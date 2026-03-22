use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[pyclass(name = "TensorDigest")]
struct PyTensorDigest {
    inner: monatq::TensorDigest,
}

#[pymethods]
impl PyTensorDigest {
    #[new]
    fn new(shape: Vec<usize>, compression: usize) -> Self {
        Self {
            inner: monatq::TensorDigest::new(&shape, compression),
        }
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn update(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let numel = self.inner.numel();

        // Fast path: buffer protocol (numpy arrays and other PEP 3118 objects)
        if let Ok(buf) = PyBuffer::<f32>::get(data) {
            if buf.item_count() != numel {
                return Err(PyValueError::new_err(format!(
                    "data element count {} does not match numel {}",
                    buf.item_count(),
                    numel,
                )));
            }
            let mut vec = vec![0.0f32; numel];
            buf.copy_to_slice(py, &mut vec)?;
            self.inner.update(&vec);
            return Ok(());
        }

        // Fast path: torch.Tensor via data_ptr() (buffer protocol not supported by torch)
        if let Ok(ptr_obj) = data.call_method0("data_ptr") {
            if let Ok(ptr) = ptr_obj.extract::<usize>() {
                // Validate dtype is float32
                let dtype_str = data.getattr("dtype")?.str()?.to_string();
                if !dtype_str.contains("float32") {
                    return Err(PyValueError::new_err(
                        "torch tensor must be float32; call .float() first",
                    ));
                }
                // Validate device is CPU
                let device_type = data
                    .getattr("device")?
                    .getattr("type")?
                    .extract::<String>()?;
                if device_type != "cpu" {
                    return Err(PyValueError::new_err(
                        "torch tensor must be on CPU; call .cpu() first",
                    ));
                }
                // Validate contiguous
                if !data.call_method0("is_contiguous")?.extract::<bool>()? {
                    return Err(PyValueError::new_err(
                        "torch tensor must be contiguous; call .contiguous() first",
                    ));
                }
                // Validate element count
                let n = data.call_method0("numel")?.extract::<usize>()?;
                if n != numel {
                    return Err(PyValueError::new_err(format!(
                        "data element count {} does not match numel {}",
                        n, numel,
                    )));
                }
                let slice =
                    unsafe { std::slice::from_raw_parts(ptr as *const f32, n) };
                self.inner.update(slice);
                return Ok(());
            }
        }

        // Fallback: Python list / any sequence of floats
        let vec = data.extract::<Vec<f32>>()?;
        if vec.len() != numel {
            return Err(PyValueError::new_err(format!(
                "data length {} does not match numel {}",
                vec.len(),
                numel,
            )));
        }
        self.inner.update(&vec);
        Ok(())
    }

    fn flush(&mut self) {
        self.inner.flush();
    }

    fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.inner.quantile(q)
    }

    fn quantiles(&mut self, qs: Vec<f32>) -> Vec<Vec<f32>> {
        self.inner.quantiles(&qs)
    }

    fn cell_quantiles(&mut self, idx: usize, qs: Vec<f32>) -> Vec<f32> {
        self.inner.flush();
        self.inner.cell_quantiles(idx, &qs)
    }

    fn analyze(&mut self) -> Vec<String> {
        self.inner
            .analyze()
            .iter()
            .map(|d| d.to_string())
            .collect()
    }

    fn merge_cells(&mut self, indices: Vec<usize>) -> PyTensorDigest {
        PyTensorDigest {
            inner: self.inner.merge_cells(&indices),
        }
    }

    fn merge_channels(&mut self, channel_indices: Vec<usize>) -> PyTensorDigest {
        PyTensorDigest {
            inner: self.inner.merge_channels(&channel_indices),
        }
    }

    fn merge_all(&mut self) -> PyTensorDigest {
        PyTensorDigest {
            inner: self.inner.merge_all(),
        }
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(PyIOError::new_err)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<PyTensorDigest> {
        monatq::TensorDigest::load(path)
            .map(|inner| PyTensorDigest { inner })
            .map_err(PyIOError::new_err)
    }

    fn visualize(&mut self, py: Python<'_>) -> PyResult<()> {
        let stop = AtomicBool::new(false);

        let result = py.detach(|| {
            std::thread::scope(|scope| -> PyResult<std::io::Result<()>> {
                let handle = scope.spawn(|| self.inner.visualize_until(&stop));
                loop {
                    if handle.is_finished() {
                        return Ok(handle.join().unwrap());
                    }
                    if let Err(err) = Python::attach(|py| py.check_signals()) {
                        stop.store(true, Ordering::Relaxed);
                        let _ = handle.join();
                        return Err(err);
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            })
        });

        result?.map_err(PyIOError::new_err)
    }
}

#[pymodule]
mod monatq_py {
    #[pymodule_export]
    use super::PyTensorDigest as TensorDigest;
}
