use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

fn normalize_dtype(obj: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    let s = obj.str()?.to_string();
    if s.contains("float32") {
        Ok("float32")
    } else if s.contains("int32") {
        Ok("int32")
    } else {
        Err(PyValueError::new_err(format!(
            "cannot interpret {s:?} as a dtype; supported: float32, int32"
        )))
    }
}

/// Probe `data` for a torch tensor. Returns `(data_ptr, numel, dtype_str)` on success,
/// `None` if `data` is not a torch tensor, or an error for invalid torch tensors.
fn try_torch(data: &Bound<'_, PyAny>, numel: usize) -> PyResult<Option<(usize, usize, String)>> {
    let Ok(ptr_obj) = data.call_method0("data_ptr") else {
        return Ok(None);
    };
    let Ok(ptr) = ptr_obj.extract::<usize>() else {
        return Ok(None);
    };
    let dtype_str = data.getattr("dtype")?.str()?.to_string();
    let device_type = data
        .getattr("device")?
        .getattr("type")?
        .extract::<String>()?;
    if device_type != "cpu" {
        return Err(PyValueError::new_err(
            "torch tensor must be on CPU; call .cpu() first",
        ));
    }
    if !data.call_method0("is_contiguous")?.extract::<bool>()? {
        return Err(PyValueError::new_err(
            "torch tensor must be contiguous; call .contiguous() first",
        ));
    }
    let n = data.call_method0("numel")?.extract::<usize>()?;
    if n != numel {
        return Err(PyValueError::new_err(format!(
            "data element count {n} does not match numel {numel}"
        )));
    }
    Ok(Some((ptr, n, dtype_str)))
}

/// Common input handling for all element types:
/// buffer protocol (numpy) → torch fast path → Python list fallback.
fn update_typed<T>(
    py: Python<'_>,
    d: &mut monatq::TensorDigest<T>,
    data: &Bound<'_, PyAny>,
    numel: usize,
    dtype_name: &'static str,
) -> PyResult<()>
where
    T: monatq::TensorValue + pyo3::buffer::Element + for<'a, 'py> FromPyObject<'a, 'py> + Default,
{
    // Buffer protocol (numpy arrays)
    if let Ok(buf) = PyBuffer::<T>::get(data) {
        if buf.item_count() != numel {
            return Err(PyValueError::new_err(format!(
                "data element count {} does not match numel {}",
                buf.item_count(),
                numel,
            )));
        }
        let mut vec = vec![T::default(); numel];
        buf.copy_to_slice(py, &mut vec)?;
        d.update(&vec);
        return Ok(());
    }
    // Torch tensor fast path
    if let Some((ptr, n, dtype_str)) = try_torch(data, numel)? {
        if !dtype_str.contains(dtype_name) {
            return Err(PyValueError::new_err(format!(
                "this digest uses dtype {dtype_name} but tensor dtype is {dtype_str}"
            )));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const T, n) };
        d.update(slice);
        return Ok(());
    }
    // Python list fallback
    let vec = data.extract::<Vec<T>>()?;
    if vec.len() != numel {
        return Err(PyValueError::new_err(format!(
            "data length {} does not match numel {}",
            vec.len(),
            numel,
        )));
    }
    d.update(&vec);
    Ok(())
}

enum Inner {
    F32(monatq::TensorDigest<f32>),
    I32(monatq::TensorDigest<i32>),
}

impl From<monatq::AnyTensorDigest> for Inner {
    fn from(any: monatq::AnyTensorDigest) -> Self {
        match any {
            monatq::AnyTensorDigest::F32(d) => Inner::F32(d),
            monatq::AnyTensorDigest::I32(d) => Inner::I32(d),
        }
    }
}

impl Inner {
    fn shape(&self) -> &[usize] {
        match self { Inner::F32(d) => d.shape(), Inner::I32(d) => d.shape() }
    }
    fn numel(&self) -> usize {
        match self { Inner::F32(d) => d.numel(), Inner::I32(d) => d.numel() }
    }
    fn dtype(&self) -> &'static str {
        match self { Inner::F32(_) => "float32", Inner::I32(_) => "int32" }
    }
    fn update(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let numel = self.numel();
        let dtype = self.dtype();
        match self {
            Inner::F32(d) => update_typed(py, d, data, numel, dtype),
            Inner::I32(d) => update_typed(py, d, data, numel, dtype),
        }
    }
    fn flush(&mut self) {
        match self { Inner::F32(d) => d.flush(), Inner::I32(d) => d.flush() }
    }
    fn quantile(&mut self, q: f32) -> Vec<f32> {
        match self { Inner::F32(d) => d.quantile(q), Inner::I32(d) => d.quantile(q) }
    }
    fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        match self { Inner::F32(d) => d.quantiles(qs), Inner::I32(d) => d.quantiles(qs) }
    }
    fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> Vec<f32> {
        self.flush();
        match self { Inner::F32(d) => d.cell_quantiles(idx, qs), Inner::I32(d) => d.cell_quantiles(idx, qs) }
    }
    fn analyze(&mut self) -> Vec<monatq::Distribution> {
        match self { Inner::F32(d) => d.analyze(), Inner::I32(d) => d.analyze() }
    }
    fn merge_cells(&mut self, indices: &[usize]) -> Self {
        match self {
            Inner::F32(d) => Inner::F32(d.merge_cells(indices)),
            Inner::I32(d) => Inner::I32(d.merge_cells(indices)),
        }
    }
    fn merge_channels(&mut self, channel_indices: &[usize]) -> Self {
        match self {
            Inner::F32(d) => Inner::F32(d.merge_channels(channel_indices)),
            Inner::I32(d) => Inner::I32(d.merge_channels(channel_indices)),
        }
    }
    fn merge_all(&mut self) -> Self {
        match self { Inner::F32(d) => Inner::F32(d.merge_all()), Inner::I32(d) => Inner::I32(d.merge_all()) }
    }
    fn save(&mut self, path: &str) -> std::io::Result<()> {
        match self { Inner::F32(d) => d.save(path), Inner::I32(d) => d.save(path) }
    }
    fn visualize_until(&mut self, stop: &AtomicBool) -> std::io::Result<()> {
        match self { Inner::F32(d) => d.visualize_until(stop), Inner::I32(d) => d.visualize_until(stop) }
    }
}

#[pyclass(name = "TensorDigest")]
struct PyTensorDigest {
    inner: Inner,
}

#[pymethods]
impl PyTensorDigest {
    #[new]
    #[pyo3(signature = (shape, compression, dtype = None))]
    fn new(shape: Vec<usize>, compression: usize, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let inner = match dtype.map(normalize_dtype).transpose()?.unwrap_or("float32") {
            "float32" => Inner::F32(monatq::TensorDigest::<f32>::new(&shape, compression)),
            "int32" => Inner::I32(monatq::TensorDigest::<i32>::new(&shape, compression)),
            other => unreachable!("normalize_dtype returned unexpected value: {other}"),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> { self.inner.shape().to_vec() }

    #[getter]
    fn numel(&self) -> usize { self.inner.numel() }

    #[getter]
    fn dtype(&self) -> &str { self.inner.dtype() }

    fn update(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.update(py, data)
    }

    fn flush(&mut self) { self.inner.flush() }

    fn quantile(&mut self, q: f32) -> Vec<f32> { self.inner.quantile(q) }

    fn quantiles(&mut self, qs: Vec<f32>) -> Vec<Vec<f32>> { self.inner.quantiles(&qs) }

    fn cell_quantiles(&mut self, idx: usize, qs: Vec<f32>) -> Vec<f32> {
        self.inner.cell_quantiles(idx, &qs)
    }

    fn analyze(&mut self) -> Vec<String> {
        self.inner.analyze().iter().map(|d| d.to_string()).collect()
    }

    fn merge_cells(&mut self, indices: Vec<usize>) -> PyTensorDigest {
        PyTensorDigest { inner: self.inner.merge_cells(&indices) }
    }

    fn merge_channels(&mut self, channel_indices: Vec<usize>) -> PyTensorDigest {
        PyTensorDigest { inner: self.inner.merge_channels(&channel_indices) }
    }

    fn merge_all(&mut self) -> PyTensorDigest {
        PyTensorDigest { inner: self.inner.merge_all() }
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(PyIOError::new_err)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<PyTensorDigest> {
        let inner = monatq::load(path).map_err(PyIOError::new_err)?.into();
        Ok(PyTensorDigest { inner })
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

#[pymodule(name = "monatq")]
mod _monatq {
    #[pymodule_export]
    use super::PyTensorDigest as TensorDigest;
}
