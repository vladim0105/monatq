use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{
    PyIOError, PyIndexError, PyNotImplementedError, PyRuntimeError, PyValueError,
};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Translate a monatq error into the closest Python exception.
///
/// Before monatq returned `Result`, these conditions were Rust panics that crossed the FFI
/// boundary as `pyo3_runtime.PanicException` with a Rust backtrace attached. Mapping them
/// gives Python callers ordinary, catchable exceptions instead.
fn to_py_err(error: monatq::Error) -> PyErr {
    match error {
        monatq::Error::Unsupported { .. } => PyNotImplementedError::new_err(error.to_string()),
        monatq::Error::IndexOutOfBounds { .. } => PyIndexError::new_err(error.to_string()),
        monatq::Error::ShapeMismatch { .. } | monatq::Error::InvalidConfig { .. } => {
            PyValueError::new_err(error.to_string())
        }
        monatq::Error::InvalidSnapshot(_) => PyValueError::new_err(error.to_string()),
        // Display no longer repeats the underlying cause, so report the chain explicitly.
        monatq::Error::Io(ref inner) => PyIOError::new_err(format!("{error}: {inner}")),
        // `monatq::Error` is `#[non_exhaustive]`, so a future variant must not break this
        // build. Anything unrecognized becomes a plain RuntimeError rather than a panic.
        ref other => PyRuntimeError::new_err(other.to_string()),
    }
}

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
fn try_torch(
    data: &Bound<'_, PyAny>,
    input_numel: usize,
) -> PyResult<Option<(usize, usize, String)>> {
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
    if n != input_numel {
        return Err(PyValueError::new_err(format!(
            "data element count {n} does not match input_numel {input_numel}"
        )));
    }
    Ok(Some((ptr, n, dtype_str)))
}

/// Common input handling for all element types:
/// buffer protocol (numpy) → torch fast path → Python list fallback.
fn update_typed<T, K>(
    py: Python<'_>,
    d: &mut monatq::TensorDigest<T, K>,
    data: &Bound<'_, PyAny>,
    input_numel: usize,
    dtype_name: &'static str,
) -> PyResult<()>
where
    T: monatq::TensorValue + pyo3::buffer::Element + for<'a, 'py> FromPyObject<'a, 'py> + Default,
    K: monatq::DigestKernel<T>,
{
    // Buffer protocol (numpy arrays)
    if let Ok(buf) = PyBuffer::<T>::get(data) {
        if buf.item_count() != input_numel {
            return Err(PyValueError::new_err(format!(
                "data element count {} does not match input_numel {}",
                buf.item_count(),
                input_numel,
            )));
        }
        let mut vec = vec![T::default(); input_numel];
        buf.copy_to_slice(py, &mut vec)?;
        d.update(&vec).map_err(to_py_err)?;
        return Ok(());
    }
    // Torch tensor fast path
    if let Some((ptr, n, dtype_str)) = try_torch(data, input_numel)? {
        if !dtype_str.contains(dtype_name) {
            return Err(PyValueError::new_err(format!(
                "this digest uses dtype {dtype_name} but tensor dtype is {dtype_str}"
            )));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const T, n) };
        d.update(slice).map_err(to_py_err)?;
        return Ok(());
    }
    // Python list fallback
    let vec = data.extract::<Vec<T>>()?;
    if vec.len() != input_numel {
        return Err(PyValueError::new_err(format!(
            "data length {} does not match input_numel {}",
            vec.len(),
            input_numel,
        )));
    }
    d.update(&vec).map_err(to_py_err)?;
    Ok(())
}

/// One digest over every supported (element type, kernel) pair.
///
/// Python cannot carry Rust type parameters, so the static choice has to be reified into an
/// enum here. Each arm is still a monomorphised `TensorDigest` with no dynamic dispatch.
enum Inner {
    TDigestF32(monatq::TensorDigest<f32, monatq::TDigest>),
    TDigestI32(monatq::TensorDigest<i32, monatq::TDigest>),
    RankKnotF32(monatq::TensorDigest<f32, monatq::RankKnot>),
    RankKnotI32(monatq::TensorDigest<i32, monatq::RankKnot>),
}

/// Run the same expression against whichever digest an [`Inner`] holds.
///
/// Written as a macro rather than a trait object so the four arms stay statically
/// dispatched, and so adding a method does not mean writing the same four-way match again.
macro_rules! dispatch {
    ($self:expr, $d:ident => $body:expr) => {
        match $self {
            Inner::TDigestF32($d) => $body,
            Inner::TDigestI32($d) => $body,
            Inner::RankKnotF32($d) => $body,
            Inner::RankKnotI32($d) => $body,
        }
    };
}

/// Like [`dispatch`], but rewraps the result back into the same variant.
macro_rules! dispatch_rewrap {
    ($self:expr, $d:ident => $body:expr) => {
        match $self {
            Inner::TDigestF32($d) => Inner::TDigestF32($body),
            Inner::TDigestI32($d) => Inner::TDigestI32($body),
            Inner::RankKnotF32($d) => Inner::RankKnotF32($body),
            Inner::RankKnotI32($d) => Inner::RankKnotI32($body),
        }
    };
}

impl From<monatq::AnyTensorDigest> for Inner {
    fn from(any: monatq::AnyTensorDigest) -> Self {
        match any {
            monatq::AnyTensorDigest::TDigestF32(d) => Inner::TDigestF32(d),
            monatq::AnyTensorDigest::TDigestI32(d) => Inner::TDigestI32(d),
            monatq::AnyTensorDigest::RankKnotF32(d) => Inner::RankKnotF32(d),
            monatq::AnyTensorDigest::RankKnotI32(d) => Inner::RankKnotI32(d),
        }
    }
}

impl Inner {
    fn shape(&self) -> &[usize] {
        dispatch!(self, d => d.shape())
    }
    fn input_shape(&self) -> &[usize] {
        dispatch!(self, d => d.input_shape())
    }
    fn input_numel(&self) -> usize {
        dispatch!(self, d => d.input_numel())
    }
    fn block_count(&self) -> usize {
        dispatch!(self, d => d.block_count())
    }
    fn block_axis(&self) -> usize {
        dispatch!(self, d => d.block_axis())
    }
    fn blocks_per_axis(&self) -> usize {
        dispatch!(self, d => d.blocks_per_axis())
    }
    fn dtype(&self) -> &'static str {
        match self {
            Inner::TDigestF32(_) | Inner::RankKnotF32(_) => "float32",
            Inner::TDigestI32(_) | Inner::RankKnotI32(_) => "int32",
        }
    }
    fn kernel(&self) -> &'static str {
        match self {
            Inner::TDigestF32(_) | Inner::TDigestI32(_) => "tdigest",
            Inner::RankKnotF32(_) | Inner::RankKnotI32(_) => "rankknot",
        }
    }
    fn update(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let input_numel = self.input_numel();
        let dtype = self.dtype();
        dispatch!(self, d => update_typed(py, d, data, input_numel, dtype))
    }
    fn flush(&mut self) {
        dispatch!(self, d => d.flush())
    }
    fn quantile(&mut self, q: f32) -> Vec<f32> {
        dispatch!(self, d => d.quantile(q))
    }
    fn quantiles(&mut self, qs: &[f32]) -> Vec<Vec<f32>> {
        dispatch!(self, d => d.quantiles(qs))
    }
    fn cell_quantiles(&mut self, idx: usize, qs: &[f32]) -> monatq::Result<Vec<f32>> {
        self.flush();
        dispatch!(self, d => d.cell_quantiles(idx, qs))
    }
    fn analyze(&mut self) -> monatq::Result<Vec<monatq::Distribution>> {
        dispatch!(self, d => d.analyze())
    }
    fn merge_cells(&mut self, indices: &[usize]) -> monatq::Result<Self> {
        Ok(dispatch_rewrap!(self, d => d.merge_cells(indices)?))
    }
    fn merge_channels(&mut self, channel_indices: &[usize]) -> monatq::Result<Self> {
        Ok(dispatch_rewrap!(self, d => d.merge_channels(channel_indices)?))
    }
    fn merge_all(&mut self) -> monatq::Result<Self> {
        Ok(dispatch_rewrap!(self, d => d.merge_all()?))
    }
    fn without_zeros(&mut self) -> monatq::Result<Self> {
        Ok(dispatch_rewrap!(self, d => d.without_zeros()?))
    }
    fn save(&mut self, path: &str) -> monatq::Result<()> {
        dispatch!(self, d => d.save(path))
    }
    #[allow(clippy::wrong_self_convention)]
    fn to_bytes(&mut self) -> monatq::Result<Vec<u8>> {
        dispatch!(self, d => d.to_bytes())
    }
    fn visualize_until(&mut self, stop: &AtomicBool) -> monatq::Result<()> {
        dispatch!(self, d => d.visualize_until(stop))
    }
}

#[pyclass(name = "TensorDigest")]
struct PyTensorDigest {
    inner: Inner,
}

#[pymethods]
impl PyTensorDigest {
    /// Build a digest over `shape`.
    ///
    /// `kernel` defaults to `"rankknot"`. The tuning knobs are kernel-specific:
    /// `compression` belongs to `"tdigest"` and `buffer_capacity` to `"rankknot"`. Passing
    /// one that does not belong to the selected kernel is an error rather than a silent
    /// no-op, because silently ignoring an accuracy knob is the kind of thing a caller only
    /// discovers from a bad result much later.
    #[new]
    #[pyo3(signature = (shape, *, kernel = "rankknot", compression = None, buffer_capacity = None, dtype = None, blocks_per_axis = None, block_axis = None))]
    fn new(
        shape: Vec<usize>,
        kernel: &str,
        compression: Option<usize>,
        buffer_capacity: Option<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        blocks_per_axis: Option<usize>,
        block_axis: Option<isize>,
    ) -> PyResult<Self> {
        let dtype = dtype.map(normalize_dtype).transpose()?.unwrap_or("float32");
        let kernel_name = kernel.trim().to_ascii_lowercase();
        let blocks = if !shape.is_empty() || blocks_per_axis.is_some() || block_axis.is_some() {
            let rank = isize::try_from(shape.len())
                .map_err(|_| PyValueError::new_err("tensor rank is too large"))?;
            let requested_axis = block_axis.unwrap_or(-1);
            let axis = if requested_axis < 0 {
                rank.checked_add(requested_axis)
            } else {
                Some(requested_axis)
            };
            let axis = axis
                .filter(|&axis| axis >= 0 && axis < rank)
                .ok_or_else(|| {
                    PyValueError::new_err("block_axis must name an existing tensor axis")
                })?;
            Some(monatq::BlockConfig::new(
                blocks_per_axis.unwrap_or(0),
                axis as usize,
            ))
        } else {
            None
        };

        let reject = |knob: &str, owner: &str| {
            Err(PyValueError::new_err(format!(
                "{knob} is a {owner} setting and has no effect on kernel {kernel_name:?}"
            )))
        };

        let inner = match kernel_name.as_str() {
            "rankknot" => {
                if compression.is_some() {
                    return reject("compression", "tdigest");
                }
                let config = match buffer_capacity {
                    Some(capacity) => monatq::RankKnotConfig {
                        buffer_capacity: capacity,
                    },
                    None => monatq::RankKnotConfig::default(),
                };
                match dtype {
                    "float32" => Inner::RankKnotF32(match blocks {
                        Some(b) => monatq::TensorDigest::with_block_config(&shape, config, b)
                            .map_err(to_py_err)?,
                        None => monatq::TensorDigest::with_config(&shape, config),
                    }),
                    _ => Inner::RankKnotI32(match blocks {
                        Some(b) => monatq::TensorDigest::with_block_config(&shape, config, b)
                            .map_err(to_py_err)?,
                        None => monatq::TensorDigest::with_config(&shape, config),
                    }),
                }
            }
            "tdigest" => {
                if buffer_capacity.is_some() {
                    return reject("buffer_capacity", "rankknot");
                }
                let config = monatq::TDigestConfig {
                    compression: compression.unwrap_or(100),
                };
                match dtype {
                    "float32" => Inner::TDigestF32(match blocks {
                        Some(b) => monatq::TensorDigest::with_block_config(&shape, config, b)
                            .map_err(to_py_err)?,
                        None => monatq::TensorDigest::with_config(&shape, config),
                    }),
                    _ => Inner::TDigestI32(match blocks {
                        Some(b) => monatq::TensorDigest::with_block_config(&shape, config, b)
                            .map_err(to_py_err)?,
                        None => monatq::TensorDigest::with_config(&shape, config),
                    }),
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown kernel {other:?}; supported: rankknot, tdigest"
                )));
            }
        };
        Ok(Self { inner })
    }

    #[getter]
    fn kernel(&self) -> &str {
        self.inner.kernel()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn input_shape(&self) -> Vec<usize> {
        self.inner.input_shape().to_vec()
    }

    #[getter]
    fn input_numel(&self) -> usize {
        self.inner.input_numel()
    }

    #[getter]
    fn dtype(&self) -> &str {
        self.inner.dtype()
    }

    #[getter]
    fn block_count(&self) -> usize {
        self.inner.block_count()
    }
    #[getter]
    fn block_axis(&self) -> usize {
        self.inner.block_axis()
    }
    #[getter]
    fn blocks_per_axis(&self) -> usize {
        self.inner.blocks_per_axis()
    }

    fn update(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.update(py, data)
    }

    fn flush(&mut self) {
        self.inner.flush()
    }

    fn quantile(&mut self, q: f32) -> Vec<f32> {
        self.inner.quantile(q)
    }

    fn quantiles(&mut self, qs: Vec<f32>) -> Vec<Vec<f32>> {
        self.inner.quantiles(&qs)
    }

    fn cell_quantiles(&mut self, idx: usize, qs: Vec<f32>) -> PyResult<Vec<f32>> {
        self.inner.cell_quantiles(idx, &qs).map_err(to_py_err)
    }

    fn analyze(&mut self) -> PyResult<Vec<String>> {
        Ok(self
            .inner
            .analyze()
            .map_err(to_py_err)?
            .iter()
            .map(|d| d.to_string())
            .collect())
    }

    fn merge_cells(&mut self, indices: Vec<usize>) -> PyResult<PyTensorDigest> {
        Ok(PyTensorDigest {
            inner: self.inner.merge_cells(&indices).map_err(to_py_err)?,
        })
    }

    fn merge_channels(&mut self, channel_indices: Vec<usize>) -> PyResult<PyTensorDigest> {
        Ok(PyTensorDigest {
            inner: self
                .inner
                .merge_channels(&channel_indices)
                .map_err(to_py_err)?,
        })
    }

    fn merge_all(&mut self) -> PyResult<PyTensorDigest> {
        Ok(PyTensorDigest {
            inner: self.inner.merge_all().map_err(to_py_err)?,
        })
    }

    /// Copy of this digest with values at zero removed.
    ///
    /// Quantiles of the result describe the nonzero subpopulation; `count` is unchanged and
    /// no longer matches it.
    fn without_zeros(&mut self) -> PyResult<PyTensorDigest> {
        Ok(PyTensorDigest {
            inner: self.inner.without_zeros().map_err(to_py_err)?,
        })
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(to_py_err)
    }

    /// Serialize to a `bytes` snapshot. Pairs with `from_bytes`.
    #[allow(clippy::wrong_self_convention)]
    fn to_bytes<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self.inner.to_bytes().map_err(to_py_err)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Load a snapshot, detecting its kernel and element type from the bytes themselves.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<PyTensorDigest> {
        let inner = monatq::from_bytes(data).map_err(to_py_err)?.into();
        Ok(PyTensorDigest { inner })
    }

    /// Load a snapshot from `path`, detecting its kernel and element type.
    #[staticmethod]
    fn load(path: &str) -> PyResult<PyTensorDigest> {
        let inner = monatq::load(path).map_err(to_py_err)?.into();
        Ok(PyTensorDigest { inner })
    }

    fn visualize(&mut self, py: Python<'_>) -> PyResult<()> {
        let stop = AtomicBool::new(false);

        let result = py.detach(|| {
            std::thread::scope(|scope| -> PyResult<monatq::Result<()>> {
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

        result?.map_err(to_py_err)
    }
}

#[pymodule(name = "monatq")]
mod _monatq {
    #[pymodule_export]
    use super::PyTensorDigest as TensorDigest;
}
