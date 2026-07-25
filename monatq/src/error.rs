//! Error type shared by every fallible operation in this crate.

/// A fallible monatq operation.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything a public monatq call can refuse to do.
///
/// Operations that cannot fail keep infallible signatures. In particular `quantile`,
/// `quantiles`, `flush`, `numel`, and `shape` never return an error, because the crate
/// documents NaN-free input as a precondition of `update` rather than validating it on the
/// hot path.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The selected kernel does not implement this operation.
    ///
    /// This is a property of the kernel, not of the data: the same call on the same digest
    /// will never start succeeding.
    #[error("{kernel} does not implement {operation}")]
    Unsupported {
        kernel: &'static str,
        operation: &'static str,
    },

    /// A tensor sample did not match the digest's element count.
    #[error("tensor sample has {actual} elements but this digest tracks {expected}")]
    ShapeMismatch { expected: usize, actual: usize },

    /// A flat tensor position was outside the digest.
    #[error("position index {index} is out of bounds for {numel} tensor positions")]
    IndexOutOfBounds { index: usize, numel: usize },

    /// A kernel configuration value was rejected at construction time.
    #[error("invalid configuration for {parameter}: {message}")]
    InvalidConfig {
        parameter: &'static str,
        message: &'static str,
    },

    /// The bytes offered to a loader are not a valid snapshot.
    ///
    /// Covers a foreign or corrupt payload, an encoding the running build cannot decode, and
    /// state that violates a kernel invariant. Distinct from [`Self::Io`]: the bytes were
    /// read successfully, they just do not describe a digest this build can trust.
    #[error("invalid snapshot: {0}")]
    InvalidSnapshot(String),

    /// Reading or writing a snapshot file failed, or the visualizer server failed.
    ///
    /// The underlying cause is available through [`std::error::Error::source`]; this message
    /// deliberately does not repeat it.
    #[error("snapshot or file access failed")]
    Io(#[from] std::io::Error),
}

impl Error {
    /// The underlying [`std::io::ErrorKind`], if this is a genuine I/O failure.
    ///
    /// Returns `None` for a malformed snapshot, which is [`Self::InvalidSnapshot`] rather
    /// than an I/O error.
    pub fn io_kind(&self) -> Option<std::io::ErrorKind> {
        match self {
            Self::Io(error) => Some(error.kind()),
            _ => None,
        }
    }

    /// True when the selected kernel simply does not implement the operation.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported { .. })
    }

    /// True when the offered bytes were readable but are not a usable snapshot.
    pub fn is_invalid_snapshot(&self) -> bool {
        matches!(self, Self::InvalidSnapshot(_))
    }

    /// Classify an I/O error raised while decoding snapshot bytes.
    ///
    /// Decoders built on `std::io` report malformed input as
    /// [`std::io::ErrorKind::InvalidData`]. That is a statement about the bytes, not about
    /// the device, so it is lifted to [`Self::InvalidSnapshot`].
    pub(crate) fn from_snapshot_io(error: std::io::Error) -> Self {
        if error.kind() == std::io::ErrorKind::InvalidData {
            Self::InvalidSnapshot(error.to_string())
        } else {
            Self::Io(error)
        }
    }
}

/// Allows callers with `io::Result`-shaped APIs, including the Python bindings, to keep
/// treating monatq failures as I/O errors without matching on every variant.
impl From<Error> for std::io::Error {
    fn from(error: Error) -> Self {
        match error {
            Error::Io(inner) => inner,
            other @ Error::InvalidSnapshot(_) => {
                std::io::Error::new(std::io::ErrorKind::InvalidData, other.to_string())
            }
            other => std::io::Error::new(std::io::ErrorKind::InvalidInput, other.to_string()),
        }
    }
}

/// Build an [`Error::Unsupported`] for a kernel operation that is still a stub.
pub(crate) fn unsupported<T>(kernel: &'static str, operation: &'static str) -> Result<T> {
    Err(Error::Unsupported { kernel, operation })
}

/// Validate a flat tensor position.
pub(crate) fn check_index(index: usize, numel: usize) -> Result<()> {
    if index < numel {
        Ok(())
    } else {
        Err(Error::IndexOutOfBounds { index, numel })
    }
}

/// Validate the element count of an incoming tensor sample.
pub(crate) fn check_sample_len(actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::ShapeMismatch { expected, actual })
    }
}
