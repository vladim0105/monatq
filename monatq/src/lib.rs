#[doc(hidden)]
pub mod dev_support;
pub mod distribution;
mod error;
mod kernels;
#[cfg(feature = "visualize")]
mod server;
mod tensor_digest;
pub mod tensor_value;

pub use distribution::Distribution;
pub use error::{Error, Result};
pub use kernels::quantile_spine::{QuantileSpineConfig, SpineLink, SpineRegime};
pub use kernels::{DigestKernel, QuantileSpine, RankKnot, RankKnotConfig, TDigest, TDigestConfig};
pub use tensor_digest::TensorDigest;
pub use tensor_value::TensorValue;

/// A loaded digest whose kernel and element type were both determined from its snapshot.
///
/// Variants are named kernel-first because the kernel is what decides which operations the
/// loaded digest supports. The set is closed: [`DigestKernel`] is sealed, so no downstream
/// crate can introduce a kernel that would need a variant here.
pub enum AnyTensorDigest {
    TDigestF32(TensorDigest<f32, TDigest>),
    TDigestI32(TensorDigest<i32, TDigest>),
    RankKnotF32(TensorDigest<f32, RankKnot>),
    RankKnotI32(TensorDigest<i32, RankKnot>),
}

/// Reports what was detected, so a mis-detected snapshot names itself in a test failure.
impl std::fmt::Debug for AnyTensorDigest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnyTensorDigest")
            .field("kernel", &self.kernel_name())
            .field("dtype", &self.dtype_name())
            .field("shape", &self.shape())
            .finish()
    }
}

impl AnyTensorDigest {
    /// Name of the kernel that wrote this snapshot.
    pub fn kernel_name(&self) -> &'static str {
        match self {
            Self::TDigestF32(_) | Self::TDigestI32(_) => "TDigest",
            Self::RankKnotF32(_) | Self::RankKnotI32(_) => "RankKnot",
        }
    }

    /// Name of the element type this snapshot was collected over.
    pub fn dtype_name(&self) -> &'static str {
        match self {
            Self::TDigestF32(_) | Self::RankKnotF32(_) => "f32",
            Self::TDigestI32(_) | Self::RankKnotI32(_) => "i32",
        }
    }

    /// Shape of the tensor this digest tracks.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::TDigestF32(d) => d.shape(),
            Self::TDigestI32(d) => d.shape(),
            Self::RankKnotF32(d) => d.shape(),
            Self::RankKnotI32(d) => d.shape(),
        }
    }
}

/// Load a digest from memory, detecting both the kernel and the element type.
///
/// Every kernel that serializes writes a distinguishing leading tag, so a snapshot is
/// self-describing and this reader never has to be told what it is about to open.
pub fn from_bytes(bytes: &[u8]) -> Result<AnyTensorDigest> {
    let payload = zstd::decode_all(bytes)
        .map_err(|e| Error::InvalidSnapshot(format!("not decodable: {e}")))?;

    // RankKnot leads with its own kernel tag; ask it first, since t-digest snapshots lead
    // with a bare dtype tag and so have no marker of their own to test for.
    if let Some(dtype_tag) = kernels::rankknot::peek_dtype_tag(&payload) {
        return match dtype_tag {
            0 => TensorDigest::<f32, RankKnot>::from_payload(&payload)
                .map(AnyTensorDigest::RankKnotF32),
            1 => TensorDigest::<i32, RankKnot>::from_payload(&payload)
                .map(AnyTensorDigest::RankKnotI32),
            other => Err(Error::InvalidSnapshot(format!(
                "RankKnot snapshot carries unknown dtype tag {other}"
            ))),
        };
    }

    match payload.first().copied() {
        Some(0) => {
            TensorDigest::<f32, TDigest>::from_payload(&payload).map(AnyTensorDigest::TDigestF32)
        }
        Some(1) => {
            TensorDigest::<i32, TDigest>::from_payload(&payload).map(AnyTensorDigest::TDigestI32)
        }
        Some(t) => Err(Error::InvalidSnapshot(format!(
            "unrecognized snapshot: leading tag {t} matches no known kernel or dtype"
        ))),
        None => Err(Error::InvalidSnapshot("empty payload".to_string())),
    }
}

/// Load a digest from `path`, detecting both the kernel and the element type.
pub fn load(path: impl AsRef<std::path::Path>) -> Result<AnyTensorDigest> {
    let bytes = std::fs::read(path).map_err(Error::Io)?;
    from_bytes(&bytes)
}
