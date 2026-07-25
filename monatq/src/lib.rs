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

/// A loaded T-Digest whose element type was determined at runtime from its snapshot.
pub enum AnyTensorDigest {
    F32(TensorDigest<f32, TDigest>),
    I32(TensorDigest<i32, TDigest>),
}

/// Load a T-Digest from memory, detecting the element type from the embedded dtype tag.
///
/// This reader is t-digest specific. Snapshots from other kernels carry a distinct leading
/// kernel tag and are reported as an unknown dtype rather than misparsed.
pub fn from_bytes(bytes: &[u8]) -> Result<AnyTensorDigest> {
    let payload = zstd::decode_all(bytes)
        .map_err(|e| Error::InvalidSnapshot(format!("not decodable: {e}")))?;
    match payload.first().copied() {
        Some(0) => TensorDigest::<f32, TDigest>::from_payload(&payload).map(AnyTensorDigest::F32),
        Some(1) => TensorDigest::<i32, TDigest>::from_payload(&payload).map(AnyTensorDigest::I32),
        Some(t) => Err(Error::InvalidSnapshot(format!(
            "unknown dtype tag {t}; snapshots from other kernels are not loadable here"
        ))),
        None => Err(Error::InvalidSnapshot("empty payload".to_string())),
    }
}

/// Load a T-Digest from `path`, detecting the element type from the embedded dtype tag.
pub fn load(path: impl AsRef<std::path::Path>) -> Result<AnyTensorDigest> {
    let bytes = std::fs::read(path).map_err(Error::Io)?;
    from_bytes(&bytes)
}
