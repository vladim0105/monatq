pub mod distribution;
#[cfg(feature = "visualize")]
pub mod server;
pub mod tensor_digest;
pub mod tensor_value;

pub use distribution::Distribution;
pub use tensor_digest::TensorDigest;
pub use tensor_value::TensorValue;

/// A loaded digest whose element type was determined at runtime from its snapshot.
pub enum AnyTensorDigest {
    F32(TensorDigest<f32>),
    I32(TensorDigest<i32>),
}

/// Load a digest from memory, detecting the element type from the embedded dtype tag.
pub fn from_bytes(bytes: &[u8]) -> std::io::Result<AnyTensorDigest> {
    let payload = zstd::decode_all(bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    match payload.first().copied() {
        Some(0) => TensorDigest::<f32>::from_payload(&payload).map(AnyTensorDigest::F32),
        Some(1) => TensorDigest::<i32>::from_payload(&payload).map(AnyTensorDigest::I32),
        Some(t) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown dtype tag {t}"),
        )),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty snapshot payload",
        )),
    }
}

/// Load a digest from `path`, detecting the element type from the embedded dtype tag.
pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<AnyTensorDigest> {
    let bytes = std::fs::read(path)?;
    from_bytes(&bytes)
}
