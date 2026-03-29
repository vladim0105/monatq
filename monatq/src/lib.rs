pub mod distribution;
#[cfg(feature = "visualize")]
pub mod server;
pub mod tensor_digest;

pub use distribution::Distribution;
pub use tensor_digest::TensorDigest;
pub use tensor_digest::TensorValue;

/// A loaded digest whose element type was determined at runtime from the file.
pub enum AnyTensorDigest {
    F32(TensorDigest<f32>),
    I32(TensorDigest<i32>),
}

/// Load a digest from `path`, detecting the element type from the embedded dtype tag.
pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<AnyTensorDigest> {
    let compressed = std::fs::read(&path)?;
    let bytes = zstd::decode_all(compressed.as_slice())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    match bytes.first().copied() {
        Some(0) => TensorDigest::<f32>::load(&path).map(AnyTensorDigest::F32),
        Some(1) => TensorDigest::<i32>::load(&path).map(AnyTensorDigest::I32),
        Some(t) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unknown dtype tag {t}"),
        )),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty file",
        )),
    }
}
