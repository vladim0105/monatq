use std::io::ErrorKind;

use monatq::{AnyTensorDigest, TensorDigest};

fn make_f32_digest() -> TensorDigest<f32> {
    let mut digest = TensorDigest::<f32>::new(&[1], 100);
    for i in 0..500u32 {
        digest.update(&[i as f32 * 0.002]);
    }
    digest
}

fn make_i32_digest() -> TensorDigest<i32> {
    let mut digest = TensorDigest::<i32>::new(&[2], 100);
    for i in 0..50 {
        digest.update(&[i, -i]);
    }
    digest
}

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("monatq_{name}_{}.bin", std::process::id()))
}

#[test]
fn file_roundtrip() {
    let mut original = make_f32_digest();
    let expected = original.quantiles(&[0.1, 0.5, 0.9]);
    let path = temp_path("file_roundtrip");

    original.save(&path).expect("save failed");
    let mut loaded = TensorDigest::<f32>::load(&path).expect("load failed");
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.shape(), original.shape());
    assert_eq!(loaded.numel(), original.numel());
    assert_eq!(loaded.quantiles(&[0.1, 0.5, 0.9]), expected);
}

#[test]
fn typed_f32_bytes_roundtrip() {
    let mut original = make_f32_digest();
    let expected = original.quantiles(&[0.1, 0.5, 0.9]);

    let bytes = original.to_bytes().expect("serialization failed");
    let mut loaded = TensorDigest::<f32>::from_bytes(&bytes).expect("deserialization failed");

    assert_eq!(loaded.shape(), original.shape());
    assert_eq!(loaded.numel(), original.numel());
    assert_eq!(loaded.quantiles(&[0.1, 0.5, 0.9]), expected);
}

#[test]
fn typed_i32_bytes_roundtrip() {
    let mut original = make_i32_digest();
    let expected = original.quantiles(&[0.25, 0.5, 0.75]);

    let bytes = original.to_bytes().expect("serialization failed");
    let mut loaded = TensorDigest::<i32>::from_bytes(&bytes).expect("deserialization failed");

    assert_eq!(loaded.shape(), original.shape());
    assert_eq!(loaded.numel(), original.numel());
    assert_eq!(loaded.quantiles(&[0.25, 0.5, 0.75]), expected);
}

#[test]
fn runtime_dtype_autodetection_from_bytes() {
    let mut f32_digest = make_f32_digest();
    let f32_bytes = f32_digest.to_bytes().expect("f32 serialization failed");
    match monatq::from_bytes(&f32_bytes).expect("f32 autodetection failed") {
        AnyTensorDigest::F32(mut loaded) => {
            assert_eq!(loaded.quantile(0.5), f32_digest.quantile(0.5));
        }
        AnyTensorDigest::I32(_) => panic!("f32 snapshot was detected as i32"),
    }

    let mut i32_digest = make_i32_digest();
    let i32_bytes = i32_digest.to_bytes().expect("i32 serialization failed");
    match monatq::from_bytes(&i32_bytes).expect("i32 autodetection failed") {
        AnyTensorDigest::I32(mut loaded) => {
            assert_eq!(loaded.quantile(0.5), i32_digest.quantile(0.5));
        }
        AnyTensorDigest::F32(_) => panic!("i32 snapshot was detected as f32"),
    }
}

#[test]
fn file_and_bytes_formats_are_cross_compatible() {
    let mut original = make_f32_digest();
    let expected = original.quantile(0.5);
    let path = temp_path("cross_compatible");

    original.save(&path).expect("save failed");
    let file_bytes = std::fs::read(&path).expect("read failed");
    let mut loaded_from_file_bytes =
        TensorDigest::<f32>::from_bytes(&file_bytes).expect("from_bytes failed");
    assert_eq!(loaded_from_file_bytes.quantile(0.5), expected);

    let memory_bytes = original.to_bytes().expect("to_bytes failed");
    std::fs::write(&path, memory_bytes).expect("write failed");
    let loaded_from_memory_bytes = monatq::load(&path).expect("load failed");
    std::fs::remove_file(&path).ok();
    match loaded_from_memory_bytes {
        AnyTensorDigest::F32(mut loaded) => assert_eq!(loaded.quantile(0.5), expected),
        AnyTensorDigest::I32(_) => panic!("f32 snapshot was detected as i32"),
    }
}

#[test]
fn to_bytes_flushes_pending_data() {
    let mut digest = TensorDigest::<f32>::new(&[1], 100);
    for value in 0..10 {
        digest.update(&[value as f32]);
    }
    assert_eq!(digest.total_weight(0), 0);

    let bytes = digest.to_bytes().expect("serialization failed");
    assert_eq!(digest.total_weight(0), 10);

    let mut loaded = TensorDigest::<f32>::from_bytes(&bytes).expect("deserialization failed");
    assert_eq!(loaded.total_weight(0), 10);
    assert_eq!(loaded.quantile(0.5), digest.quantile(0.5));
}

#[test]
fn typed_from_bytes_rejects_dtype_mismatch() {
    let mut digest = make_f32_digest();
    let bytes = digest.to_bytes().expect("serialization failed");
    let error = TensorDigest::<i32>::from_bytes(&bytes)
        .err()
        .expect("expected a dtype mismatch");

    assert_eq!(error.kind(), ErrorKind::InvalidData);
    assert!(error.to_string().contains("dtype mismatch"));
}

#[test]
fn invalid_byte_inputs_are_rejected() {
    for bytes in [&[][..], &[1, 2, 3][..]] {
        let error = monatq::from_bytes(bytes)
            .err()
            .expect("invalid input unexpectedly loaded");
        assert_eq!(error.kind(), ErrorKind::InvalidData);
    }

    let empty_payload = zstd::encode_all(&[][..], 3).expect("compression failed");
    let empty_error = monatq::from_bytes(&empty_payload)
        .err()
        .expect("empty payload unexpectedly loaded");
    assert_eq!(empty_error.kind(), ErrorKind::InvalidData);

    let unknown_payload = zstd::encode_all(&[99][..], 3).expect("compression failed");
    let unknown_error = monatq::from_bytes(&unknown_payload)
        .err()
        .expect("unknown dtype unexpectedly loaded");
    assert_eq!(unknown_error.kind(), ErrorKind::InvalidData);
    assert!(unknown_error.to_string().contains("unknown dtype tag 99"));

    let mut valid_digest = make_f32_digest();
    let mut truncated = valid_digest.to_bytes().expect("serialization failed");
    truncated.truncate(truncated.len() / 2);
    let truncated_error = TensorDigest::<f32>::from_bytes(&truncated)
        .err()
        .expect("truncated snapshot unexpectedly loaded");
    assert_eq!(truncated_error.kind(), ErrorKind::InvalidData);
}
