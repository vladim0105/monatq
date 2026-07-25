use monatq::{AnyTensorDigest, TensorDigest};

fn make_f32_digest() -> TensorDigest<f32, monatq::TDigest> {
    let mut digest = TensorDigest::<f32, monatq::TDigest>::new(&[1]);
    for i in 0..500u32 {
        digest.update(&[i as f32 * 0.002]).unwrap();
    }
    digest
}

fn make_i32_digest() -> TensorDigest<i32, monatq::TDigest> {
    let mut digest = TensorDigest::<i32, monatq::TDigest>::new(&[2]);
    for i in 0..50 {
        digest.update(&[i, -i]).unwrap();
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
    let mut loaded = TensorDigest::<f32, monatq::TDigest>::load(&path).expect("load failed");
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
    let mut loaded =
        TensorDigest::<f32, monatq::TDigest>::from_bytes(&bytes).expect("deserialization failed");

    assert_eq!(loaded.shape(), original.shape());
    assert_eq!(loaded.numel(), original.numel());
    assert_eq!(loaded.quantiles(&[0.1, 0.5, 0.9]), expected);
}

#[test]
fn typed_i32_bytes_roundtrip() {
    let mut original = make_i32_digest();
    let expected = original.quantiles(&[0.25, 0.5, 0.75]);

    let bytes = original.to_bytes().expect("serialization failed");
    let mut loaded =
        TensorDigest::<i32, monatq::TDigest>::from_bytes(&bytes).expect("deserialization failed");

    assert_eq!(loaded.shape(), original.shape());
    assert_eq!(loaded.numel(), original.numel());
    assert_eq!(loaded.quantiles(&[0.25, 0.5, 0.75]), expected);
}

#[test]
fn runtime_dtype_autodetection_from_bytes() {
    let mut f32_digest = make_f32_digest();
    let f32_bytes = f32_digest.to_bytes().expect("f32 serialization failed");
    match monatq::from_bytes(&f32_bytes).expect("f32 autodetection failed") {
        AnyTensorDigest::TDigestF32(mut loaded) => {
            assert_eq!(loaded.quantile(0.5), f32_digest.quantile(0.5));
        }
        other => panic!(
            "expected a TDigest f32 snapshot, got {} {}",
            other.kernel_name(),
            other.dtype_name()
        ),
    }

    let mut i32_digest = make_i32_digest();
    let i32_bytes = i32_digest.to_bytes().expect("i32 serialization failed");
    match monatq::from_bytes(&i32_bytes).expect("i32 autodetection failed") {
        AnyTensorDigest::TDigestI32(mut loaded) => {
            assert_eq!(loaded.quantile(0.5), i32_digest.quantile(0.5));
        }
        other => panic!(
            "expected a TDigest i32 snapshot, got {} {}",
            other.kernel_name(),
            other.dtype_name()
        ),
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
        TensorDigest::<f32, monatq::TDigest>::from_bytes(&file_bytes).expect("from_bytes failed");
    assert_eq!(loaded_from_file_bytes.quantile(0.5), expected);

    let memory_bytes = original.to_bytes().expect("to_bytes failed");
    std::fs::write(&path, memory_bytes).expect("write failed");
    let loaded_from_memory_bytes = monatq::load(&path).expect("load failed");
    std::fs::remove_file(&path).ok();
    match loaded_from_memory_bytes {
        AnyTensorDigest::TDigestF32(mut loaded) => assert_eq!(loaded.quantile(0.5), expected),
        other => panic!(
            "expected a TDigest f32 snapshot, got {} {}",
            other.kernel_name(),
            other.dtype_name()
        ),
    }
}

#[test]
fn to_bytes_flushes_pending_data() {
    let mut digest = TensorDigest::<f32, monatq::TDigest>::new(&[1]);
    for value in 0..10 {
        digest.update(&[value as f32]).unwrap();
    }
    assert_eq!(digest.total_weight(0).unwrap(), 0);

    let bytes = digest.to_bytes().expect("serialization failed");
    assert_eq!(digest.total_weight(0).unwrap(), 10);

    let mut loaded =
        TensorDigest::<f32, monatq::TDigest>::from_bytes(&bytes).expect("deserialization failed");
    assert_eq!(loaded.total_weight(0).unwrap(), 10);
    assert_eq!(loaded.quantile(0.5), digest.quantile(0.5));
}

#[test]
fn typed_from_bytes_rejects_dtype_mismatch() {
    let mut digest = make_f32_digest();
    let bytes = digest.to_bytes().expect("serialization failed");
    let error = TensorDigest::<i32, monatq::TDigest>::from_bytes(&bytes)
        .expect_err("expected a dtype mismatch");

    assert!(error.is_invalid_snapshot(), "unexpected error: {error}");
    assert!(error.to_string().contains("dtype mismatch"));
}

#[test]
fn invalid_byte_inputs_are_rejected() {
    for bytes in [&[][..], &[1, 2, 3][..]] {
        let error = monatq::from_bytes(bytes).expect_err("invalid input unexpectedly loaded");
        assert!(error.is_invalid_snapshot(), "unexpected error: {error}");
    }

    let empty_payload = zstd::encode_all(&[][..], 3).expect("compression failed");
    let empty_error =
        monatq::from_bytes(&empty_payload).expect_err("empty payload unexpectedly loaded");
    assert!(
        empty_error.is_invalid_snapshot(),
        "unexpected error: {empty_error}"
    );

    let unknown_payload = zstd::encode_all(&[99][..], 3).expect("compression failed");
    let unknown_error =
        monatq::from_bytes(&unknown_payload).expect_err("unknown dtype unexpectedly loaded");
    assert!(
        unknown_error.is_invalid_snapshot(),
        "unexpected error: {unknown_error}"
    );
    assert!(
        unknown_error.to_string().contains("leading tag 99"),
        "{unknown_error}"
    );

    let mut valid_digest = make_f32_digest();
    let mut truncated = valid_digest.to_bytes().expect("serialization failed");
    truncated.truncate(truncated.len() / 2);
    let truncated_error = TensorDigest::<f32, monatq::TDigest>::from_bytes(&truncated)
        .expect_err("truncated snapshot unexpectedly loaded");
    assert!(
        truncated_error.is_invalid_snapshot(),
        "unexpected error: {truncated_error}"
    );
}

#[test]
fn autodetection_identifies_the_kernel_as_well_as_the_dtype() {
    // All four combinations must round-trip through the untyped loader. Without kernel
    // detection a RankKnot snapshot would be read as a t-digest whose first byte happened
    // to be 0x52, or rejected outright.
    let mut rk_f32 = TensorDigest::<f32, monatq::RankKnot>::new(&[2, 2]);
    let mut rk_i32 = TensorDigest::<i32, monatq::RankKnot>::new(&[2, 2]);
    let mut td_f32 = TensorDigest::<f32, monatq::TDigest>::new(&[2, 2]);
    let mut td_i32 = TensorDigest::<i32, monatq::TDigest>::new(&[2, 2]);
    for step in 0..300 {
        let f = step as f32 * 0.25;
        let i = step - 150;
        rk_f32.update(&[f, -f, f * 2.0, 1.0]).unwrap();
        rk_i32.update(&[i, -i, i * 2, 1]).unwrap();
        td_f32.update(&[f, -f, f * 2.0, 1.0]).unwrap();
        td_i32.update(&[i, -i, i * 2, 1]).unwrap();
    }

    let cases: [(Vec<u8>, &str, &str); 4] = [
        (rk_f32.to_bytes().unwrap(), "RankKnot", "f32"),
        (rk_i32.to_bytes().unwrap(), "RankKnot", "i32"),
        (td_f32.to_bytes().unwrap(), "TDigest", "f32"),
        (td_i32.to_bytes().unwrap(), "TDigest", "i32"),
    ];
    for (bytes, kernel, dtype) in cases {
        let loaded = monatq::from_bytes(&bytes).expect("autodetection failed");
        assert_eq!(loaded.kernel_name(), kernel);
        assert_eq!(loaded.dtype_name(), dtype);
        assert_eq!(loaded.shape(), &[2, 2]);
    }

    // The detected digest must actually be usable, not just correctly labelled.
    match monatq::from_bytes(&rk_f32.to_bytes().unwrap()).unwrap() {
        AnyTensorDigest::RankKnotF32(mut d) => {
            assert_eq!(d.quantile(0.5), rk_f32.quantile(0.5));
        }
        other => panic!("expected RankKnot f32, got {}", other.kernel_name()),
    }
}

#[test]
fn a_payload_matching_no_kernel_is_rejected_with_a_useful_message() {
    let junk = zstd::encode_all(&[200_u8, 1, 2, 3, 4, 5, 6, 7, 8][..], 3).unwrap();
    let error = monatq::from_bytes(&junk).expect_err("unknown tag must be rejected");
    assert!(error.is_invalid_snapshot(), "unexpected error: {error}");
    assert!(
        error.to_string().contains("matches no known kernel"),
        "{error}"
    );
}
