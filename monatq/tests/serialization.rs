use monatq::TensorDigest;

/// Build a small digest with known data.
fn make_digest() -> TensorDigest<f32> {
    let mut td = TensorDigest::<f32>::new(&[1], 100);
    for i in 0..500u32 {
        td.update(&[i as f32 * 0.002]);
    }
    td
}

#[test]
fn roundtrip() {
    let mut original = make_digest();
    let qs = &[0.1, 0.5, 0.9];
    let original_quantiles: Vec<Vec<f32>> = qs.iter().map(|&q| original.quantile(q)).collect();
    let shape = original.shape().to_vec();
    let numel = original.numel();

    let path = std::env::temp_dir().join("monatq_roundtrip_test.bin");
    original.save(&path).expect("save failed");

    let mut loaded = TensorDigest::<f32>::load(&path).expect("load failed");
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.shape(), shape.as_slice(), "shape mismatch");
    assert_eq!(loaded.numel(), numel, "numel mismatch");

    for &q in qs {
        let orig_q = original.quantile(q);
        let load_q = loaded.quantile(q);
        assert_eq!(orig_q, load_q, "quantile {q} mismatch after roundtrip");
    }

    // Verify the previously-recorded quantiles also match.
    for (&q, orig_q) in qs.iter().zip(&original_quantiles) {
        let load_q = loaded.quantile(q);
        assert_eq!(load_q, *orig_q, "quantile {q} differs from pre-save value");
    }
}

#[test]
fn corruption_detected() {
    let mut td = make_digest();
    let path = std::env::temp_dir().join("monatq_corruption_test.bin");
    td.save(&path).expect("save failed");

    // Truncate to half the file — always breaks the zstd frame or bincode length prefix.
    let mut bytes = std::fs::read(&path).expect("read failed");
    bytes.truncate(bytes.len() / 2);
    std::fs::write(&path, &bytes).expect("write-back failed");

    let result = TensorDigest::<f32>::load(&path);
    std::fs::remove_file(&path).ok();

    assert!(result.is_err(), "expected Err for corrupted file, got Ok");
}
