//! The error surface itself: messages, classification, and the source chain.
use monatq::{QuantileSpine, RankKnot, TensorDigest};
use std::error::Error as _;

#[test]
fn display_does_not_repeat_the_source() {
    // A genuine I/O failure keeps its cause in `source()`, not in `Display`.
    let error = TensorDigest::<f32, monatq::TDigest>::load("/nonexistent/monatq/path.bin")
        .expect_err("missing file must fail");
    let message = error.to_string();
    let source = error.source().expect("io errors carry a cause").to_string();
    assert_eq!(message, "snapshot or file access failed");
    assert!(
        !message.contains(&source),
        "Display repeats its source: {message:?} contains {source:?}"
    );
    assert_eq!(error.io_kind(), Some(std::io::ErrorKind::NotFound));
    assert!(!error.is_invalid_snapshot());
}

#[test]
fn malformed_bytes_are_snapshot_errors_not_io_errors() {
    let error = TensorDigest::<f32, RankKnot>::from_bytes(&[9, 9, 9]).expect_err("must fail");
    assert!(error.is_invalid_snapshot(), "unexpected error: {error}");
    // The bytes were read fine; nothing about this is an I/O condition.
    assert_eq!(error.io_kind(), None);
    assert!(error.source().is_none());
    assert!(error.to_string().starts_with("invalid snapshot: "));
}

#[test]
fn error_messages_name_the_kernel_and_the_operation() {
    let mut spine = TensorDigest::<f32, QuantileSpine>::new(&[1]);
    let error = spine.merge_all().expect_err("spine cannot merge");
    assert_eq!(
        error.to_string(),
        "QuantileSpine does not implement merge_all"
    );
    assert!(error.is_unsupported());
}

#[test]
fn errors_convert_into_io_errors_for_io_shaped_callers() {
    let mut spine = TensorDigest::<f32, QuantileSpine>::new(&[1]);
    let converted: std::io::Error = spine.merge_all().unwrap_err().into();
    assert_eq!(converted.kind(), std::io::ErrorKind::InvalidInput);

    let snapshot_error: std::io::Error = TensorDigest::<f32, RankKnot>::from_bytes(&[9, 9, 9])
        .err()
        .unwrap()
        .into();
    assert_eq!(snapshot_error.kind(), std::io::ErrorKind::InvalidData);
}
