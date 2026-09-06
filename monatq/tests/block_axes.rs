use monatq::{BlockConfig, DigestKernel, RankKnot, TDigest, TensorDigest};

fn exercise<K: DigestKernel<f32>>() {
    let shape = [2, 5, 3];
    let values: Vec<f32> = (0..30).map(|n| n as f32).collect();
    for axis in 0..3isize {
        for constructor in [
            BlockConfig::block_size,
            BlockConfig::blocks_per_axis,
            BlockConfig::new,
        ] {
            let mut positive =
                TensorDigest::<f32, K>::with_blocks(&shape, constructor(2, axis)).unwrap();
            let mut negative =
                TensorDigest::<f32, K>::with_blocks(&shape, constructor(2, axis - 3)).unwrap();
            assert_eq!(negative.block_axis(), axis as usize);
            assert_eq!(negative.shape(), positive.shape());
            positive.update(&values).unwrap();
            negative.update(&values).unwrap();
            assert_eq!(
                negative.quantiles(&[0., 0.5, 1.]),
                positive.quantiles(&[0., 0.5, 1.])
            );
            // Normalized axes produce identical stored layouts, regardless of spelling.
            let bytes = negative.to_bytes().unwrap();
            assert_eq!(bytes, positive.to_bytes().unwrap());
            let mut restored = TensorDigest::<f32, K>::from_bytes(&bytes).unwrap();
            assert_eq!(restored.block_axis(), axis as usize);
            restored.update(&values).unwrap();
            negative.update(&values).unwrap();
            assert_eq!(restored.quantile(0.5), negative.quantile(0.5));
        }
    }
    for axis in [isize::MIN, -4, 3, isize::MAX] {
        for config in [
            BlockConfig::block_size(2, axis),
            BlockConfig::blocks_per_axis(2, axis),
        ] {
            assert!(TensorDigest::<f32, K>::with_blocks(&shape, config).is_err());
        }
    }
    for axis in [-1, 0] {
        assert!(
            TensorDigest::<f32, K>::with_blocks(&[], BlockConfig::block_size(2, axis)).is_err()
        );
    }
    assert_eq!(TensorDigest::<f32, K>::new(&[]).block_count(), 1);
    let empty =
        TensorDigest::<f32, K>::with_blocks(&[2, 0], BlockConfig::block_size(2, -1)).unwrap();
    assert_eq!(empty.block_axis(), 1);
    assert_eq!(empty.block_count(), 0);
}

#[test]
fn rankknot_signed_axes() {
    exercise::<RankKnot>();
}

#[test]
fn tdigest_signed_axes() {
    exercise::<TDigest>();
}
