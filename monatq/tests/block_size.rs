use monatq::{BlockConfig, DigestKernel, RankKnot, TDigest, TensorDigest};

fn exercise<K: DigestKernel<f32>>() {
    // Strided groups: each of the two inner coordinates remains independent.
    let mut d =
        TensorDigest::<f32, K>::with_blocks(&[2, 5, 2], BlockConfig::block_size(2, 1)).unwrap();
    assert_eq!(d.shape(), &[2, 3, 2]);
    assert_eq!(d.block_size(), Some(2));
    assert_eq!(d.blocks_per_axis(), 3);
    let values: Vec<f32> = (0..20).map(|x| x as f32).collect();
    d.update(&values).unwrap();
    d.update(&values).unwrap();
    assert_eq!(
        d.quantile(0.0),
        vec![0., 1., 4., 5., 8., 9., 10., 11., 14., 15., 18., 19.]
    );
    assert_eq!(
        d.quantile(1.0),
        vec![2., 3., 6., 7., 8., 9., 12., 13., 16., 17., 18., 19.]
    );
    for i in 0..12 {
        assert_eq!(d.total_weight(i).unwrap(), if i % 6 >= 4 { 2 } else { 4 });
    }
    let mut restored = TensorDigest::<f32, K>::from_bytes(&d.to_bytes().unwrap()).unwrap();
    assert_eq!(restored.block_size(), Some(2));
    assert_eq!(restored.shape(), d.shape());
    restored.update(&values).unwrap();
    assert_eq!(restored.total_weight(4).unwrap(), 3);
    let mut merged = restored.merge_cells(&[0, 2, 4]).unwrap();
    assert_eq!(merged.total_weight(0).unwrap(), 15);
    assert_eq!(merged.cell_quantiles(0, &[0., 1.]).unwrap(), vec![0., 8.]);

    let tail =
        TensorDigest::<f32, K>::with_blocks(&[32, 257], BlockConfig::block_size(128, 1)).unwrap();
    assert_eq!(tail.shape(), &[32, 3]);
    for size in [1, 5, 99, usize::MAX] {
        let d =
            TensorDigest::<f32, K>::with_blocks(&[2, 5], BlockConfig::block_size(size, 1)).unwrap();
        assert_eq!(d.shape(), &[2, 5usize.div_ceil(size)]);
    }
    assert!(TensorDigest::<f32, K>::with_blocks(&[2, 5], BlockConfig::block_size(0, 1)).is_err());
    assert!(TensorDigest::<f32, K>::with_blocks(&[2, 5], BlockConfig::block_size(2, 2)).is_err());
    let empty =
        TensorDigest::<f32, K>::with_blocks(&[2, 0], BlockConfig::block_size(2, 1)).unwrap();
    assert_eq!(empty.block_count(), 0);
}

#[test]
fn rankknot_fixed_size_groups() {
    exercise::<RankKnot>();
}

#[test]
fn tdigest_fixed_size_groups() {
    exercise::<TDigest>();
}
