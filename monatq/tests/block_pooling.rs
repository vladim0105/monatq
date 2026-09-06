use monatq::{BlockConfig, Error, RankKnot, RankKnotConfig, TDigest, TDigestConfig, TensorDigest};

fn sample(step: usize) -> Vec<f32> {
    (0..20).map(|i| (step * 100 + i) as f32).collect()
}

#[test]
fn k_zero_matches_existing_behavior_for_both_kernels() {
    let mut rk_old = TensorDigest::<f32, RankKnot>::with_config(
        &[2, 5, 2],
        RankKnotConfig { buffer_capacity: 3 },
    );
    let mut rk_new = TensorDigest::<f32, RankKnot>::with_block_config(
        &[2, 5, 2],
        RankKnotConfig { buffer_capacity: 3 },
        BlockConfig::new(0, 1),
    )
    .unwrap();
    let mut td_old =
        TensorDigest::<f32, TDigest>::with_config(&[2, 5, 2], TDigestConfig { compression: 40 });
    let mut td_new = TensorDigest::<f32, TDigest>::with_block_config(
        &[2, 5, 2],
        TDigestConfig { compression: 40 },
        BlockConfig::new(0, 1),
    )
    .unwrap();
    for step in 0..17 {
        let row = sample(step);
        rk_old.update(&row).unwrap();
        rk_new.update(&row).unwrap();
        td_old.update(&row).unwrap();
        td_new.update(&row).unwrap();
    }
    for q in [0.0, 0.25, 0.5, 1.0] {
        assert_eq!(rk_old.quantile(q), rk_new.quantile(q));
        assert_eq!(td_old.quantile(q), td_new.quantile(q));
    }
    // Snapshots preserve the selected axis even in element-wise mode.
    let rk_loaded = TensorDigest::<f32, RankKnot>::from_bytes(&rk_new.to_bytes().unwrap()).unwrap();
    let td_loaded = TensorDigest::<f32, TDigest>::from_bytes(&td_new.to_bytes().unwrap()).unwrap();
    assert_eq!(rk_loaded.block_axis(), 1);
    assert_eq!(td_loaded.block_axis(), 1);
}

fn exercise_blocks<K: monatq::DigestKernel<f32>>(mut d: TensorDigest<f32, K>) {
    assert_eq!(d.shape(), &[2, 2, 2]);
    assert_eq!(d.block_count(), 8);
    assert_eq!(d.input_shape(), &[2, 5, 2]);
    assert_eq!(d.input_numel(), 20);
    assert_eq!(d.block_axis(), 1);
    assert_eq!(d.blocks_per_axis(), 2);
    for step in 0..2 {
        d.update(&sample(step)).unwrap();
    }
    let mut outlier = sample(2);
    outlier[8] = 1.0e9;
    d.update(&outlier).unwrap();

    // The axis is split into balanced lengths three and two.
    assert_eq!(d.total_weight(0).unwrap(), 9);
    assert_eq!(d.total_weight(2).unwrap(), 6);
    assert_eq!(d.quantile(0.5).len(), 8);
    assert_eq!(d.cell_quantiles(2, &[0.0, 1.0]).unwrap(), vec![6.0, 1.0e9]);
    assert!(matches!(
        d.cell_quantiles(8, &[0.0]),
        Err(Error::IndexOutOfBounds { .. })
    ));

    let mut merged = d.merge_cells(&[0, 2]).unwrap();
    assert_eq!(merged.shape(), &[1]);
    assert_eq!(merged.input_shape(), &[1]);
    assert_eq!(merged.total_weight(0).unwrap(), 15);
    assert_eq!(
        merged.cell_quantiles(0, &[0.0, 1.0]).unwrap(),
        vec![0.0, 1.0e9]
    );
}

#[test]
fn rankknot_pools_nonlast_axis_and_partial_tail_across_updates() {
    exercise_blocks(
        TensorDigest::<f32, RankKnot>::with_blocks(&[2, 5, 2], BlockConfig::new(2, 1)).unwrap(),
    );
}

#[test]
fn tdigest_pools_nonlast_axis_and_partial_tail_across_updates() {
    exercise_blocks(
        TensorDigest::<f32, TDigest>::with_blocks(&[2, 5, 2], BlockConfig::new(2, 1)).unwrap(),
    );
}

fn scalar_roundtrip<K: monatq::DigestKernel<f32>>() {
    let mut d = TensorDigest::<f32, K>::new(&[]);
    d.update(&[7.0]).unwrap();
    let mut restored = TensorDigest::<f32, K>::from_bytes(&d.to_bytes().unwrap()).unwrap();
    assert_eq!(restored.shape(), &[]);
    assert_eq!(restored.quantile(0.5), vec![7.0]);
}

#[test]
fn scalar_snapshots_remain_supported() {
    scalar_roundtrip::<RankKnot>();
    scalar_roundtrip::<TDigest>();
}

fn zero_axis<K: monatq::DigestKernel<f32>>() {
    let mut digest =
        TensorDigest::<f32, K>::with_blocks(&[2, 0, 3], BlockConfig::new(16, 1)).unwrap();
    assert_eq!(digest.shape(), &[2, 0, 3]);
    assert_eq!(digest.block_count(), 0);
    digest.update(&[]).unwrap();
    assert!(digest.quantile(0.5).is_empty());
    let restored = TensorDigest::<f32, K>::from_bytes(&digest.to_bytes().unwrap()).unwrap();
    assert_eq!(restored.shape(), &[2, 0, 3]);
}

#[test]
fn zero_length_axis_is_safe() {
    zero_axis::<RankKnot>();
    zero_axis::<TDigest>();
}

fn integer_blocks<K: monatq::DigestKernel<i32>>() {
    let mut d = TensorDigest::<i32, K>::with_blocks(&[2, 5], BlockConfig::new(2, 1)).unwrap();
    d.update(&[0, 0, 0, 1000, 5, -10, -10, -10, -10, -5])
        .unwrap();
    assert_eq!(d.quantile(1.0), vec![0.0, 1000.0, -10.0, -5.0]);
    assert_eq!(d.total_weight(0).unwrap(), 3);
    assert_eq!(d.total_weight(1).unwrap(), 2);
    let filtered = d.without_zeros().unwrap();
    assert_eq!(filtered.shape(), &[2, 2]);
    let mut restored = TensorDigest::<i32, K>::from_bytes(&d.to_bytes().unwrap()).unwrap();
    restored.update(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();
    d.update(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).unwrap();
    assert_eq!(restored.quantile(1.0), d.quantile(1.0));
    assert_eq!(restored.total_weight(0).unwrap(), 6);
}

#[test]
fn integer_blocks_pool_raw_values_and_roundtrip() {
    integer_blocks::<RankKnot>();
    integer_blocks::<TDigest>();
}

#[test]
fn block_config_is_validated_without_panics() {
    let result = TensorDigest::<f32, RankKnot>::with_blocks(&[2, 3], BlockConfig::new(2, 2));
    assert!(matches!(result, Err(Error::InvalidConfig { .. })));

    let overflow =
        TensorDigest::<f32, RankKnot>::with_blocks(&[usize::MAX, 2], BlockConfig::new(1, 0));
    assert!(matches!(overflow, Err(Error::InvalidConfig { .. })));
}

fn assert_block_count_modes<K: monatq::DigestKernel<f32>>() {
    let mut identity =
        TensorDigest::<f32, K>::with_blocks(&[2, 5], BlockConfig::new(0, 1)).unwrap();
    let mut one = TensorDigest::<f32, K>::with_blocks(&[2, 5], BlockConfig::new(1, 1)).unwrap();
    let mut clamped =
        TensorDigest::<f32, K>::with_blocks(&[2, 5], BlockConfig::new(99, 1)).unwrap();
    let row: Vec<f32> = (0..10).map(|v| v as f32).collect();
    identity.update(&row).unwrap();
    one.update(&row).unwrap();
    clamped.update(&row).unwrap();

    assert_eq!(identity.shape(), &[2, 5]);
    assert_eq!(identity.blocks_per_axis(), 0);
    assert_eq!(identity.quantile(1.0), row);
    assert_eq!(one.shape(), &[2, 1]);
    assert_eq!(one.total_weight(0).unwrap(), 5);
    assert_eq!(one.total_weight(1).unwrap(), 5);
    assert_eq!(clamped.shape(), &[2, 5]);
    assert_eq!(clamped.blocks_per_axis(), 99);
    assert_eq!(clamped.quantile(1.0), row);
}

#[test]
fn zero_one_and_overlong_requested_counts_have_defined_semantics() {
    assert_block_count_modes::<RankKnot>();
    assert_block_count_modes::<TDigest>();
}

fn assert_large_balanced_layout<K: monatq::DigestKernel<f32>>() {
    let shape = [256, 129, 2];
    let mut digest = TensorDigest::<f32, K>::with_blocks(&shape, BlockConfig::new(16, 1)).unwrap();
    assert_eq!(digest.shape(), &[256, 16, 2]);
    assert_eq!(digest.block_count(), 256 * 16 * 2);
    assert_eq!(digest.input_shape(), &shape);
    assert_eq!(digest.input_numel(), 256 * 129 * 2);

    let mut row = vec![0.0; digest.input_numel()];
    for outer in 0..256 {
        for axis in 0..129 {
            for inner in 0..2 {
                row[(outer * 129 + axis) * 2 + inner] = axis as f32;
            }
        }
    }
    row[8 * 2] = 10_000.0;
    digest.update(&row).unwrap();

    // The first compact block has nine axis positions; all remaining blocks have eight.
    assert_eq!(digest.total_weight(0).unwrap(), 9);
    for block_axis in 1..16 {
        assert_eq!(digest.total_weight(block_axis * 2).unwrap(), 8);
    }
    assert_eq!(
        digest.cell_quantiles(0, &[0.0, 1.0]).unwrap(),
        vec![0.0, 10_000.0]
    );
    assert_eq!(
        digest.cell_quantiles(2, &[0.0, 1.0]).unwrap(),
        vec![9.0, 16.0]
    );
}

#[test]
fn axis_129_is_partitioned_into_sixteen_balanced_blocks() {
    assert_large_balanced_layout::<RankKnot>();
    assert_large_balanced_layout::<TDigest>();
}

fn assert_uneven_block_merges_and_continuation<K: monatq::DigestKernel<f32>>() {
    let mut blocked = TensorDigest::<f32, K>::with_blocks(&[5], BlockConfig::new(2, 0)).unwrap();
    let mut reference = TensorDigest::<f32, K>::new(&[1]);
    for _ in 0..20 {
        blocked.update(&[0.0, 0.0, 0.0, 10.0, 10.0]).unwrap();
        for value in [0.0, 0.0, 0.0, 10.0, 10.0] {
            reference.update(&[value]).unwrap();
        }
    }

    assert_eq!(blocked.total_weight(0).unwrap(), 60);
    assert_eq!(blocked.total_weight(1).unwrap(), 40);
    let mut merged = blocked.merge_all().unwrap();
    assert_eq!(merged.shape(), &[1]);
    assert_eq!(merged.input_shape(), &[1]);
    assert_eq!(merged.total_weight(0).unwrap(), 100);
    assert_eq!(
        merged.quantiles(&[0.0, 0.5, 1.0]),
        reference.quantiles(&[0.0, 0.5, 1.0])
    );

    // Continuation uses the represented observation count, not the source tensor update count.
    for _ in 0..25 {
        merged.update(&[10.0]).unwrap();
        reference.update(&[10.0]).unwrap();
    }
    merged.flush();
    reference.flush();
    assert_eq!(merged.total_weight(0).unwrap(), 125);
    let merged_q = merged.quantiles(&[0.0, 0.5, 1.0]);
    let reference_q = reference.quantiles(&[0.0, 0.5, 1.0]);
    assert_eq!(merged_q[0], reference_q[0]);
    assert_eq!(merged_q[2], reference_q[2]);
    assert!((merged_q[1][0] - reference_q[1][0]).abs() < 1.0);
}

#[test]
fn uneven_block_merges_preserve_weights_and_continue_for_both_kernels() {
    assert_uneven_block_merges_and_continuation::<RankKnot>();
    assert_uneven_block_merges_and_continuation::<TDigest>();
}

fn assert_compact_channel_and_all_merges<K: monatq::DigestKernel<f32>>() {
    let mut digest =
        TensorDigest::<f32, K>::with_blocks(&[2, 5, 2], BlockConfig::new(2, 1)).unwrap();
    digest.update(&sample(0)).unwrap();
    let channel = digest.merge_channels(&[1]).unwrap();
    assert_eq!(channel.total_weight(0).unwrap(), 10);
    let all = digest.merge_all().unwrap();
    assert_eq!(all.total_weight(0).unwrap(), 20);
}

#[test]
fn channel_and_all_merges_visit_compact_blocks_once() {
    assert_compact_channel_and_all_merges::<RankKnot>();
    assert_compact_channel_and_all_merges::<TDigest>();
}

#[test]
fn blocked_snapshot_roundtrip_preserves_layout_weights_and_queries() {
    let mut rk =
        TensorDigest::<f32, RankKnot>::with_blocks(&[2, 5, 2], BlockConfig::new(2, 1)).unwrap();
    let mut td =
        TensorDigest::<f32, TDigest>::with_blocks(&[2, 5, 2], BlockConfig::new(2, 1)).unwrap();
    for step in 0..13 {
        let row = sample(step);
        rk.update(&row).unwrap();
        td.update(&row).unwrap();
    }

    let rk_q = rk.quantiles(&[0.0, 0.5, 1.0]);
    let td_q = td.quantiles(&[0.0, 0.5, 1.0]);
    let mut rk2 = TensorDigest::<f32, RankKnot>::from_bytes(&rk.to_bytes().unwrap()).unwrap();
    let mut td2 = TensorDigest::<f32, TDigest>::from_bytes(&td.to_bytes().unwrap()).unwrap();
    for dshape in [rk2.shape(), td2.shape()] {
        assert_eq!(dshape, &[2, 2, 2]);
    }
    assert_eq!(rk2.block_axis(), 1);
    assert_eq!(td2.block_axis(), 1);
    assert_eq!(rk2.blocks_per_axis(), 2);
    assert_eq!(td2.blocks_per_axis(), 2);
    assert_eq!(rk2.input_shape(), &[2, 5, 2]);
    assert_eq!(td2.input_shape(), &[2, 5, 2]);
    assert_eq!(rk2.total_weight(6).unwrap(), 26);
    assert_eq!(td2.total_weight(6).unwrap(), 26);
    assert_eq!(rk2.quantiles(&[0.0, 0.5, 1.0]), rk_q);
    assert_eq!(td2.quantiles(&[0.0, 0.5, 1.0]), td_q);

    // Continued ingestion must use pooled weights and the original input geometry.
    rk2.update(&sample(13)).unwrap();
    td2.update(&sample(13)).unwrap();
    rk.update(&sample(13)).unwrap();
    td.update(&sample(13)).unwrap();
    assert_eq!(rk2.total_weight(0).unwrap(), 42);
    assert_eq!(td2.total_weight(6).unwrap(), 28);
    assert_eq!(
        rk2.quantiles(&[0.0, 0.5, 1.0]),
        rk.quantiles(&[0.0, 0.5, 1.0])
    );
    assert_eq!(
        td2.quantiles(&[0.0, 0.5, 1.0]),
        td.quantiles(&[0.0, 0.5, 1.0])
    );

    assert!(matches!(
        monatq::from_bytes(&rk.to_bytes().unwrap()).unwrap(),
        monatq::AnyTensorDigest::RankKnotF32(_)
    ));
    assert!(matches!(
        monatq::from_bytes(&td.to_bytes().unwrap()).unwrap(),
        monatq::AnyTensorDigest::TDigestF32(_)
    ));
}
