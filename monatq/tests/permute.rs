//! Permute reorders axes while every element and block keeps its history, including rows that
//! are still waiting to be compressed.
use monatq::{
    BlockConfig, DigestKernel, Error, RankKnot, RankKnotConfig, TDigest, TDigestConfig,
    TensorDigest,
};

fn rankknot(shape: &[usize], blocks: BlockConfig) -> TensorDigest<f32, RankKnot> {
    TensorDigest::with_block_config(shape, RankKnotConfig { buffer_capacity: 8 }, blocks).unwrap()
}

fn tdigest(shape: &[usize], blocks: BlockConfig) -> TensorDigest<f32, TDigest> {
    TensorDigest::with_block_config(
        shape,
        TDigestConfig {
            compression: 20,
            buffer_capacity: Some(8),
        },
        blocks,
    )
    .unwrap()
}

/// Reference transpose of a row-major tensor.
fn transpose(data: &[f32], shape: &[usize], axes: &[usize]) -> Vec<f32> {
    let new_shape: Vec<usize> = axes.iter().map(|&a| shape[a]).collect();
    let total: usize = shape.iter().product();
    let mut out = vec![0.0; total];
    for (new_flat, slot) in out.iter_mut().enumerate() {
        let mut rest = new_flat;
        let mut old_flat = 0;
        for (dim, &axis) in axes.iter().enumerate().rev() {
            let coord = rest % new_shape[dim];
            rest /= new_shape[dim];
            let stride: usize = shape[axis + 1..].iter().product();
            old_flat += coord * stride;
        }
        *slot = data[old_flat];
    }
    out
}

fn elementwise_permute_moves_every_element<K: DigestKernel<f32>>(mut digest: TensorDigest<f32, K>) {
    let first: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let second: Vec<f32> = first.iter().map(|value| value + 100.0).collect();
    digest.update(&first).unwrap();
    digest.update(&second).unwrap();

    digest.permute(&[2, 0, 1]).unwrap();
    assert_eq!(digest.shape(), &[4, 2, 3]);
    assert_eq!(digest.block_shape(), &[4, 2, 3]);
    assert_eq!(digest.block_config(), BlockConfig::Elementwise);
    assert_eq!(
        digest.quantile(0.0),
        transpose(&first, &[2, 3, 4], &[2, 0, 1])
    );
    assert_eq!(
        digest.quantile(1.0),
        transpose(&second, &[2, 3, 4], &[2, 0, 1])
    );

    // New samples arrive in the permuted layout and line up with the moved history.
    let third = transpose(&first, &[2, 3, 4], &[2, 0, 1]);
    digest.update(&third).unwrap();
    digest.flush();
    assert_eq!(digest.total_weight(5).unwrap(), 3);
    assert_eq!(digest.quantile(0.0), third);
}

#[test]
fn elementwise_permute_moves_every_element_on_both_kernels() {
    elementwise_permute_moves_every_element(rankknot(&[2, 3, 4], BlockConfig::Elementwise));
    elementwise_permute_moves_every_element(tdigest(&[2, 3, 4], BlockConfig::Elementwise));
}

fn negative_axes_and_round_trip<K: DigestKernel<f32>>(mut digest: TensorDigest<f32, K>) {
    let sample: Vec<f32> = (0..6).map(|value| value as f32).collect();
    digest.update(&sample).unwrap();
    digest.permute(&[-1, -2]).unwrap();
    assert_eq!(digest.shape(), &[3, 2]);
    assert_eq!(digest.quantile(1.0), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    digest.permute(&[1, 0]).unwrap();
    assert_eq!(digest.shape(), &[2, 3]);
    assert_eq!(digest.quantile(1.0), sample);
}

#[test]
fn negative_axes_and_inverse_permutation_restore_the_digest() {
    negative_axes_and_round_trip(rankknot(&[2, 3], BlockConfig::Elementwise));
    negative_axes_and_round_trip(tdigest(&[2, 3], BlockConfig::Elementwise));
}

fn blocks_follow_their_axis<K: DigestKernel<f32>>(mut digest: TensorDigest<f32, K>) {
    // Rows of [2, 8] pooled in groups of four along axis 1: blocks [[0..4), [4..8)] / [[8..12), [12..16)].
    let sample: Vec<f32> = (0..16).map(|value| value as f32).collect();
    digest.update(&sample).unwrap();
    assert_eq!(digest.block_shape(), &[2, 2]);
    assert_eq!(digest.quantile(1.0), vec![3.0, 7.0, 11.0, 15.0]);

    digest.permute(&[1, 0]).unwrap();
    assert_eq!(digest.shape(), &[8, 2]);
    assert_eq!(digest.block_shape(), &[2, 2]);
    assert_eq!(
        digest.block_config(),
        BlockConfig::Size { size: 4, axis: 0 }
    );
    // Block (row, group) moved to (group, row).
    assert_eq!(digest.quantile(1.0), vec![3.0, 11.0, 7.0, 15.0]);
    assert_eq!(digest.quantile(0.0), vec![0.0, 8.0, 4.0, 12.0]);

    // The next sample uses the permuted layout and lands in the matching blocks.
    let next = transpose(
        &sample.iter().map(|v| v + 100.0).collect::<Vec<_>>(),
        &[2, 8],
        &[1, 0],
    );
    digest.update(&next).unwrap();
    digest.flush();
    assert_eq!(digest.total_weight(0).unwrap(), 8);
    assert_eq!(digest.quantile(1.0), vec![103.0, 111.0, 107.0, 115.0]);
}

#[test]
fn blocks_follow_their_axis_on_both_kernels() {
    blocks_follow_their_axis(rankknot(&[2, 8], BlockConfig::block_size(4, -1)));
    blocks_follow_their_axis(tdigest(&[2, 8], BlockConfig::block_size(4, -1)));
}

fn pending_rows_move_with_elements_when_blocks_do_not<K: DigestKernel<f32>>(
    mut digest: TensorDigest<f32, K>,
) {
    // Block order is unchanged by this permutation ([2, 1] -> [1, 2] compact), but the
    // elements still move, so buffered rows must be transposed before they are compressed.
    let sample: Vec<f32> = (0..8).map(|value| value as f32).collect();
    digest.update(&sample).unwrap();
    digest.permute(&[1, 0]).unwrap();
    assert_eq!(digest.shape(), &[4, 2]);
    assert_eq!(digest.block_shape(), &[1, 2]);
    assert_eq!(digest.quantile(0.0), vec![0.0, 4.0]);
    assert_eq!(digest.quantile(1.0), vec![3.0, 7.0]);
}

#[test]
fn pending_rows_are_reordered_even_when_block_order_is_unchanged() {
    pending_rows_move_with_elements_when_blocks_do_not(rankknot(
        &[2, 4],
        BlockConfig::block_size(4, 1),
    ));
    pending_rows_move_with_elements_when_blocks_do_not(tdigest(
        &[2, 4],
        BlockConfig::block_size(4, 1),
    ));
}

fn invalid_axes_leave_the_digest_unchanged<K: DigestKernel<f32>>(mut digest: TensorDigest<f32, K>) {
    let sample: Vec<f32> = (0..6).map(|value| value as f32).collect();
    digest.update(&sample).unwrap();
    for axes in [&[0][..], &[0, 0], &[0, 2], &[-3, 0], &[0, 1, 0]] {
        let error = digest.permute(axes).unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidConfig {
                parameter: "axes",
                ..
            }
        ));
        assert!(!error.is_incompatible_layout());
    }
    assert_eq!(digest.shape(), &[2, 3]);
    assert_eq!(digest.quantile(1.0), sample);
}

#[test]
fn invalid_axes_are_rejected_without_changes() {
    invalid_axes_leave_the_digest_unchanged(rankknot(&[2, 3], BlockConfig::Elementwise));
    invalid_axes_leave_the_digest_unchanged(tdigest(&[2, 3], BlockConfig::Elementwise));
}

#[test]
fn scalar_and_empty_digests_permute() {
    let mut scalar = TensorDigest::<f32>::new(&[]);
    scalar.update(&[7.0]).unwrap();
    scalar.permute(&[]).unwrap();
    assert_eq!(scalar.quantile(0.5), vec![7.0]);
    assert!(scalar.permute(&[0]).is_err());

    let mut empty = TensorDigest::<f32, TDigest>::new(&[0, 4]);
    empty.permute(&[1, 0]).unwrap();
    assert_eq!(empty.shape(), &[4, 0]);
    assert_eq!(empty.block_count(), 0);
}

#[test]
fn permuted_digests_survive_a_snapshot() {
    let mut digest = rankknot(&[2, 8], BlockConfig::blocks_per_axis(2, 1));
    digest
        .update(&(0..16).map(|value| value as f32).collect::<Vec<_>>())
        .unwrap();
    digest.permute(&[1, 0]).unwrap();
    let expected = digest.quantile(0.5);
    let mut restored =
        TensorDigest::<f32, RankKnot>::from_bytes(&digest.to_bytes().unwrap()).unwrap();
    assert_eq!(restored.shape(), &[8, 2]);
    assert_eq!(
        restored.block_config(),
        BlockConfig::Count { count: 2, axis: 0 }
    );
    assert_eq!(restored.quantile(0.5), expected);
}

/// A permuted digest must match a fresh digest built in the permuted layout and fed the
/// transposed samples, including blocks along non-trivial axes and rows still buffered.
fn matches_fresh_digest<K: DigestKernel<f32>>(
    make: impl Fn(&[usize], BlockConfig) -> TensorDigest<f32, K>,
) {
    let cases: [(&[usize], BlockConfig, &[isize]); 6] = [
        (&[2, 3, 8], BlockConfig::block_size(4, 2), &[2, 0, 1]),
        (&[2, 3, 8], BlockConfig::block_size(4, 2), &[1, 2, 0]),
        (&[2, 5, 3], BlockConfig::blocks_per_axis(2, 1), &[-1, 0, 1]),
        (&[3, 7], BlockConfig::block_size(3, 1), &[1, 0]),
        (&[2, 3, 4], BlockConfig::Elementwise, &[2, 1, 0]),
        (&[4, 6], BlockConfig::block_size(2, 0), &[0, 1]),
    ];
    for (shape, blocks, axes) in cases {
        let ndim = shape.len() as isize;
        let resolved: Vec<usize> = axes
            .iter()
            .map(|&axis| axis.rem_euclid(ndim) as usize)
            .collect();
        let new_shape: Vec<usize> = resolved.iter().map(|&axis| shape[axis]).collect();
        let numel: usize = shape.iter().product();
        let sample = |step: usize| -> Vec<f32> {
            (0..numel)
                .map(|i| ((i * 7 + step * 13) % 31) as f32 - step as f32 * 0.5)
                .collect()
        };

        let mut permuted = make(shape, blocks);
        for step in 0..3 {
            permuted.update(&sample(step)).unwrap();
        }
        permuted.permute(axes).unwrap();
        for step in 3..5 {
            let data = transpose(&sample(step), shape, &resolved);
            permuted.update(&data).unwrap();
        }

        let new_blocks = permuted.block_config();
        let mut fresh = make(&new_shape, new_blocks);
        for step in 0..5 {
            fresh
                .update(&transpose(&sample(step), shape, &resolved))
                .unwrap();
        }

        assert_eq!(permuted.shape(), fresh.shape(), "{shape:?} {axes:?}");
        assert_eq!(
            permuted.block_shape(),
            fresh.block_shape(),
            "{shape:?} {axes:?}"
        );
        for q in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert_eq!(
                permuted.quantile(q),
                fresh.quantile(q),
                "{shape:?} {axes:?} q={q}"
            );
        }
    }
}

#[test]
fn permuted_digest_equals_a_fresh_digest_fed_transposed_samples() {
    matches_fresh_digest(rankknot);
    matches_fresh_digest(tdigest);
}
