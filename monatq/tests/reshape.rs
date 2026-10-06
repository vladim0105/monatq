//! Reshape and remap keep full block state, including rows still waiting to be compressed.
use monatq::{
    BlockConfig, DigestKernel, Error, RankKnot, RankKnotConfig, TDigest, TDigestConfig,
    TensorDigest,
};

fn rankknot_blocks(shape: &[usize], blocks: BlockConfig) -> TensorDigest<f32, RankKnot> {
    TensorDigest::with_block_config(
        shape,
        RankKnotConfig { buffer_capacity: 8 },
        blocks,
    )
    .unwrap()
}

fn tdigest_blocks(shape: &[usize], blocks: BlockConfig) -> TensorDigest<f32, TDigest> {
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

fn elementwise_reshape_keeps_observations<K: DigestKernel<f32>>(mut digest: TensorDigest<f32, K>) {
    let sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    digest.update(&sample).unwrap();
    digest.reshape(&[3, 2]).unwrap();
    assert_eq!(digest.shape(), &[3, 2]);
    assert_eq!(digest.block_shape(), &[3, 2]);
    assert_eq!(digest.block_config(), BlockConfig::Elementwise);
    assert_eq!(digest.quantile(0.0), sample);
    assert_eq!(digest.quantile(1.0), sample);

    let error = digest.reshape(&[4]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "cannot reshape this digest: element count must be unchanged"
    );
    assert!(error.is_incompatible_layout());
    assert_eq!(digest.shape(), &[3, 2]);
    assert_eq!(digest.quantile(1.0), sample);

    digest.remap(&[6], BlockConfig::Elementwise).unwrap();
    assert_eq!(digest.shape(), &[6]);
    digest.update(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    assert_eq!(digest.quantile(0.0), sample);
    assert_eq!(
        digest.quantile(1.0),
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    );
    assert!(digest.update(&[1.0, 2.0]).is_err());
    assert_eq!(digest.numel(), 6);
}

#[test]
fn elementwise_reshape_keeps_observations_on_both_kernels() {
    elementwise_reshape_keeps_observations(TensorDigest::<f32, RankKnot>::with_config(
        &[2, 3],
        RankKnotConfig { buffer_capacity: 8 },
    ));
    elementwise_reshape_keeps_observations(TensorDigest::<f32, TDigest>::with_config(
        &[2, 3],
        TDigestConfig {
            compression: 20,
            buffer_capacity: Some(8),
        },
    ));
}

fn compatible_blocks_keep_their_pools<K: DigestKernel<f32>>(mut digest: TensorDigest<f32, K>) {
    let sample: Vec<f32> = (0..16).map(|value| value as f32).collect();
    digest.update(&sample).unwrap();
    assert_eq!(digest.block_shape(), &[2, 2]);

    digest.reshape(&[4, 4]).unwrap();
    assert_eq!(digest.shape(), &[4, 4]);
    assert_eq!(digest.block_shape(), &[4, 1]);
    assert_eq!(digest.block_config(), BlockConfig::block_size(4, 1));
    assert_eq!(digest.quantile(0.0), vec![0.0, 4.0, 8.0, 12.0]);
    assert_eq!(digest.quantile(1.0), vec![3.0, 7.0, 11.0, 15.0]);

    let split = digest.reshape(&[8, 2]).unwrap_err();
    assert_eq!(
        split.to_string(),
        "cannot reshape this digest: it would split a pooled block"
    );
    assert_eq!(digest.shape(), &[4, 4]);
    assert_eq!(digest.block_shape(), &[4, 1]);
    assert_eq!(digest.quantile(1.0), vec![3.0, 7.0, 11.0, 15.0]);

    digest
        .remap(&[16], BlockConfig::block_size(4, 0))
        .unwrap();
    assert_eq!(digest.shape(), &[16]);
    assert_eq!(digest.block_shape(), &[4]);
    assert_eq!(
        digest.block_config(),
        BlockConfig::Size { size: 4, axis: 0 }
    );
    assert_eq!(digest.quantile(1.0), vec![3.0, 7.0, 11.0, 15.0]);
}

#[test]
fn compatible_blocks_keep_their_pools_on_both_kernels() {
    compatible_blocks_keep_their_pools(rankknot_blocks(
        &[2, 8],
        BlockConfig::block_size(4, -1),
    ));
    compatible_blocks_keep_their_pools(tdigest_blocks(&[2, 8], BlockConfig::block_size(4, -1)));
}

fn missing_axis_stays_invalid_config<K: DigestKernel<f32>>(mut digest: TensorDigest<f32, K>) {
    digest
        .update(&(0..16).map(|value| value as f32).collect::<Vec<_>>())
        .unwrap();
    let error = digest.reshape(&[16]).unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidConfig {
            parameter: "block axis",
            ..
        }
    ));
    assert!(!error.is_incompatible_layout());
    assert_eq!(digest.shape(), &[2, 8]);
    assert_eq!(digest.block_shape(), &[2, 2]);
    assert_eq!(digest.quantile(1.0), vec![3.0, 7.0, 11.0, 15.0]);
}

#[test]
fn a_missing_grouping_axis_is_invalid_config() {
    missing_axis_stays_invalid_config(rankknot_blocks(&[2, 8], BlockConfig::block_size(4, 1)));
    missing_axis_stays_invalid_config(tdigest_blocks(&[2, 8], BlockConfig::block_size(4, 1)));
}

fn merging_separate_blocks_is_rejected<K: DigestKernel<f32>>(mut digest: TensorDigest<f32, K>) {
    let sample: Vec<f32> = (0..8).map(|value| value as f32).collect();
    digest.update(&sample).unwrap();
    let error = digest
        .remap(&[8], BlockConfig::block_size(4, 0))
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "cannot remap this digest: it would merge distinct blocks"
    );
    assert_eq!(digest.shape(), &[8]);
    assert_eq!(digest.block_count(), 8);
    assert_eq!(digest.quantile(1.0), sample);
}

#[test]
fn remap_rejects_merging_distinct_blocks() {
    merging_separate_blocks_is_rejected(TensorDigest::<f32, RankKnot>::with_config(
        &[8],
        RankKnotConfig { buffer_capacity: 8 },
    ));
    merging_separate_blocks_is_rejected(TensorDigest::<f32, TDigest>::with_config(
        &[8],
        TDigestConfig {
            compression: 20,
            buffer_capacity: Some(8),
        },
    ));
}

fn crossed_axes_are_rejected<K: DigestKernel<f32>>(mut digest: TensorDigest<f32, K>) {
    digest
        .update(&(0..8).map(|value| value as f32).collect::<Vec<_>>())
        .unwrap();
    let error = digest
        .remap(&[2, 4], BlockConfig::block_size(2, 1))
        .unwrap_err();
    assert!(error.to_string().contains("split a pooled block"));
    assert_eq!(
        digest.block_config(),
        BlockConfig::Size { size: 2, axis: 0 }
    );
    assert_eq!(digest.shape(), &[2, 4]);
}

#[test]
fn remap_rejects_a_grouping_that_cuts_across_existing_pools() {
    crossed_axes_are_rejected(rankknot_blocks(&[2, 4], BlockConfig::block_size(2, 0)));
    crossed_axes_are_rejected(tdigest_blocks(&[2, 4], BlockConfig::block_size(2, 0)));
}

#[test]
fn scalar_and_empty_shapes_reshape() {
    let mut scalar = TensorDigest::<f32>::with_config(&[], RankKnotConfig { buffer_capacity: 4 });
    scalar.update(&[7.0]).unwrap();
    scalar.reshape(&[1, 1]).unwrap();
    assert_eq!(scalar.shape(), &[1, 1]);
    assert_eq!(scalar.numel(), 1);
    assert_eq!(scalar.quantile(0.5), vec![7.0]);

    let mut empty = TensorDigest::<f32, TDigest>::new(&[0, 4]);
    empty.update(&[]).unwrap();
    empty.reshape(&[2, 0, 3]).unwrap();
    assert_eq!(empty.shape(), &[2, 0, 3]);
    assert_eq!(empty.block_count(), 0);
    assert!(empty.quantile(0.5).is_empty());
}
