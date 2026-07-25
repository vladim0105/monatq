use monatq::{RankKnot, RankKnotConfig, TensorDigest};

fn digest(values: &[f32]) -> TensorDigest<f32, RankKnot> {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[1]);
    for &value in values {
        digest.update(&[value]);
    }
    digest
}

#[test]
fn default_buffer_capacity_is_256() {
    let digest = TensorDigest::<f32, RankKnot>::new(&[17]);
    assert_eq!(digest.config().buffer_capacity, 256);
}

#[test]
fn empty_and_nan_queries_match_backend_conventions() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[2]);
    assert_eq!(digest.quantile(0.5), vec![0.0, 0.0]);
    digest.update(&[1.0, 2.0]);
    assert!(digest.quantile(f32::NAN).iter().all(|value| value.is_nan()));
}

#[test]
fn partial_and_full_buffers_are_flushed_without_loss() {
    let config = RankKnotConfig { buffer_capacity: 4 };
    let mut digest = TensorDigest::<f32, RankKnot>::with_config(&[2], config);
    for i in 0..11 {
        digest.update(&[i as f32, (100 + i) as f32]);
    }
    assert_eq!(digest.sample_count(), 11);
    assert_eq!(digest.min(), vec![0.0, 100.0]);
    assert_eq!(digest.max(), vec![10.0, 110.0]);
}

#[test]
fn endpoint_queries_use_exact_extrema_after_compression() {
    let mut values = (0..2_000).map(|i| i as f32).collect::<Vec<_>>();
    values[123] = -1.0e30;
    values[1789] = 1.0e30;
    let mut digest = digest(&values);
    assert_eq!(digest.quantile(-1.0)[0], -1.0e30);
    assert_eq!(digest.quantile(0.0)[0], -1.0e30);
    assert_eq!(digest.quantile(1.0)[0], 1.0e30);
    assert_eq!(digest.quantile(2.0)[0], 1.0e30);
    assert_eq!(digest.cell_min(0), -1.0e30);
    assert_eq!(digest.cell_max(0), 1.0e30);
}

#[test]
fn retained_ties_cover_their_interior_rank_interval() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[1]);
    for i in 0..2_000 {
        digest.update(&[if i < 1_200 { 7.0 } else { 10.0 + i as f32 }]);
    }
    for q in [0.01, 0.2, 0.5, 0.59] {
        assert_eq!(digest.quantile(q)[0], 7.0, "q={q}");
    }
}

#[test]
fn quantile_curve_is_monotone() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[1]);
    let mut state = 0x1234_5678_u32;
    for _ in 0..20_000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        digest.update(&[((state as f32 / u32::MAX as f32) * 20.0 - 10.0).powi(3)]);
    }
    let qs = (0..=1_000).map(|i| i as f32 / 1_000.0).collect::<Vec<_>>();
    let estimates = digest.cell_quantiles(0, &qs);
    assert!(estimates.windows(2).all(|pair| pair[0] <= pair[1]));
}

#[test]
fn infinities_remain_pure_and_do_not_poison_finite_estimates() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[1]);
    for i in 0..10_000 {
        let value = match i % 100 {
            0 => f32::NEG_INFINITY,
            1 => f32::INFINITY,
            _ => (i % 97) as f32,
        };
        digest.update(&[value]);
    }
    assert_eq!(digest.quantile(0.0)[0], f32::NEG_INFINITY);
    assert_eq!(digest.quantile(1.0)[0], f32::INFINITY);
    assert_eq!(digest.quantile(0.005)[0], f32::NEG_INFINITY);
    assert_eq!(digest.quantile(0.995)[0], f32::INFINITY);
    assert!(digest.quantile(0.5)[0].is_finite());
}

#[test]
fn signed_zero_extrema_and_ties_are_deterministic() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[1]);
    for i in 0..1_000 {
        digest.update(&[if i % 2 == 0 { -0.0 } else { 0.0 }]);
    }
    assert!(digest.quantile(0.0)[0].is_sign_negative());
    assert!(digest.quantile(1.0)[0].is_sign_positive());
    assert_eq!(digest.quantile(0.5)[0], 0.0);
    assert!(digest.quantile(0.5)[0].is_sign_negative());
}

#[test]
fn constant_state_preserves_mass_and_value_across_many_flushes() {
    let config = RankKnotConfig { buffer_capacity: 3 };
    let mut digest = TensorDigest::<f32, RankKnot>::with_config(&[1], config);
    for _ in 0..10_001 {
        digest.update(&[42.25]);
    }
    assert_eq!(digest.sample_count(), 10_001);
    for q in [0.0, 0.001, 0.5, 0.999, 1.0] {
        assert_eq!(digest.quantile(q)[0], 42.25);
    }
}
