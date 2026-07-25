use monatq::{RankKnot, RankKnotConfig, TensorDigest};

/// Tie-aware rank error of `estimate` as an answer for probability `q`, measured against the
/// exact sorted population. A value lying inside a CDF jump is a valid generalized quantile
/// and scores zero. This mirrors `backend_accuracy`'s metric.
fn rank_interval_error(sorted: &[f32], estimate: f32, q: f32) -> f64 {
    let lower = sorted.partition_point(|&value| value < estimate) as f64 / sorted.len() as f64;
    let upper = sorted.partition_point(|&value| value <= estimate) as f64 / sorted.len() as f64;
    let q = q as f64;
    if q < lower {
        lower - q
    } else if q > upper {
        q - upper
    } else {
        0.0
    }
}

/// Deterministic LCG so fidelity thresholds are reproducible across platforms.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 40) as f32) / ((1_u64 << 24) as f32)
    }

    fn normal(&mut self) -> f32 {
        let u = self.uniform().max(1e-7);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// Collect `rows` samples of `channels` positions, merge every position, and return the
/// mean/max tie-aware rank error of the merged digest and of a reference digest fed the
/// pooled stream directly. The reference is the accuracy floor K32 imposes on this pooled
/// distribution, so it separates merge-induced loss from ordinary compression loss.
fn merge_fidelity(
    channels: usize,
    rows: usize,
    mut draw: impl FnMut(&mut Rng, usize) -> f32,
) -> ((f64, f64), (f64, f64)) {
    let mut rng = Rng(0x243f_6a88_85a3_08d3);
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[channels]);
    let mut truth = Vec::with_capacity(channels * rows);
    for _ in 0..rows {
        let sample = (0..channels).map(|c| draw(&mut rng, c)).collect::<Vec<_>>();
        truth.extend_from_slice(&sample);
        digest.update(&sample).unwrap();
    }
    truth.sort_unstable_by(f32::total_cmp);

    let mut pooled = TensorDigest::<f32, RankKnot>::new(&[1]);
    for &value in &truth {
        pooled.update(&[value]).unwrap();
    }

    let qs = (1..1_000).map(|i| i as f32 / 1_000.0).collect::<Vec<_>>();
    let mut merged = digest.merge_all().unwrap();
    assert_eq!(merged.sample_count(), (channels * rows) as u64);
    assert_eq!(merged.min()[0], truth[0]);
    assert_eq!(merged.max()[0], truth[truth.len() - 1]);

    let score = |estimates: Vec<f32>| {
        let errors = qs
            .iter()
            .zip(&estimates)
            .map(|(&q, &estimate)| rank_interval_error(&truth, estimate, q))
            .collect::<Vec<_>>();
        (
            errors.iter().sum::<f64>() / errors.len() as f64,
            errors.iter().copied().fold(0.0, f64::max),
        )
    };
    (
        score(merged.cell_quantiles(0, &qs).unwrap()),
        score(pooled.cell_quantiles(0, &qs).unwrap()),
    )
}

fn digest(values: &[f32]) -> TensorDigest<f32, RankKnot> {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[1]);
    for &value in values {
        digest.update(&[value]).unwrap();
    }
    digest
}

#[test]
fn merge_cells_unions_support_and_extrema() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[3]);
    for i in 0..5_000 {
        let x = i as f32 / 5_000.0;
        digest.update(&[x, x + 1.0, -100.0 * x]).unwrap();
    }

    let mut merged = digest.merge_cells(&[0, 1]).unwrap();
    assert_eq!(merged.sample_count(), 10_000);
    assert_eq!(merged.min(), vec![0.0]);
    assert_eq!(merged.max(), vec![1.9998]);
    // The union of U(0,1) and U(1,2) is U(0,2).
    for (q, expected) in [(0.25, 0.5), (0.5, 1.0), (0.75, 1.5)] {
        let estimate = merged.quantile(q)[0];
        assert!(
            (estimate - expected).abs() < 0.02,
            "q={q}: {estimate} vs {expected}"
        );
    }

    let mut single = digest.merge_cells(&[2]).unwrap();
    assert_eq!(single.sample_count(), 5_000);
    assert!((single.quantile(0.5)[0] + 50.0).abs() < 1.0);
}

#[test]
fn merge_cells_of_nothing_is_empty() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[2]);
    digest.update(&[1.0, 2.0]).unwrap();
    let mut merged = digest.merge_cells(&[]).unwrap();
    assert_eq!(merged.sample_count(), 0);
    assert_eq!(merged.quantile(0.5), vec![0.0]);
}

/// The merged distribution is checked against the *exact* pooled population with a tie-aware
/// rank metric at 999 probabilities, not against another approximation. Caps carry roughly
/// 2x headroom over observed values so ordinary numeric drift does not flake the suite.
#[test]
fn merged_distribution_tracks_the_exact_pooled_population() {
    struct Case {
        name: &'static str,
        channels: usize,
        rows: usize,
        draw: fn(&mut Rng, usize) -> f32,
        mean_cap: f64,
        max_cap: f64,
    }

    let cases = [
        Case {
            name: "uniform, homogeneous channels",
            channels: 8,
            rows: 4_000,
            draw: |rng, _| rng.uniform(),
            mean_cap: 0.0010,
            max_cap: 0.0035,
        },
        Case {
            name: "normal, homogeneous channels",
            channels: 8,
            rows: 4_000,
            draw: |rng, _| rng.normal(),
            mean_cap: 0.0013,
            max_cap: 0.0038,
        },
        Case {
            name: "lognormal, heavy right tail",
            channels: 8,
            rows: 4_000,
            draw: |rng, _| rng.normal().exp(),
            mean_cap: 0.0024,
            max_cap: 0.0065,
        },
        Case {
            name: "per-channel location shift",
            channels: 8,
            rows: 4_000,
            draw: |rng, channel| rng.normal() + 10.0 * channel as f32,
            mean_cap: 0.0165,
            max_cap: 0.0575,
        },
        Case {
            name: "per-channel scale spread",
            channels: 8,
            rows: 4_000,
            draw: |rng, channel| rng.normal() * (1_u32 << channel) as f32,
            mean_cap: 0.0056,
            max_cap: 0.0150,
        },
        Case {
            name: "separated bimodal channels",
            channels: 8,
            rows: 4_000,
            draw: |rng, channel| {
                if channel % 2 == 0 {
                    rng.normal()
                } else {
                    rng.normal() + 50.0
                }
            },
            mean_cap: 0.0040,
            max_cap: 0.0460,
        },
        Case {
            name: "half of the channels are zero",
            channels: 8,
            rows: 4_000,
            draw: |rng, channel| {
                if channel % 2 == 0 { 0.0 } else { rng.normal() }
            },
            mean_cap: 0.0008,
            max_cap: 0.0038,
        },
        Case {
            name: "many narrow channels",
            channels: 256,
            rows: 1_000,
            draw: |rng, channel| rng.normal() + channel as f32 * 0.1,
            mean_cap: 0.0004,
            max_cap: 0.0023,
        },
    ];

    for case in cases {
        let ((mean, max), (pooled_mean, _)) = merge_fidelity(case.channels, case.rows, case.draw);
        assert!(
            mean <= case.mean_cap,
            "{}: mean rank error {mean:.6} exceeds {:.6}",
            case.name,
            case.mean_cap
        );
        assert!(
            max <= case.max_cap,
            "{}: max rank error {max:.6} exceeds {:.6}",
            case.name,
            case.max_cap
        );
        // Merging must not cost materially more accuracy than compressing the pooled stream
        // in one pass. It is usually better, because each source resolves its own mode
        // before the union is recompressed.
        assert!(
            mean <= 2.0 * pooled_mean + 1e-4,
            "{}: merge-induced loss too large ({mean:.6} vs pooled {pooled_mean:.6})",
            case.name
        );
    }
}

/// Exactly representable level sets must survive a merge without any rank error at all.
#[test]
fn merging_preserves_discrete_levels_exactly() {
    let ((mean, max), _) = merge_fidelity(8, 4_000, |rng, _| (rng.uniform() * 32.0).floor() / 32.0);
    assert_eq!(mean, 0.0);
    assert_eq!(max, 0.0);
}

/// A merged subset must describe that subset, not the whole tensor.
#[test]
fn merge_cells_tracks_only_the_selected_population() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[4]);
    let mut truth = Vec::new();
    for _ in 0..4_000 {
        let sample = [
            rng.normal(),
            rng.normal() + 100.0,
            rng.normal() - 100.0,
            rng.normal() * 25.0,
        ];
        truth.push(sample[0]);
        truth.push(sample[2]);
        digest.update(&sample).unwrap();
    }
    truth.sort_unstable_by(f32::total_cmp);

    let mut merged = digest.merge_cells(&[0, 2]).unwrap();
    assert_eq!(merged.sample_count(), 8_000);
    assert_eq!(merged.min()[0], truth[0]);
    assert_eq!(merged.max()[0], truth[truth.len() - 1]);

    let qs = (1..1_000).map(|i| i as f32 / 1_000.0).collect::<Vec<_>>();
    let estimates = merged.cell_quantiles(0, &qs).unwrap();
    let errors = qs
        .iter()
        .zip(&estimates)
        .map(|(&q, &estimate)| rank_interval_error(&truth, estimate, q))
        .collect::<Vec<_>>();
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let worst = errors.iter().copied().fold(0.0, f64::max);
    assert!(mean < 0.005, "mean rank error {mean:.6}");
    // The two selected cells are separated modes 100 apart. One knot necessarily straddles
    // the empty gap, and every probability inside that knot's mass decodes to a value in the
    // gap where the exact CDF is flat, so a plateau-sized worst case is structural rather
    // than a merge defect. Assert it stays bounded by that knot's mass.
    assert!(worst < 0.03, "max rank error {worst:.6}");
}

#[test]
fn merge_channels_pools_only_the_selected_channels() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[3, 2, 2]);
    for i in 0..2_000 {
        let x = i as f32 / 2_000.0;
        digest
            .update(&[
                x,
                x,
                x,
                x,
                10.0 + x,
                10.0 + x,
                10.0 + x,
                10.0 + x,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
            ])
            .unwrap();
    }

    let mut merged = digest.merge_channels(&[0, 2]).unwrap();
    assert_eq!(merged.sample_count(), 16_000);
    assert_eq!(merged.min(), vec![-5.0]);
    assert_eq!(merged.max(), vec![0.9995]);
    // Half the mass is the exact tie at -5.0, the rest is U(0,1).
    assert_eq!(merged.quantile(0.25)[0], -5.0);
    assert!((merged.quantile(0.75)[0] - 0.5).abs() < 0.05);
}

#[test]
fn merged_digests_keep_accepting_updates() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[2]);
    for i in 0..1_000 {
        digest.update(&[i as f32, i as f32]).unwrap();
    }
    let mut merged = digest.merge_all().unwrap();
    for _ in 0..2_000 {
        merged.update(&[2_000.0]).unwrap();
    }
    assert_eq!(merged.sample_count(), 4_000);
    assert_eq!(merged.max(), vec![2_000.0]);
    // Half of the 4,000 observations are the appended constant.
    assert_eq!(merged.quantile(0.75)[0], 2_000.0);
    assert!(merged.quantile(0.25)[0] < 1_000.0);
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
    digest.update(&[1.0, 2.0]).unwrap();
    assert!(digest.quantile(f32::NAN).iter().all(|value| value.is_nan()));
}

#[test]
fn partial_and_full_buffers_are_flushed_without_loss() {
    let config = RankKnotConfig { buffer_capacity: 4 };
    let mut digest = TensorDigest::<f32, RankKnot>::with_config(&[2], config);
    for i in 0..11 {
        digest.update(&[i as f32, (100 + i) as f32]).unwrap();
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
    assert_eq!(digest.cell_min(0).unwrap(), -1.0e30);
    assert_eq!(digest.cell_max(0).unwrap(), 1.0e30);
}

#[test]
fn retained_ties_cover_their_interior_rank_interval() {
    let mut digest = TensorDigest::<f32, RankKnot>::new(&[1]);
    for i in 0..2_000 {
        digest
            .update(&[if i < 1_200 { 7.0 } else { 10.0 + i as f32 }])
            .unwrap();
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
        digest
            .update(&[((state as f32 / u32::MAX as f32) * 20.0 - 10.0).powi(3)])
            .unwrap();
    }
    let qs = (0..=1_000).map(|i| i as f32 / 1_000.0).collect::<Vec<_>>();
    let estimates = digest.cell_quantiles(0, &qs).unwrap();
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
        digest.update(&[value]).unwrap();
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
        digest
            .update(&[if i % 2 == 0 { -0.0 } else { 0.0 }])
            .unwrap();
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
        digest.update(&[42.25]).unwrap();
    }
    assert_eq!(digest.sample_count(), 10_001);
    for q in [0.0, 0.001, 0.5, 0.999, 1.0] {
        assert_eq!(digest.quantile(q)[0], 42.25);
    }
}

#[test]
fn snapshot_roundtrip_preserves_queries_extrema_and_count() {
    let mut original = TensorDigest::<f32, RankKnot>::new(&[2, 3]);
    let mut rng = Rng(0xcafe_f00d_1234_5678);
    for _ in 0..5_000 {
        original
            .update(&[
                rng.normal(),
                rng.normal() * 100.0,
                0.0,
                -0.0,
                f32::INFINITY,
                rng.uniform(),
            ])
            .unwrap();
    }
    let qs = (0..=100).map(|i| i as f32 / 100.0).collect::<Vec<_>>();
    let expected = original.quantiles(&qs);
    let expected_min = original.min();
    let expected_max = original.max();

    let bytes = original.to_bytes().expect("serialization failed");
    let mut loaded =
        TensorDigest::<f32, RankKnot>::from_bytes(&bytes).expect("deserialization failed");

    assert_eq!(loaded.shape(), original.shape());
    assert_eq!(loaded.numel(), original.numel());
    assert_eq!(loaded.sample_count(), original.sample_count());
    assert_eq!(loaded.quantiles(&qs), expected);
    assert_eq!(loaded.min(), expected_min);
    assert_eq!(loaded.max(), expected_max);
    // Signed zero and infinities must survive bit-exactly, not merely compare equal.
    assert!(loaded.cell_min(3).unwrap().is_sign_negative());
    assert_eq!(loaded.cell_max(4).unwrap(), f32::INFINITY);
}

#[test]
fn snapshot_roundtrip_survives_a_file_and_keeps_accepting_updates() {
    let path = std::env::temp_dir().join(format!("monatq_rankknot_{}.bin", std::process::id()));
    let mut original = TensorDigest::<f32, RankKnot>::new(&[1]);
    for i in 0..1_000 {
        original.update(&[i as f32]).unwrap();
    }
    original.save(&path).expect("save failed");
    let mut loaded = TensorDigest::<f32, RankKnot>::load(&path).expect("load failed");
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded.quantile(0.5), original.quantile(0.5));
    for _ in 0..1_000 {
        loaded.update(&[10_000.0]).unwrap();
    }
    assert_eq!(loaded.sample_count(), 2_000);
    assert_eq!(loaded.max(), vec![10_000.0]);
    // Half the observations are now the appended constant.
    assert_eq!(loaded.quantile(0.9)[0], 10_000.0);
    assert!(loaded.quantile(0.25)[0] < 1_000.0);
}

#[test]
fn empty_digest_roundtrips() {
    let mut original = TensorDigest::<f32, RankKnot>::new(&[4]);
    let bytes = original.to_bytes().expect("serialization failed");
    let mut loaded =
        TensorDigest::<f32, RankKnot>::from_bytes(&bytes).expect("deserialization failed");
    assert_eq!(loaded.sample_count(), 0);
    assert_eq!(loaded.quantile(0.5), vec![0.0; 4]);
}

#[test]
fn a_rankknot_snapshot_is_rejected_by_the_tdigest_loaders() {
    let mut original = TensorDigest::<f32, RankKnot>::new(&[1]);
    for i in 0..500 {
        original.update(&[i as f32]).unwrap();
    }
    let bytes = original.to_bytes().expect("serialization failed");

    // The crate-level loader sniffs the first payload byte as a dtype tag and assumes the
    // t-digest kernel. A distinct kernel tag must make that a clean error, never a misparse.
    assert!(monatq::from_bytes(&bytes).is_err());
    assert!(TensorDigest::<f32, monatq::TDigest>::from_bytes(&bytes).is_err());
}

#[test]
fn truncated_and_empty_snapshots_are_rejected() {
    let mut original = TensorDigest::<f32, RankKnot>::new(&[1]);
    for i in 0..500 {
        original.update(&[i as f32]).unwrap();
    }
    let bytes = original.to_bytes().expect("serialization failed");

    for candidate in [
        &[][..],
        &bytes[..bytes.len() / 2],
        &bytes[..bytes.len() - 1],
    ] {
        let error = TensorDigest::<f32, RankKnot>::from_bytes(candidate)
            .expect_err("damaged snapshot must not load");
        assert!(error.is_invalid_snapshot(), "unexpected error: {error}");
    }
}
