use monatq::{QuantileSpine, QuantileSpineConfig, SpineLink, SpineRegime};
use statrs::distribution::{ContinuousCDF, LogNormal, Normal, Uniform};

fn xorshift32(state: &mut u32) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    ((*state as f64) + 0.5) / (u32::MAX as f64 + 1.0)
}

fn feed_distribution<D: ContinuousCDF<f64, f64>>(dist: &D, n: usize) -> QuantileSpine<f32> {
    let mut spine = QuantileSpine::new(&[1]);
    let mut state = 0x8bad_f00d;
    for _ in 0..n {
        spine.update(&[dist.inverse_cdf(xorshift32(&mut state)) as f32]);
    }
    spine
}

fn assert_rank_accuracy<D: ContinuousCDF<f64, f64>>(name: &str, dist: &D) {
    let mut spine = feed_distribution(dist, 30_000);
    for &q in &[0.001f32, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999] {
        let estimate = spine.quantile(q)[0];
        let rank = dist.cdf(estimate as f64) as f32;
        let error = (rank - q).abs();
        assert!(
            error < 0.025,
            "{name} q={q}: estimate={estimate}, rank={rank}, rank error={error}"
        );
    }
}

fn rank_interval_error(sorted: &[f32], estimate: f32, q: f32) -> f32 {
    let lower = sorted.partition_point(|&value| value < estimate) as f32 / sorted.len() as f32;
    let upper = sorted.partition_point(|&value| value <= estimate) as f32 / sorted.len() as f32;
    if q < lower {
        lower - q
    } else if q > upper {
        q - upper
    } else {
        0.0
    }
}

fn mean_empirical_rank_error(
    spine: &mut QuantileSpine<f32>,
    truth: &[Vec<f32>],
    quantiles: &[f32],
) -> f32 {
    let estimates = spine.quantiles(quantiles);
    let mut error = 0.0;
    for (q_index, &q) in quantiles.iter().enumerate() {
        for (position, values) in truth.iter().enumerate() {
            error += rank_interval_error(values, estimates[q_index][position], q);
        }
    }
    error / (truth.len() * quantiles.len()) as f32
}

#[test]
fn early_life_is_exact() {
    let values = [8.0, -3.0, 4.0, 4.5, 100.0, 0.0, 7.0, 2.0, -9.0];
    let mut sorted = values;
    sorted.sort_by(f32::total_cmp);
    let mut spine = QuantileSpine::new(&[1]);
    for value in values {
        spine.update(&[value]);
    }

    for (index, expected) in sorted.into_iter().enumerate() {
        let q = index as f32 / (values.len() - 1) as f32;
        assert_eq!(spine.quantile(q)[0], expected);
    }
}

#[test]
fn all_time_record_shelves_are_exact() {
    let n = 2_000usize;
    let mut state = 0x1234_5678;
    let mut values = Vec::with_capacity(n);
    let mut spine = QuantileSpine::new(&[1]);
    for _ in 0..n {
        let value = (xorshift32(&mut state) * 10_000.0) as f32;
        values.push(value);
        spine.update(&[value]);
    }
    values.sort_by(f32::total_cmp);

    for index in (0..8).chain(n - 8..n) {
        let q = index as f32 / (n - 1) as f32;
        assert_eq!(
            spine.quantile(q)[0],
            values[index],
            "record rank {index} was not exact"
        );
    }
    assert_eq!(spine.cell_min(0), values[0]);
    assert_eq!(spine.cell_max(0), values[n - 1]);
}

#[test]
fn zero_counter_and_zero_atom_are_exact() {
    let mut spine = QuantileSpine::new(&[1]);
    for i in 0..1_000 {
        let value = if i % 2 == 0 { 0.0 } else { i as f32 + 1.0 };
        spine.update(&[value]);
    }

    assert_eq!(spine.zero_count(0), 500);
    assert_eq!(spine.quantile(0.25)[0], 0.0);
    assert_eq!(spine.quantile(0.5)[0], 0.0);
    assert!(spine.quantile(0.75)[0] > 0.0);
}

#[test]
fn secondary_atom_is_promoted_and_counted() {
    let mut spine = QuantileSpine::new(&[1]);
    for i in 0..2_000 {
        let value = if i % 3 == 0 {
            7.0
        } else {
            i as f32 * 0.01 + 10.0
        };
        spine.update(&[value]);
    }
    let (value, count) = spine
        .secondary_atom(0)
        .expect("secondary atom not promoted");
    assert_eq!(value, 7.0);
    assert_eq!(count, 667);
    assert_eq!(spine.quantile(0.2)[0], 7.0);
}

#[test]
fn mature_secondary_atom_requires_consecutive_tied_batches() {
    let config = QuantileSpineConfig {
        buffer_capacity: 16,
        atom_threshold: 0.5,
        ..QuantileSpineConfig::default()
    };
    let mut spine = QuantileSpine::with_config(&[1], config);
    for i in 0..80 {
        spine.update(&[i as f32 + 10.0]);
    }
    for _ in 0..16 {
        spine.update(&[7.0]);
    }
    assert_eq!(spine.secondary_atom(0), None);

    for _ in 0..16 {
        spine.update(&[7.0]);
    }
    assert_eq!(spine.secondary_atom(0), Some((7.0, 16)));
}

#[test]
fn quantile_rank_accuracy_normal() {
    assert_rank_accuracy("normal", &Normal::new(0.0, 1.0).unwrap());
}

#[test]
fn quantile_rank_accuracy_uniform() {
    assert_rank_accuracy("uniform", &Uniform::new(-3.0, 9.0).unwrap());
}

#[test]
fn quantile_rank_accuracy_lognormal() {
    assert_rank_accuracy("lognormal", &LogNormal::new(0.2, 0.9).unwrap());
}

#[test]
fn surprise_gain_tracks_an_abrupt_regime_change() {
    let mut spine = QuantileSpine::new(&[1]);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut state = 0xfeed_beef;
    for _ in 0..20_000 {
        spine.update(&[normal.inverse_cdf(xorshift32(&mut state)) as f32]);
    }
    for _ in 0..1_024 {
        spine.update(&[(normal.inverse_cdf(xorshift32(&mut state)) + 5.0) as f32]);
    }

    let median = spine.quantile(0.5)[0];
    assert!(
        (median - 5.0).abs() < 0.35,
        "adaptive median {median} did not converge to the shifted regime"
    );
    assert!(spine.surprise(0) > 0.02);
}

#[test]
fn sustained_surprise_restarts_windowed_records() {
    let config = QuantileSpineConfig {
        buffer_capacity: 32,
        gain_c: 0.0,
        n_max: u64::MAX,
        ..QuantileSpineConfig::default()
    };
    let mut spine = QuantileSpine::with_config(&[1], config);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut state = 0xa54f_f53a;
    for _ in 0..4_096 {
        spine.update(&[normal.inverse_cdf(xorshift32(&mut state)) as f32]);
    }
    for _ in 0..3 * config.buffer_capacity {
        spine.update(&[100.0]);
    }

    assert_eq!(spine.regime(0), SpineRegime::Restart);
    assert_eq!(spine.recent_min()[0], 100.0);
    assert_eq!(spine.recent_max()[0], 100.0);
    assert!(spine.cell_min(0) < 0.0);
}

#[test]
fn finite_memory_adapts_faster_than_no_fading() {
    let mut fading_config = QuantileSpineConfig {
        n_max: 1_024,
        gain_c: 0.0,
        restart_crossings: 0,
        ..QuantileSpineConfig::default()
    };
    fading_config.dither_seed = 42;
    let mut infinite_config = fading_config;
    infinite_config.n_max = u64::MAX;

    let mut fading = QuantileSpine::with_config(&[1], fading_config);
    let mut no_fading = QuantileSpine::with_config(&[1], infinite_config);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut state = 0x0ddc_0ffe;
    for _ in 0..20_000 {
        let value = normal.inverse_cdf(xorshift32(&mut state)) as f32;
        fading.update(&[value]);
        no_fading.update(&[value]);
    }
    for _ in 0..5_120 {
        let value = (normal.inverse_cdf(xorshift32(&mut state)) + 3.0) as f32;
        fading.update(&[value]);
        no_fading.update(&[value]);
    }

    let fading_error = (fading.quantile(0.5)[0] - 3.0).abs();
    let no_fading_error = (no_fading.quantile(0.5)[0] - 3.0).abs();
    assert!(fading_error < 0.45, "fading error was {fading_error}");
    assert!(
        no_fading_error > fading_error * 2.0,
        "no-fading error {no_fading_error} was not much worse than {fading_error}"
    );
}

#[test]
fn shared_ruler_selects_linear_and_improves_uniform_accuracy() {
    const POSITIONS: usize = 16;
    const SAMPLES: usize = 20_000;
    const QUANTILES: [f32; 9] = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999];
    let adaptive_config = QuantileSpineConfig {
        buffer_capacity: 128,
        link_refit_interval: 4,
        ..QuantileSpineConfig::default()
    };
    let fixed_config = QuantileSpineConfig {
        link_refit_interval: 0,
        ..adaptive_config
    };
    let mut adaptive = QuantileSpine::with_config(&[POSITIONS], adaptive_config);
    let mut fixed = QuantileSpine::with_config(&[POSITIONS], fixed_config);
    let uniform = Uniform::new(-3.0, 9.0).unwrap();
    let mut state = 0x16b3_1c4d;
    let mut row = [0.0f32; POSITIONS];
    let mut truth = (0..POSITIONS)
        .map(|_| Vec::with_capacity(SAMPLES))
        .collect::<Vec<_>>();
    for _ in 0..SAMPLES {
        for (position, value) in row.iter_mut().enumerate() {
            *value = uniform.inverse_cdf(xorshift32(&mut state)) as f32;
            truth[position].push(*value);
        }
        adaptive.update(&row);
        fixed.update(&row);
    }
    for values in &mut truth {
        values.sort_unstable_by(f32::total_cmp);
    }

    assert_eq!(adaptive.link(), SpineLink::Linear);
    assert_eq!(fixed.link(), SpineLink::Probit);
    let adaptive_error = mean_empirical_rank_error(&mut adaptive, &truth, &QUANTILES);
    let fixed_error = mean_empirical_rank_error(&mut fixed, &truth, &QUANTILES);
    assert!(
        adaptive_error < 0.8 * fixed_error,
        "adaptive uniform error {adaptive_error} did not improve fixed-probit error {fixed_error}"
    );
}

#[test]
fn shared_ruler_keeps_normal_probit_without_flapping() {
    const POSITIONS: usize = 16;
    let config = QuantileSpineConfig {
        buffer_capacity: 128,
        link_refit_interval: 4,
        ..QuantileSpineConfig::default()
    };
    let mut spine = QuantileSpine::with_config(&[POSITIONS], config);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut state = 0xd131_0ba6;
    let mut row = [0.0f32; POSITIONS];
    for flush in 0..80 {
        for _ in 0..config.buffer_capacity {
            for value in &mut row {
                *value = normal.inverse_cdf(xorshift32(&mut state)) as f32;
            }
            spine.update(&row);
        }
        assert_eq!(
            spine.link(),
            SpineLink::Probit,
            "normal ruler changed after flush {flush}"
        );
    }
}

#[test]
fn shared_ruler_selects_log_probit_without_accuracy_loss() {
    const POSITIONS: usize = 16;
    const SAMPLES: usize = 20_000;
    const QUANTILES: [f32; 7] = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    let adaptive_config = QuantileSpineConfig {
        buffer_capacity: 128,
        link_refit_interval: 4,
        ..QuantileSpineConfig::default()
    };
    let fixed_config = QuantileSpineConfig {
        link_refit_interval: 0,
        ..adaptive_config
    };
    let mut adaptive = QuantileSpine::with_config(&[POSITIONS], adaptive_config);
    let mut fixed = QuantileSpine::with_config(&[POSITIONS], fixed_config);
    let lognormal = LogNormal::new(0.2, 0.9).unwrap();
    let mut state = 0xa409_3822;
    let mut row = [0.0f32; POSITIONS];
    let mut truth = (0..POSITIONS)
        .map(|_| Vec::with_capacity(SAMPLES))
        .collect::<Vec<_>>();
    for _ in 0..SAMPLES {
        for (position, value) in row.iter_mut().enumerate() {
            *value = lognormal.inverse_cdf(xorshift32(&mut state)) as f32;
            truth[position].push(*value);
        }
        adaptive.update(&row);
        fixed.update(&row);
    }
    for values in &mut truth {
        values.sort_unstable_by(f32::total_cmp);
    }

    assert_eq!(adaptive.link(), SpineLink::LogProbit);
    let adaptive_error = mean_empirical_rank_error(&mut adaptive, &truth, &QUANTILES);
    let fixed_error = mean_empirical_rank_error(&mut fixed, &truth, &QUANTILES);
    assert!(
        adaptive_error <= 1.05 * fixed_error,
        "log-probit error {adaptive_error} regressed fixed-probit error {fixed_error}"
    );
}

#[test]
fn shared_ruler_switches_after_a_normal_to_uniform_change() {
    const POSITIONS: usize = 16;
    const PHASE_SAMPLES: usize = 8_192;
    const QUANTILES: [f32; 5] = [0.01, 0.1, 0.5, 0.9, 0.99];
    let config = QuantileSpineConfig {
        buffer_capacity: 128,
        link_refit_interval: 4,
        ..QuantileSpineConfig::default()
    };
    let mut spine = QuantileSpine::with_config(&[POSITIONS], config);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let uniform = Uniform::new(8.0, 12.0).unwrap();
    let mut state = 0x299f_31d0;
    let mut row = [0.0f32; POSITIONS];
    for _ in 0..PHASE_SAMPLES {
        for value in &mut row {
            *value = normal.inverse_cdf(xorshift32(&mut state)) as f32;
        }
        spine.update(&row);
    }
    assert_eq!(spine.link(), SpineLink::Probit);

    let mut recent_truth = (0..POSITIONS)
        .map(|_| Vec::with_capacity(PHASE_SAMPLES))
        .collect::<Vec<_>>();
    for _ in 0..PHASE_SAMPLES {
        for (position, value) in row.iter_mut().enumerate() {
            *value = uniform.inverse_cdf(xorshift32(&mut state)) as f32;
            recent_truth[position].push(*value);
        }
        spine.update(&row);
    }
    for values in &mut recent_truth {
        values.sort_unstable_by(f32::total_cmp);
    }

    assert_eq!(spine.link(), SpineLink::Linear);
    let error = mean_empirical_rank_error(&mut spine, &recent_truth, &QUANTILES);
    assert!(error < 0.01, "post-switch uniform rank error was {error}");
}

#[test]
fn multidimensional_and_i32_api() {
    let mut spine = QuantileSpine::<i32>::new(&[2, 2]);
    assert_eq!(spine.shape(), &[2, 2]);
    assert_eq!(spine.numel(), 4);
    for i in 0..100 {
        spine.update(&[i, -i, i * 2, 0]);
    }
    let medians = spine.quantile(0.5);
    assert!((medians[0] - 49.5).abs() < 2.0);
    assert!((medians[1] + 49.5).abs() < 2.0);
    assert_eq!(spine.zero_count(3), 100);
}

#[test]
#[should_panic]
fn wrong_length_panics() {
    let mut spine = QuantileSpine::<f32>::new(&[2]);
    spine.update(&[1.0]);
}
