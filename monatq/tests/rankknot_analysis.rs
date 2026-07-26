//! RankKnot's `analyze` and `without_zeros`, the last two kernel operations to be filled in.
use monatq::{Distribution, RankKnot, TensorDigest};
use statrs::distribution::{ContinuousCDF, Laplace, LogNormal, Normal, Uniform};

/// Deterministic inverse-CDF sampling, matching `tests/analyze.rs`.
fn samples<D: ContinuousCDF<f64, f64>>(dist: &D, n: usize) -> Vec<f32> {
    (1..=n)
        .map(|i| dist.inverse_cdf(i as f64 / (n + 1) as f64) as f32)
        .collect()
}

fn classify(values: &[f32]) -> Distribution {
    let mut td = TensorDigest::<f32, RankKnot>::new(&[1]);
    for &v in values {
        td.update(&[v]).unwrap();
    }
    td.analyze().unwrap()[0]
}

#[test]
fn analyze_identifies_each_reference_family_from_32_knots() {
    // The real question for RankKnot is whether 32 knots retain enough shape to separate
    // these families at all; t-digest gets ~100 centroids for the same job.
    assert_eq!(
        classify(&samples(&Normal::new(0.0, 1.0).unwrap(), 2000)),
        Distribution::Normal
    );
    assert_eq!(
        classify(&samples(&Uniform::new(0.0, 1.0).unwrap(), 2000)),
        Distribution::Uniform
    );
    assert_eq!(
        classify(&samples(&Laplace::new(0.0, 1.0).unwrap(), 2000)),
        Distribution::Laplace
    );
    assert_eq!(
        classify(&samples(&LogNormal::new(0.0, 1.0).unwrap(), 2000)),
        Distribution::LogNormal
    );
}

#[test]
fn analyze_reports_unknown_for_degenerate_and_empty_positions() {
    // A constant stream has no spread to standardise by.
    assert_eq!(classify(&[7.0; 500]), Distribution::Unknown);

    // A position that never saw an observation must not be classified as anything.
    let mut empty = TensorDigest::<f32, RankKnot>::new(&[3]);
    assert_eq!(
        empty.analyze().unwrap(),
        vec![
            Distribution::Unknown,
            Distribution::Unknown,
            Distribution::Unknown
        ]
    );
}

#[test]
fn analyze_classifies_each_position_independently() {
    let normal = samples(&Normal::new(0.0, 1.0).unwrap(), 2000);
    let uniform = samples(&Uniform::new(0.0, 1.0).unwrap(), 2000);

    let mut td = TensorDigest::<f32, RankKnot>::new(&[2]);
    for (&n, &u) in normal.iter().zip(uniform.iter()) {
        td.update(&[n, u]).unwrap();
    }
    assert_eq!(
        td.analyze().unwrap(),
        vec![Distribution::Normal, Distribution::Uniform]
    );
}

#[test]
fn without_zeros_recovers_the_shape_a_zero_spike_was_hiding() {
    // 80% exact zeros over a normal tail: the dominant spike should mask the shape, and
    // filtering it out should recover it.
    let normal = samples(&Normal::new(0.0, 1.0).unwrap(), 400);
    let mut values: Vec<f32> = vec![0.0; 1600];
    values.extend_from_slice(&normal);
    values.sort_by(f32::total_cmp);

    let mut td = TensorDigest::<f32, RankKnot>::new(&[1]);
    for &v in &values {
        td.update(&[v]).unwrap();
    }

    // The spike swallows the entire interquartile range: q25, median, and q75 all sit on
    // zero, so the digest reports no visible spread at all.
    assert_eq!(td.quantile(0.25)[0], 0.0);
    assert_eq!(td.quantile(0.5)[0], 0.0);
    assert_eq!(td.quantile(0.75)[0], 0.0);

    let mut filtered = td.without_zeros().unwrap();
    // The underlying normal is symmetric about zero, so the *median* legitimately stays near
    // zero after filtering. What must change is the spread: the quartiles should separate
    // and bracket the median from both sides.
    let (q25, q75) = (filtered.quantile(0.25)[0], filtered.quantile(0.75)[0]);
    assert!(q25 < -0.3, "q25 {q25} should have opened out below zero");
    assert!(q75 > 0.3, "q75 {q75} should have opened out above zero");
    // Recovered IQR of a standard normal is ~1.349.
    assert!(
        (q75 - q25 - 1.349).abs() < 0.15,
        "recovered IQR {} is not close to the normal's 1.349",
        q75 - q25
    );
}

#[test]
fn without_zeros_keeps_a_nonzero_extremum_and_stays_queryable() {
    let mut td = TensorDigest::<f32, RankKnot>::new(&[1]);
    for _ in 0..500 {
        td.update(&[0.0]).unwrap();
    }
    for step in 1..=100 {
        td.update(&[step as f32]).unwrap();
    }
    td.update(&[-42.0]).unwrap();

    let mut filtered = td.without_zeros().unwrap();
    // Both extrema were nonzero, so filtering must not pull the range inwards.
    assert_eq!(filtered.min()[0], -42.0);
    assert_eq!(filtered.max()[0], 100.0);
    assert_eq!(filtered.quantile(0.0)[0], -42.0);
    assert_eq!(filtered.quantile(1.0)[0], 100.0);
    // The filtered digest is a real digest: it still answers and still merges.
    assert!(filtered.merge_all().is_ok());
}

#[test]
fn without_zeros_on_an_all_zero_position_yields_an_empty_summary() {
    let mut td = TensorDigest::<f32, RankKnot>::new(&[1]);
    for _ in 0..200 {
        td.update(&[0.0]).unwrap();
    }
    let mut filtered = td.without_zeros().unwrap();
    // Nothing survives; extrema are pinned to zero rather than left as ±inf sentinels.
    assert_eq!(filtered.min()[0], 0.0);
    assert_eq!(filtered.max()[0], 0.0);
    assert_eq!(filtered.quantile(0.5)[0], 0.0);
    assert_eq!(filtered.analyze().unwrap()[0], Distribution::Unknown);
}

#[test]
fn without_zeros_survives_a_snapshot_round_trip() {
    // The filtered digest renormalises masses, so it must still satisfy the snapshot
    // invariants that `from_payload` validates.
    let mut td = TensorDigest::<f32, RankKnot>::new(&[2]);
    for step in 0..300 {
        let v = (step % 7) as f32 - 3.0;
        td.update(&[v, 0.0]).unwrap();
    }
    let mut filtered = td.without_zeros().unwrap();
    let bytes = filtered.to_bytes().unwrap();
    let mut restored = TensorDigest::<f32, RankKnot>::from_bytes(&bytes).unwrap();
    for &q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(restored.quantile(q), filtered.quantile(q), "q={q}");
    }
}
