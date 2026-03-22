use monatq::{Distribution, TensorDigest};
use statrs::distribution::{ContinuousCDF, LogNormal, Normal, Uniform};

fn make_digest(values: impl IntoIterator<Item = f32>) -> TensorDigest {
    let mut td = TensorDigest::new(&[1], 100);
    for v in values {
        td.update(&[v]);
    }
    td
}

fn samples<D: ContinuousCDF<f64, f64>>(dist: &D, n: usize) -> Vec<f32> {
    (1..=n)
        .map(|i| dist.inverse_cdf(i as f64 / (n + 1) as f64) as f32)
        .collect()
}

#[test]
fn analyze_normal() {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 2000));
    let result = d.analyze()[0];
    assert_eq!(result, Distribution::Normal);
    assert_ne!(result, Distribution::Uniform);
}

#[test]
fn analyze_uniform() {
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 2000));
    let result = d.analyze()[0];
    assert_eq!(result, Distribution::Uniform);
    assert_ne!(result, Distribution::Normal);
}

#[test]
fn analyze_laplace() {
    let laplace_vals: Vec<f32> = (1..=2000)
        .map(|i| {
            let u = i as f32 / 2001.0;
            let half = u - 0.5;
            let sign = if half < 0.0 { -1.0f32 } else { 1.0 };
            -sign * (1.0 - 2.0 * half.abs()).ln()
        })
        .collect();
    let mut d = make_digest(laplace_vals);
    let result = d.analyze()[0];
    assert_eq!(result, Distribution::Laplace);
    assert_ne!(result, Distribution::Normal);
}

#[test]
fn analyze_lognormal() {
    let dist = LogNormal::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 2000));
    let result = d.analyze()[0];
    assert_eq!(result, Distribution::LogNormal);
    assert_ne!(result, Distribution::Normal);
}

#[test]
fn analyze_degenerate() {
    let mut d = make_digest(std::iter::repeat(5.0f32).take(200));
    assert_eq!(d.analyze()[0], Distribution::Normal);
}

#[test]
fn analyze_tensor() {
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let uniform_dist = Uniform::new(0.0, 1.0).unwrap();
    let normal_vals = samples(&normal_dist, 2000);
    let uniform_vals = samples(&uniform_dist, 2000);

    let mut td = TensorDigest::new(&[2], 100);
    for (n, u) in normal_vals.iter().zip(uniform_vals.iter()) {
        td.update(&[*n, *u]);
    }

    let result = td.analyze();
    assert_eq!(result, vec![Distribution::Normal, Distribution::Uniform]);
}

#[test]
fn analyze_no_misclassification() {
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let uniform_dist = Uniform::new(0.0, 1.0).unwrap();
    let lognormal_dist = LogNormal::new(0.0, 1.0).unwrap();
    let laplace_vals: Vec<f32> = (1..=2000)
        .map(|i| {
            let u = i as f32 / 2001.0;
            let half = u - 0.5;
            let sign = if half < 0.0 { -1.0f32 } else { 1.0 };
            -sign * (1.0 - 2.0 * half.abs()).ln()
        })
        .collect();

    let normal_vals = samples(&normal_dist, 2000);
    let uniform_vals = samples(&uniform_dist, 2000);
    let lognormal_vals = samples(&lognormal_dist, 2000);

    let mut td = TensorDigest::new(&[4], 100);
    for i in 0..2000 {
        td.update(&[
            normal_vals[i],
            uniform_vals[i],
            laplace_vals[i],
            lognormal_vals[i],
        ]);
    }

    let result = td.analyze();
    assert_eq!(
        result[0],
        Distribution::Normal,
        "element 0 should be Normal"
    );
    assert_eq!(
        result[1],
        Distribution::Uniform,
        "element 1 should be Uniform"
    );
    assert_eq!(
        result[2],
        Distribution::Laplace,
        "element 2 should be Laplace"
    );
    assert_eq!(
        result[3],
        Distribution::LogNormal,
        "element 3 should be LogNormal"
    );

    assert_ne!(result[0], Distribution::Uniform);
    assert_ne!(result[1], Distribution::Normal);
    assert_ne!(result[2], Distribution::Normal);
    assert_ne!(result[3], Distribution::Normal);
}

#[test]
fn analyze_small_n() {
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let uniform_dist = Uniform::new(0.0, 1.0).unwrap();

    let mut normal_d = make_digest(samples(&normal_dist, 500));
    let mut uniform_d = make_digest(samples(&uniform_dist, 500));

    assert_eq!(normal_d.analyze()[0], Distribution::Normal);
    assert_eq!(uniform_d.analyze()[0], Distribution::Uniform);
}
