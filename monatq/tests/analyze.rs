use monatq::{Distribution, TensorDigest};
use statrs::distribution::{ContinuousCDF, LogNormal, Normal, Uniform};

fn make_digest(values: impl IntoIterator<Item = f32>) -> TensorDigest<f32, monatq::TDigest> {
    let mut td = TensorDigest::<f32, monatq::TDigest>::new(&[1]);
    for v in values {
        td.update(&[v]).unwrap();
    }
    td
}

fn samples<D: ContinuousCDF<f64, f64>>(dist: &D, n: usize) -> Vec<f32> {
    (1..=n)
        .map(|i| dist.inverse_cdf(i as f64 / (n + 1) as f64) as f32)
        .collect()
}

fn family_cases() -> [(Distribution, Vec<f32>); 4] {
    let laplace_vals = (1..=2000)
        .map(|i| {
            let u = i as f32 / 2001.0;
            let half = u - 0.5;
            let sign = if half < 0.0 { -1.0f32 } else { 1.0 };
            -sign * (1.0 - 2.0 * half.abs()).ln()
        })
        .collect();
    [
        (
            Distribution::Normal,
            samples(&Normal::new(0.0, 1.0).unwrap(), 2000),
        ),
        (
            Distribution::Uniform,
            samples(&Uniform::new(0.0, 1.0).unwrap(), 2000),
        ),
        (Distribution::Laplace, laplace_vals),
        (
            Distribution::LogNormal,
            samples(&LogNormal::new(0.0, 1.0).unwrap(), 2000),
        ),
    ]
}

#[test]
fn analyze_reference_families() {
    for (expected, values) in family_cases() {
        let mut d = make_digest(values);
        assert_eq!(d.analyze().unwrap(), vec![expected], "family {expected:?}");
    }
}

#[test]
fn analyze_degenerate() {
    let mut d = make_digest(std::iter::repeat_n(5.0f32, 200));
    assert_eq!(d.analyze().unwrap()[0], Distribution::Unknown);
}

#[test]
fn analyze_binormal() {
    // Generate samples from 0.5·N(-d, σ_c) + 0.5·N(+d, σ_c) with d=√3/2, σ_c=0.5
    const D: f32 = 0.8660254;
    let half = 1000usize;
    let comp = statrs::distribution::Normal::new(0.0, 0.5).unwrap();
    let mut vals: Vec<f32> = (1..=half)
        .map(|i| comp.inverse_cdf(i as f64 / (half + 1) as f64) as f32 - D)
        .chain((1..=half).map(|i| comp.inverse_cdf(i as f64 / (half + 1) as f64) as f32 + D))
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut d = make_digest(vals);
    assert_eq!(d.analyze().unwrap()[0], Distribution::BiNormal);
}

#[test]
fn analyze_tensor_positions_independently() {
    let cases = family_cases();
    let mut td = TensorDigest::<_, monatq::TDigest>::new(&[cases.len()]);
    for i in 0..cases[0].1.len() {
        let row = cases.each_ref().map(|(_, values)| values[i]);
        td.update(&row).unwrap();
    }

    let expected: Vec<_> = cases.into_iter().map(|(family, _)| family).collect();
    assert_eq!(td.analyze().unwrap(), expected);
}

#[test]
fn analyze_small_n() {
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let uniform_dist = Uniform::new(0.0, 1.0).unwrap();

    let mut normal_d = make_digest(samples(&normal_dist, 500));
    let mut uniform_d = make_digest(samples(&uniform_dist, 500));

    assert_eq!(normal_d.analyze().unwrap()[0], Distribution::Normal);
    assert_eq!(uniform_d.analyze().unwrap()[0], Distribution::Uniform);
}
