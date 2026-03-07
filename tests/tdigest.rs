use monatq::TDigest;
use statrs::distribution::{ContinuousCDF, Normal, Uniform};

fn make_digest(values: impl IntoIterator<Item = f32>) -> TDigest {
    let mut d = TDigest::new(100.0);
    for v in values {
        d.add(v);
    }
    d
}

// Evenly-spaced quantile points via inverse CDF — deterministic and no rand needed.
// Using i/(n+1) avoids the open endpoints (0 and 1) which would be ±∞ for Normal.
fn samples<D: ContinuousCDF<f64, f64>>(dist: &D, n: usize) -> Vec<f32> {
    (1..=n)
        .map(|i| dist.inverse_cdf(i as f64 / (n + 1) as f64) as f32)
        .collect()
}

#[test]
fn median_uniform() {
    let dist = Uniform::new(0.0, 1000.0).unwrap();
    let mut d = make_digest(samples(&dist, 5000));
    let median = d.quantile(0.5);
    let expected = dist.inverse_cdf(0.5) as f32;
    // k2 centroid weight at q=0.5 is compression * 0.25 = 25; allow up to 2 centroids of error
    assert!(
        (median - expected).abs() < 20.0,
        "median {median} not near {expected}"
    );
}

#[test]
fn min_max() {
    let dist = Uniform::new(1.0, 1000.0).unwrap();
    let mut d = make_digest(samples(&dist, 5000));
    assert!((d.quantile(0.0) - 1.0).abs() < 5.0);
    assert!((d.quantile(1.0) - 1000.0).abs() < 5.0);
}

#[test]
fn monotone_quantiles() {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 2000));
    let qs = [0.1, 0.25, 0.5, 0.75, 0.9];
    let values: Vec<f32> = qs.iter().map(|&q| d.quantile(q)).collect();
    for w in values.windows(2) {
        assert!(w[0] <= w[1], "quantiles not monotone: {} > {}", w[0], w[1]);
    }
}

#[test]
fn single_element() {
    let mut d = make_digest([42.0]);
    assert_eq!(d.quantile(0.0), 42.0);
    assert_eq!(d.quantile(0.5), 42.0);
    assert_eq!(d.quantile(1.0), 42.0);
}

#[test]
fn all_same_value() {
    let mut d = make_digest(std::iter::repeat(7.0f32).take(200));
    assert_eq!(d.quantile(0.5), 7.0);
}

#[test]
fn tail_accuracy() {
    // Sample from N(0,1) and compare estimated quantiles to theoretical values
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 10000));
    for &q in &[0.95f64, 0.99] {
        let expected = dist.inverse_cdf(q) as f32;
        let got = d.quantile(q as f32);
        assert!(
            (got - expected).abs() < 0.15,
            "quantile({q}) = {got}, expected {expected}"
        );
    }
}

#[test]
fn triggers_compress() {
    // 1500 samples with buffer_capacity=500 forces 3 buffer flushes before quantile
    let dist = Uniform::new(0.0, 600.0).unwrap();
    let mut d = make_digest(samples(&dist, 1500));
    let median = d.quantile(0.5);
    let expected = dist.inverse_cdf(0.5) as f32;
    assert!(
        (median - expected).abs() < 20.0,
        "median {median} not near {expected}"
    );
}
