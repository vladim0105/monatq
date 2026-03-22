use monatq::TensorDigest;
use statrs::distribution::{ContinuousCDF, Normal, Uniform};

fn make_digest(values: impl IntoIterator<Item = f32>) -> TensorDigest {
    let mut td = TensorDigest::new(&[1], 100);
    for v in values {
        td.update(&[v]);
    }
    td
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
    let median = d.quantile(0.5)[0];
    let expected = dist.inverse_cdf(0.5) as f32;
    let err = (median - expected).abs();
    assert!(err < 0.5, "median {median} not near {expected}, err={err}");
}

#[test]
fn min_max() {
    // The minimum is always an exact singleton (k2 limit = 0 at cumulative=0).
    // The maximum is peeled off as a singleton at the end of each compress.
    let dist = Uniform::new(1.0, 1000.0).unwrap();
    let values = samples(&dist, 5000);
    let expected_min = *values.first().unwrap();
    let expected_max = *values.last().unwrap();
    let mut d = make_digest(values);
    assert_eq!(d.quantile(0.0)[0], expected_min);
    assert_eq!(d.quantile(1.0)[0], expected_max);
}

#[test]
fn monotone_quantiles() {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 2000));
    let qs = [0.1, 0.25, 0.5, 0.75, 0.9];
    let values: Vec<f32> = qs.iter().map(|&q| d.quantile(q)[0]).collect();
    for w in values.windows(2) {
        assert!(w[0] <= w[1], "quantiles not monotone: {} > {}", w[0], w[1]);
    }
}

#[test]
fn single_element() {
    let mut d = make_digest([42.0]);
    assert_eq!(d.quantile(0.0)[0], 42.0);
    assert_eq!(d.quantile(0.5)[0], 42.0);
    assert_eq!(d.quantile(1.0)[0], 42.0);
}

#[test]
fn all_same_value() {
    let mut d = make_digest(std::iter::repeat(7.0f32).take(200));
    assert_eq!(d.quantile(0.5)[0], 7.0);
}

#[test]
fn tail_accuracy() {
    // Sample from N(0,1) and compare estimated quantiles to theoretical values.
    // Tests both tails — negative (q=0.01, q=0.05) and positive (q=0.95, q=0.99).
    // V2 has no special singleton-max guarantee, so tail error is slightly larger than v1.
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 10000));
    for &q in &[0.01f64, 0.05, 0.95, 0.99] {
        let expected = dist.inverse_cdf(q) as f32;
        let got = d.quantile(q as f32)[0];
        let err = (got - expected).abs();
        assert!(
            err < 0.01,
            "quantile({q}) = {got}, expected {expected}, err={err}"
        );
    }
}

#[test]
fn spanning_zero() {
    // Uniform(-500, 500): median should be near 0, q<0.5 negative, q>0.5 positive.
    let dist = Uniform::new(-500.0, 500.0).unwrap();
    let mut d = make_digest(samples(&dist, 5000));
    let median = d.quantile(0.5)[0];
    assert!(median.abs() < 0.5, "median {median} not near 0");
    let q25 = d.quantile(0.25)[0];
    let q75 = d.quantile(0.75)[0];
    assert!(q25 < 0.0, "q=0.25 should be negative");
    assert!(q75 > 0.0, "q=0.75 should be positive");
    assert!(
        (q25 - (-250.0f32)).abs() < 1.0,
        "q=0.25 {q25} not near -250"
    );
    assert!((q75 - 250.0f32).abs() < 1.0, "q=0.75 {q75} not near 250");
}

#[test]
fn negative_values() {
    // All-negative range: verifies sort order and quantile interpolation with negatives.
    let dist = Uniform::new(-1000.0, 0.0).unwrap();
    let values = samples(&dist, 5000);
    let expected_min = *values.first().unwrap();
    let mut d = make_digest(values);
    // Median should be near -500.
    let median = d.quantile(0.5)[0];
    let err = (median - (-500.0f32)).abs();
    assert!(err < 0.5, "median {median} not near -500, err={err}");
    // Min is always an exact singleton.
    assert_eq!(d.quantile(0.0)[0], expected_min);
}

#[test]
fn triggers_compress() {
    // 1500 samples with buffer_capacity=200 forces 7 buffer flushes before quantile.
    let dist = Uniform::new(0.0, 600.0).unwrap();
    let mut d = make_digest(samples(&dist, 1500));
    let median = d.quantile(0.5)[0];
    let expected = dist.inverse_cdf(0.5) as f32;
    let err = (median - expected).abs();
    assert!(err < 1.0, "median {median} not near {expected}, err={err}");
}

#[test]
fn quantile_accuracy_normal() {
    // Interior quantiles for N(0,1) with 10000 samples.
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 10000));
    for &q in &[0.1f64, 0.25, 0.5, 0.75, 0.9] {
        let expected = dist.inverse_cdf(q) as f32;
        let got = d.quantile(q as f32)[0];
        let err = (got - expected).abs();
        assert!(
            err < 0.01,
            "quantile({q}) = {got}, expected {expected}, err={err}"
        );
    }
}

#[test]
fn quantiles_consistent() {
    // quantiles() must return the same results as individual quantile() calls.
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 2000));
    let qs = [0.25f32, 0.5, 0.75];
    let batch = d.quantiles(&qs);
    for (i, &q) in qs.iter().enumerate() {
        let single = d.quantile(q)[0];
        assert_eq!(
            batch[i][0], single,
            "quantiles()[{i}] != quantile({q}): {} vs {}",
            batch[i][0], single
        );
    }
}

#[test]
fn cell_quantiles_consistent() {
    // cell_quantiles(idx, qs) must match quantile(q)[idx] for each q.
    let dist = Normal::new(0.0, 1.0).unwrap();
    let vals = samples(&dist, 2000);

    let mut td = TensorDigest::new(&[3], 100);
    for &v in &vals {
        td.update(&[v, v * 0.5, -v]);
    }

    let qs = [0.25f32, 0.5, 0.75];
    td.flush();
    let cell = td.cell_quantiles(2, &qs);
    for (i, &q) in qs.iter().enumerate() {
        let expected = td.quantile(q)[2];
        assert_eq!(
            cell[i], expected,
            "cell_quantiles(2, {q}) = {} but quantile({q})[2] = {}",
            cell[i], expected
        );
    }
}

#[test]
fn merge_cells_combines_selected_distributions() {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let vals = samples(&dist, 2000);

    let mut td = TensorDigest::new(&[3], 100);
    let mut expected = TensorDigest::new(&[1], 100);
    for &v in &vals {
        td.update(&[v, v * 0.5, -v]);
        expected.update(&[v]);
        expected.update(&[-v]);
    }

    let mut merged = td.merge_cells(&[0, 2]);
    for &q in &[0.1f32, 0.25, 0.5, 0.75, 0.9] {
        let got = merged.quantile(q)[0];
        let want = expected.quantile(q)[0];
        let err = (got - want).abs();
        assert!(
            err < 0.055,
            "merge_cells quantile({q}) = {got}, expected {want}, err={err}"
        );
    }
}

#[test]
fn merge_cells_empty_selection_returns_empty_digest() {
    let mut td = TensorDigest::new(&[2], 100);
    td.update(&[1.0, 2.0]);

    let mut merged = td.merge_cells(&[]);
    assert_eq!(merged.shape(), &[1]);
    assert_eq!(merged.numel(), 1);
    assert_eq!(merged.quantile(0.5)[0], 0.0);
}

#[test]
fn merge_cells_all_tensor_elements_matches_expected_for_large_tensor() {
    let shape = [1, 5, 64, 64];
    let numel: usize = shape.iter().product();
    let mut td = TensorDigest::new(&shape, 100);
    let mut expected = TensorDigest::new(&[1], 100);

    for sample_idx in 0..256usize {
        let mut frame = vec![0.0f32; numel];
        for (cell_idx, value) in frame.iter_mut().enumerate() {
            let v = sample_idx as f32 * 0.25 + cell_idx as f32 / numel as f32;
            *value = v;
            expected.update(&[v]);
        }
        td.update(&frame);
    }

    let indices: Vec<usize> = (0..numel).collect();
    let mut merged = td.merge_cells(&indices);

    for &q in &[0.1f32, 0.25, 0.5, 0.75, 0.9] {
        let got = merged.quantile(q)[0];
        let want = expected.quantile(q)[0];
        let err = (got - want).abs();
        assert!(
            err < 0.2,
            "merge all cells quantile({q}) = {got}, expected {want}, err={err}"
        );
    }
}

#[test]
#[should_panic]
fn update_wrong_length_panics() {
    let mut td = TensorDigest::new(&[1], 100);
    td.update(&[1.0, 2.0]);
}

#[test]
fn flush_idempotent() {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let mut d = make_digest(samples(&dist, 2000));
    let first = d.quantile(0.5)[0];
    let second = d.quantile(0.5)[0];
    assert_eq!(first, second, "quantile(0.5) changed between calls");
}

#[test]
fn multi_dim_shape() {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let vals = samples(&dist, 2000);

    let mut td = TensorDigest::new(&[2, 3], 100);
    assert_eq!(td.numel(), 6);
    assert_eq!(td.shape(), &[2, 3]);

    for chunk in vals.chunks(6) {
        let row: Vec<f32> = (0..6).map(|i| chunk[i % chunk.len()]).collect();
        td.update(&row);
    }

    let medians = td.quantile(0.5);
    for (i, &m) in medians.iter().enumerate() {
        assert!(m.abs() < 0.5, "element {i} median {m} not near 0");
    }
}

#[test]
fn merge_all_cells_large() {
    let shape = [5usize, 5, 128, 128];
    let numel: usize = shape.iter().product();
    let mut td = TensorDigest::new(&shape, 100);
    for s in 0..2000usize {
        let frame: Vec<f32> = (0..numel).map(|i| (s * numel + i) as f32 * 0.001).collect();
        td.update(&frame);
    }
    let indices: Vec<usize> = (0..numel).collect();
    let _merged = td.merge_cells(&indices);
}
