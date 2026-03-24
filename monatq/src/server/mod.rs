use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::distribution::Distribution;
use crate::tensor_digest::{TensorDigest, TensorValue};

static HTML: &str = include_str!("frontend.html");

/// Start a blocking HTTP visualizer on `127.0.0.1:{MONATQ_PORT}` (default 7777).
/// Calls `analyze()` internally before serving.
pub(crate) fn serve<T: TensorValue>(digest: &mut TensorDigest<T>) -> std::io::Result<()> {
    let stop = AtomicBool::new(false);
    serve_until(digest, &stop)
}

pub(crate) fn serve_until<T: TensorValue>(digest: &mut TensorDigest<T>, stop: &AtomicBool) -> std::io::Result<()> {
    let port = std::env::var("MONATQ_PORT").unwrap_or_else(|_| "7777".to_string());
    let addr = format!("127.0.0.1:{port}");

    let distributions = digest.analyze();
    let shape = digest.shape().to_vec();

    let listener = TcpListener::bind(&addr)?;
    listener.set_nonblocking(true)?;
    eprintln!("monatq visualizer running at http://{addr}  (Ctrl+C to stop)");

    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => handle(&stream, &shape, &distributions, digest),
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => continue,
        }
    }
    Ok(())
}

// ── request handling ─────────────────────────────────────────────────────────

fn handle<T: TensorValue>(
    stream: &std::net::TcpStream,
    shape: &[usize],
    distributions: &[Distribution],
    digest: &mut TensorDigest<T>,
) {
    let (method, path, query, body_bytes) = match parse_http_request(stream) {
        Some(r) => r,
        None => return,
    };
    let query = query.as_str();

    let (status, ct, body): (&str, &str, String) = match path.as_str() {
        "/" => ("200 OK", "text/html; charset=utf-8", HTML.to_string()),
        "/api/info" => ("200 OK", "application/json", json_info(shape)),
        "/api/slice" => {
            let q = parse_query(query);
            let b = q.get("b").and_then(|v| v.parse().ok()).unwrap_or(0usize);
            let c = q.get("c").and_then(|v| v.parse().ok()).unwrap_or(0usize);
            (
                "200 OK",
                "application/json",
                json_slice(shape, distributions, b, c),
            )
        }
        "/api/cell" => {
            let q = parse_query(query);
            let idx = q.get("idx").and_then(|v| v.parse().ok()).unwrap_or(0usize);
            let (q_lo, q_hi) = parse_quantile_window(&q);
            let exclude_zero = parse_bool_flag(&q, "exclude_zero");
            (
                "200 OK",
                "application/json",
                json_cell(digest, distributions, idx, q_lo, q_hi, exclude_zero),
            )
        }
        "/api/merge" if method == "POST" => {
            let q = parse_query(query);
            let (q_lo, q_hi) = parse_quantile_window(&q);
            let exclude_zero = parse_bool_flag(&q, "exclude_zero");
            let indices = parse_json_indices(&body_bytes);
            (
                "200 OK",
                "application/json",
                json_digest_merged(digest.merge_cells(&indices), q_lo, q_hi, exclude_zero),
            )
        }
        "/api/merge" => {
            let q = parse_query(query);
            let b = q.get("b").and_then(|v| v.parse().ok()).unwrap_or(0usize);
            let c = q.get("c").and_then(|v| v.parse().ok()).unwrap_or(0usize);
            let (q_lo, q_hi) = parse_quantile_window(&q);
            let exclude_zero = parse_bool_flag(&q, "exclude_zero");
            let body = match q.get("scope").copied() {
                Some("tensor") => json_digest_merged(digest.merge_all(), q_lo, q_hi, exclude_zero),
                Some("channel") => {
                    json_digest_merged(
                        digest.merge_channels(&[channel_idx(shape, b, c)]),
                        q_lo,
                        q_hi,
                        exclude_zero,
                    )
                }
                _ => unreachable!(),
            };
            ("200 OK", "application/json", body)
        }
        _ => ("404 Not Found", "text/plain", "Not Found".into()),
    };

    let body_bytes = body.as_bytes();
    let resp = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {ct}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body_bytes.len()
    );
    let mut stream = stream;
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.write_all(body_bytes);
}

fn parse_http_request(
    mut stream: &std::net::TcpStream,
) -> Option<(String, String, String, Vec<u8>)> {
    // Read until httparse has a complete header section.
    let mut raw: Vec<u8> = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    let (method, path, query, content_length, header_end) = loop {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        raw.extend_from_slice(&tmp[..n]);

        let mut headers = [httparse::EMPTY_HEADER; 32];
        let mut req = httparse::Request::new(&mut headers);
        match req.parse(&raw) {
            Ok(httparse::Status::Complete(header_end)) => {
                let method = req.method?.to_owned();
                let raw_path = req.path?;
                let (path, query) = match raw_path.find('?') {
                    Some(i) => (raw_path[..i].to_owned(), raw_path[i + 1..].to_owned()),
                    None => (raw_path.to_owned(), String::new()),
                };
                let content_length: usize = req
                    .headers
                    .iter()
                    .find(|h| h.name.eq_ignore_ascii_case("content-length"))
                    .and_then(|h| std::str::from_utf8(h.value).ok())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                break (method, path, query, content_length, header_end);
            }
            Ok(httparse::Status::Partial) => continue,
            Err(_) => return None,
        }
    };

    // Collect body bytes already read past the headers, then read the rest.
    let mut body = raw[header_end..].to_vec();
    while body.len() < content_length {
        let want = (content_length - body.len()).min(65536);
        let mut chunk = vec![0u8; want];
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&chunk[..n]),
        }
    }

    Some((method, path, query, body))
}

/// Parse a JSON array of integers from a POST body, e.g. `[1,2,3]`.
fn parse_json_indices(body: &[u8]) -> Vec<usize> {
    let s = std::str::from_utf8(body).unwrap_or("");
    s.split(|c: char| !c.is_ascii_digit())
        .filter(|tok| !tok.is_empty())
        .filter_map(|tok| tok.parse().ok())
        .collect()
}

// ── query string parsing ──────────────────────────────────────────────────────

fn parse_query(query: &str) -> HashMap<&str, &str> {
    let mut map = HashMap::new();
    for pair in query.split('&') {
        if let Some(eq) = pair.find('=') {
            map.insert(&pair[..eq], &pair[eq + 1..]);
        }
    }
    map
}

fn parse_quantile_window(query: &HashMap<&str, &str>) -> (f32, f32) {
    let q_lo = query
        .get("q0")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0_f32);
    let q_hi = query
        .get("q1")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0_f32);
    normalize_quantile_window(q_lo, q_hi)
}

fn normalize_quantile_window(q_lo: f32, q_hi: f32) -> (f32, f32) {
    let lo = q_lo.clamp(0.0, 1.0);
    let hi = q_hi.clamp(0.0, 1.0);
    if hi > lo { (lo, hi) } else { (0.0, 1.0) }
}

fn parse_bool_flag(query: &HashMap<&str, &str>, key: &str) -> bool {
    matches!(query.get(key).copied(), Some("1" | "true" | "yes" | "on"))
}


/// Flat channel index for the given (b, c) coordinates within `shape`.
fn channel_idx(shape: &[usize], b: usize, c: usize) -> usize {
    match shape.len() {
        0..=2 => 0,
        3 => c,
        _ => b * shape[1] + c,
    }
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

fn json_info(shape: &[usize]) -> String {
    let arr = shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(r#"{{"shape":[{arr}],"ndim":{}}}"#, shape.len())
}

fn json_slice(shape: &[usize], distributions: &[Distribution], b: usize, c: usize) -> String {
    let ndim = shape.len();
    let (h, w) = (shape[ndim - 2], shape[ndim - 1]);

    let offset = channel_idx(shape, b, c) * h * w;

    let end = (offset + h * w).min(distributions.len());
    let slice = &distributions[offset..end];

    let list = slice
        .iter()
        .map(|&d| format!(r#""{d}""#))
        .collect::<Vec<_>>()
        .join(",");

    format!(r#"{{"rows":{h},"cols":{w},"distributions":[{list}]}}"#)
}

fn json_digest_cell<T: TensorValue>(
    digest: &TensorDigest<T>,
    dist: Distribution,
    label: &str,
    q_lo: f32,
    q_hi: f32,
) -> String {
    // 402 quantile points: window endpoints plus 400 evenly spaced interior quantiles.
    let qs: Vec<f32> = std::iter::once(q_lo)
        .chain((0..400).map(|i| q_lo + (q_hi - q_lo) * ((i as f32 + 1.0) / 401.0)))
        .chain(std::iter::once(q_hi))
        .collect();
    let vals = digest.cell_quantiles(0, &qs);

    let min = vals[0];
    let max = vals[vals.len() - 1];
    let box_qs = digest.cell_quantiles(0, &[0.25, 0.50, 0.75]);
    let (q25, q50, q75) = (box_qs[0], box_qs[1], box_qs[2]);

    let count = digest.total_weight(0) as u64;

    let pdf_pts = qs
        .iter()
        .zip(vals.iter())
        .map(|(&q, &x)| format!(r#"{{"x":{x:.6},"y":{q:.6}}}"#))
        .collect::<Vec<_>>()
        .join(",");

    format!(
        r#"{{"label":"{label}","distribution":"{dist}","count":{count},"min":{min:.6},"q25":{q25:.6},"q50":{q50:.6},"q75":{q75:.6},"max":{max:.6},"q0":{q_lo:.6},"q1":{q_hi:.6},"pdf":[{pdf_pts}]}}"#
    )
}

fn json_cell<T: TensorValue>(
    digest: &mut TensorDigest<T>,
    distributions: &[Distribution],
    idx: usize,
    q_lo: f32,
    q_hi: f32,
    exclude_zero: bool,
) -> String {
    let dist = distributions
        .get(idx)
        .copied()
        .unwrap_or(Distribution::Normal);
    let single = digest.merge_cells(&[idx]);
    let filtered = if exclude_zero {
        single.without_zeros()
    } else {
        single
    };
    json_digest_cell(&filtered, dist, "cell", q_lo, q_hi)
}

fn json_digest_merged<T: TensorValue>(mut merged: TensorDigest<T>, q_lo: f32, q_hi: f32, exclude_zero: bool) -> String {
    if exclude_zero {
        merged = merged.without_zeros();
    }
    let dist = merged
        .analyze()
        .into_iter()
        .next()
        .unwrap_or(Distribution::Unknown);
    json_digest_cell(&merged, dist, "merged", q_lo, q_hi)
}
