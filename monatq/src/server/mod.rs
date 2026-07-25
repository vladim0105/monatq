use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::distribution::Distribution;
use crate::tensor_digest::StorageOperations;
use crate::tensor_value::TensorValue;

static HTML: &str = include_str!("frontend.html");

/// Start a blocking HTTP visualizer on `127.0.0.1:{MONATQ_PORT}` (default 7777).
/// Calls `analyze()` internally before serving.
///
/// Generic over kernel storage rather than tied to one kernel: every endpoint is expressible
/// in [`StorageOperations`], so any kernel implementing merge, analysis, and zero filtering
/// is servable without a second copy of this module.
pub(crate) fn serve<T: TensorValue, S: StorageOperations<T>>(digest: &mut S) -> crate::Result<()> {
    let stop = AtomicBool::new(false);
    serve_until(digest, &stop)
}

pub(crate) fn serve_until<T: TensorValue, S: StorageOperations<T>>(
    digest: &mut S,
    stop: &AtomicBool,
) -> crate::Result<()> {
    let port = std::env::var("MONATQ_PORT").unwrap_or_else(|_| "7777".to_string());
    let addr = format!("127.0.0.1:{port}");

    // Fail before binding the port if this kernel cannot analyse: reporting `Unsupported`
    // immediately is far better than serving a window that errors on every request.
    let distributions = digest.analyze()?;
    let shape = digest.shape().to_vec();

    let listener = TcpListener::bind(&addr).map_err(crate::Error::Io)?;
    listener.set_nonblocking(true).map_err(crate::Error::Io)?;
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

fn handle<T: TensorValue, S: StorageOperations<T>>(
    stream: &std::net::TcpStream,
    shape: &[usize],
    distributions: &[Distribution],
    digest: &mut S,
) {
    let (method, path, query, body_bytes) = match parse_http_request(stream) {
        Some(r) => r,
        None => return,
    };
    let query = query.as_str();

    // Endpoints that touch the digest are fallible, so each arm yields a `Result` and one
    // place below turns a failure into a 500 rather than tearing down the server.
    let routed: crate::Result<(&str, &str, String)> = match path.as_str() {
        "/" => Ok(("200 OK", "text/html; charset=utf-8", HTML.to_string())),
        "/api/info" => Ok(("200 OK", "application/json", json_info(shape))),
        "/api/slice" => {
            let q = parse_query(query);
            match parse_coordinates(&q).and_then(|(b, c)| {
                if shape.len() < 2 {
                    Err("slice requires a tensor with at least 2 dimensions".to_string())
                } else {
                    channel_idx_checked(shape, b, c)
                }
            }) {
                Ok(channel) => Ok((
                    "200 OK",
                    "application/json",
                    json_slice(shape, distributions, channel),
                )),
                Err(message) => Ok(bad_request(message)),
            }
        }
        "/api/cell" => {
            let q = parse_query(query);
            let idx = q.get("idx").and_then(|v| v.parse().ok()).unwrap_or(0usize);
            let (q_lo, q_hi) = parse_quantile_window(&q);
            let exclude_zero = parse_bool_flag(&q, "exclude_zero");
            json_cell(digest, distributions, idx, q_lo, q_hi, exclude_zero)
                .map(|body| ("200 OK", "application/json", body))
        }
        "/api/merge" if method == "POST" => {
            let q = parse_query(query);
            let (q_lo, q_hi) = parse_quantile_window(&q);
            let exclude_zero = parse_bool_flag(&q, "exclude_zero");
            let indices = parse_json_indices(&body_bytes);
            digest
                .merge_cells(&indices)
                .and_then(|merged| json_digest_merged(merged, q_lo, q_hi, exclude_zero))
                .map(|body| ("200 OK", "application/json", body))
        }
        "/api/merge" => {
            let q = parse_query(query);
            let (q_lo, q_hi) = parse_quantile_window(&q);
            let exclude_zero = parse_bool_flag(&q, "exclude_zero");
            match q.get("scope").copied() {
                Some("tensor") => digest
                    .merge_all()
                    .and_then(|merged| json_digest_merged(merged, q_lo, q_hi, exclude_zero))
                    .map(|body| ("200 OK", "application/json", body)),
                Some("channel") => match parse_coordinates(&q)
                    .and_then(|(b, c)| channel_idx_checked(shape, b, c))
                {
                    Ok(channel) => digest
                        .merge_channels(&[channel])
                        .and_then(|merged| json_digest_merged(merged, q_lo, q_hi, exclude_zero))
                        .map(|body| ("200 OK", "application/json", body)),
                    Err(message) => Ok(bad_request(message)),
                },
                Some(scope) => Ok(bad_request(format!("invalid merge scope: {scope}"))),
                None => Ok(bad_request("missing merge scope".to_string())),
            }
        }
        _ => Ok(("404 Not Found", "text/plain", "Not Found".into())),
    };

    let (status, ct, body) = match routed {
        Ok(response) => response,
        Err(error) => (
            "500 Internal Server Error",
            "application/json",
            format!(r#"{{"error":"{}"}}"#, json_escape(&error.to_string())),
        ),
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

fn parse_coordinates(query: &HashMap<&str, &str>) -> Result<(usize, usize), String> {
    let parse = |key| match query.get(key) {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| format!("invalid {key} coordinate: {value}")),
        None => Ok(0),
    };
    Ok((parse("b")?, parse("c")?))
}

/// Flat channel index for valid (b, c) coordinates within `shape`.
fn channel_idx_checked(shape: &[usize], b: usize, c: usize) -> Result<usize, String> {
    let valid = match shape.len() {
        0..=2 => b == 0 && c == 0,
        3 => b == 0 && c < shape[0],
        _ => b < shape[0] && c < shape[1],
    };
    if !valid {
        return Err(format!(
            "channel coordinates b={b}, c={c} are out of range for shape {shape:?}"
        ));
    }

    match shape.len() {
        0..=2 => Ok(0),
        3 => Ok(c),
        _ => b
            .checked_mul(shape[1])
            .and_then(|base| base.checked_add(c))
            .ok_or_else(|| "channel index overflow".to_string()),
    }
}

fn bad_request(message: String) -> (&'static str, &'static str, String) {
    (
        "400 Bad Request",
        "application/json",
        format!(r#"{{"error":"{}"}}"#, json_escape(&message)),
    )
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

fn json_slice(shape: &[usize], distributions: &[Distribution], channel: usize) -> String {
    let ndim = shape.len();
    let (h, w) = (shape[ndim - 2], shape[ndim - 1]);

    let offset = channel * h * w;

    let end = (offset + h * w).min(distributions.len());
    let slice = &distributions[offset..end];

    let list = slice
        .iter()
        .map(|&d| format!(r#""{d}""#))
        .collect::<Vec<_>>()
        .join(",");

    format!(r#"{{"rows":{h},"cols":{w},"distributions":[{list}]}}"#)
}

/// Escape a string for embedding in a JSON string literal.
///
/// Error messages are the only untrusted text reaching the response body, and they can carry
/// quotes from a decoder.
fn json_escape(text: &str) -> String {
    text.chars()
        .flat_map(|c| match c {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            '\n' => "\\n".chars().collect(),
            '\r' => "\\r".chars().collect(),
            '\t' => "\\t".chars().collect(),
            c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32).chars().collect(),
            c => vec![c],
        })
        .collect()
}

fn json_digest_cell<T: TensorValue, S: StorageOperations<T>>(
    digest: &mut S,
    dist: Distribution,
    label: &str,
    q_lo: f32,
    q_hi: f32,
) -> crate::Result<String> {
    // 402 quantile points: window endpoints plus 400 evenly spaced interior quantiles.
    let qs: Vec<f32> = std::iter::once(q_lo)
        .chain((0..400).map(|i| q_lo + (q_hi - q_lo) * ((i as f32 + 1.0) / 401.0)))
        .chain(std::iter::once(q_hi))
        .collect();
    let vals = digest.cell_quantiles(0, &qs)?;

    let min = vals[0];
    let max = vals[vals.len() - 1];
    let box_qs = digest.cell_quantiles(0, &[0.25, 0.50, 0.75])?;
    let (q25, q50, q75) = (box_qs[0], box_qs[1], box_qs[2]);

    let count = digest.total_weight(0)?;

    let pdf_pts = qs
        .iter()
        .zip(vals.iter())
        .map(|(&q, &x)| format!(r#"{{"x":{x:.6},"y":{q:.6}}}"#))
        .collect::<Vec<_>>()
        .join(",");

    Ok(format!(
        r#"{{"label":"{label}","distribution":"{dist}","count":{count},"min":{min:.6},"q25":{q25:.6},"q50":{q50:.6},"q75":{q75:.6},"max":{max:.6},"q0":{q_lo:.6},"q1":{q_hi:.6},"pdf":[{pdf_pts}]}}"#
    ))
}

fn json_cell<T: TensorValue, S: StorageOperations<T>>(
    digest: &mut S,
    distributions: &[Distribution],
    idx: usize,
    q_lo: f32,
    q_hi: f32,
    exclude_zero: bool,
) -> crate::Result<String> {
    let dist = distributions
        .get(idx)
        .copied()
        .unwrap_or(Distribution::Normal);
    let mut single = digest.merge_cells(&[idx])?;
    let mut filtered = if exclude_zero {
        single.without_zeros()?
    } else {
        single
    };
    json_digest_cell(&mut filtered, dist, "cell", q_lo, q_hi)
}

fn json_digest_merged<T: TensorValue, S: StorageOperations<T>>(
    mut merged: S,
    q_lo: f32,
    q_hi: f32,
    exclude_zero: bool,
) -> crate::Result<String> {
    if exclude_zero {
        merged = merged.without_zeros()?;
    }
    let dist = merged
        .analyze()?
        .into_iter()
        .next()
        .unwrap_or(Distribution::Unknown);
    json_digest_cell(&mut merged, dist, "merged", q_lo, q_hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tdigest::TDigestStorage;
    use std::net::{Shutdown, TcpListener, TcpStream};

    fn route(shape: &[usize], target: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();

        let request =
            format!("GET {target} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        client.write_all(request.as_bytes()).unwrap();
        client.shutdown(Shutdown::Write).unwrap();

        let mut digest = TDigestStorage::<f32>::new(shape, 100);
        let distributions = vec![Distribution::Normal; shape.iter().product()];
        handle(&server, shape, &distributions, &mut digest);
        drop(server);

        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        response
    }

    fn assert_bad_request(response: &str) {
        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request\r\n"),
            "unexpected response: {response}"
        );
    }

    #[test]
    fn get_merge_rejects_missing_and_invalid_scope() {
        assert_bad_request(&route(&[1, 2, 2], "/api/merge"));
        assert_bad_request(&route(&[1, 2, 2], "/api/merge?scope=bogus"));
    }

    #[test]
    fn slice_rejects_too_few_dimensions_and_bad_coordinates() {
        assert_bad_request(&route(&[4], "/api/slice"));
        assert_bad_request(&route(&[2, 3, 4, 5], "/api/slice?b=2&c=0"));
        assert_bad_request(&route(&[2, 3, 4, 5], "/api/slice?b=0&c=3"));
        assert_bad_request(&route(&[2, 3, 4, 5], "/api/slice?b=nope&c=0"));
    }
}
