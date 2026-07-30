//! A minimal HTTP key/value server over `StoreDirect`, demonstrating the M3
//! store and — the point of this example — **fail-fast overload**: a write that
//! cannot immediately reserve admission returns `503 Service Unavailable` rather
//! than blocking the request thread. This follows the rule that "no
//! correctness progress depends on the HTTP executor continuing to poll a
//! particular request future": the store's single writer owns durability; the
//! server just sheds load when the admission budget is momentarily full.
//!
//! No async runtime and no HTTP framework — a tiny hand-rolled HTTP/1.1 parse on
//! a fixed thread pool, each request resolved with `futures_lite::block_on`.
//! A real Tokio/Hyper service is the M5 deliverable.
//!
//! Routes:
//!   GET    /<key>   → 200 <value> | 404
//!   PUT    /<key>   → 200 (body is the value) | 503 when overloaded
//!   DELETE /<key>   → 200
//!
//! Run: `cargo run --example http_kv -- 127.0.0.1:7878`
//! Try: `curl -s -XPUT localhost:7878/hello -d world; curl -s localhost:7878/hello`

use mapdb_uring::admission::AdmissionLimits;
use mapdb_uring::error::WriteError;
use mapdb_uring::io::FakeIo;
use mapdb_uring::store::Options;
use mapdb_uring::{ApplyOutcome, BTreeMap, Direct, Store, WriteBatch};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

/// Nominal per-batch depth term for the admission-cost estimate (an engine
/// detail the writer reconciles to actual; an example may estimate).
const DEPTH_BOUND: u32 = 8;

fn main() {
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:7878".to_string());

    // A deliberately small admission budget so concurrent writes demonstrate
    // fail-fast overload (503) rather than unbounded queuing.
    let options = Options {
        limits: AdmissionLimits {
            max_queue_slots: 32,
            max_command_bytes: 8 * 1024 * 1024,
            max_dirty_pages: 512,
            max_ops_per_batch: 1024,
        },
        ..Options::default()
    };
    let store =
        Store::<Direct>::create_with_io(Arc::new(FakeIo::new(0)), options).expect("create store");
    let map = BTreeMap::over(store);

    let listener = TcpListener::bind(&addr).expect("bind");
    println!("http_kv listening on http://{addr}  (Ctrl-C to stop)");

    // A fixed worker pool; each shares the cheap store handle.
    let workers = 8;
    let listener = Arc::new(listener);
    let mut handles = Vec::new();
    for _ in 0..workers {
        let listener = Arc::clone(&listener);
        let map = map.clone();
        handles.push(std::thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(s) => handle_conn(s, &map),
                    Err(_) => continue,
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
}

fn handle_conn(mut stream: TcpStream, map: &BTreeMap<Direct>) {
    let peer = stream.try_clone();
    let mut reader = BufReader::new(match peer {
        Ok(s) => s,
        Err(_) => return,
    });

    // Request line: METHOD SP TARGET SP HTTP/1.1
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() || line.is_empty() {
        return;
    }
    let mut parts = line.split_whitespace();
    let (method, target) = match (parts.next(), parts.next()) {
        (Some(m), Some(t)) => (m.to_string(), t.to_string()),
        _ => return respond(&mut stream, 400, "Bad Request", b"bad request line"),
    };
    let key = target.trim_start_matches('/').as_bytes().to_vec();

    // Headers → find Content-Length.
    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).is_err() {
            return;
        }
        let trimmed = h.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some(v) = trimmed.strip_prefix("Content-Length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }

    match method.as_str() {
        "GET" => match futures_lite::future::block_on(map.get(key)) {
            Ok(Some(v)) => respond(&mut stream, 200, "OK", &v),
            Ok(None) => respond(&mut stream, 404, "Not Found", b"not found"),
            Err(e) => respond(
                &mut stream,
                500,
                "Internal Server Error",
                format!("{e}").as_bytes(),
            ),
        },
        "PUT" | "POST" => write_batch(&mut stream, map, WriteBatch::new().insert(key, body)),
        "DELETE" => write_batch(&mut stream, map, WriteBatch::new().remove(key)),
        _ => respond(
            &mut stream,
            405,
            "Method Not Allowed",
            b"method not allowed",
        ),
    }
}

/// Applies a write with **fail-fast overload**: reserve non-blocking; a
/// momentarily-full budget yields `503`, never a blocked request thread.
fn write_batch(stream: &mut TcpStream, map: &BTreeMap<Direct>, batch: WriteBatch) {
    let cost = batch.cost(DEPTH_BOUND);
    let permit = match map.try_reserve_write(cost) {
        Ok(p) => p,
        Err(WriteError::Overloaded) => {
            return respond(
                stream,
                503,
                "Service Unavailable",
                b"overloaded, retry later",
            );
        }
        Err(WriteError::Compacting) => {
            return respond(
                stream,
                503,
                "Service Unavailable",
                b"maintenance, retry later",
            );
        }
        Err(e) => {
            return respond(
                stream,
                500,
                "Internal Server Error",
                format!("{e}").as_bytes(),
            )
        }
    };
    let fut = match permit.apply(batch) {
        Ok(f) => f,
        Err((_batch, _mismatch)) => {
            return respond(stream, 500, "Internal Server Error", b"cost mismatch");
        }
    };
    match futures_lite::future::block_on(fut) {
        Ok(ApplyOutcome::Applied { .. }) => respond(stream, 200, "OK", b"ok"),
        Ok(ApplyOutcome::ConditionFailed { .. }) => {
            respond(stream, 409, "Conflict", b"condition failed")
        }
        Ok(_) => respond(stream, 200, "OK", b"ok"),
        Err(e) => respond(
            stream,
            500,
            "Internal Server Error",
            format!("{e}").as_bytes(),
        ),
    }
}

fn respond(stream: &mut TcpStream, code: u16, reason: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}
