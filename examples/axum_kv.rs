//! A durable HTTP key/value service on **Tokio + Axum** over `StoreWal` — the M5
//! interoperability deliverable.
//!
//! The point of this example is that the store is **runtime-neutral**: its single
//! writer actor owns durability on its own thread with `futures_lite::block_on`,
//! and every public future (`get`, `apply`, `commit`, `close`) is `Send` and
//! holds no lock across `.await`. So it drops straight into Axum's handler
//! signatures and Tokio's work-stealing scheduler with no adapter. Contrast
//! `http_kv.rs` (M3), which hand-rolls HTTP on a blocking pool to make the same
//! point without a framework.
//!
//! It demonstrates three load behaviours:
//!   * **fail-fast overload** — a write that cannot immediately reserve admission
//!     returns `503`, never a blocked task (invariant: correctness never depends
//!     on the HTTP executor continuing to poll a request future);
//!   * **timeout** — each store call runs under a `tokio::time::timeout`; a slow
//!     call returns `504` and abandons its future without corrupting state (a
//!     dropped read just stops reading; a granted write permit, once `apply`
//!     returns, owns the command and survives future-drop — invariant 4);
//!   * **graceful shutdown** — `Ctrl-C` stops intake, then `close().await` drains
//!     accepted writes and syncs the WAL (a clean close preserves visible state).
//!
//! Routes:
//!   GET    /kv/:key   → 200 <value> | 404 | 504 timeout
//!   PUT    /kv/:key   → 200 | 503 overloaded | 504 timeout | 413 too large |
//!                       400 empty key | 507 store full
//!   DELETE /kv/:key   → 200 | 503 | 504
//!   GET    /metrics   → 200 text metrics snapshot
//!
//! Status mapping follows the rule "a client-caused fault is 4xx, a capacity limit
//! is 503/507, only an internal fault is 5xx" — a malformed body never reads as a
//! server error.
//!
//! Run: `cargo run --example axum_kv -- 127.0.0.1:7878 /tmp/axum_kv.db`
//! Try: `curl -s -XPUT localhost:7878/kv/hello -d world; curl -s localhost:7878/kv/hello`

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use mapdb_uring::admission::AdmissionLimits;
use mapdb_uring::batch::BatchError;
use mapdb_uring::error::WriteError;
use mapdb_uring::store::Options;
use mapdb_uring::{ApplyOutcome, BTreeMap, Wal, WriteBatch};
use std::time::Duration;

/// Per-request store-call budget. A store call slower than this sheds the request
/// rather than pinning a connection.
const CALL_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
struct AppState {
    map: BTreeMap<Wal>,
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let addr = args.next().unwrap_or_else(|| "127.0.0.1:7878".to_string());
    let path = args.next().unwrap_or_else(|| "/tmp/axum_kv.db".to_string());

    // A deliberately small admission budget so concurrent writes demonstrate
    // fail-fast overload (503) rather than unbounded queuing.
    let options = Options {
        limits: AdmissionLimits {
            max_queue_slots: 64,
            max_command_bytes: 16 * 1024 * 1024,
            max_dirty_pages: 1024,
            max_ops_per_batch: 1024,
        },
        ..Options::default()
    };

    // A real recoverable two-file StoreWal: `<path>` + `<path>.wal`. Open the
    // existing store (replaying its WAL — this service survives a restart) if the
    // files are there, otherwise create a fresh one. `create_path` *truncates*,
    // so it must be the fallback, never the first attempt.
    let data_exists = std::path::Path::new(&path).exists();
    let store = if data_exists {
        mapdb_uring::Store::<Wal>::open_path(&path, options).expect("open StoreWal")
    } else {
        mapdb_uring::Store::<Wal>::create_path(&path, options).expect("create StoreWal")
    };
    let map = BTreeMap::over(store);
    let state = AppState { map };

    let app = Router::new()
        .route(
            "/kv/:key",
            get(get_key).put(put_key).post(put_key).delete(delete_key),
        )
        .route("/metrics", get(metrics))
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    println!("axum_kv listening on http://{addr}  (StoreWal at {path}, Ctrl-C to stop)");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown(state.clone()))
        .await
        .expect("serve");

    // `with_graceful_shutdown` returns once the signal fired and in-flight
    // requests drained; now close the store to sync the WAL and release the lock.
    // A durability service MUST surface a failed close: if the final sync did not
    // land, exit non-zero so an orchestrator does not treat the shutdown as clean.
    if let Err(e) = state.map.close().await {
        eprintln!("close FAILED — durability not confirmed: {e}");
        std::process::exit(1);
    }
    println!("stopped cleanly");
}

/// Resolves when a shutdown signal arrives. Axum then stops accepting new
/// connections and lets in-flight handlers finish before `serve` returns.
async fn shutdown(_state: AppState) {
    let _ = tokio::signal::ctrl_c().await;
    println!("\nshutdown signal — draining");
}

async fn get_key(State(st): State<AppState>, Path(key): Path<String>) -> impl IntoResponse {
    match tokio::time::timeout(CALL_TIMEOUT, st.map.get(key.into_bytes())).await {
        Ok(Ok(Some(v))) => (StatusCode::OK, v),
        Ok(Ok(None)) => (StatusCode::NOT_FOUND, b"not found".to_vec()),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e}").into_bytes(),
        ),
        Err(_) => (StatusCode::GATEWAY_TIMEOUT, b"timeout".to_vec()),
    }
}

async fn put_key(
    State(st): State<AppState>,
    Path(key): Path<String>,
    body: Bytes,
) -> impl IntoResponse {
    write(
        &st,
        WriteBatch::new().insert(key.into_bytes(), body.to_vec()),
    )
    .await
}

async fn delete_key(State(st): State<AppState>, Path(key): Path<String>) -> impl IntoResponse {
    write(&st, WriteBatch::new().remove(key.into_bytes())).await
}

/// Applies a write with **fail-fast overload** and a **timeout**: reserve
/// non-blocking (a full budget → 503), then run the apply under a deadline.
async fn write(st: &AppState, batch: WriteBatch) -> (StatusCode, Vec<u8>) {
    // Size the reservation with the engine's own default cost, so the permit can
    // never mismatch the batch the writer recomputes (no duplicated depth term).
    let cost = batch.default_cost();
    let permit = match st.map.try_reserve_write(cost) {
        Ok(p) => p,
        Err(WriteError::Overloaded) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                b"overloaded, retry later".to_vec(),
            );
        }
        Err(WriteError::Compacting) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                b"maintenance, retry later".to_vec(),
            );
        }
        Err(WriteError::StoreFull) => {
            return (StatusCode::INSUFFICIENT_STORAGE, b"store full".to_vec());
        }
        // A cost above a hard cap (e.g. a batch worst-case over half the dirty
        // budget) is a permanent client error, not a transient one: 413, not 503.
        Err(WriteError::CostExceedsLimit) => {
            return (StatusCode::PAYLOAD_TOO_LARGE, b"batch too large".to_vec());
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{e}").into_bytes(),
            )
        }
    };
    let fut = match permit.apply(batch) {
        Ok(f) => f,
        Err((_batch, _mismatch)) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, b"cost mismatch".to_vec());
        }
    };
    match tokio::time::timeout(CALL_TIMEOUT, fut).await {
        Ok(Ok(ApplyOutcome::Applied { .. })) => (StatusCode::OK, b"ok".to_vec()),
        Ok(Ok(ApplyOutcome::ConditionFailed { .. })) => {
            (StatusCode::CONFLICT, b"condition failed".to_vec())
        }
        Ok(Ok(_)) => (StatusCode::OK, b"ok".to_vec()),
        // Apply-side errors surface here (batch validation and the bidirectional
        // WAL-byte reconcile both resolve the *future*, not the reservation). Map
        // client-caused ones to 4xx/507 consistently with the reservation arm; a
        // malformed body must not read as a server fault.
        Ok(Err(WriteError::InvalidBatch(be))) => match be {
            BatchError::EmptyKey => (StatusCode::BAD_REQUEST, b"empty key".to_vec()),
            // `OldValuesTooLarge` is about the size of the *response* the caller
            // asked for, not the request — but it is still the caller's doing and
            // still a size problem, so it maps the same way.
            BatchError::KeyTooLong
            | BatchError::ValueTooLong
            | BatchError::TooManyOps
            | BatchError::OldValuesTooLarge => (
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("{be:?}").into_bytes(),
            ),
        },
        Ok(Err(WriteError::StoreFull)) => {
            (StatusCode::INSUFFICIENT_STORAGE, b"store full".to_vec())
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e}").into_bytes(),
        ),
        // The apply future is dropped here, but the permit already handed the
        // command to the writer, so the write still lands durably; the client
        // just doesn't learn the outcome on this connection.
        Err(_) => (StatusCode::GATEWAY_TIMEOUT, b"timeout".to_vec()),
    }
}

async fn metrics(State(st): State<AppState>) -> impl IntoResponse {
    let m = st.map.metrics();
    let body = format!(
        "entry_count {}\nvisible_txid {}\ndurable_txid {}\nadmission_slots {}\nadmission_bytes {}\ndirty_pages {}\nwal_logical_bytes {}\nwal_reserved_bytes {}\nwal_physical_bytes {}\noverlay_pages {}\nopen_scan_ms {}\nopen_scan_bytes {}\n",
        m.entry_count,
        m.visible_txid,
        m.durable_txid,
        m.admission_slots,
        m.admission_bytes,
        m.dirty_pages,
        m.wal_logical_bytes,
        m.wal_reserved_bytes,
        m.wal_physical_bytes,
        m.overlay_pages,
        m.open_scan_ms,
        m.open_scan_bytes,
    );
    (StatusCode::OK, body)
}
