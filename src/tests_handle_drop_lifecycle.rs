//! Acceptance for last-handle drop being safe.
//!
//! The sidecar `flock` is released by the writer thread at exit, never at handle
//! drop, so a dropped store's path stays locked until every admitted command has
//! drained. The maintenance park wakes on channel closure, so a store dropped
//! while paused drains — applying its deferred admitted writes — instead of
//! leaking its writer thread forever. These run against the real filesystem via
//! `create_path`/`open_path` on both backends.
//!
//! The drop-while-paused case is the sharp one: without the fix it fails twice
//! over, since the writer leaks and never applies the deferred write *and* the
//! lock releases at drop, so an immediate reopen succeeds against a store
//! missing an admitted write.

use crate::backend::{Direct, Wal};
use crate::error::OpenError;
use crate::map::BTreeMap;
use crate::store::{Options, Store, DEPTH_BOUND};
use crate::WriteBatch;
use futures_lite::future::block_on;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mapdb-t12-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Retry `open` until it succeeds or `deadline` passes. Every failed attempt
/// must be `AlreadyOpen` — the writer still draining — never any other error;
/// the terminal failure mode (pre-fix: a leaked writer holding nothing, path
/// unlocked-but-stale, or a wedged lock) surfaces as a panic here.
fn open_when_free<T>(deadline: Duration, mut open: impl FnMut() -> Result<T, OpenError>) -> T {
    let start = Instant::now();
    loop {
        match open() {
            Ok(store) => return store,
            Err(OpenError::AlreadyOpen) => {
                assert!(
                    start.elapsed() < deadline,
                    "path still locked after {deadline:?}: writer never exited"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => panic!("reopen failed with a non-lock error: {e:?}"),
        }
    }
}

/// The T1.2 core scenario, Wal backend: a write admitted *before* a maintenance
/// pause is applied even when every handle is dropped mid-pause, and the path
/// is reopenable only after that write has landed.
#[test]
fn wal_drop_while_paused_applies_deferred_write_then_frees_path() {
    let dir = tmp_dir("wal-paused");
    let path = dir.join("store.data");
    let store = Store::<Wal>::create_path(&path, Options::default()).unwrap();
    let map = BTreeMap::over(store.clone());

    // Admit the write while Running (a permit granted before the pause stays
    // valid across it — state.rs: Maintenance is enqueue-allowed).
    let batch = WriteBatch::new().insert(b"k".to_vec(), b"deferred-value".to_vec());
    let cost = batch.cost(DEPTH_BOUND);
    let permit = map.try_reserve_write(cost).unwrap();

    // Pause, then enqueue on the held permit: the command lands in the parked
    // writer's deferred queue (or the channel — equivalent) and CANNOT apply
    // until the pause ends... which no one will ever end.
    assert!(store.begin_maintenance());
    let outcome = permit.apply(batch);
    drop(outcome); // invariant 4: the enqueued command survives its future

    // Drop every handle while paused. Pre-T1.2 the writer thread leaks forever
    // (the park never observes channel closure) and the lock releases here.
    drop(map);
    drop(store);

    // The path must become free — and only with the deferred write applied.
    let reopened = open_when_free(Duration::from_secs(10), || {
        Store::<Wal>::open_path(&path, Options::default())
    });
    let map = BTreeMap::over(reopened);
    let got = block_on(map.get(b"k".to_vec())).unwrap();
    assert_eq!(
        got.as_deref(),
        Some(&b"deferred-value"[..]),
        "the admitted-then-deferred write must be applied by the drop-drain"
    );
    block_on(map.close()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Same scenario on the Direct backend, dropped while Running: the drop-drain
/// must flush the applied write durably before the lock releases, so the
/// reopened store always sees it.
#[test]
fn direct_drop_while_running_flushes_before_freeing_path() {
    let dir = tmp_dir("direct-running");
    let path = dir.join("store.data");
    let store = Store::<Direct>::create_path(&path, Options::default()).unwrap();
    let map = BTreeMap::over(store);

    let batch = WriteBatch::new().insert(b"k".to_vec(), b"running-value".to_vec());
    block_on(map.apply(batch)).unwrap();
    // No close(), no flush(): drop is the only teardown.
    drop(map);

    let reopened = open_when_free(Duration::from_secs(10), || {
        Store::<Direct>::open_path(&path, Options::default())
    });
    let map = BTreeMap::over(reopened);
    let got = block_on(map.get(b"k".to_vec())).unwrap();
    assert_eq!(
        got.as_deref(),
        Some(&b"running-value"[..]),
        "the drop-drain must flush applied state before the lock releases"
    );
    block_on(map.close()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// The lock is genuinely held while the writer is parked in Maintenance with
/// live handles — a second opener fails fast `AlreadyOpen` — and released by
/// an explicit close() exactly as before.
#[test]
fn paused_store_holds_its_lock_until_close() {
    let dir = tmp_dir("direct-paused-lock");
    let path = dir.join("store.data");
    let store = Store::<Direct>::create_path(&path, Options::default()).unwrap();

    assert!(store.begin_maintenance());
    match Store::<Direct>::open_path(&path, Options::default()) {
        Err(OpenError::AlreadyOpen) => {}
        Err(other) => panic!("expected AlreadyOpen, got {other:?}"),
        Ok(_) => panic!("a paused store's path must stay locked"),
    }
    store.end_maintenance();
    block_on(store.close()).unwrap();

    // close() joined the writer, which released the lock on exit: the very
    // first reopen attempt must succeed (no retry loop).
    let reopened = Store::<Direct>::open_path(&path, Options::default()).unwrap();
    block_on(reopened.close()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Wal drop while Running with a command still in flight: the immediate reopen
/// attempt is either `AlreadyOpen` (writer mid-drain) or a complete store — a
/// success that is missing the admitted write is the two-writers bug.
#[test]
fn wal_drop_immediate_reopen_never_sees_partial_state() {
    let dir = tmp_dir("wal-race");
    let path = dir.join("store.data");
    let store = Store::<Wal>::create_path(&path, Options::default()).unwrap();
    let map = BTreeMap::over(store);

    let batch = WriteBatch::new().insert(b"k".to_vec(), b"raced-value".to_vec());
    // Enqueue synchronously and drop the future + handles immediately: the
    // writer may still be applying/committing when the drop happens.
    let fut = map.apply(batch);
    // Drive admission to the enqueue point, then abandon the outcome. (With an
    // unContended budget this resolves without parking; polling to completion
    // would serialize with the writer and shrink the race window under test.)
    drop(block_on(fut));
    drop(map);

    let reopened = open_when_free(Duration::from_secs(10), || {
        Store::<Wal>::open_path(&path, Options::default())
    });
    let map = BTreeMap::over(reopened);
    let got = block_on(map.get(b"k".to_vec())).unwrap();
    assert_eq!(got.as_deref(), Some(&b"raced-value"[..]));
    block_on(map.close()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
