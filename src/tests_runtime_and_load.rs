//! Foreign-runtime interoperability and load behaviour for the public futures.
//!
//! Compile-time `Send`/`Sync` assertions are necessary but not sufficient: the
//! store's futures must also run correctly when polled by a foreign reactor and
//! move between worker threads under a work-stealing scheduler. These tests
//! drive the real read/write/commit/close paths under such runtimes; exercise
//! the load surface (overload fail-fast, timeout leaking no capacity, a parked
//! reservation granted by a release, retry after overload, shutdown draining
//! then rejecting); assert the WAL byte and overlay metrics across a reopen; pin
//! the tree builder's split-policy density for ascending, descending and random
//! insertion orders; and pin merge-apply semantics for multi-op batches that
//! touch one key repeatedly or empty and refill the map mid-batch.
//!
//! The foreign runtimes used here are dev-dependencies: available to these unit
//! tests, never linked into a plain library build.

use crate::admission::WriteCost;
use crate::backend::{Backend, Direct, Wal};
use crate::batch::{ApplyOutcome, WriteBatch};
use crate::error::WriteError;
use crate::io::{FakeIo, PageIo};
use crate::map::BTreeMap;
use crate::page::PageKind;
use crate::store::{Options, Store};
use crate::version::Version;
use futures_lite::future::block_on;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// -- Helpers ----------------------------------------------------------------

fn direct_map() -> BTreeMap<Direct> {
    let store = Store::<Direct>::create_with_io(Arc::new(FakeIo::new(0)), Options::default())
        .expect("create direct");
    BTreeMap::over(store)
}

/// A Direct map with a deliberately small dirty-page budget (for overload/timeout
/// exercises). A legal single reservation is at most `budget / 2` (invariant 16).
fn small_budget_map(budget: u32) -> BTreeMap<Direct> {
    let store = Store::<Direct>::create_with_io(
        Arc::new(FakeIo::new(0)),
        Options {
            limits: crate::admission::AdmissionLimits {
                max_dirty_pages: budget,
                ..crate::admission::AdmissionLimits::default()
            },
            ..Options::default()
        },
    )
    .expect("create direct");
    BTreeMap::over(store)
}

fn wal_devices() -> (Arc<FakeIo>, Arc<FakeIo>) {
    (Arc::new(FakeIo::new(0)), Arc::new(FakeIo::new(0)))
}

fn wal_map_with(data: &Arc<FakeIo>, wal: &Arc<FakeIo>, options: Options) -> BTreeMap<Wal> {
    let store = Store::<Wal>::create_with_io(
        Arc::clone(data) as Arc<dyn PageIo>,
        Arc::clone(wal) as Arc<dyn PageIo>,
        options,
    )
    .expect("create wal");
    BTreeMap::over(store)
}

/// A `WriteCost` of `pages` worst-case dirty pages and nothing else — enough to
/// drive admission accounting without a real batch (the exercise tests only need
/// to occupy budget).
fn dirty_cost(pages: u32) -> WriteCost {
    WriteCost {
        encoded_bytes: 0,
        op_count: 1,
        worst_case_pages: pages,
        wal_worst_case_bytes: 0,
    }
}

fn applied_version(out: ApplyOutcome) -> Version {
    match out {
        ApplyOutcome::Applied { version, .. } => version,
        other => panic!("expected Applied, got {other:?}"),
    }
}

/// Counts the pages **reachable from the currently visible root** by walking the
/// live tree through the cache — the true tree size, as opposed to
/// `overlay_pages` (the cumulative count of every page the WAL ever indexed).
fn reachable_pages<B: Backend>(map: &BTreeMap<B>) -> u64 {
    let core = &map.store().inner.core;
    let desc = core.root();
    let cache = &core.cache;
    let mut seen = std::collections::HashSet::new();
    block_on(async {
        let mut stack = vec![desc.root];
        while let Some(r) = stack.pop() {
            if r.is_empty() || !seen.insert(r.page_id) {
                continue;
            }
            let page = cache.get_page(r, &desc.file).await.unwrap();
            if page.kind() == PageKind::Branch {
                for c in page.branch_children() {
                    stack.push(*c);
                }
            }
        }
    });
    seen.len() as u64
}

fn unique_path(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "mapdb-uring-m5-{}-{}-{}.data",
        tag,
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_file(&p);
    p
}

// ===========================================================================
// D2 — interoperability: the futures run under foreign reactors and move
// across worker threads.
// ===========================================================================

/// A single-threaded, non-`futures_lite` foreign executor (smol) drives a
/// write-then-read round trip. Proves the store makes no assumption about which
/// runtime polls its futures.
#[test]
fn interop_smol_round_trip() {
    let map = direct_map();
    smol::block_on(async {
        let out = map
            .apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))
            .await
            .unwrap();
        let _ = applied_version(out);
        assert_eq!(map.get(b"k".to_vec()).await.unwrap(), Some(b"v".to_vec()));
        assert_eq!(map.get(b"absent".to_vec()).await.unwrap(), None);
    });
}

/// A Tokio current-thread runtime drives the same round trip.
#[test]
fn interop_tokio_current_thread() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let map = direct_map();
    rt.block_on(async {
        map.apply(WriteBatch::new().insert(b"a".to_vec(), b"1".to_vec()))
            .await
            .unwrap();
        assert_eq!(map.get(b"a".to_vec()).await.unwrap(), Some(b"1".to_vec()));
    });
}

/// A Tokio **multi-thread** runtime `spawn`s the store's futures onto its
/// work-stealing pool. `spawn` statically requires `Send + 'static`, and the
/// `apply` reply oneshot is woken from the writer thread, so the apply future's
/// waker genuinely crosses threads. (Warm-cache `get`s over `FakeIo` complete on
/// first poll — the cross-thread *suspension* of a read is exercised separately
/// by `interop_syscall_cold_read_suspends_under_tokio`.) All reads observe the
/// single committed version.
#[test]
fn interop_tokio_multithread_spawn() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let map = direct_map();
    rt.block_on(async {
        for i in 0..50u32 {
            let m = map.clone();
            let v = applied_version(
                tokio::spawn(m.apply(
                    WriteBatch::new().insert(format!("k{i:03}").into_bytes(), vec![i as u8; 32]),
                ))
                .await
                .unwrap()
                .unwrap(),
            );
            assert_eq!(v.incarnation(), map.store().incarnation());
        }
        let mut handles = Vec::new();
        for i in 0..50u32 {
            let m = map.clone();
            handles.push(tokio::spawn(async move {
                m.get(format!("k{i:03}").into_bytes()).await
            }));
        }
        for (i, h) in handles.into_iter().enumerate() {
            let got = h.await.unwrap().unwrap();
            assert_eq!(got, Some(vec![i as u8; 32]), "key k{i:03}");
        }
    });
}

/// The most interop-sensitive wait in the engine is the StoreWal **commit
/// frontier** wait: it is notified from the writer thread via an
/// `event_listener`, so a foreign runtime's waker must drive it to
/// completion. Spawn `apply` + `commit` onto the Tokio pool and await the
/// durable receipt.
#[test]
fn interop_wal_commit_under_tokio_multithread() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let (data, wal) = wal_devices();
    let map = wal_map_with(&data, &wal, Options::default());
    rt.block_on(async {
        let mut handles = Vec::new();
        for i in 0..20u32 {
            let m = map.clone();
            handles.push(tokio::spawn(async move {
                let out = m
                    .apply(
                        WriteBatch::new()
                            .insert(format!("w{i:03}").into_bytes(), vec![i as u8; 40]),
                    )
                    .await
                    .unwrap();
                let v = applied_version(out);
                // The commit frontier wait — polled here by a Tokio worker.
                m.commit(v).await.unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(map.metrics().durable_txid, map.metrics().visible_txid);
        assert_eq!(
            map.get(b"w000".to_vec()).await.unwrap(),
            Some(vec![0u8; 40])
        );
    });
}

/// A **genuinely suspending** read under a foreign runtime: reopen a real-file
/// Direct store with a cold cache so `get` faults a page in from `SyscallIo`
/// (whose worker-pool future actually returns `Pending`), and drive it under the
/// Tokio multi-thread runtime. This exercises a physical page-load future being
/// woken on a foreign executor (invariants 7/12).
#[test]
fn interop_syscall_cold_read_suspends_under_tokio() {
    let path = unique_path("cold-read");
    // Seed a store on a real file, then close it.
    {
        let store = Store::<Direct>::create_path(&path, Options::default()).unwrap();
        let map = BTreeMap::over(store);
        block_on(async {
            for i in 0..200u32 {
                map.apply(
                    WriteBatch::new().insert(format!("k{i:04}").into_bytes(), vec![i as u8; 48]),
                )
                .await
                .unwrap();
            }
            map.close().await.unwrap();
        });
    }
    // Reopen cold and read under Tokio.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let store = Store::<Direct>::open_path(&path, Options::default()).unwrap();
    let map = BTreeMap::over(store);
    rt.block_on(async {
        let mut handles = Vec::new();
        for i in 0..200u32 {
            let m = map.clone();
            handles.push(tokio::spawn(async move {
                (i, m.get(format!("k{i:04}").into_bytes()).await.unwrap())
            }));
        }
        for h in handles {
            let (i, got) = h.await.unwrap();
            assert_eq!(got, Some(vec![i as u8; 48]), "cold read k{i:04}");
        }
    });
    let _ = std::fs::remove_file(&path);
    // Best-effort cleanup of the sidecar lock (`<path>.lock`).
    let mut lock = path.clone().into_os_string();
    lock.push(".lock");
    let _ = std::fs::remove_file(PathBuf::from(lock));
}

// ===========================================================================
// D3 — overload / timeout / retry / shutdown exercises.
// ===========================================================================

/// Overload is **fail-fast**: with the dirty-page budget occupied by held
/// permits, `try_reserve_write` returns `Overloaded` immediately (it never
/// blocks). Releasing the held permits restores capacity.
#[test]
fn exercise_overload_fail_fast() {
    let map = small_budget_map(20);
    // A legal single batch is at most half the budget (invariant 16), so occupy
    // the whole budget with two half-budget permits.
    let held1 = map
        .try_reserve_write(dirty_cost(10))
        .expect("first half fits");
    let held2 = map
        .try_reserve_write(dirty_cost(10))
        .expect("second half fits");
    match map.try_reserve_write(dirty_cost(1)) {
        Err(WriteError::Overloaded) => {}
        Err(e) => panic!("expected Overloaded, got {e:?}"),
        Ok(_) => panic!("expected Overloaded, got a granted permit"),
    }
    drop(held1);
    drop(held2);
    let _p = map
        .try_reserve_write(dirty_cost(10))
        .expect("fits after release");
}

/// A store call can be bounded by a runtime timeout, and abandoning the future
/// leaks **zero** admission capacity. The dirty budget is saturated by two held
/// permits, so `reserve_write` awaits indefinitely; `tokio::time::timeout` fires,
/// we drop the future, then — after releasing the holders — we re-reserve the
/// **entire** budget (two half-budget permits). If the abandoned future had
/// leaked any of its 10-page charge, the second half would not fit.
#[test]
fn exercise_timeout_leaks_no_capacity() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let map = small_budget_map(20);
    rt.block_on(async {
        let held1 = map
            .try_reserve_write(dirty_cost(10))
            .expect("first half fits");
        let held2 = map
            .try_reserve_write(dirty_cost(10))
            .expect("second half fits");
        let res = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            map.reserve_write(dirty_cost(10)),
        )
        .await;
        assert!(res.is_err(), "expected the reservation to time out");
        drop(held1);
        drop(held2);
        // Zero-leak proof: the full budget is reservable again.
        let a = map
            .try_reserve_write(dirty_cost(10))
            .expect("first half free");
        let b = map
            .try_reserve_write(dirty_cost(10))
            .expect("second half free — no leak");
        drop((a, b));
        // And the store still applies and reads.
        map.apply(WriteBatch::new().insert(b"after".to_vec(), b"ok".to_vec()))
            .await
            .unwrap();
        assert_eq!(
            map.get(b"after".to_vec()).await.unwrap(),
            Some(b"ok".to_vec())
        );
    });
}

/// A **parked** `reserve_write` is woken and granted when a holder releases
/// capacity — driven to completion by a foreign (Tokio) waker. This exercises the
/// release → waiter-notify → grant path the M5 exercises otherwise never
/// complete: every prior test either drops the parked future or wakes it
/// by poison.
#[test]
fn exercise_parked_reserve_granted_by_release() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let map = small_budget_map(20);
    rt.block_on(async {
        let held1 = map
            .try_reserve_write(dirty_cost(10))
            .expect("first half fits");
        let held2 = map
            .try_reserve_write(dirty_cost(10))
            .expect("second half fits");
        // Park a reservation on the pool, sized to the batch it will apply (a real
        // permit rejects a batch whose recomputed cost exceeds the reservation).
        let batch = WriteBatch::new().insert(b"granted".to_vec(), b"y".to_vec());
        let cost = batch.cost(crate::store::DEPTH_BOUND);
        let m = map.clone();
        let task = tokio::spawn(async move { m.reserve_write(cost).await });
        // Wait until the spawned reservation has actually *parked* as the FIFO
        // waiter before releasing capacity — otherwise (slow scheduler) the holders
        // could drop first and the grant would never exercise the
        // release → notify → foreign-waker path this test exists
        // for. Deterministic, not a timer race.
        let budget = &map.store().inner.core.budget;
        while budget.waiter_count() == 0 {
            tokio::task::yield_now().await;
        }
        drop(held1);
        drop(held2);
        // The parked reservation is granted; the permit is usable.
        let permit = task.await.unwrap().expect("parked reservation granted");
        let fut = permit.apply(batch).expect("cost within permit");
        fut.await.unwrap();
        assert_eq!(
            map.get(b"granted".to_vec()).await.unwrap(),
            Some(b"y".to_vec())
        );
    });
}

/// Fail-fast + caller retry: after an `Overloaded` on a real batch's cost,
/// retrying **the same batch** once capacity frees succeeds and is
/// visible. The rejected and retried reservations use one identical cost.
#[test]
fn exercise_retry_after_overload() {
    let map = small_budget_map(20);
    let held1 = map.try_reserve_write(dirty_cost(10)).unwrap();
    let held2 = map.try_reserve_write(dirty_cost(10)).unwrap();

    let batch = WriteBatch::new().insert(b"retry".to_vec(), b"done".to_vec());
    let cost = batch.cost(crate::store::DEPTH_BOUND);
    // The same cost is rejected while the budget is full…
    assert!(matches!(
        map.try_reserve_write(cost),
        Err(WriteError::Overloaded)
    ));
    // …and accepted after capacity frees; retry the unchanged batch.
    drop(held1);
    drop(held2);
    let permit = map.try_reserve_write(cost).expect("retry fits");
    let fut = permit.apply(batch).expect("cost within permit");
    block_on(fut).unwrap();
    assert_eq!(
        block_on(map.get(b"retry".to_vec())).unwrap(),
        Some(b"done".to_vec())
    );
}

/// Graceful shutdown actually **drains pending work**: the writes are applied
/// (visible) but left **uncommitted**, and one is enqueued via a permit whose
/// outcome future is dropped without awaiting (invariant 4). `close()` must drain
/// and sync all of it, so the durable image recovers every write on reopen — the
/// assertion would fail if `close` dropped pending batches. Every post-close
/// `apply` **and** `get` returns the terminal `Closed`.
#[test]
fn exercise_shutdown_drains_then_rejects() {
    let (data, wal) = wal_devices();
    let map = wal_map_with(&data, &wal, Options::default());

    block_on(async {
        // Visible-but-uncommitted writes (no commit awaited).
        for i in 0..6u32 {
            map.apply(WriteBatch::new().insert(format!("s{i}").into_bytes(), vec![i as u8; 16]))
                .await
                .unwrap();
        }
        // One enqueued via a permit whose reply future is dropped un-awaited: the
        // command owns the permit and must still be drained by close (invariant 4).
        let batch = WriteBatch::new().insert(b"s6".to_vec(), vec![6u8; 16]);
        let cost = batch.cost(crate::store::DEPTH_BOUND);
        let permit = map.try_reserve_write(cost).unwrap();
        let fut = permit.apply(batch).expect("cost within permit");
        drop(fut); // enqueued synchronously; outcome abandoned.

        map.close().await.unwrap();

        // Post-close: both intake and reads are terminal.
        match map
            .apply(WriteBatch::new().insert(b"late".to_vec(), b"x".to_vec()))
            .await
        {
            Err(WriteError::Closed) => {}
            other => panic!("expected Closed apply, got {other:?}"),
        }
        match map.get(b"s0".to_vec()).await {
            Err(crate::error::ReadError::Closed) => {}
            other => panic!("expected Closed get, got {other:?}"),
        }
    });

    // Reopen over the durable images: every drained write recovered, including
    // the one whose outcome future was dropped.
    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let w = Arc::new(FakeIo::from_vec(wal.synced_bytes()));
    let store = Store::<Wal>::open_over(
        Arc::clone(&d) as Arc<dyn PageIo>,
        Arc::clone(&w) as Arc<dyn PageIo>,
        Options::default(),
    )
    .unwrap();
    let map = BTreeMap::over(store);
    block_on(async {
        for i in 0..7u32 {
            assert_eq!(
                map.get(format!("s{i}").into_bytes()).await.unwrap(),
                Some(vec![i as u8; 16]),
                "recovered s{i}"
            );
        }
    });
}

// ===========================================================================
// D6 — metric assertions: wal bytes, overlay, and the scan marker.
// ===========================================================================

/// WAL byte and overlay metrics track real appends: logical grows, reserved is
/// exactly logical once quiescent (no permits outstanding), and physical equals
/// logical on a clean (untorn) append history.
#[test]
fn metrics_wal_bytes_and_overlay() {
    let (data, wal) = wal_devices();
    let map = wal_map_with(&data, &wal, Options::default());

    let before = map.metrics();
    assert_eq!(before.wal_logical_bytes, before.wal_physical_bytes);
    assert_eq!(before.open_scan_bytes, 0, "fresh create ran no scan");

    block_on(async {
        for i in 0..30u32 {
            let out = map
                .apply(
                    WriteBatch::new().insert(format!("m{i:03}").into_bytes(), vec![i as u8; 128]),
                )
                .await
                .unwrap();
            let v = applied_version(out);
            map.commit(v).await.unwrap();
        }
    });

    let after = map.metrics();
    assert!(
        after.wal_logical_bytes > before.wal_logical_bytes,
        "logical bytes should grow with appends"
    );
    // Quiescent (all committed, no permits): reserved == logical exactly. `>=`
    // would not catch a leaked reservation.
    assert_eq!(
        after.wal_reserved_bytes, after.wal_logical_bytes,
        "quiescent reserved must equal logical"
    );
    // Clean append history: physical == logical exactly.
    assert_eq!(
        after.wal_physical_bytes, after.wal_logical_bytes,
        "clean history: physical == logical"
    );
    assert!(
        after.overlay_pages > 0,
        "overlay indexes the committed pages"
    );
    assert_eq!(after.entry_count, 30);
}

/// After reopening a WAL store with records, the recovery scan is unambiguously
/// observable via `open_scan_bytes > 0` (a fresh create reports `0`), the overlay
/// count matches the pre-close overlay exactly (recovery rebuilds it from the
/// same accepted records), and the tree recovers intact.
#[test]
fn metrics_open_scan_and_overlay_after_reopen() {
    let (data, wal) = wal_devices();
    let (pre_close_overlay, pre_close_logical) = {
        let map = wal_map_with(&data, &wal, Options::default());
        assert_eq!(
            map.metrics().open_scan_bytes,
            0,
            "fresh create does not scan"
        );
        assert_eq!(map.metrics().open_scan_ms, 0);
        block_on(async {
            for i in 0..40u32 {
                let out = map
                    .apply(
                        WriteBatch::new()
                            .insert(format!("r{i:03}").into_bytes(), vec![i as u8; 64]),
                    )
                    .await
                    .unwrap();
                let v = applied_version(out);
                map.commit(v).await.unwrap();
            }
            let m = map.metrics();
            let snapshot = (m.overlay_pages, m.wal_logical_bytes);
            map.close().await.unwrap();
            snapshot
        })
    };

    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let w = Arc::new(FakeIo::from_vec(wal.synced_bytes()));
    let store = Store::<Wal>::open_over(
        Arc::clone(&d) as Arc<dyn PageIo>,
        Arc::clone(&w) as Arc<dyn PageIo>,
        Options::default(),
    )
    .unwrap();
    let map = BTreeMap::over(store);
    let m = map.metrics();
    // Unambiguous "a scan ran", pinned exactly: `open_scan_bytes` is the recovered
    // logical frontier, so on this clean (untorn) history it equals both the
    // pre-close logical frontier and the reopened logical-bytes metric — not just
    // `> 0` or `>=`.
    assert_eq!(
        m.open_scan_bytes, pre_close_logical,
        "recovery accepted exactly the records"
    );
    assert_eq!(
        m.open_scan_bytes, m.wal_logical_bytes,
        "scan frontier == reopened logical"
    );
    assert_eq!(
        m.overlay_pages, pre_close_overlay,
        "recovery rebuilds the same overlay cardinality"
    );
    assert_eq!(m.entry_count, 40, "all committed entries recovered");
    assert_eq!(
        m.wal_physical_bytes, m.wal_logical_bytes,
        "clean recovered history: physical == logical"
    );
}

// ===========================================================================
// Builder density regression.
// ===========================================================================

/// Split-policy density regression, **updated for the M6 Phase C fix**.
///
/// Pre-M6, greedy leaf packing was dense for ascending/bulk (append) but fragmented
/// badly under random/interior/descending single-op inserts (the old `[full][tiny]`
/// tail re-split every insert — measured at 1,731 leaves for 1,500 descending
/// keys vs ~10 ascending). The insertion-direction-aware split (byte-balance the
/// last two chunks on a non-append overflow, greedy only for appends) collapses
/// that: **every order now packs to within a small factor of a bulk build.** This
/// test pins the *fixed* behaviour — ascending stays optimal AND random/descending
/// no longer fragment — so a future split-policy regression is caught either way.
/// Reachable pages (not cumulative `overlay_pages`) are the density measure.
#[test]
fn builder_split_policy_dense_across_orders() {
    const N: u32 = 1_000;
    let opts = || Options {
        limits: crate::admission::AdmissionLimits {
            max_dirty_pages: 4_000_000,
            max_ops_per_batch: 100_000,
            ..crate::admission::AdmissionLimits::default()
        },
        ..Options::default()
    };
    let mk = || {
        BTreeMap::over(Store::<Direct>::create_with_io(Arc::new(FakeIo::new(0)), opts()).unwrap())
    };

    // Ascending single-op.
    let asc = mk();
    block_on(async {
        for i in 0..N {
            asc.apply(WriteBatch::new().insert(format!("k{i:08}").into_bytes(), vec![0u8; 64]))
                .await
                .unwrap();
        }
    });

    // Random single-op: a deterministic *permutation* of 0..N (a bijection since
    // gcd(2_654_435_761, 1000) == 1), computed in u64 to avoid a u32 wrap that
    // would collapse it to a non-injective map. So all N keys are distinct and
    // inserted in scattered order — the fragmenting workload.
    let rnd = mk();
    block_on(async {
        for j in 0..N {
            let i = ((j as u64).wrapping_mul(2_654_435_761) % N as u64) as u32;
            rnd.apply(WriteBatch::new().insert(format!("k{i:08}").into_bytes(), vec![0u8; 64]))
                .await
                .unwrap();
        }
    });

    // Descending single-op — the worst case pre-fix (every insert hits the hot
    // left leaf). Each insert lands before the current minimum, so it is never an
    // append: the balanced path must fire.
    let dsc = mk();
    block_on(async {
        for i in (0..N).rev() {
            dsc.apply(WriteBatch::new().insert(format!("k{i:08}").into_bytes(), vec![0u8; 64]))
                .await
                .unwrap();
        }
    });

    // Bulk (one batch) — the reference for maximal density.
    let bulk = mk();
    block_on(async {
        let mut b = WriteBatch::new();
        for i in 0..N {
            b = b.insert(format!("k{i:08}").into_bytes(), vec![0u8; 64]);
        }
        bulk.apply(b).await.unwrap();
    });

    let asc_pages = reachable_pages(&asc);
    let rnd_pages = reachable_pages(&rnd);
    let dsc_pages = reachable_pages(&dsc);
    let bulk_pages = reachable_pages(&bulk);

    // Correctness is order-independent (verify passes; counts exact on every order).
    assert_eq!(block_on(asc.verify()).unwrap().entry_count, N as u64);
    assert_eq!(block_on(rnd.verify()).unwrap().entry_count, N as u64);
    assert_eq!(block_on(dsc.verify()).unwrap().entry_count, N as u64);

    // The M6 fix: ascending stays optimal, and random/descending — which used to
    // fragment to ≥3× ascending — now pack to within a small factor of a bulk
    // build. `body_capacity/2` amortized fill on the balanced path means a scattered
    // order can at most ~double the leaf count of a dense build; assert ≤ 2×, well
    // under the old ≥3×-fragmentation these very inputs produced.
    assert!(
        asc_pages <= bulk_pages * 2,
        "ascending should be near-dense: asc={asc_pages} bulk={bulk_pages}"
    );
    assert!(
        rnd_pages <= bulk_pages * 2,
        "random must not fragment post-fix: rnd={rnd_pages} bulk={bulk_pages}"
    );
    assert!(
        dsc_pages <= bulk_pages * 2,
        "descending must not fragment post-fix: dsc={dsc_pages} bulk={bulk_pages}"
    );
}

// == Merge-apply semantics ===================================================
//
// `btree::apply_batch` key-sorts a batch and applies it as one merge descent.
// These pin the semantics the rewrite must preserve: same-key ops keep their
// batch order (stable sort), a batch that empties and refills the tree
// mid-sequence lands on the same final content, remove-all canonicalizes to
// the EMPTY root, and a wide multi-leaf mixed batch matches a model exactly.

#[test]
fn merge_apply_same_key_ops_keep_batch_order() {
    let map = direct_map();
    block_on(async {
        // insert → remove → insert on one key inside ONE batch: last insert wins.
        map.apply(
            WriteBatch::new()
                .insert(b"k".to_vec(), b"v1".to_vec())
                .remove(b"k".to_vec())
                .insert(b"k".to_vec(), b"v2".to_vec()),
        )
        .await
        .unwrap();
        assert_eq!(map.get(b"k".to_vec()).await.unwrap(), Some(b"v2".to_vec()));
        // insert → remove ends absent, and the no-op delta nets to zero.
        map.apply(
            WriteBatch::new()
                .insert(b"g".to_vec(), b"x".to_vec())
                .remove(b"g".to_vec()),
        )
        .await
        .unwrap();
        assert_eq!(map.get(b"g".to_vec()).await.unwrap(), None);
        assert_eq!(map.metrics().entry_count, 1);
    });
}

#[test]
fn merge_apply_batch_that_empties_and_refills_mid_batch() {
    let map = direct_map();
    block_on(async {
        map.apply(
            WriteBatch::new()
                .insert(b"a".to_vec(), b"1".to_vec())
                .insert(b"b".to_vec(), b"2".to_vec()),
        )
        .await
        .unwrap();
        // One batch: remove everything, then insert fresh keys. In batch order
        // the tree is momentarily empty; final content must be exactly the
        // fresh keys.
        map.apply(
            WriteBatch::new()
                .remove(b"a".to_vec())
                .remove(b"b".to_vec())
                .insert(b"c".to_vec(), b"3".to_vec())
                .insert(b"z".to_vec(), b"26".to_vec()),
        )
        .await
        .unwrap();
        assert_eq!(map.get(b"a".to_vec()).await.unwrap(), None);
        assert_eq!(map.get(b"b".to_vec()).await.unwrap(), None);
        assert_eq!(map.get(b"c".to_vec()).await.unwrap(), Some(b"3".to_vec()));
        assert_eq!(map.get(b"z".to_vec()).await.unwrap(), Some(b"26".to_vec()));
        assert_eq!(map.metrics().entry_count, 2);
        map.verify().await.unwrap();
    });
}

#[test]
fn merge_apply_remove_all_canonicalizes_empty() {
    let map = direct_map();
    block_on(async {
        map.apply(
            WriteBatch::new()
                .insert(b"a".to_vec(), b"1".to_vec())
                .insert(b"b".to_vec(), b"2".to_vec()),
        )
        .await
        .unwrap();
        map.apply(
            WriteBatch::new()
                .remove(b"a".to_vec())
                .remove(b"b".to_vec()),
        )
        .await
        .unwrap();
        assert_eq!(map.metrics().entry_count, 0);
        assert_eq!(map.get(b"a".to_vec()).await.unwrap(), None);
        // The emptied tree accepts new writes.
        map.apply(WriteBatch::new().insert(b"n".to_vec(), b"9".to_vec()))
            .await
            .unwrap();
        assert_eq!(map.get(b"n".to_vec()).await.unwrap(), Some(b"9".to_vec()));
        assert_eq!(map.metrics().entry_count, 1);
    });
}

/// A wide mixed batch over a multi-leaf tree — overwrites, removes, interior
/// inserts, and inserts beyond the rightmost key, all in one batch — matches a
/// `std::collections::BTreeMap` model exactly (content and count).
#[test]
fn merge_apply_wide_mixed_batch_matches_model() {
    let map = direct_map();
    let mut model = std::collections::BTreeMap::<Vec<u8>, Vec<u8>>::new();
    block_on(async {
        // Multi-leaf base: 2000 keys × 100-byte values (> 10 leaves at 16 KiB).
        for chunk in (0..2000u32).collect::<Vec<_>>().chunks(500) {
            let mut b = WriteBatch::new();
            for &i in chunk {
                let k = format!("key:{i:06}").into_bytes();
                let v = vec![b'a' + (i % 26) as u8; 100];
                model.insert(k.clone(), v.clone());
                b = b.insert(k, v);
            }
            map.apply(b).await.unwrap();
        }
        // One 400-op mixed batch across the whole key space.
        let mut b = WriteBatch::new();
        let mut x: u64 = 0xDEAD_BEEF;
        for _ in 0..400 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let i = (x % 2600) as u32; // beyond 2000 → routes past the rightmost leaf
            let k = format!("key:{i:06}").into_bytes();
            match x % 3 {
                0 => {
                    let v = vec![b'z'; 120];
                    model.insert(k.clone(), v.clone());
                    b = b.insert(k, v);
                }
                1 => {
                    model.remove(&k);
                    b = b.remove(k);
                }
                _ => {
                    let v = format!("fresh:{i}").into_bytes();
                    model.insert(k.clone(), v.clone());
                    b = b.insert(k, v);
                }
            }
        }
        map.apply(b).await.unwrap();
        assert_eq!(map.metrics().entry_count, model.len() as u64);
        for i in 0..2600u32 {
            let k = format!("key:{i:06}").into_bytes();
            assert_eq!(
                map.get(k.clone()).await.unwrap(),
                model.get(&k).cloned(),
                "key {i}"
            );
        }
        map.verify().await.unwrap();
    });
}

/// A batch that removes the last live key
/// and reinserts (same batch) on a sparse multi-level no-merge skeleton. The
/// old per-op writer happened to reset to EMPTY mid-batch and discard the
/// skeleton; the merge descent deliberately rebuilds within it (module doc —
/// same sparse shape multi-batch deletes always leave; `compact` reclaims).
/// Content, count, verify, and continued usability are what this pins, for
/// replacement keys below the first, inside, and beyond the last separator.
#[test]
fn merge_apply_transient_empty_on_sparse_skeleton() {
    for replacement in [&b"key:000000"[..], &b"key:001000"[..], &b"zzz"[..]] {
        let map = direct_map();
        block_on(async {
            // Multi-level tree (2000 × 100B ≈ >10 leaves), then strip to ONE
            // live key across earlier batches — the no-merge sparse skeleton.
            for chunk in (0..2000u32).collect::<Vec<_>>().chunks(500) {
                let mut b = WriteBatch::new();
                for &i in chunk {
                    b = b.insert(format!("key:{i:06}").into_bytes(), vec![b'v'; 100]);
                }
                map.apply(b).await.unwrap();
            }
            for chunk in (0..2000u32)
                .filter(|&i| i != 1000)
                .collect::<Vec<_>>()
                .chunks(500)
            {
                let mut b = WriteBatch::new();
                for &i in chunk {
                    b = b.remove(format!("key:{i:06}").into_bytes());
                }
                map.apply(b).await.unwrap();
            }
            assert_eq!(map.metrics().entry_count, 1);
            // One batch: remove the last live key, insert the replacement.
            map.apply(
                WriteBatch::new()
                    .remove(b"key:001000".to_vec())
                    .insert(replacement.to_vec(), b"fresh".to_vec()),
            )
            .await
            .unwrap();
            assert_eq!(map.metrics().entry_count, 1);
            assert_eq!(
                map.get(replacement.to_vec()).await.unwrap(),
                Some(b"fresh".to_vec())
            );
            if replacement != b"key:001000" {
                assert_eq!(map.get(b"key:001000".to_vec()).await.unwrap(), None);
            }
            map.verify().await.unwrap();
            // The tree stays fully usable after the transient-empty batch.
            map.apply(WriteBatch::new().insert(b"after".to_vec(), b"ok".to_vec()))
                .await
                .unwrap();
            assert_eq!(map.metrics().entry_count, 2);
            assert_eq!(
                map.get(b"after".to_vec()).await.unwrap(),
                Some(b"ok".to_vec())
            );
        });
    }
}
