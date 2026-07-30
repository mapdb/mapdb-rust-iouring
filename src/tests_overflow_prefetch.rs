//! Acceptance for overflow-chain prefetch.
//!
//! The oracle is the FakeIo read gate: park physical reads in submission order,
//! observe the speculative window's depth while frozen, then release. Four
//! properties are under test.
//!
//! Issue-ahead: with the head demand read parked, the driver already has the
//! full window of speculative link reads submitted. Advisory prediction: a wrong
//! guess — non-consecutive ids, a predicted location disagreeing with the
//! authoritative `locate`, or a failed or corrupt speculative read — degrades to
//! demand reads with the identical verdict the naive path gives, never an error
//! of speculation's own. No shadowing: the cache's `Loading` or resident state
//! always wins, a private image is never consumed when the cache holds the page,
//! and speculation never inserts into the cache. Accounting: permits recover on
//! every path including caller cancellation, issued reads drain to terminal
//! completion, and the counters expose waste rather than hiding it.

use crate::backend::{DataFileLocator, FileGeneration, Location, Member, PageLocator};
use crate::cache::{CacheState, PageCache};
use crate::io::{FakeIo, Faults, PageIo};
use crate::page::{
    encode_overflow, overflow_chunks, Checksum, OverflowCursor, OverflowRef, Page, PageId, PageRef,
    PAGE_SIZE,
};
use crate::read::{read_overflow, TreeBuilder, TreeReader, CHAIN_PREFETCH_WINDOW};
use futures_lite::future::block_on;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Per-step deadline for gated waits (same rationale as `tests_get_many_concurrency`).
const STEP_TIMEOUT: Duration = Duration::from_secs(10);

/// A deterministic large value spanning exactly `links` chain links (the tail
/// link one byte short of full).
fn big_value(seed: u8, links: usize) -> Vec<u8> {
    let len = crate::page::overflow_payload_capacity() * links - 1;
    (0..len)
        .map(|i| seed.wrapping_add((i % 249) as u8))
        .collect()
}

fn key(i: usize) -> Vec<u8> {
    format!("key{i:06}").into_bytes()
}

/// Wraps any locator with the id-only prediction capability by forging a ref
/// with a zero checksum — sound **for `DataFileLocator` fixtures only**, whose
/// `locate` documents an id-keyed table (the production capability
/// implementations do not forge refs, which is why the capability is a
/// distinct method).
struct Predicting(Arc<dyn PageLocator>);

impl PageLocator for Predicting {
    fn locate(&self, page_ref: PageRef) -> Option<Location> {
        self.0.locate(page_ref)
    }
    fn predict_overflow_location(&self, page_id: PageId) -> Option<Location> {
        self.0.locate(PageRef {
            page_id,
            expected_checksum: Checksum(0),
        })
    }
}

/// Rebuilds a `TreeBuilder` fixture's generation with the prediction capability
/// declared (the scaffold `DataFileLocator` deliberately stays opaque).
fn declare_prediction(built: crate::read::BuiltTree) -> crate::read::BuiltTree {
    let dev = built.file.device(Member::Data);
    let locator = Arc::new(Predicting(Arc::clone(built.file.locator())));
    crate::read::BuiltTree {
        file: FileGeneration::direct(dev, locator),
        root: built.root,
        entry_count: built.entry_count,
    }
}

/// Writes a single chain on a raw device: `ids[i]` is link i's page id, placed
/// at slot-sequential offsets (`i × PAGE_SIZE`). Returns the head ref.
fn raw_chain(dev: &Arc<FakeIo>, ids: &[u64], value: &[u8]) -> OverflowRef {
    let chunks: Vec<&[u8]> = overflow_chunks(value).collect();
    assert_eq!(chunks.len(), ids.len(), "ids must cover the chain exactly");
    let mut pages: Vec<Page> = Vec::with_capacity(ids.len());
    let mut next: Option<PageRef> = None;
    for (i, chunk) in chunks.iter().enumerate().rev() {
        let page = encode_overflow(PageId(ids[i]), 0, chunk, next);
        next = Some(page.page_ref());
        pages.push(page);
    }
    pages.reverse(); // head-first
    for (slot, page) in pages.iter().enumerate() {
        block_on(dev.write_all_at(
            slot as u64 * PAGE_SIZE as u64,
            page.bytes().to_vec().into_boxed_slice(),
        ))
        .result
        .unwrap();
    }
    OverflowRef {
        head: next.unwrap(),
        total_len: value.len() as u64,
    }
}

/// The `id → slot-offset` truth table for a [`raw_chain`] fixture.
fn chain_locator(ids: &[u64]) -> DataFileLocator {
    let locator = DataFileLocator::new();
    for (slot, id) in ids.iter().enumerate() {
        locator.record(*id, slot as u64 * PAGE_SIZE as u64);
    }
    locator
}

/// A raw-chain fixture with a fresh prefetch-enabled cache and a *predicting*
/// truth locator. `file` is `Arc` so gated tests can hand it to a thread.
struct RawRun {
    cache: Arc<PageCache>,
    file: Arc<FileGeneration>,
    oref: OverflowRef,
}

fn raw_run(dev: &Arc<FakeIo>, ids: &[u64], value: &[u8]) -> RawRun {
    let oref = raw_chain(dev, ids, value);
    let file = Arc::new(FileGeneration::direct(
        Arc::clone(dev) as Arc<dyn PageIo>,
        Arc::new(Predicting(Arc::new(chain_locator(ids)))),
    ));
    let cache = Arc::new(PageCache::new());
    cache.set_chain_prefetch(true);
    RawRun { cache, file, oref }
}

impl RawRun {
    fn read(&self) -> Result<Vec<u8>, crate::error::ReadError> {
        block_on(read_overflow(&self.cache, &self.file, self.oref))
    }
}

// ---------------------------------------------------------------------------
// Issue-ahead and the happy path.
// ---------------------------------------------------------------------------

/// The structural issue-ahead evidence: while the head's demand read is still
/// parked at the device, the full speculative window is already submitted and
/// parked beside it. A serial chain reader can never show more than one parked
/// read here.
#[test]
fn window_is_issued_while_head_read_is_parked() {
    let links = CHAIN_PREFETCH_WINDOW + 4; // window-capped, replenished later
    let dev = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (10..10 + links as u64).collect();
    let run = raw_run(&dev, &ids, &big_value(1, links));

    dev.reset_read_stats();
    dev.arm_read_gate(CHAIN_PREFETCH_WINDOW + 1);
    let permits_before = run.cache.chain_prefetch_permits();

    let h = {
        let (cache, file, oref) = (Arc::clone(&run.cache), Arc::clone(&run.file), run.oref);
        std::thread::spawn(move || block_on(read_overflow(&cache, &file, oref)))
    };

    let deadline = Instant::now() + STEP_TIMEOUT;
    while dev.parked_read_count() < CHAIN_PREFETCH_WINDOW + 1 {
        if Instant::now() >= deadline {
            dev.release_reads(1_000_000);
            let _ = h.join();
            panic!("window never parked");
        }
        std::thread::yield_now();
    }
    // Frozen: exactly head + W submissions — the window is capped, not the
    // whole chain, and nothing has completed yet.
    assert_eq!(dev.reads_submitted(), CHAIN_PREFETCH_WINDOW + 1);
    assert_eq!(
        run.cache.chain_prefetch_permits(),
        permits_before - CHAIN_PREFETCH_WINDOW
    );

    dev.release_reads(1_000_000);
    let got = h.join().unwrap().unwrap();
    assert_eq!(got, big_value(1, links));

    let s = run.cache.chain_prefetch_stats();
    assert_eq!(s.engaged_chains, 1);
    assert_eq!(s.issued, links as u64 - 1, "every link past the head");
    assert_eq!(s.consumed, links as u64 - 1, "zero waste on the happy path");
    assert_eq!((s.unused, s.redundant, s.fallbacks), (0, 0, 0));
    assert_eq!(run.cache.chain_prefetch_permits(), permits_before);
    assert_eq!(dev.reads_in_flight(), 0, "all speculative reads terminal");
    // No shadowing/pollution: only the head went through the cache.
    assert_eq!(run.cache.metrics().loads_issued, 1);
    assert_eq!(run.cache.metrics().clean_resident, 1);
}

/// Chains below the engagement threshold never speculate.
#[test]
fn short_chains_do_not_engage() {
    let dev = Arc::new(FakeIo::new(0));
    let run = raw_run(&dev, &[5, 6, 7], &big_value(2, 3));
    assert_eq!(run.read().unwrap(), big_value(2, 3));
    let s = run.cache.chain_prefetch_stats();
    assert_eq!((s.engaged_chains, s.issued), (0, 0));
}

/// A warm head disqualifies the chain (re-reading a cached chain from the
/// device is pure redundancy).
#[test]
fn warm_head_does_not_engage() {
    let links = 8;
    let dev = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (3..3 + links as u64).collect();
    let run = raw_run(&dev, &ids, &big_value(3, links));

    assert_eq!(run.read().unwrap(), big_value(3, links));
    assert_eq!(run.cache.chain_prefetch_stats().engaged_chains, 1);

    // The head is clean-resident now (its demand load); the re-read must run
    // demand-only.
    assert_eq!(run.read().unwrap(), big_value(3, links));
    let s = run.cache.chain_prefetch_stats();
    assert_eq!(s.engaged_chains, 1, "warm-head chain re-engaged");
    assert_eq!(s.issued, links as u64 - 1, "no new speculative reads");
}

/// Without the switch nothing engages, even with the capability declared and a
/// long cold chain.
#[test]
fn prefetch_is_off_by_default() {
    let links = 12;
    let dev = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (10..10 + links as u64).collect();
    let run = raw_run(&dev, &ids, &big_value(4, links));
    run.cache.set_chain_prefetch(false);
    dev.reset_read_stats();
    assert_eq!(run.read().unwrap(), big_value(4, links));
    assert_eq!(run.cache.chain_prefetch_stats().engaged_chains, 0);
    assert_eq!(dev.reads_submitted(), links, "demand reads only");
}

/// The scaffold `DataFileLocator` does not declare the capability: an
/// undeclared locator must never engage regardless of the switch (the
/// "deliberately non-contiguous locator" case in its default form).
#[test]
fn undeclared_locator_never_engages() {
    let links = 12;
    let dev = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (10..10 + links as u64).collect();
    let oref = raw_chain(&dev, &ids, &big_value(5, links));
    let file = FileGeneration::direct(
        Arc::clone(&dev) as Arc<dyn PageIo>,
        Arc::new(chain_locator(&ids)),
    );
    let cache = PageCache::new();
    cache.set_chain_prefetch(true);
    let got = block_on(read_overflow(&cache, &file, oref)).unwrap();
    assert_eq!(got, big_value(5, links));
    assert_eq!(cache.chain_prefetch_stats().engaged_chains, 0);
}

// ---------------------------------------------------------------------------
// Misprediction, fallback, and the location tooth.
// ---------------------------------------------------------------------------

/// Non-consecutive chain ids under a *declaring* locator: each predicted id
/// resolves to a real page with a different embedded id, the reference-
/// independent decode rejects it, and the chain falls back to demand reads —
/// correct value, one fallback, every issued read counted wasted and drained.
#[test]
fn non_consecutive_chain_falls_back() {
    let links = 8;
    let dev = Arc::new(FakeIo::new(0));
    // head=20, remaining links on a far id range (40, 41, …) — nothing at
    // 21..27. The plausible-but-wrong table maps the *predicted* ids 21..27
    // (disjoint from every real id, so the truth rows stay intact) onto the
    // real links' offsets, so speculation actually issues instead of trivially
    // declining — and each speculative decode sees a valid page whose embedded
    // id is not the predicted one.
    let mut ids: Vec<u64> = vec![20];
    ids.extend((1..links as u64).map(|i| 39 + i));
    let oref = raw_chain(&dev, &ids, &big_value(6, links));
    let locator = chain_locator(&ids);
    for slot in 1..links {
        locator.record(20 + slot as u64, slot as u64 * PAGE_SIZE as u64);
    }
    let file = FileGeneration::direct(
        Arc::clone(&dev) as Arc<dyn PageIo>,
        Arc::new(Predicting(Arc::new(locator))),
    );
    let cache = PageCache::new();
    cache.set_chain_prefetch(true);
    dev.reset_read_stats();

    let got = block_on(read_overflow(&cache, &file, oref)).unwrap();
    assert_eq!(got, big_value(6, links));
    let s = cache.chain_prefetch_stats();
    assert_eq!(s.engaged_chains, 1);
    assert_eq!(s.fallbacks, 1, "exactly one fallback per chain");
    assert_eq!(s.consumed, 0, "nothing consumable from a wrong id stream");
    assert_eq!(s.unused, s.issued, "every speculative read wasted, counted");
    assert_eq!(
        dev.reads_in_flight(),
        0,
        "wasted reads drained, not dropped"
    );
}

/// The location-equality tooth: the predicted location holds
/// a byte-identical copy of the *correct* page, but authoritative `locate`
/// names a different offset. Full-ref equality alone would consume it; the
/// mandated location comparison must reject it and fall back.
#[test]
fn predicted_location_must_match_authoritative_locate() {
    let links = 6;
    let dev = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (30..30 + links as u64).collect();
    let oref = raw_chain(&dev, &ids, &big_value(7, links));

    // Duplicate every link's bytes into a decoy region.
    let decoy_base = 100 * PAGE_SIZE as u64;
    for slot in 0..links {
        let bytes = block_on(dev.read_exact_at(slot as u64 * PAGE_SIZE as u64, PAGE_SIZE)).unwrap();
        block_on(dev.write_all_at(decoy_base + slot as u64 * PAGE_SIZE as u64, bytes))
            .result
            .unwrap();
    }

    struct DecoyPredict {
        truth: DataFileLocator,
        decoy_base: u64,
        head: u64,
    }
    impl PageLocator for DecoyPredict {
        fn locate(&self, page_ref: PageRef) -> Option<Location> {
            self.truth.locate(page_ref)
        }
        fn predict_overflow_location(&self, page_id: PageId) -> Option<Location> {
            let pos = page_id.as_u64().checked_sub(self.head)?;
            Some(Location {
                member: Member::Data,
                offset: self.decoy_base + pos * PAGE_SIZE as u64,
                length: PAGE_SIZE as u32,
            })
        }
    }
    let file = FileGeneration::direct(
        Arc::clone(&dev) as Arc<dyn PageIo>,
        Arc::new(DecoyPredict {
            truth: chain_locator(&ids),
            decoy_base,
            head: ids[0],
        }),
    );
    let cache = PageCache::new();
    cache.set_chain_prefetch(true);

    let got = block_on(read_overflow(&cache, &file, oref)).unwrap();
    assert_eq!(got, big_value(7, links));
    let s = cache.chain_prefetch_stats();
    assert_eq!(s.engaged_chains, 1);
    assert_eq!(
        s.consumed, 0,
        "byte-identical page at the wrong location must not be consumed"
    );
    assert_eq!(s.fallbacks, 1);
    assert_eq!(dev.reads_in_flight(), 0);
}

/// A failed speculative read is not an error: the demand fallback re-reads the
/// same offset successfully (single-shot order fault), and the value is intact.
#[test]
fn spec_read_failure_falls_back_without_error() {
    let links = 8;
    let dev = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (40..40 + links as u64).collect();
    let run = raw_run(&dev, &ids, &big_value(8, links));

    dev.reset_read_stats();
    // The driver polls its window before the demand future, so submission #0 is
    // the speculative read of link 1. Fail exactly that one, once.
    dev.set_faults(Faults {
        fail_read_orders: vec![0],
        ..Default::default()
    });

    let got = run.read().unwrap();
    assert_eq!(got, big_value(8, links), "speculation must not surface EIO");
    let s = run.cache.chain_prefetch_stats();
    assert_eq!(s.fallbacks, 1);
    assert_eq!(s.consumed, 0, "fallback abandons the whole window");
    assert_eq!(s.unused, s.issued);
    assert_eq!(dev.reads_in_flight(), 0);
}

/// Error parity: a genuinely corrupt link yields the *same* verdict with and
/// without prefetch — the demand path owns every error.
#[test]
fn corrupt_link_error_matches_naive_path() {
    let links = 8;
    let corrupt_slot = 3u64;
    let make_dev = || {
        let dev = Arc::new(FakeIo::new(0));
        let ids: Vec<u64> = (50..50 + links as u64).collect();
        let oref = raw_chain(&dev, &ids, &big_value(9, links));
        block_on(dev.write_all_at(
            corrupt_slot * PAGE_SIZE as u64,
            vec![0xEE; PAGE_SIZE].into_boxed_slice(),
        ))
        .result
        .unwrap();
        (dev, ids, oref)
    };

    // Naive arm: opaque locator, switch irrelevant.
    let (dev_n, ids_n, oref_n) = make_dev();
    let file_n = FileGeneration::direct(
        Arc::clone(&dev_n) as Arc<dyn PageIo>,
        Arc::new(chain_locator(&ids_n)),
    );
    let err_n = block_on(read_overflow(&PageCache::new(), &file_n, oref_n)).unwrap_err();

    // Prefetch arm: engaged, speculative pos-3 read decodes garbage → fallback
    // → the demand read owns the verdict.
    let (dev_p, ids_p, oref_p) = make_dev();
    let file_p = FileGeneration::direct(
        Arc::clone(&dev_p) as Arc<dyn PageIo>,
        Arc::new(Predicting(Arc::new(chain_locator(&ids_p)))),
    );
    let cache_p = PageCache::new();
    cache_p.set_chain_prefetch(true);
    let err_p = block_on(read_overflow(&cache_p, &file_p, oref_p)).unwrap_err();

    assert_eq!(format!("{err_n:?}"), format!("{err_p:?}"));
    assert_eq!(cache_p.chain_prefetch_stats().fallbacks, 1);
    assert_eq!(dev_p.reads_in_flight(), 0, "drained even on the error path");
}

// ---------------------------------------------------------------------------
// Cache arbitration.
// ---------------------------------------------------------------------------

/// A resident link wins over a confirmed speculative image: the private page is
/// discarded (`redundant`), the cache serves the position, and replenishment
/// stops — but the already-issued window tail is still consumed.
#[test]
fn resident_link_wins_over_speculation() {
    let links = 10;
    let warm_pos = 4usize;
    let dev = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (60..60 + links as u64).collect();
    let run = raw_run(&dev, &ids, &big_value(10, links));

    // Resolve link refs by walking the chain on a throwaway cache, then warm
    // exactly `warm_pos` on the cache under test.
    let refs: Vec<PageRef> = {
        let scratch = PageCache::new();
        let mut refs = vec![run.oref.head];
        let mut cursor = OverflowCursor::new(run.oref);
        while let Some(r) = cursor.pending() {
            let page = block_on(scratch.get_page(r, &run.file)).unwrap();
            cursor.accept(&page).unwrap();
            if let Some(n) = cursor.pending() {
                refs.push(n);
            }
        }
        refs
    };
    block_on(run.cache.get_page(refs[warm_pos], &run.file)).unwrap();
    assert_eq!(run.cache.metrics().clean_resident, 1);

    let got = run.read().unwrap();
    assert_eq!(got, big_value(10, links));
    let s = run.cache.chain_prefetch_stats();
    assert_eq!(s.engaged_chains, 1);
    assert_eq!(s.redundant, 1, "exactly the warm position");
    assert_eq!(s.fallbacks, 0, "resident-wins is not a fallback");
    // Window (9 ≤ W) was fully issued up front; the warm position's image is
    // the only waste, and every other position was consumed privately.
    assert_eq!(s.issued, links as u64 - 1);
    assert_eq!(s.unused, 1);
    assert_eq!(s.consumed, links as u64 - 2);
    assert_eq!(dev.reads_in_flight(), 0);
}

/// Concurrent demand for a predicted page: another reader's cache load of a
/// mid-chain link is parked in flight when the prefetching chain reaches that
/// position. The cache's single-flight token wins — the chain *joins* the load
/// instead of consuming its private image, both readers complete with the same
/// bytes, and the cache issued that load exactly once.
#[test]
fn inflight_cache_load_wins_over_speculation() {
    let links = 10;
    let contested = 5usize;
    let dev = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (70..70 + links as u64).collect();
    let run = raw_run(&dev, &ids, &big_value(11, links));

    // Resolve the contested ref straight off the device images (no cache).
    let mut refs = vec![run.oref.head];
    for pos in 0..links - 1 {
        let bytes = block_on(dev.read_exact_at(pos as u64 * PAGE_SIZE as u64, PAGE_SIZE)).unwrap();
        let page = Page::decode(Vec::from(bytes).into(), refs[pos]).unwrap();
        refs.push(page.overflow_next().expect("interior link has a successor"));
    }
    let contested_ref = refs[contested];

    dev.reset_read_stats();
    // Only the *first* submission parks: reader B's demand load, started before
    // the chain read; everything else runs ungated.
    dev.arm_read_gate(1);
    let hb = {
        let (cache, file) = (Arc::clone(&run.cache), Arc::clone(&run.file));
        std::thread::spawn(move || block_on(cache.get_page(contested_ref, &file)))
    };
    let deadline = Instant::now() + STEP_TIMEOUT;
    while dev.parked_read_count() < 1 {
        assert!(Instant::now() < deadline, "B's load never parked");
        std::thread::yield_now();
    }
    assert_eq!(
        run.cache.state_of(contested_ref.page_id),
        Some(CacheState::Loading)
    );

    let ha = {
        let (cache, file, oref) = (Arc::clone(&run.cache), Arc::clone(&run.file), run.oref);
        std::thread::spawn(move || block_on(read_overflow(&cache, &file, oref)))
    };

    // A must classify the contested position redundant (Loading wins) and then
    // block joining B's parked load — it cannot finish before the release.
    let deadline = Instant::now() + STEP_TIMEOUT;
    while run.cache.chain_prefetch_stats().redundant < 1 {
        if Instant::now() >= deadline {
            dev.release_reads(1_000_000);
            let _ = ha.join();
            let _ = hb.join();
            panic!("chain never classified the contested position redundant");
        }
        std::thread::yield_now();
    }
    assert!(!ha.is_finished(), "A is blocked joining B's parked load");
    dev.release_reads(1_000_000);

    let a = ha.join().unwrap().unwrap();
    let b = hb.join().unwrap().unwrap();
    assert_eq!(a, big_value(11, links));
    assert_eq!(b.page_ref(), contested_ref);
    // The contested page was loaded by the cache exactly once (B's load; A
    // joined): loads = head + contested.
    assert_eq!(run.cache.metrics().loads_issued, 2);
    assert_eq!(dev.reads_in_flight(), 0);
}

// ---------------------------------------------------------------------------
// Permits: cancellation recovery and the aggregate bound.
// ---------------------------------------------------------------------------

/// Caller cancellation mid-window: permits recover via the guards' drop, the
/// parked speculative reads un-park (`SyncGate` drop guard), and a fresh chain
/// engages the full window afterwards. (`reads_in_flight` deliberately strands
/// on FakeIo for dropped futures — the documented divergence — so this test
/// never asserts it.)
#[test]
fn cancellation_recovers_permits() {
    let links = CHAIN_PREFETCH_WINDOW + 4;
    let dev = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (90..90 + links as u64).collect();
    let run = raw_run(&dev, &ids, &big_value(13, links));
    let permits_full = run.cache.chain_prefetch_permits();

    dev.reset_read_stats();
    dev.arm_read_gate(CHAIN_PREFETCH_WINDOW + 1);
    {
        let fut = read_overflow(&run.cache, &run.file, run.oref);
        futures_lite::pin!(fut);
        let deadline = Instant::now() + STEP_TIMEOUT;
        while dev.parked_read_count() < CHAIN_PREFETCH_WINDOW + 1 {
            assert!(Instant::now() < deadline, "window never parked");
            assert!(
                block_on(futures_lite::future::poll_once(fut.as_mut())).is_none(),
                "chain finished while every read should be parked"
            );
            std::thread::yield_now();
        }
    } // the chain future drops here, mid-window
    assert_eq!(
        run.cache.chain_prefetch_permits(),
        permits_full,
        "permit guards must release on drop"
    );
    // The 16 dropped *speculative* futures un-parked via the gate's drop guard.
    // The 1 still parked is the head's **cache** load: single-flight is
    // cancellation-independent, so the orphaned `LoadState` retains its device
    // future — exactly the contrast between cache-owned demand reads and
    // driver-owned speculation.
    assert_eq!(
        dev.parked_read_count(),
        1,
        "only the cache's orphan remains"
    );

    // The claimed slots are dead but the pool is healthy. (The *same* chain
    // cannot re-engage: its head is still `Loading` in the cache — the orphaned
    // single-flight load — and the cold-head check correctly declines.) A
    // fresh chain on the same cache must engage the full window from the
    // recovered pool.
    dev.release_reads(1_000_000);
    let dev2 = Arc::new(FakeIo::new(0));
    let ids2: Vec<u64> = (200..200 + links as u64).collect();
    let oref2 = raw_chain(&dev2, &ids2, &big_value(14, links));
    let file2 = FileGeneration::direct(
        Arc::clone(&dev2) as Arc<dyn PageIo>,
        Arc::new(Predicting(Arc::new(chain_locator(&ids2)))),
    );
    let got = block_on(read_overflow(&run.cache, &file2, oref2)).unwrap();
    assert_eq!(got, big_value(14, links));
    assert_eq!(run.cache.chain_prefetch_stats().engaged_chains, 2);
    assert_eq!(run.cache.chain_prefetch_permits(), permits_full);
    assert_eq!(dev2.reads_in_flight(), 0);
}

/// The global permit pool bounds aggregate speculation across concurrent
/// chains (T2.1 runs chains concurrently): with every read parked, total
/// speculative submissions plateau at exactly the pool size, not chains × W —
/// and recover fully afterwards.
#[test]
fn permit_pool_bounds_aggregate_speculation() {
    // 8 chains × window 16 = 128 wanted > 64 permits.
    let chains = 8usize;
    let links = CHAIN_PREFETCH_WINDOW + 2;
    let n = chains * 4;
    let mut data: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| (key(i), format!("value-{i}").into_bytes()))
        .collect();
    for c in 0..chains {
        data[c * 4].1 = big_value(c as u8, links);
    }
    let dev = Arc::new(FakeIo::new(0));
    let built = TreeBuilder {
        max_fanout: 100,
        max_leaf_entries: 4,
        ..Default::default()
    }
    .build(Arc::clone(&dev) as Arc<dyn PageIo>, &data)
    .unwrap();
    let cache = Arc::new(PageCache::new());
    cache.set_chain_prefetch(true);
    let reader = Arc::new(TreeReader::with_cache(
        declare_prediction(built),
        Arc::clone(&cache),
    ));
    let permits_full = cache.chain_prefetch_permits();

    // Warm the tree pages with inline-key reads so the gated phase sees only
    // chain traffic; heads and links stay cold.
    let warm: Vec<Vec<u8>> = (0..n).filter(|i| i % 4 != 0).map(key).collect();
    block_on(reader.get_many(warm)).unwrap();

    dev.reset_read_stats();
    dev.arm_read_gate(usize::MAX / 2); // park everything
    let query: Vec<Vec<u8>> = (0..chains).map(|c| key(c * 4)).collect();
    let h = {
        let reader = Arc::clone(&reader);
        std::thread::spawn(move || block_on(reader.get_many_grouped(query)))
    };

    // Plateau: `chains` demand head loads + exactly the permit pool.
    let expect = chains + permits_full;
    let deadline = Instant::now() + STEP_TIMEOUT;
    while dev.parked_read_count() < expect {
        if Instant::now() >= deadline {
            dev.release_reads(1_000_000);
            let _ = h.join();
            panic!(
                "aggregate plateau never reached: parked={} want={expect}",
                dev.parked_read_count()
            );
        }
        std::thread::yield_now();
    }
    assert_eq!(
        dev.reads_submitted(),
        expect,
        "pool-dry: no more submissions"
    );
    assert_eq!(cache.chain_prefetch_permits(), 0);
    assert!(cache.chain_prefetch_stats().permit_denials > 0);

    // The completed-buffer half of the bound: release exactly the FIRST
    // chain's window (slots 0..15 — it
    // was polled first, so its 16 speculative reads own the first 16 gate
    // slots). Those reads COMPLETE while every demand head stays parked; the
    // 16 private pages are now completed-but-unconsumed, and the pool must
    // still be empty — a permit released at read completion (instead of at
    // consumption) would show 16 free permits here and let later chains
    // recycle them into more private buffers.
    dev.release_reads(CHAIN_PREFETCH_WINDOW);
    let deadline = Instant::now() + STEP_TIMEOUT;
    while dev.parked_read_count() > expect - CHAIN_PREFETCH_WINDOW {
        if Instant::now() >= deadline {
            dev.release_reads(1_000_000);
            let _ = h.join();
            panic!("first chain's window never completed");
        }
        std::thread::yield_now();
    }
    assert_eq!(
        cache.chain_prefetch_permits(),
        0,
        "completed private buffers must stay charged until consumed"
    );
    assert_eq!(
        dev.reads_submitted(),
        expect,
        "no permit recycling occurred"
    );

    dev.release_reads(1_000_000);
    let got = h.join().unwrap().unwrap();
    for (c, v) in got.iter().enumerate() {
        assert_eq!(v.as_ref().unwrap(), &big_value(c as u8, links));
    }
    assert_eq!(cache.chain_prefetch_permits(), permits_full);
    assert_eq!(dev.reads_in_flight(), 0);
}

/// Permit denial **stops** a chain's speculation outright: a later
/// permit-availability window must never issue a
/// slot for a position the cursor already passed on demand — the stale front
/// slot would block every later one. The denied chain runs demand-only,
/// counts exactly one denial, and no fallback (nothing mispredicted).
#[test]
fn permit_denial_stops_speculation_cleanly() {
    let links = CHAIN_PREFETCH_WINDOW + 4;
    let cache = Arc::new(PageCache::new());
    cache.set_chain_prefetch(true);
    let permits_full = cache.chain_prefetch_permits();
    let holders = permits_full / CHAIN_PREFETCH_WINDOW; // 4 chains drain 64

    // Each holder chain parks on its own gated device, holding a full window.
    let mut hs = Vec::new();
    let mut hdevs = Vec::new();
    for c in 0..holders {
        let dev = Arc::new(FakeIo::new(0));
        let ids: Vec<u64> =
            (1_000 + 100 * c as u64..1_000 + 100 * c as u64 + links as u64).collect();
        let oref = raw_chain(&dev, &ids, &big_value(30 + c as u8, links));
        let file = Arc::new(FileGeneration::direct(
            Arc::clone(&dev) as Arc<dyn PageIo>,
            Arc::new(Predicting(Arc::new(chain_locator(&ids)))),
        ));
        dev.arm_read_gate(usize::MAX / 2);
        let (cache2, file2) = (Arc::clone(&cache), Arc::clone(&file));
        hs.push((
            c,
            std::thread::spawn(move || block_on(read_overflow(&cache2, &file2, oref))),
        ));
        let deadline = Instant::now() + STEP_TIMEOUT;
        while dev.parked_read_count() < CHAIN_PREFETCH_WINDOW + 1 {
            if Instant::now() >= deadline {
                for d in &hdevs {
                    let d: &Arc<FakeIo> = d;
                    d.release_reads(1_000_000);
                }
                dev.release_reads(1_000_000);
                panic!("holder {c} never parked its window");
            }
            std::thread::yield_now();
        }
        hdevs.push(dev);
    }
    assert_eq!(cache.chain_prefetch_permits(), 0, "pool drained by holders");
    let before = cache.chain_prefetch_stats();

    // The denied chain: own (ungated) device, same cache. It must engage,
    // hit one denial, and finish demand-only — correct bytes, no fallback.
    let dev_b = Arc::new(FakeIo::new(0));
    let ids_b: Vec<u64> = (5_000..5_000 + links as u64).collect();
    let oref_b = raw_chain(&dev_b, &ids_b, &big_value(40, links));
    let file_b = FileGeneration::direct(
        Arc::clone(&dev_b) as Arc<dyn PageIo>,
        Arc::new(Predicting(Arc::new(chain_locator(&ids_b)))),
    );
    let got = block_on(read_overflow(&cache, &file_b, oref_b)).unwrap();
    assert_eq!(got, big_value(40, links));
    let after = cache.chain_prefetch_stats();
    assert_eq!(after.engaged_chains - before.engaged_chains, 1);
    assert_eq!(after.issued, before.issued, "denied chain issued nothing");
    assert_eq!(after.consumed, before.consumed);
    assert_eq!(
        after.permit_denials - before.permit_denials,
        1,
        "stop-on-denial tries exactly once"
    );
    assert_eq!(
        after.fallbacks, before.fallbacks,
        "denial is not a fallback"
    );
    assert_eq!(dev_b.reads_submitted(), links, "pure demand reads");

    for d in &hdevs {
        d.release_reads(1_000_000);
    }
    for (c, h) in hs {
        let v = h.join().unwrap().unwrap();
        assert_eq!(v, big_value(30 + c as u8, links));
    }
    assert_eq!(cache.chain_prefetch_permits(), permits_full);
}

/// Prediction across member boundaries: one predicted id sequence whose
/// consecutive positions resolve into
/// **different members** (alternating data/WAL devices) through the
/// authoritative locator. The driver reads each speculative link from its own
/// predicted member's device — no offset or member contiguity is assumed
/// anywhere — and consumes the whole chain with zero fallbacks.
#[test]
fn cross_member_chain_prefetches() {
    let links = 12;
    let dev_data = Arc::new(FakeIo::new(0));
    let dev_wal = Arc::new(FakeIo::new(0));
    let ids: Vec<u64> = (300..300 + links as u64).collect();
    let value = big_value(50, links);

    // Encode the chain, then land link i on the data member (even i) or the
    // WAL member (odd i), each at slot-sequential offsets on its device.
    let chunks: Vec<&[u8]> = overflow_chunks(&value).collect();
    assert_eq!(chunks.len(), links);
    let mut pages: Vec<Page> = Vec::with_capacity(links);
    let mut next: Option<PageRef> = None;
    for (i, chunk) in chunks.iter().enumerate().rev() {
        let page = encode_overflow(PageId(ids[i]), 0, chunk, next);
        next = Some(page.page_ref());
        pages.push(page);
    }
    pages.reverse();
    struct SplitLocator(std::collections::HashMap<u64, Location>);
    impl PageLocator for SplitLocator {
        fn locate(&self, page_ref: PageRef) -> Option<Location> {
            self.0.get(&page_ref.page_id.as_u64()).copied()
        }
        fn predict_overflow_location(&self, page_id: PageId) -> Option<Location> {
            self.0.get(&page_id.as_u64()).copied()
        }
    }
    let mut table = std::collections::HashMap::new();
    for (i, page) in pages.iter().enumerate() {
        let member = if i % 2 == 0 {
            Member::Data
        } else {
            Member::Wal
        };
        let dev = if i % 2 == 0 { &dev_data } else { &dev_wal };
        let offset = i as u64 * PAGE_SIZE as u64;
        table.insert(
            ids[i],
            Location {
                member,
                offset,
                length: PAGE_SIZE as u32,
            },
        );
        block_on(dev.write_all_at(offset, page.bytes().to_vec().into_boxed_slice()))
            .result
            .unwrap();
    }
    let file = FileGeneration::wal(
        Arc::clone(&dev_data) as Arc<dyn PageIo>,
        Arc::clone(&dev_wal) as Arc<dyn PageIo>,
        Arc::new(SplitLocator(table)),
    );
    let cache = PageCache::new();
    cache.set_chain_prefetch(true);
    dev_data.reset_read_stats();
    dev_wal.reset_read_stats();

    let got = block_on(read_overflow(
        &cache,
        &file,
        OverflowRef {
            head: next.unwrap(),
            total_len: value.len() as u64,
        },
    ))
    .unwrap();
    assert_eq!(got, value);
    let s = cache.chain_prefetch_stats();
    assert_eq!(s.engaged_chains, 1);
    assert_eq!(
        s.consumed,
        links as u64 - 1,
        "every predicted link consumed"
    );
    assert_eq!((s.unused, s.redundant, s.fallbacks), (0, 0, 0));
    // Each member's device served exactly its half: 6 even positions on data
    // (the head's demand read + 5 speculative), 6 odd on WAL (all speculative).
    assert_eq!(dev_data.reads_submitted(), links / 2);
    assert_eq!(dev_wal.reads_submitted(), links / 2);
    assert_eq!(
        (dev_data.reads_in_flight(), dev_wal.reads_in_flight()),
        (0, 0)
    );
}

// ---------------------------------------------------------------------------
// End-to-end store layouts (Direct, WAL overlay, WAL post-checkpoint).
// ---------------------------------------------------------------------------

/// StoreDirect end-to-end: the production `DirectLocator` declares the
/// capability; a cold reopen read engages and consumes the whole chain.
#[test]
fn direct_store_chain_prefetches_after_reopen() {
    use crate::batch::WriteBatch;
    use crate::store::{Options, Store};
    let data = Arc::new(FakeIo::new(0));
    {
        let store = Store::<crate::Direct>::create_with_io(
            Arc::clone(&data) as Arc<dyn PageIo>,
            Options::default(),
        )
        .unwrap();
        let map = crate::map::BTreeMap::over(store);
        block_on(map.apply(WriteBatch::new().insert(b"big".to_vec(), big_value(23, 12)))).unwrap();
        block_on(map.close()).unwrap();
    }
    let store =
        Store::<crate::Direct>::open_over(Arc::clone(&data) as Arc<dyn PageIo>, Options::default())
            .unwrap();
    let map = crate::map::BTreeMap::over(store);
    map.set_chain_prefetch(true);
    let got = block_on(map.get(b"big".to_vec())).unwrap().unwrap();
    assert_eq!(got, big_value(23, 12));
    let s = map.chain_prefetch_stats();
    assert_eq!(s.engaged_chains, 1);
    assert_eq!(s.consumed, 11, "all predicted links consumed");
    assert_eq!((s.unused, s.fallbacks), (0, 0));
}

/// StoreWal, uncheckpointed reopen: chain pages resolve into WAL records
/// (member=Wal) through the overlay. Prediction works there too — the
/// experiment declares the capability for both members; shipping WAL
/// default-on has its own gate (see `wal.rs` capability doc).
#[test]
fn wal_overlay_chain_prefetches_after_reopen() {
    use crate::batch::WriteBatch;
    use crate::store::{Options, Store};
    let data = Arc::new(FakeIo::new(0));
    let wal = Arc::new(FakeIo::new(0));
    {
        let store = Store::<crate::Wal>::create_with_io(
            Arc::clone(&data) as Arc<dyn PageIo>,
            Arc::clone(&wal) as Arc<dyn PageIo>,
            Options::default(),
        )
        .unwrap();
        let map = crate::map::BTreeMap::over(store);
        block_on(map.apply(WriteBatch::new().insert(b"big".to_vec(), big_value(21, 12)))).unwrap();
        block_on(map.close()).unwrap();
    }
    let store = Store::<crate::Wal>::open_over(
        Arc::clone(&data) as Arc<dyn PageIo>,
        Arc::clone(&wal) as Arc<dyn PageIo>,
        Options::default(),
    )
    .unwrap();
    let map = crate::map::BTreeMap::over(store);
    assert!(
        map.metrics().overlay_pages > 0,
        "chain must be WAL-resident"
    );
    map.set_chain_prefetch(true);
    let got = block_on(map.get(b"big".to_vec())).unwrap().unwrap();
    assert_eq!(got, big_value(21, 12));
    let s = map.chain_prefetch_stats();
    assert_eq!(s.engaged_chains, 1);
    assert_eq!(s.consumed, 11);
    assert_eq!((s.unused, s.fallbacks), (0, 0));
}

/// StoreWal, post-checkpoint reopen: chain pages were relocated into the data
/// file (offsets interleaved by live-page walk order — a layout
/// unpredictable by *offset* stride), the cache is cold, and the
/// chain still prefetches: id-stride plus the authoritative locator needs no
/// offset contiguity.
#[test]
fn wal_checkpoint_chain_prefetches_after_reopen() {
    use crate::batch::WriteBatch;
    use crate::fileset::{create_mem_store, open_mem_store, MemWalFileSet};
    use crate::store::Options;
    let data = Arc::new(FakeIo::new(0));
    let file_set = MemWalFileSet::new();
    {
        let store = create_mem_store(
            Arc::clone(&data) as Arc<dyn PageIo>,
            Arc::clone(&file_set),
            Options::default(),
        )
        .unwrap();
        let map = crate::map::BTreeMap::over(store);
        let mut batch = WriteBatch::new();
        for i in 0..8 {
            batch = batch.insert(key(i), format!("value-{i}").into_bytes());
        }
        batch = batch.insert(b"zzbig".to_vec(), big_value(22, 12));
        block_on(map.apply(batch)).unwrap();
        block_on(map.store().checkpoint()).unwrap();
        block_on(map.close()).unwrap();
    }
    // Reopen over the durable images (clean shutdown): a fresh cold cache.
    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let fresh = MemWalFileSet::new();
    for seq in file_set.live_segments() {
        let seg = file_set.segment(seq).unwrap();
        fresh.insert(seq, Arc::new(FakeIo::from_vec(seg.synced_bytes())));
    }
    let store =
        open_mem_store(Arc::clone(&d) as Arc<dyn PageIo>, fresh, Options::default()).unwrap();
    let map = crate::map::BTreeMap::over(store);
    assert_eq!(
        map.metrics().overlay_pages,
        0,
        "checkpoint must have landed"
    );
    map.set_chain_prefetch(true);
    let got = block_on(map.get(b"zzbig".to_vec())).unwrap().unwrap();
    assert_eq!(got, big_value(22, 12));
    let s = map.chain_prefetch_stats();
    assert_eq!(s.engaged_chains, 1, "post-checkpoint chain must engage");
    assert_eq!(s.consumed, 11);
    assert_eq!((s.unused, s.fallbacks), (0, 0));
}
