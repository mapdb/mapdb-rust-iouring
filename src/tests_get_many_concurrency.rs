//! Acceptance for the concurrent grouped `get_many` driver.
//!
//! The oracle throughout is the FakeIo read gate (`arm_read_gate` /
//! `release_reads` / `parked_read_count`): a test parks the traversal's physical
//! reads at the device, observes the fan-out depth while frozen, then releases.
//! A peak counter alone would prove nothing, since inline-resolving reads make
//! any driver — serial or concurrent — show a peak of 1. Expected peak at a
//! level is the number of distinct cold child pages at that level: not
//! `min(keys, siblings)`, because keys share children, and not the key count,
//! because the cache single-flights duplicates. Error selection is pinned too:
//! the reported failure is the lowest input position, independent of completion
//! order and of sort order.
//!
//! Every test runs on [`TreeReader`], which has no writer thread, so nothing
//! exists to backstop-sweep an abandoned cache load; `reads_in_flight() == 0`
//! after an errored call is therefore hard evidence that the driver drained
//! every started child rather than dropping one mid-flight.

use crate::io::{FakeIo, Faults, PageIo};
use crate::page::{overflow_chunks, PAGE_SIZE};
use crate::read::{TreeBuilder, TreeReader};
use futures_lite::future::block_on;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// `key{i:06}` / `value-{i}`, the shape the whole suite uses.
fn entries(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..n)
        .map(|i| {
            (
                format!("key{i:06}").into_bytes(),
                format!("value-{i}").into_bytes(),
            )
        })
        .collect()
}

fn key(i: usize) -> Vec<u8> {
    format!("key{i:06}").into_bytes()
}

fn val(i: usize) -> Vec<u8> {
    format!("value-{i}").into_bytes()
}

/// A 2-level tree: `n` entries in leaves of 4, one root branch. With inline-only
/// values the builder assigns leaf `j` (keys `4j..4j+4`) the device offset
/// `j * PAGE_SIZE`, which the fault-injection tests below rely on.
fn two_level(dev: Arc<FakeIo>, n: usize) -> TreeReader {
    let built = TreeBuilder {
        max_fanout: 100,
        max_leaf_entries: 4,
        ..Default::default()
    }
    .build(dev, &entries(n))
    .unwrap();
    TreeReader::new(built)
}

/// Per-step deadline for every gated wait and join below. A concurrency
/// regression must fail the test, never wedge the suite.
const STEP_TIMEOUT: Duration = Duration::from_secs(10);

/// A gated `get_many_grouped` call running on a scratch thread, with **bounded**
/// waits and joins: any deadline failure
/// releases every remaining gate slot and gives the scratch thread a bounded
/// window to finish; if it finishes it is joined, and if it is still wedged
/// after the window it is deliberately **leaked** before the panic — a raw
/// `join()` on a wedged thread would turn the regression into a suite hang,
/// which is the exact failure mode this harness exists to prevent. Every path
/// out of the harness is bounded.
type GetManyResult = Result<Vec<Option<Vec<u8>>>, crate::error::ReadError>;

struct GatedRun {
    dev: Arc<FakeIo>,
    h: Option<std::thread::JoinHandle<GetManyResult>>,
}

impl GatedRun {
    fn spawn(dev: &Arc<FakeIo>, reader: &Arc<TreeReader>, query: &[Vec<u8>]) -> Self {
        let reader = Arc::clone(reader);
        let query = query.to_vec();
        GatedRun {
            dev: Arc::clone(dev),
            h: Some(std::thread::spawn(move || {
                block_on(reader.get_many_grouped(query))
            })),
        }
    }

    /// Spins until `cond` holds; on deadline, bounded cleanup then panic.
    fn wait_until(&mut self, what: &str, cond: impl Fn() -> bool) {
        let deadline = Instant::now() + STEP_TIMEOUT;
        while !cond() {
            if Instant::now() >= deadline {
                self.cleanup();
                panic!("timeout waiting for {what}");
            }
            std::thread::yield_now();
        }
    }

    /// The bounded final join: the call must already be unblocked (every slot
    /// the test will ever need released); a wedge here is a driver bug and
    /// fails after releasing whatever remains.
    fn finish(mut self) -> GetManyResult {
        let h = self.h.take().expect("finish is called once");
        let deadline = Instant::now() + STEP_TIMEOUT;
        while !h.is_finished() {
            if Instant::now() >= deadline {
                self.dev.release_reads(1_000_000);
                let grace = Instant::now() + STEP_TIMEOUT;
                while !h.is_finished() && Instant::now() < grace {
                    std::thread::yield_now();
                }
                assert!(
                    h.is_finished(),
                    "gated run wedged even after releasing every slot"
                );
                break;
            }
            std::thread::yield_now();
        }
        h.join().expect("scratch thread panicked")
    }

    /// Failure-path cleanup: release everything, join bounded.
    fn cleanup(&mut self) {
        self.dev.release_reads(1_000_000);
        if let Some(h) = self.h.take() {
            let deadline = Instant::now() + STEP_TIMEOUT;
            while !h.is_finished() && Instant::now() < deadline {
                std::thread::yield_now();
            }
            if h.is_finished() {
                let _ = h.join();
            }
            // Otherwise: leak the thread — a panic with context beats a hang.
        }
    }
}

/// Distinct cold children drive the fan-out: a batch touching 4 distinct leaves
/// (two keys sharing one leaf, plus a literal duplicate key) parks exactly 4
/// concurrent leaf reads — not 6 (the key count), not 1 (the old serial form).
#[test]
fn peak_reads_equal_distinct_cold_children() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = Arc::new(two_level(dev.clone(), 32));

    // Scrambled input order; leaves touched: {7, 2, 0, 5}; key 29 duplicated;
    // keys 8+9 share leaf 2.
    let query = vec![key(29), key(8), key(9), key(0), key(29), key(21)];
    dev.reset_read_stats();
    dev.arm_read_gate(1 + 4);
    let mut run = GatedRun::spawn(&dev, &reader, &query);

    // Level 0: the root read parks alone.
    run.wait_until("root read parked", || dev.parked_read_count() == 1);
    assert_eq!(dev.reads_submitted(), 1);
    dev.release_reads(1);

    // Level 1: all four distinct leaf reads must be in flight simultaneously.
    run.wait_until("4 leaf reads parked", || dev.parked_read_count() == 4);
    assert_eq!(
        dev.reads_submitted(),
        5,
        "root + one read per distinct leaf"
    );
    assert_eq!(dev.reads_in_flight(), 4);
    assert_eq!(dev.reads_peak(), 4, "peak = distinct cold children");
    dev.release_reads(4);

    let got = run.finish().unwrap();
    let want = vec![
        Some(val(29)),
        Some(val(8)),
        Some(val(9)),
        Some(val(0)),
        Some(val(29)),
        Some(val(21)),
    ];
    assert_eq!(got, want, "input order and duplicate keys preserved");
    assert_eq!(dev.reads_in_flight(), 0);
}

/// A 3-level tree shows the per-level plateaus: 4 distinct branches park
/// concurrently, then their 4 distinct leaves.
#[test]
fn multilevel_fanout_plateaus_per_level() {
    let dev = Arc::new(FakeIo::new(0));
    let built = TreeBuilder {
        max_fanout: 4,
        max_leaf_entries: 4,
        ..Default::default()
    }
    .build(dev.clone(), &entries(64))
    .unwrap();
    let reader = Arc::new(TreeReader::new(built));

    // 16 leaves under 4 branches; one key under each branch, each in a
    // different leaf: branch j covers keys 16j..16j+16.
    let query = vec![key(0), key(17), key(34), key(51)];
    dev.reset_read_stats();
    dev.arm_read_gate(1 + 4 + 4);
    let mut run = GatedRun::spawn(&dev, &reader, &query);

    run.wait_until("root parked", || dev.parked_read_count() == 1);
    dev.release_reads(1);
    run.wait_until("4 branch reads parked", || dev.parked_read_count() == 4);
    assert_eq!(dev.reads_submitted(), 5, "root + 4 distinct branches");
    dev.release_reads(4);
    run.wait_until("4 leaf reads parked", || {
        dev.parked_read_count() == 4 && dev.reads_submitted() == 9
    });
    dev.release_reads(4);

    let got = run.finish().unwrap();
    assert_eq!(
        got,
        vec![Some(val(0)), Some(val(17)), Some(val(34)), Some(val(51))]
    );
    assert_eq!(dev.reads_peak(), 4);
}

/// Clustered keys collapse to one shared child load: the whole batch parks a
/// single leaf read.
#[test]
fn clustered_keys_share_one_child_load() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = Arc::new(two_level(dev.clone(), 32));

    // All in leaf 3 (keys 12..16), plus an absent key routing into the same leaf.
    let query = vec![key(12), key(15), b"key000013a".to_vec(), key(14)];
    dev.reset_read_stats();
    dev.arm_read_gate(1 + 1);
    let mut run = GatedRun::spawn(&dev, &reader, &query);

    run.wait_until("root parked", || dev.parked_read_count() == 1);
    dev.release_reads(1);
    run.wait_until("single leaf read parked", || {
        dev.parked_read_count() == 1 && dev.reads_submitted() == 2
    });
    dev.release_reads(1);

    let got = run.finish().unwrap();
    assert_eq!(got, vec![Some(val(12)), Some(val(15)), None, Some(val(14))]);
    assert_eq!(dev.reads_peak(), 1, "one distinct child page → one read");
}

/// Warm pages join no fan-out: only the cold leaves issue (and park) reads, and
/// results still assemble correctly across hit, miss, and absent keys.
#[test]
fn mixed_hit_miss_parks_only_cold_children() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = Arc::new(two_level(dev.clone(), 32));

    // Warm the root + leaf 2 (keys 8..12).
    block_on(reader.get(key(8))).unwrap().unwrap();
    dev.reset_read_stats();

    // leaf2 warm (present + absent probes), leaves 0 and 7 cold.
    let query = vec![
        key(9),                 // leaf 2, warm
        key(0),                 // leaf 0, cold
        key(30),                // leaf 7, cold
        b"key000009a".to_vec(), // absent, routes to warm leaf 2
        b"zzz".to_vec(),        // absent, routes to cold leaf 7
    ];
    dev.arm_read_gate(2);
    let mut run = GatedRun::spawn(&dev, &reader, &query);

    run.wait_until("2 cold leaf reads parked", || dev.parked_read_count() == 2);
    assert_eq!(
        dev.reads_submitted(),
        2,
        "warm root and warm leaf issue no reads; cold leaves issue one each"
    );
    dev.release_reads(2);

    let got = run.finish().unwrap();
    assert_eq!(
        got,
        vec![Some(val(9)), Some(val(0)), Some(val(30)), None, None]
    );
    assert_eq!(dev.reads_peak(), 2);
}

/// The error contract, half 1: the winner is the error at the **lowest sorted
/// position**, not the first to complete. Leaf 1's image is garbled (→ `Corrupt`)
/// and parked at the gate so it completes LAST; leaf 5 fails with EIO (→ `Io`)
/// long before. The call must still return `Corrupt` — and, with an `Ok` sibling
/// (leaf 7) in the group, keep driving everything to completion first
/// (`reads_in_flight == 0` on return; no writer exists here to clean up).
#[test]
fn error_winner_is_lowest_position_not_first_completed_corrupt() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = Arc::new(two_level(dev.clone(), 32));

    // Garble leaf 1's page image on the device (offset 1 * PAGE_SIZE).
    block_on(async {
        dev.write_all_at(PAGE_SIZE as u64, vec![0xEE; PAGE_SIZE].into_boxed_slice())
            .await
            .result
            .unwrap();
    });
    // EIO leaf 5's read at completion.
    dev.set_faults(Faults {
        fail_read_offsets: vec![5 * PAGE_SIZE as u64],
        ..Default::default()
    });

    // Sorted positions: leaf1 (pos 0) < leaf5 (pos 1) < leaf7 (pos 2, healthy).
    let query = vec![key(4), key(20), key(28)];
    dev.reset_read_stats();
    // Gate only the root and the FIRST leaf submission (= leaf 1, the lowest
    // sorted run): its siblings complete unclaimed, so the Io error and the Ok
    // sibling resolve while the Corrupt child is still parked.
    dev.arm_read_gate(2);
    let mut run = GatedRun::spawn(&dev, &reader, &query);

    run.wait_until("root parked", || dev.parked_read_count() == 1);
    dev.release_reads(1);
    // All three leaf reads submitted; leaf1 parked; leaf5 already failed with
    // Io; leaf7 already Ok — and the driver is still driving leaf1 (drain).
    run.wait_until("leaf1 parked after siblings resolved", || {
        dev.reads_submitted() == 4 && dev.parked_read_count() == 1 && dev.reads_in_flight() == 1
    });
    dev.release_reads(1);

    let err = run.finish().unwrap_err();
    assert!(
        matches!(err, crate::error::ReadError::Corrupt(_)),
        "lowest-position error (Corrupt) must win over the earlier-completed Io, got {err:?}"
    );
    assert_eq!(
        dev.reads_in_flight(),
        0,
        "every started child was drained to completion"
    );
    assert_eq!(dev.parked_read_count(), 0);
}

/// The error contract, half 2 (the converse arrangement): now the EIO child is
/// the lowest sorted position and completes last; the garbled child's `Corrupt`
/// lands first. The call must return `Io`.
#[test]
fn error_winner_is_lowest_position_not_first_completed_io() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = Arc::new(two_level(dev.clone(), 32));

    block_on(async {
        dev.write_all_at(
            5 * PAGE_SIZE as u64,
            vec![0xEE; PAGE_SIZE].into_boxed_slice(),
        )
        .await
        .result
        .unwrap();
    });
    dev.set_faults(Faults {
        fail_read_offsets: vec![PAGE_SIZE as u64],
        ..Default::default()
    });

    let query = vec![key(4), key(20), key(28)];
    dev.reset_read_stats();
    dev.arm_read_gate(2);
    let mut run = GatedRun::spawn(&dev, &reader, &query);

    run.wait_until("root parked", || dev.parked_read_count() == 1);
    dev.release_reads(1);
    run.wait_until("leaf1 parked after siblings resolved", || {
        dev.reads_submitted() == 4 && dev.parked_read_count() == 1 && dev.reads_in_flight() == 1
    });
    dev.release_reads(1);

    let err = run.finish().unwrap_err();
    assert!(
        matches!(err, crate::error::ReadError::Io(_)),
        "lowest-position error (Io) must win over the earlier-completed Corrupt, got {err:?}"
    );
    assert_eq!(dev.reads_in_flight(), 0);
}

/// The error contract, half 3: both arrangements again, but with the OTHER
/// completion order (the failing lowest-rank child completes FIRST via plain
/// prefix release). Together with halves 1–2, each ranking arrangement is
/// exercised under both completion orders.
#[test]
fn error_winner_stable_when_lowest_position_completes_first() {
    for (garble_leaf, eio_leaf, expect_corrupt) in [(1u64, 5u64, true), (5, 1, false)] {
        let dev = Arc::new(FakeIo::new(0));
        let reader = Arc::new(two_level(dev.clone(), 32));
        block_on(async {
            dev.write_all_at(
                garble_leaf * PAGE_SIZE as u64,
                vec![0xEE; PAGE_SIZE].into_boxed_slice(),
            )
            .await
            .result
            .unwrap();
        });
        dev.set_faults(Faults {
            fail_read_offsets: vec![eio_leaf * PAGE_SIZE as u64],
            ..Default::default()
        });

        let query = vec![key(4), key(20), key(28)];
        dev.reset_read_stats();
        // Full gate + prefix release: completions land in submission (sorted)
        // order, i.e. the lowest-rank child (leaf 1) completes FIRST here.
        dev.arm_read_gate(1 + 3);
        let mut run = GatedRun::spawn(&dev, &reader, &query);
        run.wait_until("root parked", || dev.parked_read_count() == 1);
        dev.release_reads(1);
        run.wait_until("3 leaf reads parked", || dev.parked_read_count() == 3);
        dev.release_reads(3);

        let err = run.finish().unwrap_err();
        if expect_corrupt {
            assert!(
                matches!(err, crate::error::ReadError::Corrupt(_)),
                "rank-0 Corrupt must win in this completion order too, got {err:?}"
            );
        } else {
            assert!(
                matches!(err, crate::error::ReadError::Io(_)),
                "rank-0 Io must win in this completion order too, got {err:?}"
            );
        }
        assert_eq!(dev.reads_in_flight(), 0);
    }
}

/// The winner's rank is the smallest **original input** index, not the smallest
/// sorted position. The input is scrambled so the two
/// rules disagree: key 20 (input position 0, EIO leaf) sorts AFTER key 4 (input
/// position 1, garbled leaf). Sorted-position ranking would return `Corrupt`;
/// the contract requires input-position ranking → `Io`.
#[test]
fn error_rank_follows_original_input_order_not_sort_order() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = Arc::new(two_level(dev.clone(), 32));
    block_on(async {
        dev.write_all_at(PAGE_SIZE as u64, vec![0xEE; PAGE_SIZE].into_boxed_slice())
            .await
            .result
            .unwrap();
    });
    dev.set_faults(Faults {
        fail_read_offsets: vec![5 * PAGE_SIZE as u64],
        ..Default::default()
    });

    // Input positions: key20 → 0 (leaf 5, Io), key4 → 1 (leaf 1, Corrupt).
    let query = vec![key(20), key(4), key(28)];
    let err = block_on(reader.get_many_grouped(query)).unwrap_err();
    assert!(
        matches!(err, crate::error::ReadError::Io(_)),
        "the error covering input position 0 must win regardless of sort order, got {err:?}"
    );
    assert_eq!(dev.reads_in_flight(), 0);
}

/// Overflow chains advance **concurrently across result keys** (serially within
/// each chain — every link names its successor): three distinct chains park 3
/// chain reads at every chain step — and a duplicate key naming one of those
/// chains adds a fourth result WITHOUT adding physical reads (per-link
/// single-flight), so the plateau stays 3.
#[test]
fn overflow_chains_advance_concurrently_across_keys() {
    let mut data = entries(12);
    // ~40 KiB values → multi-link chains for keys 3, 6, 9 (leaves 0, 1, 2).
    let big = |seed: u8| -> Vec<u8> {
        (0..40_000u32)
            .map(|i| seed.wrapping_add((i % 251) as u8))
            .collect()
    };
    data[3].1 = big(1);
    data[6].1 = big(2);
    data[9].1 = big(3);
    let links = overflow_chunks(&data[3].1).count();
    assert!(links >= 2, "test needs multi-link chains, got {links}");

    let dev = Arc::new(FakeIo::new(0));
    let built = TreeBuilder {
        max_fanout: 100,
        max_leaf_entries: 4,
        ..Default::default()
    }
    .build(dev.clone(), &data)
    .unwrap();
    let reader = Arc::new(TreeReader::new(built));

    // key 3 appears twice: its chain must load once per link, not twice.
    let query = vec![key(3), key(6), key(9), key(3)];
    dev.reset_read_stats();
    dev.arm_read_gate(1 + 3 + 3 * links);
    let mut run = GatedRun::spawn(&dev, &reader, &query);

    run.wait_until("root parked", || dev.parked_read_count() == 1);
    dev.release_reads(1);
    run.wait_until("3 leaf reads parked", || dev.parked_read_count() == 3);
    assert_eq!(dev.reads_submitted(), 4, "duplicate key adds no leaf read");
    dev.release_reads(3);

    // Each chain step: all three chains' link reads are in flight together —
    // exactly 3, never 4 (the duplicate joins per-link single-flight) and never
    // more than one link per chain (a link names its successor).
    for step in 0..links {
        run.wait_until("3 chain link reads parked", || {
            dev.parked_read_count() == 3 && dev.reads_submitted() == 4 + 3 * (step + 1)
        });
        dev.release_reads(3);
    }

    let got = run.finish().unwrap();
    assert_eq!(
        got,
        vec![
            Some(data[3].1.clone()),
            Some(data[6].1.clone()),
            Some(data[9].1.clone()),
            Some(data[3].1.clone())
        ],
        "reassembled overflow values must round-trip (duplicate included)"
    );
    assert_eq!(dev.reads_in_flight(), 0);
    assert_eq!(
        dev.reads_submitted(),
        4 + 3 * links,
        "total physical reads: root + 3 leaves + one per link per distinct chain"
    );
}

/// The production route (T2.1 threshold policy): a `BTreeMap` batch
/// at/above [`GROUPED_ROUTE_MIN_BATCH`](crate::map::GROUPED_ROUTE_MIN_BATCH)
/// takes the concurrent grouped traversal, below it the sequential one, and the
/// explicit `get_many_grouped` opt-in exists at any size — all three agree with
/// each other and with ground truth on a live store.
#[test]
fn btreemap_threshold_route_and_opt_in_agree() {
    use crate::admission::AdmissionLimits;
    use crate::backend::Direct;
    use crate::batch::WriteBatch;
    use crate::map::{BTreeMap, GROUPED_ROUTE_MIN_BATCH};
    use crate::store::{Options, Store};

    let store = Store::<Direct>::create_with_io(
        Arc::new(FakeIo::new(0)),
        Options {
            limits: AdmissionLimits {
                max_dirty_pages: 4_000_000,
                max_ops_per_batch: 4096,
                ..Default::default()
            },
            ..Options::default()
        },
    )
    .unwrap();
    let map = BTreeMap::over(store);
    let n = GROUPED_ROUTE_MIN_BATCH + 100;
    block_on(async {
        for chunk in (0..n).collect::<Vec<_>>().chunks(1024) {
            let mut b = WriteBatch::new();
            for &i in chunk {
                b = b.insert(key(i), val(i));
            }
            map.apply(b).await.unwrap();
        }

        // Reversed order + an absent probe; length crosses the threshold.
        let mut query: Vec<Vec<u8>> = (0..n).rev().map(key).collect();
        query.push(b"absent".to_vec());
        let mut want: Vec<Option<Vec<u8>>> = (0..n).rev().map(|i| Some(val(i))).collect();
        want.push(None);

        let routed = map.get_many(query.clone()).await.unwrap();
        assert_eq!(routed, want, "above-threshold route (grouped)");
        let opt_in = map.get_many_grouped(query.clone()).await.unwrap();
        assert_eq!(opt_in, want, "explicit grouped opt-in");
        let below = map.get_many(query[..100].to_vec()).await.unwrap();
        assert_eq!(below, want[..100], "below-threshold route (sequential)");
        map.close().await.unwrap();
    });
}

/// Ungated agreement over a value-size mix: naive and concurrent-grouped answer
/// identically (and match ground truth) across inline values, multi-link
/// overflow values, duplicates, and absent keys.
#[test]
fn naive_and_concurrent_grouped_agree_with_overflow_values() {
    let mut data = entries(100);
    for i in (0..100).step_by(7) {
        data[i].1 = (0..30_000u32)
            .map(|j| (i as u8).wrapping_add((j % 253) as u8))
            .collect();
    }
    let truth: std::collections::BTreeMap<Vec<u8>, Vec<u8>> = data.iter().cloned().collect();
    let dev = Arc::new(FakeIo::new(0));
    let built = TreeBuilder {
        max_fanout: 4,
        max_leaf_entries: 4,
        ..Default::default()
    }
    .build(dev, &data)
    .unwrap();
    let reader = TreeReader::new(built);

    let mut query: Vec<Vec<u8>> = Vec::new();
    for i in (0..100).rev().step_by(3) {
        query.push(key(i));
    }
    query.push(key(7)); // duplicate of a big value
    query.push(key(7));
    query.push(b"absent-low".to_vec());
    query.push(b"zzz-absent-high".to_vec());

    block_on(async {
        let naive = reader.get_many(query.clone()).await.unwrap();
        let grouped = reader.get_many_grouped(query.clone()).await.unwrap();
        assert_eq!(naive, grouped, "naive and concurrent grouped must agree");
        let want: Vec<Option<Vec<u8>>> = query.iter().map(|k| truth.get(k).cloned()).collect();
        assert_eq!(grouped, want);
    });
}
