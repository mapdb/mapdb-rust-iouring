//! Acceptance for the sorted-map layer: consumer-callback range scans in both
//! directions (`tree_scan_range` / `for_each_range` / `for_each_range_rev`),
//! edge lookups (`first`/`last`), navigation (floor/ceiling/lower/higher), and
//! the [`SortedMap`] abstraction — put, remove, compare-and-set, prefix and
//! bounded scans, pagination, and snapshot isolation.
//!
//! The scan oracle throughout is `std::collections::BTreeMap`: every range shape
//! is checked against the standard map's `range` over the same entries, so bound
//! semantics (included, excluded, unbounded, empty, inverted) are defined by the
//! standard library rather than by this suite. Beyond the happy path it covers
//! structural edges (single-leaf and minimum-fanout tall trees, binary key
//! alphabets, a cache capacity of one), overflow values in scans and at the
//! edges, and adverse paths: an I/O or corruption fault mid-scan must deliver the
//! intact prefix then error, a range skipping a broken leaf must never touch it,
//! and a dropped in-flight scan must leave the reader reusable.

use crate::io::FakeIo;
use crate::page::MAX_INLINE_VALUE_LEN;
use crate::read::{TreeBuilder, TreeReader};
use crate::sorted::SortedMap;
use crate::store::{Options, Store};
use futures_lite::future::block_on;
use std::collections::BTreeMap as StdMap;
use std::ops::Bound;
use std::ops::ControlFlow;
use std::sync::Arc;

fn key(i: usize) -> Vec<u8> {
    format!("key{i:06}").into_bytes()
}

fn val(i: usize) -> Vec<u8> {
    format!("value-{i}").into_bytes()
}

fn entries(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..n).map(|i| (key(i), val(i))).collect()
}

/// A three-level tree (fanout 4, leaves of 4) so scans cross real branch
/// windows, not one root leaf.
fn deep_reader(n: usize) -> TreeReader {
    let built = TreeBuilder {
        max_fanout: 4,
        max_leaf_entries: 4,
        ..Default::default()
    }
    .build(Arc::new(FakeIo::new(0)), &entries(n))
    .unwrap();
    TreeReader::new(built)
}

/// Collects a reader scan into owned pairs.
fn scan<R: std::ops::RangeBounds<Vec<u8>>>(
    reader: &TreeReader,
    range: R,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    let summary = block_on(reader.for_each_range(range, |k, v| {
        out.push((k.to_vec(), v.to_vec()));
        ControlFlow::Continue(())
    }))
    .unwrap();
    assert!(summary.completed);
    assert_eq!(summary.visited as usize, out.len());
    out
}

/// The same range evaluated by the standard map over the same entries.
fn oracle<R: std::ops::RangeBounds<Vec<u8>>>(n: usize, range: R) -> Vec<(Vec<u8>, Vec<u8>)> {
    let std: StdMap<Vec<u8>, Vec<u8>> = entries(n).into_iter().collect();
    std.range(range)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Reader-level scan semantics (TreeBuilder fixtures, no writer).
// ---------------------------------------------------------------------------

#[test]
fn full_scan_is_sorted_and_complete() {
    let reader = deep_reader(100);
    assert_eq!(scan(&reader, ..), oracle(100, ..));
}

#[test]
fn range_bounds_match_std_btreemap() {
    let n = 100;
    let reader = deep_reader(n);
    // Boundary keys sit on leaf edges (leaves of 4) and separator keys, plus
    // keys absent from the map (suffix pushes between entries).
    let probes: Vec<Vec<u8>> = vec![
        key(0),
        key(3),
        key(4),
        key(15),
        key(16),
        key(50),
        key(99),
        b"key000015zzz".to_vec(),
        b"a".to_vec(),
        b"z".to_vec(),
    ];
    for lo in &probes {
        for hi in &probes {
            type Bnd = Bound<Vec<u8>>;
            let cases: Vec<(Bnd, Bnd)> = vec![
                (Bound::Included(lo.clone()), Bound::Included(hi.clone())),
                (Bound::Included(lo.clone()), Bound::Excluded(hi.clone())),
                (Bound::Excluded(lo.clone()), Bound::Included(hi.clone())),
                (Bound::Excluded(lo.clone()), Bound::Excluded(hi.clone())),
                (Bound::Unbounded, Bound::Included(hi.clone())),
                (Bound::Included(lo.clone()), Bound::Unbounded),
            ];
            for (start, end) in cases {
                // std's `range` panics on an inverted range (start > end, or
                // start == end with both sides excluded); for exactly those
                // shapes the scan must simply return empty instead.
                let skey = match &start {
                    Bound::Included(k) | Bound::Excluded(k) => Some(k),
                    Bound::Unbounded => None,
                };
                let ekey = match &end {
                    Bound::Included(k) | Bound::Excluded(k) => Some(k),
                    Bound::Unbounded => None,
                };
                let std_refuses = match (skey, ekey) {
                    (Some(s), Some(e)) => {
                        s > e
                            || (s == e
                                && matches!(
                                    (&start, &end),
                                    (Bound::Excluded(_), Bound::Excluded(_))
                                ))
                    }
                    _ => false,
                };
                if std_refuses {
                    let got = scan(&reader, (start, end));
                    assert!(got.is_empty(), "inverted range must be empty");
                    continue;
                }
                let got = scan(&reader, (start.clone(), end.clone()));
                assert_eq!(got, oracle(n, (start, end)));
            }
        }
    }
}

#[test]
fn early_stop_reports_partial_scan() {
    let reader = deep_reader(50);
    let mut seen = 0u64;
    let summary = block_on(
        reader.for_each_range::<&[u8], _, _>(.., |_k: &[u8], _v: &[u8]| {
            seen += 1;
            if seen == 7 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }),
    )
    .unwrap();
    assert_eq!(seen, 7);
    assert_eq!(summary.visited, 7);
    assert!(!summary.completed);
}

#[test]
fn empty_tree_scans_empty() {
    let reader = deep_reader(0);
    assert_eq!(scan(&reader, ..), Vec::<(Vec<u8>, Vec<u8>)>::new());
}

#[test]
fn overflow_values_are_reassembled_in_scans_and_edges() {
    // Values above MAX_INLINE_VALUE_LEN spill to chains in the builder; the
    // scan must hand the consumer the reassembled bytes.
    let big = |i: usize| {
        let mut v = vec![b'a' + (i % 26) as u8; MAX_INLINE_VALUE_LEN * 3 + i];
        v.extend_from_slice(format!("-{i}").as_bytes());
        v
    };
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..6).map(|i| (key(i), big(i))).collect();
    let built = TreeBuilder {
        max_fanout: 4,
        max_leaf_entries: 2,
        ..Default::default()
    }
    .build(Arc::new(FakeIo::new(0)), &entries)
    .unwrap();
    let reader = TreeReader::new(built);
    let got = scan(&reader, ..);
    assert_eq!(got, entries);
}

// ---------------------------------------------------------------------------
// SortedMap over a real store (FakeIo-backed StoreDirect / StoreWal).
// ---------------------------------------------------------------------------

fn direct_sorted() -> SortedMap<crate::Direct> {
    let dev = Arc::new(FakeIo::new(0));
    let store = Store::<crate::Direct>::create_with_io(dev, Options::default()).unwrap();
    SortedMap::over(store)
}

#[test]
fn sorted_map_put_scan_remove() {
    let m = direct_sorted();
    block_on(async {
        m.put_all((0..40).map(|i| (key(i), val(i)))).await.unwrap();
        m.insert(key(7), b"replaced".to_vec()).await.unwrap();
        m.remove(key(8)).await.unwrap();

        let mut expect: StdMap<Vec<u8>, Vec<u8>> = (0..40).map(|i| (key(i), val(i))).collect();
        expect.insert(key(7), b"replaced".to_vec());
        expect.remove(&key(8));

        let mut got = Vec::new();
        let summary = m
            .for_each(|k, v| {
                got.push((k.to_vec(), v.to_vec()));
                ControlFlow::Continue(())
            })
            .await
            .unwrap();
        assert!(summary.completed);
        let want: Vec<_> = expect.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        assert_eq!(got, want);

        assert_eq!(m.count_range(key(10)..key(20)).await.unwrap(), 10);
        assert_eq!(m.first_entry().await.unwrap().unwrap(), (key(0), val(0)),);
        assert_eq!(m.last_entry().await.unwrap().unwrap(), (key(39), val(39)),);

        // Bounded collection: limit 0, partial, and beyond-the-end.
        assert!(m.collect_range::<&[u8], _>(.., 0).await.unwrap().is_empty());
        let three = m.collect_range(key(10)..key(20), 3).await.unwrap();
        assert_eq!(
            three,
            vec![(key(10), val(10)), (key(11), val(11)), (key(12), val(12))]
        );
        assert_eq!(m.collect_range(key(35).., 100).await.unwrap().len(), 5);

        m.close().await.unwrap();
    });
}

#[test]
fn sorted_map_prefix_scan() {
    let m = direct_sorted();
    block_on(async {
        m.put_all([
            (b"app".to_vec(), b"1".to_vec()),
            (b"apple".to_vec(), b"2".to_vec()),
            (b"apple-pie".to_vec(), b"3".to_vec()),
            (b"apricot".to_vec(), b"4".to_vec()),
            (b"banana".to_vec(), b"5".to_vec()),
            (b"\xff\xff".to_vec(), b"top".to_vec()),
            (b"\xff\xff\x01".to_vec(), b"top2".to_vec()),
        ])
        .await
        .unwrap();

        let collect_prefix = |prefix: Vec<u8>| {
            let m = m.clone();
            async move {
                let mut got = Vec::new();
                m.for_each_prefix(prefix, |k, _v| {
                    got.push(k.to_vec());
                    ControlFlow::Continue(())
                })
                .await
                .unwrap();
                got
            }
        };

        assert_eq!(
            collect_prefix(b"apple".to_vec()).await,
            vec![b"apple".to_vec(), b"apple-pie".to_vec()]
        );
        assert_eq!(
            collect_prefix(b"ap".to_vec()).await,
            vec![
                b"app".to_vec(),
                b"apple".to_vec(),
                b"apple-pie".to_vec(),
                b"apricot".to_vec()
            ]
        );
        assert_eq!(collect_prefix(b"c".to_vec()).await, Vec::<Vec<u8>>::new());
        // All-0xFF prefix: the range is unbounded above and must still stop at
        // the prefix's own extent — here it covers both 0xff-keys.
        assert_eq!(
            collect_prefix(b"\xff\xff".to_vec()).await,
            vec![b"\xff\xff".to_vec(), b"\xff\xff\x01".to_vec()]
        );
        m.close().await.unwrap();
    });
}

#[test]
fn sorted_map_snapshot_isolation() {
    let m = direct_sorted();
    block_on(async {
        m.put_all((0..10).map(|i| (key(i), val(i)))).await.unwrap();
        let snap = m.snapshot();

        m.insert(key(100), val(100)).await.unwrap();
        m.remove(key(0)).await.unwrap();

        // The snapshot scan sees the capture-time state…
        let mut snap_keys = Vec::new();
        snap.for_each_range::<&[u8], _, _>(.., |k: &[u8], _v: &[u8]| {
            snap_keys.push(k.to_vec());
            ControlFlow::Continue(())
        })
        .await
        .unwrap();
        assert_eq!(snap_keys, (0..10).map(key).collect::<Vec<_>>());
        assert_eq!(snap.first_entry().await.unwrap().unwrap().0, key(0));

        // …while a live scan sees the new version.
        assert_eq!(m.count_range::<&[u8], _>(..).await.unwrap(), 10);
        assert_eq!(m.first_entry().await.unwrap().unwrap().0, key(1));
        assert_eq!(m.last_entry().await.unwrap().unwrap().0, key(100));
        m.close().await.unwrap();
    });
}

#[test]
fn sorted_map_conditional_ops() {
    let m = direct_sorted();
    block_on(async {
        assert!(m
            .put_if_absent(b"k".to_vec(), b"v1".to_vec())
            .await
            .unwrap()
            .is_some());
        assert!(m
            .put_if_absent(b"k".to_vec(), b"v2".to_vec())
            .await
            .unwrap()
            .is_none());
        assert_eq!(m.get(b"k").await.unwrap().unwrap(), b"v1");

        // CAS success, CAS stale, CAS-to-remove.
        assert!(m
            .compare_and_set(b"k".to_vec(), Some(b"v1".to_vec()), Some(b"v3".to_vec()))
            .await
            .unwrap()
            .is_some());
        assert!(m
            .compare_and_set(b"k".to_vec(), Some(b"v1".to_vec()), Some(b"v4".to_vec()))
            .await
            .unwrap()
            .is_none());
        assert_eq!(m.get(b"k").await.unwrap().unwrap(), b"v3");
        assert!(m
            .compare_and_set(b"k".to_vec(), Some(b"v3".to_vec()), None)
            .await
            .unwrap()
            .is_some());
        assert!(!m.contains_key(b"k").await.unwrap());
        m.close().await.unwrap();
    });
}

#[test]
fn sorted_map_wal_commit_and_scan() {
    let data = Arc::new(FakeIo::new(0));
    let wal = Arc::new(FakeIo::new(0));
    let store = Store::<crate::Wal>::create_with_io(data, wal, Options::default()).unwrap();
    let m = SortedMap::over(store);
    block_on(async {
        let v = m.put_all((0..25).map(|i| (key(i), val(i)))).await.unwrap();
        m.commit(v).await.unwrap();
        assert_eq!(m.count_range::<&[u8], _>(..).await.unwrap(), 25);
        let two = m.collect_range(key(3)..=key(4), 10).await.unwrap();
        assert_eq!(two, vec![(key(3), val(3)), (key(4), val(4))]);
        m.close().await.unwrap();
    });
}

#[test]
fn sorted_map_scans_overflow_values_from_store() {
    let m = direct_sorted();
    block_on(async {
        let big = vec![b'x'; MAX_INLINE_VALUE_LEN * 2 + 17];
        m.insert(b"big".to_vec(), big.clone()).await.unwrap();
        m.insert(b"small".to_vec(), b"s".to_vec()).await.unwrap();
        let got = m.collect_range::<&[u8], _>(.., 10).await.unwrap();
        assert_eq!(
            got,
            vec![(b"big".to_vec(), big), (b"small".to_vec(), b"s".to_vec())]
        );
        m.close().await.unwrap();
    });
}

// ---------------------------------------------------------------------------
// Structural edges: tree shapes, key alphabets, bound geometry.
// ---------------------------------------------------------------------------

#[test]
fn single_root_leaf_tree_scans_and_bounds() {
    // Large fanout/leaf caps: 3 entries fit one root leaf — no branch at all.
    let built = TreeBuilder {
        max_fanout: 100,
        max_leaf_entries: 100,
        ..Default::default()
    }
    .build(Arc::new(FakeIo::new(0)), &entries(3))
    .unwrap();
    let reader = TreeReader::new(built);
    assert_eq!(scan(&reader, ..), oracle(3, ..));
    assert_eq!(scan(&reader, key(1)..key(2)), oracle(3, key(1)..key(2)));
    assert_eq!(scan(&reader, key(0)..=key(0)), oracle(3, key(0)..=key(0)));
}

#[test]
fn minimum_fanout_tall_tree_scans_exactly() {
    // Fanout 2 / leaves of 1: 64 entries force the tallest shape the builder
    // can produce — the path stack must handle every up-and-over transition.
    let built = TreeBuilder {
        max_fanout: 2,
        max_leaf_entries: 1,
        ..Default::default()
    }
    .build(Arc::new(FakeIo::new(0)), &entries(64))
    .unwrap();
    let reader = TreeReader::new(built);
    assert_eq!(scan(&reader, ..), oracle(64, ..));
    assert_eq!(
        scan(&reader, key(13)..key(51)),
        oracle(64, key(13)..key(51))
    );
    assert_eq!(
        scan(
            &reader,
            (Bound::Excluded(key(31)), Bound::Included(key(32)))
        ),
        oracle(64, (Bound::Excluded(key(31)), Bound::Included(key(32))))
    );
}

#[test]
fn binary_alphabet_keys_scan_in_bytewise_order() {
    // 0x00 / 0xFF keys of mixed length: bytewise order, not text order. The
    // shortest key sorts first among its extensions (b"\x00" < b"\x00\x00").
    let mut keys: Vec<Vec<u8>> = vec![
        vec![0x00],
        vec![0x00, 0x00],
        vec![0x00, 0xFF],
        vec![0x01],
        vec![0xFE, 0xFF],
        vec![0xFF],
        vec![0xFF, 0x00],
        vec![0xFF, 0xFF],
    ];
    keys.sort();
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = keys.iter().map(|k| (k.clone(), b"v".to_vec())).collect();
    let built = TreeBuilder {
        max_fanout: 3,
        max_leaf_entries: 2,
        ..Default::default()
    }
    .build(Arc::new(FakeIo::new(0)), &pairs)
    .unwrap();
    let reader = TreeReader::new(built);
    let got = scan(&reader, ..);
    assert_eq!(got, pairs);
    // A range pinched between two adjacent keys is empty.
    assert_eq!(
        scan(
            &reader,
            (
                Bound::Excluded(vec![0x00, 0x00]),
                Bound::Excluded(vec![0x00, 0xFF])
            )
        ),
        Vec::<(Vec<u8>, Vec<u8>)>::new()
    );
    // Included-both single-key pinch yields exactly that key.
    assert_eq!(
        scan(&reader, vec![0xFF]..=vec![0xFF]),
        vec![(vec![0xFF], b"v".to_vec())]
    );
}

#[test]
fn scan_after_clearing_every_entry_is_empty_then_repopulates() {
    let m = direct_sorted();
    block_on(async {
        m.put_all((0..12).map(|i| (key(i), val(i)))).await.unwrap();
        for i in 0..12 {
            m.remove(key(i)).await.unwrap();
        }
        assert_eq!(m.count_range::<&[u8], _>(..).await.unwrap(), 0);
        assert_eq!(m.first_entry().await.unwrap(), None);
        assert_eq!(m.last_entry().await.unwrap(), None);

        // Repopulate: the scan sees only the new generation of entries.
        m.insert(key(5), b"back".to_vec()).await.unwrap();
        let got = m.collect_range::<&[u8], _>(.., 10).await.unwrap();
        assert_eq!(got, vec![(key(5), b"back".to_vec())]);
        m.close().await.unwrap();
    });
}

#[test]
fn scan_is_exact_under_cache_capacity_one() {
    // A clean tier that can hold one page evicts constantly mid-scan; the
    // path stack's Arc pins keep the traversal exact regardless.
    let built = TreeBuilder {
        max_fanout: 4,
        max_leaf_entries: 4,
        ..Default::default()
    }
    .build(Arc::new(FakeIo::new(0)), &entries(80))
    .unwrap();
    let reader = TreeReader::with_cache(built, Arc::new(crate::cache::PageCache::with_capacity(1)));
    assert_eq!(scan(&reader, ..), oracle(80, ..));
    assert_eq!(
        scan(&reader, key(20)..key(60)),
        oracle(80, key(20)..key(60))
    );
}

#[test]
fn one_byte_minimum_keys_and_empty_key_rejection() {
    let m = direct_sorted();
    block_on(async {
        // The engine rejects empty keys outright (BatchError::EmptyKey);
        // b"\x00" is the smallest legal key and must sort first.
        let err = m.insert(Vec::new(), b"v".to_vec()).await.unwrap_err();
        assert!(
            matches!(
                err,
                crate::error::WriteError::InvalidBatch(crate::batch::BatchError::EmptyKey)
            ),
            "empty key must be rejected, got {err:?}"
        );
        m.insert(vec![0x00], b"zero".to_vec()).await.unwrap();
        m.insert(vec![0xFF], b"top".to_vec()).await.unwrap();
        m.insert(b"middle".to_vec(), b"m".to_vec()).await.unwrap();
        assert_eq!(m.first_entry().await.unwrap().unwrap().0, vec![0x00]);
        assert_eq!(m.last_entry().await.unwrap().unwrap().0, vec![0xFF]);
        // An unbounded-start range naturally includes the minimum key.
        let got = m.collect_range(..=vec![0x00], 10).await.unwrap();
        assert_eq!(got, vec![(vec![0x00], b"zero".to_vec())]);
        m.close().await.unwrap();
    });
}

// ---------------------------------------------------------------------------
// Overflow values: thresholds, early stop on a chain, chain edges.
// ---------------------------------------------------------------------------

#[test]
fn inline_overflow_threshold_boundary_values_scan_exactly() {
    let m = direct_sorted();
    block_on(async {
        let at = vec![b'a'; MAX_INLINE_VALUE_LEN]; // largest inline
        let over = vec![b'b'; MAX_INLINE_VALUE_LEN + 1]; // smallest chain
        m.insert(key(0), at.clone()).await.unwrap();
        m.insert(key(1), over.clone()).await.unwrap();
        let got = m.collect_range::<&[u8], _>(.., 10).await.unwrap();
        assert_eq!(got, vec![(key(0), at), (key(1), over)]);
        m.close().await.unwrap();
    });
}

#[test]
fn early_stop_on_an_overflow_entry_counts_it_as_visited() {
    let m = direct_sorted();
    block_on(async {
        let big = vec![b'x'; MAX_INLINE_VALUE_LEN * 2];
        m.insert(key(0), b"small".to_vec()).await.unwrap();
        m.insert(key(1), big.clone()).await.unwrap();
        m.insert(key(2), b"after".to_vec()).await.unwrap();
        let mut seen: Vec<(Vec<u8>, usize)> = Vec::new();
        let summary = m
            .for_each(|k, v| {
                seen.push((k.to_vec(), v.len()));
                if k == key(1).as_slice() {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .await
            .unwrap();
        // The chain was fully reassembled before delivery, the break landed on
        // it, and key(2) was never visited.
        assert_eq!(seen, vec![(key(0), 5), (key(1), big.len())]);
        assert_eq!(summary.visited, 2);
        assert!(!summary.completed);
        m.close().await.unwrap();
    });
}

#[test]
fn first_and_last_reassemble_overflow_edge_values() {
    let m = direct_sorted();
    block_on(async {
        let lo = vec![b'l'; MAX_INLINE_VALUE_LEN * 2 + 3];
        let hi = vec![b'h'; MAX_INLINE_VALUE_LEN * 3 + 1];
        m.insert(vec![0x01], lo.clone()).await.unwrap();
        m.insert(b"middle".to_vec(), b"m".to_vec()).await.unwrap();
        m.insert(vec![0xFE], hi.clone()).await.unwrap();
        assert_eq!(m.first_entry().await.unwrap().unwrap(), (vec![0x01], lo));
        assert_eq!(m.last_entry().await.unwrap().unwrap(), (vec![0xFE], hi));
        m.close().await.unwrap();
    });
}

// ---------------------------------------------------------------------------
// Faults, corruption, cancellation. TreeReader has no writer thread, so
// `reads_in_flight() == 0` after a failed call proves the scan abandoned
// nothing to a backstop that does not exist (same discipline as the T2.1
// suite).
// ---------------------------------------------------------------------------

use crate::error::ReadError;
use crate::io::{Faults, PageIo};
use crate::page::PAGE_SIZE;

/// The T2.1 layout trick: fanout 100 + leaves of 4 + inline values puts leaf
/// `j` (keys `4j..4j+4`) at device offset `j * PAGE_SIZE`, so fault knobs can
/// target an exact leaf.
fn flat_reader(dev: Arc<FakeIo>, n: usize) -> TreeReader {
    let built = TreeBuilder {
        max_fanout: 100,
        max_leaf_entries: 4,
        ..Default::default()
    }
    .build(dev, &entries(n))
    .unwrap();
    TreeReader::new(built)
}

#[test]
fn scan_surfaces_io_error_and_leaves_no_reads_in_flight() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = flat_reader(Arc::clone(&dev), 48);
    dev.set_faults(Faults {
        fail_reads_from: Some(0), // every read fails, starting with the root
        ..Faults::default()
    });
    let mut delivered = 0u64;
    let err = block_on(
        reader.for_each_range::<&[u8], _, _>(.., |_k: &[u8], _v: &[u8]| {
            delivered += 1;
            ControlFlow::Continue(())
        }),
    )
    .unwrap_err();
    assert!(matches!(err, ReadError::Io { .. }), "got {err:?}");
    assert_eq!(delivered, 0);
    assert_eq!(
        dev.reads_in_flight(),
        0,
        "no read left in flight after error"
    );
}

#[test]
fn mid_scan_leaf_failure_delivers_the_prefix_then_errors() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = flat_reader(Arc::clone(&dev), 48);
    // EIO exactly leaf 3's offset: leaves 0..=2 (entries 0..12) deliver, then
    // the scan fails — and never silently skips the broken leaf.
    dev.set_faults(Faults {
        fail_read_offsets: vec![3 * PAGE_SIZE as u64],
        ..Faults::default()
    });
    let mut got = Vec::new();
    let err = block_on(
        reader.for_each_range::<&[u8], _, _>(.., |k: &[u8], _v: &[u8]| {
            got.push(k.to_vec());
            ControlFlow::Continue(())
        }),
    )
    .unwrap_err();
    assert!(matches!(err, ReadError::Io { .. }), "got {err:?}");
    assert_eq!(got, (0..12).map(key).collect::<Vec<_>>());
    assert_eq!(dev.reads_in_flight(), 0);
}

/// The descending mirror of the mid-scan fault: leaves 11..=4 (entries
/// 47..16, descending) deliver, then leaf 3 EIOs — the broken leaf is never
/// silently skipped, and the delivered part is exactly the descending prefix.
#[test]
fn rev_mid_scan_leaf_failure_delivers_the_descending_prefix_then_errors() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = flat_reader(Arc::clone(&dev), 48);
    dev.set_faults(Faults {
        fail_read_offsets: vec![3 * PAGE_SIZE as u64],
        ..Faults::default()
    });
    let mut got = Vec::new();
    let err = block_on(
        reader.for_each_range_rev::<&[u8], _, _>(.., |k: &[u8], _v: &[u8]| {
            got.push(k.to_vec());
            ControlFlow::Continue(())
        }),
    )
    .unwrap_err();
    assert!(matches!(err, ReadError::Io { .. }), "got {err:?}");
    assert_eq!(got, (16..48).rev().map(key).collect::<Vec<_>>());
    assert_eq!(dev.reads_in_flight(), 0);
}

/// Early stop on the descending path never touches leaves beyond the break:
/// every leaf below the stop point is faulted, and the scan still succeeds.
#[test]
fn rev_early_stop_reads_no_leaf_beyond_the_break() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = flat_reader(Arc::clone(&dev), 48);
    // Leaves 0..=9 (entries 0..40) all EIO; only leaves 10 and 11 stay
    // readable. A descending scan that stops within leaf 10 must not care.
    dev.set_faults(Faults {
        fail_read_offsets: (0..10).map(|j| j * PAGE_SIZE as u64).collect(),
        ..Faults::default()
    });
    let mut got = Vec::new();
    let summary = block_on(
        reader.for_each_range_rev::<&[u8], _, _>(.., |k: &[u8], _v: &[u8]| {
            got.push(k.to_vec());
            if got.len() == 6 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }),
    )
    .unwrap();
    assert!(!summary.completed);
    assert_eq!(got, (42..48).rev().map(key).collect::<Vec<_>>());
}

#[test]
fn corrupt_leaf_fails_the_scan_after_the_intact_prefix() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = flat_reader(Arc::clone(&dev), 48);
    // Flip one byte inside leaf 5's image: checksum validation at decode must
    // reject it. Entries 0..20 (leaves 0..=4) deliver first.
    block_on(async {
        let mut bytes = dev.snapshot_bytes();
        bytes[5 * PAGE_SIZE + 200] ^= 0xFF;
        let page: Vec<u8> = bytes[5 * PAGE_SIZE..6 * PAGE_SIZE].to_vec();
        dev.write_all_at(5 * PAGE_SIZE as u64, page.into_boxed_slice())
            .await
            .result
            .unwrap();
    });
    let mut got = Vec::new();
    let err = block_on(
        reader.for_each_range::<&[u8], _, _>(.., |k: &[u8], _v: &[u8]| {
            got.push(k.to_vec());
            ControlFlow::Continue(())
        }),
    )
    .unwrap_err();
    assert!(matches!(err, ReadError::Corrupt(_)), "got {err:?}");
    assert_eq!(got, (0..20).map(key).collect::<Vec<_>>());
    assert_eq!(dev.reads_in_flight(), 0);
}

#[test]
fn range_that_skips_a_broken_leaf_never_touches_it() {
    let dev = Arc::new(FakeIo::new(0));
    let reader = flat_reader(Arc::clone(&dev), 48);
    // Leaf 0 is fatally broken — but a scan bounded to leaves 6.. must route
    // around it at the branch level and never issue its read.
    dev.set_faults(Faults {
        fail_read_offsets: vec![0],
        ..Faults::default()
    });
    let got = scan(&reader, key(24)..key(36));
    assert_eq!(got, oracle(48, key(24)..key(36)));
}

#[test]
fn dropped_mid_flight_scan_leaves_the_reader_reusable() {
    // Cancellation: park the scan's first physical read at the device, drop
    // the whole scan future, release the gate — then a fresh scan over the
    // same reader/cache must still see every entry (the cache's
    // cancellation-independent single-flight hands the abandoned load to the
    // next caller; nothing is wedged or double-counted).
    let dev = Arc::new(FakeIo::new(0));
    let reader = flat_reader(Arc::clone(&dev), 48);
    dev.arm_read_gate(1);
    {
        let mut sink = Vec::new();
        let mut fut = Box::pin(
            reader.for_each_range::<&[u8], _, _>(.., |k: &[u8], _v: &[u8]| {
                sink.push(k.to_vec());
                ControlFlow::Continue(())
            }),
        );
        assert!(
            block_on(futures_lite::future::poll_once(fut.as_mut())).is_none(),
            "the cold scan must park on its gated first read"
        );
        // Drop the future with its read parked.
    }
    dev.release_reads(1);
    assert_eq!(scan(&reader, ..), oracle(48, ..));
    assert_eq!(dev.reads_in_flight(), 0);
}

// ---------------------------------------------------------------------------
// WAL integration: recovery, retention expiry, and a genuinely in-flight scan
// racing a live writer.
// ---------------------------------------------------------------------------

use crate::fileset::{create_mem_store, open_mem_store, MemWalFileSet};

fn as_dev(dev: &Arc<FakeIo>) -> Arc<dyn PageIo> {
    Arc::clone(dev) as Arc<dyn PageIo>
}

fn wal_sorted_bounded(
    bound: Option<u32>,
) -> (SortedMap<crate::Wal>, Arc<FakeIo>, Arc<MemWalFileSet>) {
    let data = Arc::new(FakeIo::new(0));
    let file_set = MemWalFileSet::new();
    let options = Options {
        max_retained_generations: bound,
        ..Options::default()
    };
    let store = create_mem_store(as_dev(&data), Arc::clone(&file_set), options).unwrap();
    (SortedMap::over(store), data, file_set)
}

/// Clean-crash reopen: fresh devices frozen at each file's last datasync.
fn reopen_synced(data: &Arc<FakeIo>, file_set: &Arc<MemWalFileSet>) -> SortedMap<crate::Wal> {
    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let fresh = MemWalFileSet::new();
    for seq in file_set.live_segments() {
        let seg = file_set.segment(seq).unwrap();
        fresh.insert(seq, Arc::new(FakeIo::from_vec(seg.synced_bytes())));
    }
    let store = open_mem_store(as_dev(&d), fresh, Options::default()).unwrap();
    SortedMap::over(store)
}

#[test]
fn scan_after_crash_recovery_sees_exactly_the_committed_state() {
    let (m, data, file_set) = wal_sorted_bounded(None);
    block_on(async {
        let v = m.put_all((0..30).map(|i| (key(i), val(i)))).await.unwrap();
        m.commit(v).await.unwrap();
    });
    // Apply-without-commit, with the WAL segment's datasync PARKED at the
    // device: the entry is visible live but its commit record can never reach
    // durability before the images are taken. (A plain `close()` would drain
    // and sync it — that is the whole reason for the gate.)
    let seg0 = file_set.segment(0).unwrap();
    seg0.arm_sync_gate(1);
    block_on(async {
        m.insert(b"uncommitted".to_vec(), b"lost".to_vec())
            .await
            .unwrap();
        assert!(m.contains_key(b"uncommitted").await.unwrap());
    });
    // Freeze the crash images while the tail sync is parked.
    let recovered = reopen_synced(&data, &file_set);
    seg0.release_syncs(usize::MAX);
    block_on(async {
        let got = recovered.collect_range::<&[u8], _>(.., 100).await.unwrap();
        let want: Vec<_> = (0..30).map(|i| (key(i), val(i))).collect();
        assert_eq!(
            got, want,
            "recovery scan = every committed entry, nothing else"
        );
        assert_eq!(recovered.first_entry().await.unwrap().unwrap().0, key(0));
        assert_eq!(recovered.last_entry().await.unwrap().unwrap().0, key(29));
        recovered.close().await.unwrap();
        m.close().await.unwrap();
    });
}

#[test]
fn expired_snapshot_scan_and_edges_fail_with_snapshot_expired() {
    let (m, _data, _fs) = wal_sorted_bounded(Some(1));
    block_on(async {
        let v = m.put_all((0..10).map(|i| (key(i), val(i)))).await.unwrap();
        m.commit(v).await.unwrap();
        let snap = m.snapshot();

        // Two checkpoints push the snapshot's generation past the retention
        // bound (the phase-E recipe from tests_checkpoint_compaction).
        m.store().checkpoint().await.unwrap();
        let v2 = m.insert(b"newer".to_vec(), b"v".to_vec()).await.unwrap();
        m.commit(v2).await.unwrap();
        let _pin_newer = m.snapshot();
        m.store().checkpoint().await.unwrap();

        assert!(snap.is_expired());
        let err = snap
            .for_each_range::<&[u8], _, _>(.., |_k: &[u8], _v: &[u8]| ControlFlow::Continue(()))
            .await
            .unwrap_err();
        assert!(matches!(err, ReadError::SnapshotExpired), "got {err:?}");
        let err = snap
            .for_each_range_rev::<&[u8], _, _>(.., |_k: &[u8], _v: &[u8]| ControlFlow::Continue(()))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ReadError::SnapshotExpired),
            "rev: got {err:?}"
        );
        let err = snap.first_entry().await.unwrap_err();
        assert!(matches!(err, ReadError::SnapshotExpired), "got {err:?}");

        // The live map is untouched by the snapshot's expiry.
        assert_eq!(m.count_range::<&[u8], _>(..).await.unwrap(), 11);
        m.close().await.unwrap();
    });
}

#[test]
fn parked_cold_scan_stays_isolated_from_concurrent_writes() {
    // A scan that is genuinely in flight (a cold leaf read parked at the
    // device) must deliver its snapshot's exact state even though the writer
    // publishes a new version while it is parked.
    //
    // Gate discipline: the read gate parks EVERY cold read, including the
    // writer's own CoW descent — so the writer must be given a fully warm
    // path before the gate arms. A point get of key(0) warms root + leaf 0;
    // the scan then delivers leaf 0 from cache and parks on leaf 1, while
    // overwrites of key(0) run entirely warm.
    const STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    let (m, data, file_set) = wal_sorted_bounded(None);
    block_on(async {
        // Enough entries to span several production-size leaves.
        let v = m.put_all((0..800).map(|i| (key(i), val(i)))).await.unwrap();
        m.commit(v).await.unwrap();
        // Checkpoint so the tree pages live in the DATA file: an
        // un-checkpointed page is served from the WAL segment, and the gate
        // below is on the data device.
        m.store().checkpoint().await.unwrap();
        m.close().await.unwrap();
    });
    // Reopen over frozen images with an explicit device handle so the gate is
    // ours; the cache starts cold, forcing the scan to physically read.
    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let fresh = MemWalFileSet::new();
    for seq in file_set.live_segments() {
        let seg = file_set.segment(seq).unwrap();
        fresh.insert(seq, Arc::new(FakeIo::from_vec(seg.synced_bytes())));
    }
    let store = open_mem_store(as_dev(&d), fresh, Options::default()).unwrap();
    let m2 = SortedMap::over(store);

    // Warm the writer's whole path for key(0) before arming the gate.
    assert_eq!(block_on(m2.get(key(0))).unwrap().unwrap(), val(0));

    d.arm_read_gate(1);
    let scan_map = m2.clone();
    let handle = std::thread::spawn(move || {
        let mut got: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let summary = block_on(scan_map.for_each(|k, v| {
            got.push((k.to_vec(), v.to_vec()));
            ControlFlow::Continue(())
        }))
        .unwrap();
        (summary, got)
    });
    let deadline = std::time::Instant::now() + STEP_TIMEOUT;
    while d.parked_read_count() == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "scan never parked at the gate"
        );
        std::thread::yield_now();
    }
    // While the scan is parked mid-tree, overwrite key(0) — a fully-warm
    // write path — publishing a new root.
    block_on(async {
        m2.insert(key(0), b"overwritten".to_vec()).await.unwrap();
    });
    d.release_reads(1);
    let (summary, got) = handle.join().unwrap();
    assert!(summary.completed);
    assert_eq!(got.len(), 800);
    assert_eq!(
        got[0],
        (key(0), val(0)),
        "the parked scan sees its snapshot's value, not the overwrite"
    );
    // A fresh read sees the new version.
    assert_eq!(
        block_on(m2.get(key(0))).unwrap().unwrap(),
        b"overwritten".to_vec()
    );
    block_on(m2.close()).unwrap();
}

// ---------------------------------------------------------------------------
// SortedMap API edges.
// ---------------------------------------------------------------------------

#[test]
fn put_many_empty_is_a_versioned_no_op() {
    let m = direct_sorted();
    block_on(async {
        m.insert(key(1), val(1)).await.unwrap();
        let before = m.collect_range::<&[u8], _>(.., 10).await.unwrap();
        let v = m
            .put_all(std::iter::empty::<(Vec<u8>, Vec<u8>)>())
            .await
            .unwrap();
        // A version was published and nothing changed.
        let after = m.collect_range::<&[u8], _>(.., 10).await.unwrap();
        assert_eq!(before, after);
        let v2 = m.insert(key(2), val(2)).await.unwrap();
        assert!(v2.txid() > v.txid());
        m.close().await.unwrap();
    });
}

#[test]
fn remove_absent_key_is_a_versioned_no_op() {
    let m = direct_sorted();
    block_on(async {
        m.insert(key(1), val(1)).await.unwrap();
        let v = m.remove(b"never-existed".to_vec()).await.unwrap();
        assert!(v.txid() > 0);
        assert_eq!(m.count_range::<&[u8], _>(..).await.unwrap(), 1);
        m.close().await.unwrap();
    });
}

#[test]
fn cas_on_an_absent_key_with_wrong_expectation_fails_clean() {
    let m = direct_sorted();
    block_on(async {
        // expected Some(..) on an absent key → precondition fails, nothing
        // published.
        assert!(m
            .compare_and_set(b"ghost".to_vec(), Some(b"x".to_vec()), Some(b"y".to_vec()))
            .await
            .unwrap()
            .is_none());
        assert!(!m.contains_key(b"ghost").await.unwrap());
        // expected None on an absent key with new = None: a conditional
        // no-op remove that succeeds.
        assert!(m
            .compare_and_set(b"ghost".to_vec(), None, None)
            .await
            .unwrap()
            .is_some());
        // put_if_absent works again after a real remove.
        assert!(m
            .put_if_absent(b"k".to_vec(), b"v1".to_vec())
            .await
            .unwrap()
            .is_some());
        m.remove(b"k".to_vec()).await.unwrap();
        assert!(m
            .put_if_absent(b"k".to_vec(), b"v2".to_vec())
            .await
            .unwrap()
            .is_some());
        assert_eq!(m.get(b"k").await.unwrap().unwrap(), b"v2");
        m.close().await.unwrap();
    });
}

#[test]
fn clones_share_one_store() {
    let m = direct_sorted();
    let m2 = m.clone();
    block_on(async {
        m2.insert(key(1), val(1)).await.unwrap();
        assert_eq!(m.get(key(1)).await.unwrap().unwrap(), val(1));
        assert_eq!(m.len(), m2.len());
        m.close().await.unwrap();
    });
}

#[test]
fn collect_range_limit_exactly_total_returns_everything() {
    let m = direct_sorted();
    block_on(async {
        m.put_all((0..8).map(|i| (key(i), val(i)))).await.unwrap();
        let got = m.collect_range::<&[u8], _>(.., 8).await.unwrap();
        assert_eq!(got.len(), 8);
        let one = m.collect_range::<&[u8], _>(.., 1).await.unwrap();
        assert_eq!(one, vec![(key(0), val(0))]);
        m.close().await.unwrap();
    });
}

#[test]
fn prefix_scan_of_an_exact_full_key_matches_only_extensions_and_itself() {
    let m = direct_sorted();
    block_on(async {
        m.insert(b"a".to_vec(), b"1".to_vec()).await.unwrap();
        m.insert(b"ab".to_vec(), b"2".to_vec()).await.unwrap();
        m.insert(b"b".to_vec(), b"3".to_vec()).await.unwrap();
        let mut got = Vec::new();
        m.for_each_prefix(b"a", |k, _v| {
            got.push(k.to_vec());
            ControlFlow::Continue(())
        })
        .await
        .unwrap();
        assert_eq!(got, vec![b"a".to_vec(), b"ab".to_vec()]);
        // Empty prefix covers the whole map.
        let mut n = 0u32;
        m.for_each_prefix(b"", |_k, _v| {
            n += 1;
            ControlFlow::Continue(())
        })
        .await
        .unwrap();
        assert_eq!(n, 3);
        m.close().await.unwrap();
    });
}

#[test]
fn get_many_through_sorted_map_matches_point_gets() {
    let m = direct_sorted();
    block_on(async {
        m.put_all((0..10).map(|i| (key(i), val(i)))).await.unwrap();
        let keys = vec![key(3), b"missing".to_vec(), key(7), key(3)];
        let many = m.get_many(keys.clone()).await.unwrap();
        for (k, got) in keys.iter().zip(&many) {
            assert_eq!(got, &m.get(k).await.unwrap());
        }
        m.close().await.unwrap();
    });
}

// ---------------------------------------------------------------------------
// Properties: randomized shapes, ranges, and op interleavings against the
// std::collections::BTreeMap oracle (same proptest config family as
// tests_crash_recovery_fuzz).
// ---------------------------------------------------------------------------

use proptest::prelude::*;

/// Short keys over a tiny alphabet: maximal boundary collisions (equal keys,
/// shared prefixes, keys straddling separators).
fn small_key() -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(0u8..6, 1..4)
}

fn bound_of(k: Vec<u8>, kind: u8) -> Bound<Vec<u8>> {
    match kind % 3 {
        0 => Bound::Included(k),
        1 => Bound::Excluded(k),
        _ => Bound::Unbounded,
    }
}

/// std's `range` refuses inverted bounds; mirror its rule.
fn std_refuses(start: &Bound<Vec<u8>>, end: &Bound<Vec<u8>>) -> bool {
    match (start, end) {
        (Bound::Included(s) | Bound::Excluded(s), Bound::Included(e) | Bound::Excluded(e)) => {
            s > e || (s == e && matches!((start, end), (Bound::Excluded(_), Bound::Excluded(_))))
        }
        _ => false,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    /// Any key set, any tree shape, any range: the scan equals the std range.
    #[test]
    fn prop_reader_scan_matches_std_range(
        keys in proptest::collection::btree_set(small_key(), 0..80),
        fanout in 2usize..6,
        leaf_cap in 1usize..5,
        b1 in small_key(),
        b2 in small_key(),
        k1 in 0u8..3,
        k2 in 0u8..3,
    ) {
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| (k.clone(), format!("v{i}").into_bytes()))
            .collect();
        let built = TreeBuilder {
            max_fanout: fanout,
            max_leaf_entries: leaf_cap,
            ..Default::default()
        }
        .build(Arc::new(FakeIo::new(0)), &pairs)
        .unwrap();
        let reader = TreeReader::new(built);
        let std_map: StdMap<Vec<u8>, Vec<u8>> = pairs.iter().cloned().collect();

        let start = bound_of(b1, k1);
        let end = bound_of(b2, k2);
        let got = scan(&reader, (start.clone(), end.clone()));
        if std_refuses(&start, &end) {
            prop_assert!(got.is_empty());
        } else {
            let want: Vec<(Vec<u8>, Vec<u8>)> = std_map
                .range((start, end))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            prop_assert_eq!(got, want);
        }
        // Full scan agrees with the whole map regardless of the range case.
        let full = scan(&reader, ..);
        let all: Vec<(Vec<u8>, Vec<u8>)> =
            std_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        prop_assert_eq!(full, all);
    }

    /// Early stop after N entries is exactly the N-prefix of the full scan.
    #[test]
    fn prop_early_stop_is_a_prefix(
        keys in proptest::collection::btree_set(small_key(), 1..60),
        stop in 0usize..70,
    ) {
        let pairs: Vec<(Vec<u8>, Vec<u8>)> =
            keys.iter().map(|k| (k.clone(), b"v".to_vec())).collect();
        let built = TreeBuilder {
            max_fanout: 3,
            max_leaf_entries: 2,
            ..Default::default()
        }
        .build(Arc::new(FakeIo::new(0)), &pairs)
        .unwrap();
        let reader = TreeReader::new(built);
        let full = scan(&reader, ..);

        let mut got = Vec::new();
        let summary = block_on(reader.for_each_range::<&[u8], _, _>(.., |k: &[u8], _v: &[u8]| {
            got.push(k.to_vec());
            if got.len() >= stop {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }))
        .unwrap();
        // The consumer decides after seeing each entry, so even stop == 0
        // delivers the first entry of a non-empty scan; a break lands exactly
        // when the running count first reaches `stop`.
        let expect_n = if full.is_empty() {
            0
        } else {
            stop.clamp(1, full.len())
        };
        prop_assert_eq!(got.len(), expect_n);
        let want: Vec<Vec<u8>> = full.iter().take(expect_n).map(|(k, _)| k.clone()).collect();
        prop_assert_eq!(got, want);
        prop_assert_eq!(summary.visited as usize, expect_n);
        // The scan completes iff the break never fired: only when the whole
        // map is shorter than the stop threshold (or empty).
        prop_assert_eq!(summary.completed, full.is_empty() || stop > full.len());
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    /// Random put/remove interleavings on a REAL store (writer thread, CoW
    /// builder, admission) end in exactly the oracle's sorted state — scans,
    /// count, first/last, and a random prefix all agree.
    #[test]
    fn prop_store_ops_then_scan_matches_oracle(
        ops in proptest::collection::vec((any::<bool>(), small_key(), 0u8..250), 0..60),
        prefix in proptest::collection::vec(0u8..6, 0..2),
    ) {
        let m = direct_sorted();
        let mut oracle: StdMap<Vec<u8>, Vec<u8>> = StdMap::new();
        block_on(async {
            for (is_put, k, vb) in &ops {
                if *is_put {
                    m.insert(k.clone(), vec![*vb]).await.unwrap();
                    oracle.insert(k.clone(), vec![*vb]);
                } else {
                    m.remove(k.clone()).await.unwrap();
                    oracle.remove(k);
                }
            }
            let got = m.collect_range::<&[u8], _>(.., 1000).await.unwrap();
            let want: Vec<(Vec<u8>, Vec<u8>)> =
                oracle.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            prop_assert_eq!(&got, &want);
            prop_assert_eq!(
                m.count_range::<&[u8], _>(..).await.unwrap(),
                oracle.len() as u64
            );
            prop_assert_eq!(m.len(), oracle.len() as u64);
            prop_assert_eq!(
                m.first_entry().await.unwrap(),
                oracle.iter().next().map(|(k, v)| (k.clone(), v.clone()))
            );
            prop_assert_eq!(
                m.last_entry().await.unwrap(),
                oracle.iter().next_back().map(|(k, v)| (k.clone(), v.clone()))
            );
            // A random prefix scan agrees with the oracle filter.
            let mut got_pref = Vec::new();
            m.for_each_prefix(&prefix, |k, _v| {
                got_pref.push(k.to_vec());
                ControlFlow::Continue(())
            })
            .await
            .unwrap();
            let want_pref: Vec<Vec<u8>> = oracle
                .keys()
                .filter(|k| k.starts_with(&prefix))
                .cloned()
                .collect();
            prop_assert_eq!(got_pref, want_pref);
            m.close().await.unwrap();
            Ok(())
        })?;
    }
}

// ---------------------------------------------------------------------------
// NavigableMap alignment (floor/ceiling/lower/higher, mapdb-collections
// vocabulary) and range removal.
// ---------------------------------------------------------------------------

#[test]
fn navigation_family_matches_std_btreemap_semantics() {
    let m = direct_sorted();
    block_on(async {
        // Sparse keys so probes land between entries: 10, 20, ..., 100.
        m.put_all((1..=10).map(|i| (key(i * 10), val(i * 10))))
            .await
            .unwrap();
        let oracle: StdMap<Vec<u8>, Vec<u8>> =
            (1..=10).map(|i| (key(i * 10), val(i * 10))).collect();

        // Probes: below all, on an entry, between entries, above all.
        for probe in [
            key(5),
            key(10),
            key(15),
            key(50),
            key(55),
            key(100),
            key(101),
        ] {
            let floor = oracle
                .range(..=probe.clone())
                .next_back()
                .map(|(k, v)| (k.clone(), v.clone()));
            let lower = oracle
                .range(..probe.clone())
                .next_back()
                .map(|(k, v)| (k.clone(), v.clone()));
            let ceiling = oracle
                .range(probe.clone()..)
                .next()
                .map(|(k, v)| (k.clone(), v.clone()));
            let higher = oracle
                .range((Bound::Excluded(probe.clone()), Bound::Unbounded))
                .next()
                .map(|(k, v)| (k.clone(), v.clone()));
            assert_eq!(
                m.floor_entry(&probe).await.unwrap(),
                floor,
                "floor {probe:?}"
            );
            assert_eq!(
                m.lower_entry(&probe).await.unwrap(),
                lower,
                "lower {probe:?}"
            );
            assert_eq!(
                m.ceiling_entry(&probe).await.unwrap(),
                ceiling,
                "ceiling {probe:?}"
            );
            assert_eq!(
                m.higher_entry(&probe).await.unwrap(),
                higher,
                "higher {probe:?}"
            );
        }
        m.close().await.unwrap();
    });
}

#[test]
fn navigation_on_empty_map_and_overflow_values() {
    let m = direct_sorted();
    block_on(async {
        assert_eq!(m.floor_entry(key(1)).await.unwrap(), None);
        assert_eq!(m.ceiling_entry(key(1)).await.unwrap(), None);
        // An overflow value comes back reassembled through every nav path.
        let big = vec![b'n'; MAX_INLINE_VALUE_LEN * 2 + 9];
        m.insert(key(50), big.clone()).await.unwrap();
        assert_eq!(
            m.floor_entry(key(99)).await.unwrap().unwrap(),
            (key(50), big.clone())
        );
        assert_eq!(
            m.ceiling_entry(key(1)).await.unwrap().unwrap(),
            (key(50), big.clone())
        );
        assert_eq!(m.lower_entry(key(50)).await.unwrap(), None);
        assert_eq!(m.higher_entry(key(50)).await.unwrap(), None);
        m.close().await.unwrap();
    });
}

/// Floor's fallback subtree: probe keys whose descent leaf holds NO
/// qualifying entry, so the answer must come from the nearest left sibling
/// subtree — exercised across every leaf boundary of a deep tree.
#[test]
fn floor_falls_back_across_leaf_boundaries() {
    let m = direct_sorted();
    block_on(async {
        // 200 entries at even positions; probe every odd position. With
        // production leaf sizes this crosses multiple leaves; the same probe
        // set also runs strict (lower) on the entries themselves.
        m.put_all((0..200).map(|i| (key(i * 2), val(i * 2))))
            .await
            .unwrap();
        for i in 0..200 {
            let probe = key(i * 2 + 1);
            assert_eq!(
                m.floor_entry(&probe).await.unwrap().unwrap().0,
                key(i * 2),
                "floor of odd probe {i}"
            );
            if i > 0 {
                assert_eq!(
                    m.lower_entry(key(i * 2)).await.unwrap().unwrap().0,
                    key((i - 1) * 2),
                    "lower of even entry {i}"
                );
            }
        }
        assert_eq!(m.lower_entry(key(0)).await.unwrap(), None);
        m.close().await.unwrap();
    });
}

#[test]
fn remove_range_deletes_exactly_the_bounded_keys() {
    let m = direct_sorted();
    block_on(async {
        m.put_all((0..50).map(|i| (key(i), val(i)))).await.unwrap();
        let removed = m.remove_range(key(10)..key(40)).await.unwrap();
        assert_eq!(removed, 30);
        assert_eq!(m.len(), 20);
        let left: Vec<Vec<u8>> = m
            .collect_range::<&[u8], _>(.., 100)
            .await
            .unwrap()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        let want: Vec<Vec<u8>> = (0..10).chain(40..50).map(key).collect();
        assert_eq!(left, want);
        // Removing an already-empty range is a zero-count no-op.
        assert_eq!(m.remove_range(key(10)..key(40)).await.unwrap(), 0);
        m.close().await.unwrap();
    });
}

#[test]
fn remove_range_chunks_past_one_batch() {
    let m = direct_sorted();
    block_on(async {
        // > 512 keys forces several remove chunks. (Inserted in three
        // batches: a single 1500-op batch's worst-case page cost exceeds the
        // default admission budget — put_all is atomic, so it must fit.)
        for chunk in [0..500, 500..1000, 1000..1500] {
            m.put_all(chunk.map(|i| (key(i), b"v".to_vec())))
                .await
                .unwrap();
        }
        assert_eq!(m.clear().await.unwrap(), 1500);
        assert!(m.is_empty());
        assert_eq!(m.first_entry().await.unwrap(), None);
        m.close().await.unwrap();
    });
}

// ---------------------------------------------------------------------------
// Descending scans (`for_each_range_rev` / `collect_range_rev` — reverse
// iteration). Oracle: the std range reversed. The traversal is the
// path-stack mirror (branches right-to-left, leaves back-to-front); no format
// change.
// ---------------------------------------------------------------------------

/// Collects a reader's descending scan into owned pairs.
fn scan_rev<R: std::ops::RangeBounds<Vec<u8>>>(
    reader: &TreeReader,
    range: R,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    let summary = block_on(reader.for_each_range_rev(range, |k, v| {
        out.push((k.to_vec(), v.to_vec()));
        ControlFlow::Continue(())
    }))
    .unwrap();
    assert!(summary.completed);
    assert_eq!(summary.visited as usize, out.len());
    out
}

fn oracle_rev<R: std::ops::RangeBounds<Vec<u8>>>(n: usize, range: R) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut want = oracle(n, range);
    want.reverse();
    want
}

#[test]
fn rev_full_scan_is_descending_and_complete() {
    let reader = deep_reader(100);
    assert_eq!(scan_rev(&reader, ..), oracle_rev(100, ..));
    // And an empty tree scans to nothing, completed.
    let empty = TreeReader::new(
        TreeBuilder::default()
            .build(Arc::new(FakeIo::new(0)), &[])
            .unwrap(),
    );
    assert!(scan_rev(&empty, ..).is_empty());
}

#[test]
fn rev_range_bounds_match_std_btreemap() {
    let n = 100;
    let reader = deep_reader(n);
    let probes: Vec<Vec<u8>> = vec![
        key(0),
        key(3),
        key(4),
        key(15),
        key(16),
        key(50),
        key(99),
        b"key000015zzz".to_vec(),
        b"a".to_vec(),
        b"z".to_vec(),
    ];
    for lo in &probes {
        for hi in &probes {
            type Bnd = Bound<Vec<u8>>;
            let cases: Vec<(Bnd, Bnd)> = vec![
                (Bound::Included(lo.clone()), Bound::Included(hi.clone())),
                (Bound::Included(lo.clone()), Bound::Excluded(hi.clone())),
                (Bound::Excluded(lo.clone()), Bound::Included(hi.clone())),
                (Bound::Excluded(lo.clone()), Bound::Excluded(hi.clone())),
                (Bound::Unbounded, Bound::Included(hi.clone())),
                (Bound::Included(lo.clone()), Bound::Unbounded),
            ];
            for (start, end) in cases {
                let got = scan_rev(&reader, (start.clone(), end.clone()));
                if std_refuses(&start, &end) {
                    assert!(got.is_empty(), "inverted {start:?}..{end:?}");
                } else {
                    assert_eq!(got, oracle_rev(n, (start, end)));
                }
            }
        }
    }
}

/// Early stop after N entries is exactly the N-*suffix* of the key space, in
/// descending order, reported incomplete.
#[test]
fn rev_early_stop_is_a_descending_suffix() {
    let reader = deep_reader(100);
    let mut out: Vec<Vec<u8>> = Vec::new();
    let summary = block_on(reader.for_each_range_rev::<&[u8], _, _>(.., |k, _v| {
        out.push(k.to_vec());
        if out.len() == 7 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }))
    .unwrap();
    assert!(!summary.completed);
    assert_eq!(summary.visited, 7);
    let want: Vec<Vec<u8>> = (93..100).rev().map(key).collect();
    assert_eq!(out, want);
}

#[test]
fn rev_scan_reassembles_overflow_values() {
    let big = |i: usize| {
        let mut v = vec![b'a' + (i % 26) as u8; MAX_INLINE_VALUE_LEN * 3 + i];
        v.extend_from_slice(format!("-{i}").as_bytes());
        v
    };
    let entries: Vec<(Vec<u8>, Vec<u8>)> = (0..6).map(|i| (key(i), big(i))).collect();
    let built = TreeBuilder {
        max_fanout: 4,
        max_leaf_entries: 2,
        ..Default::default()
    }
    .build(Arc::new(FakeIo::new(0)), &entries)
    .unwrap();
    let reader = TreeReader::new(built);
    let mut want = entries;
    want.reverse();
    assert_eq!(scan_rev(&reader, ..), want);
}

/// Newest-first pagination through `collect_range_rev`: each page resumes
/// with `(Unbounded, Excluded(previous page's smallest key))`; the
/// concatenation is the whole map descending, with no duplicate or hole.
#[test]
fn sorted_map_descending_pagination_walks_the_whole_map() {
    let m = direct_sorted();
    block_on(async {
        m.put_all((0..300).map(|i| (key(i), val(i)))).await.unwrap();
        assert!(m
            .collect_range_rev::<&[u8], _>(.., 0)
            .await
            .unwrap()
            .is_empty());

        let mut pages: Vec<Vec<u8>> = Vec::new();
        let mut resume: Option<Vec<u8>> = None;
        loop {
            let page = match &resume {
                None => m.collect_range_rev::<&[u8], _>(.., 37).await.unwrap(),
                Some(last) => m
                    .collect_range_rev((Bound::Unbounded, Bound::Excluded(last.clone())), 37)
                    .await
                    .unwrap(),
            };
            if page.is_empty() {
                break;
            }
            assert!(page.len() <= 37);
            resume = Some(page.last().unwrap().0.clone());
            pages.extend(page.into_iter().map(|(k, _)| k));
        }
        let want: Vec<Vec<u8>> = (0..300).rev().map(key).collect();
        assert_eq!(pages, want);
        m.close().await.unwrap();
    });
}

/// A snapshot captured before an overwrite scans (descending) only the
/// original values; a snapshot captured after sees only the new ones.
#[test]
fn rev_scan_is_snapshot_consistent_across_writes() {
    let m = direct_sorted();
    block_on(async {
        m.put_all((0..50).map(|i| (key(i), val(i)))).await.unwrap();
        let before = m.map().snapshot();
        m.put_all((0..50).map(|i| (key(i), b"NEW".to_vec())))
            .await
            .unwrap();
        let mut old_vals: Vec<Vec<u8>> = Vec::new();
        before
            .for_each_range_rev::<&[u8], _, _>(.., |_k, v| {
                old_vals.push(v.to_vec());
                ControlFlow::Continue(())
            })
            .await
            .unwrap();
        let want: Vec<Vec<u8>> = (0..50).rev().map(val).collect();
        assert_eq!(
            old_vals, want,
            "the pinned snapshot must predate the overwrite"
        );
        let after = m.map().snapshot();
        after
            .for_each_range_rev::<&[u8], _, _>(.., |_k, v| {
                assert_eq!(v, b"NEW");
                ControlFlow::Continue(())
            })
            .await
            .unwrap();
        m.close().await.unwrap();
    });
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    /// Any key set, any tree shape, any range: the descending scan equals the
    /// std range reversed.
    #[test]
    fn prop_rev_scan_matches_std_range_reversed(
        keys in proptest::collection::btree_set(small_key(), 0..80),
        fanout in 2usize..6,
        leaf_cap in 1usize..5,
        b1 in small_key(),
        b2 in small_key(),
        k1 in 0u8..3,
        k2 in 0u8..3,
    ) {
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| (k.clone(), format!("v{i}").into_bytes()))
            .collect();
        let built = TreeBuilder {
            max_fanout: fanout,
            max_leaf_entries: leaf_cap,
            ..Default::default()
        }
        .build(Arc::new(FakeIo::new(0)), &pairs)
        .unwrap();
        let reader = TreeReader::new(built);
        let std_map: StdMap<Vec<u8>, Vec<u8>> = pairs.iter().cloned().collect();

        let start = bound_of(b1, k1);
        let end = bound_of(b2, k2);
        let got = scan_rev(&reader, (start.clone(), end.clone()));
        if std_refuses(&start, &end) {
            prop_assert!(got.is_empty());
        } else {
            let want: Vec<(Vec<u8>, Vec<u8>)> = std_map
                .range((start, end))
                .rev()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            prop_assert_eq!(got, want);
        }
        // The full descending scan is exactly the full ascending scan reversed.
        let mut fwd = scan(&reader, ..);
        fwd.reverse();
        prop_assert_eq!(scan_rev(&reader, ..), fwd);
    }
}

/// `for_each_prefix_rev` mirrors `for_each_prefix` exactly reversed —
/// including the saturating-0xFF prefix successor edge, and the
/// latest-under-a-prefix idiom (break on the first delivery).
#[test]
fn prefix_rev_scans_descending_and_finds_the_latest() {
    let m = direct_sorted();
    block_on(async {
        for (k, v) in [
            (&b"a:1"[..], &b"1"[..]),
            (b"b:1", b"2"),
            (b"b:2", b"3"),
            (b"b:3", b"4"),
            (b"c:1", b"5"),
            (b"\xff\xff", b"6"),
        ] {
            m.insert(k.to_vec(), v.to_vec()).await.unwrap();
        }
        let mut got: Vec<Vec<u8>> = Vec::new();
        m.for_each_prefix_rev(b"b:", |k, _v| {
            got.push(k.to_vec());
            ControlFlow::Continue(())
        })
        .await
        .unwrap();
        assert_eq!(got, vec![b"b:3".to_vec(), b"b:2".to_vec(), b"b:1".to_vec()]);

        // Latest under the prefix: first delivery, then break.
        let mut latest: Option<Vec<u8>> = None;
        let summary = m
            .for_each_prefix_rev(b"b:", |k, _v| {
                latest = Some(k.to_vec());
                ControlFlow::Break(())
            })
            .await
            .unwrap();
        assert_eq!(latest.as_deref(), Some(&b"b:3"[..]));
        assert!(!summary.completed);

        // All-0xFF prefix saturates to an unbounded end and still scans.
        let mut ff: Vec<Vec<u8>> = Vec::new();
        m.for_each_prefix_rev(b"\xff", |k, _v| {
            ff.push(k.to_vec());
            ControlFlow::Continue(())
        })
        .await
        .unwrap();
        assert_eq!(ff, vec![b"\xff\xff".to_vec()]);

        // for_each_rev covers the whole map descending.
        let mut all = 0usize;
        let mut prev: Option<Vec<u8>> = None;
        m.for_each_rev(|k, _v| {
            if let Some(p) = &prev {
                assert!(k < p.as_slice(), "descending order violated");
            }
            prev = Some(k.to_vec());
            all += 1;
            ControlFlow::Continue(())
        })
        .await
        .unwrap();
        assert_eq!(all, 6);
        m.close().await.unwrap();
    });
}
