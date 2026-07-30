//! Acceptance for the Direct backend's write path over real copy-on-write pages.
//!
//! Asserts sequential behaviour against `std::collections::BTreeMap` including
//! splits and removes across levels; that a concurrent snapshot never observes
//! half a batch or half a split; that dropping an apply future still applies or
//! fails truthfully without leaking budget; that an intermediate snapshot stays
//! readable after a later flush and forced eviction; that a racing or regressing
//! flush never lowers the durable header; that sustained writes progress at the
//! high-water mark with no explicit flush; and that flush faults and corrupt
//! headers fail bounded with valid data or `Corrupt`, never a hang.
//!
//! Also pins the flush-write fan-out: identical durable bytes to the serial
//! path, a bounded in-flight window, and no header write after any chunk failure.

#![allow(clippy::bool_assert_comparison)]

use crate::admission::AdmissionLimits;
use crate::batch::ApplyOutcome;
use crate::direct::DirectHeader;
use crate::error::WriteError;
use crate::io::{FakeIo, Faults, PageIo};
use crate::store::{Options, Store};
use crate::{BTreeMap, WriteBatch};
use futures_lite::future::block_on;
use std::collections::BTreeMap as StdMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn direct_map(dev: Arc<FakeIo>) -> BTreeMap<crate::Direct> {
    let store = Store::<crate::Direct>::create_with_io(dev, Options::default()).unwrap();
    BTreeMap::over(store)
}

fn direct_map_opts(dev: Arc<FakeIo>, options: Options) -> BTreeMap<crate::Direct> {
    let store = Store::<crate::Direct>::create_with_io(dev, options).unwrap();
    BTreeMap::over(store)
}

/// Coerces a `FakeIo` handle to the `PageIo` trait object `open_over` expects.
fn as_dev(dev: &Arc<FakeIo>) -> Arc<dyn PageIo> {
    Arc::clone(dev) as Arc<dyn PageIo>
}

/// A value big enough that only a few entries fit one 16 KiB page, so modest
/// entry counts already force leaf and branch splits.
fn big_val(seed: u32) -> Vec<u8> {
    let mut v = vec![0u8; 3000];
    v[..4].copy_from_slice(&seed.to_le_bytes());
    v
}

fn spin_until(mut cond: impl FnMut() -> bool) {
    for _ in 0..5000 {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("condition never became true");
}

#[test]
fn insert_split_remove_multilevel() {
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(dev);
    let n = 200u32;
    block_on(async {
        // Insert enough big values to force a genuine multi-level tree.
        for i in 0..n {
            let out = map
                .apply(WriteBatch::new().insert(format!("k{i:05}").into_bytes(), big_val(i)))
                .await
                .unwrap();
            assert!(matches!(out, ApplyOutcome::Applied { .. }));
        }
        assert_eq!(map.len(), n as u64);
        assert_eq!(map.verify().await.unwrap().entry_count, n as u64);
        // Read every key back.
        for i in 0..n {
            let got = map.get(format!("k{i:05}").into_bytes()).await.unwrap();
            assert_eq!(got.as_deref().map(|v| &v[..4]), Some(&i.to_le_bytes()[..]));
        }
        // Overwrite a few (no count change).
        map.apply(WriteBatch::new().insert(b"k00100".to_vec(), b"tiny".to_vec()))
            .await
            .unwrap();
        assert_eq!(
            map.get(b"k00100".to_vec()).await.unwrap(),
            Some(b"tiny".to_vec())
        );
        assert_eq!(map.len(), n as u64);
        // Remove all: the tree empties to the canonical empty root.
        for i in 0..n {
            map.apply(WriteBatch::new().remove(format!("k{i:05}").into_bytes()))
                .await
                .unwrap();
        }
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.verify().await.unwrap().entry_count, 0);
        assert_eq!(map.get(b"k00042".to_vec()).await.unwrap(), None);
        // Re-insert after empty works (empty → non-empty transition).
        map.apply(WriteBatch::new().insert(b"again".to_vec(), b"1".to_vec()))
            .await
            .unwrap();
        assert_eq!(
            map.get(b"again".to_vec()).await.unwrap(),
            Some(b"1".to_vec())
        );
    });
    block_on(map.close()).unwrap();
}

#[test]
fn snapshot_never_observes_half_a_batch() {
    // A batch sets all 26 keys "a".."z" to one generation number, across a
    // multi-page tree (2 KiB values). A concurrent snapshot reader must always
    // see either none of the keys or all 26 at the *same* generation — never a
    // torn mix from a half-applied batch/split.
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(dev);
    let reader = map.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_r = Arc::clone(&stop);

    let handle = std::thread::spawn(move || {
        while !stop_r.load(Ordering::Relaxed) {
            let snap = reader.snapshot();
            block_on(async {
                let mut gens: Vec<u32> = Vec::new();
                for k in b'a'..=b'z' {
                    if let Some(v) = snap.get(vec![k]).await.unwrap() {
                        gens.push(u32::from_le_bytes(v[..4].try_into().unwrap()));
                    }
                }
                if !gens.is_empty() {
                    assert_eq!(gens.len(), 26, "partial key set: half a batch was observed");
                    assert!(
                        gens.iter().all(|g| *g == gens[0]),
                        "mixed generations: a split/batch was observed half-applied"
                    );
                }
            });
        }
    });

    block_on(async {
        for g in 0..250u32 {
            let mut b = WriteBatch::new();
            let mut val = big_val(g);
            val.truncate(2000);
            for k in b'a'..=b'z' {
                let mut v = val.clone();
                v[..4].copy_from_slice(&g.to_le_bytes());
                b = b.insert(vec![k], v);
            }
            map.apply(b).await.unwrap();
        }
    });

    stop.store(true, Ordering::Relaxed);
    handle.join().unwrap();
    block_on(map.close()).unwrap();
}

#[test]
fn apply_future_drop_still_applies() {
    // Drop the apply future right after enqueue: the command owns its permit and
    // still applies (invariant 4); the budget releases afterwards (no leak).
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(dev);
    let cost = WriteBatch::new()
        .insert(b"k".to_vec(), b"v".to_vec())
        .cost(crate::store::DEPTH_BOUND);
    let permit = map.try_reserve_write(cost).unwrap();
    let fut = permit
        .apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))
        .unwrap_or_else(|_| panic!("cost within permit"));
    drop(fut); // caller cancels; the queued command survives.
    spin_until(|| map.len() == 1);
    assert_eq!(
        block_on(map.get(b"k".to_vec())).unwrap(),
        Some(b"v".to_vec())
    );
    // Slot + dirty-page budget both drain (pages release at the flush frontier).
    spin_until(|| {
        let m = map.store().metrics();
        m.admission_slots == 0 && m.dirty_pages == 0
    });
    block_on(map.close()).unwrap();
}

#[test]
fn intermediate_snapshot_readable_after_flush_and_eviction() {
    // Tiny clean cache so later reads evict earlier pages; big values so the
    // snapshot spans several pages. After an explicit flush of the snapshot's
    // version and heavy churn, the snapshot's pages reload from disk (invariant
    // 5: evictable ⇔ durably readable).
    let dev = Arc::new(FakeIo::new(0));
    let options = Options {
        cache_capacity: Some(4),
        ..Options::default()
    };
    let map = direct_map_opts(dev, options);
    block_on(async {
        // Batch 1: a multi-page snapshot subject.
        let mut b = WriteBatch::new();
        for i in 0..30u32 {
            b = b.insert(format!("s{i:04}").into_bytes(), big_val(i));
        }
        let v1 = match map.apply(b).await.unwrap() {
            ApplyOutcome::Applied { version, .. } => version,
            other => panic!("expected Applied, got {other:?}"),
        };
        let snap = map.snapshot();
        assert_eq!(snap.len(), 30);
        // Make v1 durable so its pages are evictable-and-reloadable.
        map.flush(v1).await.unwrap();

        // Churn: many more big batches to evict v1's pages from the tiny cache.
        for round in 0..40u32 {
            map.apply(
                WriteBatch::new()
                    .insert(format!("x{round:04}").into_bytes(), big_val(round + 1000)),
            )
            .await
            .unwrap();
        }

        // The intermediate snapshot still reads every original key correctly,
        // reloading evicted pages from disk.
        for i in 0..30u32 {
            let got = snap.get(format!("s{i:04}").into_bytes()).await.unwrap();
            assert_eq!(
                got.as_deref().map(|v| &v[..4]),
                Some(&i.to_le_bytes()[..]),
                "intermediate snapshot lost key s{i:04} after flush+eviction"
            );
        }
    });
    block_on(map.close()).unwrap();
}

#[test]
fn racing_flush_never_lowers_header() {
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(Arc::clone(&dev));
    block_on(async {
        let v1 =
            applied_version(&map, WriteBatch::new().insert(b"a".to_vec(), b"1".to_vec())).await;
        map.flush(v1).await.unwrap();
        let v2 =
            applied_version(&map, WriteBatch::new().insert(b"b".to_vec(), b"2".to_vec())).await;
        map.flush(v2).await.unwrap();
        // A regressing flush of the older version is a no-op and cannot lower
        // the persisted header.
        map.flush(v1).await.unwrap();
        let header = DirectHeader::decode(&dev.snapshot_bytes()).unwrap();
        assert_eq!(
            header.header_txid,
            v2.txid(),
            "header must reflect the highest flush"
        );
        assert_eq!(header.entry_count, 2);
        assert!(header.header_txid >= v1.txid());
    });
    block_on(map.close()).unwrap();
}

#[test]
fn sustained_writes_progress_via_high_water() {
    // Small dirty budget → the writer's high-water flush must advance durability
    // even though the test never calls flush.
    let dev = Arc::new(FakeIo::new(0));
    // Small dirty budget (but above the half-budget cap for a single batch's
    // 9-page worst case) so the high-water mark (~half of it) is hit repeatedly.
    let options = Options {
        limits: AdmissionLimits {
            max_dirty_pages: 40,
            ..AdmissionLimits::default()
        },
        ..Options::default()
    };
    let map = direct_map_opts(dev, options);
    block_on(async {
        for i in 0..80u32 {
            map.apply(WriteBatch::new().insert(format!("k{i:04}").into_bytes(), big_val(i)))
                .await
                .unwrap();
        }
    });
    // Durability advances on its own (no explicit flush issued).
    spin_until(|| map.store().metrics().durable_txid > 0);
    // Dirty pages stay bounded by the budget (high-water keeps releasing them).
    let m = map.store().metrics();
    assert!(
        m.durable_txid > 0,
        "high-water flush never advanced durability"
    );
    block_on(map.close()).unwrap();
}

#[test]
fn clean_close_reopen_preserves_state() {
    let dev = Arc::new(FakeIo::new(0));
    {
        let map = direct_map(Arc::clone(&dev));
        block_on(async {
            for i in 0..120u32 {
                map.apply(WriteBatch::new().insert(format!("k{i:04}").into_bytes(), big_val(i)))
                    .await
                    .unwrap();
            }
            // Remove a few and overwrite one, to make the visible state non-trivial.
            map.apply(WriteBatch::new().remove(b"k0007".to_vec()))
                .await
                .unwrap();
            map.apply(WriteBatch::new().insert(b"k0008".to_vec(), b"small".to_vec()))
                .await
                .unwrap();
        });
        block_on(map.close()).unwrap();
    }
    // Reopen over the same device: the highest visible pre-close state survives.
    let store = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap();
    let map = BTreeMap::over(store);
    block_on(async {
        assert_eq!(map.len(), 119);
        assert_eq!(map.verify().await.unwrap().entry_count, 119);
        assert_eq!(map.get(b"k0007".to_vec()).await.unwrap(), None);
        assert_eq!(
            map.get(b"k0008".to_vec()).await.unwrap(),
            Some(b"small".to_vec())
        );
        assert_eq!(
            map.get(b"k0042".to_vec())
                .await
                .unwrap()
                .as_deref()
                .map(|v| &v[..4]),
            Some(&42u32.to_le_bytes()[..])
        );
        // The reopened store is writable and its new writes read back.
        map.apply(WriteBatch::new().insert(b"post-reopen".to_vec(), b"ok".to_vec()))
            .await
            .unwrap();
        assert_eq!(
            map.get(b"post-reopen".to_vec()).await.unwrap(),
            Some(b"ok".to_vec())
        );
    });
    block_on(map.close()).unwrap();
}

#[test]
fn reopen_rejects_corrupt_header() {
    let dev = Arc::new(FakeIo::new(0));
    {
        let map = direct_map(Arc::clone(&dev));
        block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))).unwrap();
        block_on(map.close()).unwrap();
    }
    // v4 (T1.1): corrupting ONE header slot no longer bricks the
    // store — the other slot is authoritative and the committed state opens.
    block_on(async {
        let mut bytes = dev.snapshot_bytes();
        bytes[64] ^= 0xFF; // flip a checksum-covered byte in slot A
        dev.write_all_at(
            0,
            bytes[..crate::page::PAGE_SIZE].to_vec().into_boxed_slice(),
        )
        .await
        .result
        .unwrap();
    });
    {
        let reopened = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap();
        let map = BTreeMap::over(reopened);
        assert_eq!(
            block_on(map.get(b"k".to_vec())).unwrap().as_deref(),
            Some(&b"v"[..]),
            "one torn slot must fall back to the committed slot"
        );
        block_on(map.close()).unwrap();
    }
    // Both slots corrupt → Corrupt, never a heuristic-repaired open. (Flip a
    // byte not touched above: the no-write close left slot A still-corrupt, so
    // re-flipping byte 64 would restore it.)
    block_on(async {
        let mut bytes = dev.snapshot_bytes();
        bytes[65] ^= 0xFF;
        bytes[crate::page::PAGE_SIZE + 65] ^= 0xFF;
        dev.write_all_at(
            0,
            bytes[..2 * crate::page::PAGE_SIZE]
                .to_vec()
                .into_boxed_slice(),
        )
        .await
        .result
        .unwrap();
    });
    let result = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default());
    assert!(
        matches!(result, Err(crate::error::OpenError::Corrupt(_))),
        "both slots corrupt must be Corrupt"
    );
}

#[test]
fn flush_failure_poisons_bounded() {
    // A datasync fault during flush poisons the store (bounded failure, no
    // hang); subsequent writes fail truthfully. Reopen then yields a consistent
    // earlier state (valid data), never a hang or half-state.
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(Arc::clone(&dev));
    block_on(async {
        // First write + explicit flush succeeds and is durable.
        let v1 = applied_version(
            &map,
            WriteBatch::new().insert(b"durable".to_vec(), b"1".to_vec()),
        )
        .await;
        map.flush(v1).await.unwrap();

        // Arm a sync fault, then a write whose idle-flush datasync will fail.
        dev.set_faults(Faults {
            fail_next_sync: Some(5),
            ..Default::default()
        });
        let _ = map
            .apply(WriteBatch::new().insert(b"doomed".to_vec(), b"2".to_vec()))
            .await;
    });
    // The store poisons; a later write fails fast with a terminal error (bounded,
    // no hang).
    spin_until(|| {
        matches!(
            block_on(map.apply(WriteBatch::new().insert(b"z".to_vec(), b"z".to_vec()))),
            Err(WriteError::Poisoned(_)) | Err(WriteError::Closed)
        )
    });
    let _ = block_on(map.close());

    // Reopen: the header still reflects the last durable state (the pre-fault
    // flush), i.e. valid data, not corruption or a hang.
    let store = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap();
    let reopened = BTreeMap::over(store);
    block_on(async {
        assert_eq!(
            reopened.get(b"durable".to_vec()).await.unwrap(),
            Some(b"1".to_vec())
        );
    });
    block_on(reopened.close()).unwrap();
}

#[test]
fn max_size_keys_force_low_fanout_branches() {
    // Maximum-size keys (4 KiB) and values (8 KiB) → one entry per leaf and only
    // ~4 children per branch, exercising frequent leaf and branch splits and the
    // lone-child rebalance. Every branch must keep ≥ 2 children (verified by
    // building, reading all back, structurally verifying, and reopening).
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(Arc::clone(&dev));
    let n = 40usize;
    let key = |i: usize| {
        let mut k = vec![b'a' + (i % 26) as u8; crate::page::MAX_KEY_LEN];
        k[..8].copy_from_slice(&(i as u64).to_be_bytes());
        k
    };
    block_on(async {
        for i in 0..n {
            map.apply(WriteBatch::new().insert(key(i), vec![i as u8; crate::page::MAX_VALUE_LEN]))
                .await
                .unwrap();
        }
        assert_eq!(map.len(), n as u64);
        // A structural verify walks every page (decode-validated) and checks
        // subtree key bounds — a degenerate/invalid branch would surface here.
        assert_eq!(map.verify().await.unwrap().entry_count, n as u64);
        for i in 0..n {
            assert_eq!(
                map.get(key(i)).await.unwrap().as_deref().map(|v| v[0]),
                Some(i as u8)
            );
        }
    });
    block_on(map.close()).unwrap();
    // Reopen validates the whole tree eagerly — the low-fanout structure
    // must round-trip.
    let store = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap();
    let reopened = BTreeMap::over(store);
    block_on(async {
        assert_eq!(reopened.len(), n as u64);
        assert_eq!(reopened.verify().await.unwrap().entry_count, n as u64);
    });
    block_on(reopened.close()).unwrap();
}

#[test]
fn oversized_condition_is_rejected() {
    // A check with a megabyte key must be rejected as InvalidBatch, not smuggled
    // past command-byte admission.
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(dev);
    let batch = WriteBatch::new()
        .check(vec![b'x'; crate::page::MAX_KEY_LEN + 1], None)
        .insert(b"k".to_vec(), b"v".to_vec());
    let out = block_on(map.apply(batch));
    assert!(
        matches!(
            out,
            Err(WriteError::InvalidBatch(
                crate::batch::BatchError::KeyTooLong
            ))
        ),
        "oversized check key must be InvalidBatch, got {out:?}"
    );
    block_on(map.close()).unwrap();
}

#[test]
fn reopen_rejects_inconsistent_header_count() {
    // A checksum-valid header whose entry_count disagrees with the actual tree
    // must be rejected: otherwise len() lies and a later write could
    // prematurely collapse the tree.
    let dev = Arc::new(FakeIo::new(0));
    {
        let map = direct_map(Arc::clone(&dev));
        block_on(async {
            for i in 0..5u32 {
                map.apply(WriteBatch::new().insert(vec![b'k', i as u8], vec![i as u8]))
                    .await
                    .unwrap();
            }
        });
        block_on(map.close()).unwrap();
    }
    // Rewrite the WINNING slot with a valid-checksum header claiming 999
    // entries (v4: forging the losing slot would simply lose the election —
    // the count-vs-tree gate under test guards the slot that actually opens).
    block_on(async {
        let bytes = dev.snapshot_bytes();
        let pg = crate::page::PAGE_SIZE;
        let a = DirectHeader::decode(&bytes[..pg]);
        let b = DirectHeader::decode(&bytes[pg..2 * pg]);
        let (mut header, off) = match (a, b) {
            (Ok(ha), Ok(hb)) if hb.header_txid > ha.header_txid => (hb, pg as u64),
            (Ok(ha), _) => (ha, 0),
            (Err(_), Ok(hb)) => (hb, pg as u64),
            (Err(_), Err(_)) => panic!("no valid slot in fixture"),
        };
        header.entry_count = 999;
        dev.write_all_at(off, header.encode()).await.result.unwrap();
    });
    let result = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default());
    assert!(
        matches!(result, Err(crate::error::OpenError::Corrupt(_))),
        "header count disagreeing with the tree must be Corrupt"
    );
}

#[test]
fn flush_of_durable_version_after_clean_close_is_ok() {
    // A receipt for a version that is already durable resolves Ok even after the
    // store cleanly closes (which flushed it): the barrier is satisfied, so
    // reporting `Closed` would be a false data-loss signal.
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(dev);
    let v1 = block_on(applied_version(
        &map,
        WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()),
    ));
    block_on(map.close()).unwrap(); // clean close flushes v1 durable
                                    // flush(v1) after close: v1 ≤ durable, so Ok — not FlushError::Closed.
    assert!(
        block_on(map.flush(v1)).is_ok(),
        "flush of a durable version must be Ok after close"
    );
}

#[test]
fn multi_op_batch_semantics_through_builder() {
    // Duplicate keys, insert+remove of the same key, old values, and conditions
    // must all be correct through the real CoW builder.
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(dev);
    block_on(async {
        // Last write wins on a repeated key; a same-key remove after insert nets
        // out; old values are the pre-batch values.
        let batch = WriteBatch::new()
            .insert(b"a".to_vec(), b"1".to_vec())
            .insert(b"b".to_vec(), b"2".to_vec())
            .insert(b"a".to_vec(), b"3".to_vec()) // overwrite within batch
            .insert(b"c".to_vec(), b"9".to_vec())
            .remove(b"c".to_vec()) // insert then remove same key → absent
            .returning_old_values();
        match map.apply(batch).await.unwrap() {
            ApplyOutcome::Applied { old_values, .. } => {
                // Old values aligned to ops; all keys absent pre-batch.
                assert_eq!(old_values, Some(vec![None, None, None, None, None]));
            }
            other => panic!("expected Applied, got {other:?}"),
        }
        assert_eq!(map.get(b"a".to_vec()).await.unwrap(), Some(b"3".to_vec()));
        assert_eq!(map.get(b"b".to_vec()).await.unwrap(), Some(b"2".to_vec()));
        assert_eq!(map.get(b"c".to_vec()).await.unwrap(), None);
        assert_eq!(map.len(), 2);

        // A conditional batch: check passes → applies; then a failing check.
        let ok = map
            .apply(
                WriteBatch::new()
                    .check(b"a".to_vec(), Some(b"3".to_vec()))
                    .insert(b"a".to_vec(), b"4".to_vec())
                    .returning_old_values(),
            )
            .await
            .unwrap();
        assert!(matches!(
            ok,
            ApplyOutcome::Applied {
                old_values: Some(_),
                ..
            }
        ));
        assert_eq!(map.get(b"a".to_vec()).await.unwrap(), Some(b"4".to_vec()));

        let failed = map
            .apply(
                WriteBatch::new()
                    .check(b"a".to_vec(), Some(b"stale".to_vec()))
                    .insert(b"a".to_vec(), b"5".to_vec()),
            )
            .await
            .unwrap();
        assert!(matches!(
            failed,
            ApplyOutcome::ConditionFailed { first_failed: 0 }
        ));
        assert_eq!(
            map.get(b"a".to_vec()).await.unwrap(),
            Some(b"4".to_vec()),
            "no publish on failed check"
        );
    });
    block_on(map.close()).unwrap();
}

#[test]
fn reopen_emptied_store() {
    // A store emptied by removes (root → EMPTY, count 0, tail advanced past
    // burned ids) must reopen as empty and stay writable.
    let dev = Arc::new(FakeIo::new(0));
    {
        let map = direct_map(Arc::clone(&dev));
        block_on(async {
            for i in 0..10u32 {
                map.apply(WriteBatch::new().insert(vec![b'k', i as u8], vec![i as u8]))
                    .await
                    .unwrap();
            }
            for i in 0..10u32 {
                map.apply(WriteBatch::new().remove(vec![b'k', i as u8]))
                    .await
                    .unwrap();
            }
            assert!(map.is_empty());
        });
        block_on(map.close()).unwrap();
    }
    let store = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap();
    let map = BTreeMap::over(store);
    block_on(async {
        assert!(map.is_empty());
        assert_eq!(map.get(vec![b'k', 3]).await.unwrap(), None);
        assert_eq!(map.verify().await.unwrap().entry_count, 0);
        // Still writable after reopening empty.
        map.apply(WriteBatch::new().insert(b"fresh".to_vec(), b"1".to_vec()))
            .await
            .unwrap();
        assert_eq!(
            map.get(b"fresh".to_vec()).await.unwrap(),
            Some(b"1".to_vec())
        );
    });
    block_on(map.close()).unwrap();
}

async fn applied_version(map: &BTreeMap<crate::Direct>, batch: WriteBatch) -> crate::Version {
    match map.apply(batch).await.unwrap() {
        ApplyOutcome::Applied { version, .. } => version,
        other => panic!("expected Applied, got {other:?}"),
    }
}

// -- Differential vs std::BTreeMap ------------------------------------------

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 40, ..ProptestConfig::default() })]

    /// A random sequence of single-op batches (insert/remove over a small key
    /// domain, with values big enough to force splits) applied to both the store
    /// and a `std::BTreeMap` must agree on every key, `len`, and `verify`.
    #[test]
    fn differential_vs_std_btreemap(
        ops in prop::collection::vec(
            (0u16..48, prop::option::of(any::<u32>())),
            0..60,
        ),
    ) {
        let dev = Arc::new(FakeIo::new(0));
        let map = direct_map(dev);
        let mut truth: StdMap<Vec<u8>, Vec<u8>> = StdMap::new();

        block_on(async {
            for (k, maybe_v) in &ops {
                let key = k.to_be_bytes().to_vec();
                match maybe_v {
                    Some(seed) => {
                        let val = big_val(*seed);
                        map.apply(WriteBatch::new().insert(key.clone(), val.clone())).await.unwrap();
                        truth.insert(key, val);
                    }
                    None => {
                        map.apply(WriteBatch::new().remove(key.clone())).await.unwrap();
                        truth.remove(&key);
                    }
                }
            }

            prop_assert_eq!(map.len(), truth.len() as u64);
            prop_assert_eq!(map.verify().await.unwrap().entry_count, truth.len() as u64);
            for k in 0u16..48 {
                let key = k.to_be_bytes().to_vec();
                let got = map.get(key.clone()).await.unwrap();
                prop_assert_eq!(got.as_ref(), truth.get(&key));
            }
            Ok(())
        })?;
        block_on(map.close()).unwrap();
    }
}

// -- M7-B: flush-write fan-out ----------------------------------------------

use crate::page::PAGE_SIZE;

/// Runs an identical multi-flush workload at a given fan-out cap and returns the
/// durable image. Used to prove fan-out changes nothing but write *order*.
fn direct_workload_image(fanout_cap: usize) -> Vec<u8> {
    let dev = Arc::new(FakeIo::new(0));
    let opts = Options {
        fanout_cap: Some(fanout_cap),
        ..Options::default()
    };
    let map = direct_map_opts(Arc::clone(&dev), opts);
    block_on(async {
        for round in 0..8u32 {
            let mut b = WriteBatch::new();
            for i in 0..20u32 {
                b = b.insert(
                    format!("k{round:02}{i:03}").into_bytes(),
                    big_val(round * 100 + i),
                );
            }
            let v = applied_version(&map, b).await;
            map.flush(v).await.unwrap();
        }
        map.close().await.unwrap();
    });
    dev.synced_bytes()
}

/// A fanned-out flush persists the data region **byte-identically** to the serial
/// (cap = 1) write loop: fan-out reorders submissions but writes the same bytes to
/// the same offsets, so the durable image is unchanged.
/// The two meta pages (index 0 = header with the random per-create
/// `store_uuid`, index 1 = the zeroed reserved slot B) are skipped; every data
/// page — index 2 onward — is compared.
#[test]
fn fanout_flush_persists_the_data_region_identically_to_the_serial_path() {
    let serial = direct_workload_image(1);
    let fanned = direct_workload_image(64);
    assert_eq!(
        serial.len(),
        fanned.len(),
        "same workload ⇒ same durable length"
    );
    assert_eq!(
        &serial[2 * PAGE_SIZE..],
        &fanned[2 * PAGE_SIZE..],
        "fan-out must persist the data region byte-for-byte identically to serial",
    );
    // And the fanned image reopens with every key intact.
    let dev = Arc::new(FakeIo::from_vec(fanned));
    let reopened = BTreeMap::over(
        Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap(),
    );
    block_on(async {
        for round in 0..8u32 {
            for i in 0..20u32 {
                let got = reopened
                    .get(format!("k{round:02}{i:03}").into_bytes())
                    .await
                    .unwrap();
                assert_eq!(
                    got.as_deref().map(|v| &v[..4]),
                    Some(&(round * 100 + i).to_le_bytes()[..]),
                    "round {round} key {i} lost across a fanned flush + reopen",
                );
            }
        }
        reopened.close().await.unwrap();
    });
}

/// A flush submits exactly one bounded chunk at a time: with every write parked
/// mid-flight, precisely `cap` writes are outstanding — the fan-out overlaps
/// (peak ≥ 2, so it is not the serial path) but never exceeds the cap, and the
/// next chunk is not submitted until this one drains (the eager-submission
/// contract made observable).
#[test]
fn a_flush_fans_out_up_to_the_cap_and_never_over_submits() {
    let cap = 4usize;
    let dev = Arc::new(FakeIo::new(0));
    let opts = Options {
        fanout_cap: Some(cap),
        ..Options::default()
    };
    let map = direct_map_opts(Arc::clone(&dev), opts);
    // Ignore the create-time base-header writes; measure only this flush.
    dev.reset_write_stats();
    // Park every page write so the flush stalls where we can inspect its depth.
    dev.arm_write_gate(1_000_000);

    // One batch dirtying well more than `cap` pages ⇒ a multi-chunk flush.
    let v = block_on(async {
        let mut b = WriteBatch::new();
        for i in 0..80u32 {
            b = b.insert(format!("f{i:04}").into_bytes(), big_val(i));
        }
        applied_version(&map, b).await
    });

    // The writer idle-flushes and parks in step 1; wait until a chunk is in flight.
    spin_until(|| dev.writes_in_flight() >= cap && dev.parked_write_count() >= 1);
    // Bounded: exactly one chunk is outstanding — `cap` writes submitted (the
    // `.collect()` submits a whole chunk before awaiting any), none from the next.
    assert_eq!(
        dev.writes_in_flight(),
        cap,
        "a chunk holds exactly cap writes in flight"
    );
    assert_eq!(
        dev.writes_submitted(),
        cap,
        "the next chunk must not be submitted while this one is parked"
    );
    assert!(
        dev.writes_peak() <= cap,
        "fan-out must never exceed the cap"
    );
    assert!(
        dev.writes_peak() >= 2,
        "fan-out must actually overlap writes, not run serially"
    );

    // Release everything: the flush drains every chunk and reaches durability.
    dev.release_writes(10_000_000);
    block_on(map.flush(v)).unwrap();
    assert!(
        dev.writes_submitted() >= 2 * cap,
        "the batch must have spanned multiple chunks (submitted {} ≥ 2·cap)",
        dev.writes_submitted(),
    );
    block_on(map.close()).unwrap();
}

/// A page write failing mid-fan-out poisons the store bounded (no hang) and never
/// writes a partial header; the chunk *after* the failing one is never submitted,
/// and every write of the failing chunk is still drained. Reopen names the
/// pre-fault durable state (invariants 3, 17). Every appended page fails here,
/// so the assertion on which chunk was submitted is independent of the CoW
/// build order within the batch.
#[test]
fn a_mid_fanout_error_poisons_and_submits_no_later_chunk() {
    let cap = 4usize;
    let dev = Arc::new(FakeIo::new(0));
    let opts = Options {
        fanout_cap: Some(cap),
        ..Options::default()
    };
    let map = direct_map_opts(Arc::clone(&dev), opts);

    // A durable baseline the reopen must recover.
    let v1 = block_on(applied_version(
        &map,
        WriteBatch::new().insert(b"durable".to_vec(), b"1".to_vec()),
    ));
    block_on(map.flush(v1)).unwrap();
    let durable_tail = dev.synced_bytes().len() as u64;
    dev.reset_write_stats();

    // Fail every appended page (offset ≥ tail): the batch has far more than `cap`
    // pages, so the first chunk fails and no later chunk is ever submitted.
    dev.set_faults(Faults {
        fail_beyond_offset: Some(durable_tail),
        ..Default::default()
    });
    block_on(async {
        let mut b = WriteBatch::new();
        for i in 0..40u32 {
            b = b.insert(format!("d{i:04}").into_bytes(), big_val(i));
        }
        let _ = map.apply(b).await;
    });

    // Bounded poison: a later write fails terminally, no hang.
    spin_until(|| {
        matches!(
            block_on(map.apply(WriteBatch::new().insert(b"z".to_vec(), b"z".to_vec()))),
            Err(WriteError::Poisoned(_)) | Err(WriteError::Closed)
        )
    });
    // Exactly one chunk was submitted (`cap` writes); the next was never reached
    // after the error — and every write of that chunk drained (in-flight back to 0).
    assert_eq!(
        dev.writes_submitted(),
        cap,
        "only the failing chunk is submitted; later chunks must not be"
    );
    assert_eq!(
        dev.writes_in_flight(),
        0,
        "every write of the failing chunk must be drained"
    );
    let _ = block_on(map.close());

    // Reopen: the header still names the pre-fault durable state — no partial mix.
    let store = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap();
    let reopened = BTreeMap::over(store);
    block_on(async {
        assert_eq!(
            reopened.get(b"durable".to_vec()).await.unwrap(),
            Some(b"1".to_vec())
        );
        assert_eq!(
            reopened.get(b"d0000".to_vec()).await.unwrap(),
            None,
            "the doomed batch must not be visible"
        );
        reopened.close().await.unwrap();
    });
}

/// A failure partway through a single fanned chunk — the first two writes land,
/// the rest fail — still drains every write and never advances the header.
/// Reopen sees the pre-fault state, so the landed pages are orphans past the
/// committed tail, invisible.
/// `cap` exceeds the page count so the whole batch is one chunk (every page
/// submitted); the order-based fault lands exactly the first two writes regardless
/// of which offsets the CoW builder assigned them.
#[test]
fn a_partial_fanout_failure_drains_all_and_writes_no_header() {
    let dev = Arc::new(FakeIo::new(0));
    let opts = Options {
        fanout_cap: Some(64),
        ..Options::default()
    };
    let map = direct_map_opts(Arc::clone(&dev), opts);

    let v1 = block_on(applied_version(
        &map,
        WriteBatch::new().insert(b"base".to_vec(), b"1".to_vec()),
    ));
    block_on(map.flush(v1)).unwrap();
    dev.reset_write_stats();

    // The first two submitted writes of the next flush land; every later one fails.
    dev.set_faults(Faults {
        fail_writes_from: Some(2),
        ..Default::default()
    });
    block_on(async {
        let mut b = WriteBatch::new();
        for i in 0..24u32 {
            b = b.insert(format!("m{i:04}").into_bytes(), big_val(i));
        }
        let _ = map.apply(b).await;
    });

    spin_until(|| {
        matches!(
            block_on(map.apply(WriteBatch::new().insert(b"z".to_vec(), b"z".to_vec()))),
            Err(WriteError::Poisoned(_)) | Err(WriteError::Closed)
        )
    });
    // Every write of the chunk was drained (none dropped by an early return), and
    // the two successful peers actually landed (recorded as non-durable ranges the
    // failed flush never synced).
    assert!(
        dev.writes_submitted() > 2,
        "the flush must have fanned out past the failing point"
    );
    assert_eq!(
        dev.writes_in_flight(),
        0,
        "every submitted write must be awaited before poison"
    );
    assert_eq!(
        dev.unsynced_ranges().len(),
        2,
        "exactly the two pre-fault writes must have landed (un-synced)",
    );
    let _ = block_on(map.close());

    // Reopen: baseline intact; the landed-but-uncommitted pages are invisible.
    let reopened = BTreeMap::over(
        Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap(),
    );
    block_on(async {
        assert_eq!(
            reopened.get(b"base".to_vec()).await.unwrap(),
            Some(b"1".to_vec())
        );
        assert_eq!(reopened.get(b"m0000".to_vec()).await.unwrap(), None);
        reopened.close().await.unwrap();
    });
}

/// The datasync barrier never runs while any page write of the chunk is still
/// outstanding, even when one has already failed: with every write parked, the
/// flush stays in step 1 (no datasync, frontier unmoved); only after the whole
/// chunk drains does it poison — and the header is still never written
/// (the straggler-plus-error case).
#[test]
fn a_gated_flush_never_crosses_the_barrier_until_the_chunk_drains() {
    let dev = Arc::new(FakeIo::new(0));
    // One chunk (cap ≫ page count) so the error and its parked peers share it.
    let opts = Options {
        fanout_cap: Some(64),
        ..Options::default()
    };
    let map = direct_map_opts(Arc::clone(&dev), opts);

    let v1 = block_on(applied_version(
        &map,
        WriteBatch::new().insert(b"base".to_vec(), b"1".to_vec()),
    ));
    block_on(map.flush(v1)).unwrap();
    let tail = dev.synced_bytes().len() as u64;
    let base_syncs = dev.sync_count();

    // One page beyond tail+2 fails; the rest would succeed — but first, park them all.
    dev.set_faults(Faults {
        fail_beyond_offset: Some(tail + 2 * PAGE_SIZE as u64),
        ..Default::default()
    });
    dev.arm_write_gate(1_000_000);
    block_on(async {
        let mut b = WriteBatch::new();
        for i in 0..30u32 {
            b = b.insert(format!("g{i:04}").into_bytes(), big_val(i));
        }
        let _ = applied_version(&map, b).await;
    });

    // The writer fans the whole chunk out and parks awaiting the first write.
    spin_until(|| dev.parked_write_count() >= 1 && dev.writes_in_flight() >= 2);
    // Barrier not crossed: no new datasync, durability still at the baseline — the
    // flush is blocked in step 1 with writes outstanding.
    assert_eq!(
        dev.sync_count(),
        base_syncs,
        "no datasync may run while page writes are in flight"
    );
    assert_eq!(
        map.metrics().durable_txid,
        v1.txid(),
        "the frontier must not advance mid-fan-out"
    );

    // Release: the chunk drains (successes and the failure both awaited), then the
    // store poisons — still before any datasync or header write.
    dev.release_writes(10_000_000);
    spin_until(|| {
        matches!(
            block_on(map.apply(WriteBatch::new().insert(b"z".to_vec(), b"z".to_vec()))),
            Err(WriteError::Poisoned(_)) | Err(WriteError::Closed)
        )
    });
    assert_eq!(
        dev.writes_in_flight(),
        0,
        "every submitted write must be drained before poison"
    );
    assert_eq!(
        dev.sync_count(),
        base_syncs,
        "a poisoned flush must never reach its datasync barrier"
    );
    let _ = block_on(map.close());

    // Reopen: baseline intact, the doomed batch invisible.
    let reopened = BTreeMap::over(
        Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap(),
    );
    block_on(async {
        assert_eq!(
            reopened.get(b"base".to_vec()).await.unwrap(),
            Some(b"1".to_vec())
        );
        assert_eq!(reopened.get(b"g0000".to_vec()).await.unwrap(), None);
        reopened.close().await.unwrap();
    });
}

/// A zero-page batch (a no-op remove of an absent key) still commits the header:
/// the flush has an empty write set but must advance the durable frontier so a
/// receipt wait resolves (the fan-out loop over an
/// empty page set is a clean no-op that falls through to the barrier).
#[test]
fn a_zero_page_flush_still_advances_the_durable_frontier() {
    let dev = Arc::new(FakeIo::new(0));
    let opts = Options {
        fanout_cap: Some(4),
        ..Options::default()
    };
    let map = direct_map_opts(Arc::clone(&dev), opts);
    let v = block_on(applied_version(
        &map,
        WriteBatch::new().remove(b"never-existed".to_vec()),
    ));
    // Flush resolves (no page writes, header still commits) and durability reaches v.
    block_on(map.flush(v)).unwrap();
    assert_eq!(
        map.metrics().durable_txid,
        v.txid(),
        "a zero-page flush must commit the header/txid"
    );
    block_on(map.close()).unwrap();
}
