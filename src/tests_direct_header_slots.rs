//! Acceptance for the Direct backend's double-slot header.
//!
//! The crash matrix uses a four-phase oracle for a flush over committed state P
//! producing N: before the stale-slot header write is submitted, only P is
//! recoverable; from submission through an incomplete header sync, the result is
//! P or a complete N, never a hybrid and never older than P; after the header
//! sync, N; and a page-write failure prevents header submission, leaving P.
//!
//! Every reopen runs with `verify_tree_on_open`, so "opens" means the whole map
//! is readable and page-valid, not merely that a header decoded. The alternation
//! tests assert *which* offsets a flush actually wrote — a suite that never
//! checked offsets could pass while page 0 was still overwritten every time.

use crate::direct::{DirectHeader, DirectSlot};
use crate::io::{FakeIo, PageIo, Persist};
use crate::page::PAGE_SIZE;
use crate::store::{Options, Store};
use crate::version::Version;
use crate::{ApplyOutcome, BTreeMap, WriteBatch};
use futures_lite::future::block_on;
use std::sync::Arc;

fn as_dev(dev: &Arc<FakeIo>) -> Arc<dyn PageIo> {
    Arc::clone(dev) as Arc<dyn PageIo>
}

fn direct_map(dev: Arc<FakeIo>) -> BTreeMap<crate::Direct> {
    let store = Store::<crate::Direct>::create_with_io(dev, Options::default()).unwrap();
    BTreeMap::over(store)
}

/// Applies `{key -> value}` and flushes it durable, returning the version.
fn put_flushed(map: &BTreeMap<crate::Direct>, key: &[u8], value: &[u8]) -> Version {
    let out = block_on(map.apply(WriteBatch::new().insert(key.to_vec(), value.to_vec()))).unwrap();
    let ApplyOutcome::Applied { version, .. } = out else {
        panic!("apply failed")
    };
    block_on(map.flush(version)).unwrap();
    version
}

type SlotDecode = Result<DirectHeader, crate::error::OpenError>;
type Decision = (&'static str, fn(usize) -> Persist);

/// Decodes both slots of a device image: `(slot A, slot B)`.
fn slots(bytes: &[u8]) -> (SlotDecode, SlotDecode) {
    (
        DirectHeader::decode(&bytes[..PAGE_SIZE]),
        DirectHeader::decode(&bytes[PAGE_SIZE..2 * PAGE_SIZE]),
    )
}

/// Opens a crash image (default options: full tree verification) and returns
/// the value of `key`, or the open error.
fn open_and_get(img: Vec<u8>, key: &[u8]) -> Result<Option<Vec<u8>>, crate::error::OpenError> {
    let dev = Arc::new(FakeIo::from_vec(img));
    let store = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default())?;
    let map = BTreeMap::over(store);
    let got = block_on(map.get(key.to_vec())).unwrap();
    block_on(map.close()).unwrap();
    Ok(got)
}

/// Successive flushes write ALTERNATING slot offsets, the two slots always hold
/// consecutive txids, and the winner flips parity — asserted from the device
/// bytes, not the coordinator's word for it.
#[test]
fn flushes_alternate_slots_with_consecutive_txids() {
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(Arc::clone(&dev));

    // Creation: both slots identical at txid 0 (the legal tie, A canonical).
    let (a, b) = slots(&dev.snapshot_bytes());
    let (a, b) = (a.unwrap(), b.unwrap());
    assert_eq!((a.header_txid, b.header_txid), (0, 0));

    for k in 1..=6u64 {
        put_flushed(&map, format!("k{k}").as_bytes(), b"v");
        let (a, b) = slots(&dev.snapshot_bytes());
        let (a, b) = (a.unwrap(), b.unwrap());
        // Odd flush → B holds the new txid; even flush → A. Never page 0 twice.
        let (winner, loser) = if k % 2 == 1 { (b, a) } else { (a, b) };
        assert_eq!(winner.header_txid, k, "flush {k} winner txid");
        assert_eq!(loser.header_txid, k - 1, "flush {k} stale txid");
    }
    block_on(map.close()).unwrap();
}

/// A reopen resumes alternation against the RECOVERED stale slot: the next
/// flush must overwrite the open-time loser, never the winner.
#[test]
fn reopen_resumes_alternation_at_the_stale_slot() {
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(Arc::clone(&dev));
    put_flushed(&map, b"k1", b"v1"); // winner B (txid 1), stale A (txid 0)
    block_on(map.close()).unwrap();

    let store = Store::<crate::Direct>::open_over(as_dev(&dev), Options::default()).unwrap();
    let map = BTreeMap::over(store);
    put_flushed(&map, b"k2", b"v2"); // must land in A
    let (a, b) = slots(&dev.snapshot_bytes());
    let (a, b) = (a.unwrap(), b.unwrap());
    assert!(
        a.header_txid > b.header_txid,
        "the post-reopen flush must target the recovered stale slot (A), \
         got A txid {} vs B txid {}",
        a.header_txid,
        b.header_txid
    );
    block_on(map.close()).unwrap();
}

/// Phases 1/2/3 of the oracle, driven by the sync gate: park the flush at each
/// durability boundary, take crash images with per-granule Old/New/Garble
/// decisions, and assert every image reopens to exactly P or complete-N.
#[test]
fn flush_crash_matrix_previous_or_complete_new_never_hybrid() {
    // Committed P: k -> "P". N: adds k2 and overwrites k -> "N".
    let build = || {
        let dev = Arc::new(FakeIo::new(0));
        let map = direct_map(Arc::clone(&dev));
        put_flushed(&map, b"k", b"P");
        (dev, map)
    };
    let apply_n = |map: &BTreeMap<crate::Direct>| -> Version {
        let out = block_on(
            map.apply(
                WriteBatch::new()
                    .insert(b"k".to_vec(), b"N".to_vec())
                    .insert(b"k2".to_vec(), b"n2".to_vec()),
            ),
        )
        .unwrap();
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!("apply failed")
        };
        version
    };
    // Every granule decision to sweep at each parked phase. 128-byte blast
    // radius tears within a page; page-sized tears are covered a fortiori.
    let decisions: &[Decision] = &[
        ("old", |_| Persist::Old),
        ("new", |_| Persist::New),
        ("garble", |_| Persist::Garble),
        ("odd-new", |g| {
            if g % 2 == 0 {
                Persist::Old
            } else {
                Persist::New
            }
        }),
        ("odd-garble", |g| {
            if g % 2 == 0 {
                Persist::New
            } else {
                Persist::Garble
            }
        }),
    ];

    // Phase 1: park BEFORE the page fdatasync (header not yet submitted).
    // Every crash image must open to exactly P.
    for (name, decide) in decisions {
        let (dev, map) = build();
        dev.arm_sync_gate(1);
        let version = apply_n(&map);
        let m = map.clone();
        let flusher = std::thread::spawn(move || {
            let _ = block_on(m.flush(version));
        });
        while dev.parked_sync_count() == 0 {
            std::thread::yield_now();
        }
        let img = dev.crash_image(128, decide);
        let got = open_and_get(img, b"k").expect("phase-1 crash must reopen (P)");
        assert_eq!(
            got.as_deref(),
            Some(&b"P"[..]),
            "phase 1 ({name}): header untouched, P must open"
        );
        dev.release_syncs(8);
        flusher.join().unwrap();
        block_on(map.close()).unwrap();
    }

    // Phase 2: pages durable, header WRITTEN to the stale slot, its sync
    // parked. The unsynced range is exactly the stale slot: Old/Garble → P,
    // New → complete N — and mixed per-granule decisions must never produce a
    // hybrid (a torn slot loses the election; a fully-persisted one wins).
    for (name, decide) in decisions {
        let (dev, map) = build();
        dev.arm_sync_gate(2); // gate the page sync AND the header sync
        let version = apply_n(&map);
        let m = map.clone();
        let flusher = std::thread::spawn(move || {
            let _ = block_on(m.flush(version));
        });
        while dev.parked_sync_count() == 0 {
            std::thread::yield_now();
        }
        let syncs_before = dev.sync_count();
        dev.release_syncs(1); // pages durable; flush proceeds to the header write
                              // The released page sync must actually FOLD (sync_count advances)
                              // before the next park can be the header sync — spinning on the parked
                              // count alone can observe the still-parked first sync.
        while dev.sync_count() == syncs_before {
            std::thread::yield_now();
        }
        while dev.parked_sync_count() == 0 {
            std::thread::yield_now();
        }
        // Parked at the header sync: unsynced = the stale-slot header write.
        let img = dev.crash_image(128, decide);
        match open_and_get(img, b"k") {
            Ok(got) => {
                let v = got.as_deref();
                assert!(
                    v == Some(&b"P"[..]) || v == Some(&b"N"[..]),
                    "phase 2 ({name}): P or complete-N, got {v:?}"
                );
                // Complete-N means ALL of N: k2 present iff k reads N.
                // (Re-derive on a fresh image: open_and_get consumed the map.)
                let img2 = dev.crash_image(128, decide);
                let got2 = open_and_get(img2, b"k2").unwrap();
                if v == Some(&b"N"[..]) {
                    assert_eq!(
                        got2.as_deref(),
                        Some(&b"n2"[..]),
                        "phase 2 ({name}): N must be complete, not partial"
                    );
                } else {
                    assert_eq!(got2, None, "phase 2 ({name}): P must not leak pieces of N");
                }
            }
            Err(e) => panic!("phase 2 ({name}): a one-slot tear must never brick the store: {e:?}"),
        }
        dev.release_syncs(8);
        flusher.join().unwrap();
        block_on(map.close()).unwrap();
    }

    // Phase 3: flush completed (header synced) → complete N, always.
    {
        let (dev, map) = build();
        let version = apply_n(&map);
        block_on(map.flush(version)).unwrap();
        let img = dev.crash_image(128, |_| Persist::Old); // nothing unsynced remains
        assert_eq!(
            open_and_get(img, b"k").unwrap().as_deref(),
            Some(&b"N"[..]),
            "phase 3: after the header sync the new state is committed"
        );
        block_on(map.close()).unwrap();
    }
}

/// Phase 4: a page-write failure poisons the flush BEFORE any header
/// submission — the durable image still opens to exactly P.
#[test]
fn page_write_failure_never_submits_the_header() {
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(Arc::clone(&dev));
    put_flushed(&map, b"k", b"P");
    let synced_before = dev.synced_bytes();

    dev.set_faults(crate::io::Faults {
        fail_beyond_offset: Some(synced_before.len() as u64),
        ..Default::default()
    });
    let out = block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"N".to_vec()))).unwrap();
    let ApplyOutcome::Applied { version, .. } = out else {
        panic!("apply failed")
    };
    assert!(
        block_on(map.flush(version)).is_err(),
        "the failing flush must poison, not succeed"
    );
    // The durable image never saw a header submission: it opens to exactly P.
    let got = open_and_get(dev.synced_bytes(), b"k").unwrap();
    assert_eq!(got.as_deref(), Some(&b"P"[..]), "phase 4: P only");
}

/// Creation oracle: a partially-persisted create is absent/unopenable or the
/// complete empty store; the acknowledged create always reopens.
#[test]
fn creation_crash_is_unopenable_or_complete_empty() {
    // Acknowledged create reopens (complete empty store).
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(Arc::clone(&dev));
    block_on(map.close()).unwrap();
    let full = dev.snapshot_bytes();
    assert!(open_and_get(full.clone(), b"k").unwrap().is_none());

    // Nothing persisted: unopenable (empty file), never a junk open.
    assert!(open_and_get(Vec::new(), b"k").is_err());
    // Only slot A persisted (file too short for a v4 store): unopenable.
    assert!(open_and_get(full[..PAGE_SIZE].to_vec(), b"k").is_err());
    // Slot A persisted, slot B garbled: the tie is gone — slot A (valid, txid
    // 0, empty) wins and the complete empty store opens.
    let mut torn = full.clone();
    for b in &mut torn[PAGE_SIZE + 4..PAGE_SIZE + 96] {
        *b = 0xA5;
    }
    assert!(open_and_get(torn, b"k").unwrap().is_none());
}

/// Equal-txid slots are legal ONLY byte-identical; a divergent tie and a
/// cross-store slot pair are Corrupt, never a coin flip.
#[test]
fn divergent_tie_and_uuid_mismatch_are_corrupt() {
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(Arc::clone(&dev));
    block_on(map.close()).unwrap();
    let base = dev.snapshot_bytes();

    // Divergent tie: same txid (0), different logical_tail, valid checksum.
    let mut hdr = DirectHeader::decode(&base[..PAGE_SIZE]).unwrap();
    hdr.logical_tail += PAGE_SIZE as u64;
    let mut img = base.clone();
    img[PAGE_SIZE..2 * PAGE_SIZE].copy_from_slice(&hdr.encode());
    match open_and_get(img, b"k") {
        Err(crate::error::OpenError::Corrupt(what)) => {
            assert!(what.contains("tie"), "unexpected corrupt reason: {what}")
        }
        other => panic!("divergent tie must be Corrupt, got {other:?}"),
    }

    // UUID mismatch with a higher txid: refuse, never adopt the imposter.
    let mut hdr = DirectHeader::decode(&base[..PAGE_SIZE]).unwrap();
    hdr.store_uuid = [0xEE; 16];
    hdr.header_txid = 99;
    let mut img = base.clone();
    img[PAGE_SIZE..2 * PAGE_SIZE].copy_from_slice(&hdr.encode());
    match open_and_get(img, b"k") {
        Err(crate::error::OpenError::Corrupt(what)) => assert!(
            what.contains("different stores"),
            "unexpected corrupt reason: {what}"
        ),
        other => panic!("uuid mismatch must be Corrupt, got {other:?}"),
    }
}

/// Format policy: a v3 store is refused as older, a v5 store as
/// newer — surfaced as `UnsupportedFormat`, never misread as a torn slot.
#[test]
fn cross_format_stores_are_refused_not_repaired() {
    let dev = Arc::new(FakeIo::new(0));
    let map = direct_map(Arc::clone(&dev));
    block_on(map.close()).unwrap();
    let base = dev.snapshot_bytes();

    for (byte, newer) in [(3u8, false), (5u8, true)] {
        let mut img = base.clone();
        // Patch BOTH slots' format bytes and re-checksum, so the refusal is
        // format policy, not a slot-damage fallback.
        for off in [0usize, PAGE_SIZE] {
            img[off + 4] = byte;
            let ck = crc32c::crc32c(&img[off..off + 80]);
            img[off + 80..off + 84].copy_from_slice(&ck.to_le_bytes());
        }
        match open_and_get(img, b"k") {
            Err(crate::error::OpenError::UnsupportedFormat {
                found, newer: n, ..
            }) => {
                assert_eq!((found, n), (byte, newer));
            }
            other => panic!("format {byte} must be UnsupportedFormat, got {other:?}"),
        }
    }
}

/// One-slot loss at EVERY committed flush depth: whichever single slot a crash
/// tears, the store opens to the committed state (winner) or the previous one
/// (when the torn slot WAS the winner — the crash-mid-flush case), never less.
#[test]
fn single_slot_tear_at_every_depth_opens_committed_or_previous() {
    for depth in 1..=4u64 {
        let dev = Arc::new(FakeIo::new(0));
        let map = direct_map(Arc::clone(&dev));
        for k in 1..=depth {
            put_flushed(&map, b"seq", format!("v{k}").as_bytes());
        }
        block_on(map.close()).unwrap();
        let base = dev.snapshot_bytes();
        for slot in [DirectSlot::A, DirectSlot::B] {
            let mut img = base.clone();
            let off = slot.offset() as usize;
            for b in &mut img[off + 8..off + 60] {
                *b = 0x5A;
            }
            let got = open_and_get(img, b"seq")
                .unwrap_or_else(|e| panic!("depth {depth}, torn {slot:?}: must open: {e:?}"));
            let want_now = format!("v{depth}");
            let want_prev = format!("v{}", depth.saturating_sub(1));
            let v = got
                .as_deref()
                .map(|v| String::from_utf8_lossy(v).into_owned());
            let committed_or_prev = v.as_deref() == Some(want_now.as_str())
                || (depth > 1 && v.as_deref() == Some(want_prev.as_str()))
                || (depth == 1 && v.is_none());
            assert!(
                committed_or_prev,
                "depth {depth}, torn {slot:?}: got {v:?}, want {want_now} or previous"
            );
        }
    }
}
