//! Acceptance for the WAL backend: record append, group commit, the
//! `max_wal_bytes` bound, and crash recovery against a write-range crash model.
//!
//! The crash model: [`FakeIo`] tracks the last `fdatasync`'d image
//! (`synced_bytes`) plus the ranges written since (`unsynced_ranges`). A crash
//! image is the synced image optionally extended with a subset — possibly torn —
//! of the unsynced writes, preserving every byte outside a written range
//! (powersafe overwrite), and recovery opens a fresh device over that image.
//!
//! The invariants: an acknowledged commit is synced before its ack, so it is
//! always in the synced image and must always recover; a torn or garbage
//! in-flight record must never be accepted and must never resurrect stale bytes;
//! and recovery itself must be read-only and idempotent.

use crate::batch::{ApplyOutcome, WriteBatch};
use crate::error::{CommitError, OpenError, WriteError};
use crate::io::{FakeIo, Faults, PageIo, Persist};
use crate::map::BTreeMap;
use crate::store::{Options, Store};
use crate::version::Version;
use futures_lite::future::block_on;
use std::sync::Arc;

// -- Helpers ----------------------------------------------------------------

fn opts() -> Options {
    Options::default()
}

fn dev(a: &Arc<FakeIo>) -> Arc<dyn PageIo> {
    Arc::clone(a) as Arc<dyn PageIo>
}

/// A fresh data+WAL device pair.
fn devices() -> (Arc<FakeIo>, Arc<FakeIo>) {
    (Arc::new(FakeIo::new(0)), Arc::new(FakeIo::new(0)))
}

fn create(data: &Arc<FakeIo>, wal: &Arc<FakeIo>, options: Options) -> Store<Wal> {
    Store::<Wal>::create_with_io(dev(data), dev(wal), options).unwrap()
}

use crate::backend::Wal;

/// Applies a single-insert batch and commits it durable, returning the version.
fn put(map: &BTreeMap<Wal>, k: &[u8], v: &[u8]) -> Version {
    block_on(async {
        let out = map
            .apply(WriteBatch::new().insert(k.to_vec(), v.to_vec()))
            .await
            .unwrap();
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!("expected Applied, got {out:?}");
        };
        map.commit(version).await.unwrap();
        version
    })
}

/// Reopens over the **durable** (synced) images of `data`/`wal` — i.e. a clean
/// crash after the last `fdatasync`.
fn reopen_synced(
    data: &Arc<FakeIo>,
    wal: &Arc<FakeIo>,
    options: Options,
) -> Result<Store<Wal>, OpenError> {
    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let w = Arc::new(FakeIo::from_vec(wal.synced_bytes()));
    Store::<Wal>::open_over(dev(&d), dev(&w), options)
}

/// Reopens over a caller-supplied WAL image (crash variant) and the durable data
/// image.
fn reopen_wal_image(
    data: &Arc<FakeIo>,
    wal_image: Vec<u8>,
    options: Options,
) -> Result<Store<Wal>, OpenError> {
    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let w = Arc::new(FakeIo::from_vec(wal_image));
    Store::<Wal>::open_over(dev(&d), dev(&w), options)
}

/// A recovered store plus its concrete data/WAL devices, for inspection.
type OpenedDevs = (Store<Wal>, Arc<FakeIo>, Arc<FakeIo>);

/// Opens over explicit data/WAL images and returns the (concrete) devices so the
/// test can inspect/reuse their post-run durable bytes.
fn open_over_devs(
    data_image: Vec<u8>,
    wal_image: Vec<u8>,
    options: Options,
) -> Result<OpenedDevs, OpenError> {
    let d = Arc::new(FakeIo::from_vec(data_image));
    let w = Arc::new(FakeIo::from_vec(wal_image));
    let s = Store::<Wal>::open_over(dev(&d), dev(&w), options)?;
    Ok((s, d, w))
}

// -- Tests ------------------------------------------------------------------

#[test]
fn create_write_commit_reopen_recovers() {
    let (data, wal) = devices();
    {
        let map = BTreeMap::over(create(&data, &wal, opts()));
        for i in 0..40u32 {
            put(&map, format!("k{i:03}").as_bytes(), &vec![i as u8; 300]);
        }
        block_on(map.apply(WriteBatch::new().remove(b"k005".to_vec()))).unwrap();
        // commit the remove
        let v = block_on(async {
            let out = map
                .apply(WriteBatch::new().insert(b"tail".to_vec(), b"z".to_vec()))
                .await
                .unwrap();
            let ApplyOutcome::Applied { version, .. } = out else {
                panic!()
            };
            version
        });
        block_on(map.commit(v)).unwrap();
    }
    // Recover from the durable images.
    let store = reopen_synced(&data, &wal, opts()).unwrap();
    let map = BTreeMap::over(store);
    block_on(async {
        assert_eq!(
            map.get(b"k000".to_vec()).await.unwrap(),
            Some(vec![0u8; 300])
        );
        assert_eq!(
            map.get(b"k039".to_vec()).await.unwrap(),
            Some(vec![39u8; 300])
        );
        assert_eq!(map.get(b"k005".to_vec()).await.unwrap(), None);
        assert_eq!(
            map.get(b"tail".to_vec()).await.unwrap(),
            Some(b"z".to_vec())
        );
        // Recovery continued the txid; a fresh write appends past the prefix.
        let out = map
            .apply(WriteBatch::new().insert(b"new".to_vec(), b"v".to_vec()))
            .await
            .unwrap();
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!()
        };
        map.commit(version).await.unwrap();
    });
    block_on(map.close()).unwrap();
}

#[test]
fn recovers_last_committed_prefix_at_every_truncation() {
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    // Commit N records one at a time, recording each record's end offset.
    let mut boundaries = vec![crate::wal::WAL_HEADER_LEN as usize];
    let n = 12;
    for i in 0..n {
        put(
            &map,
            format!("key{i:02}").as_bytes(),
            format!("val{i:02}").as_bytes(),
        );
        boundaries.push(wal.synced_bytes().len());
    }
    let full = wal.synced_bytes();

    // For every truncation length, recovery must accept exactly the records
    // fully contained in [0, L): the last complete committed prefix.
    for l in (crate::wal::WAL_HEADER_LEN as usize)..=full.len() {
        let img = full[..l].to_vec();
        let store = reopen_wal_image(&data, img, opts()).unwrap();
        // Expected accepted record count = number of boundaries ≤ l, minus the
        // header boundary.
        let accepted = boundaries.iter().filter(|&&b| b <= l).count() - 1;
        let map2 = BTreeMap::over(store);
        assert_eq!(
            map2.len(),
            accepted as u64,
            "truncation at {l}: expected {accepted} entries"
        );
        // Every accepted key present; the first un-accepted key absent.
        block_on(async {
            for i in 0..accepted {
                let k = format!("key{i:02}");
                assert_eq!(
                    map2.get(k.clone().into_bytes()).await.unwrap(),
                    Some(format!("val{i:02}").into_bytes()),
                    "truncation {l}: key{i:02} must be present"
                );
            }
            if accepted < n {
                let k = format!("key{accepted:02}");
                assert_eq!(map2.get(k.into_bytes()).await.unwrap(), None);
            }
        });
    }
}

#[test]
fn torn_or_garbage_tail_rejected_prefix_survives() {
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    for i in 0..6u32 {
        put(&map, format!("k{i}").as_bytes(), b"v");
    }
    let committed = wal.synced_bytes();

    // Append assorted garbage tails; the committed prefix must always survive and
    // no invalid record may be accepted.
    for tail in [
        vec![0u8; 10],
        vec![0xFFu8; 5000],
        vec![0x4D, 0x50, 0x53, 0x52, 1, 0, 0, 0], // record magic + partial
        vec![7u8; crate::page::PAGE_SIZE * 2],
    ] {
        let mut img = committed.clone();
        img.extend_from_slice(&tail);
        let store = reopen_wal_image(&data, img, opts()).unwrap();
        let map2 = BTreeMap::over(store);
        assert_eq!(map2.len(), 6, "garbage tail must not add or drop entries");
        block_on(async {
            assert_eq!(map2.get(b"k5".to_vec()).await.unwrap(), Some(b"v".to_vec()));
        });
    }
}

#[test]
fn corrupt_middle_record_stops_scan_no_resurrection() {
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    let mut boundaries = vec![crate::wal::WAL_HEADER_LEN as usize];
    for i in 0..8u32 {
        put(&map, format!("k{i}").as_bytes(), b"v");
        boundaries.push(wal.synced_bytes().len());
    }
    let mut img = wal.synced_bytes();
    // Corrupt one byte inside record 4 (0-indexed record 3): between
    // boundaries[3] and boundaries[4]. This must reject records 3.. (the scan
    // stops; later, still-intact records must NOT be resurrected).
    let corrupt_at = boundaries[3] + 20;
    img[corrupt_at] ^= 0xFF;
    let store = reopen_wal_image(&data, img, opts()).unwrap();
    let map2 = BTreeMap::over(store);
    assert_eq!(map2.len(), 3, "scan stops at the first corrupt record");
    block_on(async {
        assert_eq!(map2.get(b"k2".to_vec()).await.unwrap(), Some(b"v".to_vec()));
        assert_eq!(map2.get(b"k3".to_vec()).await.unwrap(), None);
        assert_eq!(map2.get(b"k7".to_vec()).await.unwrap(), None);
    });
}

#[test]
fn shorter_replacement_does_not_resurrect_stale_tail() {
    // Recover a torn tail, commit a *shorter* replacement over the accepted-prefix
    // offset (never truncating), then a second crash: the stale bytes of the long
    // record beyond the shorter replacement never resurrect (invariant 13).
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    put(&map, b"a", b"1"); // R1 — one small page
    let r1_end = wal.synced_bytes().len();
    // R2 — a large multi-key batch spanning many leaf pages (a long record), so a
    // later single-page replacement is genuinely shorter.
    block_on(async {
        let mut b = WriteBatch::new();
        for i in 0..60u32 {
            b = b.insert(format!("b{i:03}").into_bytes(), vec![2u8; 6000]);
        }
        let out = map.apply(b).await.unwrap();
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!()
        };
        map.commit(version).await.unwrap();
    });
    let full = wal.synced_bytes();
    drop(map);

    // A crash leaves R1 durable and R2 physically present but damaged (one flipped
    // byte inside it) → recovery accepts R1 only, with R2's bytes still on disk.
    let mut img = full.clone();
    img[r1_end + 30] ^= 0xFF;
    let (store, _d, w2) = open_over_devs(data.synced_bytes(), img, opts()).unwrap();
    let map2 = BTreeMap::over(store);
    assert_eq!(map2.len(), 1, "recovery stops at R1");

    // Commit a SHORT replacement R2' at R1's end, overwriting the head of stale R2.
    block_on(async {
        let out = map2
            .apply(WriteBatch::new().insert(b"c".to_vec(), b"x".to_vec()))
            .await
            .unwrap();
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!()
        };
        map2.commit(version).await.unwrap();
    });
    let met = map2.metrics();
    assert!(
        met.wal_physical_bytes > met.wal_logical_bytes,
        "stale long-record tail leaves the file physically longer than the logical frontier"
    );
    let final_wal = w2.synced_bytes();
    drop(map2);

    // Second crash + recovery: R1 + R2' ("a","c") present; stale R2's "b" gone.
    let (fs, _, _) = open_over_devs(data.synced_bytes(), final_wal, opts()).unwrap();
    let fm = BTreeMap::over(fs);
    block_on(async {
        assert_eq!(fm.get(b"a".to_vec()).await.unwrap(), Some(b"1".to_vec()));
        assert_eq!(fm.get(b"c".to_vec()).await.unwrap(), Some(b"x".to_vec()));
        assert_eq!(
            fm.get(b"b000".to_vec()).await.unwrap(),
            None,
            "stale record must not resurrect"
        );
        assert_eq!(
            fm.get(b"b059".to_vec()).await.unwrap(),
            None,
            "stale record must not resurrect"
        );
    });
    assert_eq!(fm.len(), 2);
}

#[test]
fn uuid_mismatch_fails_open() {
    let (data1, wal1) = devices();
    let _ = create(&data1, &wal1, opts());
    let (data2, wal2) = devices();
    let _ = create(&data2, &wal2, opts());
    // Pair data from store 1 with WAL from store 2 → UUID mismatch.
    let d = Arc::new(FakeIo::from_vec(data1.synced_bytes()));
    let w = Arc::new(FakeIo::from_vec(wal2.synced_bytes()));
    let r = Store::<Wal>::open_over(dev(&d), dev(&w), opts());
    assert!(
        matches!(r, Err(OpenError::UuidMismatch)),
        "expected UuidMismatch"
    );
}

#[test]
fn store_full_is_pre_publication() {
    // A tiny max_wal_bytes: a batch whose worst case would exceed it fails
    // admission with StoreFull, consumes no txid, and never becomes visible.
    let (data, wal) = devices();
    let mut options = opts();
    options.max_wal_bytes = crate::wal::WAL_HEADER_LEN + crate::page::PAGE_SIZE as u64; // ~1 page room
    let map = BTreeMap::over(create(&data, &wal, options));
    // A big multi-op batch's worst case exceeds one page of WAL room.
    let mut b = WriteBatch::new();
    for i in 0..50u32 {
        b = b.insert(format!("k{i}").into_bytes(), vec![9u8; 500]);
    }
    let r = block_on(map.apply(b));
    assert!(matches!(r, Err(WriteError::StoreFull)), "got {r:?}");
    assert_eq!(map.len(), 0, "StoreFull batch must not be visible");
    assert_eq!(map.metrics().visible_txid, 0, "consumes no txid");
}

#[test]
fn wal_write_error_poisons_permanently() {
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    // Fail every WAL write beyond the header region.
    wal.set_faults(Faults {
        fail_beyond_offset: Some(crate::wal::WAL_HEADER_LEN),
        ..Default::default()
    });
    let r = block_on(async {
        let out = map
            .apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))
            .await?;
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!()
        };
        map.commit(version).await.map_err(|e| match e {
            CommitError::Poisoned(_) => {
                WriteError::Poisoned(crate::error::PoisonReason::Internal("p"))
            }
            CommitError::Closed => WriteError::Closed,
            CommitError::StaleIncarnation => WriteError::Closed,
        })
    });
    assert!(r.is_err(), "a WAL write EIO must surface as an error");
    // Subsequent writes fail terminally (poisoned, never retried into success).
    let r2 = block_on(map.apply(WriteBatch::new().insert(b"x".to_vec(), b"y".to_vec())));
    assert!(matches!(r2, Err(WriteError::Poisoned(_))), "got {r2:?}");
}

#[test]
fn metrics_track_wal_frontiers_and_overlay() {
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    for i in 0..5u32 {
        put(&map, format!("k{i}").as_bytes(), &[i as u8; 200]);
    }
    let m = map.metrics();
    assert!(
        m.wal_logical_bytes > crate::wal::WAL_HEADER_LEN,
        "logical grew"
    );
    assert!(
        m.wal_reserved_bytes >= m.wal_logical_bytes,
        "reserved ≥ logical"
    );
    assert!(m.overlay_pages >= 1, "overlay populated");
    assert!(
        m.wal_physical_bytes >= m.wal_logical_bytes,
        "physical ≥ logical"
    );
    assert_eq!(m.durable_txid, 5);
}

#[test]
fn rejected_record_does_not_leak_page_into_overlay() {
    // Regression: a torn record whose first page section is valid must
    // not leave that page in the accepted overlay.
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    put(&map, b"a", b"1");
    let good = wal.synced_bytes();
    // Add a second committed record, then corrupt its *trailer* only (body +
    // page checksums intact). The record must be rejected and its page(s) not
    // installed.
    put(&map, b"b", b"2");
    let mut img = wal.synced_bytes();
    // Corrupt the last 4 bytes (trailer checksum) of the file.
    let n = img.len();
    img[n - 1] ^= 0xFF;
    let store = reopen_wal_image(&data, img, opts()).unwrap();
    let map2 = BTreeMap::over(store);
    // Only the first record accepted; overlay has exactly its page(s).
    assert_eq!(map2.len(), 1);
    let overlay_after = map2.metrics().overlay_pages;
    // Recover the *good* prefix independently and compare overlay size.
    let clean = reopen_wal_image(&data, good, opts()).unwrap();
    let clean_overlay = BTreeMap::over(clean).metrics().overlay_pages;
    assert_eq!(
        overlay_after, clean_overlay,
        "rejected record must not add overlay pages"
    );
}

#[test]
fn reopen_with_lowered_limit_serves_reads_fails_writes() {
    // Reopening with a max_wal_bytes below the recovered
    // frontier succeeds, serves reads, and fails writes with StoreFull.
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    for i in 0..8u32 {
        put(&map, format!("k{i}").as_bytes(), &vec![i as u8; 400]);
    }
    let logical = map.metrics().wal_logical_bytes;
    drop(map);

    let mut lowered = opts();
    lowered.max_wal_bytes = logical / 2; // below the recovered frontier
    let store = reopen_synced(&data, &wal, lowered).unwrap();
    let map2 = BTreeMap::over(store);
    block_on(async {
        // Reads work.
        assert_eq!(
            map2.get(b"k0".to_vec()).await.unwrap(),
            Some(vec![0u8; 400])
        );
        // Writes fail fast with StoreFull.
        let r = map2
            .apply(WriteBatch::new().insert(b"z".to_vec(), b"z".to_vec()))
            .await;
        assert!(matches!(r, Err(WriteError::StoreFull)), "got {r:?}");
    });
}

#[test]
fn stale_incarnation_receipt_never_resolves_after_reopen() {
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    let v = put(&map, b"k", b"v");
    drop(map);
    let store = reopen_synced(&data, &wal, opts()).unwrap();
    let map2 = BTreeMap::over(store);
    // A receipt from the previous open must not resolve against the new one.
    let r = block_on(map2.commit(v));
    assert!(matches!(r, Err(CommitError::StaleIncarnation)), "got {r:?}");
}

#[test]
fn idempotent_recovery() {
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    for i in 0..7u32 {
        put(&map, format!("k{i}").as_bytes(), b"v");
    }
    let wal_img = wal.synced_bytes();
    let data_img = data.synced_bytes();
    // Recover twice from the same image; identical accepted state, no mutation.
    let a = Store::<Wal>::open_over(
        dev(&Arc::new(FakeIo::from_vec(data_img.clone()))),
        dev(&Arc::new(FakeIo::from_vec(wal_img.clone()))),
        opts(),
    )
    .unwrap();
    let am = BTreeMap::over(a);
    assert_eq!(am.len(), 7);
    let b = Store::<Wal>::open_over(
        dev(&Arc::new(FakeIo::from_vec(data_img))),
        dev(&Arc::new(FakeIo::from_vec(wal_img))),
        opts(),
    )
    .unwrap();
    assert_eq!(BTreeMap::over(b).len(), 7);
    drop(am);
}

#[test]
fn grouped_commit_waiters_wake_at_frontier() {
    // Many pending batches committed as a group; every waiter resolves.
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    block_on(async {
        let mut versions = Vec::new();
        for i in 0..50u32 {
            let out = map
                .apply(WriteBatch::new().insert(format!("k{i}").into_bytes(), b"v".to_vec()))
                .await
                .unwrap();
            let ApplyOutcome::Applied { version, .. } = out else {
                panic!()
            };
            versions.push(version);
        }
        // Commit the highest; all lower txids are durable at/under that frontier.
        for v in &versions {
            map.commit(*v).await.unwrap();
        }
        assert_eq!(map.metrics().durable_txid, 50);
    });
}

// -- Crash model ------------------------------------------------------------

/// Builds a WAL device with record R1 (`a→1`) **durable** (synced) and a second
/// record R2 (`b→2`) written-but-**unsynced** (in flight at crash time), plus the
/// durable data image and R1's end offset. Recovery over crash images of this
/// device must always recover R1; R2 only if it fully persisted.
fn r1_durable_r2_inflight() -> (Vec<u8>, Arc<FakeIo>, usize) {
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    put(&map, b"a", b"1");
    let r1_end = wal.synced_bytes().len();
    put(&map, b"b", b"2");
    let full = wal.synced_bytes();
    let data_img = data.synced_bytes();
    let r1_only = full[..r1_end].to_vec();
    let r2_bytes = full[r1_end..].to_vec();
    drop(map);
    // Fresh WAL device: R1 durable, R2 written but not synced.
    let wal2 = Arc::new(FakeIo::from_vec(r1_only));
    block_on(async {
        wal2.write_all_at(r1_end as u64, r2_bytes.into_boxed_slice())
            .await
            .result
            .unwrap();
    });
    (data_img, wal2, r1_end)
}

#[test]
fn shared_unit_crash_preserves_accepted_record_at_multiple_radii() {
    // Normative crash-model rule: damage anywhere in the in-flight record's write
    // range (which shares a physical unit with the accepted record's tail) must
    // never harm the acknowledged R1, at more than one blast radius.
    for radius in [512usize, 4096] {
        let (data_img, wal, _r1_end) = r1_durable_r2_inflight();
        // Every granule of R2 tears (garbage).
        let img = wal.crash_image(radius, |_| Persist::Garble);
        let (store, _, _) = open_over_devs(data_img, img, opts()).unwrap();
        let map = BTreeMap::over(store);
        assert_eq!(map.len(), 1, "radius {radius}: R1 must survive");
        block_on(async {
            assert_eq!(map.get(b"a".to_vec()).await.unwrap(), Some(b"1".to_vec()));
            assert_eq!(map.get(b"b".to_vec()).await.unwrap(), None);
        });
    }
}

#[test]
fn fully_persisted_unsynced_record_may_be_accepted() {
    // A completed write can reach stable media before
    // the crashing fdatasync; such a complete, valid record MAY be accepted.
    let (data_img, wal, _r1_end) = r1_durable_r2_inflight();
    let img = wal.crash_image(4096, |_| Persist::New); // R2 fully persisted
    let (store, _, _) = open_over_devs(data_img, img, opts()).unwrap();
    let map = BTreeMap::over(store);
    assert_eq!(map.len(), 2, "a fully-persisted valid R2 is acceptable");
    block_on(async {
        assert_eq!(map.get(b"b".to_vec()).await.unwrap(), Some(b"2".to_vec()));
    });
}

#[test]
fn arbitrary_unsynced_persistence_never_accepts_invalid() {
    // Enumerate persisted-prefix crash variants of the in-flight R2 and a holed
    // subset: R1 always recovers; an incomplete/holed R2 is never accepted.
    let radius = 512usize;
    let (data_img0, wal0, r1_end) = r1_durable_r2_inflight();
    let ranges = wal0.unsynced_ranges();
    let (r2_off, r2_len) = ranges[0];
    let first_g = r2_off as usize / radius;
    let last_g = (r2_off as usize + r2_len - 1) / radius;
    let n_g = last_g - first_g + 1;
    let _ = r1_end;

    // Prefix persistence: keep the first k granules of R2, drop the rest.
    for k in 0..=n_g {
        let (data_img, wal, _) = r1_durable_r2_inflight();
        let keep = first_g + k;
        let img = wal.crash_image(
            radius,
            |g| if g < keep { Persist::New } else { Persist::Old },
        );
        let (store, _, _) = open_over_devs(data_img, img, opts()).unwrap();
        let map = BTreeMap::over(store);
        let expect = if k == n_g { 2 } else { 1 };
        assert_eq!(
            map.len(),
            expect,
            "prefix k={k}/{n_g}: expected {expect} entries"
        );
    }
    let _ = (data_img0, wal0);

    // Holed subset: head + trailer persist but a middle granule tears (garbage,
    // distinct from the page's zero padding) → R2 must be rejected (checksum
    // fails), leaving only R1.
    if n_g >= 3 {
        let (data_img, wal, _) = r1_durable_r2_inflight();
        let mid = first_g + n_g / 2;
        let img = wal.crash_image(radius, |g| {
            if g == mid {
                Persist::Garble
            } else {
                Persist::New
            }
        });
        let (store, _, _) = open_over_devs(data_img, img, opts()).unwrap();
        assert_eq!(
            BTreeMap::over(store).len(),
            1,
            "a holed R2 must not be accepted"
        );
    }
}

#[test]
fn crash_during_post_recovery_overwrite_over_stale_bytes() {
    // Crash while a post-recovery append is **in flight,
    // overwriting stale bytes** left by a previous crashed run. R1 (acknowledged)
    // must survive at every persisted prefix of the new record, and neither the
    // partially-written new record nor the stale bytes beyond its write range may
    // be accepted (they do not chain from R1).
    let radius = 512usize;
    // Reuse the fixture to get R1's durable bytes and a well-formed R2 record.
    let (data_img, wal_fix, r1_end) = r1_durable_r2_inflight();
    let r2 = {
        let ranges = wal_fix.unsynced_ranges();
        let (off, len) = ranges[0];
        wal_fix.snapshot_bytes()[off as usize..off as usize + len].to_vec()
    };
    // Durable device = R1 + garbage "stale tail" (a prior crashed append).
    let mut durable = wal_fix.synced_bytes(); // R1 only
    durable.extend(std::iter::repeat(0x5Au8).take(r2.len() + 4096));
    let n_g = r2.len().div_ceil(radius);
    let first_g = r1_end / radius;

    for k in 0..=n_g {
        let dev = Arc::new(FakeIo::from_vec(durable.clone()));
        // Write the new record R2' at the accepted-prefix offset, UNSYNCED
        // (in flight), overwriting the head of the stale tail.
        block_on(async {
            dev.write_all_at(r1_end as u64, r2.clone().into_boxed_slice())
                .await
                .result
                .unwrap();
        });
        // Crash: persist the first k granules of R2'; the rest reverts to the
        // stale bytes (Old), and the stale tail beyond R2' is preserved.
        let keep = first_g + k;
        let img = dev.crash_image(
            radius,
            |g| if g < keep { Persist::New } else { Persist::Old },
        );
        let (store, _, _) = open_over_devs(data_img.clone(), img, opts()).unwrap();
        let map = BTreeMap::over(store);
        // R1 always survives; R2' only if fully persisted (all granules New).
        let expect = if k == n_g { 2 } else { 1 };
        assert_eq!(
            map.len(),
            expect,
            "post-recovery overwrite prefix k={k}/{n_g}"
        );
        block_on(async {
            assert_eq!(
                map.get(b"a".to_vec()).await.unwrap(),
                Some(b"1".to_vec()),
                "R1 survives"
            );
        });
    }
}

#[test]
fn group_commit_shares_one_fdatasync_and_recovers() {
    // Submit many applies via permits enqueued back-to-back (no awaits between),
    // so the writer drains and group-commits them; assert far fewer fdatasyncs
    // than records, and that every record recovers.
    let (data, wal) = devices();
    let mut options = opts();
    // High dirty high-water so no mid-batch flush trips; large budget.
    options.limits.max_dirty_pages = 1 << 20;
    options.limits.max_queue_slots = 4096;
    let map = BTreeMap::over(create(&data, &wal, options));
    let n = 300u32;
    let syncs_before = wal.sync_count();
    block_on(async {
        // Enqueue all N synchronously (WritePermit::apply enqueues at call), then
        // await them together so the writer sees a full queue and groups.
        let mut futs = Vec::new();
        for i in 0..n {
            let cost = WriteBatch::new()
                .insert(format!("k{i:04}").into_bytes(), b"v".to_vec())
                .cost(8);
            let permit = map.try_reserve_write(cost).expect("capacity");
            let fut = permit
                .apply(WriteBatch::new().insert(format!("k{i:04}").into_bytes(), b"v".to_vec()))
                .expect("cost matches");
            futs.push(fut);
        }
        for f in futs {
            f.await.unwrap();
        }
        // Commit the highest txid: everything at/under it is durable.
        map.commit(Version::new(map.store().incarnation(), u64::from(n)))
            .await
            .unwrap();
    });
    let syncs = wal.sync_count() - syncs_before;
    assert!(
        syncs < u64::from(n),
        "group commit should share fdatasyncs: {syncs} syncs for {n} records"
    );
    // Every record is durable and recovers.
    assert_eq!(map.metrics().durable_txid, u64::from(n));
    drop(map);
    let store = reopen_synced(&data, &wal, opts()).unwrap();
    let m2 = BTreeMap::over(store);
    assert_eq!(m2.len(), u64::from(n));
    block_on(async {
        assert_eq!(
            m2.get(b"k0000".to_vec()).await.unwrap(),
            Some(b"v".to_vec())
        );
        assert_eq!(
            m2.get(b"k0299".to_vec()).await.unwrap(),
            Some(b"v".to_vec())
        );
    });
}

#[test]
fn trailer_persists_without_body_is_rejected() {
    // A trailer without its body is an invalid record and must never be
    // accepted. Persist only the last granule (the commit trailer) of R2; its
    // body reverts.
    let radius = 512usize;
    let (data_img, wal, _) = r1_durable_r2_inflight();
    let ranges = wal.unsynced_ranges();
    let (r2_off, r2_len) = ranges[0];
    let last_g = (r2_off as usize + r2_len - 1) / radius;
    let img = wal.crash_image(radius, |g| {
        if g == last_g {
            Persist::New
        } else {
            Persist::Old
        }
    });
    let (store, _, _) = open_over_devs(data_img, img, opts()).unwrap();
    assert_eq!(
        BTreeMap::over(store).len(),
        1,
        "trailer-without-body must not be accepted"
    );
}

#[test]
fn fdatasync_eio_poisons_permanently() {
    // An injected WAL fdatasync EIO poisons permanently, never retry-acked.
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    wal.set_faults(Faults {
        fail_next_sync: Some(5),
        ..Default::default()
    });
    let r = block_on(async {
        let out = map
            .apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))
            .await?;
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!()
        };
        map.commit(version).await.map_err(|_| WriteError::Closed)
    });
    assert!(r.is_err(), "fdatasync EIO must surface");
    let r2 = block_on(map.apply(WriteBatch::new().insert(b"x".to_vec(), b"y".to_vec())));
    assert!(
        matches!(r2, Err(WriteError::Poisoned(_))),
        "poisoned, never retried; got {r2:?}"
    );
}

#[test]
fn recovery_is_read_only() {
    // Recovery touches no bytes: the device's durable image is unchanged and no
    // unsynced writes are issued.
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    for i in 0..6u32 {
        put(&map, format!("k{i}").as_bytes(), b"v");
    }
    let wal_before = wal.synced_bytes();
    drop(map);
    let recov_wal = Arc::new(FakeIo::from_vec(wal_before.clone()));
    let _store = Store::<Wal>::open_over(
        dev(&Arc::new(FakeIo::from_vec(data.synced_bytes()))),
        dev(&recov_wal),
        opts(),
    )
    .unwrap();
    assert!(
        recov_wal.unsynced_ranges().is_empty(),
        "recovery issued no writes"
    );
    assert_eq!(
        recov_wal.snapshot_bytes(),
        wal_before,
        "recovery mutated no bytes"
    );
}

#[test]
fn longer_replacement_after_torn_tail_survives() {
    // Recover a torn tail, commit a *longer* (multi-page) replacement, crash
    // again: the replacement survives and the stale tail never resurrects.
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    put(&map, b"a", b"1"); // R1
    let r1_end = wal.synced_bytes().len();
    put(&map, b"b", b"2"); // R2 (short)
    let full = wal.synced_bytes();
    drop(map);
    // Crash leaves R1 durable, R2 damaged (present but corrupt).
    let mut img = full.clone();
    img[r1_end + 30] ^= 0xFF;
    let (store, _d, w2) = open_over_devs(data.synced_bytes(), img, opts()).unwrap();
    let map2 = BTreeMap::over(store);
    assert_eq!(map2.len(), 1);
    // Commit a LONGER replacement: a multi-key batch spanning several pages.
    block_on(async {
        let mut b = WriteBatch::new();
        for i in 0..40u32 {
            b = b.insert(format!("r{i:03}").into_bytes(), vec![9u8; 5000]);
        }
        let out = map2.apply(b).await.unwrap();
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!()
        };
        map2.commit(version).await.unwrap();
    });
    let final_wal = w2.synced_bytes();
    drop(map2);
    let (fs, _, _) = open_over_devs(data.synced_bytes(), final_wal, opts()).unwrap();
    let fm = BTreeMap::over(fs);
    block_on(async {
        assert_eq!(fm.get(b"a".to_vec()).await.unwrap(), Some(b"1".to_vec()));
        assert_eq!(
            fm.get(b"r000".to_vec()).await.unwrap(),
            Some(vec![9u8; 5000])
        );
        assert_eq!(
            fm.get(b"r039".to_vec()).await.unwrap(),
            Some(vec![9u8; 5000])
        );
        assert_eq!(
            fm.get(b"b".to_vec()).await.unwrap(),
            None,
            "stale short R2 must not resurrect"
        );
    });
    assert_eq!(fm.len(), 41);
}

#[test]
fn store_full_measured_by_logical_not_estimate_or_physical() {
    // Sustained small batches approach max_wal_bytes by
    // wal_logical_bytes; admission is by reserved (logical + outstanding), never
    // by st_size. Fill until StoreFull, then assert logical is near the limit.
    let (data, wal) = devices();
    let mut options = opts();
    options.max_wal_bytes = crate::wal::WAL_HEADER_LEN + 40 * (crate::page::PAGE_SIZE as u64 + 256);
    let map = BTreeMap::over(create(&data, &wal, options.clone()));
    let mut committed = 0u64;
    block_on(async {
        loop {
            match map
                .apply(
                    WriteBatch::new().insert(format!("k{committed}").into_bytes(), b"v".to_vec()),
                )
                .await
            {
                Ok(ApplyOutcome::Applied { version, .. }) => {
                    map.commit(version).await.unwrap();
                    committed += 1;
                }
                Err(WriteError::StoreFull) => break,
                other => panic!("unexpected {other:?}"),
            }
            assert!(committed < 1000, "should hit StoreFull well before this");
        }
    });
    let m = map.metrics();
    assert!(committed > 0, "some small batches fit");
    assert!(
        m.wal_logical_bytes <= options.max_wal_bytes,
        "logical never exceeds the limit"
    );
    // In a clean append (no torn-tail recovery) the physical file equals the
    // logical frontier — admission is measured by logical/reserved, never by an
    // inflated `st_size`.
    assert_eq!(
        m.wal_physical_bytes, m.wal_logical_bytes,
        "clean append: physical == logical (admission is not by st_size)"
    );
    // The store stopped only because the *next* batch's pessimistic worst-case
    // reservation no longer fit — logical is within one such reservation of the
    // limit, not prematurely far below it.
    let one_batch_worst = WriteBatch::new()
        .insert(b"k".to_vec(), b"v".to_vec())
        .cost(8)
        .wal_worst_case_bytes;
    assert!(
        options.max_wal_bytes - m.wal_logical_bytes <= one_batch_worst,
        "logical={} is within one worst-case reservation ({}) of limit={}",
        m.wal_logical_bytes,
        one_batch_worst,
        options.max_wal_bytes
    );
}

#[test]
fn batch_exceeding_page_estimate_never_poisons() {
    // Regression: a batch whose actual rebuilt pages
    // exceed the admission estimate (a tall multi-level tree + a multi-op batch)
    // must reconcile up — succeeding, or `StoreFull` if it no longer fits
    // `max_wal_bytes` — but NEVER poison the store.
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    // Build a multi-level tree with large keys/values (low fanout, many leaves).
    block_on(async {
        for i in 0..300u32 {
            let out = map
                .apply(WriteBatch::new().insert(format!("key{i:05}").into_bytes(), vec![7u8; 6000]))
                .await
                .unwrap();
            let ApplyOutcome::Applied { version, .. } = out else {
                panic!()
            };
            map.commit(version).await.unwrap();
        }
    });
    assert!(map.metrics().visible_txid > 0);
    // A multi-op batch touching many distinct leaves rebuilds far more pages than
    // `2·op_count + depth_bound` estimates.
    let out = block_on(async {
        let mut b = WriteBatch::new();
        for i in (0..300u32).step_by(7) {
            b = b.insert(format!("key{i:05}").into_bytes(), vec![8u8; 6000]);
        }
        map.apply(b).await
    });
    match out {
        Ok(ApplyOutcome::Applied { version, .. }) => {
            block_on(map.commit(version)).unwrap();
        }
        Ok(ApplyOutcome::ConditionFailed { .. }) => panic!("no conditions"),
        Err(WriteError::StoreFull) => {} // acceptable healthy fail-fast
        Err(WriteError::Poisoned(r)) => panic!("legal batch poisoned the store: {r:?}"),
        Err(e) => panic!("unexpected {e:?}"),
    }
    // The store is still healthy: a further write succeeds.
    let ok = block_on(map.apply(WriteBatch::new().insert(b"z".to_vec(), b"z".to_vec())));
    assert!(
        matches!(ok, Ok(ApplyOutcome::Applied { .. })),
        "store still healthy; got {ok:?}"
    );
}

#[test]
fn many_concurrent_commit_waiters_all_wake() {
    // Waiters across grouped txids wake at the correct frontier. Apply N
    // batches, then have N threads concurrently await each version's commit.
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    let n = 200u32;
    let versions: Vec<Version> = block_on(async {
        let mut vs = Vec::new();
        for i in 0..n {
            let out = map
                .apply(WriteBatch::new().insert(format!("k{i:04}").into_bytes(), b"v".to_vec()))
                .await
                .unwrap();
            let ApplyOutcome::Applied { version, .. } = out else {
                panic!()
            };
            vs.push(version);
        }
        vs
    });
    let handles: Vec<_> = versions
        .into_iter()
        .map(|v| {
            let m = map.clone();
            std::thread::spawn(move || block_on(m.commit(v)))
        })
        .collect();
    for h in handles {
        h.join()
            .unwrap()
            .expect("every waiter resolves Ok at the frontier");
    }
    assert_eq!(map.metrics().durable_txid, u64::from(n));
}

#[test]
fn empty_wal_recovers_empty_store() {
    let (data, wal) = devices();
    let _ = create(&data, &wal, opts()); // writes headers only, no records
    let store = reopen_synced(&data, &wal, opts()).unwrap();
    let map = BTreeMap::over(store);
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
}

#[test]
fn explicit_verify_detects_recovered_wal_count_mismatch() {
    let (data, wal) = devices();
    let map = BTreeMap::over(create(&data, &wal, opts()));
    block_on(async {
        let mut batch = WriteBatch::new();
        for key in 0..5u8 {
            batch = batch.insert(vec![key], vec![key]);
        }
        let ApplyOutcome::Applied { version, .. } = map.apply(batch).await.unwrap() else {
            panic!("expected apply");
        };
        map.commit(version).await.unwrap();
        map.close().await.unwrap();
    });
    let original = wal.synced_bytes();
    for count in [1u64, 999, u64::MAX] {
        let mut image = original.clone();
        let start = crate::page::PAGE_SIZE;
        let tail = image.len() - 52;
        image[tail + 20..tail + 28].copy_from_slice(&count.to_le_bytes());
        let previous = u32::from_le_bytes(image[start + 24..start + 28].try_into().unwrap());
        let checksum = crc32c::crc32c_append(previous, &image[start..tail + 28]);
        image[tail + 28..tail + 32].copy_from_slice(&checksum.to_le_bytes());
        let end = image.len();
        image[end - 4..].copy_from_slice(&checksum.to_le_bytes());
        let (store, data_image, wal_image) =
            open_over_devs(data.synced_bytes(), image, opts()).unwrap();
        let recovered = BTreeMap::over(store);
        assert_eq!(
            recovered.len(),
            count,
            "fixture must retain its checksum-valid declaration"
        );
        for key in 0..5u8 {
            assert_eq!(block_on(recovered.get(vec![key])).unwrap(), Some(vec![key]));
        }
        let before = (data_image.snapshot_bytes(), wal_image.snapshot_bytes());
        assert!(matches!(
            block_on(recovered.verify()),
            Err(crate::error::ReadError::Corrupt(_))
        ));
        assert_eq!(
            before,
            (data_image.snapshot_bytes(), wal_image.snapshot_bytes())
        );
        block_on(recovered.close()).unwrap();
    }
}
