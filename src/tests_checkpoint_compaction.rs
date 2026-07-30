//! Acceptance for the checkpoint, compaction and generation-retention
//! machinery, plus the overflow-value and disk-space limits layered on them.
//!
//! Runs over `FakeIo` so the durability barrier is controllable: a test arms the
//! device to hold an in-flight `datasync`, making the window between "records
//! written" and "sync durable" observable — the window where dirty capacity must
//! release only at the completion frontier, where dropping a commit waiter must
//! not cancel the group commit, and where `close()` must wait for a mid-sync
//! writer.
//!
//! On that seam the module asserts: the data header is recovery's only source of
//! truth and rejects hostile or stale variants; a checkpoint relocates the live
//! set, rotates the segment, never reuses page ids, and lands on one side of
//! every crash boundary; snapshots hold revocable generation leases that expire
//! at the retention bound and free their files exactly once; compaction rebuilds
//! both backends densely, preserves content, and aborts cleanly before promotion
//! or poisons after it; overflow values round-trip, relocate and renumber across
//! every size and backend; and space limits are enforced and reported rather
//! than silently breached.
//!
//! It also carries the long soak that bounds space growth and leaks, the guard
//! keeping the clean cache tier's adopted behaviour pinned, the idle/maintenance
//! writer's servicing of orphaned cache loads, and the bench-only device-wrapper
//! seam.

use crate::backend::Wal;
use crate::batch::{ApplyOutcome, WriteBatch};
use crate::io::{FakeIo, PageIo};
use crate::map::BTreeMap;
use crate::store::{Options, Store};
use crate::version::Version;
use futures_lite::future::{block_on, poll_once};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn dev(a: &Arc<FakeIo>) -> Arc<dyn PageIo> {
    Arc::clone(a) as Arc<dyn PageIo>
}

/// A fresh StoreWal over inspectable data + WAL `FakeIo` devices. The WAL device
/// is returned so a test can arm its sync-gate. Header syncs run inside
/// `create_with_io` before this returns, so the first gated `datasync` is the
/// first *commit*, never a create-time header sync.
fn wal_store() -> (BTreeMap<Wal>, Arc<FakeIo>, Arc<FakeIo>) {
    let data = Arc::new(FakeIo::new(0));
    let wal = Arc::new(FakeIo::new(0));
    let store = Store::<Wal>::create_with_io(dev(&data), dev(&wal), Options::default()).unwrap();
    (BTreeMap::over(store), data, wal)
}

/// Applies a single-insert batch, returning its version (published, not yet
/// necessarily durable).
fn apply(map: &BTreeMap<Wal>, k: &[u8], v: &[u8]) -> Version {
    block_on(async {
        let out = map
            .apply(WriteBatch::new().insert(k.to_vec(), v.to_vec()))
            .await
            .unwrap();
        match out {
            ApplyOutcome::Applied { version, .. } => version,
            other => panic!("expected Applied, got {other:?}"),
        }
    })
}

/// Spin until the WAL device reports a `datasync` parked at the barrier — i.e. the
/// writer has written the commit's records and is blocked making them durable.
fn wait_until_mid_sync(wal: &Arc<FakeIo>) {
    let mut spins = 0u64;
    while wal.parked_sync_count() == 0 {
        std::thread::yield_now();
        spins += 1;
        assert!(
            spins < 100_000_000,
            "writer never reached the gated datasync"
        );
    }
}

/// Invariant 5: dirty-page capacity is charged at publication and released **only
/// at the completion frontier** (after `fdatasync`), not merely once the records
/// are written. With the sync gated, the batch's dirty pages must stay charged;
/// they drop to zero only after the sync is released and durability advances.
#[test]
fn dirty_capacity_held_until_completion_frontier() {
    let (map, _data, wal) = wal_store();
    wal.arm_sync_gate(1); // hold the next (commit) datasync in-flight

    let v = apply(&map, b"k", b"v"); // publishes; writer idle-flushes → parks mid-sync
    wait_until_mid_sync(&wal);

    // Records are written but not yet durable: the dirty pages remain charged and
    // durability has not advanced past the published version.
    let m = map.metrics();
    assert!(m.dirty_pages > 0, "dirty pages must stay charged mid-sync");
    assert!(
        m.durable_txid < v.txid(),
        "durability must not advance before the sync lands"
    );

    // Release the sync: durability reaches the frontier and the dirty pages free.
    wal.release_syncs(1);
    block_on(map.commit(v)).unwrap();
    let m = map.metrics();
    assert_eq!(m.durable_txid, v.txid());
    assert_eq!(
        m.dirty_pages, 0,
        "dirty pages release at the completion frontier"
    );
}

/// Invariant 9: dropping a commit waiter only detaches that listener; it never
/// cancels the in-flight group commit. Abandon a `commit` future while its group
/// commit is mid-`fdatasync`, then prove the commit still lands durably (a reopen
/// recovers it) and a *later* barrier progresses.
#[test]
fn dropping_commit_waiter_mid_sync_does_not_cancel_group_commit() {
    let (map, data, wal) = wal_store();
    wal.arm_sync_gate(1);

    let v1 = apply(&map, b"a", b"1");
    wait_until_mid_sync(&wal);

    // Register a commit waiter for v1, observe it is pending (the sync is held),
    // then drop it — the abandonment. `poll_once` polls the future exactly once
    // and drops it, which is precisely a waiter that registered then went away.
    let polled = block_on(poll_once(map.commit(v1)));
    assert!(
        polled.is_none(),
        "commit must be pending while the sync is gated"
    );

    // Release: the group commit the abandoned waiter was listening on completes on
    // its own. Durability reaches v1 despite the dropped waiter.
    wal.release_syncs(1);
    block_on(map.commit(v1)).unwrap();
    assert_eq!(map.metrics().durable_txid, v1.txid());

    // A later barrier still progresses: a fresh write commits durably.
    let v2 = apply(&map, b"b", b"2");
    block_on(map.commit(v2)).unwrap();
    assert_eq!(map.metrics().durable_txid, v2.txid());

    // And both survive a crash after the last sync (acknowledged commits are in
    // the durable image): reopen recovers them.
    block_on(map.close()).unwrap();
    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let w = Arc::new(FakeIo::from_vec(wal.synced_bytes()));
    let reopened =
        BTreeMap::over(Store::<Wal>::open_over(dev(&d), dev(&w), Options::default()).unwrap());
    assert_eq!(
        block_on(reopened.get(b"a".to_vec())).unwrap().as_deref(),
        Some(&b"1"[..])
    );
    assert_eq!(
        block_on(reopened.get(b"b".to_vec())).unwrap().as_deref(),
        Some(&b"2"[..])
    );
    block_on(reopened.close()).unwrap();
}

/// `close()` must wait for a writer that is mid-`fdatasync`: a clean close
/// preserves visible state, so it cannot resolve while a
/// durability barrier for accepted work is still in flight. With the commit sync
/// gated, `close()` stays pending until the sync is released.
#[test]
fn close_waits_for_mid_sync_writer() {
    let (map, data, wal) = wal_store();
    wal.arm_sync_gate(1);

    let v = apply(&map, b"k", b"v");
    wait_until_mid_sync(&wal); // writer is blocked inside the commit datasync

    // Drive close() on another thread; it must not finish while the writer is
    // parked mid-sync.
    let closed = Arc::new(AtomicBool::new(false));
    let map2 = map.clone();
    let closed2 = Arc::clone(&closed);
    let h = std::thread::spawn(move || {
        block_on(map2.close()).unwrap();
        closed2.store(true, Ordering::Release);
    });

    // Give the close a real chance to (wrongly) complete, then assert it hasn't:
    // it is genuinely blocked on the in-flight sync, not merely not-yet-scheduled.
    for _ in 0..10_000 {
        std::thread::yield_now();
    }
    assert!(
        !closed.load(Ordering::Acquire),
        "close resolved while a durability barrier was still mid-sync"
    );
    assert!(
        wal.parked_sync_count() >= 1,
        "the writer should still be parked"
    );

    // Release the sync: the writer finishes the commit, then drains and closes.
    wal.release_syncs(1);
    h.join().unwrap();
    assert!(closed.load(Ordering::Acquire));

    // The write was durable before close returned: reopen recovers it.
    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let w = Arc::new(FakeIo::from_vec(wal.synced_bytes()));
    let reopened =
        BTreeMap::over(Store::<Wal>::open_over(dev(&d), dev(&w), Options::default()).unwrap());
    assert_eq!(
        block_on(reopened.get(b"k".to_vec())).unwrap().as_deref(),
        Some(&b"v"[..])
    );
    assert_eq!(reopened.metrics().durable_txid, v.txid());
    block_on(reopened.close()).unwrap();
}

// -- Phase B: the data header is recovery's source of truth ------------------
//
// A fresh store writes the `checkpoint_seq == 0` header (empty root, empty
// manifest, WAL segment 0, scan origin at the segment header end), so a
// never-checkpointed store is just the seq-0 case of the general recovery path.
// Recovery takes *all* reset state from the winning slot and refuses to scan a
// segment that slot does not name.

use crate::checkpoint::{read_winning_header, CheckpointHeader};
use crate::error::OpenError;
use crate::page::PAGE_SIZE;
use crate::wal::{encode_wal_header, seed_checksum, WAL_HEADER_LEN};

/// The winning header of a data-file image.
fn header_of(image: Vec<u8>) -> CheckpointHeader {
    let dev = FakeIo::from_vec(image);
    block_on(read_winning_header(&dev)).unwrap()
}

/// Opens over explicit data/WAL images, discarding the store on success so the
/// error case is inspectable (`Store` is not `Debug`).
fn open_images(data: Vec<u8>, wal: Vec<u8>) -> Result<(), OpenError> {
    let d = Arc::new(FakeIo::from_vec(data));
    let w = Arc::new(FakeIo::from_vec(wal));
    Store::<Wal>::open_over(dev(&d), dev(&w), Options::default()).map(|s| {
        let _ = block_on(s.close());
    })
}

#[test]
fn fresh_store_writes_the_seq_zero_checkpoint_header() {
    let (map, data, _wal) = wal_store();
    apply(&map, b"k", b"v");
    block_on(map.close()).unwrap();
    // Read the *durable* image: the create protocol syncs the header before it
    // acknowledges, so a crash right after create still reopens.
    let h = header_of(data.synced_bytes());
    assert_eq!(h.checkpoint_seq, 0, "a never-checkpointed store is seq 0");
    assert_eq!(
        h.checkpoint_txid, 0,
        "nothing is superseded by the data file yet"
    );
    assert!(h.root.is_empty(), "no checkpoint has published a base root");
    assert_eq!(h.manifest_len, 0);
    assert_eq!(h.wal_segment_seq, 0);
    assert_eq!(
        h.wal_scan_origin, WAL_HEADER_LEN,
        "scan starts past the segment header"
    );
    assert_eq!(
        h.wal_chain_seed,
        seed_checksum(h.store_uuid, 0),
        "the header records segment 0's chain seed"
    );
}

#[test]
fn recovery_refuses_a_segment_the_header_does_not_name() {
    let (map, data, wal) = wal_store();
    apply(&map, b"k", b"v");
    block_on(map.commit(apply(&map, b"k2", b"v2"))).unwrap();
    block_on(map.close()).unwrap();
    let uuid = header_of(data.synced_bytes()).store_uuid;

    // Re-stamp the segment as seq 1 while the data header still names seq 0. The
    // records' chain is now unverifiable against this identity — recovery must
    // refuse the whole segment rather than scan it as if it were segment 0.
    let mut img = wal.synced_bytes();
    img[..PAGE_SIZE].copy_from_slice(&encode_wal_header(uuid, 1));
    let err = open_images(data.synced_bytes(), img).unwrap_err();
    assert!(
        matches!(err, OpenError::Corrupt(m) if m.contains("segment seq")),
        "a mis-named segment must not be scanned"
    );
}

#[test]
fn recovery_refuses_a_v1_data_header() {
    // The format byte is bumped and the old format refused the day the M6 header
    // ships — never silently misread. This is the v1
    // "MP5B" base header a pre-M6 store left at page 0.
    let mut v1 = vec![0u8; PAGE_SIZE];
    v1[0..4].copy_from_slice(&0x4D50_5342u32.to_le_bytes()); // "MP5B"
    v1[4] = 1; // format v1
    v1[5] = 1; // crc32c
    v1[8..12].copy_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
    let ck = crc32c::crc32c(&v1[..32]);
    v1[32..36].copy_from_slice(&ck.to_le_bytes());
    assert!(
        matches!(
            open_images(
                [v1, vec![0u8; PAGE_SIZE]].concat(),
                encode_wal_header([0u8; 16], 0).to_vec()
            ),
            Err(OpenError::Corrupt(_))
        ),
        "a v1 store must be refused, not misread"
    );
}

#[test]
fn recovery_refuses_a_manifest_extent_outside_the_data_file() {
    let (map, data, wal) = wal_store();
    block_on(map.commit(apply(&map, b"k", b"v"))).unwrap();
    block_on(map.close()).unwrap();
    let mut img = data.synced_bytes();
    let h = header_of(img.clone());
    // Forge a checksum-valid slot whose manifest lies past EOF: the extent check
    // must fire *before* any read or allocation is driven by the length.
    let forged = CheckpointHeader {
        manifest_offset: 2 * PAGE_SIZE as u64,
        manifest_len: 20 * 1_000_000,
        allocator_frontier: 1 << 40,
        ..h
    };
    img[..PAGE_SIZE].copy_from_slice(&forged.encode());
    let err = open_images(img, wal.synced_bytes()).unwrap_err();
    assert!(
        matches!(err, OpenError::Corrupt(m) if m.contains("manifest extent")),
        "an out-of-file manifest is Corrupt, got {err:?}"
    );
}

// -- Phase B: the checkpoint ------------------------------------------------
//
// The marquee M6 gap: without a checkpoint, committed pages live in the WAL
// forever — `overlay_pages`, WAL bytes, and reopen cost all grow with total
// writes ever made. A checkpoint relocates the *live* set into the data file,
// rotates to a fresh WAL segment, and unlinks the retired one.

use crate::checkpoint::read_manifest;
use crate::error::CheckpointError;
use crate::fileset::{create_mem_store, open_mem_store, MemWalFileSet, WalFileSet};

/// A StoreWal that owns an in-memory segment namespace, so it can checkpoint.
/// Returns the map, its data device, and the file set (which a test inspects for
/// rotation/retention).
fn owned_store() -> (BTreeMap<Wal>, Arc<FakeIo>, Arc<MemWalFileSet>) {
    let data = Arc::new(FakeIo::new(0));
    let file_set = MemWalFileSet::new();
    let store = create_mem_store(dev(&data), Arc::clone(&file_set), Options::default()).unwrap();
    (BTreeMap::over(store), data, file_set)
}

/// Applies and commits `k -> v` durable.
fn put(map: &BTreeMap<Wal>, k: &[u8], v: &[u8]) {
    block_on(map.commit(apply(map, k, v))).unwrap();
}

/// Reopens over the durable images: the data file plus every segment the file set
/// still holds, each frozen at its last `datasync` — i.e. a clean crash. Returns
/// the *new* devices, since the reopened store writes to those, not the originals.
fn reopen_synced(
    data: &Arc<FakeIo>,
    file_set: &Arc<MemWalFileSet>,
) -> (BTreeMap<Wal>, Arc<FakeIo>, Arc<MemWalFileSet>) {
    let d = Arc::new(FakeIo::from_vec(data.synced_bytes()));
    let fresh = MemWalFileSet::new();
    for seq in file_set.live_segments() {
        let seg = file_set.segment(seq).unwrap();
        fresh.insert(seq, Arc::new(FakeIo::from_vec(seg.synced_bytes())));
    }
    let store = open_mem_store(dev(&d), Arc::clone(&fresh), Options::default()).unwrap();
    (BTreeMap::over(store), d, fresh)
}

#[test]
fn checkpoint_relocates_the_live_set_and_rotates_the_segment() {
    let (map, _data, file_set) = owned_store();
    for i in 0..60u32 {
        put(&map, format!("k{i:03}").as_bytes(), &vec![i as u8; 400]);
    }
    // Overwrite every key: each rewrite CoW-copies its path, so the WAL now holds
    // far more pages than the tree actually needs.
    for i in 0..60u32 {
        put(&map, format!("k{i:03}").as_bytes(), &vec![i as u8; 500]);
    }
    let before = map.store().metrics();
    let wal_before = before.wal_logical_bytes;
    assert_eq!(before.checkpoints, 0);
    assert_eq!(before.checkpoint_pages, 0, "nothing checkpointed yet");

    let report = block_on(map.store().checkpoint()).unwrap();
    let after = map.store().metrics();

    assert_eq!(report.checkpoint_seq, 1);
    assert_eq!(report.checkpoint_txid, before.durable_txid);
    assert_eq!(report.retired_wal_bytes, wal_before);
    // The live set is far smaller than every page ever written: superseded CoW
    // ancestors are unreachable and so are simply dropped.
    assert!(
        report.relocated_pages < before.overlay_pages / 2,
        "checkpoint relocated {} of {} overlay pages — superseded pages should be dropped",
        report.relocated_pages,
        before.overlay_pages
    );
    assert_eq!(after.checkpoint_pages, report.relocated_pages);
    assert_eq!(
        after.overlay_pages, 0,
        "the fresh segment holds no pages yet"
    );
    assert_eq!(after.checkpoints, 1);
    assert_eq!(after.last_checkpoint_txid, report.checkpoint_txid);
    assert_eq!(after.leaked_segments, 0);
    // WAL bytes are *reclaimed*, not merely capped: the meter is rebased onto the
    // fresh segment.
    assert_eq!(after.wal_logical_bytes, WAL_HEADER_LEN);
    assert!(wal_before > WAL_HEADER_LEN * 10, "the WAL had really grown");

    // Rotation: segment 0 is unlinked, segment 1 is in force.
    assert_eq!(file_set.live_segments(), vec![1], "old segment unlinked");

    // Reads are unaffected — every page now resolves in the data file.
    block_on(async {
        for i in 0..60u32 {
            assert_eq!(
                map.get(format!("k{i:03}").into_bytes()).await.unwrap(),
                Some(vec![i as u8; 500])
            );
        }
    });
    block_on(map.close()).unwrap();
}

#[test]
fn acked_post_checkpoint_commits_survive_reopen() {
    // A commit acked *after* a checkpoint must never be lost, because the
    // winning header names the segment those records live in.
    let (map, data, file_set) = owned_store();
    for i in 0..30u32 {
        put(&map, format!("old{i:02}").as_bytes(), b"before");
    }
    block_on(map.store().checkpoint()).unwrap();
    for i in 0..30u32 {
        put(&map, format!("new{i:02}").as_bytes(), b"after");
    }
    block_on(map.close()).unwrap();

    let (reopened, _, _) = reopen_synced(&data, &file_set);
    block_on(async {
        for i in 0..30u32 {
            assert_eq!(
                reopened
                    .get(format!("old{i:02}").into_bytes())
                    .await
                    .unwrap(),
                Some(b"before".to_vec()),
                "a pre-checkpoint key must come back from the data file"
            );
            assert_eq!(
                reopened
                    .get(format!("new{i:02}").into_bytes())
                    .await
                    .unwrap(),
                Some(b"after".to_vec()),
                "an acked post-checkpoint commit must never be lost"
            );
        }
        // The store keeps working after recovery.
        let v = apply(&reopened, b"newer", b"v");
        reopened.commit(v).await.unwrap();
    });
    block_on(reopened.close()).unwrap();
}

#[test]
fn reopen_after_checkpoints_scans_only_the_live_segment() {
    // Reopen cost is O(post-checkpoint segment), not O(total history) — the whole
    // point of the milestone.
    let (map, data, file_set) = owned_store();
    for i in 0..200u32 {
        put(&map, format!("k{i:03}").as_bytes(), &vec![7u8; 300]);
    }
    let history = map.store().metrics().wal_logical_bytes;
    block_on(map.store().checkpoint()).unwrap();
    put(&map, b"tail", b"v");
    let tail_bytes = map.store().metrics().wal_logical_bytes;
    block_on(map.close()).unwrap();

    let (reopened, _, _) = reopen_synced(&data, &file_set);
    let m = reopened.store().metrics();
    assert_eq!(
        m.open_scan_bytes, tail_bytes,
        "the scan consumed only the live segment"
    );
    assert!(
        m.open_scan_bytes * 20 < history,
        "reopen scanned {} bytes vs {} of history",
        m.open_scan_bytes,
        history
    );
    assert_eq!(
        m.last_checkpoint_txid, 200,
        "the checkpoint txid survives reopen"
    );
    block_on(async {
        assert_eq!(
            reopened.get(b"k000".to_vec()).await.unwrap(),
            Some(vec![7u8; 300])
        );
        assert_eq!(
            reopened.get(b"tail".to_vec()).await.unwrap(),
            Some(b"v".to_vec())
        );
    });
    block_on(reopened.close()).unwrap();
}

#[test]
fn snapshot_taken_before_a_checkpoint_stays_readable() {
    // The retired segment is unlinked at cutover, but a snapshot holds the old
    // generation — and with it the segment's handle. POSIX keeps the inode alive;
    // the bytes free when the snapshot drops (invariant 8).
    let (map, _data, file_set) = owned_store();
    for i in 0..40u32 {
        put(&map, format!("k{i:02}").as_bytes(), b"v1");
    }
    let snap = map.snapshot();

    block_on(map.store().checkpoint()).unwrap();
    assert!(
        file_set.segment(0).is_none(),
        "segment 0 is unlinked — only the snapshot's own handle keeps it alive"
    );

    block_on(async {
        for i in 0..40u32 {
            assert_eq!(
                snap.get(format!("k{i:02}").into_bytes()).await.unwrap(),
                Some(b"v1".to_vec()),
                "a snapshot must stay readable across the checkpoint that retired its segment"
            );
        }
    });
    // Writes keep flowing into the new segment while the old snapshot lives.
    put(&map, b"after", b"v2");
    block_on(async {
        assert_eq!(
            snap.get(b"after".to_vec()).await.unwrap(),
            None,
            "snapshot is stable"
        );
        assert_eq!(
            map.get(b"after".to_vec()).await.unwrap(),
            Some(b"v2".to_vec())
        );
    });
    drop(snap);
    block_on(map.close()).unwrap();
}

#[test]
fn a_retired_segment_frees_exactly_when_the_last_snapshot_drops() {
    // The other half of the retention claim, and the reason B needs no refcount:
    // unlink-at-cutover is safe because POSIX keeps the inode alive for every open
    // handle. `Weak` here stands in for the inode's link count — the bytes live
    // while some holder exists, and vanish the instant the last one drops.
    let (map, _data, file_set) = owned_store();
    for i in 0..30u32 {
        put(&map, format!("k{i:02}").as_bytes(), b"v");
    }
    let snap = map.snapshot();
    let retired = Arc::downgrade(&file_set.segment(0).unwrap());

    block_on(map.store().checkpoint()).unwrap();
    assert!(
        retired.upgrade().is_some(),
        "the unlinked segment's bytes must survive for the snapshot still reading them"
    );
    // Everything else has let go: the writer rebound, the published generation was
    // replaced. Only the snapshot holds it.
    drop(snap);
    assert!(
        retired.upgrade().is_none(),
        "the retired segment's bytes must free the instant the last snapshot drops"
    );
    block_on(map.close()).unwrap();
}

#[test]
fn page_ids_are_never_reused_across_a_checkpoint() {
    // The allocator frontier F is *persisted* in the header, not merely used to
    // gate minting: a reopened store must mint past every id the
    // dropped records burned, or a stale ref could resolve to the wrong page.
    let (map, data, file_set) = owned_store();
    for i in 0..40u32 {
        put(&map, format!("k{i:02}").as_bytes(), b"v");
    }
    // Empty the tree, then checkpoint: the live set is empty, so *nothing* records
    // the ids the 40 inserts burned except F.
    block_on(async {
        for i in 0..40u32 {
            let out = map
                .apply(WriteBatch::new().remove(format!("k{i:02}").into_bytes()))
                .await
                .unwrap();
            if let ApplyOutcome::Applied { version, .. } = out {
                map.commit(version).await.unwrap();
            }
        }
    });
    block_on(map.store().checkpoint()).unwrap();
    let frontier = header_of(data.synced_bytes()).allocator_frontier;
    assert!(frontier > 40, "the 40 inserts burned ids: F = {frontier}");
    block_on(map.close()).unwrap();

    // Reopen (the segment is empty, so *only* the header seeds the allocator) and
    // write again.
    let (reopened, data2, _fs2) = reopen_synced(&data, &file_set);
    assert!(reopened.is_empty());
    for i in 0..5u32 {
        put(&reopened, format!("n{i}").as_bytes(), b"v");
    }
    block_on(reopened.store().checkpoint()).unwrap();
    block_on(reopened.close()).unwrap();

    let h = header_of(data2.synced_bytes());
    assert_eq!(
        h.checkpoint_seq, 2,
        "the reopened store continued the checkpoint sequence"
    );
    let devd = FakeIo::from_vec(data2.synced_bytes());
    let manifest = block_on(read_manifest(&devd, &h)).unwrap();
    assert!(!manifest.is_empty());
    for e in &manifest {
        assert!(
            e.page_id.as_u64() >= frontier,
            "page id {} was reused below the persisted frontier {frontier}",
            e.page_id.as_u64()
        );
    }
}

#[test]
fn zero_page_commit_after_a_checkpoint_recovers() {
    // A no-op batch commits a record with no pages, so its root resolves only if
    // recovery seeded the root-resolve checksum set from the manifest.
    let (map, data, file_set) = owned_store();
    for i in 0..20u32 {
        put(&map, format!("k{i:02}").as_bytes(), b"v");
    }
    block_on(map.store().checkpoint()).unwrap();
    // Removing an absent key publishes a root-unchanged, zero-page batch.
    let v = block_on(async {
        let out = map
            .apply(WriteBatch::new().remove(b"absent".to_vec()))
            .await
            .unwrap();
        match out {
            ApplyOutcome::Applied { version, .. } => version,
            other => panic!("expected Applied, got {other:?}"),
        }
    });
    block_on(map.commit(v)).unwrap();
    block_on(map.close()).unwrap();

    let (reopened, _, _) = reopen_synced(&data, &file_set);
    block_on(async {
        assert_eq!(
            reopened.get(b"k00".to_vec()).await.unwrap(),
            Some(b"v".to_vec())
        );
        assert_eq!(
            reopened.get(b"k19".to_vec()).await.unwrap(),
            Some(b"v".to_vec())
        );
    });
    assert_eq!(
        reopened.len(),
        20,
        "the zero-page record kept the checkpoint root"
    );
    block_on(reopened.close()).unwrap();
}

#[test]
fn a_store_over_caller_supplied_devices_cannot_checkpoint() {
    // A bare `Arc<dyn PageIo>` has no name to rotate, so the store owns no
    // namespace and says so instead of pretending.
    let (map, _data, _wal) = wal_store();
    put(&map, b"k", b"v");
    assert!(matches!(
        block_on(map.store().checkpoint()),
        Err(CheckpointError::Unsupported)
    ));
    block_on(map.close()).unwrap();
}

#[test]
fn checkpoint_on_a_closed_store_is_terminal_not_a_hang() {
    let (map, _data, _fs) = owned_store();
    put(&map, b"k", b"v");
    block_on(map.close()).unwrap();
    assert!(matches!(
        block_on(map.store().checkpoint()),
        Err(CheckpointError::Closed)
    ));
}

#[test]
fn checkpoint_during_maintenance_fails_fast() {
    // The writer processes no commands while parked, so a checkpoint must not
    // queue behind an operator-held pause.
    let (map, _data, _fs) = owned_store();
    put(&map, b"k", b"v");
    assert!(map.store().begin_maintenance());
    assert!(matches!(
        block_on(map.store().checkpoint()),
        Err(CheckpointError::Compacting)
    ));
    map.store().end_maintenance();
    assert!(
        block_on(map.store().checkpoint()).is_ok(),
        "works again after resume"
    );
    block_on(map.close()).unwrap();
}

// -- Phase B: the checkpoint crash matrix ------------------------------------
//
// The checkpoint's atomic commit point is the step-5 data-header `datasync`. The
// Phase A sync-gate lets a test park exactly there — pages and manifest durable,
// the new segment minted, the header written but *not* yet durable — and then
// decide what a crash persists of that header page. Every outcome must be the
// pre- or the post-checkpoint state, never a mix, and never a lost commit.

use crate::io::Persist;

/// A checkpoint parked at its commit point, plus the durable images of the whole
/// file set at that instant. Dropping `finish` lets the checkpoint complete.
struct ParkedCheckpoint {
    data: Arc<FakeIo>,
    file_set: Arc<MemWalFileSet>,
    finish: std::thread::JoinHandle<()>,
}

/// Drives a store to `keys` committed keys, then runs a checkpoint on another
/// thread and parks it at the header `datasync` — the commit point.
fn park_at_commit_point(keys: u32) -> ParkedCheckpoint {
    let data = Arc::new(FakeIo::new(0));
    let file_set = MemWalFileSet::new();
    let store = create_mem_store(dev(&data), Arc::clone(&file_set), Options::default()).unwrap();
    let map = BTreeMap::over(store.clone());
    for i in 0..keys {
        put(&map, format!("k{i:02}").as_bytes(), b"v");
    }
    // The checkpoint syncs the data device twice: pages+manifest, then the header.
    // Hold both, release the first, and it parks on the second.
    //
    // The device has already been synced by the create protocol, so "a sync
    // landed" must be measured against that baseline — not against zero, which
    // would let this fall through while the *pages* sync is still parked and
    // silently move the whole matrix off the commit point.
    let base_syncs = data.sync_count();
    data.arm_sync_gate(2);
    let finish = std::thread::spawn(move || {
        block_on(store.checkpoint()).unwrap();
        block_on(map.close()).unwrap();
    });
    while data.parked_sync_count() == 0 {
        std::thread::yield_now();
    }
    data.release_syncs(1);
    // Wait for the pages+manifest sync to actually fold, and only then for the
    // *next* sync to park: gate slot 0 is spent, so the sync parked after this
    // can only be the header commit.
    while data.sync_count() < base_syncs + 1 {
        std::thread::yield_now();
    }
    while data.parked_sync_count() == 0 {
        std::thread::yield_now();
    }
    // Assert we are where we claim: the *only* write in flight is one header slot.
    // Without this the matrix below could be deciding the fate of the page writes
    // instead of the commit, and would still pass.
    let in_flight = data.unsynced_ranges();
    assert_eq!(
        in_flight.len(),
        1,
        "expected exactly the header write in flight, got {in_flight:?}"
    );
    let (off, len) = in_flight[0];
    assert!(
        (off == 0 || off == PAGE_SIZE as u64) && len == PAGE_SIZE,
        "in-flight write {in_flight:?} is not a header slot — not parked at the commit point"
    );
    ParkedCheckpoint {
        data,
        file_set,
        finish,
    }
}

impl ParkedCheckpoint {
    /// Opens over a crash image of the data file in which the in-flight header
    /// write persisted as `decide`, and over the durable image of every segment
    /// that exists at this instant (the retired one is not yet unlinked).
    fn recover_with(&self, decide: Persist) -> Result<BTreeMap<Wal>, OpenError> {
        self.recover_torn(PAGE_SIZE, move |_| decide)
    }

    /// As `recover_with`, but decides the in-flight header write per
    /// `blast_radius`-byte granule — so a *partially* persisted slot (the real
    /// shape of a torn write: some granules new, some still old) can be built,
    /// not just an all-or-nothing one.
    fn recover_torn(
        &self,
        blast_radius: usize,
        decide: impl Fn(usize) -> Persist,
    ) -> Result<BTreeMap<Wal>, OpenError> {
        let img = self.data.crash_image(blast_radius, decide);
        let fresh = MemWalFileSet::new();
        for seq in self.file_set.live_segments() {
            let seg = self.file_set.segment(seq).unwrap();
            fresh.insert(seq, Arc::new(FakeIo::from_vec(seg.synced_bytes())));
        }
        let d = Arc::new(FakeIo::from_vec(img));
        open_mem_store(dev(&d), fresh, Options::default()).map(BTreeMap::over)
    }

    fn release(self) {
        self.data.release_syncs(1);
        self.finish.join().unwrap();
    }
}

/// Asserts every key `0..keys` is readable and the tree is structurally sound.
fn assert_all_readable(map: &BTreeMap<Wal>, keys: u32) {
    block_on(async {
        map.verify().await.expect("recovered tree verifies");
        for i in 0..keys {
            assert_eq!(
                map.get(format!("k{i:02}").into_bytes()).await.unwrap(),
                Some(b"v".to_vec()),
                "key k{i:02} lost"
            );
        }
    });
}

#[test]
fn crash_before_the_header_commits_recovers_the_pre_checkpoint_state() {
    // The header never reached stable storage → the *old* slot wins. The relocated
    // pages and manifest sitting in the data file are orphans no valid header
    // references; the retired segment is still linked and is what recovery scans.
    let parked = park_at_commit_point(30);
    let recovered = parked.recover_with(Persist::Old).unwrap();
    let m = recovered.store().metrics();
    assert_eq!(m.last_checkpoint_txid, 0, "no checkpoint is durable");
    assert_eq!(m.checkpoint_pages, 0, "the orphan pages are unreferenced");
    assert!(
        m.open_scan_bytes > WAL_HEADER_LEN,
        "recovery scanned the retired segment"
    );
    assert_all_readable(&recovered, 30);
    block_on(recovered.close()).unwrap();
    parked.release();
}

#[test]
fn crash_after_the_header_commits_recovers_the_post_checkpoint_state() {
    // The header landed → the *new* slot wins: the base root comes from the data
    // file and recovery scans the fresh (empty) segment named by that header. The
    // retired segment still exists — the unlink had not run — but nothing
    // references it.
    let parked = park_at_commit_point(30);
    let recovered = parked.recover_with(Persist::New).unwrap();
    let m = recovered.store().metrics();
    assert_eq!(m.last_checkpoint_txid, 30, "the checkpoint is durable");
    assert!(
        m.checkpoint_pages > 0,
        "the base root resolves in the data file"
    );
    assert_eq!(m.overlay_pages, 0, "the new segment holds no records yet");
    assert_eq!(
        m.open_scan_bytes, WAL_HEADER_LEN,
        "only the empty new segment was scanned"
    );
    assert_all_readable(&recovered, 30);
    block_on(recovered.close()).unwrap();
    parked.release();
}

#[test]
fn a_torn_header_slot_falls_back_to_the_previous_checkpoint() {
    // A torn slot fails its own checksum, so the *other* slot — the previous
    // checkpoint, whose segment is still linked — wins. This is the property that
    // makes the double slot worth having: no torn write can brick the store.
    let parked = park_at_commit_point(30);
    let recovered = parked.recover_with(Persist::Garble).unwrap();
    let m = recovered.store().metrics();
    assert_eq!(
        m.last_checkpoint_txid, 0,
        "the torn slot lost to its predecessor"
    );
    assert_all_readable(&recovered, 30);
    block_on(recovered.close()).unwrap();
    parked.release();
}

#[test]
fn every_crash_boundary_of_a_checkpoint_lands_on_one_side_or_the_other() {
    // The matrix as one statement: whatever a crash persists of the in-flight
    // header, the store recovers, verifies, and holds exactly the committed keys.
    // Only *which* side (pre- or post-checkpoint) differs — never the data.
    for decide in [Persist::Old, Persist::New, Persist::Garble] {
        let parked = park_at_commit_point(24);
        let recovered = parked
            .recover_with(decide)
            .unwrap_or_else(|e| panic!("{decide:?} must recover, got {e:?}"));
        assert_all_readable(&recovered, 24);
        assert_eq!(recovered.len(), 24);
        // The recovered store is writable and durable again.
        block_on(async {
            let v = apply(&recovered, b"after-crash", b"v");
            recovered.commit(v).await.unwrap();
            assert_eq!(
                recovered.get(b"after-crash".to_vec()).await.unwrap(),
                Some(b"v".to_vec())
            );
        });
        block_on(recovered.close()).unwrap();
        parked.release();
    }
}

#[test]
fn a_crashed_checkpoint_does_not_brick_the_next_one() {
    // A checkpoint that crashed before its commit point leaves the segment it had
    // already minted on disk as an orphan. The retry must reclaim that name: the
    // orphan is referenced by no valid header slot (had its header committed, this
    // store would have reopened at seq+1), so it is safe to drop — and it *must*
    // be dropped, or `create_new` refuses and, because a checkpoint I/O error
    // poisons, one crash would permanently brick the store's ability to checkpoint
    // at all.
    let parked = park_at_commit_point(20);
    let recovered = parked.recover_with(Persist::Old).unwrap();
    assert_eq!(
        recovered.store().metrics().last_checkpoint_txid,
        0,
        "pre-checkpoint"
    );

    let retry = block_on(recovered.store().checkpoint())
        .expect("a checkpoint retry after a crashed checkpoint must succeed");
    assert_eq!(
        retry.checkpoint_seq, 1,
        "the retry reuses the crashed attempt's seq"
    );
    assert!(retry.relocated_pages > 0);
    let m = recovered.store().metrics();
    assert_eq!(m.last_checkpoint_txid, 20);
    assert_eq!(m.checkpoints, 1);

    // The store is healthy — not poisoned — and still writable and durable.
    block_on(async {
        let v = apply(&recovered, b"after-retry", b"v");
        recovered.commit(v).await.unwrap();
        assert_eq!(
            recovered.get(b"after-retry".to_vec()).await.unwrap(),
            Some(b"v".to_vec())
        );
    });
    assert_all_readable(&recovered, 20);
    block_on(recovered.close()).unwrap();
    parked.release();
}

// -- Phase B: hostile-header validation --------------------------------------
//
// The data header is a trust boundary: a crafted or corrupted one must fail here,
// as `Corrupt`, and never as an arithmetic panic on the writer thread or a check
// that silently wraps past the value it exists to bound. Both cases below were
// found by probe programs feeding crafted headers.

#[test]
fn a_manifest_page_extent_near_the_address_ceiling_is_corrupt() {
    use crate::checkpoint::{encode_manifest, ManifestEntry};
    use crate::page::{Checksum, PageId, PageRef};
    // Page-aligned and near the top of the address space, so the extent check's
    // `offset + PAGE_SIZE` overflows: unchecked, that wraps to a small number and
    // *passes* — admitting a locator entry pointing outside the file (and
    // panicking outright in debug).
    let hostile = (u64::MAX / PAGE_SIZE as u64) * PAGE_SIZE as u64;
    let entries = vec![ManifestEntry {
        page_id: PageId(2),
        offset: hostile,
        checksum: Checksum(1),
    }];
    let (blob, ck) = encode_manifest(&entries);
    let h = CheckpointHeader {
        store_uuid: [1u8; 16],
        checkpoint_seq: 1,
        checkpoint_txid: 0,
        root: PageRef::EMPTY,
        entry_count: 0,
        allocator_frontier: 100,
        wal_segment_seq: 0,
        wal_scan_origin: WAL_HEADER_LEN,
        wal_chain_seed: 0,
        manifest_offset: 2 * PAGE_SIZE as u64,
        manifest_len: blob.len() as u64,
        manifest_cksum: ck,
    };
    let mut img = vec![0u8; 2 * PAGE_SIZE + blob.len()];
    img[2 * PAGE_SIZE..].copy_from_slice(&blob);
    let d = FakeIo::from_vec(img);
    assert!(
        matches!(block_on(read_manifest(&d, &h)), Err(OpenError::Corrupt(m)) if m.contains("overflow")),
        "an overflowing page extent must be Corrupt, not a panic or a wrapped pass"
    );
}

#[test]
fn a_header_at_the_sequence_bound_is_refused_at_open() {
    use crate::page::PageRef;
    // A forged seq of u64::MAX would make the writer's `seq + 1` wrap — a panic on
    // the writer thread, which poisons the store. Bound it at the trust boundary
    // instead, so it is a clean `Corrupt` at open.
    let uuid = [2u8; 16];
    let h = CheckpointHeader {
        store_uuid: uuid,
        checkpoint_seq: u64::MAX,
        checkpoint_txid: 0,
        root: PageRef::EMPTY,
        entry_count: 0,
        allocator_frontier: 2,
        wal_segment_seq: u64::MAX,
        wal_scan_origin: WAL_HEADER_LEN,
        wal_chain_seed: seed_checksum(uuid, u64::MAX),
        manifest_offset: 0,
        manifest_len: 0,
        manifest_cksum: 0,
    };
    assert!(
        matches!(CheckpointHeader::decode(&h.encode()), Err(OpenError::Corrupt(m)) if m.contains("sequence")),
        "a sequence at the exhaustion bound must be refused"
    );
    let mut img = vec![0u8; 2 * PAGE_SIZE];
    img[PAGE_SIZE..].copy_from_slice(&h.encode());
    let d = Arc::new(FakeIo::from_vec(img));
    let files = MemWalFileSet::new();
    files.insert(
        u64::MAX,
        Arc::new(FakeIo::from_vec(encode_wal_header(uuid, u64::MAX).to_vec())),
    );
    assert!(
        open_mem_store(dev(&d), files, Options::default()).is_err(),
        "the store must refuse to open rather than panic its writer on the next checkpoint"
    );
}

#[test]
fn a_forged_scan_origin_outside_the_segment_is_corrupt() {
    // Every header the engine writes uses exactly WAL_HEADER_LEN, so an origin past
    // the segment end is corruption. Unbounded, it opens "successfully" and seeds
    // the append frontier and the meter past EOF, failing every write with
    // StoreFull.
    let (map, data, file_set) = owned_store();
    put(&map, b"k", b"v");
    block_on(map.close()).unwrap();
    let mut img = data.synced_bytes();
    let h = header_of(img.clone());
    let forged = CheckpointHeader {
        wal_scan_origin: 1 << 60,
        ..h
    };
    img[..PAGE_SIZE].copy_from_slice(&forged.encode());
    let d = Arc::new(FakeIo::from_vec(img));
    let fresh = MemWalFileSet::new();
    for seq in file_set.live_segments() {
        fresh.insert(
            seq,
            Arc::new(FakeIo::from_vec(
                file_set.segment(seq).unwrap().synced_bytes(),
            )),
        );
    }
    let err = open_mem_store(dev(&d), fresh, Options::default())
        .err()
        .expect("must refuse");
    assert!(
        matches!(err, OpenError::Corrupt(m) if m.contains("scan origin")),
        "got {err:?}"
    );
}

#[test]
fn a_crash_between_commit_and_unlink_does_not_leak_the_retired_segment() {
    // The rotation's unlink runs *after* the header commit, so a crash in between
    // leaves the retired segment behind — and nothing would ever remove it: the
    // next checkpoint retires only its own predecessor. That is up to
    // `max_wal_bytes` pinned forever, per crash, with no holder to free it — which
    // would falsify the retention claim. Open completes the interrupted rotation
    // instead.
    let parked = park_at_commit_point(20);
    // Crash with the header durable but the unlink never reached.
    let (recovered, data2, fs2) = {
        let img = parked.data.crash_image(PAGE_SIZE, |_| Persist::New);
        let fresh = MemWalFileSet::new();
        for seq in parked.file_set.live_segments() {
            let seg = parked.file_set.segment(seq).unwrap();
            fresh.insert(seq, Arc::new(FakeIo::from_vec(seg.synced_bytes())));
        }
        assert_eq!(
            fresh.live_segments(),
            vec![0, 1],
            "the crash left both segments"
        );
        let d = Arc::new(FakeIo::from_vec(img));
        let s = open_mem_store(dev(&d), Arc::clone(&fresh), Options::default()).unwrap();
        (BTreeMap::over(s), d, fresh)
    };
    // Open completed the rotation the crash interrupted.
    assert_eq!(
        fs2.live_segments(),
        vec![1],
        "the retired segment must not survive the reopen"
    );
    assert_all_readable(&recovered, 20);

    // And it stays swept as checkpoints continue — no accumulation across crashes.
    put(&recovered, b"more", b"v");
    block_on(recovered.store().checkpoint()).unwrap();
    assert_eq!(
        fs2.live_segments(),
        vec![2],
        "only the segment in force survives"
    );
    assert_eq!(recovered.store().metrics().leaked_segments, 0);
    block_on(recovered.close()).unwrap();
    let _ = data2;
    parked.release();
}

#[test]
fn a_partially_persisted_header_slot_lands_on_one_side_at_every_blast_radius() {
    // The realistic torn write: the device persisted *some* granules of the new
    // header and left the rest holding the old bytes. The granule size a device
    // tears at is not ours to choose, so enumerate several.
    //
    // The invariant is **one side or the other, never a mix** — not "always falls
    // back". A header's payload is only the first 128 bytes (the rest of the slot
    // is zero padding), so any tear granule at or above that size is all-or-nothing
    // for the whole payload: the slot comes out either fully valid (post-checkpoint
    // wins) or fully zero (invalid → the predecessor wins). Both are correct
    // outcomes; asserting one of them would be asserting an accident of the
    // granule size.
    for radius in [512usize, 4096, PAGE_SIZE / 2] {
        let parked = park_at_commit_point(16);
        let recovered = parked
            .recover_torn(radius, |g| {
                if g % 2 == 0 {
                    Persist::New
                } else {
                    Persist::Old
                }
            })
            .unwrap_or_else(|e| panic!("radius {radius}: must recover, got {e:?}"));
        let t = recovered.store().metrics().last_checkpoint_txid;
        assert!(
            t == 0 || t == 16,
            "radius {radius}: recovered to neither side (last_checkpoint_txid = {t})"
        );
        assert_all_readable(&recovered, 16);
        assert_eq!(recovered.len(), 16);
        block_on(recovered.close()).unwrap();
        parked.release();
    }
}

#[test]
fn a_header_torn_within_its_payload_is_rejected() {
    // The case the coarse radii above cannot construct: a granule *smaller* than
    // the 128-byte payload, so the slot really does end up half new and half old.
    // That must fail its own checksum and lose to the predecessor — the property
    // the double slot exists for.
    let parked = park_at_commit_point(16);
    // 64-byte granules: the payload spans two of them, and only the first lands.
    let recovered = parked
        .recover_torn(64, |g| {
            if g % 2 == 0 {
                Persist::New
            } else {
                Persist::Old
            }
        })
        .expect("a torn slot must not brick the store");
    assert_eq!(
        recovered.store().metrics().last_checkpoint_txid,
        0,
        "a slot torn *within* its payload must lose to its predecessor"
    );
    assert_all_readable(&recovered, 16);
    block_on(recovered.close()).unwrap();
    parked.release();
}

#[test]
fn a_checkpoint_queued_before_maintenance_fails_fast_rather_than_waiting() {
    // Gating at enqueue closes only half the race: the phase can flip *after* the
    // command is already in the channel, and a parked writer dequeues nothing — so
    // the checkpoint would wait on an operator, not on I/O, which is precisely what
    // `CheckpointError::Compacting` promises cannot happen.
    let data = Arc::new(FakeIo::new(0));
    let file_set = MemWalFileSet::new();
    let store = create_mem_store(dev(&data), Arc::clone(&file_set), Options::default()).unwrap();
    let map = BTreeMap::over(store.clone());
    put(&map, b"k", b"v");

    // Pin the writer inside a commit's datasync so it cannot reach the command.
    let seg = file_set.segment(0).unwrap();
    seg.arm_sync_gate(1);
    let v = apply(&map, b"k2", b"v2");
    wait_until_mid_sync(&seg);

    // Enqueue the checkpoint while still Running (it lands in the channel), then
    // pause. The writer will see Maintenance before it ever dequeues.
    let mut ckpt = Box::pin(store.checkpoint());
    assert!(
        block_on(poll_once(&mut ckpt)).is_none(),
        "queued, not yet run"
    );
    assert!(store.begin_maintenance());

    // Release: the writer finishes the commit, sees Maintenance, and must fail the
    // queued checkpoint instead of parking with it still in the channel.
    seg.release_syncs(1);
    let outcome = block_on(ckpt);
    assert!(
        matches!(outcome, Err(CheckpointError::Compacting)),
        "a checkpoint overtaken by a pause must fail fast, got {outcome:?}"
    );

    // The write admitted before the pause is not lost — it is held in order and
    // applied on resume.
    store.end_maintenance();
    block_on(map.commit(v)).unwrap();
    block_on(async {
        assert_eq!(map.get(b"k2".to_vec()).await.unwrap(), Some(b"v2".to_vec()));
    });
    let again = block_on(store.checkpoint());
    assert!(again.is_ok(), "and checkpoints work again, got {again:?}");
    block_on(map.close()).unwrap();
}

#[test]
fn a_header_at_the_txid_bound_is_refused_at_open() {
    // A forged `checkpoint_txid` seeds the *visible version*, and the next write's
    // `Version::next` does `txid + 1`: unbounded, that panics the writer thread in
    // debug and silently wraps in release, breaking the monotonic-txid invariant
    // the whole receipt system rests on. Reject it at the trust boundary instead.
    use crate::page::PageRef;
    let uuid = [3u8; 16];
    let h = CheckpointHeader {
        store_uuid: uuid,
        checkpoint_seq: 1,
        checkpoint_txid: u64::MAX,
        root: PageRef::EMPTY,
        entry_count: 0,
        allocator_frontier: 2,
        wal_segment_seq: 0,
        wal_scan_origin: WAL_HEADER_LEN,
        wal_chain_seed: seed_checksum(uuid, 0),
        manifest_offset: 0,
        manifest_len: 0,
        manifest_cksum: 0,
    };
    assert!(
        matches!(CheckpointHeader::decode(&h.encode()), Err(OpenError::Corrupt(m)) if m.contains("txid")),
        "a txid at the exhaustion bound must be refused"
    );
    // And the store refuses to open rather than publishing an un-incrementable
    // visible version.
    let mut img = vec![0u8; 2 * PAGE_SIZE];
    img[PAGE_SIZE..].copy_from_slice(&h.encode());
    let d = Arc::new(FakeIo::from_vec(img));
    let files = MemWalFileSet::new();
    files.insert(
        0,
        Arc::new(FakeIo::from_vec(encode_wal_header(uuid, 0).to_vec())),
    );
    assert!(open_mem_store(dev(&d), files, Options::default()).is_err());
}

// ---------------------------------------------------------------------------
// M6 Phase E: revocable generation leases.
//
// A snapshot no longer holds the generation's files irrevocably: it holds a
// counted hold on the generation's revocable holder, and each read runs under a
// short-lived lease. `Options::max_retained_generations` bounds how many retired
// generations snapshots may pin (a space bound of that many retired WAL
// segments, NOT a wall-clock age); past it, the oldest is revoked and its
// snapshots read `SnapshotExpired`. The lease-drain protocol itself (a lease
// acquired an instant before revocation keeps its device until it drains) is
// pinned by unit tests in `retention.rs`; these exercise the store-level wiring.
// ---------------------------------------------------------------------------

/// An owned store with an explicit retention bound.
fn owned_store_bounded(bound: Option<u32>) -> (BTreeMap<Wal>, Arc<FakeIo>, Arc<MemWalFileSet>) {
    let data = Arc::new(FakeIo::new(0));
    let file_set = MemWalFileSet::new();
    let options = Options {
        max_retained_generations: bound,
        ..Options::default()
    };
    let store = create_mem_store(dev(&data), Arc::clone(&file_set), options).unwrap();
    (BTreeMap::over(store), data, file_set)
}

/// The headline Phase E behavior: a snapshot pushed past the retention bound by
/// newer pinned generations expires — its reads return `SnapshotExpired`, its
/// retired segment's bytes actually free, and the store stays healthy — while a
/// within-bound snapshot stays fully readable across the same checkpoints.
#[test]
fn snapshot_past_the_retention_bound_expires_and_frees_its_segment() {
    use crate::error::ReadError;
    let (map, _data, file_set) = owned_store_bounded(Some(1));
    for i in 0..20u32 {
        put(&map, format!("k{i:02}").as_bytes(), b"v1");
    }
    let snap0 = map.snapshot();
    let seg0 = Arc::downgrade(&file_set.segment(0).unwrap());

    // Checkpoint 1 retires generation 0; one pinned retained generation is
    // within the bound, so the snapshot stays *fully* readable (every key).
    block_on(map.store().checkpoint()).unwrap();
    assert!(!snap0.is_expired());
    block_on(async {
        for i in 0..20u32 {
            assert_eq!(
                snap0.get(format!("k{i:02}").into_bytes()).await.unwrap(),
                Some(b"v1".to_vec()),
                "a within-bound snapshot must stay fully readable across a checkpoint"
            );
        }
    });
    let m = map.metrics();
    assert_eq!(m.retained_generations, 1);
    assert!(
        m.retained_segment_bytes > 0,
        "the pinned retired segment's bytes are attributed"
    );
    assert_eq!(m.expired_generations, 0);

    // A second snapshot pins generation 1; checkpoint 2 retires it. Two pinned
    // retained generations exceed the bound → the oldest (gen 0) is revoked.
    put(&map, b"newer", b"v2");
    let snap1 = map.snapshot();
    block_on(map.store().checkpoint()).unwrap();

    assert!(snap0.is_expired(), "oldest snapshot expired past the bound");
    assert!(!snap1.is_expired(), "newest snapshot retained");
    let err = block_on(snap0.get(b"k00".to_vec())).unwrap_err();
    assert!(
        matches!(err, ReadError::SnapshotExpired),
        "expired snapshot reads fail with SnapshotExpired, got {err:?}"
    );
    assert!(
        seg0.upgrade().is_none(),
        "the expired snapshot's retired segment bytes must actually free"
    );
    // Expiration is per-snapshot-generation, not per-store: the newer snapshot
    // and the live map read on.
    block_on(async {
        assert_eq!(
            snap1.get(b"newer".to_vec()).await.unwrap(),
            Some(b"v2".to_vec())
        );
        assert_eq!(
            map.get(b"k00".to_vec()).await.unwrap(),
            Some(b"v1".to_vec())
        );
    });
    // The expired snapshot's plain copies still answer; only reads are gone.
    assert_eq!(snap0.len(), 20);
    let m = map.metrics();
    assert_eq!(m.expired_generations, 1);
    assert_eq!(m.retained_generations, 1, "gen 1 still pinned by snap1");
    block_on(map.close()).unwrap();
}

/// A pending snapshot-read future takes its own hold at `get()` call: dropping
/// the `Snapshot` first neither invalidates the read nor lets the rebind free
/// the generation under it.
#[test]
fn pending_snapshot_read_survives_snapshot_drop_and_checkpoint() {
    let (map, _data, _file_set) = owned_store_bounded(Some(2));
    for i in 0..10u32 {
        put(&map, format!("k{i:02}").as_bytes(), b"v");
    }
    let snap = map.snapshot();
    let pending = snap.get(b"k03".to_vec());
    drop(snap);

    // The retired generation is pinned by the pending read's hold alone.
    block_on(map.store().checkpoint()).unwrap();
    assert_eq!(map.metrics().retained_generations, 1);

    assert_eq!(block_on(pending).unwrap(), Some(b"v".to_vec()));
    block_on(map.close()).unwrap();
}

/// The dual: cancelling the pending read (dropping the future unpolled) drops
/// the last hold, and the retired generation releases naturally — no policy
/// pressure needed, and the segment's bytes free on the spot.
#[test]
fn cancelled_pending_read_releases_the_retired_generation() {
    let (map, _data, file_set) = owned_store_bounded(Some(8));
    for i in 0..10u32 {
        put(&map, format!("k{i:02}").as_bytes(), b"v");
    }
    let snap = map.snapshot();
    let pending = snap.get(b"k03".to_vec());
    drop(snap);
    let seg0 = Arc::downgrade(&file_set.segment(0).unwrap());

    block_on(map.store().checkpoint()).unwrap();
    assert_eq!(
        map.metrics().retained_generations,
        1,
        "pending read pins gen 0"
    );
    assert!(seg0.upgrade().is_some());

    drop(pending);
    assert!(
        seg0.upgrade().is_none(),
        "dropping the last hold releases the retired generation immediately"
    );
    assert_eq!(map.metrics().retained_generations, 0);
    assert_eq!(
        map.metrics().expired_generations,
        0,
        "natural release, not a revocation"
    );
    block_on(map.close()).unwrap();
}

/// `max_retained_generations: None` opts out: snapshots stay readable across
/// arbitrarily many checkpoints and retained space grows — the documented
/// unbounded policy, a deliberate product-decision escape hatch.
#[test]
fn unbounded_retention_never_expires_snapshots() {
    let (map, _data, _file_set) = owned_store_bounded(None);
    put(&map, b"a", b"v");
    let mut snaps = Vec::new();
    for round in 0..3u32 {
        snaps.push(map.snapshot());
        put(&map, format!("r{round}").as_bytes(), b"v");
        block_on(map.store().checkpoint()).unwrap();
    }
    for snap in &snaps {
        assert!(!snap.is_expired());
        assert_eq!(
            block_on(snap.get(b"a".to_vec())).unwrap(),
            Some(b"v".to_vec()),
            "unbounded retention keeps every snapshot readable"
        );
    }
    let m = map.metrics();
    assert_eq!(
        m.retained_generations, 3,
        "every retired generation stays pinned"
    );
    assert_eq!(m.expired_generations, 0);
    block_on(map.close()).unwrap();
}

/// A retired generation nobody captured releases at the rebind itself: no
/// retained bytes, nothing for the policy to do. (The store's own descriptor
/// handle is not a snapshot hold.)
#[test]
fn uncaptured_generations_release_at_the_rebind() {
    let (map, _data, _file_set) = owned_store_bounded(Some(1));
    for round in 0..3u32 {
        put(&map, format!("r{round}").as_bytes(), b"v");
        block_on(map.store().checkpoint()).unwrap();
    }
    let m = map.metrics();
    assert_eq!(m.retained_generations, 0, "no snapshot → nothing retained");
    assert_eq!(m.retained_segment_bytes, 0);
    assert_eq!(m.expired_generations, 0, "bound never engaged");
    block_on(map.close()).unwrap();
}

// ---- Phase E hardening -----------------------------------------------------

/// A `PageIo` adapter that parks the single next read issued after
/// [`arm_next_read`](GatedReads::arm_next_read) until
/// [`release_reads`](GatedReads::release_reads), delegating everything else.
/// One-shot by design: the snapshot read under test claims the gate, while the
/// checkpoint's own relocation reads (same device, same offsets) pass through —
/// that selectivity is why FakeIo itself has no read gate.
struct GatedReads {
    inner: Arc<dyn PageIo>,
    claim: AtomicBool,
    held: AtomicBool,
    parked: std::sync::atomic::AtomicU64,
    event: event_listener::Event,
}

impl GatedReads {
    fn new(inner: Arc<dyn PageIo>) -> Arc<Self> {
        Arc::new(GatedReads {
            inner,
            claim: AtomicBool::new(false),
            held: AtomicBool::new(false),
            parked: std::sync::atomic::AtomicU64::new(0),
            event: event_listener::Event::new(),
        })
    }
    fn arm_next_read(&self) {
        self.held.store(true, Ordering::Release);
        self.claim.store(true, Ordering::Release);
    }
    fn release_reads(&self) {
        self.held.store(false, Ordering::Release);
        self.event.notify(usize::MAX);
    }
    fn parked(&self) -> u64 {
        self.parked.load(Ordering::Acquire)
    }
}

impl PageIo for GatedReads {
    fn read_exact_at(
        &self,
        offset: u64,
        len: usize,
    ) -> crate::io::IoFuture<'_, std::io::Result<Box<[u8]>>> {
        Box::pin(async move {
            if self.claim.swap(false, Ordering::AcqRel) {
                self.parked.fetch_add(1, Ordering::AcqRel);
                loop {
                    if !self.held.load(Ordering::Acquire) {
                        break;
                    }
                    let listener = self.event.listen();
                    if !self.held.load(Ordering::Acquire) {
                        break;
                    }
                    listener.await;
                }
                self.parked.fetch_sub(1, Ordering::AcqRel);
            }
            self.inner.read_exact_at(offset, len).await
        })
    }
    fn write_all_at(
        &self,
        offset: u64,
        buffer: Box<[u8]>,
    ) -> crate::io::IoFuture<'_, crate::io::BufResult> {
        self.inner.write_all_at(offset, buffer)
    }
    fn datasync(&self) -> crate::io::IoFuture<'_, std::io::Result<()>> {
        self.inner.datasync()
    }
    fn size(&self) -> crate::io::IoFuture<'_, std::io::Result<u64>> {
        self.inner.size()
    }
    fn shutdown(&self) -> crate::io::IoFuture<'_, std::io::Result<()>> {
        self.inner.shutdown()
    }
}

/// A `WalFileSet` that wraps segment 0's device in [`GatedReads`], delegating
/// the namespace to a real [`MemWalFileSet`].
struct GatingFileSet {
    inner: Arc<MemWalFileSet>,
    gate0: parking_lot::Mutex<Option<Arc<GatedReads>>>,
}

impl WalFileSet for GatingFileSet {
    fn create_segment(&self, seq: u64, store_uuid: [u8; 16]) -> std::io::Result<Arc<dyn PageIo>> {
        let dev = self.inner.create_segment(seq, store_uuid)?;
        if seq == 0 {
            let gated = GatedReads::new(dev);
            *self.gate0.lock() = Some(Arc::clone(&gated));
            Ok(gated as Arc<dyn PageIo>)
        } else {
            Ok(dev)
        }
    }
    fn open_segment(&self, seq: u64) -> std::io::Result<Arc<dyn PageIo>> {
        self.inner.open_segment(seq)
    }
    fn unlink_segment(&self, seq: u64) -> std::io::Result<()> {
        self.inner.unlink_segment(seq)
    }
    fn retire_other_segments(&self, keep: u64) -> std::io::Result<u64> {
        self.inner.retire_other_segments(keep)
    }
}

/// The store-level invariant-8 wiring test: a snapshot read
/// parked **mid-traversal** — lease held, device I/O in flight — survives a
/// checkpoint whose bound revokes its generation at that very moment. The read
/// completes with the correct value, the segment's bytes stay pinned (and
/// metric-visible as *draining*) until the lease drains, and they free at the
/// drain.
#[test]
fn a_leased_read_survives_revocation_mid_traversal() {
    use crate::error::ReadError;
    let data = Arc::new(FakeIo::new(0));
    let mem = MemWalFileSet::new();
    let fs = Arc::new(GatingFileSet {
        inner: Arc::clone(&mem),
        gate0: parking_lot::Mutex::new(None),
    });
    let store_uuid = crate::direct::generate_store_uuid();
    let wal = fs.create_segment(0, store_uuid).unwrap();
    let store = Store::<Wal>::create_with_io_owned(
        dev(&data),
        wal,
        store_uuid,
        Some(Arc::clone(&fs) as Arc<dyn WalFileSet>),
        None,
        Options {
            max_retained_generations: Some(0),
            // Force real device reads: with one cache slot, a descent must load
            // at least one page from the device, so the gate engages.
            cache_capacity: Some(1),
            ..Options::default()
        },
        None,
    )
    .unwrap();
    let map = BTreeMap::over(store);
    // Values big enough that the tree spans several pages: a one-slot cache
    // then cannot hold a whole descent, so the read must hit the device.
    let big = vec![7u8; 800];
    for i in 0..40u32 {
        put(&map, format!("k{i:02}").as_bytes(), &big);
    }
    let snap = map.snapshot();
    // Overwrite the probed key (and its path) AFTER the snapshot: the
    // snapshot's descent now reads *superseded* pages, which the checkpoint's
    // relocation walk never touches. This matters beyond realism — the walk
    // shares the page cache's single-flight, so if the parked read targeted a
    // live-set page, the checkpoint would join that same gated load and the
    // writer would deadlock behind the test's own gate.
    put(&map, b"k07", b"v2-overwrite");
    let seg0 = Arc::downgrade(&mem.segment(0).unwrap());
    let gate = fs.gate0.lock().clone().unwrap();

    // Park the snapshot read mid-traversal: lease acquired, device read in
    // flight on the gen-0 WAL segment.
    gate.arm_next_read();
    let reader = std::thread::spawn({
        let snap = snap.clone();
        move || block_on(snap.get(b"k07".to_vec()))
    });
    let mut spins = 0u64;
    while gate.parked() == 0 {
        std::thread::yield_now();
        spins += 1;
        assert!(
            spins < 100_000_000,
            "snapshot read never reached the device"
        );
    }

    // Checkpoint retires gen 0; bound 0 revokes it immediately — while the
    // read's lease is in flight.
    block_on(map.store().checkpoint()).unwrap();
    assert!(snap.is_expired(), "revoked while a lease drains");
    let m = map.metrics();
    assert_eq!(m.retained_generations, 0);
    assert_eq!(
        m.draining_generations, 1,
        "the draining generation stays metric-visible"
    );
    assert!(
        m.retained_segment_bytes > 0,
        "draining bytes are still pinned and still counted"
    );
    assert_eq!(m.expired_generations, 1);
    assert!(
        seg0.upgrade().is_some(),
        "the leased read's device must stay alive through the revocation"
    );
    // New reads on the expired snapshot refuse; the in-flight one is untouched.
    assert!(matches!(
        block_on(snap.get(b"k01".to_vec())),
        Err(ReadError::SnapshotExpired)
    ));

    // Release the gate: the parked read completes correctly, the lease drains,
    // and the drain performs the release — the segment's bytes free.
    gate.release_reads();
    assert_eq!(
        reader.join().unwrap().unwrap(),
        Some(big),
        "a read that held its lease across the revocation completes correctly \
         and sees the snapshot's pre-overwrite value"
    );
    // Drop the test's own wrapper handles: `GatedReads.inner` holds the segment
    // device, so the fixture itself pins it — after this, any survival would be
    // the engine's pin.
    *fs.gate0.lock() = None;
    drop(gate);
    assert!(
        seg0.upgrade().is_none(),
        "the segment frees exactly at the lease drain"
    );
    let m = map.metrics();
    assert_eq!(m.draining_generations, 0);
    assert_eq!(m.retained_segment_bytes, 0);
    block_on(map.close()).unwrap();
}

/// `Some(0)` semantics, pinned: no retired generation may
/// stay pinned at all — a held snapshot expires at the very next checkpoint and
/// its segment frees on the spot (no lease in flight → release inside retire).
#[test]
fn bound_zero_expires_every_snapshot_at_the_next_checkpoint() {
    use crate::error::ReadError;
    let (map, _data, file_set) = owned_store_bounded(Some(0));
    for i in 0..10u32 {
        put(&map, format!("k{i:02}").as_bytes(), b"v");
    }
    let snap = map.snapshot();
    let seg0 = Arc::downgrade(&file_set.segment(0).unwrap());

    block_on(map.store().checkpoint()).unwrap();
    assert!(snap.is_expired());
    assert!(matches!(
        block_on(snap.get(b"k00".to_vec())),
        Err(ReadError::SnapshotExpired)
    ));
    assert!(
        seg0.upgrade().is_none(),
        "with no lease in flight, bound 0 frees the segment at the checkpoint"
    );
    let m = map.metrics();
    assert_eq!(
        (
            m.retained_generations,
            m.draining_generations,
            m.expired_generations
        ),
        (0, 0, 1)
    );
    // The store and fresh snapshots are unaffected.
    block_on(async {
        assert_eq!(map.get(b"k00".to_vec()).await.unwrap(), Some(b"v".to_vec()));
    });
    assert!(!map.snapshot().is_expired());
    block_on(map.close()).unwrap();
}

// ---------------------------------------------------------------------------
// M6 Phase D — data-file compaction.
// ---------------------------------------------------------------------------

use crate::backend::Direct;
use crate::compact::MemDataHost;
use crate::fileset::{create_mem_store_hosted, open_mem_store_hosted};

/// A fresh hosted StoreDirect over an inspectable `MemDataHost` (so it can
/// compact and a test can reach the active/temp devices).
fn hosted_direct_store(options: Options) -> (BTreeMap<Direct>, Arc<MemDataHost>) {
    let data = Arc::new(FakeIo::new(0));
    let host = MemDataHost::new(Arc::clone(&data));
    let store = Store::<Direct>::create_with_io_hosted(
        dev(&data),
        Some(Arc::clone(&host) as Arc<dyn crate::compact::DataFileHost>),
        options,
        None,
    )
    .unwrap();
    (BTreeMap::over(store), host)
}

/// A fresh hosted StoreWal over a `MemDataHost` + `MemWalFileSet`.
fn hosted_wal_store(options: Options) -> (BTreeMap<Wal>, Arc<MemDataHost>, Arc<MemWalFileSet>) {
    let data = Arc::new(FakeIo::new(0));
    let host = MemDataHost::new(data);
    let file_set = MemWalFileSet::new();
    let store = create_mem_store_hosted(Arc::clone(&host), Arc::clone(&file_set), options).unwrap();
    (BTreeMap::over(store), host, file_set)
}

/// Applies one insert on a Direct map (durability advances at idle flush; a
/// compaction establishes its own flushed barrier).
fn put_direct(map: &BTreeMap<Direct>, k: &[u8], v: &[u8]) {
    block_on(async {
        let out = map
            .apply(WriteBatch::new().insert(k.to_vec(), v.to_vec()))
            .await
            .unwrap();
        assert!(matches!(out, ApplyOutcome::Applied { .. }));
    });
}

/// The expected content of the churned fixture: `k00..k39` with the overwrite
/// generation `gen`, minus removed keys `k00`/`k07`.
fn probe_churned<B: crate::backend::Backend>(map: &BTreeMap<B>, gen: u8) {
    block_on(async {
        for i in 0..40u32 {
            let key = format!("k{i:02}").into_bytes();
            let got = map.get(key).await.unwrap();
            if i == 0 || i == 7 {
                assert_eq!(got, None, "removed key k{i:02} must stay gone");
            } else {
                assert_eq!(got, Some(vec![gen ^ i as u8; 800]), "key k{i:02}");
            }
        }
    });
}

/// Writes the churned fixture: insert 40 keys, overwrite them all (CoW garbage),
/// remove two — so the data file holds plenty of superseded pages.
fn churn_direct(map: &BTreeMap<Direct>, gen: u8) {
    for i in 0..40u32 {
        put_direct(map, format!("k{i:02}").as_bytes(), &[1 ^ i as u8; 800]);
    }
    for i in 0..40u32 {
        put_direct(map, format!("k{i:02}").as_bytes(), &[gen ^ i as u8; 800]);
    }
    block_on(async {
        map.apply(
            WriteBatch::new()
                .remove(b"k00".to_vec())
                .remove(b"k07".to_vec()),
        )
        .await
        .unwrap();
    });
}

fn churn_wal(map: &BTreeMap<Wal>, gen: u8) {
    for i in 0..40u32 {
        put(map, format!("k{i:02}").as_bytes(), &[1 ^ i as u8; 800]);
    }
    for i in 0..40u32 {
        put(map, format!("k{i:02}").as_bytes(), &[gen ^ i as u8; 800]);
    }
    block_on(async {
        let out = map
            .apply(
                WriteBatch::new()
                    .remove(b"k00".to_vec())
                    .remove(b"k07".to_vec()),
            )
            .await
            .unwrap();
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!()
        };
        map.commit(version).await.unwrap();
    });
}

/// Direct: a compaction rebuilds a dense file containing only live pages, the
/// content is unchanged, the superseded inode's bytes free (no snapshot holds
/// it), and the store keeps writing — including across a reopen that seeds the
/// v2 header's id delta.
#[test]
fn direct_compaction_rebuilds_dense_and_preserves_content() {
    let (map, host) = hosted_direct_store(Options::default());
    churn_direct(&map, 3);

    let old = host.active();
    let old_weak = Arc::downgrade(&old);
    drop(old);

    let report = block_on(map.store().compact()).unwrap();
    assert!(report.live_pages > 0);
    assert!(
        report.data_bytes_after < report.data_bytes_before,
        "a churned store must shrink: {report:?}"
    );
    // The promoted file is exactly headers + live pages — dense.
    assert_eq!(
        host.active().snapshot_bytes().len() as u64,
        report.data_bytes_after
    );
    assert_eq!(
        host.active().snapshot_bytes().len() as u64,
        (2 + report.live_pages) * crate::page::PAGE_SIZE as u64
    );
    assert!(host.temp().is_none(), "temp slot consumed by the promotion");
    // No snapshot holds the superseded generation: its inode frees at the
    // rebind's retire (M6 Phase E natural release).
    assert!(
        old_weak.upgrade().is_none(),
        "superseded data inode must free"
    );
    assert_eq!(map.metrics().compactions, 1);

    // Content identical, and the store keeps accepting writes.
    probe_churned(&map, 3);
    put_direct(&map, b"after", b"v");
    block_on(async {
        assert_eq!(
            map.get(b"after".to_vec()).await.unwrap(),
            Some(b"v".to_vec())
        );
    });
    block_on(map.close()).unwrap();

    // Reopen over the compacted file: the v2 header's delta seeds the locator
    // and allocator; the full tree verifies at open.
    let map = BTreeMap::over(
        Store::<Direct>::open_over(dev(&host.active()), Options::default()).unwrap(),
    );
    block_on(async {
        assert_eq!(
            map.get(b"k05".to_vec()).await.unwrap(),
            Some(vec![3 ^ 5u8; 800])
        );
        assert_eq!(
            map.get(b"after".to_vec()).await.unwrap(),
            Some(b"v".to_vec())
        );
        assert_eq!(map.get(b"k07".to_vec()).await.unwrap(), None);
    });
    put_direct(&map, b"post-reopen", b"w");
    block_on(map.close()).unwrap();
}

/// M6 Phase H — the maintenance metrics track the live set, reclaimed bytes, and
/// generation lineage across compactions. Direct arm: `data_live_pages` is
/// unknown (`0`) until the first compaction walks the live set, then equals the
/// compaction's `live_pages` and fits inside the file's total page count;
/// `reclaimed_bytes` accumulates the space each shrinking compaction frees; and
/// `generations_created` advances one per rebind.
#[test]
fn direct_maintenance_metrics_track_live_reclaim_and_generations() {
    let (map, _host) = hosted_direct_store(Options::default());
    let fresh = map.metrics();
    assert_eq!(
        fresh.generations_created, 1,
        "open mints exactly one generation"
    );
    assert_eq!(fresh.reclaimed_bytes, 0);
    assert_eq!(
        fresh.data_live_pages, 0,
        "no compaction has walked the live set yet — 0 means unknown, not empty"
    );

    churn_direct(&map, 3);
    let first = block_on(map.store().compact()).unwrap();
    let m1 = map.metrics();
    assert_eq!(m1.generations_created, 2, "one rebind past open");
    assert_eq!(
        m1.data_live_pages, first.live_pages,
        "the live gauge is the compaction's walked live-page count"
    );
    let total_pages = m1.data_physical_bytes / crate::page::PAGE_SIZE as u64;
    assert!(
        m1.data_live_pages <= total_pages && m1.data_live_pages > 0,
        "live {} must be a positive fraction of total {total_pages}",
        m1.data_live_pages
    );
    assert_eq!(
        m1.reclaimed_bytes,
        first.data_bytes_before - first.data_bytes_after,
        "reclaimed = the bytes the dense rewrite dropped"
    );

    // A second compaction with no intervening writes reclaims (near) nothing but
    // still advances the lineage and updates the live gauge; reclaimed is
    // monotonic.
    churn_direct(&map, 2);
    let second = block_on(map.store().compact()).unwrap();
    let m2 = map.metrics();
    assert_eq!(m2.generations_created, 3);
    assert_eq!(m2.data_live_pages, second.live_pages);
    assert!(
        m2.reclaimed_bytes >= m1.reclaimed_bytes,
        "reclaimed never goes backwards"
    );
    block_on(map.close()).unwrap();
}

/// M6 Phase H — the WAL arm of the maintenance metrics. A checkpoint *appends*
/// the live set into the data file, so it reclaims nothing but publishes a fresh
/// generation and sets `data_live_pages` to exactly `checkpoint_pages` (the field
/// exists so the same ratio reads on Direct, where `checkpoint_pages` stays 0).
#[test]
fn wal_checkpoint_metrics_track_live_set_and_generations() {
    let (map, _data, _file_set) = owned_store();
    for i in 0..60u32 {
        put(&map, format!("k{i:03}").as_bytes(), &vec![i as u8; 500]);
    }
    let before = map.store().metrics();
    assert_eq!(before.generations_created, 1);
    assert_eq!(before.data_live_pages, 0);

    let report = block_on(map.store().checkpoint()).unwrap();
    let after = map.store().metrics();
    assert_eq!(
        after.generations_created, 2,
        "the checkpoint rebind publishes a generation"
    );
    assert_eq!(
        after.data_live_pages, report.relocated_pages,
        "the live gauge is the pages the checkpoint relocated"
    );
    assert_eq!(
        after.data_live_pages, after.checkpoint_pages,
        "on WAL the live gauge and checkpoint_pages agree"
    );
    assert_eq!(
        after.reclaimed_bytes, 0,
        "a checkpoint appends live pages — it reclaims nothing"
    );
    block_on(map.close()).unwrap();
}

/// Direct: a pre-compaction snapshot stays fully readable across the cutover
/// (it reads the superseded inode through its retained generation), and the
/// inode frees exactly when the snapshot drops.
#[test]
fn direct_snapshot_stays_readable_across_compaction() {
    let (map, host) = hosted_direct_store(Options::default());
    churn_direct(&map, 3);
    let snap = map.snapshot();

    let old = host.active();
    let old_weak = Arc::downgrade(&old);
    drop(old);

    block_on(map.store().compact()).unwrap();
    assert!(
        old_weak.upgrade().is_some(),
        "a held snapshot must pin the superseded inode"
    );
    assert!(!snap.is_expired());
    // Every key of the snapshot's state reads correctly through the old inode.
    block_on(async {
        for i in 0..40u32 {
            let got = snap.get(format!("k{i:02}").into_bytes()).await.unwrap();
            if i == 0 || i == 7 {
                assert_eq!(got, None);
            } else {
                assert_eq!(got, Some(vec![3 ^ i as u8; 800]));
            }
        }
    });
    // Retention accounting: one retained generation, its bytes = the old file.
    let m = map.metrics();
    assert_eq!(m.retained_generations, 1);
    assert!(m.retained_segment_bytes > 0);

    drop(snap);
    assert!(
        old_weak.upgrade().is_none(),
        "inode frees when the last snapshot drops"
    );
    assert_eq!(map.metrics().retained_generations, 0);
    probe_churned(&map, 3);
    block_on(map.close()).unwrap();
}

/// Direct + bound `Some(0)`: the compaction's retire immediately revokes the
/// superseded generation — the snapshot expires, later reads fail terminally,
/// and the old inode frees on the spot.
#[test]
fn direct_bound_zero_expires_snapshot_at_compaction() {
    let (map, host) = hosted_direct_store(Options {
        max_retained_generations: Some(0),
        ..Options::default()
    });
    churn_direct(&map, 3);
    let snap = map.snapshot();
    let old = host.active();
    let old_weak = Arc::downgrade(&old);
    drop(old);

    block_on(map.store().compact()).unwrap();
    assert!(snap.is_expired());
    assert!(matches!(
        block_on(snap.get(b"k01".to_vec())),
        Err(crate::error::ReadError::SnapshotExpired)
    ));
    assert!(
        old_weak.upgrade().is_none(),
        "bound 0 frees the inode at the compaction"
    );
    assert_eq!(map.metrics().expired_generations, 1);
    // The live map is unaffected.
    probe_churned(&map, 3);
    block_on(map.close()).unwrap();
}

/// Compacting an already-dense store is idempotent: same live set, same size,
/// same content — and each pass raises the id delta without disturbing reads.
#[test]
fn direct_compaction_is_idempotent_when_dense() {
    let (map, host) = hosted_direct_store(Options::default());
    churn_direct(&map, 3);
    let first = block_on(map.store().compact()).unwrap();
    let second = block_on(map.store().compact()).unwrap();
    assert_eq!(second.live_pages, first.live_pages);
    assert_eq!(second.data_bytes_after, first.data_bytes_after);
    assert_eq!(
        second.data_bytes_before, first.data_bytes_after,
        "the second pass starts from the first pass's dense file"
    );
    probe_churned(&map, 3);
    assert_eq!(map.metrics().compactions, 2);
    block_on(map.close()).unwrap();
    // Still reopens cleanly after two renumberings.
    let map = BTreeMap::over(
        Store::<Direct>::open_over(dev(&host.active()), Options::default()).unwrap(),
    );
    probe_churned(&map, 3);
    block_on(map.close()).unwrap();
}

/// An empty store compacts to just its header pages and stays writable.
#[test]
fn direct_empty_store_compacts_to_headers_only() {
    let (map, host) = hosted_direct_store(Options::default());
    let report = block_on(map.store().compact()).unwrap();
    assert_eq!(report.live_pages, 0);
    assert_eq!(report.data_bytes_after, 2 * crate::page::PAGE_SIZE as u64);
    put_direct(&map, b"k", b"v");
    block_on(async {
        assert_eq!(map.get(b"k".to_vec()).await.unwrap(), Some(b"v".to_vec()));
    });
    block_on(map.close()).unwrap();
    let map = BTreeMap::over(
        Store::<Direct>::open_over(dev(&host.active()), Options::default()).unwrap(),
    );
    assert_eq!(map.len(), 1);
    block_on(map.close()).unwrap();
}

/// Crash on either side of the Direct promote: the old image (crash before) is
/// the untouched pre-compaction store — the compaction never writes the old
/// inode; the new image (crash after) is the dense store. Both reopen through
/// the unchanged header protocol.
#[test]
fn direct_compaction_crash_images_recover_on_either_side() {
    let (map, host) = hosted_direct_store(Options::default());
    churn_direct(&map, 3);
    // Make the churn durable, then capture the pre-compaction durable image —
    // what a crash before the promote leaves at the path.
    let out =
        block_on(map.apply(WriteBatch::new().insert(b"pin".to_vec(), b"p".to_vec()))).unwrap();
    let ApplyOutcome::Applied { version, .. } = out else {
        panic!()
    };
    block_on(map.flush(version)).unwrap();
    let data_a = host.active().synced_bytes();

    block_on(map.store().compact()).unwrap();
    let data_b = host.active().synced_bytes();
    block_on(map.close()).unwrap();

    for image in [data_a, data_b] {
        let active = Arc::new(FakeIo::from_vec(image));
        let host = MemDataHost::new(Arc::clone(&active));
        let map = BTreeMap::over(
            Store::<Direct>::open_over_hosted(
                dev(&active),
                Some(Arc::clone(&host) as Arc<dyn crate::compact::DataFileHost>),
                Options::default(),
                None,
            )
            .unwrap(),
        );
        probe_churned(&map, 3);
        block_on(async {
            assert_eq!(map.get(b"pin".to_vec()).await.unwrap(), Some(b"p".to_vec()));
        });
        // A compaction retried after the crash recovery succeeds outright.
        block_on(map.store().compact()).unwrap();
        probe_churned(&map, 3);
        block_on(map.close()).unwrap();
    }
}

/// StoreWal: a compaction folds checkpoint-extent garbage and outstanding WAL
/// records into one dense data file, rotates the segment, preserves content,
/// and reopens through unchanged recovery.
#[test]
fn wal_compaction_folds_history_and_preserves_content() {
    let (map, host, fs) = hosted_wal_store(Options::default());
    churn_wal(&map, 3);
    // Two checkpoints strand two relocated extents + manifests in the data file
    // (each checkpoint appends past the previous one's — the orphan garbage
    // compaction exists to fold), then a further commit appends fresh WAL
    // records — every kind of history exists at once.
    block_on(map.store().checkpoint()).unwrap();
    put(&map, b"y-mid", b"yyy");
    block_on(map.store().checkpoint()).unwrap();
    put(&map, b"z-post-ckpt", b"zzz");
    let visible = map.metrics().visible_txid;

    let old_data = host.active();
    let old_data_weak = Arc::downgrade(&old_data);
    drop(old_data);
    let old_seg = fs
        .segment(2)
        .expect("segment 2 in force after two checkpoints");
    let old_seg_weak = Arc::downgrade(&old_seg);
    drop(old_seg);

    let report = block_on(map.store().compact()).unwrap();
    assert!(
        report.data_bytes_after < report.data_bytes_before,
        "{report:?}"
    );
    assert_eq!(
        fs.live_segments(),
        vec![3],
        "the segment rotated at the cutover"
    );
    // No snapshot holds the superseded generation: both inodes free at retire.
    assert!(
        old_data_weak.upgrade().is_none(),
        "superseded data inode must free"
    );
    assert!(
        old_seg_weak.upgrade().is_none(),
        "retired segment must free"
    );

    let m = map.metrics();
    assert_eq!(m.compactions, 1);
    assert_eq!(m.checkpoints, 3, "a compaction is a checkpoint too");
    assert_eq!(m.last_checkpoint_txid, visible);
    assert_eq!(
        m.wal_physical_bytes,
        crate::page::PAGE_SIZE as u64,
        "fresh segment"
    );

    probe_churned(&map, 3);
    block_on(async {
        assert_eq!(
            map.get(b"y-mid".to_vec()).await.unwrap(),
            Some(b"yyy".to_vec())
        );
        assert_eq!(
            map.get(b"z-post-ckpt".to_vec()).await.unwrap(),
            Some(b"zzz".to_vec())
        );
    });
    // Writes, commits, and a further checkpoint all continue over the new file.
    put(&map, b"after", b"v");
    block_on(map.store().checkpoint()).unwrap();
    block_on(map.close()).unwrap();

    // Reopen resolves everything from the promoted file's header.
    let map = BTreeMap::over(open_mem_store_hosted(host, fs, Options::default()).unwrap());
    probe_churned(&map, 3);
    block_on(async {
        assert_eq!(
            map.get(b"after".to_vec()).await.unwrap(),
            Some(b"v".to_vec())
        );
    });
    block_on(map.close()).unwrap();
}

/// StoreWal: a within-bound snapshot stays fully readable across a compaction —
/// pinning *both* superseded inodes (data + segment) — and both free at drop.
#[test]
fn wal_snapshot_stays_readable_across_compaction() {
    let (map, host, fs) = hosted_wal_store(Options::default());
    churn_wal(&map, 3);
    let snap = map.snapshot();

    let old_data = host.active();
    let old_data_weak = Arc::downgrade(&old_data);
    drop(old_data);
    let old_seg = fs.segment(0).unwrap();
    let old_seg_weak = Arc::downgrade(&old_seg);
    drop(old_seg);

    block_on(map.store().compact()).unwrap();
    assert!(
        old_data_weak.upgrade().is_some(),
        "snapshot pins the old data inode"
    );
    assert!(
        old_seg_weak.upgrade().is_some(),
        "snapshot pins the old segment inode"
    );
    block_on(async {
        for i in 1..40u32 {
            if i == 7 {
                continue;
            }
            assert_eq!(
                snap.get(format!("k{i:02}").into_bytes()).await.unwrap(),
                Some(vec![3 ^ i as u8; 800])
            );
        }
    });
    drop(snap);
    assert!(old_data_weak.upgrade().is_none());
    assert!(old_seg_weak.upgrade().is_none());
    probe_churned(&map, 3);
    block_on(map.close()).unwrap();
}

/// StoreWal + bound `Some(0)`: the compaction expires the snapshot and frees
/// both superseded inodes immediately.
#[test]
fn wal_bound_zero_expires_snapshot_at_compaction() {
    let (map, host, fs) = hosted_wal_store(Options {
        max_retained_generations: Some(0),
        ..Options::default()
    });
    churn_wal(&map, 3);
    let snap = map.snapshot();
    let old_data = host.active();
    let old_data_weak = Arc::downgrade(&old_data);
    drop(old_data);
    let old_seg = fs.segment(0).unwrap();
    let old_seg_weak = Arc::downgrade(&old_seg);
    drop(old_seg);

    block_on(map.store().compact()).unwrap();
    assert!(snap.is_expired());
    assert!(matches!(
        block_on(snap.get(b"k01".to_vec())),
        Err(crate::error::ReadError::SnapshotExpired)
    ));
    assert!(old_data_weak.upgrade().is_none());
    assert!(old_seg_weak.upgrade().is_none());
    probe_churned(&map, 3);
    block_on(map.close()).unwrap();
}

/// StoreWal crash matrix around the promote: reconstruct the exact durable
/// state on each side of the cutover from synced images and reopen each through
/// the **unchanged** recovery path. Neither loses an acknowledged commit.
#[test]
fn wal_compaction_crash_images_recover_on_either_side() {
    let (map, host, fs) = hosted_wal_store(Options::default());
    churn_wal(&map, 3);
    block_on(map.store().checkpoint()).unwrap(); // segment 1 in force
    put(&map, b"z-post-ckpt", b"zzz");

    // The durable pre-compaction images.
    let data_a = host.active().synced_bytes();
    let seg1_a = fs.segment(1).unwrap().synced_bytes();

    block_on(map.store().compact()).unwrap(); // → segment 2, dense data file

    // The durable post-compaction images.
    let data_b = host.active().synced_bytes();
    let seg2_b = fs.segment(2).unwrap().synced_bytes();
    block_on(map.close()).unwrap();

    // Crash BEFORE the promote became durable: the path still holds the old
    // data file; segment 1 is still linked; the minted segment 2 is an orphan.
    {
        let fresh = MemWalFileSet::new();
        fresh.insert(1, Arc::new(FakeIo::from_vec(seg1_a.clone())));
        fresh.insert(2, Arc::new(FakeIo::from_vec(seg2_b.clone())));
        let host = MemDataHost::new(Arc::new(FakeIo::from_vec(data_a)));
        let map = BTreeMap::over(
            open_mem_store_hosted(host, Arc::clone(&fresh), Options::default()).unwrap(),
        );
        // The orphan segment the crashed compaction minted was cleaned up.
        assert_eq!(fresh.live_segments(), vec![1]);
        probe_churned(&map, 3);
        block_on(async {
            assert_eq!(
                map.get(b"z-post-ckpt".to_vec()).await.unwrap(),
                Some(b"zzz".to_vec()),
                "the acknowledged post-checkpoint commit survives"
            );
        });
        // A compaction retried after the crash recovery succeeds outright.
        block_on(map.store().compact()).unwrap();
        probe_churned(&map, 3);
        block_on(map.close()).unwrap();
    }

    // Crash AFTER the promote but BEFORE the retired segment's unlink: the path
    // holds the dense file naming segment 2; segment 1 is the orphan.
    {
        let fresh = MemWalFileSet::new();
        fresh.insert(1, Arc::new(FakeIo::from_vec(seg1_a)));
        fresh.insert(2, Arc::new(FakeIo::from_vec(seg2_b)));
        let host = MemDataHost::new(Arc::new(FakeIo::from_vec(data_b)));
        let map = BTreeMap::over(
            open_mem_store_hosted(host, Arc::clone(&fresh), Options::default()).unwrap(),
        );
        assert_eq!(
            fresh.live_segments(),
            vec![2],
            "the retired segment was cleaned up"
        );
        probe_churned(&map, 3);
        block_on(async {
            assert_eq!(
                map.get(b"z-post-ckpt".to_vec()).await.unwrap(),
                Some(b"zzz".to_vec())
            );
        });
        block_on(map.store().compact()).unwrap();
        probe_churned(&map, 3);
        block_on(map.close()).unwrap();
    }
}

/// A never-checkpointed store compacts too: the live set is read from the WAL
/// overlay and lands dense in the data file.
#[test]
fn wal_compaction_of_a_never_checkpointed_store() {
    let (map, host, fs) = hosted_wal_store(Options::default());
    churn_wal(&map, 3);
    let report = block_on(map.store().compact()).unwrap();
    assert!(report.live_pages > 0);
    assert_eq!(fs.live_segments(), vec![1]);
    probe_churned(&map, 3);
    // Idempotent when already dense (WAL half — Direct's is tested above).
    let second = block_on(map.store().compact()).unwrap();
    assert_eq!(second.live_pages, report.live_pages);
    assert_eq!(second.data_bytes_after, report.data_bytes_after);
    assert_eq!(second.data_bytes_before, report.data_bytes_after);
    assert_eq!(fs.live_segments(), vec![2]);
    probe_churned(&map, 3);
    block_on(map.close()).unwrap();
    let map = BTreeMap::over(open_mem_store_hosted(host, fs, Options::default()).unwrap());
    probe_churned(&map, 3);
    block_on(map.close()).unwrap();
}

/// No data namespace → no compaction, exactly like B0's checkpoint rule.
#[test]
fn compaction_unsupported_without_a_namespace() {
    // StoreWal over caller-supplied devices.
    let (map, _data, _wal) = wal_store();
    assert!(matches!(
        block_on(map.store().compact()),
        Err(CheckpointError::Unsupported)
    ));
    block_on(map.close()).unwrap();
    // StoreDirect over a caller-supplied device.
    let data = Arc::new(FakeIo::new(0));
    let map =
        BTreeMap::over(Store::<Direct>::create_with_io(dev(&data), Options::default()).unwrap());
    assert!(matches!(
        block_on(map.store().compact()),
        Err(CheckpointError::Unsupported)
    ));
    block_on(map.close()).unwrap();
}

/// A compact command fails fast during an operator maintenance pause (it is a
/// writer command; the writer is parked), and works after resume.
#[test]
fn compaction_fails_fast_during_maintenance() {
    let (map, _host) = hosted_direct_store(Options::default());
    put_direct(&map, b"k", b"v");
    assert!(map.store().begin_maintenance());
    assert!(matches!(
        block_on(map.store().compact()),
        Err(CheckpointError::Compacting)
    ));
    map.store().end_maintenance();
    assert!(block_on(map.store().compact()).is_ok());
    block_on(map.close()).unwrap();
}

/// A fault-injecting data host: each flag fires once and clears, so the abort
/// path and the retry that follows it run in one test — the clean-abort
/// deviation must be guarded by tests, not only argued.
struct FlakyHost {
    inner: Arc<MemDataHost>,
    fail_create: AtomicBool,
    fail_temp_sync: AtomicBool,
    fail_promote: AtomicBool,
}

impl crate::compact::DataFileHost for FlakyHost {
    fn create_temp(&self) -> std::io::Result<Arc<dyn PageIo>> {
        if self.fail_create.swap(false, Ordering::SeqCst) {
            return Err(std::io::Error::other("injected temp-create failure"));
        }
        let dev = self.inner.create_temp()?;
        if self.fail_temp_sync.swap(false, Ordering::SeqCst) {
            // The compaction's own temp datasync will fail once.
            self.inner.temp().unwrap().set_faults(crate::io::Faults {
                fail_next_sync: Some(libc::EIO),
                ..Default::default()
            });
        }
        Ok(dev)
    }

    fn promote_temp(&self) -> std::io::Result<()> {
        if self.fail_promote.swap(false, Ordering::SeqCst) {
            return Err(std::io::Error::other("injected promotion failure"));
        }
        self.inner.promote_temp()
    }

    fn remove_temp(&self) {
        self.inner.remove_temp()
    }
}

fn flaky_direct_store() -> (BTreeMap<Direct>, Arc<FlakyHost>) {
    let data = Arc::new(FakeIo::new(0));
    let host = Arc::new(FlakyHost {
        inner: MemDataHost::new(Arc::clone(&data)),
        fail_create: AtomicBool::new(false),
        fail_temp_sync: AtomicBool::new(false),
        fail_promote: AtomicBool::new(false),
    });
    let store = Store::<Direct>::create_with_io_hosted(
        dev(&data),
        Some(Arc::clone(&host) as Arc<dyn crate::compact::DataFileHost>),
        Options::default(),
        None,
    )
    .unwrap();
    (BTreeMap::over(store), host)
}

/// Every pre-promotion failure is a clean abort: `CheckpointError::Io`, the
/// temp discarded, the store unpoisoned and fully usable — and a retry with
/// the fault cleared succeeds outright.
#[test]
fn a_pre_promotion_failure_aborts_cleanly_and_the_store_runs_on() {
    let (map, host) = flaky_direct_store();
    churn_direct(&map, 3);

    // Temp creation fails.
    host.fail_create.store(true, Ordering::SeqCst);
    let out = block_on(map.store().compact());
    assert!(
        matches!(out, Err(CheckpointError::Io(_))),
        "create: {out:?}"
    );
    probe_churned(&map, 3);
    put_direct(&map, b"alive-1", b"a");

    // The temp's own datasync fails (after the full rebuild was written).
    host.fail_temp_sync.store(true, Ordering::SeqCst);
    let out = block_on(map.store().compact());
    assert!(matches!(out, Err(CheckpointError::Io(_))), "sync: {out:?}");
    assert!(
        host.inner.temp().is_none(),
        "the aborted temp was discarded"
    );
    put_direct(&map, b"alive-2", b"b");

    // Nothing counted, nothing poisoned; the retry succeeds.
    assert_eq!(map.metrics().compactions, 0);
    let report = block_on(map.store().compact()).unwrap();
    assert!(report.live_pages > 0);
    assert_eq!(map.metrics().compactions, 1);
    probe_churned(&map, 3);
    block_on(async {
        assert_eq!(
            map.get(b"alive-1".to_vec()).await.unwrap(),
            Some(b"a".to_vec())
        );
        assert_eq!(
            map.get(b"alive-2".to_vec()).await.unwrap(),
            Some(b"b".to_vec())
        );
    });
    block_on(map.close()).unwrap();
}

/// Phase-H debt: the pre-temp `size()` probe's abort
/// arm gets direct fault injection. A failed probe is a clean
/// `CheckpointError::Io` — read-only, pre-mutation, nothing poisoned — and the
/// retry with the fault cleared compacts outright. Direct backend.
#[test]
fn a_size_probe_failure_aborts_direct_compaction_cleanly() {
    let data = Arc::new(FakeIo::new(0));
    let host = MemDataHost::new(Arc::clone(&data));
    let store = Store::<Direct>::create_with_io_hosted(
        dev(&data),
        Some(Arc::clone(&host) as Arc<dyn crate::compact::DataFileHost>),
        Options::default(),
        None,
    )
    .unwrap();
    let map = BTreeMap::over(store);
    churn_direct(&map, 3);

    data.set_faults(crate::io::Faults {
        fail_next_size: Some(5), // EIO
        ..Default::default()
    });
    let out = block_on(map.store().compact());
    assert!(matches!(out, Err(CheckpointError::Io(_))), "probe: {out:?}");
    assert!(host.temp().is_none(), "the probe aborts before any temp");

    // Unpoisoned and fully usable; the fault self-cleared, so a retry runs.
    probe_churned(&map, 3);
    put_direct(&map, b"alive", b"a");
    assert_eq!(map.metrics().compactions, 0);
    let report = block_on(map.store().compact()).unwrap();
    assert!(report.live_pages > 0);
    assert_eq!(map.metrics().compactions, 1);
    block_on(map.close()).unwrap();
}

/// The Wal twin of the size-probe abort: same clean `CheckpointError::Io`,
/// same no-poison guarantee, retry succeeds (wal.rs carries its own copy of
/// the probe, so both arms need teeth).
#[test]
fn a_size_probe_failure_aborts_wal_compaction_cleanly() {
    let data = Arc::new(FakeIo::new(0));
    let host = MemDataHost::new(Arc::clone(&data));
    let file_set = MemWalFileSet::new();
    let store =
        create_mem_store_hosted(Arc::clone(&host), Arc::clone(&file_set), Options::default())
            .unwrap();
    let map = BTreeMap::over(store);
    churn_wal(&map, 3);

    data.set_faults(crate::io::Faults {
        fail_next_size: Some(5), // EIO
        ..Default::default()
    });
    let out = block_on(map.store().compact());
    assert!(matches!(out, Err(CheckpointError::Io(_))), "probe: {out:?}");
    assert!(host.temp().is_none(), "the probe aborts before any temp");

    probe_churned(&map, 3);
    put(&map, b"alive", b"a");
    assert_eq!(map.metrics().compactions, 0);
    let report = block_on(map.store().compact()).unwrap();
    assert!(report.live_pages > 0);
    assert_eq!(map.metrics().compactions, 1);
    block_on(map.close()).unwrap();
}

/// A promotion failure is the one compaction failure that poisons: the commit
/// is in doubt (invariant 17), so the store goes terminal with the original
/// reason.
#[test]
fn a_promotion_failure_poisons() {
    let (map, host) = flaky_direct_store();
    churn_direct(&map, 3);
    host.fail_promote.store(true, Ordering::SeqCst);
    let out = block_on(map.store().compact());
    assert!(matches!(out, Err(CheckpointError::Poisoned(_))), "{out:?}");
    let w = block_on(map.apply(WriteBatch::new().insert(b"x".to_vec(), b"y".to_vec())));
    assert!(
        matches!(w, Err(crate::error::WriteError::Poisoned(_))),
        "the poison is sticky: {w:?}"
    );
    let _ = block_on(map.close());
}

/// A checkpoint-retired generation pins the data inode a
/// later compaction supersedes. The superseded inode's bytes must stay in
/// `retained_segment_bytes` until the *last* generation reading through it
/// releases — even though the compaction-retired generation itself released
/// born-dead (nothing held it) and took its exclusive segment bytes with it.
#[test]
fn wal_retained_bytes_cover_the_shared_data_inode() {
    let (map, host, _fs) = hosted_wal_store(Options::default());
    churn_wal(&map, 3);
    let snap = map.snapshot(); // holds generation 0: data inode D0 + segment 0
    let seg0_bytes = map.metrics().wal_physical_bytes;
    block_on(map.store().checkpoint()).unwrap(); // retires gen 0 (exclusive: segment 0)
    let d0_bytes = host.active().snapshot_bytes().len() as u64;
    block_on(map.store().compact()).unwrap(); // supersedes D0; its retiree has no holders
    let m = map.metrics();
    assert_eq!(
        m.retained_generations, 1,
        "only the snapshot's generation is retained"
    );
    assert_eq!(
        m.retained_segment_bytes,
        seg0_bytes + d0_bytes,
        "the superseded inode is counted once, via the surviving reader"
    );
    // The snapshot still reads everything through D0 + segment 0.
    block_on(async {
        assert_eq!(
            snap.get(b"k05".to_vec()).await.unwrap(),
            Some(vec![3 ^ 5u8; 800])
        );
    });
    drop(snap);
    assert_eq!(
        map.metrics().retained_segment_bytes,
        0,
        "all pins gone at the last release"
    );
    probe_churned(&map, 3);
    block_on(map.close()).unwrap();
}

/// The WAL compactor's abort arms, fault-injected (the Direct-only `FlakyHost`
/// coverage left wal.rs's separately-implemented abort
/// paths unguarded): a temp-sync failure aborts cleanly — the store commits,
/// checkpoints, and retries on — and a promotion failure poisons.
#[test]
fn a_wal_pre_promotion_failure_aborts_cleanly_and_promotion_failure_poisons() {
    let data = Arc::new(FakeIo::new(0));
    let host = Arc::new(FlakyHost {
        inner: MemDataHost::new(Arc::clone(&data)),
        fail_create: AtomicBool::new(false),
        fail_temp_sync: AtomicBool::new(false),
        fail_promote: AtomicBool::new(false),
    });
    let store_uuid = crate::direct::generate_store_uuid();
    let fs = MemWalFileSet::new();
    let wal = fs.create_segment(0, store_uuid).unwrap();
    let store = Store::<Wal>::create_with_io_owned(
        dev(&data),
        wal,
        store_uuid,
        Some(Arc::clone(&fs) as Arc<dyn WalFileSet>),
        Some(Arc::clone(&host) as Arc<dyn crate::compact::DataFileHost>),
        Options::default(),
        None,
    )
    .unwrap();
    let map = BTreeMap::over(store);
    churn_wal(&map, 3);

    // The temp's datasync fails after the whole live set was relocated: clean
    // abort, temp discarded, segment 0 still in force, store fully usable.
    host.fail_temp_sync.store(true, Ordering::SeqCst);
    let out = block_on(map.store().compact());
    assert!(matches!(out, Err(CheckpointError::Io(_))), "sync: {out:?}");
    assert!(host.inner.temp().is_none(), "aborted temp discarded");
    assert_eq!(
        fs.live_segments(),
        vec![0],
        "no rotation on an aborted compaction"
    );
    assert_eq!(map.metrics().compactions, 0);
    put(&map, b"alive", b"a");

    // The retry succeeds outright.
    let report = block_on(map.store().compact()).unwrap();
    assert!(report.live_pages > 0);
    assert_eq!(fs.live_segments(), vec![1]);
    probe_churned(&map, 3);

    // A promotion failure is the commit-in-doubt case: poison, sticky.
    put(&map, b"more", b"b");
    host.fail_promote.store(true, Ordering::SeqCst);
    let out = block_on(map.store().compact());
    assert!(matches!(out, Err(CheckpointError::Poisoned(_))), "{out:?}");
    let w = block_on(map.apply(WriteBatch::new().insert(b"x".to_vec(), b"y".to_vec())));
    assert!(matches!(w, Err(crate::error::WriteError::Poisoned(_))));
    let _ = block_on(map.close());
}

// ---------------------------------------------------------------------------
// M6 Phase F — overflow values + disk-space limits.
// ---------------------------------------------------------------------------

use crate::page::{MAX_INLINE_VALUE_LEN, MAX_VALUE_LEN, PAGE_SIZE as PGSZ};

/// A deterministic pseudo-random blob of `n` bytes. Content varies with `seed`
/// and position, so a chain that reassembles in the wrong order, drops a link, or
/// duplicates one cannot pass by luck (a constant fill would).
fn blob(seed: u8, n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            (i as u8)
                .wrapping_mul(31)
                .wrapping_add(seed)
                .wrapping_add((i >> 8) as u8)
        })
        .collect()
}

/// Values that bracket every interesting chunking boundary of the codec.
fn spill_sizes() -> Vec<usize> {
    let cap = crate::page::overflow_payload_capacity();
    vec![
        MAX_INLINE_VALUE_LEN + 1, // the smallest value that spills at all
        cap - 1,                  // one short of a single full link
        cap,                      // exactly one full link (tail == cap)
        cap + 1,                  // two links, tail of 1 byte
        2 * cap,                  // exactly two full links
        MAX_VALUE_LEN,            // the largest legal value (65 links)
    ]
}

#[test]
fn overflow_values_round_trip_across_sizes_and_backends() {
    // The inline/overflow boundary must be invisible through the public API: the
    // same key/value contract holds on both sides of the spill threshold.
    let (wal, _d, _fs) = owned_store();
    let (direct, _host) = hosted_direct_store(Options::default());
    for (i, n) in spill_sizes().into_iter().enumerate() {
        let k = format!("big{i:02}");
        let v = blob(i as u8, n);
        put(&wal, k.as_bytes(), &v);
        put_direct(&direct, k.as_bytes(), &v);
        block_on(async {
            assert_eq!(
                wal.get(k.clone().into_bytes()).await.unwrap(),
                Some(v.clone()),
                "wal {n}"
            );
            assert_eq!(
                direct.get(k.into_bytes()).await.unwrap(),
                Some(v),
                "direct {n}"
            );
        });
    }
    // An inline value still stores inline — the threshold moved nothing below it.
    put(&wal, b"small", &blob(9, MAX_INLINE_VALUE_LEN));
    block_on(async {
        assert_eq!(
            wal.get(b"small".to_vec()).await.unwrap(),
            Some(blob(9, MAX_INLINE_VALUE_LEN))
        );
        wal.verify().await.unwrap();
        direct.verify().await.unwrap();
    });
}

#[test]
fn oversize_value_is_still_rejected_at_the_batch_cap() {
    // Raising MAX_VALUE_LEN moves the cap; it does not remove it.
    let (map, _d, _fs) = owned_store();
    let too_big = WriteBatch::new().insert(b"k".to_vec(), vec![0u8; MAX_VALUE_LEN + 1]);
    assert!(matches!(
        block_on(map.apply(too_big)),
        Err(crate::error::WriteError::InvalidBatch(
            crate::batch::BatchError::ValueTooLong
        ))
    ));
    // Exactly at the cap is legal.
    put(&map, b"k", &blob(1, MAX_VALUE_LEN));
    block_on(async {
        assert_eq!(
            map.get(b"k".to_vec()).await.unwrap(),
            Some(blob(1, MAX_VALUE_LEN))
        );
    });
}

#[test]
fn overflow_survives_reopen_and_recovery() {
    // The chain rides the WAL as page images like any other page, so a value
    // written but never checkpointed must come back from the log alone.
    let (map, data, file_set) = owned_store();
    let v = blob(3, 300_000);
    put(&map, b"huge", &v);
    put(&map, b"tiny", b"t");
    let (re, _d2, _fs2) = reopen_synced(&data, &file_set);
    block_on(async {
        assert_eq!(re.get(b"huge".to_vec()).await.unwrap(), Some(v));
        assert_eq!(re.get(b"tiny".to_vec()).await.unwrap(), Some(b"t".to_vec()));
        re.verify().await.unwrap();
    });
}

#[test]
fn checkpoint_relocates_overflow_chains() {
    // The headline Phase F risk: if the live-set walker
    // does not traverse overflow refs, the checkpoint drops every chain page and
    // the value is silently gone after the WAL is reclaimed. Reopening from the
    // *data file alone* — the rotated segment no longer holds the record — is what
    // proves the walker followed them.
    let (map, data, file_set) = owned_store();
    let v = blob(5, 400_000);
    put(&map, b"huge", &v);
    for i in 0..20u32 {
        put(&map, format!("k{i:02}").as_bytes(), b"small");
    }
    let report = block_on(map.store().checkpoint()).unwrap();
    assert!(report.skipped_space.is_none());
    // ceil(400_000/16_336) == 25 chain pages, plus the tree's own pages.
    assert!(
        report.relocated_pages > 25,
        "the live set must include the chain: {report:?}"
    );
    let (re, _d2, _fs2) = reopen_synced(&data, &file_set);
    block_on(async {
        assert_eq!(
            re.get(b"huge".to_vec()).await.unwrap(),
            Some(v),
            "chain lost by checkpoint"
        );
        re.verify().await.unwrap();
    });
}

#[test]
fn compaction_renumbers_overflow_chains_on_both_backends() {
    // Direct compaction renumbers onto fresh ids, so a chain must be rebuilt
    // tail-first and its leaf ref rewritten. WAL compaction preserves ids but must
    // still reach the chain through the walker.
    let (direct, host) = hosted_direct_store(Options::default());
    let v = blob(7, 200_000);
    put_direct(&direct, b"huge", &v);
    put_direct(&direct, b"huge", &blob(8, 200_000)); // supersede: old chain is garbage
    let v2 = blob(8, 200_000);
    let report = block_on(direct.store().compact()).unwrap();
    assert!(
        report.data_bytes_after < report.data_bytes_before,
        "{report:?}"
    );
    block_on(async {
        assert_eq!(
            direct.get(b"huge".to_vec()).await.unwrap(),
            Some(v2.clone())
        );
        direct.verify().await.unwrap();
    });
    // The promoted file is dense: exactly headers + live pages, so the superseded
    // chain's 13 pages were dropped, not copied.
    assert_eq!(
        host.active().snapshot_bytes().len() as u64,
        (2 + report.live_pages) * PGSZ as u64
    );

    let (wal, _h2, _fs) = hosted_wal_store(Options::default());
    put(&wal, b"huge", &v);
    let r = block_on(wal.store().compact()).unwrap();
    assert!(r.live_pages > 12, "{r:?}");
    block_on(async {
        assert_eq!(wal.get(b"huge".to_vec()).await.unwrap(), Some(v));
        wal.verify().await.unwrap();
    });
}

#[test]
fn reinserting_an_identical_large_value_is_a_no_op() {
    // Without an equality check against the stored
    // chain, re-inserting identical bytes rebuilds the whole chain and the root
    // path. The observable is **WAL bytes** — the record carries a full image per
    // rebuilt page, and WAL bytes are exactly what `max_wal_bytes` refuses writes
    // on, so the waste is what would turn an idempotent retry into `StoreFull`.
    // (A no-op batch still consumes a txid; `Unchanged` elides pages, not versions.)
    let (map, _d, _fs) = owned_store();
    let v = blob(11, 200_000);
    put(&map, b"huge", &v);

    let before = map.store().metrics().wal_logical_bytes;
    put(&map, b"huge", &v); // byte-identical
    let idempotent_cost = map.store().metrics().wal_logical_bytes - before;

    // A genuinely different value of the *same* length must still be written, and
    // is the yardstick: the identical re-insert must cost a small fraction of it.
    let before = map.store().metrics().wal_logical_bytes;
    let v2 = blob(12, 200_000);
    put(&map, b"huge", &v2);
    let real_write_cost = map.store().metrics().wal_logical_bytes - before;

    assert!(
        real_write_cost > 200_000,
        "a real 200 KB overwrite must carry its page images: {real_write_cost}"
    );
    assert!(
        idempotent_cost * 10 < real_write_cost,
        "an identical overflow re-insert must not rewrite the chain: \
         cost {idempotent_cost} vs a real write's {real_write_cost}"
    );
    block_on(async {
        assert_eq!(map.get(b"huge".to_vec()).await.unwrap(), Some(v2));
    });
}

#[test]
fn a_maximum_value_chain_is_longer_than_the_tree_depth_bound() {
    // A 1 MiB value is 65 links, and
    // MAX_TREE_DEPTH is 64. If chain length were charged to tree depth, every one
    // of these paths would report Corrupt and poison a healthy store.
    let v = blob(13, MAX_VALUE_LEN);
    assert!(
        (MAX_VALUE_LEN as u64).div_ceil(crate::page::overflow_payload_capacity() as u64)
            > crate::read::MAX_TREE_DEPTH as u64,
        "this test is only meaningful while a max chain outruns the depth bound"
    );
    let (map, data, file_set) = owned_store();
    put(&map, b"max", &v);
    block_on(async {
        assert_eq!(map.get(b"max".to_vec()).await.unwrap(), Some(v.clone()));
        map.verify().await.unwrap(); // walks the chain
    });
    block_on(map.store().checkpoint()).unwrap(); // walks the chain
    let (re, _d, _f) = reopen_synced(&data, &file_set);
    block_on(async {
        assert_eq!(re.get(b"max".to_vec()).await.unwrap(), Some(v));
    });

    let (direct, _host) = hosted_direct_store(Options::default());
    put_direct(&direct, b"max", &blob(13, MAX_VALUE_LEN));
    block_on(direct.store().compact()).unwrap(); // renumbers the chain
    block_on(async {
        assert_eq!(
            direct.get(b"max".to_vec()).await.unwrap(),
            Some(blob(13, MAX_VALUE_LEN))
        );
    });
}

/// An owned StoreWal with caller options (the `max_data_bytes` tests).
fn owned_store_with(options: Options) -> (BTreeMap<Wal>, Arc<FakeIo>, Arc<MemWalFileSet>) {
    let data = Arc::new(FakeIo::new(0));
    let file_set = MemWalFileSet::new();
    let store = create_mem_store(dev(&data), Arc::clone(&file_set), options).unwrap();
    (BTreeMap::over(store), data, file_set)
}

#[test]
fn checkpoint_skips_when_it_would_breach_max_data_bytes() {
    // The whole point of the limit: the store must stay *healthy* when it binds.
    // A skip is a backpressure signal, not a failure.
    let (map, _d, file_set) = owned_store_with(Options {
        max_data_bytes: Some(64 * 1024), // smaller than any real live set + headroom
        ..Options::default()
    });
    for i in 0..40u32 {
        put(&map, format!("k{i:02}").as_bytes(), &[i as u8; 400]);
    }
    let segments_before = file_set.live_segments();

    let report = block_on(map.store().checkpoint()).unwrap();
    let skip = report
        .skipped_space
        .expect("checkpoint must skip on the space limit");
    assert_eq!(skip.limit, 64 * 1024);
    assert!(skip.projected > skip.limit, "{skip:?}");
    // A skip is a *no-op*, not a partial checkpoint.
    assert_eq!(report.relocated_pages, 0);
    assert_eq!(report.retired_wal_bytes, 0);
    assert_eq!(
        file_set.live_segments(),
        segments_before,
        "a skipped checkpoint must not rotate the segment"
    );
    let m = map.store().metrics();
    assert_eq!(m.checkpoints_skipped_space, 1);
    assert_eq!(m.checkpoints, 0, "a skip is not a checkpoint");

    // Still fully healthy: reads work, writes work, and it can checkpoint again.
    block_on(async {
        assert_eq!(
            map.get(b"k00".to_vec()).await.unwrap(),
            Some(vec![0u8; 400])
        );
        map.verify().await.unwrap();
    });
    put(&map, b"after", b"still writable");
    block_on(async {
        assert_eq!(
            map.get(b"after".to_vec()).await.unwrap(),
            Some(b"still writable".to_vec())
        );
    });
    assert!(block_on(map.store().checkpoint())
        .unwrap()
        .skipped_space
        .is_some());
    assert_eq!(map.store().metrics().checkpoints_skipped_space, 2);
    let _ = block_on(map.close());
}

#[test]
fn a_generous_limit_lets_the_checkpoint_run_and_is_reported() {
    // The other side of the branch: the guard must not fire when the store fits,
    // and `store_physical_bytes` must be the number the guard actually decided on.
    let (map, _d, _fs) = owned_store_with(Options {
        max_data_bytes: Some(256 * 1024 * 1024),
        ..Options::default()
    });
    for i in 0..40u32 {
        put(&map, format!("k{i:02}").as_bytes(), &[i as u8; 400]);
    }
    let report = block_on(map.store().checkpoint()).unwrap();
    assert!(report.skipped_space.is_none(), "{report:?}");
    assert!(report.relocated_pages > 0);

    let m = map.store().metrics();
    assert_eq!(m.checkpoints_skipped_space, 0);
    assert_eq!(m.checkpoints, 1);
    assert!(
        m.data_physical_bytes > 0,
        "the checkpoint grew the data file"
    );
    assert_eq!(
        m.store_physical_bytes,
        m.data_physical_bytes + m.wal_physical_bytes + m.retained_segment_bytes,
        "store_physical_bytes must be the sum the space check projects from"
    );
    let _ = block_on(map.close());
}

#[test]
fn the_default_is_unbounded_and_pays_nothing() {
    // `None` must not merely permit everything — it must skip the counting walk
    // entirely. The metric is the proxy: no skip is ever recorded.
    let (map, _d, _fs) = owned_store(); // Options::default() => max_data_bytes: None
    for i in 0..30u32 {
        put(&map, format!("k{i:02}").as_bytes(), &[i as u8; 400]);
    }
    assert!(block_on(map.store().checkpoint())
        .unwrap()
        .skipped_space
        .is_none());
    assert_eq!(map.store().metrics().checkpoints_skipped_space, 0);
    let _ = block_on(map.close());
}

#[test]
fn a_write_failure_despite_the_reservation_still_poisons() {
    // The reservation is against the store's own accounting, not the filesystem's:
    // another process can still take the space. A limit that passes therefore
    // changes nothing about invariant 17 — a failed write to the *shared* data
    // file leaves the store unable to vouch for what a recovery would read.
    // (`fail_beyond_offset` raises EIO; the policy keys on which resource failed,
    // not on the errno, so this models the ENOSPC case exactly.)
    let (map, data, _fs) = owned_store_with(Options {
        max_data_bytes: Some(1024 * 1024 * 1024), // generous: the guard passes
        ..Options::default()
    });
    for i in 0..40u32 {
        put(&map, format!("k{i:02}").as_bytes(), &[i as u8; 400]);
    }
    // The device refuses everything past the current data file: the checkpoint's
    // relocation append is the first write to land there.
    data.set_faults(crate::io::Faults {
        fail_beyond_offset: Some(2 * PGSZ as u64),
        ..Default::default()
    });
    let err = block_on(map.store().checkpoint()).unwrap_err();
    assert!(
        matches!(err, CheckpointError::Poisoned(_)),
        "a failed relocation write into the shared data file must poison: {err:?}"
    );
    let w = block_on(map.apply(WriteBatch::new().insert(b"x".to_vec(), b"y".to_vec())));
    assert!(matches!(w, Err(crate::error::WriteError::Poisoned(_))));
    let _ = block_on(map.close());
}

#[test]
fn the_data_space_metric_tracks_direct_writes_and_compaction() {
    // Direct grows its data file at *flush*, not at a checkpoint (it has none), so
    // that is where the metric has to move. Seeded at open, it would otherwise sit
    // at the empty-store value forever and under-report the whole store.
    let (map, _host) = hosted_direct_store(Options::default());
    let empty = map.store().metrics().data_physical_bytes;
    assert_eq!(
        empty,
        2 * PGSZ as u64,
        "a fresh store is exactly its two header slots"
    );

    churn_direct(&map, 3);
    // A compaction runs at a flushed barrier, so it is also the barrier this
    // metric reads after: force the writer to flush by compacting below.
    let grown = map.store().metrics().data_physical_bytes;
    assert!(
        grown > empty,
        "writes must grow the data-file metric: {grown} vs {empty}"
    );
    // Direct owns no WAL and retains nothing here, so the total is the data file.
    let m = map.store().metrics();
    assert_eq!(m.store_physical_bytes, m.data_physical_bytes);

    // A compaction shrinks it, and the metric must follow the cutover down.
    let report = block_on(map.store().compact()).unwrap();
    assert_eq!(
        map.store().metrics().data_physical_bytes,
        report.data_bytes_after
    );
    assert!(map.store().metrics().data_physical_bytes < grown);
    let _ = block_on(map.close());
}

#[test]
fn many_overflow_entries_share_one_leaf_and_all_renumber() {
    // An overflow entry costs ~40 bytes in its leaf (slot + key + 20-byte ref), so
    // a leaf of huge values is *tiny* — dozens of chains hang off one leaf. That
    // makes the interesting cases concurrent: one leaf pushing many chain walks,
    // and a compaction renumbering all of them plus interleaved inline values.
    let (map, host) = hosted_direct_store(Options::default());
    let n = 12usize;
    let big: Vec<Vec<u8>> = (0..n).map(|i| blob(i as u8, 40_000 + i * 3_000)).collect();
    for (i, v) in big.iter().enumerate() {
        put_direct(&map, format!("b{i:02}").as_bytes(), v);
        // An inline neighbour between each pair: the leaf holds both kinds.
        put_direct(&map, format!("s{i:02}").as_bytes(), &[i as u8; 60]);
    }
    block_on(async { map.verify().await.unwrap() });

    let report = block_on(map.store().compact()).unwrap();
    // Dense: every chain page was renumbered and written, none dropped or doubled.
    assert_eq!(
        host.active().snapshot_bytes().len() as u64,
        (2 + report.live_pages) * PGSZ as u64,
        "the rebuilt file must be exactly headers + live pages: {report:?}"
    );
    block_on(async {
        for (i, v) in big.iter().enumerate() {
            assert_eq!(
                map.get(format!("b{i:02}").into_bytes())
                    .await
                    .unwrap()
                    .as_ref(),
                Some(v),
                "chain {i} lost or mis-linked by the renumbering"
            );
            assert_eq!(
                map.get(format!("s{i:02}").into_bytes()).await.unwrap(),
                Some(vec![i as u8; 60])
            );
        }
        map.verify().await.unwrap();
    });
    let _ = block_on(map.close());
}

#[test]
fn a_spilled_value_removed_in_the_same_batch_writes_nothing() {
    // `collect_reachable` returns only pages reachable from the *final* root, so a
    // chain built and then orphaned inside one batch must never be pinned Dirty or
    // written. Otherwise a batch could charge (and flush) pages nothing points at.
    let (map, _d, _fs) = owned_store();
    put(&map, b"keep", b"v");
    let before = map.store().metrics().wal_logical_bytes;
    block_on(async {
        // Spill, then overwrite the same key with a tiny value in one batch: the
        // chain is superseded before the batch ever publishes.
        map.apply(
            WriteBatch::new()
                .insert(b"k".to_vec(), blob(4, 300_000))
                .insert(b"k".to_vec(), b"small".to_vec()),
        )
        .await
        .unwrap();
    });
    let cost = map.store().metrics().wal_logical_bytes - before;
    assert!(
        cost < 100_000,
        "the superseded chain must not ride the record: {cost} bytes for a 5-byte value"
    );
    block_on(async {
        assert_eq!(
            map.get(b"k".to_vec()).await.unwrap(),
            Some(b"small".to_vec())
        );
        map.verify().await.unwrap();
    });
    let _ = block_on(map.close());
}

/// A forged overflow ref reads as `Corrupt` on every read path, rather than
/// resolving to whatever page happens to sit at that id.
///
/// This is the *cache* half of the story; the writer's private overlay is the half
/// that was actually broken, and is unit-tested at
/// `btree::tests::an_overlay_hit_is_validated_against_its_reference`.
#[test]
fn a_forged_overflow_ref_reads_as_corrupt() {
    use crate::backend::{DataFileLocator, FileGeneration};
    use crate::cache::PageCache;
    use crate::page::{
        encode_leaf, encode_overflow, EntryValue, OverflowRef, PageId, PageRef, PAGE_SIZE,
    };

    let val = blob(2, 20_000); // > 8 KiB: spills, and spans two links
                               // A leaf whose entry names page id 500 — which does not exist — with a
                               // plausible checksum. This is the forged image.
    let forged = OverflowRef {
        head: PageRef {
            page_id: PageId(500),
            expected_checksum: crate::page::Checksum(0xDEAD_BEEF),
        },
        total_len: val.len() as u64,
    };
    let leaf = encode_leaf(
        PageId(3),
        0,
        &[(b"z".to_vec(), EntryValue::Overflow(forged))],
    );

    let dev = Arc::new(FakeIo::new(0));
    let loc = DataFileLocator::new();
    block_on(async {
        loc.record(leaf.id().as_u64(), PAGE_SIZE as u64);
        dev.write_all_at(PAGE_SIZE as u64, leaf.bytes().to_vec().into_boxed_slice())
            .await
            .result
            .unwrap();
        // A page that *does* exist at the forged id, so the failure can only come
        // from the checksum check and not from a missing locator entry.
        let decoy = encode_overflow(PageId(500), 0, b"not the value", None);
        loc.record(500, 2 * PAGE_SIZE as u64);
        dev.write_all_at(
            2 * PAGE_SIZE as u64,
            decoy.bytes().to_vec().into_boxed_slice(),
        )
        .await
        .result
        .unwrap();
        dev.datasync().await.unwrap();

        let file = FileGeneration::direct(dev.clone() as Arc<dyn PageIo>, Arc::new(loc));
        let cache = PageCache::new();
        // The forged ref must never resolve to the decoy: the checksum it names is
        // not the decoy's, so every path reports Corrupt rather than the wrong bytes.
        let err = crate::read::tree_get(&cache, &file, leaf.page_ref(), b"z")
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::error::ReadError::Corrupt(_)),
            "{err:?}"
        );
        assert!(matches!(
            crate::read::tree_verify(&cache, &file, leaf.page_ref()).await,
            Err(crate::error::ReadError::Corrupt(_))
        ));
    });
}

/// Follows the overflow chain in a dense Direct image and returns the page
/// indices it visits, head to tail.
fn chain_page_order(bytes: &[u8]) -> Vec<u64> {
    use crate::page::{PageKind, PAGE_SIZE as PS};
    let img = |i: u64| &bytes[i as usize * PS..(i as usize + 1) * PS];
    let id_of = |i: u64| u64::from_le_bytes(img(i)[8..16].try_into().unwrap());
    let next_of = |i: u64| u64::from_le_bytes(img(i)[32..40].try_into().unwrap());
    let links: Vec<u64> = (2..(bytes.len() / PS) as u64)
        .filter(|&i| img(i)[5] == PageKind::Overflow as u8)
        .collect();
    assert!(!links.is_empty(), "no chain in this image");
    let nexts: Vec<u64> = links.iter().map(|&i| next_of(i)).collect();
    // The head is the only link nothing else points at.
    let head = links
        .iter()
        .copied()
        .find(|&i| !nexts.contains(&id_of(i)))
        .expect("a chain has exactly one head");
    let mut visited = vec![head];
    let mut cur = head;
    while next_of(cur) != 0 {
        let n = next_of(cur);
        cur = links
            .iter()
            .copied()
            .find(|&i| id_of(i) == n)
            .expect("every link is in the file");
        visited.push(cur);
    }
    assert_eq!(visited.len(), links.len(), "the walk must reach every link");
    visited
}

#[test]
fn direct_lays_an_overflow_chain_out_in_value_order() {
    // On StoreDirect a page's offset is a pure function of its id, so id order *is*
    // disk order. A chain must be *encoded* tail-first (each link embeds its
    // successor's id and checksum) — but if ids are drawn in that same order the
    // value lands backwards on disk, and reading it walks descending pages,
    // defeating readahead for exactly the objects this phase adds.
    // Ids are therefore reserved head-first and filled in reverse. Both emitters
    // must do it: the writer (`btree::emit_overflow`) and compaction's renumbering
    // rebuild (`direct::rebuild_chain`).
    let (map, host) = hosted_direct_store(Options::default());
    let v = blob(21, 300_000); // 19 links
    put_direct(&map, b"big", &v);
    block_on(map.store().compact()).unwrap(); // force a flush + a dense rebuild

    let after = chain_page_order(&host.active().snapshot_bytes());
    let mut ascending = after.clone();
    ascending.sort_unstable();
    assert_eq!(
        after, ascending,
        "a rebuilt chain must run head->tail *up* the file, not down it: {after:?}"
    );
    block_on(async { assert_eq!(map.get(b"big".to_vec()).await.unwrap(), Some(v.clone())) });
    let _ = block_on(map.close());

    // The writer's own emitter, with **no compaction**: the case above inspects the
    // file *after* `rebuild_chain` rewrote it, so it would pass even with
    // `btree::emit_overflow` still descending. Flush and read the live file.
    let (map2, host2) = hosted_direct_store(Options::default());
    let ver = block_on(async {
        map2.apply(WriteBatch::new().insert(b"big".to_vec(), v.clone()))
            .await
            .unwrap()
    });
    let ver = match ver {
        ApplyOutcome::Applied { version, .. } => version,
        other => panic!("expected Applied, got {other:?}"),
    };
    block_on(map2.flush(ver)).unwrap();
    let fresh = chain_page_order(&host2.active().synced_bytes());
    let mut asc2 = fresh.clone();
    asc2.sort_unstable();
    assert_eq!(
        fresh, asc2,
        "a freshly written chain must run up the file: {fresh:?}"
    );
    let _ = block_on(map2.close());
}

#[test]
fn old_values_cannot_amplify_a_tiny_batch_into_gigabytes() {
    // Admission charges a command's *input*
    // encoding; the old-value response is an output and nothing charged it. That
    // only stopped mattering because values were small — Phase F raised
    // MAX_VALUE_LEN 128x and the two multiply. A ~2 KiB batch of N remove(k) ops
    // with returning_old_values, against a 1 MiB value at k, would have the writer
    // materialise N MiB before it can reply, and dropping the caller's future does
    // not stop it. Here max_command_bytes is small so the test is quick; the
    // default (64 MiB) against 1,000 ops of a 1 MiB value is the real ~1 GiB case.
    let (map, _d, _fs) = owned_store_with(Options {
        limits: crate::admission::AdmissionLimits {
            max_command_bytes: 512 * 1024,
            ..Default::default()
        },
        ..Options::default()
    });
    let v = blob(31, 200_000);
    put(&map, b"k", &v);

    // Two old values (400 KB) fit under the 512 KB ceiling.
    let ok = block_on(
        map.apply(
            WriteBatch::new()
                .remove(b"k".to_vec())
                .remove(b"k".to_vec())
                .returning_old_values(),
        ),
    );
    assert!(matches!(ok, Ok(ApplyOutcome::Applied { .. })), "{ok:?}");

    // Five (1 MB) do not: refused, healthily, without materialising them all.
    put(&map, b"k", &v);
    let mut batch = WriteBatch::new();
    for _ in 0..5 {
        batch = batch.remove(b"k".to_vec());
    }
    let err = block_on(map.apply(batch.returning_old_values()));
    assert!(
        matches!(
            err,
            Err(crate::error::WriteError::InvalidBatch(
                crate::batch::BatchError::OldValuesTooLarge
            ))
        ),
        "an oversized old-value response must be refused, got {err:?}"
    );
    // Healthy: not poisoned, and the same batch without the old values applies.
    let ok = block_on(map.apply(WriteBatch::new().remove(b"k".to_vec())));
    assert!(matches!(ok, Ok(ApplyOutcome::Applied { .. })), "{ok:?}");
    block_on(async {
        assert_eq!(map.get(b"k".to_vec()).await.unwrap(), None);
    });
    let _ = block_on(map.close());
}
#[test]
fn an_idempotent_large_reinsert_still_needs_admission_headroom() {
    // The honest boundary of the same-value optimisation.
    // `overflow_equals` runs in the *writer*, but admission gates the
    // command first on `WriteBatch::cost`'s worst case — which must upper-bound the
    // record, and cannot know the batch will turn out to be a no-op without doing
    // the batch. So the compare removes the WAL *consumption*, not the
    // *reservation*: with the WAL nearly full, re-inserting bytes the store already
    // holds is refused even though it would have written nothing.
    //
    // This is not a regression — pre-Phase-F a 200 KB value could not exist at all
    // (MAX_VALUE_LEN was 8 KiB) — but it is a real limitation, and it is pinned
    // here so it cannot be silently believed fixed.
    let (map, _d, _fs) = owned_store_with(Options {
        max_wal_bytes: 400_000,
        ..Options::default()
    });
    let v = blob(41, 200_000);
    put(&map, b"k", &v);
    // ~246 KB logical; the re-insert's worst-case reservation (~377 KB) does not fit.
    assert!(matches!(
        block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), v.clone()))),
        Err(crate::error::WriteError::StoreFull)
    ));
    // Healthy and unchanged: StoreFull is a fail-fast, not a poison.
    block_on(async {
        assert_eq!(map.get(b"k".to_vec()).await.unwrap(), Some(v.clone()));
    });

    // With headroom, the same re-insert is admitted *and* the compare then makes it
    // free — which is what the optimisation actually buys.
    let (map2, _d2, _fs2) = owned_store_with(Options {
        max_wal_bytes: 16 * 1024 * 1024,
        ..Options::default()
    });
    put(&map2, b"k", &v);
    let before = map2.store().metrics().wal_logical_bytes;
    put(&map2, b"k", &v);
    let cost = map2.store().metrics().wal_logical_bytes - before;
    assert!(
        cost < 20_000,
        "the identical re-insert must not rewrite the chain: {cost}"
    );
    let _ = block_on(map.close());
    let _ = block_on(map2.close());
}

#[test]
fn default_limits_admit_any_inline_old_value_response() {
    // The old-value response shares `max_command_bytes` with the command's input,
    // which means tuning that limit down also caps the response.
    // Under the **defaults** no batch of pre-Phase-F-shaped values can reach the
    // ceiling, and that margin is the safety argument for sharing the knob — so it
    // is pinned here rather than left as arithmetic in a doc comment.
    let d = crate::admission::AdmissionLimits::default();
    let worst_inline_response = u64::from(d.max_ops_per_batch) * MAX_INLINE_VALUE_LEN as u64;
    assert!(
        worst_inline_response < d.max_command_bytes,
        "a full batch of inline-sized old values ({worst_inline_response} B) must fit \
         the default command ceiling ({} B), or sharing the knob would refuse batches \
         that predate overflow values",
        d.max_command_bytes
    );
    // Overflow values deliberately blow past it: that is the case the bound exists
    // for, and why the ceiling cannot simply be raised to cover every batch.
    let worst_overflow_response = u64::from(d.max_ops_per_batch) * MAX_VALUE_LEN as u64;
    assert!(worst_overflow_response > d.max_command_bytes);
}

#[test]
fn the_data_space_metric_survives_a_reopen_unrounded() {
    // `data_physical_bytes` is documented as the data file's *length*, and it must
    // report the same value before and after a restart (seeding it from the rounded
    // frontier at open once broke that). Since M7-A the
    // checkpoint manifest write is padded up to a full page so an
    // O_DIRECT open can read it, which extends the real EOF to the page boundary — the
    // length now coincides with the rounded frontier rather than sitting below it. The
    // invariant under test is unchanged: whatever the real length is, a reopen reports
    // the same value.
    let (map, data, file_set) = owned_store();
    put(&map, b"k", b"v");
    let report = block_on(map.store().checkpoint()).unwrap();
    assert!(report.skipped_space.is_none());

    let live = map.store().metrics().data_physical_bytes;
    // Headers + one leaf + a manifest entry padded up to a full page: 4 pages.
    assert_eq!(live, 4 * PGSZ as u64, "the checkpoint's real (padded) EOF");
    assert_eq!(
        live % PGSZ as u64,
        0,
        "the padded manifest aligns the file length"
    );
    let _ = block_on(map.close());

    let (re, _d, _f) = reopen_synced(&data, &file_set);
    assert_eq!(
        re.store().metrics().data_physical_bytes,
        live,
        "a reopen must report the same file length it reported before the restart"
    );
    let _ = block_on(re.close());
}

// ---- M6 Phase G / G3: whole-store exit guard over the quick_cache clean tier ----
// The G3 exit guard: the quick_cache
// clean tier must demonstrably keep pinning and generation accounting intact on
// the REAL store read path, not just in the isolated cache. This drives a Direct
// store under cache pressure, through write -> flush (dirty->clean into qc) ->
// evict -> reload -> reopen (fresh generation, fresh qc), asserting every value
// and the accounting invariants.
#[test]
fn quick_tier_store_read_path_and_accounting_intact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("g3.db");
    let opts = || Options {
        // Tiny clean tier so flushed pages evict from qc and must reload from disk
        // (invariant 5: evictable iff durably readable).
        cache_capacity: Some(8),
        ..Options::default()
    };

    let n = 2000u32;
    block_on(async {
        let store = Store::<crate::Direct>::create_path(&path, opts()).expect("create");
        let map = BTreeMap::over(store);
        for chunk in (0..n).collect::<Vec<_>>().chunks(200) {
            let mut b = WriteBatch::new();
            for &i in chunk {
                b = b.insert(
                    format!("k{i:06}").into_bytes(),
                    format!("val{i:06}").into_bytes(),
                );
            }
            assert!(matches!(
                map.apply(b).await.unwrap(),
                ApplyOutcome::Applied { .. }
            ));
        }
        // Read every key back: descents reload evicted pages through qc, exercise
        // single-flight, and validate_hit on every warm hit.
        for i in 0..n {
            let got = map.get(format!("k{i:06}").into_bytes()).await.unwrap();
            assert_eq!(
                got.as_deref(),
                Some(format!("val{i:06}").as_bytes()),
                "key {i}"
            );
        }
        let m = map.cache_metrics();
        // Per-shard bound: cap 8 may overshoot by up to (shards-1); assert the
        // contract, not an exact 8.
        let shards = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        assert!(
            m.clean_resident <= 8 + shards as u64,
            "clean tier bounded: {}",
            m.clean_resident
        );
        assert!(
            m.evictions > 0,
            "cache pressure evicted through the qc tier"
        );
        assert!(
            m.dirty_resident == 0,
            "no dirty pins linger after quiescence"
        );
        // The lock-free clean path actually served traffic in the store context —
        // with cap 8 the hot upper tree levels are served from qc on every descent:
        // prove hits happened, not just that residency/eviction moved.
        assert!(
            m.hits > 0,
            "warm hits served from the quick_cache clean tier"
        );
        map.close().await.expect("close");
    });

    // Reopen with a FRESH generation and a FRESH qc: every value must survive, which
    // proves the clean tier holds no cross-generation state (ids are globally unique;
    // the qc is per-store, rebuilt on open).
    block_on(async {
        let store = Store::<crate::Direct>::open_path(&path, opts()).expect("reopen");
        let map = BTreeMap::over(store);
        for i in (0..n).step_by(7) {
            let got = map.get(format!("k{i:06}").into_bytes()).await.unwrap();
            assert_eq!(
                got.as_deref(),
                Some(format!("val{i:06}").as_bytes()),
                "reopened key {i}"
            );
        }
        map.close().await.expect("close");
    });
}

/// M6 Phase H — **long soak**. A sustained mixed
/// read / write / checkpoint / compaction workload over a *bounded* key set must
/// keep bounded space and leak nothing: with periodic checkpoints (which rotate
/// and reclaim the WAL) and compactions (which fold the data file dense), the
/// store's owned bytes must stay near the working-set size no matter how many
/// times every key is rewritten — the exact property that fails without
/// maintenance (the WAL, `overlay_pages`, and reopen cost all grow with *total*
/// writes ever made). Also exercises retention: a snapshot held across a
/// checkpoint pins a generation, which must release cleanly when it drops.
///
/// Compressed for CI by default; set `MAPDB_SOAK_ROUNDS` to run a real long soak
/// (e.g. `MAPDB_SOAK_ROUNDS=5000 cargo test --release -- --nocapture soak`).
#[test]
fn long_soak_bounds_space_and_leaks_nothing() {
    const KEYS: u32 = 300;
    const VAL_LEN: usize = 256;
    let rounds: u32 = std::env::var("MAPDB_SOAK_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let (map, _host, file_set) = hosted_wal_store(Options::default());

    // The live working set is bounded: KEYS entries of VAL_LEN. Owned bytes must
    // stay within a generous multiple of it across the whole run.
    let live_bytes = KEYS as u64 * VAL_LEN as u64;
    let mut post_compaction_sizes: Vec<u64> = Vec::new();
    let mut held_snapshot = None;

    block_on(async {
        for round in 0..rounds {
            // Overwrite every key this round (churns the WAL with superseded pages).
            let mut b = WriteBatch::new();
            for k in 0..KEYS {
                b = b.insert(
                    format!("key{k:04}").into_bytes(),
                    vec![round as u8; VAL_LEN],
                );
            }
            let out = map.apply(b).await.unwrap();
            let ApplyOutcome::Applied { version, .. } = out else {
                panic!()
            };
            map.commit(version).await.unwrap();

            // Interleave reads (mixed workload; also validates the current round).
            if round % 3 == 0 {
                for k in (0..KEYS).step_by(50) {
                    assert_eq!(
                        map.get(format!("key{k:04}").into_bytes()).await.unwrap(),
                        Some(vec![round as u8; VAL_LEN])
                    );
                }
            }

            // Hold a snapshot across a checkpoint once, then drop it a few rounds
            // later — exercises the retention pin + its release.
            if round == rounds / 2 {
                held_snapshot = Some(map.snapshot());
            }
            if round == rounds / 2 + 3 {
                drop(held_snapshot.take());
            }

            // Periodic maintenance: checkpoint often, compact occasionally.
            if round % 4 == 3 {
                map.store().checkpoint().await.unwrap();
            }
            if round % 12 == 11 {
                map.store().compact().await.unwrap();
                post_compaction_sizes.push(map.store().metrics().store_physical_bytes);
            }
        }
    });

    let m = map.store().metrics();

    // Space is bounded, not merely capped: owned bytes stay within a generous
    // multiple of the live working set despite `rounds` full rewrites.
    assert!(
        m.store_physical_bytes < live_bytes * 12,
        "owned bytes {} must stay near the {live_bytes}-byte working set after {rounds} rounds",
        m.store_physical_bytes
    );
    // No unbounded growth: post-compaction sizes must not trend up with rounds —
    // each compaction returns space to ~the working set. Assert the spread is
    // bounded (largest ≤ 2× smallest), which linear growth would violate.
    if post_compaction_sizes.len() >= 2 {
        let lo = *post_compaction_sizes.iter().min().unwrap();
        let hi = *post_compaction_sizes.iter().max().unwrap();
        assert!(
            hi <= lo * 2,
            "post-compaction owned bytes must not grow with rounds: {post_compaction_sizes:?}"
        );
    }

    // Correctness held throughout: the bounded key set is all present.
    assert_eq!(m.entry_count, KEYS as u64);
    // Maintenance actually ran and reclaimed.
    assert!(m.compactions >= 1 && m.checkpoints >= m.compactions);
    assert!(
        m.reclaimed_bytes > 0,
        "compactions reclaimed data-file bytes"
    );
    // One generation per open + one per rebind. Every rebind (checkpoint OR
    // compaction) advances `checkpoints` — a compaction is also a checkpoint — so
    // the rebind count is exactly `checkpoints` and generations_created = 1 + that.
    assert_eq!(m.generations_created, 1 + m.checkpoints);
    // No leak: the held snapshot was dropped, so no generation stays pinned, and
    // no segment leaked.
    assert_eq!(
        m.retained_generations, 0,
        "no generation pinned after the snapshot dropped"
    );
    assert_eq!(m.draining_generations, 0);
    assert_eq!(m.retained_segment_bytes, 0);
    assert_eq!(m.leaked_segments, 0);
    assert_eq!(
        file_set.live_segments().len(),
        1,
        "exactly one live WAL segment remains"
    );

    block_on(map.close()).unwrap();
}

/// Builds a store whose root is a **branch** with at least one cold (non-resident)
/// leaf child, and returns the map plus that cold leaf's `PageRef`. Shared by the
/// Running- and Maintenance-arm idle-drive end-to-end tests: a cold child is what
/// lets a fresh single-flight load be installed and then orphaned.
fn branch_root_with_cold_leaf() -> (BTreeMap<crate::Direct>, crate::page::PageRef) {
    use crate::page::PageKind;

    let store = Store::<crate::Direct>::create_with_io(
        Arc::new(FakeIo::new(0)),
        Options {
            // A tiny clean tier keeps leaf children cold (non-resident) after we
            // read only the root, so a fresh load can be installed and orphaned.
            cache_capacity: Some(1),
            ..Options::default()
        },
    )
    .unwrap();
    let map = BTreeMap::over(store);

    // Grow the tree until the root is a branch (so it has cold leaf children).
    // Values are large enough that a few hundred entries overflow one leaf.
    for i in 0..400u32 {
        block_on(
            map.apply(
                WriteBatch::new().insert(format!("k{i:04}").into_bytes(), vec![i as u8; 512]),
            ),
        )
        .unwrap();
    }

    let leaf = {
        let core = &map.store().inner.core;
        let desc = core.root();
        let root = block_on(core.cache.get_page(desc.root, &desc.file)).unwrap();
        assert_eq!(
            root.kind(),
            PageKind::Branch,
            "test needs a branch root so a cold leaf child exists"
        );
        // A child leaf that is not resident (capacity 1 holds at most the root).
        *root
            .branch_children()
            .iter()
            .find(|c| !c.is_empty() && core.cache.state_of(c.page_id).is_none())
            .expect("a cold child leaf exists under a branch root")
    };
    (map, leaf)
}

/// H5 idle drive (end-to-end) — the writer, while **idle**, wakes on load activity
/// and sweeps an orphaned single-flight load to terminal completion, with no
/// rebind, command, or state change to drive it. This is the one part of the idle
/// drive the cache-level tests cannot reach: the writer wiring itself (the third
/// idle `or` arm, the `LoadActivity → continue` routing, and the sweep-on-park).
/// Deleting either the activity arm or the sweep line makes this test time out
/// while every cache-level test still passes.
///
/// Deterministic despite being a liveness wait: because the writer registers its
/// activity listener *before* sweeping, an install is caught either by that sweep
/// (slot already in the map) or by the listener (notify after registration) — no
/// lost-wakeup window — so a healthy writer always terminalizes the orphan and a
/// regression fails as a clean timeout, never a flaky pass.
#[test]
fn idle_writer_drives_an_orphaned_load() {
    use crate::cache::CacheState;

    let (map, leaf) = branch_root_with_cold_leaf();
    // The last apply's reply is awaited, so that write is published; the writer is
    // draining toward its next idle park (the reply is sent before the idle
    // durability flush/park). That is enough: a busy writer unconditionally sweeps
    // on its next idle entry, so the orphan below is driven regardless of the exact
    // moment the writer reaches the park.
    let core = &map.store().inner.core;
    let desc = core.root();

    // Orphan a load: constructing the read future runs `begin` (install + activity
    // notify) synchronously; dropping it unpolled leaves a parked `Loading` slot
    // with no driver — issuing no device read, so *only* the install edge (not a
    // completion) can signal the writer here.
    drop(core.cache.get_page(leaf, &desc.file));
    // The orphan is installed `Loading` with no driver. We do NOT assert `Loading`
    // here: a healthy writer may already have woken on the install edge and swept
    // the never-polled orphan to terminal (FakeIo reads complete inline, so the
    // sweep's first poll issues *and* completes the read) — a legal schedule, not a
    // bug. Any of `Loading` (not yet swept), `Clean`
    // (swept, resident), or `None` (swept, evicted) is correct; `Failed` is not
    // (that would mean the sweep used the Reader policy). The bounded wait below is
    // the decisive assertion.
    assert!(
        !matches!(core.cache.state_of(leaf.page_id), Some(CacheState::Failed)),
        "the orphan must never be cached Failed (sweep must use the Maintenance policy)"
    );

    // No command, no state change, no rebind — the writer's idle activity arm is
    // the only thing that can move this. Bounded liveness wait.
    let start = std::time::Instant::now();
    loop {
        match core.cache.state_of(leaf.page_id) {
            Some(CacheState::Loading) => {
                assert!(
                    start.elapsed() < std::time::Duration::from_secs(10),
                    "idle writer did not sweep the orphan within 10s — wiring regression"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            other => {
                // Clean (resident) or evicted to None — both are terminal. `Failed`
                // would mean the sweep used the Reader policy (wrong): it must not
                // cache a verdict for an orphan.
                assert!(
                    matches!(other, Some(CacheState::Clean) | None),
                    "orphan terminalized to {other:?}; expected Clean or None (Failed = wrong policy)"
                );
                break;
            }
        }
    }

    block_on(map.close()).unwrap();
}

/// H5 idle drive (end-to-end), **Maintenance arm** — the same orphan-drive, but
/// while the writer is parked in `Lifecycle::Maintenance` (an operator pause), not
/// the Running idle arm. Reads stay admitted during a pause, which may be arbitrarily
/// long, so an orphan born here must get the same wall-clock bound. This is the one
/// part the Running test cannot reach: the *independently implemented* Maintenance
/// park arm (`writer.rs`, `or(state_listener, activity)` + its own sweep). Deleting
/// that arm's activity listener + sweep leaves every other test (incl. the Running
/// e2e) green while stranding the orphan until resume — this test then times out.
///
/// No resume, command, or rebind occurs before the assertion, and in Maintenance the
/// Running idle arm is never entered, so the Maintenance arm's sweep is the *only*
/// thing that can move the orphan. Same listen-before-sweep determinism as above.
#[test]
fn maintenance_writer_drives_an_orphaned_load() {
    use crate::cache::CacheState;

    let (map, leaf) = branch_root_with_cold_leaf();
    let core = &map.store().inner.core;
    let desc = core.root();

    // Pause into Maintenance: the writer leaves the Running idle arm and parks in
    // the Maintenance arm. (Reads remain admitted — that is the pause contract.)
    assert!(
        map.store().begin_maintenance(),
        "begin_maintenance must succeed on a live store"
    );

    // Orphan a load *during the pause* — install + activity notify run synchronously
    // in `begin`; dropping the future unpolled leaves a parked `Loading` slot with no
    // driver and no device read (only the install edge can signal the writer).
    drop(core.cache.get_page(leaf, &desc.file));
    assert!(
        !matches!(core.cache.state_of(leaf.page_id), Some(CacheState::Failed)),
        "the orphan must never be cached Failed (sweep must use the Maintenance policy)"
    );

    // No resume, no command, no rebind — only the Maintenance park arm's sweep can
    // move this. Bounded liveness wait; a wiring regression is a clean timeout.
    let start = std::time::Instant::now();
    loop {
        match core.cache.state_of(leaf.page_id) {
            Some(CacheState::Loading) => {
                assert!(
                    start.elapsed() < std::time::Duration::from_secs(10),
                    "maintenance-parked writer did not sweep the orphan within 10s — wiring regression"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            other => {
                assert!(
                    matches!(other, Some(CacheState::Clean) | None),
                    "orphan terminalized to {other:?}; expected Clean or None (Failed = wrong policy)"
                );
                break;
            }
        }
    }

    // Resume, then close cleanly.
    map.store().end_maintenance();
    block_on(map.close()).unwrap();
}

/// End-to-end idle-drive test — the idle writer, driving parked loads through the
/// bounded `drive_pending` queue instead of the old full-set sweep, still services
/// **every** orphaned load. Orphan a load for *each* cold leaf child of a branch
/// root (many distinct single-flight loads at once), then bound-wait for the idle
/// writer to terminalize all of them and the pending-drive queue to drain to zero —
/// the eventual-service + no-leak property the queue must preserve. A wiring
/// regression (e.g. the bounded drain never re-loops on `more`, so a batch's
/// overflow is stranded) fails as a clean timeout, never a flaky pass.
#[test]
fn idle_writer_services_every_orphaned_load() {
    use crate::cache::CacheState;

    let (map, _one) = branch_root_with_cold_leaf();
    let core = &map.store().inner.core;
    let desc = core.root();
    let root = block_on(core.cache.get_page(desc.root, &desc.file)).unwrap();
    assert_eq!(root.kind(), crate::page::PageKind::Branch);

    // Every non-empty leaf child that is currently cold — orphan a load for each.
    let cold: Vec<_> = root
        .branch_children()
        .iter()
        .filter(|c| !c.is_empty() && core.cache.state_of(c.page_id).is_none())
        .copied()
        .collect();
    assert!(
        cold.len() >= 3,
        "need several cold children for a multi-orphan test, got {}",
        cold.len()
    );

    for leaf in &cold {
        // `begin` (install + enqueue + activity notify) runs synchronously; dropping
        // the never-polled future orphans the load with no driver.
        drop(core.cache.get_page(*leaf, &desc.file));
    }
    // None may be cached Failed (that would be a corrupt/verdict bug, not an orphan
    // drive).
    for leaf in &cold {
        assert!(
            !matches!(core.cache.state_of(leaf.page_id), Some(CacheState::Failed)),
            "orphan {:?} must never be Failed",
            leaf.page_id
        );
    }

    // No command / state change / rebind — only the idle writer's bounded drive can
    // move these. Bounded liveness wait for ALL of them to terminalize.
    let start = std::time::Instant::now();
    loop {
        let pending: Vec<_> = cold
            .iter()
            .filter(|leaf| matches!(core.cache.state_of(leaf.page_id), Some(CacheState::Loading)))
            .collect();
        if pending.is_empty() {
            break;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "idle writer did not service all orphans within 10s ({} still Loading) — wiring regression",
            pending.len()
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    for leaf in &cold {
        assert!(
            matches!(
                core.cache.state_of(leaf.page_id),
                Some(CacheState::Clean) | None
            ),
            "orphan {:?} terminalized wrong",
            leaf.page_id
        );
    }
    // The pending-drive queue drained to empty (no accumulated dead edges).
    let start = std::time::Instant::now();
    while core.cache.metrics().pending_drive_depth != 0 {
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "pending-drive queue did not drain to zero — leak"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    block_on(map.close()).unwrap();
}

/// End-to-end busy-path test — orphaned loads are serviced and the
/// pending-drive queue stays bounded while the writer is under a **sustained command
/// stream** (it runs the bounded post-apply `drive_pending` in its command arms, not
/// only the idle arm). Orphan the cold leaves, then push a stream of write batches
/// through the writer; by the end every orphan has terminalized and the queue has
/// drained to zero — the eventual-service + no-leak property under load. Exercises the
/// writer's busy-path command/drain wiring; a leak (queue growing unbounded, orphans
/// never serviced under load) fails as a bounded-liveness timeout.
#[test]
fn orphans_are_serviced_under_a_command_stream() {
    use crate::cache::CacheState;

    let (map, _one) = branch_root_with_cold_leaf();
    let cold: Vec<_> = {
        let core = &map.store().inner.core;
        let desc = core.root();
        let root = block_on(core.cache.get_page(desc.root, &desc.file)).unwrap();
        root.branch_children()
            .iter()
            .filter(|c| !c.is_empty() && core.cache.state_of(c.page_id).is_none())
            .copied()
            .collect()
    };
    assert!(
        cold.len() >= 3,
        "need several cold children, got {}",
        cold.len()
    );

    // Orphan a load for each cold leaf.
    {
        let core = &map.store().inner.core;
        let desc = core.root();
        for leaf in &cold {
            drop(core.cache.get_page(*leaf, &desc.file));
        }
    }

    // Drive a sustained command stream through the writer: every apply reply is
    // awaited, so the writer keeps cycling its command arms (which run the busy-path
    // drain) rather than sitting idle.
    for i in 0..40u32 {
        block_on(
            map.apply(WriteBatch::new().insert(format!("s{i:04}").into_bytes(), vec![i as u8; 64])),
        )
        .unwrap();
    }

    // Every orphan terminalized (Clean or evicted None; never Failed), and the
    // pending-drive queue drained to zero — no leak under load.
    let core = &map.store().inner.core;
    let start = std::time::Instant::now();
    loop {
        let stuck: Vec<_> = cold
            .iter()
            .filter(|l| matches!(core.cache.state_of(l.page_id), Some(CacheState::Loading)))
            .collect();
        let depth = core.cache.metrics().pending_drive_depth;
        if stuck.is_empty() && depth == 0 {
            break;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "under a command stream, {} orphans still Loading / queue depth {} — busy-path service regression",
            stuck.len(),
            depth
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    for leaf in &cold {
        assert!(
            !matches!(core.cache.state_of(leaf.page_id), Some(CacheState::Failed)),
            "orphan {:?} must never be Failed",
            leaf.page_id
        );
    }

    block_on(map.close()).unwrap();
}

// M7-E: the bench-only `Options::device_wrapper`
// must decorate *every* device the production path mints, **including those minted
// after `create_path` returns** — a rotated WAL segment (via `FsWalFileSet`) and a
// compaction temp (via `FsDataHost`), both of which hold a clone of the store's
// `DeviceFactory`. A lifetime argument that "the factory persists" is not proof;
// this drives a real rotation + compaction and asserts the wrapper saw the
// post-open mints, role-tagged. Without this the most production-specific part of
// the M7-E instrumentation seam could silently regress.
#[cfg(feature = "bench-instrument")]
#[test]
fn the_device_wrapper_decorates_every_role_minted_after_open() {
    use crate::io::{IoRole, PageIo};
    use std::sync::{Arc, Mutex};

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m7e-wrap.db");

    // Records the role of every device the factory mints, in mint order. The
    // wrapper passes the device through unchanged — this test proves only that the
    // seam fires per role, which is all the future-device claim needs.
    let roles: Arc<Mutex<Vec<IoRole>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&roles);
    let wrapper: crate::io::DeviceWrapper = Arc::new(move |dev: Arc<dyn PageIo>, role: IoRole| {
        sink.lock().unwrap().push(role);
        dev
    });

    let opts = Options {
        device_wrapper: Some(wrapper),
        ..Options::default()
    };

    block_on(async {
        let store = Store::<Wal>::create_path(&path, opts).expect("create");
        let map = BTreeMap::over(store);
        // Build a live set worth relocating; overwrite so the WAL holds superseded
        // CoW ancestors a checkpoint will drop.
        for pass in 0..2 {
            for chunk in (0..120u32).collect::<Vec<_>>().chunks(40) {
                let mut b = WriteBatch::new();
                for &i in chunk {
                    b = b.insert(format!("k{i:05}").into_bytes(), vec![(i + pass) as u8; 300]);
                }
                assert!(matches!(
                    map.apply(b).await.unwrap(),
                    ApplyOutcome::Applied { .. }
                ));
            }
        }

        // Snapshot what create_path minted (the data file + segment 0) before any
        // post-open mint. Writes do not mint devices, so this is stable here.
        let after_open = roles.lock().unwrap().clone();
        assert!(
            after_open.contains(&IoRole::Data) && after_open.contains(&IoRole::WalSegment),
            "create_path minted a wrapped Data + WalSegment(0): {after_open:?}"
        );

        // Checkpoint rotates the WAL segment → a NEW WalSegment is minted by the
        // factory the FsWalFileSet kept. Compaction mints a CompactTemp via the
        // factory the FsDataHost kept. Both are strictly after `after_open`.
        map.store().checkpoint().await.unwrap();
        map.store().compact().await.unwrap();
        map.close().await.unwrap();

        let all = roles.lock().unwrap().clone();
        let post = &all[after_open.len()..];
        assert!(
            post.contains(&IoRole::WalSegment),
            "checkpoint rotation minted a wrapped WalSegment after open: all={all:?}"
        );
        assert!(
            post.contains(&IoRole::CompactTemp),
            "compaction minted a wrapped CompactTemp after open: all={all:?}"
        );
    });
}
