//! The durable real-file create/open protocol for StoreWal: a data file, a
//! numbered append-only WAL **segment**, and a stable sidecar lock, all named
//! from one `path` (`<path>`, `<path>.wal.<seq>`, `<path>.lock`).
//!
//! Creation is a durability protocol mirroring StoreDirect's: take
//! the exclusive sidecar lock (a file separate from the replaceable data inode),
//! create/truncate both files so their directory entries exist, write + fdatasync
//! both headers (the paired `store_uuid`), then `fsync` the parent directory so
//! the new entries are durable — so a crash right after a "successful" create
//! either loses the whole (unacknowledged) store or reopens it cleanly, never
//! leaving an acknowledged-but-unrecoverable file.
//!
//! **Segment naming (M6 Phase B).** WAL space is reclaimed by rotating
//! to a fresh segment at each checkpoint and unlinking the superseded one, never
//! by in-place reuse. The segment in force is therefore not a fixed name: the
//! *data header* names it (`wal_segment_seq`), and `open_path` resolves
//! `<path>.wal.<seq>` from the header it just read rather than guessing. A
//! segment orphaned by a crash between the header commit and the unlink is
//! unreferenced by the winning header and simply ignored.
//!
//! Open takes the same lock, opens the *existing* files only, and runs crash
//! recovery (`open_over`). The lock's `File` is owned by the store's close hook
//! (released on close and on last-handle drop), exactly as for Direct.

use crate::backend::Wal;
use crate::error::OpenError;
use crate::fileset::WalFileSet;
use crate::io::{DeviceFactory, FileRole, PageIo};
use crate::store::{Options, Store};
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The path of WAL segment `seq` for the store at `path`.
pub(crate) fn segment_path(path: &Path, seq: u64) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(format!(".wal.{seq}"));
    PathBuf::from(s)
}

use crate::fs_util::{
    acquire_lock, fsync_parent_dir, only_format_refusal, read_prefix_unlocked, sidecar_missing,
    CreateClaim,
};

/// Whether ANY `<stem>.wal.<digits>` segment exists for `path` — the WAL
/// member classification of the fail-safe create (T1.5). Scans the directory
/// exactly like `retire_other_segments` (leftovers need not be contiguous).
fn any_segment_exists(path: &Path) -> io::Result<bool> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = dir.unwrap_or_else(|| Path::new("."));
    let Some(stem) = path.file_name().and_then(|s| s.to_str()) else {
        return Ok(false);
    };
    let prefix = format!("{stem}.wal.");
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(rest) = name.strip_prefix(&prefix) {
            if rest.parse::<u64>().is_ok() {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The real-filesystem WAL segment namespace for the store at `path`: segments
/// are `<path>.wal.<seq>`. Every mutation runs under the store's already-held
/// sidecar `flock` (the store owns this file set for its whole lifetime), so no
/// two processes can race over the same names.
pub(crate) struct FsWalFileSet {
    path: PathBuf,
    /// The store's device backend (M7-0). A rotated WAL segment is minted with the
    /// same backend as the store that owns the file set. **`direct_io` (M7-A) will
    /// never apply here** — the WAL record stream is byte-granular (`wal.rs`
    /// append) and cannot be O_DIRECT; only the data-file role gets O_DIRECT.
    factory: DeviceFactory,
}

impl WalFileSet for FsWalFileSet {
    fn create_segment(&self, seq: u64, store_uuid: [u8; 16]) -> io::Result<Arc<dyn PageIo>> {
        let p = segment_path(&self.path, seq);
        // `create_new`: a live segment is never recreated, and an orphan left by a
        // crashed checkpoint must not be silently reused — its stale bytes past
        // the header would be scanned as records of the new segment.
        OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&p)?;
        let dev = self.factory.open(&p, FileRole::WalSegment)?;
        futures_lite::future::block_on(async {
            dev.write_all_at(0, crate::wal::encode_wal_header(store_uuid, seq))
                .await
                .result?;
            dev.datasync().await
        })?;
        // The header is durable, but the *directory entry* is not until the parent
        // is fsynced — without this a crash could commit a data header naming a
        // segment that does not exist, which is exactly the state recovery cannot
        // resolve.
        fsync_parent_dir(&p)?;
        Ok(dev)
    }

    fn open_segment(&self, seq: u64) -> io::Result<Arc<dyn PageIo>> {
        self.factory
            .open_existing(&segment_path(&self.path, seq), FileRole::WalSegment)
    }

    fn unlink_segment(&self, seq: u64) -> io::Result<()> {
        match std::fs::remove_file(segment_path(&self.path, seq)) {
            Ok(()) => {}
            // Idempotent: a crash may have unlinked it already.
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        }
        // Make the *removal* durable too, so the freed space is not resurrected by
        // a crash right after a checkpoint.
        fsync_parent_dir(&self.path)
    }

    fn retire_other_segments(&self, keep: u64) -> io::Result<u64> {
        // Scan the directory rather than guessing sequence numbers: after several
        // crashes the leftovers need not be contiguous with `keep`.
        let dir = self.path.parent().filter(|p| !p.as_os_str().is_empty());
        let dir = dir.unwrap_or_else(|| Path::new("."));
        let Some(stem) = self.path.file_name().and_then(|s| s.to_str()) else {
            return Ok(0);
        };
        let prefix = format!("{stem}.wal.");
        let mut removed = 0u64;
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(rest) = name.strip_prefix(&prefix) else {
                continue;
            };
            // Only exact `<stem>.wal.<digits>` — never anything else in the dir.
            let Ok(seq) = rest.parse::<u64>() else {
                continue;
            };
            if seq == keep {
                continue;
            }
            match std::fs::remove_file(entry.path()) {
                Ok(()) => removed += 1,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        if removed > 0 {
            fsync_parent_dir(&self.path)?;
        }
        Ok(removed)
    }
}

/// The pre-lock format probe for a store with no sidecar ([`sidecar_missing`]):
/// read-only, it refuses only an authenticated unsupported data-header or
/// in-force segment-header format; anything else is left to the locked open.
fn refuse_unsupported_format_unlocked(path: &Path) -> Result<(), OpenError> {
    use crate::page::PAGE_SIZE;
    let Some(slots) = read_prefix_unlocked(path, 2 * PAGE_SIZE) else {
        return Ok(());
    };
    let (slot0, slot1) = slots.split_at(PAGE_SIZE);
    let Some(header) = only_format_refusal(crate::checkpoint::pick_winning_header(slot0, slot1))?
    else {
        return Ok(());
    };
    let seg = segment_path(path, header.wal_segment_seq);
    if let Some(bytes) = read_prefix_unlocked(&seg, PAGE_SIZE) {
        only_format_refusal(crate::wal::decode_wal_header(&bytes))?;
    }
    Ok(())
}

impl Store<Wal> {
    /// Creates a fresh StoreWal at `path` (data) + `<path>.wal.0` under the
    /// durable creation protocol (atomic sidecar claim, paired headers +
    /// `fdatasync`, directory fsync). **Fail-safe (T1.5):** refuses
    /// with [`OpenError::AlreadyExists`] if any store member — data file, any
    /// `<path>.wal.<seq>` segment, or a compaction temp — already exists;
    /// pre-existing bytes are never touched. [`OpenError::AlreadyOpen`] if the
    /// sidecar lock is held. Replacing an existing store is
    /// [`create_path_overwrite`](Self::create_path_overwrite).
    ///
    /// Recovery of a store created here assumes powersafe overwrite on the
    /// underlying filesystem — see the [`Wal`] backend docs for the supported set.
    pub fn create_path(path: impl AsRef<Path>, options: Options) -> Result<Store<Wal>, OpenError> {
        let path = path.as_ref();
        let mut claim = CreateClaim::claim(path)?;
        // Classify the whole WAL namespace UNDER the held lock.
        let collision = if path.exists() {
            Some("data file")
        } else if any_segment_exists(path)? {
            Some("wal segment")
        } else if crate::compact::temp_path(path).exists() {
            Some("compaction temp")
        } else {
            None
        };
        if let Some(member) = collision {
            claim.abandon(); // removes only a sidecar this attempt minted
            return Err(OpenError::AlreadyExists(member));
        }
        // Claim the data inode with `create_new` (a lockless squatter racing in
        // is refused, never truncated).
        if let Err(e) = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)
        {
            claim.abandon();
            return match e.kind() {
                io::ErrorKind::AlreadyExists => Err(OpenError::AlreadyExists("data file")),
                _ => Err(OpenError::from(e)),
            };
        }
        let lock = claim.take_lock();
        match Self::create_over_claimed(path, options, lock) {
            Ok(store) => Ok(store),
            Err(e) => {
                // Roll back exactly what this attempt created: data inode,
                // segment 0 (create_over_claimed mints it with create_new — a
                // pre-existing one was refused above), sidecar if ours.
                let _ = std::fs::remove_file(segment_path(path, 0));
                let _ = std::fs::remove_file(path);
                claim.abandon();
                Err(e)
            }
        }
    }

    /// **Destructively** creates a fresh StoreWal at `path`, replacing whatever
    /// store or partial file set exists there (the pre-T1.5 `create_path`
    /// behavior: truncate the data file, delete every segment, remove a stale
    /// compaction temp). Fails with [`OpenError::AlreadyOpen`] if the sidecar
    /// lock is held. **Warning:** a path mix-up silently destroys a
    /// store — reach for [`create_path`](Self::create_path) unless replacement
    /// is the explicit intent.
    pub fn create_path_overwrite(
        path: impl AsRef<Path>,
        options: Options,
    ) -> Result<Store<Wal>, OpenError> {
        let path = path.as_ref();
        let lock = acquire_lock(path)?;
        // The destructive steps live ONLY here (T1.5): stale compaction temp,
        // the whole segment set, and the data file's stale bytes.
        crate::compact::remove_stale_temp(path)?;
        OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let file_set = FsWalFileSet {
            path: path.to_path_buf(),
            factory: DeviceFactory::new(options.io_backend, options.direct_io),
        };
        // Truncating a store means truncating its whole file set: a leftover
        // segment from the previous store at this path would otherwise be
        // `create_new`-rejected in the shared tail. `keep = 0` then gets
        // recreated immediately.
        file_set.retire_other_segments(0)?;
        let _ = std::fs::remove_file(segment_path(path, 0));
        Self::create_over_claimed(path, options, lock)
    }

    /// The shared durable-create tail over an already-locked, already-created
    /// (empty) data inode: mint segment 0 (`create_new`), write both headers
    /// durably, fsync the directory, and build the store. `lock` is the held
    /// sidecar flock, handed to the store (writer-thread ownership, T1.2).
    fn create_over_claimed(
        path: &Path,
        options: Options,
        lock: File,
    ) -> Result<Store<Wal>, OpenError> {
        let factory = DeviceFactory::new(options.io_backend, options.direct_io);
        #[cfg(feature = "bench-instrument")]
        let factory = factory.with_wrapper(options.device_wrapper.clone());
        let data: Arc<dyn PageIo> = factory.open(path, FileRole::Data)?;
        let file_set = Arc::new(FsWalFileSet {
            path: path.to_path_buf(),
            factory: factory.clone(),
        });
        let store_uuid = crate::direct::generate_store_uuid();
        // Mints segment 0 with its identity header, syncs it, and fsyncs the
        // parent directory; then the data header is written + synced.
        let wal = file_set.create_segment(0, store_uuid)?;
        let host = Arc::new(crate::compact::FsDataHost {
            path: path.to_path_buf(),
            factory,
        });
        let store = Store::<Wal>::create_with_io_owned(
            data,
            wal,
            store_uuid,
            Some(file_set as Arc<dyn WalFileSet>),
            Some(host),
            options,
            Some(lock),
        )?;
        // The new files' headers and directory entries are now durable. If the
        // directory fsync fails, stop the just-spawned writer (drain/join) before
        // returning; the writer releases the sidecar lock as it exits, so the
        // lock is never freed while the writer still runs (T1.2).
        if let Err(e) = fsync_parent_dir(path) {
            let _ = futures_lite::future::block_on(store.close());
            return Err(OpenError::from(e));
        }
        Ok(store)
    }

    /// Opens an existing StoreWal at `path` under the sidecar lock, running crash
    /// recovery. The WAL segment to open is **resolved from the data header**
    /// (`<path>.wal.<wal_segment_seq>`), not assumed. The lock is held for the
    /// store's lifetime.
    pub fn open_path(path: impl AsRef<Path>, options: Options) -> Result<Store<Wal>, OpenError> {
        let path = path.as_ref();
        let factory = DeviceFactory::new(options.io_backend, options.direct_io);
        #[cfg(feature = "bench-instrument")]
        let factory = factory.with_wrapper(options.device_wrapper.clone());
        if !path.exists() {
            return Err(OpenError::from(io::Error::new(
                io::ErrorKind::NotFound,
                "store data file does not exist",
            )));
        }
        // A store without its sidecar: refuse an unsupported format before
        // `acquire_lock` mints one (fable72 F3).
        if sidecar_missing(path) {
            refuse_unsupported_format_unlocked(path)?;
        }
        let lock = acquire_lock(path)?;
        let data: Arc<dyn PageIo> = factory.open_existing(path, FileRole::Data)?;
        // Read the winning header once here purely to name the segment; `open_over`
        // re-reads it under the lock as recovery's authority (both reads see the
        // same bytes — the file is not written until the writer starts).
        let header =
            futures_lite::future::block_on(crate::checkpoint::read_winning_header(&*data))?;
        let wpath = segment_path(path, header.wal_segment_seq);
        if !wpath.exists() {
            return Err(OpenError::from(io::Error::new(
                io::ErrorKind::NotFound,
                "wal segment named by the data header does not exist",
            )));
        }
        let wal: Arc<dyn PageIo> = factory.open_existing(&wpath, FileRole::WalSegment)?;
        // Every open-time gate — the data-header format above, then the segment
        // header's format and identity here — refuses before the namespace
        // cleanup below: a downgrade refusal must preserve even an orphaned
        // compaction image or a newer release's segments (M6 Phase D/H, fable72
        // F1). Recovery re-runs the same check as its own authority.
        futures_lite::future::block_on(crate::wal::check_segment_identity(&header, &*wal))?;
        // Active-path-wins: remove an orphaned compaction temp.
        crate::compact::remove_stale_temp(path)?;
        let file_set = Arc::new(FsWalFileSet {
            path: path.to_path_buf(),
            factory: factory.clone(),
        });
        // Complete any rotation a crash interrupted before its unlink: under the
        // held lock, before the writer starts, no snapshot can exist yet, and the
        // winning header names the only segment anything may still read.
        file_set.retire_other_segments(header.wal_segment_seq)?;
        let host = Arc::new(crate::compact::FsDataHost {
            path: path.to_path_buf(),
            factory,
        });
        Store::<Wal>::open_owned(
            data,
            wal,
            Some(file_set as Arc<dyn WalFileSet>),
            Some(host),
            options,
            Some(lock),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{ApplyOutcome, WriteBatch};
    use crate::fs_util::lock_path;
    use crate::map::BTreeMap;
    use futures_lite::future::block_on;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_path(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "mapdb-uring-m4-{}-{}-{}.data",
            tag,
            std::process::id(),
            n
        ));
        cleanup(&p);
        p
    }

    /// Removes the whole file set: the data file, the lock, and every segment a
    /// rotation may have left behind.
    fn cleanup(p: &Path) {
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_file(lock_path(p));
        for seq in 0..8 {
            let _ = std::fs::remove_file(segment_path(p, seq));
        }
    }

    /// A fixture path under `target/` (real disk, O_DIRECT-capable) with the whole
    /// file set pre-cleaned — the O_DIRECT tests need a filesystem that honors it
    /// (M7-A).
    fn real_path(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push("target");
        dir.push("m7a-direct-tests");
        let _ = std::fs::create_dir_all(&dir);
        dir.push(format!("wal-{}-{}-{}.data", tag, std::process::id(), n));
        cleanup(&dir);
        dir
    }

    #[test]
    fn direct_io_wal_checkpoint_reopens_with_buffered_wal_segments() {
        // M7-A: a StoreWal opened `direct_io` puts its **data file**
        // (the checkpoint target, with the page-padded manifest) on O_DIRECT while the
        // byte-granular **WAL record segments** stay buffered (the role split). A
        // checkpoint writes the manifest through the O_DIRECT data fd and rotates a
        // (buffered) segment; a reopen reads the manifest back under O_DIRECT.
        let direct = Options {
            direct_io: true,
            ..Options::default()
        };
        // Probe-and-skip if this host cannot open O_DIRECT.
        {
            let probe = real_path("probe");
            let available = Store::<Wal>::create_path(&probe, direct.clone()).is_ok();
            cleanup(&probe);
            if !available {
                eprintln!("skip: O_DIRECT unavailable under target/ on this host");
                return;
            }
        }
        let path = real_path("ckpt");
        let n = 80u32;
        {
            let map = BTreeMap::over(Store::<Wal>::create_path(&path, direct.clone()).unwrap());
            block_on(async {
                for i in 0..n {
                    map.apply(
                        WriteBatch::new()
                            .insert(format!("k{i:04}").into_bytes(), vec![i as u8; 250]),
                    )
                    .await
                    .unwrap();
                }
                // Checkpoint: manifest written to the O_DIRECT data file, WAL rotated.
                let report = map.store().checkpoint().await.unwrap();
                assert!(report.skipped_space.is_none());
                // A post-checkpoint write goes to the fresh (buffered) segment.
                map.apply(WriteBatch::new().insert(b"after".to_vec(), b"v".to_vec()))
                    .await
                    .unwrap();
            });
            block_on(map.close()).unwrap();
        }
        // Reopen O_DIRECT: recovery reads the winning header + padded manifest from the
        // O_DIRECT data file and replays the buffered tail segment.
        let map = BTreeMap::over(Store::<Wal>::open_path(&path, direct).unwrap());
        block_on(async {
            assert_eq!(map.len() as u32, n + 1);
            assert_eq!(
                map.get(b"k0042".to_vec()).await.unwrap(),
                Some(vec![42u8; 250])
            );
            assert_eq!(
                map.get(b"after".to_vec()).await.unwrap(),
                Some(b"v".to_vec())
            );
            let out = map
                .apply(WriteBatch::new().insert(b"new".to_vec(), b"v".to_vec()))
                .await
                .unwrap();
            assert!(matches!(out, ApplyOutcome::Applied { .. }));
        });
        block_on(map.close()).unwrap();
        cleanup(&path);
    }

    #[cfg(feature = "io-uring")]
    #[test]
    fn uring_direct_io_wal_checkpoint_reopens() {
        // M7-A commit 2 (the 2x2 upper-right cell): a StoreWal on the io_uring backend
        // *and* O_DIRECT — the data file (checkpoint target + padded manifest) rides
        // the ring with aligned bounce buffers while the WAL segments stay buffered.
        let opts = || crate::store::Options {
            io_backend: crate::io::IoBackend::Uring,
            direct_io: true,
            ..Options::default()
        };
        {
            let probe = real_path("uprobe");
            let available = Store::<Wal>::create_path(&probe, opts()).is_ok();
            cleanup(&probe);
            if !available {
                eprintln!("skip: io_uring + O_DIRECT unavailable on this host");
                return;
            }
        }
        let path = real_path("uckpt");
        let n = 80u32;
        {
            let map = BTreeMap::over(Store::<Wal>::create_path(&path, opts()).unwrap());
            block_on(async {
                for i in 0..n {
                    map.apply(
                        WriteBatch::new()
                            .insert(format!("k{i:04}").into_bytes(), vec![i as u8; 250]),
                    )
                    .await
                    .unwrap();
                }
                map.store().checkpoint().await.unwrap();
                map.apply(WriteBatch::new().insert(b"after".to_vec(), b"v".to_vec()))
                    .await
                    .unwrap();
            });
            block_on(map.close()).unwrap();
        }
        let map = BTreeMap::over(Store::<Wal>::open_path(&path, opts()).unwrap());
        block_on(async {
            assert_eq!(map.len() as u32, n + 1);
            assert_eq!(
                map.get(b"k0042".to_vec()).await.unwrap(),
                Some(vec![42u8; 250])
            );
            assert_eq!(
                map.get(b"after".to_vec()).await.unwrap(),
                Some(b"v".to_vec())
            );
        });
        block_on(map.close()).unwrap();
        cleanup(&path);
    }

    #[test]
    fn direct_io_reopens_a_legacy_unpadded_manifest() {
        // Before M7-A the checkpoint manifest was written *unpadded*,
        // so the data file ended at a non-page-aligned offset. Opening such a file
        // `direct_io` must still work — the device rounds the manifest read up past EOF
        // and tolerates the short return. No shipped path writes an unpadded
        // manifest anymore, so synthesize one: checkpoint (which pads), then truncate
        // the page padding back off, leaving an old-format, unaligned-EOF data file.
        let direct = Options {
            direct_io: true,
            ..Options::default()
        };
        {
            let probe = real_path("probe2");
            let available = Store::<Wal>::create_path(&probe, direct.clone()).is_ok();
            cleanup(&probe);
            if !available {
                eprintln!("skip: O_DIRECT unavailable under target/ on this host");
                return;
            }
        }
        let path = real_path("legacy");
        let n = 70u32;
        {
            let map = BTreeMap::over(Store::<Wal>::create_path(&path, Options::default()).unwrap());
            block_on(async {
                for i in 0..n {
                    map.apply(
                        WriteBatch::new()
                            .insert(format!("k{i:04}").into_bytes(), vec![i as u8; 300]),
                    )
                    .await
                    .unwrap();
                }
                map.store().checkpoint().await.unwrap();
            });
            block_on(map.close()).unwrap();
        }
        // Read the winning header to find the manifest's true (unpadded) end.
        let true_end = {
            let dev = crate::io::SyscallIo::open_existing(&path).unwrap();
            let h = block_on(crate::checkpoint::read_winning_header(&dev)).unwrap();
            block_on(dev.shutdown()).unwrap();
            h.manifest_offset + h.manifest_len
        };
        assert_ne!(
            true_end % crate::page::PAGE_SIZE as u64,
            0,
            "the legacy fixture must end unaligned to exercise the past-EOF read"
        );
        // Truncate off the page padding → an old-format, unaligned-EOF data file.
        {
            let f = OpenOptions::new().write(true).open(&path).unwrap();
            f.set_len(true_end).unwrap();
        }
        // Reopen O_DIRECT: read_manifest rounds the read up past the now-unpadded EOF
        // and the device tolerates the short return.
        let map = BTreeMap::over(Store::<Wal>::open_path(&path, direct).unwrap());
        block_on(async {
            assert_eq!(map.len() as u32, n);
            assert_eq!(
                map.get(b"k0007".to_vec()).await.unwrap(),
                Some(vec![7u8; 300])
            );
            assert_eq!(
                map.get(b"k0069".to_vec()).await.unwrap(),
                Some(vec![69u8; 300])
            );
        });
        block_on(map.close()).unwrap();
        cleanup(&path);
    }

    #[test]
    fn create_write_commit_close_reopen_recovers_real_files() {
        let path = unique_path("reopen");
        {
            let store = Store::<Wal>::create_path(&path, Options::default()).unwrap();
            let map = BTreeMap::over(store);
            block_on(async {
                for i in 0..40u32 {
                    let out = map
                        .apply(
                            WriteBatch::new()
                                .insert(format!("k{i:03}").into_bytes(), vec![i as u8; 250]),
                        )
                        .await
                        .unwrap();
                    let ApplyOutcome::Applied { version, .. } = out else {
                        panic!()
                    };
                    map.commit(version).await.unwrap();
                }
                map.apply(WriteBatch::new().remove(b"k007".to_vec()))
                    .await
                    .unwrap();
            });
            block_on(map.close()).unwrap();
        }
        // Reopen the real files and recover the committed prefix.
        let store = Store::<Wal>::open_path(&path, Options::default()).unwrap();
        let map = BTreeMap::over(store);
        block_on(async {
            assert_eq!(
                map.get(b"k000".to_vec()).await.unwrap(),
                Some(vec![0u8; 250])
            );
            assert_eq!(map.get(b"k007".to_vec()).await.unwrap(), None);
            assert_eq!(
                map.get(b"k039".to_vec()).await.unwrap(),
                Some(vec![39u8; 250])
            );
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
        cleanup(&path);
    }

    #[test]
    fn exclusive_open_rejected_then_released_on_close() {
        let path = unique_path("excl");
        let map = BTreeMap::over(Store::<Wal>::create_path(&path, Options::default()).unwrap());
        let second = Store::<Wal>::open_path(&path, Options::default());
        assert!(
            matches!(second, Err(OpenError::AlreadyOpen)),
            "second open must be rejected"
        );
        block_on(map.close()).unwrap();
        let reopened = Store::<Wal>::open_path(&path, Options::default());
        assert!(reopened.is_ok(), "open after close should succeed");
        block_on(BTreeMap::over(reopened.unwrap()).close()).unwrap();
        drop(map);
        cleanup(&path);
    }

    #[test]
    fn checkpoint_rotates_the_real_segment_and_reopens_from_the_header() {
        let path = unique_path("ckpt");
        {
            let map = BTreeMap::over(Store::<Wal>::create_path(&path, Options::default()).unwrap());
            block_on(async {
                for i in 0..50u32 {
                    let out = map
                        .apply(
                            WriteBatch::new()
                                .insert(format!("k{i:02}").into_bytes(), vec![i as u8; 400]),
                        )
                        .await
                        .unwrap();
                    let ApplyOutcome::Applied { version, .. } = out else {
                        panic!()
                    };
                    map.commit(version).await.unwrap();
                }
            });
            assert!(segment_path(&path, 0).exists(), "segment 0 is in force");

            let report = block_on(map.store().checkpoint()).unwrap();
            assert_eq!(report.checkpoint_seq, 1);
            assert!(report.relocated_pages > 0);
            // Rotation on a real filesystem: the retired segment's directory entry
            // is gone, and the new one is durable.
            assert!(
                !segment_path(&path, 0).exists(),
                "segment 0 unlinked at cutover"
            );
            assert!(segment_path(&path, 1).exists(), "segment 1 is in force");

            // Writes keep flowing into the new segment.
            block_on(async {
                let out = map
                    .apply(WriteBatch::new().insert(b"after".to_vec(), b"v".to_vec()))
                    .await
                    .unwrap();
                let ApplyOutcome::Applied { version, .. } = out else {
                    panic!()
                };
                map.commit(version).await.unwrap();
            });
            block_on(map.close()).unwrap();
        }
        // Reopen resolves the segment from the data header — nothing knows the
        // name `.wal.1` except the header.
        let map = BTreeMap::over(Store::<Wal>::open_path(&path, Options::default()).unwrap());
        block_on(async {
            assert_eq!(
                map.get(b"k00".to_vec()).await.unwrap(),
                Some(vec![0u8; 400])
            );
            assert_eq!(
                map.get(b"k49".to_vec()).await.unwrap(),
                Some(vec![49u8; 400])
            );
            assert_eq!(
                map.get(b"after".to_vec()).await.unwrap(),
                Some(b"v".to_vec())
            );
        });
        assert_eq!(map.store().metrics().last_checkpoint_txid, 50);
        block_on(map.close()).unwrap();
        cleanup(&path);
    }

    /// M7-0: a StoreWal created on the `Uring` backend runs the whole production
    /// path — write + commit, checkpoint (rotates the segment through the factory,
    /// mints segment 1 on the ring), reopen from the header on the ring, read back.
    /// Exercises the data-file, WAL-segment, and rotation device sites on io_uring.
    #[cfg(feature = "io-uring")]
    #[test]
    fn uring_wal_roundtrips_and_checkpoints() {
        if !crate::io::io_uring_available() {
            eprintln!("skip: io_uring unavailable on this host");
            return;
        }
        let path = unique_path("uring-ckpt");
        let opts = || Options {
            io_backend: crate::io::IoBackend::Uring,
            ..Options::default()
        };
        {
            let map = BTreeMap::over(Store::<Wal>::create_path(&path, opts()).unwrap());
            block_on(async {
                for i in 0..50u32 {
                    let out = map
                        .apply(
                            WriteBatch::new()
                                .insert(format!("k{i:02}").into_bytes(), vec![i as u8; 400]),
                        )
                        .await
                        .unwrap();
                    let ApplyOutcome::Applied { version, .. } = out else {
                        panic!()
                    };
                    map.commit(version).await.unwrap();
                }
            });
            let report = block_on(map.store().checkpoint()).unwrap();
            assert_eq!(report.checkpoint_seq, 1);
            assert!(
                !segment_path(&path, 0).exists(),
                "segment 0 unlinked at cutover"
            );
            assert!(segment_path(&path, 1).exists(), "segment 1 is in force");
            block_on(async {
                let out = map
                    .apply(WriteBatch::new().insert(b"after".to_vec(), b"v".to_vec()))
                    .await
                    .unwrap();
                let ApplyOutcome::Applied { version, .. } = out else {
                    panic!()
                };
                map.commit(version).await.unwrap();
            });
            block_on(map.close()).unwrap();
        }
        let map = BTreeMap::over(Store::<Wal>::open_path(&path, opts()).unwrap());
        block_on(async {
            assert_eq!(
                map.get(b"k00".to_vec()).await.unwrap(),
                Some(vec![0u8; 400])
            );
            assert_eq!(
                map.get(b"k49".to_vec()).await.unwrap(),
                Some(vec![49u8; 400])
            );
            assert_eq!(
                map.get(b"after".to_vec()).await.unwrap(),
                Some(b"v".to_vec())
            );
        });
        block_on(map.close()).unwrap();
        cleanup(&path);
    }

    /// M7-0: retention across a rebind on the ring. A snapshot taken before a
    /// checkpoint keeps reading the retired (unlinked) segment through the old
    /// generation's `CompioIo` handle (POSIX keeps the inode alive for the open fd —
    /// invariant 8), and the segment's bytes free when the snapshot drops. Exercises
    /// the Phase-E retention machinery on io_uring devices, not just Syscall.
    #[cfg(feature = "io-uring")]
    #[test]
    fn uring_snapshot_stays_readable_across_a_checkpoint() {
        if !crate::io::io_uring_available() {
            eprintln!("skip: io_uring unavailable on this host");
            return;
        }
        let path = unique_path("uring-retain");
        let opts = Options {
            io_backend: crate::io::IoBackend::Uring,
            ..Options::default()
        };
        let map = BTreeMap::over(Store::<Wal>::create_path(&path, opts).unwrap());
        block_on(async {
            for i in 0..40u32 {
                let out = map
                    .apply(
                        WriteBatch::new().insert(format!("k{i:02}").into_bytes(), b"v1".to_vec()),
                    )
                    .await
                    .unwrap();
                let ApplyOutcome::Applied { version, .. } = out else {
                    panic!()
                };
                map.commit(version).await.unwrap();
            }
        });
        let snap = map.snapshot();
        // Checkpoint rotates segment 0 → 1 and unlinks segment 0; the snapshot's own
        // handle keeps the old inode alive.
        block_on(map.store().checkpoint()).unwrap();
        assert!(
            !segment_path(&path, 0).exists(),
            "segment 0 unlinked at cutover"
        );
        assert_eq!(
            map.store().metrics().retained_generations,
            1,
            "snapshot pins the retired gen"
        );
        block_on(async {
            for i in 0..40u32 {
                assert_eq!(
                    snap.get(format!("k{i:02}").into_bytes()).await.unwrap(),
                    Some(b"v1".to_vec()),
                    "snapshot must read the retired segment across the checkpoint (ring)"
                );
            }
        });
        drop(snap);
        // The retained generation releases once the snapshot is gone.
        assert_eq!(
            map.store().metrics().retained_generations,
            0,
            "retired gen frees on snapshot drop"
        );
        block_on(map.close()).unwrap();
        cleanup(&path);
    }

    /// M7-0: the WAL-segment mint site respects the factory — an
    /// `FsWalFileSet` carrying a `Uring` factory mints `Uring` segment devices at
    /// both create and open, so a rotated segment is never a silently-syscall one.
    #[cfg(feature = "io-uring")]
    #[test]
    fn wal_segments_are_minted_with_the_selected_backend() {
        if !crate::io::io_uring_available() {
            eprintln!("skip: io_uring unavailable on this host");
            return;
        }
        let path = unique_path("uring-seg");
        // Create the data path so the parent dir exists for segment fsync.
        std::fs::write(&path, b"").unwrap();
        let fs = FsWalFileSet {
            path: path.clone(),
            factory: DeviceFactory::new(crate::io::IoBackend::Uring, false),
        };
        let uuid = crate::direct::generate_store_uuid();
        let created = fs.create_segment(7, uuid).unwrap();
        assert_eq!(created.kind(), crate::io::IoKind::Uring);
        block_on(created.shutdown()).unwrap();
        let opened = fs.open_segment(7).unwrap();
        assert_eq!(opened.kind(), crate::io::IoKind::Uring);
        block_on(opened.shutdown()).unwrap();
        fs.unlink_segment(7).unwrap();
        cleanup(&path);
    }

    #[test]
    fn unsupported_checkpoint_refuses_without_namespace_mutation() {
        use crate::checkpoint::{CheckpointHeader, FORMAT_V3};
        use crate::page::PAGE_SIZE;
        use std::collections::BTreeMap as FileImages;

        for slot in 0..2 {
            for format in [FORMAT_V3 - 1, FORMAT_V3 + 1] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("store");
                let store = Store::<Wal>::create_path(&path, Options::default()).unwrap();
                block_on(store.close()).unwrap();
                let mut data = std::fs::read(&path).unwrap();
                let mut header = CheckpointHeader::decode(&data[..PAGE_SIZE]).unwrap();
                header.checkpoint_seq = 2;
                header.wal_segment_seq = 2;
                let mut image = header.encode();
                image[4] = format;
                let crc = crc32c::crc32c(&image[..124]);
                image[124..128].copy_from_slice(&crc.to_le_bytes());
                data[slot * PAGE_SIZE..(slot + 1) * PAGE_SIZE].copy_from_slice(&image);
                std::fs::write(&path, data).unwrap();
                std::fs::copy(segment_path(&path, 0), segment_path(&path, 2)).unwrap();
                std::fs::write(crate::compact::temp_path(&path), b"preserve pending image")
                    .unwrap();
                let snapshot = || -> FileImages<_, _> {
                    std::fs::read_dir(dir.path())
                        .unwrap()
                        .map(|entry| {
                            let entry = entry.unwrap();
                            (entry.file_name(), std::fs::read(entry.path()).unwrap())
                        })
                        .collect()
                };
                let before = snapshot();
                let result = Store::<Wal>::open_path(&path, Options::default());
                if let Ok(store) = &result {
                    block_on(store.close()).unwrap();
                }
                assert!(
                    matches!(result, Err(OpenError::UnsupportedFormat { found, .. }) if found == format)
                );
                assert_eq!(
                    snapshot(),
                    before,
                    "refusal must preserve every file and byte"
                );
            }
        }
    }

    #[test]
    fn open_missing_wal_is_not_found() {
        let path = unique_path("missing");
        let r = Store::<Wal>::open_path(&path, Options::default());
        assert!(
            matches!(r, Err(OpenError::Io(_))),
            "missing store → NotFound Io"
        );
        cleanup(&path);
    }

    /// M6 Phase D end-to-end on a real filesystem: a compaction replaces the
    /// data file with a dense one (smaller), rotates the real segment, leaves
    /// no temp, and the store reopens from the promoted header and continues.
    #[test]
    fn compaction_replaces_the_real_data_file_and_reopens() {
        let path = unique_path("compact");
        let tmp = crate::compact::temp_path(&path);
        {
            let map = BTreeMap::over(Store::<Wal>::create_path(&path, Options::default()).unwrap());
            block_on(async {
                for i in 0..40u32 {
                    let out = map
                        .apply(
                            WriteBatch::new()
                                .insert(format!("k{i:02}").into_bytes(), vec![i as u8; 300]),
                        )
                        .await
                        .unwrap();
                    let ApplyOutcome::Applied { version, .. } = out else {
                        panic!()
                    };
                    map.commit(version).await.unwrap();
                }
            });
            // Two checkpoints strand two relocated extents in the data file.
            block_on(map.store().checkpoint()).unwrap();
            block_on(map.store().checkpoint()).unwrap();
            let before = std::fs::metadata(&path).unwrap().len();

            let report = block_on(map.store().compact()).unwrap();
            let after = std::fs::metadata(&path).unwrap().len();
            assert_eq!(after, report.data_bytes_after);
            assert!(
                after < before,
                "two stranded extents must fold: {before} -> {after}"
            );
            assert!(!tmp.exists(), "the temp was renamed away");
            assert!(!segment_path(&path, 2).exists(), "retired segment unlinked");
            assert!(segment_path(&path, 3).exists(), "fresh segment in force");

            block_on(async {
                let out = map
                    .apply(WriteBatch::new().insert(b"after".to_vec(), b"v".to_vec()))
                    .await
                    .unwrap();
                let ApplyOutcome::Applied { version, .. } = out else {
                    panic!()
                };
                map.commit(version).await.unwrap();
            });
            block_on(map.close()).unwrap();
        }
        // Reopen resolves the segment and the dense layout from the header.
        let map = BTreeMap::over(Store::<Wal>::open_path(&path, Options::default()).unwrap());
        block_on(async {
            assert_eq!(
                map.get(b"k00".to_vec()).await.unwrap(),
                Some(vec![0u8; 300])
            );
            assert_eq!(
                map.get(b"k39".to_vec()).await.unwrap(),
                Some(vec![39u8; 300])
            );
            assert_eq!(
                map.get(b"after".to_vec()).await.unwrap(),
                Some(b"v".to_vec())
            );
        });
        block_on(map.close()).unwrap();
        cleanup(&path);
    }

    /// Every pre-existing file name and byte in `dir` (the lock sidecar included).
    fn namespace_images(dir: &Path) -> std::collections::BTreeMap<std::ffi::OsString, Vec<u8>> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (entry.file_name(), std::fs::read(entry.path()).unwrap())
            })
            .collect()
    }

    /// Asserts `dir` holds exactly the names in `before`, each byte-identical
    /// (names compared first so a failure reports them, not megabytes of bytes).
    fn assert_same_namespace(
        dir: &Path,
        before: &std::collections::BTreeMap<std::ffi::OsString, Vec<u8>>,
        case: &str,
    ) {
        let after = namespace_images(dir);
        assert_eq!(
            after.keys().collect::<Vec<_>>(),
            before.keys().collect::<Vec<_>>(),
            "{case}: refusal must preserve every pre-existing file name and create none"
        );
        for (name, bytes) in before {
            assert!(after[name] == *bytes, "{case}: {name:?} bytes changed");
        }
    }

    /// A closed fresh StoreWal plus the leftovers an interrupted rotation /
    /// compaction (or a newer release) may leave: two unreferenced segments and a
    /// compaction temp. Every one of them is destroyed by a *successful* open.
    fn store_with_leftovers(dir: &Path) -> PathBuf {
        let path = dir.join("store");
        let store = Store::<Wal>::create_path(&path, Options::default()).unwrap();
        block_on(store.close()).unwrap();
        std::fs::copy(segment_path(&path, 0), segment_path(&path, 1)).unwrap();
        std::fs::write(segment_path(&path, 7), b"future segment").unwrap();
        std::fs::write(crate::compact::temp_path(&path), b"preserve pending image").unwrap();
        path
    }

    /// Rewrites the in-force segment header (seq 0) through `edit`, resealing its
    /// CRC iff `reseal`.
    fn edit_segment_header(path: &Path, reseal: bool, edit: impl FnOnce(&mut [u8])) {
        let seg = segment_path(path, 0);
        let mut bytes = std::fs::read(&seg).unwrap();
        edit(&mut bytes[..crate::page::PAGE_SIZE]);
        if reseal {
            let crc = crc32c::crc32c(&bytes[..40]);
            bytes[40..44].copy_from_slice(&crc.to_le_bytes());
        }
        std::fs::write(&seg, bytes).unwrap();
    }

    /// fable72 F1: the WAL *segment-header* gate (and the rest of the segment
    /// identity check) refuses before `remove_stale_temp` and
    /// `retire_other_segments` touch the namespace. An authenticated older/newer
    /// format is `UnsupportedFormat`; an unsealed format byte is a torn header,
    /// `Corrupt` (CRC before format, as for the checkpoint header); a foreign
    /// store's segment is `UuidMismatch`. Every case leaves every pre-existing
    /// file name and byte unchanged.
    #[test]
    fn segment_header_refusal_preserves_every_file_and_byte() {
        type HeaderEdit = Box<dyn Fn(&mut [u8])>;
        let mut cases: Vec<(String, bool, HeaderEdit)> = Vec::new();
        for format in [0u8, 1, 3, 4, 255] {
            cases.push((
                format!("format {format}"),
                true,
                Box::new(move |h: &mut [u8]| h[4] = format),
            ));
        }
        cases.push((
            "torn format byte".into(),
            false,
            Box::new(|h: &mut [u8]| h[4] = 3),
        ));
        cases.push((
            "foreign uuid".into(),
            true,
            Box::new(|h: &mut [u8]| h[16] ^= 0xff),
        ));
        cases.push((
            "wrong segment seq".into(),
            true,
            Box::new(|h: &mut [u8]| h[32] = 9),
        ));
        // Collect every case's verdict so one run reports the whole matrix.
        let mut failures = Vec::new();
        for (name, reseal, edit) in cases {
            let dir = tempfile::tempdir().unwrap();
            let path = store_with_leftovers(dir.path());
            edit_segment_header(&path, reseal, edit);
            let before = namespace_images(dir.path());
            let result = Store::<Wal>::open_path(&path, Options::default());
            if let Ok(store) = &result {
                block_on(store.close()).unwrap();
            }
            let err = result
                .err()
                .unwrap_or_else(|| panic!("{name}: open must refuse"));
            let classified = match name.strip_prefix("format ") {
                Some(f) => {
                    let f: u8 = f.parse().unwrap();
                    matches!(err, OpenError::UnsupportedFormat {
                        component: "WAL segment header", found, supported: 2, newer
                    } if found == f && newer == (f > 2))
                }
                None if name == "foreign uuid" => matches!(err, OpenError::UuidMismatch),
                None => matches!(err, OpenError::Corrupt(_)),
            };
            if !classified {
                failures.push(format!("{name}: wrong error {err:?}"));
            }
            let caught = std::panic::catch_unwind(|| {
                assert_same_namespace(dir.path(), &before, &name);
            });
            if let Err(panic) = caught {
                let msg = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_else(|| "namespace changed".into());
                failures.push(msg);
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// fable72 F3: a refused open of a store that has no lock sidecar (copied
    /// without it, or pre-T1.5) must not create one. Covers both WAL open-time
    /// version gates: the checkpoint (data) header and the segment header.
    #[test]
    fn refusal_without_sidecar_does_not_create_one() {
        use crate::checkpoint::FORMAT_V3;
        use crate::page::PAGE_SIZE;
        for gate in ["data header", "WAL segment header"] {
            for slot in 0..2 {
                let dir = tempfile::tempdir().unwrap();
                let path = store_with_leftovers(dir.path());
                if gate == "data header" {
                    let mut data = std::fs::read(&path).unwrap();
                    let mut image = crate::checkpoint::CheckpointHeader::decode(&data[..PAGE_SIZE])
                        .unwrap()
                        .encode();
                    image[4] = FORMAT_V3 + 1;
                    let crc = crc32c::crc32c(&image[..124]);
                    image[124..128].copy_from_slice(&crc.to_le_bytes());
                    data[slot * PAGE_SIZE..(slot + 1) * PAGE_SIZE].copy_from_slice(&image);
                    std::fs::write(&path, data).unwrap();
                } else if slot == 0 {
                    edit_segment_header(&path, true, |h| h[4] = 3);
                } else {
                    continue;
                }
                std::fs::remove_file(lock_path(&path)).unwrap();
                let before = namespace_images(dir.path());
                let result = Store::<Wal>::open_path(&path, Options::default());
                if let Ok(store) = &result {
                    block_on(store.close()).unwrap();
                }
                assert!(
                    matches!(result, Err(OpenError::UnsupportedFormat { component, .. }) if component == gate),
                    "{gate} slot {slot}: {:?}",
                    result.err()
                );
                assert_same_namespace(dir.path(), &before, &format!("{gate} slot {slot}"));
            }
        }
    }

    /// The supported path keeps its cleanup: orphan segments and the compaction
    /// temp are removed, the store writes, and it reopens — with and without a
    /// pre-existing sidecar (a missing one is created and locked as before).
    #[test]
    fn supported_open_still_retires_leftovers() {
        for drop_sidecar in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = store_with_leftovers(dir.path());
            if drop_sidecar {
                std::fs::remove_file(lock_path(&path)).unwrap();
            }
            let map = BTreeMap::over(Store::<Wal>::open_path(&path, Options::default()).unwrap());
            assert!(lock_path(&path).exists());
            assert!(!crate::compact::temp_path(&path).exists());
            assert!(!segment_path(&path, 1).exists());
            assert!(!segment_path(&path, 7).exists());
            assert!(segment_path(&path, 0).exists());
            assert!(matches!(
                Store::<Wal>::open_path(&path, Options::default()),
                Err(OpenError::AlreadyOpen)
            ));
            block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))).unwrap();
            block_on(map.close()).unwrap();
            let map = BTreeMap::over(Store::<Wal>::open_path(&path, Options::default()).unwrap());
            assert_eq!(
                block_on(map.get(b"k".to_vec())).unwrap(),
                Some(b"v".to_vec())
            );
            block_on(map.close()).unwrap();
        }
    }
}
