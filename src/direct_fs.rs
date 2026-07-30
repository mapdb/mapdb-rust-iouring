//! The durable real-file create/open protocol for StoreDirect.
//!
//! Creation is a durability protocol: obtain an exclusive **stable sidecar
//! lock** (`<path>.lock`, `flock` — a file separate from the data inode so it
//! survives a future data-file rename/compaction), create the data file, let
//! the writer lay down the header + slot B and `fdatasync`, then `fsync` the
//! parent directory so the new file's directory entry is durable. A second open
//! while the lock is held fails fast with [`OpenError::AlreadyOpen`]. Open of an
//! existing file takes the same lock but skips the directory fsync and validates
//! the header (any defect → `Corrupt`, never heuristic repair).
//!
//! The lock's `File` is owned by the **writer thread** (`Core::namespace_lock`,
//! injected at spawn — T1.2) and dropped as the writer's final act: `close()`
//! joins the writer, so a cleanly-closed store's path is free the moment close
//! returns, even while a closed handle lingers; if `close()` is never
//! called, dropping the last handle closes the command channel, the writer
//! drains every admitted command, flushes, exits, and only *then* releases the
//! lock — the path is never exposed to a second opener mid-drain. Compaction-temp naming and active-path-wins cleanup bind to
//! this same seam (M6 Phase D, [`crate::compact`]): the temp is
//! `<path>.compact.tmp`, and both constructors remove a stale one right after
//! taking the lock — whatever it contains, it is by construction uncommitted.

use crate::backend::Direct;
use crate::direct::generate_store_uuid;
use crate::error::OpenError;
use crate::io::{DeviceFactory, FileRole, PageIo};
use crate::page::PAGE_SIZE;
use crate::store::{Options, Store};
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::fs_util::{acquire_lock, fsync_parent_dir, CreateClaim};

impl Store<Direct> {
    /// Creates a fresh StoreDirect at `path` under the durable creation protocol
    /// (atomic sidecar claim, synchronous header + `fdatasync`, directory
    /// fsync). **Fail-safe (T1.5):** refuses with
    /// [`OpenError::AlreadyExists`] if any store member already exists at the
    /// path — pre-existing bytes are never touched — and with
    /// [`OpenError::AlreadyOpen`] if the sidecar lock is held. Replacing an
    /// existing store is [`create_path_overwrite`](Self::create_path_overwrite).
    ///
    /// Note: a `Snapshot` that outlives a `close()` may begin failing
    /// with `Io`/`Corrupt` (bounded, never wrong data) if the path is
    /// concurrently re-created by another opener after the lock releases.
    pub fn create_path(
        path: impl AsRef<Path>,
        options: Options,
    ) -> Result<Store<Direct>, OpenError> {
        let path = path.as_ref();
        let claim = CreateClaim::claim(path)?;
        // Classify the namespace UNDER the held lock: any real member refuses.
        let collision = if path.exists() {
            Some("data file")
        } else if crate::compact::temp_path(path).exists() {
            Some("compaction temp")
        } else {
            None
        };
        if let Some(member) = collision {
            claim.abandon(); // removes only a sidecar this attempt minted
            return Err(OpenError::AlreadyExists(member));
        }
        // Claim the data inode itself with `create_new`: a squatter racing in
        // without the lock protocol still cannot be silently destroyed.
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
        let mut claim = claim;
        let result = Self::create_over_claimed(path, options, claim.take_lock());
        match result {
            Ok(store) => Ok(store),
            Err(e) => {
                // Roll back exactly what this attempt created: the data inode
                // and (via the claim) a sidecar this attempt minted.
                let _ = std::fs::remove_file(path);
                claim.abandon();
                Err(e)
            }
        }
    }

    /// **Destructively** creates a fresh StoreDirect at `path`, replacing
    /// whatever store or partial file set exists there (the pre-T1.5
    /// `create_path` behavior: truncate the data file, remove a stale
    /// compaction temp). Fails with [`OpenError::AlreadyOpen`] if the sidecar
    /// lock is held. **Warning:** a path mix-up silently destroys a
    /// store — reach for [`create_path`](Self::create_path) unless replacement
    /// is the explicit intent.
    pub fn create_path_overwrite(
        path: impl AsRef<Path>,
        options: Options,
    ) -> Result<Store<Direct>, OpenError> {
        let path = path.as_ref();
        let lock = acquire_lock(path)?;
        // Active-path-wins: a compaction temp a crash left behind is uncommitted
        // by construction — remove it before laying down the fresh store. These
        // destructive steps live ONLY here (T1.5); the default create refuses.
        crate::compact::remove_stale_temp(path)?;
        // Fresh data file: truncate any stale bytes.
        OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        Self::create_over_claimed(path, options, lock)
    }

    /// The shared durable-create tail over an already-locked, already-created
    /// (empty) data inode: header + fdatasync + directory fsync, then open.
    /// `lock` is the held sidecar flock, handed to the store (writer-thread
    /// ownership, T1.2).
    fn create_over_claimed(
        path: &Path,
        options: Options,
        lock: File,
    ) -> Result<Store<Direct>, OpenError> {
        let factory = DeviceFactory::new(options.io_backend, options.direct_io);
        #[cfg(feature = "bench-instrument")]
        let factory = factory.with_wrapper(options.device_wrapper.clone());
        let dev: Arc<dyn PageIo> = factory.open(path, FileRole::Data)?;
        // Durable create protocol: both header slots are written and
        // **fdatasync'd** here, before
        // this call returns — otherwise a crash right after a "successful"
        // create leaves an empty/invalid header that `open_path` rejects, i.e.
        // an acknowledged create that is not durable.
        crate::direct::write_base_header_blocking(&dev, generate_store_uuid())?;
        // The new file's header and directory entry are now durable.
        fsync_parent_dir(path)?;
        let host = Arc::new(crate::compact::FsDataHost {
            path: path.to_path_buf(),
            factory,
        });
        Store::<Direct>::open_over_hosted(dev, Some(host), options, Some(lock))
    }

    /// Opens an existing StoreDirect at `path` under the sidecar lock, validating
    /// the header. The lock is held until the store's writer thread exits.
    pub fn open_path(path: impl AsRef<Path>, options: Options) -> Result<Store<Direct>, OpenError> {
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
        let lock = acquire_lock(path)?;
        // Active-path-wins: remove an orphaned compaction temp (M6 Phase D).
        crate::compact::remove_stale_temp(path)?;
        // Open the *existing* file only (never create): a file deleted between
        // the `exists()` check and here is reported as `NotFound`, not
        // re-created as a junk store.
        let dev = factory.open_existing(path, FileRole::Data)?;
        // Reject an impossibly short file up front (a real header needs page 0).
        let len = futures_lite::future::block_on(dev.size())?;
        if len < 2 * PAGE_SIZE as u64 {
            return Err(OpenError::Corrupt(
                "store data file shorter than its two header pages",
            ));
        }
        // `open_over` reads + validates the header and rebuilds the allocator
        // frontier; the writer thread holds the lock until it exits (T1.2).
        let host = Arc::new(crate::compact::FsDataHost {
            path: path.to_path_buf(),
            factory,
        });
        Store::<Direct>::open_over_hosted(dev, Some(host), options, Some(lock))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::ApplyOutcome;
    use crate::fs_util::lock_path;
    use crate::map::BTreeMap;
    use crate::WriteBatch;
    use futures_lite::future::block_on;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_path(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "mapdb-uring-m3-{}-{}-{}.data",
            tag,
            std::process::id(),
            n
        ));
        // Best-effort clean of any leftovers.
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(lock_path(&p));
        p
    }

    /// A fixture path under the repo's `target/` (real disk — btrfs on this host and
    /// CI), so `direct_io` (O_DIRECT) fixtures are on a filesystem that can honor it,
    /// never `/tmp` (tmpfs, where `open(O_DIRECT)` fails). The O_DIRECT tests still
    /// probe-and-skip if even this fs refuses O_DIRECT (M7-A).
    fn real_store_path(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push("target");
        dir.push("m7a-direct-tests");
        let _ = std::fs::create_dir_all(&dir);
        dir.push(format!("{}-{}-{}.data", tag, std::process::id(), n));
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(lock_path(&dir));
        dir
    }

    fn direct_opts() -> Options {
        Options {
            direct_io: true,
            ..Options::default()
        }
    }

    /// Whether O_DIRECT can be opened under `target/` on this host — probes by
    /// attempting a create with `direct_io`, so the O_DIRECT store tests skip cleanly
    /// on a filesystem/kernel that does not honor it.
    fn odirect_store_available() -> bool {
        let p = real_store_path("probe");
        let ok = Store::<Direct>::create_path(&p, direct_opts()).is_ok();
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(lock_path(&p));
        ok
    }

    #[test]
    fn direct_io_data_page_path_reopens_across_modes() {
        // M7-A: `direct_io` is orthogonal, so a store may be created
        // in one mode and reopened in the other. StoreDirect's data file is pure
        // page-granular I/O (no manifest — that is a StoreWal checkpoint artifact), so
        // this drives the O_DIRECT page read/write path end-to-end, including a
        // compaction's dense rewrite through the O_DIRECT fd. Verifies both directions:
        // created-direct → reopened-{direct,buffered}, and created-buffered →
        // reopened-direct.
        if !odirect_store_available() {
            eprintln!("skip: O_DIRECT unavailable under target/ on this host");
            return;
        }
        let n = 60u32;
        let fill = |map: &BTreeMap<Direct>| {
            block_on(async {
                for i in 0..n {
                    map.apply(
                        WriteBatch::new()
                            .insert(format!("k{i:04}").into_bytes(), vec![i as u8; 300]),
                    )
                    .await
                    .unwrap();
                }
                // Compaction densely rewrites the live pages through the O_DIRECT fd
                // (StoreDirect has no manifest) and promotes the result.
                map.store().compact().await.unwrap();
            });
        };
        let verify = |map: &BTreeMap<Direct>| {
            block_on(async {
                assert_eq!(map.len() as u32, n);
                assert_eq!(
                    map.get(b"k0021".to_vec()).await.unwrap(),
                    Some(vec![21u8; 300])
                );
                assert_eq!(
                    map.get(b"k0059".to_vec()).await.unwrap(),
                    Some(vec![59u8; 300])
                );
            });
        };

        // Flow A — created with O_DIRECT, reopened both ways.
        let path_a = real_store_path("created-direct");
        {
            let map = BTreeMap::over(Store::<Direct>::create_path(&path_a, direct_opts()).unwrap());
            fill(&map);
            block_on(map.close()).unwrap();
        }
        // reopen O_DIRECT: reads the padded manifest back through the O_DIRECT fd.
        let map = BTreeMap::over(Store::<Direct>::open_path(&path_a, direct_opts()).unwrap());
        verify(&map);
        block_on(map.close()).unwrap();
        // reopen buffered: the same on-disk file opens without O_DIRECT.
        let map = BTreeMap::over(Store::<Direct>::open_path(&path_a, Options::default()).unwrap());
        verify(&map);
        block_on(map.close()).unwrap();
        let _ = std::fs::remove_file(&path_a);
        let _ = std::fs::remove_file(lock_path(&path_a));

        // Flow B — created buffered, reopened with O_DIRECT (padded manifest is
        // written even in buffered mode, so the O_DIRECT read succeeds).
        let path_b = real_store_path("created-buffered");
        {
            let map =
                BTreeMap::over(Store::<Direct>::create_path(&path_b, Options::default()).unwrap());
            fill(&map);
            block_on(map.close()).unwrap();
        }
        let map = BTreeMap::over(Store::<Direct>::open_path(&path_b, direct_opts()).unwrap());
        verify(&map);
        block_on(map.close()).unwrap();
        let _ = std::fs::remove_file(&path_b);
        let _ = std::fs::remove_file(lock_path(&path_b));
    }

    #[test]
    fn create_write_close_reopen_real_file() {
        let path = unique_path("reopen");
        {
            let store = Store::<Direct>::create_path(&path, Options::default()).unwrap();
            let map = BTreeMap::over(store);
            block_on(async {
                for i in 0..50u32 {
                    map.apply(
                        WriteBatch::new()
                            .insert(format!("k{i:04}").into_bytes(), vec![i as u8; 200]),
                    )
                    .await
                    .unwrap();
                }
                map.apply(WriteBatch::new().remove(b"k0003".to_vec()))
                    .await
                    .unwrap();
            });
            block_on(map.close()).unwrap();
        }
        // Reopen the real file: the highest visible pre-close state survives.
        let store = Store::<Direct>::open_path(&path, Options::default()).unwrap();
        let map = BTreeMap::over(store);
        block_on(async {
            assert_eq!(map.len(), 49);
            assert_eq!(map.get(b"k0003".to_vec()).await.unwrap(), None);
            assert_eq!(
                map.get(b"k0021".to_vec()).await.unwrap(),
                Some(vec![21u8; 200])
            );
            let out = map
                .apply(WriteBatch::new().insert(b"new".to_vec(), b"v".to_vec()))
                .await
                .unwrap();
            assert!(matches!(out, ApplyOutcome::Applied { .. }));
        });
        block_on(map.close()).unwrap();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(lock_path(&path));
    }

    #[test]
    fn exclusive_open_is_rejected_then_released_on_close() {
        let path = unique_path("excl");
        let map = BTreeMap::over(Store::<Direct>::create_path(&path, Options::default()).unwrap());
        // A second open while the first holds the sidecar lock fails fast.
        let second = Store::<Direct>::open_path(&path, Options::default());
        assert!(
            matches!(second, Err(OpenError::AlreadyOpen)),
            "second open must be rejected"
        );

        // Close releases the sidecar lock even though the closed handle `map` is
        // still in scope: a reopen now succeeds.
        block_on(map.close()).unwrap();
        let reopened = Store::<Direct>::open_path(&path, Options::default());
        assert!(
            reopened.is_ok(),
            "open after close (handle still alive) should succeed"
        );
        block_on(BTreeMap::over(reopened.unwrap()).close()).unwrap();
        drop(map);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(lock_path(&path));
    }

    /// M6 Phase D end-to-end on a real filesystem: compaction shrinks the file
    /// at `path`, leaves no temp behind, and the store reopens (v2 delta header)
    /// and keeps writing. A stale temp planted before open is removed
    /// (active-path-wins cleanup).
    #[test]
    fn compaction_shrinks_the_real_file_and_reopens() {
        let path = unique_path("compact");
        let tmp = crate::compact::temp_path(&path);
        {
            let map =
                BTreeMap::over(Store::<Direct>::create_path(&path, Options::default()).unwrap());
            block_on(async {
                for round in 0..3u8 {
                    for i in 0..30u32 {
                        map.apply(
                            WriteBatch::new().insert(
                                format!("k{i:03}").into_bytes(),
                                vec![round ^ i as u8; 300],
                            ),
                        )
                        .await
                        .unwrap();
                    }
                }
            });
            let before = std::fs::metadata(&path).unwrap().len();
            let report = block_on(map.store().compact()).unwrap();
            let after = std::fs::metadata(&path).unwrap().len();
            assert_eq!(after, report.data_bytes_after);
            assert!(
                after < before,
                "churned real file must shrink: {before} -> {after}"
            );
            assert!(!tmp.exists(), "the temp was renamed away");
            block_on(async {
                assert_eq!(
                    map.get(b"k012".to_vec()).await.unwrap(),
                    Some(vec![2 ^ 12u8; 300])
                );
            });
            block_on(map.close()).unwrap();
        }
        // Plant a junk temp: open must remove it and succeed.
        std::fs::write(&tmp, b"junk from a crashed compaction").unwrap();
        let map = BTreeMap::over(Store::<Direct>::open_path(&path, Options::default()).unwrap());
        assert!(!tmp.exists(), "stale temp removed at open");
        block_on(async {
            assert_eq!(
                map.get(b"k000".to_vec()).await.unwrap(),
                Some(vec![2u8; 300])
            );
            let out = map
                .apply(WriteBatch::new().insert(b"new".to_vec(), b"v".to_vec()))
                .await
                .unwrap();
            assert!(matches!(out, ApplyOutcome::Applied { .. }));
        });
        block_on(map.close()).unwrap();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(lock_path(&path));
    }

    /// M7-0: a StoreDirect created on the `Uring` backend runs the whole
    /// production path — write, compact (mints the temp through the factory),
    /// reopen on the ring, read back. If any device site mis-minted or the ring
    /// integration broke, compaction or reopen would fail.
    #[cfg(feature = "io-uring")]
    #[test]
    fn uring_direct_roundtrips_and_compacts() {
        if !crate::io::io_uring_available() {
            eprintln!("skip: io_uring unavailable on this host");
            return;
        }
        let path = unique_path("uring-direct");
        let opts = || Options {
            io_backend: crate::io::IoBackend::Uring,
            ..Options::default()
        };
        {
            let map = BTreeMap::over(Store::<Direct>::create_path(&path, opts()).unwrap());
            block_on(async {
                for round in 0..3u8 {
                    for i in 0..40u32 {
                        map.apply(
                            WriteBatch::new().insert(
                                format!("k{i:03}").into_bytes(),
                                vec![round ^ i as u8; 250],
                            ),
                        )
                        .await
                        .unwrap();
                    }
                }
                let before = std::fs::metadata(&path).unwrap().len();
                map.store().compact().await.unwrap();
                let after = std::fs::metadata(&path).unwrap().len();
                assert!(
                    after < before,
                    "churned file must shrink: {before} -> {after}"
                );
                assert_eq!(
                    map.get(b"k012".to_vec()).await.unwrap(),
                    Some(vec![2 ^ 12u8; 250])
                );
            });
            block_on(map.close()).unwrap();
        }
        let map = BTreeMap::over(Store::<Direct>::open_path(&path, opts()).unwrap());
        block_on(async {
            assert_eq!(
                map.get(b"k000".to_vec()).await.unwrap(),
                Some(vec![2u8; 250])
            );
            let out = map
                .apply(WriteBatch::new().insert(b"new".to_vec(), b"v".to_vec()))
                .await
                .unwrap();
            assert!(matches!(out, ApplyOutcome::Applied { .. }));
        });
        block_on(map.close()).unwrap();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(lock_path(&path));
    }

    /// M7-0: the compaction-temp mint site respects the factory — a
    /// `FsDataHost` carrying a `Uring` factory mints a `Uring` temp device, not a
    /// silently-syscall one. Pins the "no measurement mislabeling" property at the
    /// exact site.
    #[cfg(feature = "io-uring")]
    #[test]
    fn compaction_temp_is_minted_with_the_selected_backend() {
        use crate::compact::DataFileHost;
        if !crate::io::io_uring_available() {
            eprintln!("skip: io_uring unavailable on this host");
            return;
        }
        let path = unique_path("uring-temp");
        let host = crate::compact::FsDataHost {
            path: path.clone(),
            factory: crate::io::DeviceFactory::new(crate::io::IoBackend::Uring, false),
        };
        let dev = host.create_temp().unwrap();
        assert_eq!(dev.kind(), crate::io::IoKind::Uring);
        block_on(dev.shutdown()).unwrap();
        host.remove_temp();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(crate::compact::temp_path(&path));
    }
}
