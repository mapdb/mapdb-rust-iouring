//! Data-file compaction namespace seam + report (M6 Phase D).
//!
//! [`PageIo`](crate::io::PageIo) is a byte device; replacing the data file is a
//! *namespace* capability — create a temp, rename it over the active path,
//! make the rename durable — exactly parallel to [`WalFileSet`]
//! (crate::fileset::WalFileSet) for segments. This trait is that capability. It
//! is held only by stores that own their data path (the path-backed
//! constructors, plus the in-memory [`MemDataHost`] the deterministic tests
//! drive); a store over a caller-supplied `Arc<dyn PageIo>` owns no namespace
//! and cannot compact ([`CheckpointError::Unsupported`]
//! (crate::error::CheckpointError)).
//!
//! **Fixed temp name** (`<path>.compact.tmp`), not unique names: the exclusive
//! sidecar `flock` — a sidecar precisely so it survives the data-inode rename —
//! guarantees at most one live opener per path, hence at most one compactor.
//! Uniqueness buys nothing under that lock; a fixed name makes orphan cleanup
//! one `remove_file`. **Active-path-wins**: whatever a temp contains, it is by
//! construction uncommitted — `create_path`/`open_path` remove it right after
//! taking the lock, and `create_temp` removes a stale one before minting.
//!
//! Implementations are blocking, like `WalFileSet`: they run on the writer
//! thread inside a compaction, which is rare and already pays several `fsync`s.
//!
//! The `.compact.tmp` suffix is **reserved** next to a store's data path: an
//! open removes `<path>.compact.tmp` unconditionally, so placing an unrelated
//! file (or another store) at that exact name loses it (the same class of
//! contract as `<path>.lock` and `<path>.wal.<seq>`).

use crate::io::{DeviceFactory, PageIo};
use std::fs::OpenOptions;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Owns the data file's namespace for compaction: temp creation, durable
/// promotion (rename + parent-dir fsync), and abort cleanup.
pub(crate) trait DataFileHost: Send + Sync + 'static {
    /// Removes any stale compaction temp and creates a fresh, empty one,
    /// returning its device. The handle is opened before the rename, so after
    /// [`promote_temp`](Self::promote_temp) it *is* the active file's device.
    fn create_temp(&self) -> io::Result<Arc<dyn PageIo>>;

    /// Promotes the temp to the active data path: `rename(temp, path)`, then
    /// `fsync` the parent directory so the swap is durable. The commit point of
    /// a compaction. The old inode leaves the namespace but stays alive for
    /// every open handle (retained generations keep reading it — invariant 8).
    fn promote_temp(&self) -> io::Result<()>;

    /// Best-effort removal of the temp after an aborted compaction.
    fn remove_temp(&self);
}

/// The compaction temp path for a data file: `<path>.compact.tmp`.
pub(crate) fn temp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".compact.tmp");
    PathBuf::from(s)
}

/// Removes a stale compaction temp at open (active-path-wins). Missing is fine.
pub(crate) fn remove_stale_temp(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(temp_path(path)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

use crate::fs_util::fsync_parent_dir;

/// The real-filesystem data-file host for the store at `path`. Every mutation
/// runs under the store's already-held sidecar `flock`. Carries the store's
/// [`DeviceFactory`] so a compaction temp is minted with the store's selected
/// backend (M7-0) — the compaction temp is promoted into the active data file, so
/// it must be the same backend the store reads through.
pub(crate) struct FsDataHost {
    pub path: PathBuf,
    pub factory: DeviceFactory,
}

impl DataFileHost for FsDataHost {
    fn create_temp(&self) -> io::Result<Arc<dyn PageIo>> {
        let tmp = temp_path(&self.path);
        remove_stale_temp(&self.path)?;
        // `create_new`: after the remove above, an existing file here means
        // something else owns this name — refuse rather than truncate it.
        OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&tmp)?;
        match self
            .factory
            .open_existing(&tmp, crate::io::FileRole::CompactTemp)
        {
            Ok(dev) => Ok(dev),
            Err(e) => {
                // Keep the clean-abort promise: never strand the temp we just
                // created.
                let _ = std::fs::remove_file(&tmp);
                Err(e)
            }
        }
    }

    fn promote_temp(&self) -> io::Result<()> {
        std::fs::rename(temp_path(&self.path), &self.path)?;
        fsync_parent_dir(&self.path)
    }

    fn remove_temp(&self) {
        let _ = std::fs::remove_file(temp_path(&self.path));
    }
}

/// An in-memory data-file host over [`FakeIo`](crate::io::FakeIo): the
/// deterministic-test counterpart of [`FsDataHost`], mirroring
/// [`MemWalFileSet`](crate::fileset::MemWalFileSet). Holds the *active* data
/// device and at most one temp; `promote_temp` swaps the temp into the active
/// slot, dropping this host's reference to the old device — exactly modelling
/// the rename removing the old inode from the namespace while retained
/// generations keep their handles.
pub struct MemDataHost {
    active: parking_lot::Mutex<Arc<crate::io::FakeIo>>,
    temp: parking_lot::Mutex<Option<Arc<crate::io::FakeIo>>>,
}

impl MemDataHost {
    pub fn new(active: Arc<crate::io::FakeIo>) -> Arc<Self> {
        Arc::new(MemDataHost {
            active: parking_lot::Mutex::new(active),
            temp: parking_lot::Mutex::new(None),
        })
    }

    /// The currently active data device (post-compaction, the promoted temp).
    /// Reopen-after-crash tests read the store back through this.
    pub fn active(&self) -> Arc<crate::io::FakeIo> {
        Arc::clone(&self.active.lock())
    }

    /// The pending temp, if a compaction is mid-flight (crash-image tests).
    pub fn temp(&self) -> Option<Arc<crate::io::FakeIo>> {
        self.temp.lock().clone()
    }
}

impl DataFileHost for MemDataHost {
    fn create_temp(&self) -> io::Result<Arc<dyn PageIo>> {
        let dev = Arc::new(crate::io::FakeIo::new(0));
        *self.temp.lock() = Some(Arc::clone(&dev));
        Ok(dev as Arc<dyn PageIo>)
    }

    fn promote_temp(&self) -> io::Result<()> {
        let Some(tmp) = self.temp.lock().take() else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no compaction temp to promote",
            ));
        };
        *self.active.lock() = tmp;
        Ok(())
    }

    fn remove_temp(&self) {
        *self.temp.lock() = None;
    }
}

/// What one compaction did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactReport {
    /// Live pages written into the dense file.
    pub live_pages: u64,
    /// Data-file length before (the superseded inode's bytes — they free when
    /// its retired generation releases, or the retention bound revokes it).
    pub data_bytes_before: u64,
    /// Data-file length after (headers + live pages, plus the manifest for WAL).
    pub data_bytes_after: u64,
}
