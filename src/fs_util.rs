//! Shared real-filesystem durability helpers (M6 Phase H consolidation).
//!
//! Linux-first. The one durability primitive the create/rotate/compact protocols
//! all need is a **parent-directory fsync**: on Linux a file's data being
//! `fdatasync`'d does not make its *directory entry* durable, so a freshly
//! created, renamed, or unlinked name can vanish (or reappear) across a crash
//! until the containing directory is itself fsynced. `direct_fs`, `wal_fs`, and
//! `compact` previously each carried a verbatim copy of this; there is no
//! durability reason to duplicate it, and one home keeps the Linux-first
//! assumption and its portability caveats in a single documented place.

use crate::error::OpenError;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

/// `fsync` the directory containing `path`, so a just-created / renamed /
/// unlinked entry for `path` is durable across a crash.
///
/// **Same-parent assumption.** Every namespace mutation in this store is within
/// one directory — the data file `<path>`, its WAL segments `<path>.wal.<seq>`,
/// and the compaction temp `<path>.compact.tmp` all share `<path>`'s parent, and
/// the compaction promote is a *same-directory* `rename(temp, path)`. So syncing
/// the single parent is sufficient; a cross-directory rename (which would need
/// *both* parents synced) does not occur, and if one is ever introduced this
/// helper must be revisited.
///
/// The error carries the directory path so a caller propagating it (into
/// `OpenError::Io`, or a poison) records *which* directory's fsync failed, not a
/// bare "sync_all failed". Callers must treat a failure as a durability failure —
/// never as a completed commit/rotation.
pub(crate) fn fsync_parent_dir(path: &Path) -> io::Result<()> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| io::Error::new(e.kind(), format!("fsync parent dir {}: {e}", dir.display())))
}

/// The sidecar lock path for a store's data path: `<path>.lock`.
pub(crate) fn lock_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".lock");
    PathBuf::from(s)
}

/// Takes the exclusive non-blocking `flock` on an open sidecar, or
/// [`OpenError::AlreadyOpen`] if another handle holds it.
fn flock_exclusive_nb(lock: &File) -> Result<(), OpenError> {
    let ret = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if ret != 0 {
        let err = io::Error::last_os_error();
        // EWOULDBLOCK == EAGAIN on Linux: the lock is held by another handle.
        if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            return Err(OpenError::AlreadyOpen);
        }
        return Err(OpenError::from(err));
    }
    Ok(())
}

/// Reads the first `len` bytes of an existing `path` through a read-only,
/// non-creating handle, **without** the sidecar lock; `None` on any failure
/// (missing, short, unreadable). Only for the pre-lock refusal probe
/// ([`sidecar_missing`]), whose sole power is to refuse: whatever it reads is
/// re-read and re-validated under the lock before anything is trusted.
pub(crate) fn read_prefix_unlocked(path: &Path, len: usize) -> Option<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let file = OpenOptions::new().read(true).open(path).ok()?;
    let mut buf = vec![0u8; len];
    file.read_exact_at(&mut buf, 0).ok()?;
    Some(buf)
}

/// `true` when the store at `path` has no lock sidecar (copied without it, or
/// created before T1.5). Opening such a store must create the sidecar to lock
/// it, so `open_path` first probes the open-time format gates read-only and
/// refuses an unsupported format *without* minting the sidecar. The probe only
/// ever refuses; it grants nothing, so it cannot weaken the lock: a probe that
/// passes (or is inconclusive) falls through to the normal locked open, which
/// re-validates every gate under the lock. A sidecar that appears
/// concurrently changes nothing — the probe mutates nothing either way.
pub(crate) fn sidecar_missing(path: &Path) -> bool {
    matches!(
        std::fs::symlink_metadata(lock_path(path)),
        Err(e) if e.kind() == io::ErrorKind::NotFound
    )
}

/// Maps a pre-lock probe verdict: an authenticated format refusal propagates;
/// every other outcome (a torn slot, an I/O error, a concurrent writer's
/// in-flight state) is inconclusive and left to the locked open to judge.
pub(crate) fn only_format_refusal<T>(r: Result<T, OpenError>) -> Result<Option<T>, OpenError> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e @ OpenError::UnsupportedFormat { .. }) => Err(e),
        Err(_) => Ok(None),
    }
}

/// Acquires the exclusive sidecar lock for **open** (creating the sidecar if
/// absent — an open of a pre-T1.5 store may find none), or
/// [`OpenError::AlreadyOpen`] if held. The returned `File` keeps the flock
/// until it drops (owned by the store's writer thread, T1.2).
pub(crate) fn acquire_lock(path: &Path) -> Result<File, OpenError> {
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(lock_path(path))?;
    flock_exclusive_nb(&lock)?;
    Ok(lock)
}

/// An atomically-claimed create right (T1.5): the held sidecar flock
/// plus the memory of whether *this attempt* created the sidecar inode — the
/// only namespace member a failed fail-safe create is allowed to remove.
pub(crate) struct CreateClaim {
    lock: Option<File>,
    created_sidecar: bool,
    lock_file_path: PathBuf,
}

impl CreateClaim {
    /// Atomically create-or-classify the sidecar and take its flock.
    ///
    /// `create_new` distinguishes "we minted the sidecar" from "it pre-existed"
    /// without a TOCTOU window (the sidecar itself is a store member, so any
    /// exists-check *before* locking would race). Either way
    /// the flock is then taken non-blocking: a raced flock on our own fresh
    /// inode, or a held lock on a pre-existing one, is `AlreadyOpen`. A
    /// pre-existing but *unheld* sidecar alone is claimable (a leftover from a
    /// crashed create or a deleted store); it is the caller's job to classify
    /// the rest of the namespace under the now-held lock and refuse
    /// `AlreadyExists` if any real member exists.
    pub(crate) fn claim(path: &Path) -> Result<CreateClaim, OpenError> {
        let lp = lock_path(path);
        let (lock, created_sidecar) = match OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&lp)
        {
            Ok(f) => (f, true),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                let f = OpenOptions::new().read(true).write(true).open(&lp)?;
                (f, false)
            }
            Err(e) => return Err(e.into()),
        };
        flock_exclusive_nb(&lock)?;
        Ok(CreateClaim {
            lock: Some(lock),
            created_sidecar,
            lock_file_path: lp,
        })
    }

    /// Hands the lock `File` to the store (writer-thread ownership, T1.2) on
    /// the success path. The claim itself stays alive so a construction failure
    /// *after* this can still [`abandon`](Self::abandon) the sidecar.
    pub(crate) fn take_lock(&mut self) -> File {
        self.lock.take().expect("lock already taken")
    }

    /// Rolls back a FAILED create attempt: removes the sidecar iff this attempt
    /// created it, releases the flock (if still held here). Pre-existing bytes
    /// are never touched — the caller separately removes only the members it
    /// itself created.
    pub(crate) fn abandon(mut self) {
        if self.created_sidecar {
            let _ = std::fs::remove_file(&self.lock_file_path);
        }
        drop(self.lock.take());
    }
}
