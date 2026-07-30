//! Backend marker types, the `Backend` trait, and multi-handle file
//! generations.
//!
//! The two stores differ only in their durability protocol and physical layout;
//! the B+tree, cache, and read path are shared. The differences are
//! captured by:
//!
//! - marker types [`Direct`] and [`Wal`] that make the crash guarantee visible
//!   in Rust types (`Store<Direct>` vs `Store<Wal>`);
//! - a [`FileGeneration`] that owns *one or more* device handles, so a
//!   compaction/cutover swaps a whole `(root, files)` atom rather than a bare
//!   file (invariant 1);
//! - a [`PageLocator`] that maps a `PageRef` to `(member, offset, length)` so
//!   the read path never assumes `page_id * page_size`.

use crate::io::PageIo;
use crate::page::{PageId, PageRef, PAGE_SIZE};
use std::sync::Arc;

mod sealed {
    pub trait Sealed {}
}

/// The backend contract. Sealed: only [`Direct`] and [`Wal`] implement it.
pub trait Backend: sealed::Sealed + Send + Sync + 'static {
    /// A short name for diagnostics/metrics.
    const NAME: &'static str;
}

/// Clean-close persistence, no crash-recovery guarantee.
#[derive(Debug)]
pub struct Direct;

/// Checksummed write-ahead page-batch log with crash recovery.
///
/// **Crash-recovery scope (normative).** Recovery yielding the
/// last complete committed batch assumes **powersafe overwrite** in SQLite's
/// sense (`SQLITE_POWERSAFE_OVERWRITE`): a write of byte range `[a, b)` never
/// alters any byte outside `[a, b)`, even across a crash. This is a supported-
/// configuration claim — ordinary local **ext4/xfs on conventional block
/// devices**, where SQLite ships with PSOW enabled. CoW filesystems (btrfs, ZFS),
/// compressed/network filesystems, and exotic devices are **outside the v1
/// supported set** unless separately verified: without PSOW, a crash during the
/// post-recovery append (which begins in the physical unit holding the last
/// accepted record's tail) could corrupt that acknowledged record and lose a
/// committed batch, with no protocol bug. It is an environmental prerequisite,
/// like "fdatasync reaches stable storage."
#[derive(Debug)]
pub struct Wal;

impl sealed::Sealed for Direct {}
impl sealed::Sealed for Wal {}
impl Backend for Direct {
    const NAME: &'static str = "direct";
}
impl Backend for Wal {
    const NAME: &'static str = "wal";
}

/// Which member of a [`FileGeneration`] a page lives in.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Member {
    /// The data file (base header + Direct pages, or WAL-checkpointed pages).
    Data,
    /// The append-only WAL file (StoreWal only).
    Wal,
}

/// A physical location: which member, byte offset, and length.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Location {
    pub member: Member,
    pub offset: u64,
    pub length: u32,
}

/// A set of owned device handles that move together across a cutover.
///
/// Direct holds just [`Member::Data`]; Wal holds `Data` and `Wal`. Every
/// submitted operation clones the selected member's `Arc<dyn PageIo>` from the
/// generation and retains it until terminal completion (invariant 8), so a
/// generation swap cannot pull a device out from under an in-flight op.
pub struct FileGeneration {
    data: Arc<dyn PageIo>,
    wal: Option<Arc<dyn PageIo>>,
    /// Maps a `PageRef` to a physical [`Location`] within *this* generation.
    /// Lives inside the generation (not the cache) because the same immutable
    /// page id can sit at different offsets in different generations after a
    /// cutover; the cache is keyed by id and is generation-independent.
    locator: Arc<dyn PageLocator>,
}

impl FileGeneration {
    /// A Direct generation with a single data device and its locator.
    pub fn direct(data: Arc<dyn PageIo>, locator: Arc<dyn PageLocator>) -> Self {
        FileGeneration {
            data,
            wal: None,
            locator,
        }
    }

    /// A Wal generation with paired data + WAL devices and a locator.
    pub fn wal(data: Arc<dyn PageIo>, wal: Arc<dyn PageIo>, locator: Arc<dyn PageLocator>) -> Self {
        FileGeneration {
            data,
            wal: Some(wal),
            locator,
        }
    }

    /// This generation's page locator.
    pub fn locator(&self) -> &Arc<dyn PageLocator> {
        &self.locator
    }

    /// Clones the device handle for `member`. The clone is retained by the
    /// caller until its operation's terminal completion.
    pub fn device(&self, member: Member) -> Arc<dyn PageIo> {
        match member {
            Member::Data => Arc::clone(&self.data),
            Member::Wal => Arc::clone(
                self.wal
                    .as_ref()
                    .expect("WAL member requested on a Direct generation"),
            ),
        }
    }

    pub fn has_wal(&self) -> bool {
        self.wal.is_some()
    }
}

/// Maps a `PageRef` to a physical [`Location`] within a generation.
///
/// - Direct: pages are append-only in the data file; the production locator is
///   arithmetic over the header's id delta (`direct.rs`), not a table.
/// - Wal: a page lives either in a WAL record (the runtime overlay index) or,
///   after a checkpoint relocated it, in the data file — the locator consults
///   the overlay first and falls back to the checkpointed extent (`wal.rs`).
///
/// The mapping is opaque by design: no consumer may assume
/// `page_id * page_size`, contiguity, or single-member residency — offsets and
/// members come only from `locate` (this is what makes checkpoint/compaction
/// rebinds invisible to readers).
pub trait PageLocator: Send + Sync {
    /// Resolves where to read the image for `page_ref`. Returns `None` if the
    /// page is unknown to this generation (a corruption or a stale ref).
    fn locate(&self, page_ref: PageRef) -> Option<Location>;

    /// EXPERIMENT (T2.2): id-only speculative lookup for
    /// overflow-chain prefetch. Chain link ids are consecutive by construction
    /// (`btree.rs::emit_overflow` reserves the range head-first; Direct
    /// compaction's `rebuild_chain` does the same; WAL checkpoints relocate
    /// offsets, never ids), so a reader holding a chain head can *predict* the
    /// ids of the links it has not decoded yet. This method is the capability
    /// declaration: a locator that can resolve a bare id returns the location a
    /// full-ref [`locate`](Self::locate) would; a locator that cannot promise
    /// that returns `None` (the default) and prefetch never engages.
    ///
    /// The result is **advisory, never trusted**: before a speculative image is
    /// consumed, the reader re-resolves the authoritative `PageRef` through
    /// `locate` and requires exact `Location` equality *and* full-ref identity
    /// of the decoded page; any mismatch discards the speculation and falls
    /// back to demand reads. Correctness never depends on this method.
    fn predict_overflow_location(&self, page_id: PageId) -> Option<Location> {
        let _ = page_id;
        None
    }
}

/// A simple append-only data-file locator: `offset = table[page_id]`.
///
/// **Test scaffolding only** — no production generation uses it (Direct's real
/// locator is arithmetic in `direct.rs`; WAL's is the overlay index in
/// `wal.rs`). Kept for fixtures that need an arbitrary id→offset mapping; the
/// map shape also documents that no `page_id * page_size` assumption may leak.
pub struct DataFileLocator {
    offsets: parking_lot::RwLock<std::collections::HashMap<u64, u64>>,
}

impl DataFileLocator {
    pub fn new() -> Self {
        DataFileLocator {
            offsets: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Records that `page_id` lives at `offset` in the data file.
    pub fn record(&self, page_id: u64, offset: u64) {
        self.offsets.write().insert(page_id, offset);
    }
}

impl Default for DataFileLocator {
    fn default() -> Self {
        Self::new()
    }
}

impl PageLocator for DataFileLocator {
    fn locate(&self, page_ref: PageRef) -> Option<Location> {
        let offset = *self.offsets.read().get(&page_ref.page_id.as_u64())?;
        Some(Location {
            member: Member::Data,
            offset,
            length: PAGE_SIZE as u32,
        })
    }
}
