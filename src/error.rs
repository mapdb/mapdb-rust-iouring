//! Error and terminal-state types.
//!
//! Terminal store states (`Poisoned`, `Closed`) can surface on nearly every
//! path, so they are modelled once as [`Terminal`] and folded into each public
//! error via `From`. This keeps every wait loop's "check poison/closed before
//! listening and after registering" (cross-cutting invariant 6) uniform.

use std::sync::Arc;

/// A terminal store condition. Clonable and structured (not a bare string) so a
/// poisoned store replays the *original* failure to every waiter class
/// (invariant 6); the prototype's `ErrorKind::Other` flattening is rejected.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Terminal {
    /// An unrecoverable I/O or protocol failure poisoned the store. Every later
    /// operation fails with this same reason; it is never retried into success.
    Poisoned(PoisonReason),
    /// The store was cleanly closed. No further intake.
    Closed,
}

/// Why a store poisoned. Cheaply clonable (`Arc<io::Error>` keeps the original
/// `ErrorKind`).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum PoisonReason {
    /// A read/write/datasync at the I/O boundary failed fatally.
    Io(Arc<std::io::Error>),
    /// A decoded page/record failed structural or checksum validation on a path
    /// that cannot continue.
    Corrupt(&'static str),
    /// An internal invariant was violated (codec overage, budget underflow).
    /// Indicates a bug; surfaced rather than panicking the writer thread.
    Internal(&'static str),
}

impl Terminal {
    pub fn io(err: std::io::Error) -> Self {
        Terminal::Poisoned(PoisonReason::Io(Arc::new(err)))
    }
    pub fn corrupt(what: &'static str) -> Self {
        Terminal::Poisoned(PoisonReason::Corrupt(what))
    }
    pub fn internal(what: &'static str) -> Self {
        Terminal::Poisoned(PoisonReason::Internal(what))
    }
}

/// Failure to open or create a store.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum OpenError {
    /// The file/header/root failed validation and no recovery is attempted
    /// (StoreDirect) or no committed prefix validated (StoreWal base header).
    Corrupt(&'static str),
    /// A well-formed header (valid magic, sufficient length) whose format version
    /// this build does not support — either **older** than this build reads
    /// (media predating a format bump) or **newer** (written by a later release).
    ///
    /// Distinct from [`Self::Corrupt`], which is a damaged or unrecognizable
    /// image: this store is intact but outside this binary's supported version.
    /// It is **never** recovered, migrated, created-over, truncated, or replayed
    /// — clean refusal *before any mutation* is the whole forward-format policy
    /// (M6 Phase H). Migration tooling and read-only-open-of-newer are deliberate
    /// non-goals until there is a real compatible evolution to represent; a single
    /// format byte does not encode compatibility, so this build supports exactly
    /// one version per structure.
    UnsupportedFormat {
        /// The on-disk structure whose version was refused: `"direct header"`,
        /// `"data header"` (StoreWal checkpoint slot), or `"WAL segment header"`.
        component: &'static str,
        /// The format byte found on disk.
        found: u8,
        /// The single format version this build reads and writes.
        supported: u8,
        /// `true` when `found > supported` (a future release wrote it); `false`
        /// when `found < supported` (media predating a format bump).
        newer: bool,
    },
    /// Store UUID mismatch between paired data and WAL files.
    UuidMismatch,
    /// Another process holds the exclusive sidecar lock.
    AlreadyOpen,
    /// The fail-safe `create_path` found a pre-existing store member at the
    /// path while the sidecar lock was **not** held (T1.5): refused
    /// without touching a byte. The payload names the colliding member kind
    /// (`"data file"`, `"wal segment"`, `"compaction temp"`). Replacing an
    /// existing store is the explicitly destructive
    /// `create_path_overwrite`'s job, never the default's.
    AlreadyExists(&'static str),
    /// I/O error during open/create.
    Io(Arc<std::io::Error>),
}

impl OpenError {
    /// Classifies a header's on-disk `found` format byte against the single
    /// `supported` version this build implements. `Ok(())` on an exact match,
    /// else [`OpenError::UnsupportedFormat`] with the older/newer distinction
    /// resolved.
    ///
    /// This is the one classification point (M6 Phase H): the independently
    /// versioned open-time headers (direct, StoreWal data/checkpoint, WAL segment)
    /// each call it *after* their own magic + length + checksum checks. It
    /// deliberately does **not** unify their versions — they evolve separately, so
    /// it judges one at a time. Sub-item format bytes (WAL *record*, *page*) are
    /// not open-time gates: they sit behind an already-validated header that has
    /// vouched for the format, so a mismatch there is corruption, handled on the
    /// read/recovery path, not a version refusal here. (A future compatibility
    /// window would widen `supported` to a range at this seam; today it is single-
    /// version by construction.)
    pub(crate) fn check_open_format(
        component: &'static str,
        found: u8,
        supported: u8,
    ) -> Result<(), OpenError> {
        if found == supported {
            Ok(())
        } else {
            Err(OpenError::UnsupportedFormat {
                component,
                found,
                supported,
                newer: found > supported,
            })
        }
    }
}

impl From<std::io::Error> for OpenError {
    fn from(err: std::io::Error) -> Self {
        OpenError::Io(Arc::new(err))
    }
}

/// Failure of a read-path operation (`get`, `get_many`, `snapshot`-backed
/// reads, `len`, `verify`).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ReadError {
    /// A page failed structural or checksum validation.
    Corrupt(&'static str),
    /// A device read failed at the I/O boundary. Unlike [`Self::Corrupt`] this
    /// is not cached (a retry re-issues the load) and does not itself poison the
    /// store.
    Io(Arc<std::io::Error>),
    /// The store is poisoned.
    Poisoned(PoisonReason),
    /// The store is closed.
    Closed,
    /// The snapshot's generation was expired by the retention policy
    /// (`Options::max_retained_generations`, oldest-wins): its retired WAL
    /// segment has been reclaimed (or is draining its last in-flight reads).
    /// Terminal for this snapshot — every later read fails the same way; the
    /// store itself is healthy. Capture a fresh snapshot (M6 Phase E).
    SnapshotExpired,
}

impl From<std::io::Error> for ReadError {
    fn from(err: std::io::Error) -> Self {
        ReadError::Io(Arc::new(err))
    }
}

/// Failure of a write-path operation (`apply`, `reserve_write`,
/// `try_reserve_write`, `WritePermit::apply`).
///
/// `ConditionFailed` is **not** here: a failed condition is `Ok(ApplyOutcome::
/// ConditionFailed)`, publishes nothing, and consumes no txid.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum WriteError {
    /// Transient lack of capacity. Only `try_reserve_write` yields this; the
    /// awaiting forms wait instead. Retry later. Note it is **also** returned
    /// while an awaiting `reserve` caller is queued ahead — a non-blocking grab
    /// yields to the FIFO head even when raw capacity is free — so a fail-fast
    /// layer should treat it as "retry", not "hard saturation".
    Overloaded,
    /// The requested cost exceeds a hard cap no amount of waiting can satisfy
    /// (e.g. a batch worst case above half the dirty budget). Both reserve
    /// forms yield this. Permanent for this cost.
    CostExceedsLimit,
    /// A StoreWal batch's reserved WAL bytes would exceed `max_wal_bytes`.
    /// Always a pre-publication result: the batch never became visible and
    /// consumed no txid. Healthy fail-fast, not poison.
    StoreFull,
    /// The store is in maintenance (e.g. compaction); admission is fail-fast
    /// paused. Retry after maintenance completes.
    Compacting,
    /// The batch is malformed (empty/oversized key, oversized value, too many
    /// ops). A caller error caught before admission; carries the reason.
    InvalidBatch(crate::batch::BatchError),
    /// The store is poisoned.
    Poisoned(PoisonReason),
    /// The store is closed.
    Closed,
}

/// Returned by [`WritePermit::apply`](crate::WritePermit) when the batch's cost
/// exceeds what the permit reserved. Carries the batch back so the caller can
/// re-reserve. Distinct from `WriteError` because it is a caller-side cost
/// mismatch, never a store state.
#[derive(Clone, Copy, Debug)]
pub struct PermitMismatch;

/// Failure of [`BTreeMap::<Direct>::flush`](crate::BTreeMap).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FlushError {
    /// The receipt's incarnation does not match the current open.
    StaleIncarnation,
    Poisoned(PoisonReason),
    Closed,
}

/// Failure of [`BTreeMap::<Wal>::commit`](crate::BTreeMap).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum CommitError {
    /// The receipt's incarnation does not match the current open.
    StaleIncarnation,
    Poisoned(PoisonReason),
    Closed,
}

/// Failure of [`Store::<Wal>::checkpoint`](crate::Store).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum CheckpointError {
    /// This store owns no WAL segment namespace, so it cannot rotate: a
    /// checkpoint reclaims space by minting a fresh segment and unlinking the
    /// old one, and a caller-supplied `Arc<dyn PageIo>` device (`create_with_io`
    /// / `open_over`) is a bare byte device with no name to rotate.
    /// Use the path-backed constructors for a checkpointing store.
    Unsupported,
    /// The store is parked in maintenance, where the writer processes no
    /// commands — so a checkpoint (which *is* a writer command) fails fast rather
    /// than queueing behind an operator-held pause. Retry after `end_maintenance`.
    Compacting,
    /// A compaction aborted **cleanly before its commit point**: an I/O failure
    /// while probing or building the private temp file (size probe,
    /// create/write/sync, or the replacement segment's mint). Nothing shared
    /// changed — the store runs on, unpoisoned; retry when the cause (e.g. disk
    /// space) clears. A *promotion* (rename + directory fsync) failure is not
    /// this: it poisons, because the commit is then in doubt. A checkpoint
    /// never yields this either: it mutates the shared data file, so its I/O
    /// failures poison (M6 Phase D).
    Io(Arc<std::io::Error>),
    /// The store is poisoned — including by a failure *during* the checkpoint:
    /// once the header commit is in doubt, no truthful non-terminal answer
    /// exists, since the store cannot know which WAL segment a later recovery
    /// would scan (invariant 17).
    Poisoned(PoisonReason),
    /// The store is closed.
    Closed,
    /// An engine-internal invariant did not hold at the operation's entry (e.g.
    /// a maintenance command reached the coordinator off its committed barrier).
    /// A **clean pre-mutation refusal** — nothing was written, the store runs
    /// on — but it indicates a writer-loop bug: report it. Added by T3.4:
    /// these were release-elided `debug_assert`s, and an off-barrier
    /// compaction silently publishing a wrong root is exactly the failure a
    /// release build must refuse loudly.
    Internal(&'static str),
}

impl From<Terminal> for CheckpointError {
    fn from(t: Terminal) -> Self {
        match t {
            Terminal::Poisoned(r) => CheckpointError::Poisoned(r),
            Terminal::Closed => CheckpointError::Closed,
        }
    }
}

/// Failure of `close()`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum CloseError {
    /// A final flush/commit failed; the store poisoned. Recovery yields the
    /// last previously durable prefix.
    Poisoned(PoisonReason),
    /// Already closed by a concurrent caller.
    AlreadyClosed,
}

impl From<Terminal> for ReadError {
    fn from(t: Terminal) -> Self {
        match t {
            Terminal::Poisoned(r) => ReadError::Poisoned(r),
            Terminal::Closed => ReadError::Closed,
        }
    }
}

impl From<Terminal> for WriteError {
    fn from(t: Terminal) -> Self {
        match t {
            Terminal::Poisoned(r) => WriteError::Poisoned(r),
            Terminal::Closed => WriteError::Closed,
        }
    }
}

impl From<Terminal> for FlushError {
    fn from(t: Terminal) -> Self {
        match t {
            Terminal::Poisoned(r) => FlushError::Poisoned(r),
            Terminal::Closed => FlushError::Closed,
        }
    }
}

impl From<Terminal> for CommitError {
    fn from(t: Terminal) -> Self {
        match t {
            Terminal::Poisoned(r) => CommitError::Poisoned(r),
            Terminal::Closed => CommitError::Closed,
        }
    }
}

macro_rules! impl_display_debug {
    ($($t:ty),*) => {$(
        impl std::fmt::Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:?}", self)
            }
        }
        impl std::error::Error for $t {}
    )*};
}
impl_display_debug!(
    OpenError,
    ReadError,
    WriteError,
    FlushError,
    CommitError,
    CheckpointError,
    CloseError,
    PermitMismatch
);
