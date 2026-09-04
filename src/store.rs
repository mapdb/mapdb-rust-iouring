//! `Store<B>`, its shared inner core, the published root descriptor, and the
//! writer command type.
//!
//! `Store<B>` is a cheap `Arc` handle. All reads go through the lock-free
//! `ArcSwap<RootDescriptor>` publication point; all writes are serialized by the
//! single writer actor (see [`crate::writer`]). The backend marker `B` makes the
//! crash guarantee visible in the type and selects the backend-specific
//! durability barrier (`flush` for Direct, `commit` for Wal).

use crate::admission::{AdmissionLimits, Budget, Reservation, WalMeter, WalReservation, WriteCost};
use crate::backend::{Backend, Direct, FileGeneration, Member, Wal};
use crate::batch::{ApplyOutcome, PageBatch, WriteBatch};
use crate::direct::{direct_generation, Allocator, DirectCoordinator, FIRST_DATA_INDEX};
use crate::error::{CheckpointError, OpenError, WriteError};
use crate::io::PageIo;
use crate::metrics::Metrics;
use crate::page::{PageRef, PAGE_SIZE};
use crate::retention::{GenHold, Retention, SnapshotGen};
use crate::state::{EngineState, IntakeDecision};
use crate::version::{Incarnation, Version};
use crate::wal::{wal_generation, WalCoordinator, WalLocator};
use crate::writer::Coordinator;
use arc_swap::ArcSwap;
use event_listener::Event;
use futures_channel::oneshot;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Nominal per-batch tree-depth term for the admission-cost *estimate*
/// ([`WriteBatch::cost`]). Not a hard bound — the writer reconciles the dirty and
/// WAL charges to actual before publication.
pub(crate) const DEPTH_BOUND: u32 = 8;

/// The atomically published `(version, root, files)` triple.
///
/// The `file` generation lives *inside* the descriptor so a compaction/cutover
/// swaps a whole `(root, files)` atom and a reader can never capture a torn
/// `(root, wrong-file)` pair (invariant 1). All reads traverse `root` through
/// the page cache (the M0 `logical` placeholder was deleted at M3).
pub(crate) struct RootDescriptor {
    pub(crate) version: Version,
    pub(crate) root: PageRef,
    pub(crate) file: Arc<FileGeneration>,
    pub(crate) entry_count: u64,
    /// The generation's revocable holder (M6 Phase E): snapshots bind to this,
    /// not to `file`, so a retired generation can be force-reclaimed. One
    /// `SnapshotGen` per generation — descriptors published over the same
    /// generation share it (`next_descriptor` clones the `Arc`); a rebind mints
    /// a fresh one and retires the old.
    pub(crate) retain: Arc<SnapshotGen>,
}

/// One admitted write, carrying its permit into the writer. The command owns the
/// reservation, so it survives the caller dropping the reply future (invariant
/// 4): the reservation releases only when the command is fully handled.
pub(crate) enum Command {
    Apply {
        batch: WriteBatch,
        reservation: Reservation,
        reply: oneshot::Sender<Result<ApplyOutcome, WriteError>>,
    },
    /// Run a checkpoint (StoreWal). Carries no reservation: it consumes no
    /// admission capacity and publishes no txid. Being a *command* is the whole
    /// design — the writer reaches it only between commands, after committing
    /// everything queued ahead of it, which is what establishes the quiescence the
    /// protocol needs.
    Checkpoint {
        reply: oneshot::Sender<Result<crate::wal::CheckpointReport, CheckpointError>>,
    },
    /// Run a data-file compaction (both backends — M6 Phase D). A command for
    /// the same reason a checkpoint is: the writer reaches it only at a
    /// drained, fully-flushed barrier.
    Compact {
        reply: oneshot::Sender<Result<crate::compact::CompactReport, CheckpointError>>,
    },
}

/// Backend-neutral shared state. Non-generic so the writer loop and reads need
/// no `B`.
pub(crate) struct Core {
    pub state: EngineState,
    pub root: ArcSwap<RootDescriptor>,
    pub budget: Arc<Budget>,
    pub incarnation: Incarnation,
    /// The last-completed backend-durable version's txid. M0 treats a published
    /// version as immediately durable; M3/M4 advance this at the flush/commit
    /// frontier.
    pub durable_txid: std::sync::atomic::AtomicU64,
    /// The highest txid an explicit `flush`/`commit` has requested be made
    /// durable. The writer flushes when this exceeds `flushed_txid`, so a
    /// receipt wait resolves promptly even under a continuously busy command
    /// channel. Advanced monotonically via `max`.
    pub flush_request: std::sync::atomic::AtomicU64,
    /// The shared page cache. Keyed by immutable `PageId`, so it is
    /// generation-independent and lives here on the backend-neutral core. The
    /// read path traverses real pages through it once `root` is non-empty.
    pub cache: Arc<crate::cache::PageCache>,
    /// The path-backed open protocol's sidecar `flock` file (`None` for
    /// caller-device stores and fixtures). Injected at writer spawn
    /// (`build_bound`) and dropped — releasing the lock — by the **writer
    /// thread as its final act** (`writer::run`), on the clean, poisoned, and
    /// panicked exits alike. Tying release to writer exit (not handle drop, as
    /// the pre-T1.2 `close_hook` did) closes the two-writers window where the
    /// last handle's drop freed the lock while the detached writer was still
    /// draining admitted commands (no path-backed store is writerless).
    /// `close()` still joins the writer, so a closed store's path is
    /// reopenable while handles linger; a dropped store's path becomes
    /// reopenable as soon as its writer drains.
    pub namespace_lock: parking_lot::Mutex<Option<std::fs::File>>,
    /// StoreWal only: the monotonic WAL-byte meter admission charges against
    /// (invariant 18). `None` for StoreDirect. The overlay locator lives in the
    /// `FileGeneration`; this is the byte budget the record reservations release
    /// through.
    pub wal: Option<Arc<WalMeter>>,
    /// StoreWal only: the current locator, kept here so metrics can read
    /// `overlay_pages` without threading the generation. Swapped (not mutated) by
    /// a checkpoint's rebind, so a reader that already captured a generation
    /// keeps resolving through the locator that generation was published with.
    pub wal_locator: arc_swap::ArcSwapOption<WalLocator>,
    /// StoreWal last open-scan duration in milliseconds (metric `open_scan_ms`);
    /// 0 for StoreDirect and fresh creates.
    pub open_scan_ms: u64,
    /// StoreWal bytes accepted by the last open recovery scan — the recovered
    /// logical frontier (metric `open_scan_bytes`), excluding any rejected torn
    /// tail; the unambiguous "a scan ran" signal — **0 iff no scan** (StoreDirect
    /// and fresh creates), `> 0` after any recovery. Unlike `open_scan_ms` this
    /// cannot round to 0 for a fast scan.
    pub open_scan_bytes: u64,
    /// StoreWal WAL file length (metric `wal_physical_bytes`). Seeded from the
    /// file size at open and advanced to the highest write end; may exceed the
    /// logical frontier after a torn-tail overwrite. `0` for StoreDirect. A
    /// checkpoint's rotation resets it to the fresh segment's header length.
    pub wal_physical_bytes: std::sync::atomic::AtomicU64,
    /// Checkpoints completed **this open** (metric `checkpoints`).
    pub checkpoints: std::sync::atomic::AtomicU64,
    /// Compactions completed **this open** (metric `compactions`, M6 Phase D).
    pub compactions: std::sync::atomic::AtomicU64,
    /// The txid of the last completed checkpoint — the point up to which the
    /// data file holds every live page outright (metric `last_checkpoint_txid`).
    /// Seeded at open from the winning header, so it survives reopen.
    pub last_checkpoint_txid: std::sync::atomic::AtomicU64,
    /// Retired WAL segments a rotation could not unlink (metric
    /// `leaked_segments`). Never a correctness problem — the winning header does
    /// not reference them — but an operator wants to know.
    pub leaked_segments: std::sync::atomic::AtomicU64,
    /// Retired-generation registry + expiration policy (M6 Phase E). A rebind
    /// retires the superseded generation here; the bound revokes oldest-first.
    pub retention: Retention,
    /// Data-file length (metric `data_physical_bytes`). Seeded at open and
    /// advanced by the durable transitions that change it: a checkpoint's append
    /// and a compaction's cutover. Held here — rather than probed with `size()` —
    /// so `metrics()` stays synchronous and free.
    pub data_physical_bytes: std::sync::atomic::AtomicU64,
    /// Live pages in the data file **as of the last maintenance** (metric
    /// `data_live_pages`, M6 Phase H). Set by every checkpoint (WAL) and
    /// compaction (both backends) from the walked live set; `0` before the first
    /// one has walked it (which on Direct means "unknown", not "empty"). Stale
    /// between maintenances — a fragmentation signal, not a live gauge.
    pub data_live_pages: std::sync::atomic::AtomicU64,
    /// Data-file bytes reclaimed by compaction so far this open (metric
    /// `reclaimed_bytes`, M6 Phase H): the cumulative sum of
    /// `data_bytes_before − data_bytes_after` over compactions that shrank the
    /// file. Checkpoints (which *append* live pages) never add to it.
    pub reclaimed_bytes: std::sync::atomic::AtomicU64,
    /// Generations published this open (metric `generations_created`, M6 Phase
    /// H): seeded at `1` for the generation open mints, then `+1` per rebind
    /// (each checkpoint/compaction retires one and publishes its successor).
    /// Monotonic; pairs with `expired_generations` and `retained_generations`.
    pub generations_created: std::sync::atomic::AtomicU64,
    /// The admission byte ceiling (`AdmissionLimits::max_command_bytes`), kept
    /// here because the writer needs it too: it bounds what a command may carry
    /// *out* (`returning_old_values`) as well as in (M6 Phase F).
    pub max_command_bytes: u64,
    /// Total-owned-physical-bytes ceiling (`Options::max_data_bytes`, M6 Phase F).
    /// `None` = unbounded. Enforced at the checkpoint, which **skips** rather
    /// than fails when the store would not fit.
    pub max_data_bytes: Option<u64>,
    /// Checkpoints skipped because they would breach `max_data_bytes` (metric
    /// `checkpoints_skipped_space`, M6 Phase F). A healthy counter, not an error
    /// count: the store keeps serving while the WAL has room.
    pub checkpoints_skipped_space: std::sync::atomic::AtomicU64,
}

impl Core {
    /// The currently visible root.
    pub fn root(&self) -> Arc<RootDescriptor> {
        self.root.load_full()
    }

    pub fn metrics(&self) -> Metrics {
        let root = self.root();
        let (slots, bytes, pages) = self.budget.usage();
        let (wal_logical_bytes, wal_reserved_bytes) =
            self.wal.as_ref().map(|m| m.usage()).unwrap_or((0, 0));
        let locator = self.wal_locator.load();
        let overlay_pages = locator.as_ref().map(|l| l.len()).unwrap_or(0);
        let checkpoint_pages = locator.as_ref().map(|l| l.base_len()).unwrap_or(0);
        let retention = self.retention.metrics();
        let data_bytes = self
            .data_physical_bytes
            .load(std::sync::atomic::Ordering::Acquire);
        let wal_physical = self
            .wal_physical_bytes
            .load(std::sync::atomic::Ordering::Acquire);
        Metrics {
            entry_count: root.entry_count,
            visible_txid: root.version.txid(),
            durable_txid: self.durable_txid.load(std::sync::atomic::Ordering::Acquire),
            admission_slots: slots,
            admission_bytes: bytes,
            dirty_pages: pages,
            wal_logical_bytes,
            wal_reserved_bytes,
            wal_physical_bytes: wal_physical,
            overlay_pages,
            checkpoint_pages,
            checkpoints: self.checkpoints.load(std::sync::atomic::Ordering::Acquire),
            compactions: self.compactions.load(std::sync::atomic::Ordering::Acquire),
            last_checkpoint_txid: self
                .last_checkpoint_txid
                .load(std::sync::atomic::Ordering::Acquire),
            leaked_segments: self
                .leaked_segments
                .load(std::sync::atomic::Ordering::Acquire),
            retained_generations: retention.retained,
            draining_generations: retention.draining,
            retained_segment_bytes: retention.bytes,
            expired_generations: retention.expired,
            data_physical_bytes: data_bytes,
            data_live_pages: self
                .data_live_pages
                .load(std::sync::atomic::Ordering::Acquire),
            reclaimed_bytes: self
                .reclaimed_bytes
                .load(std::sync::atomic::Ordering::Acquire),
            generations_created: self
                .generations_created
                .load(std::sync::atomic::Ordering::Acquire),
            store_physical_bytes: owned_bytes(data_bytes, wal_physical, retention.bytes),
            checkpoints_skipped_space: self
                .checkpoints_skipped_space
                .load(std::sync::atomic::Ordering::Acquire),
            open_scan_ms: self.open_scan_ms,
            open_scan_bytes: self.open_scan_bytes,
            ..Metrics::default()
        }
    }
}

/// The store's total owned bytes — **the one accounting rule** (M6 Phase F).
///
/// Both the `store_physical_bytes` metric and `max_data_bytes` enforcement go
/// through here, so the number an operator reads is the number the checkpoint
/// decides against; two accountings that drift apart would make the limit
/// unexplainable. The checkpoint calls it with *projected* terms.
///
/// Summed file **lengths**, not blocks the filesystem allocated. See
/// [`Metrics::store_physical_bytes`](crate::Metrics::store_physical_bytes) for
/// what it is blind to; it is a lower bound.
pub fn owned_bytes(data_bytes: u64, wal_bytes: u64, retained_bytes: u64) -> u64 {
    data_bytes
        .saturating_add(wal_bytes)
        .saturating_add(retained_bytes)
}

/// What differs between the two backends at store construction, passed to
/// [`Store::build_bound`].
pub(crate) enum BackendSetup {
    Direct {
        store_uuid: [u8; 16],
        /// The data-path namespace owner (M6 Phase D) — present only for stores
        /// that own their data path; `None` disables compaction.
        data_host: Option<Arc<dyn crate::compact::DataFileHost>>,
        /// Data-file length at open (metric `data_physical_bytes`): the header's
        /// logical tail.
        data_bytes: u64,
        /// The header slot the next flush overwrites (T1.1): the open
        /// protocol's loser, or slot B after a create (identical slots, A wins).
        stale_slot: crate::direct::DirectSlot,
    },
    Wal {
        locator: Arc<WalLocator>,
        /// The data device (checkpoints relocate live pages into it) and the
        /// identity/namespace/frontier state the checkpoint protocol needs.
        binding: crate::wal::WalBinding,
        /// Current logical/append frontier (segment header + accepted records).
        append_offset: u64,
        /// Accepted chain head (record checksum or WAL header seed).
        chain_head: u32,
        max_wal_bytes: u64,
        /// Recovery scan duration (metric `open_scan_ms`); 0 for a fresh create.
        open_scan_ms: u64,
        /// Bytes accepted by the recovery scan — the recovered logical frontier
        /// (metric `open_scan_bytes`), excluding a rejected torn tail; 0 for a
        /// fresh create, `> 0` after any recovery.
        open_scan_bytes: u64,
        /// Initial WAL segment length (metric `wal_physical_bytes`).
        physical_bytes: u64,
        /// Data-file length at open (metric `data_physical_bytes`). The **real**
        /// EOF, kept separate from `binding.data_frontier`, which rounds it up to a
        /// page for appends: seeding the metric from the frontier would round the
        /// file's reported size up at every unaligned reopen, so the same store
        /// would report one size before a restart and another after.
        data_bytes: u64,
        /// The winning header's checkpoint txid, so the metric survives reopen.
        checkpoint_txid: u64,
    },
}

/// Backend-specific inner handle state.
pub(crate) struct StoreInner<B: Backend> {
    pub core: Arc<Core>,
    pub commands: async_channel::Sender<Command>,
    /// Joined once, by the first `close()` that observes the writer finished.
    writer: parking_lot::Mutex<Option<JoinHandle<()>>>,
    /// Guards single-shot close initiation among racing callers.
    closing: AtomicBool,
    // Note: the sidecar-lock release moved from a handle-owned `close_hook` to
    // `Core::namespace_lock`, dropped by the writer thread at exit (T1.2) — a
    // last-handle drop therefore keeps the path locked until the writer has
    // drained and stopped, instead of releasing it mid-drain.
    _backend: PhantomData<B>,
}

impl<B: Backend> StoreInner<B> {
    /// The store's sticky terminal reason as a `CheckpointError` — for the paths
    /// where the command could not be delivered or its reply sender was dropped
    /// (a closed store, or a writer that panicked on the way out).
    fn terminal_checkpoint_error(&self) -> CheckpointError {
        self.core
            .state
            .check_terminal()
            .err()
            .map(CheckpointError::from)
            .unwrap_or(CheckpointError::Closed)
    }
}

impl<B: Backend> StoreInner<B> {
    /// Enqueues an already-admitted batch, transferring `reservation` into the
    /// command under the intake gate. Returns a `Send` future
    /// that resolves with the apply outcome; dropping it after this call returns
    /// does not drop the command (invariant 4).
    ///
    /// This is the **single** choke point where a batch reaches the writer, so
    /// batch validation lives here — both the `apply` sugar and the permit path
    /// go through it, and neither can smuggle a malformed batch past the size
    /// caps into the (future) page codec. A malformed batch resolves to
    /// `InvalidBatch` without enqueueing, consuming permit and batch locally.
    pub(crate) fn enqueue(
        &self,
        batch: WriteBatch,
        reservation: Reservation,
    ) -> impl std::future::Future<Output = Result<ApplyOutcome, WriteError>> + Send {
        let (tx, rx) = oneshot::channel();
        let max_ops = self.core.budget.limits().max_ops_per_batch;
        match batch.validate(max_ops) {
            Err(e) => {
                drop(reservation);
                let _ = tx.send(Err(WriteError::InvalidBatch(e)));
            }
            Ok(()) => self.core.state.with_intake(|decision| match decision {
                IntakeDecision::Open => {
                    let cmd = Command::Apply {
                        batch,
                        reservation,
                        reply: tx,
                    };
                    // The command channel is unbounded (admission already bounds
                    // it, invariant 15), so a send fails only if the writer
                    // receiver is gone — which, under this lock, means the store
                    // went terminal (e.g. a panicked writer that poisoned on the
                    // way out).
                    if let Err(err) = self.commands.try_send(cmd) {
                        let Command::Apply {
                            reply, reservation, ..
                        } = err.into_inner()
                        else {
                            unreachable!("sent an Apply")
                        };
                        drop(reservation);
                        let term = self
                            .core
                            .state
                            .check_terminal()
                            .err()
                            .map(WriteError::from)
                            .unwrap_or(WriteError::Closed);
                        let _ = reply.send(Err(term));
                    }
                }
                IntakeDecision::Terminal(t) => {
                    // Consume permit + batch locally; resolve with the terminal
                    // error. Nothing is enqueued. (batch dropped here.)
                    let _ = batch;
                    drop(reservation);
                    let _ = tx.send(Err(t.into()));
                }
            }),
        }
        // On oneshot cancellation the reply sender was dropped without a value —
        // most benignly the store closed, but it also happens when the writer
        // *panicked mid-command* (e.g. a hostile `PageIo` panic): the in-flight
        // command's sender unwinds off the writer stack while `catch_unwind`
        // poisons the store. Return the store's sticky terminal reason so the
        // triggering caller sees the true `Poisoned(...)`, not a false `Closed`.
        let core = Arc::clone(&self.core);
        async move {
            rx.await.unwrap_or_else(|_| {
                Err(core
                    .state
                    .check_terminal()
                    .err()
                    .map(WriteError::from)
                    .unwrap_or(WriteError::Closed))
            })
        }
    }
}

/// A cheap, clonable store handle. `B` is [`Direct`] or [`Wal`].
pub struct Store<B: Backend> {
    pub(crate) inner: Arc<StoreInner<B>>,
}

impl<B: Backend> Clone for Store<B> {
    fn clone(&self) -> Self {
        Store {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Open/create options.
#[derive(Clone)]
pub struct Options {
    pub limits: AdmissionLimits,
    /// StoreWal WAL-byte ceiling (ignored by Direct).
    pub max_wal_bytes: u64,
    /// Fix the open incarnation (tests). `None` draws a fresh random nonce.
    pub incarnation: Option<Incarnation>,
    /// Nominal clean-tier page-cache capacity, in 16-KiB pages. An explicit
    /// `Some(n)` is used unchanged (`PageCache` clamps zero to one), which is the
    /// escape hatch for a fixed process budget, multiple stores in one process,
    /// or a known working set.
    ///
    /// `None` derives a conservative per-store capacity from one eighth of the
    /// memory currently available at construction. On Linux it uses
    /// `/proc/meminfo`'s `MemAvailable`, constrained by remaining cgroup v1/v2
    /// headroom when detectable, and clamps the result to 1,024–32,768 pages
    /// (16–512 MiB of clean page images). If detection fails, it falls back to
    /// 1,024 pages. Cache metadata and pinned dirty pages are additional, so
    /// tightly budgeted deployments and processes opening several stores should
    /// set this field explicitly rather than treating the default as a total
    /// memory limit.
    ///
    /// Capacity is nominal: `quick_cache` divides it across shards. Ceiling
    /// division can permit up to `shards - 1` extra resident pages, while a
    /// skewed key distribution can make a busy shard evict before the aggregate
    /// reaches `n`. Size `Some(n)` to cover the hot page working set, not the raw
    /// key/value payload alone; undersizing makes random reads pay page I/O and
    /// decode/checksum work despite a warm OS page cache.
    pub cache_capacity: Option<usize>,
    /// Whether [`Store::<Direct>::open_over`] structurally walks the whole tree
    /// at open and requires the actual entry count to equal the header's.
    /// Defaults to `true` (correctness-first: it defends against a
    /// checksum-valid but *forged* count that could drive a premature empty
    /// collapse). Set `false` for a fast, header-checksum-only open of a large
    /// trusted store — the header CRC still catches accidental corruption; only
    /// a deliberately forged count slips through, and reads still validate every
    /// page lazily at the decode trust boundary.
    ///
    /// **Cost changed in M6 Phase F.** The walk now follows overflow chains too —
    /// it has to, since a chain's structure is only checkable across its pages
    /// ([`OverflowCursor`](crate::page::OverflowCursor)) — so a store whose bytes
    /// live mostly in large values pays an open proportional to **the live data**,
    /// not to the tree's own pages. For inline-sized values this is unchanged.
    /// A store of many big values that wants a fast open is exactly the case
    /// `false` exists for.
    pub verify_tree_on_open: bool,
    /// How many **retired** generations may stay pinned by snapshots before the
    /// oldest is force-expired (M6 Phase E). A generation retires when a
    /// checkpoint's rebind supersedes it; each retired generation pins its
    /// retired WAL segment's bytes for as long as some snapshot holds it. Past
    /// the bound, the oldest retained generation is revoked oldest-wins: its
    /// snapshots' later reads return [`ReadError::SnapshotExpired`]
    /// (crate::error::ReadError), in-flight reads finish safely, and the
    /// segment's bytes free at that drain. `None` opts out: snapshots stay
    /// readable indefinitely and bounded physical space is explicitly **not**
    /// offered under pinned snapshots. The bound is a
    /// generation count rather than a wall-clock age: deterministic, clockless,
    /// and 1:1 with what pins space — so a *lone* old snapshot within the bound
    /// stays readable across arbitrarily many checkpoints (this is a space
    /// bound, not an age bound). `Some(0)` is legal and means **no** retired
    /// generation may stay pinned: every snapshot expires at the first
    /// checkpoint that retires its generation, before its first read if need
    /// be. The bound governs snapshot holds only; an in-flight current-root
    /// read future (root captured at call, per the frozen API contract) pins
    /// its generation for the future's own lifetime regardless.
    pub max_retained_generations: Option<u32>,
    /// A **budget the StoreWal checkpoint holds itself to** (M6 Phase F).
    /// `None` (the default) is unbounded. **Ignored by StoreDirect**, exactly as
    /// `max_wal_bytes` is — for the same reason: neither backend has the other's
    /// enforcement point.
    ///
    /// # This is not a ceiling on the store's size
    ///
    /// Say plainly what it is, because the tempting reading is wrong.
    /// It bounds **one transition** — a checkpoint
    /// declines to run if the store would not fit afterwards. It does **not**
    /// bound the store:
    ///
    /// - writes keep flowing after a check passes, and the WAL keeps growing, so
    ///   the total can and does exceed this between checkpoints. `max_wal_bytes`
    ///   is what actually refuses writes;
    /// - StoreDirect has no checkpoint, hence no enforcement point at all;
    /// - a store that is *already* over the limit at open is not refused; its next
    ///   checkpoint simply skips.
    ///
    /// The bound worth reasoning about is therefore roughly
    /// `max_data_bytes + max_wal_bytes`, and only for StoreWal. What this option
    /// buys is narrower and real: **the data file cannot be grown into a corner it
    /// cannot compact out of.**
    ///
    /// # What it measures
    ///
    /// The projection is a total-footprint estimate at the moment of the
    /// checkpoint — the live data file, the WAL (old segment plus the fresh one
    /// the rotation mints; they coexist), bytes retained for snapshots, and
    /// **maintenance headroom** for a compaction's second copy of the live set.
    /// See `WalCoordinator::space_check`. It uses the same accounting as
    /// [`Metrics::store_physical_bytes`](crate::Metrics::store_physical_bytes),
    /// which documents what that accounting is blind to; both are lower bounds, so
    /// the effective limit is *softer* than the number set here, never tighter.
    ///
    /// The headroom is sized on the live set as it stands. Pages added after this
    /// checkpoint enlarge the copy a later compaction needs, so it is an estimate,
    /// not a guarantee.
    ///
    /// # Cost and failure
    ///
    /// When `Some`, each checkpoint pays one extra live-set walk to count the set
    /// before relocating it; `None` pays nothing. A skipped checkpoint reports
    /// [`CheckpointReport::skipped_space`](crate::CheckpointReport::skipped_space)
    /// — a healthy outcome, counted in `checkpoints_skipped_space`, not an error.
    ///
    /// If checkpoints keep skipping, the WAL eventually fills and writes get
    /// `StoreFull`: the store goes read-only until space frees (a retained
    /// generation releasing, or an operator raising the limit). That is the
    /// intended terminal state of the backpressure, and it is recoverable —
    /// unlike filling the disk.
    ///
    /// This is a **reservation against the store's own accounting**, not a promise
    /// about the filesystem: another process can still consume the space, and a
    /// genuine ENOSPC follows the ordinary rules (invariant 17) — poison when the
    /// shared data file or WAL is being mutated, a clean abort when only a private
    /// compaction temp is.
    pub max_data_bytes: Option<u64>,
    /// Which durable-byte backend the **path-backed** constructors
    /// (`create_path`/`open_path`) mint every device with — the data file, each
    /// rotated WAL segment, and the compaction temp (M7-0). Defaults
    /// to [`IoBackend::Syscall`] (G1's closed M6 decision). Selecting
    /// [`IoBackend::Uring`] requires the `io-uring` feature (without it the variant
    /// does not exist — a compile error, not an open-time one); with the feature but
    /// on a host where io_uring is unavailable, the ring fails to build at open.
    /// Ignored by the caller-supplied-device
    /// constructors (`create_with_io`/`open_over`), which own no namespace. This
    /// makes the io_uring backend reachable on the production path *with
    /// checkpoint/compaction intact* — G1's #1 reopen precondition; M7-E re-decides
    /// the default with production-path data.
    pub io_backend: crate::io::IoBackend,

    /// Open the **data file** and **compaction temp** with `O_DIRECT`, bypassing the
    /// page cache (M7-A). Orthogonal to [`io_backend`](Self::io_backend)
    /// (valid on either backend, so M7-E can attribute a Syscall+O_DIRECT arm). The
    /// byte-granular WAL record segment is **never** O_DIRECT — its record appends are
    /// not page-aligned, so `direct_io` applies per file role.
    /// Defaults to `false`. Linux-only: opening `direct_io` on another OS is refused
    /// at open. If the filesystem does not honor O_DIRECT at the store's alignment the
    /// open is **refused**, never silently downgraded to buffered (a silent fallback
    /// would make a "cold" benchmark arm secretly page-cached). On the
    /// io_uring backend `direct_io` is refused until M7-A commit 2. Ignored by the
    /// caller-supplied-device constructors (`create_with_io`/`open_over`).
    pub direct_io: bool,

    /// Max concurrent in-flight page writes per StoreDirect flush (M7-B fan-out).
    /// `None` uses the built-in [`FLUSH_FANOUT_CAP`](crate::direct)
    /// default (64). A tuning knob for M7-E, which sweeps this across backends and
    /// O_DIRECT to size the write-arm queue depth; also lets a test force the
    /// serial path (`Some(1)`) or a small multi-chunk boundary. Clamped to ≥ 1.
    /// StoreWal ignores it (its checkpoint has a separate write path).
    pub fanout_cap: Option<usize>,

    /// M7-E decision harness: a bench-only decorator wrapping every
    /// device the production `create_path`/`open_path` mints — data file, rotated
    /// WAL segments, compaction temps — so the harness can count/trace the
    /// *production* path's I/O without falling back to the caller-supplied-device
    /// constructors (which have no checkpoint). `None` (the default) mints devices
    /// exactly as the shipping path. Ignored by `create_with_io`/`open_over`, which
    /// bring their own device. Only present under `bench-instrument`.
    #[cfg(feature = "bench-instrument")]
    pub device_wrapper: Option<crate::io::DeviceWrapper>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            limits: AdmissionLimits::default(),
            max_wal_bytes: 256 * 1024 * 1024,
            incarnation: None,
            cache_capacity: None,
            verify_tree_on_open: true,
            max_retained_generations: Some(2),
            max_data_bytes: None,
            io_backend: crate::io::IoBackend::Syscall,
            direct_io: false,
            fanout_cap: None,
            #[cfg(feature = "bench-instrument")]
            device_wrapper: None,
        }
    }
}

impl<B: Backend> Store<B> {
    /// Shared construction: builds the core, publishes the initial root, spawns
    /// the writer thread, and hands it a Direct flush coordinator bound to
    /// `files`/`alloc` (the generation-binding seam).
    ///
    /// `initial_txid` is the visible root's txid — `0` for a fresh create, or the
    /// reopened header's `header_txid` so the persisted header field never
    /// regresses across opens. Versions still compare only
    /// within one incarnation (invariant 11), so a stale receipt cannot match a
    /// later open. The base header is written durably by the create protocol
    /// before this is called; nothing is laid down here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_bound(
        files: FileGeneration,
        alloc: Allocator,
        setup: BackendSetup,
        root: PageRef,
        entry_count: u64,
        initial_txid: u64,
        options: &Options,
        namespace_lock: Option<std::fs::File>,
    ) -> Store<B> {
        let incarnation = options.incarnation.unwrap_or_else(Incarnation::generate);
        let event = Arc::new(Event::new());
        let state = EngineState::new_running(Arc::clone(&event));
        let budget = Budget::new(options.limits, Arc::clone(&event));
        let files = Arc::new(files);
        let high_water = (options.limits.max_dirty_pages / 2).max(1) as usize;

        // Build the backend coordinator and the (StoreWal-only) WAL meter/overlay.
        let (
            coordinator,
            wal_meter,
            wal_locator,
            open_scan_ms,
            open_scan_bytes,
            physical_bytes,
            checkpoint_txid,
            data_physical_bytes,
        ) = match setup {
            BackendSetup::Direct {
                store_uuid,
                data_host,
                data_bytes,
                stale_slot,
            } => {
                let dev = files.device(Member::Data);
                let coord = DirectCoordinator::new(
                    dev,
                    store_uuid,
                    alloc,
                    initial_txid,
                    high_water,
                    options.fanout_cap,
                    data_host,
                    stale_slot,
                );
                (
                    Coordinator::Direct(coord),
                    None,
                    None,
                    0,
                    0,
                    0,
                    0,
                    data_bytes,
                )
            }
            BackendSetup::Wal {
                locator,
                binding,
                append_offset,
                chain_head,
                max_wal_bytes,
                open_scan_ms,
                open_scan_bytes,
                physical_bytes,
                checkpoint_txid,
                data_bytes,
            } => {
                let wal_dev = files.device(Member::Wal);
                let meter = crate::wal::wal_meter(max_wal_bytes, append_offset, Arc::clone(&event));
                let coord = WalCoordinator::new(
                    wal_dev,
                    incarnation.as_u64(),
                    alloc,
                    Arc::clone(&locator),
                    append_offset,
                    chain_head,
                    initial_txid,
                    high_water,
                    binding,
                );
                (
                    Coordinator::Wal(coord),
                    Some(meter),
                    Some(locator),
                    open_scan_ms,
                    open_scan_bytes,
                    physical_bytes,
                    checkpoint_txid,
                    data_bytes,
                )
            }
        };

        let base = RootDescriptor {
            version: Version::new(incarnation, initial_txid),
            root,
            file: Arc::clone(&files),
            entry_count,
            retain: SnapshotGen::active(Arc::clone(&files)),
        };
        let core = Arc::new(Core {
            state,
            root: ArcSwap::from_pointee(base),
            budget,
            incarnation,
            durable_txid: std::sync::atomic::AtomicU64::new(initial_txid),
            flush_request: std::sync::atomic::AtomicU64::new(0),
            cache: Arc::new(make_cache(options)),
            wal: wal_meter,
            wal_locator: arc_swap::ArcSwapOption::from(wal_locator),
            open_scan_ms,
            open_scan_bytes,
            wal_physical_bytes: std::sync::atomic::AtomicU64::new(physical_bytes),
            checkpoints: std::sync::atomic::AtomicU64::new(0),
            compactions: std::sync::atomic::AtomicU64::new(0),
            last_checkpoint_txid: std::sync::atomic::AtomicU64::new(checkpoint_txid),
            leaked_segments: std::sync::atomic::AtomicU64::new(0),
            retention: Retention::new(options.max_retained_generations),
            data_physical_bytes: std::sync::atomic::AtomicU64::new(data_physical_bytes),
            data_live_pages: std::sync::atomic::AtomicU64::new(0),
            reclaimed_bytes: std::sync::atomic::AtomicU64::new(0),
            generations_created: std::sync::atomic::AtomicU64::new(1),
            max_command_bytes: options.limits.max_command_bytes,
            max_data_bytes: options.max_data_bytes,
            checkpoints_skipped_space: std::sync::atomic::AtomicU64::new(0),
            namespace_lock: parking_lot::Mutex::new(namespace_lock),
        });
        let (tx, rx) = async_channel::unbounded::<Command>();
        let writer_core = Arc::clone(&core);
        let handle = std::thread::Builder::new()
            .name(format!("mapdb-uring-writer-{}", B::NAME))
            .spawn(move || crate::writer::run(writer_core, rx, coordinator))
            .expect("spawn writer thread");
        Store {
            inner: Arc::new(StoreInner {
                core,
                commands: tx,
                writer: parking_lot::Mutex::new(Some(handle)),
                closing: AtomicBool::new(false),
                _backend: PhantomData,
            }),
        }
    }

    /// Builds a **read-only** store over a prebuilt page tree: no writer thread
    /// is spawned and the store starts in `Maintenance`, so every fresh-admission
    /// attempt fails fast with `Compacting` and no write can publish an empty
    /// root over the tree. This is the M2 read-path fixture; a real
    /// writable Direct store comes from [`Store::<Direct>::create_with_io`] /
    /// [`Store::<Direct>::open_over`].
    pub(crate) fn build_readonly(
        files: FileGeneration,
        root: PageRef,
        entry_count: u64,
        options: &Options,
    ) -> Store<B> {
        let incarnation = options.incarnation.unwrap_or_else(Incarnation::generate);
        let event = Arc::new(Event::new());
        let state = EngineState::new_running(Arc::clone(&event));
        // Park admission: fresh writes fail fast with `Compacting`.
        state.enter_maintenance();
        let budget = Budget::new(options.limits, Arc::clone(&event));
        let files = Arc::new(files);
        let base = RootDescriptor {
            version: Version::base(incarnation),
            root,
            file: Arc::clone(&files),
            entry_count,
            retain: SnapshotGen::active(files),
        };
        let core = Arc::new(Core {
            state,
            root: ArcSwap::from_pointee(base),
            budget,
            incarnation,
            durable_txid: std::sync::atomic::AtomicU64::new(0),
            flush_request: std::sync::atomic::AtomicU64::new(0),
            cache: Arc::new(make_cache(options)),
            wal: None,
            wal_locator: arc_swap::ArcSwapOption::empty(),
            open_scan_ms: 0,
            open_scan_bytes: 0,
            wal_physical_bytes: std::sync::atomic::AtomicU64::new(0),
            checkpoints: std::sync::atomic::AtomicU64::new(0),
            compactions: std::sync::atomic::AtomicU64::new(0),
            last_checkpoint_txid: std::sync::atomic::AtomicU64::new(0),
            leaked_segments: std::sync::atomic::AtomicU64::new(0),
            retention: Retention::new(options.max_retained_generations),
            data_physical_bytes: std::sync::atomic::AtomicU64::new(0),
            data_live_pages: std::sync::atomic::AtomicU64::new(0),
            reclaimed_bytes: std::sync::atomic::AtomicU64::new(0),
            generations_created: std::sync::atomic::AtomicU64::new(1),
            max_command_bytes: options.limits.max_command_bytes,
            max_data_bytes: options.max_data_bytes,
            checkpoints_skipped_space: std::sync::atomic::AtomicU64::new(0),
            namespace_lock: parking_lot::Mutex::new(None),
        });
        // A command channel is required by the type, but with no writer nothing
        // is ever sent (admission is parked). Drop the receiver.
        let (tx, _rx) = async_channel::unbounded::<Command>();
        Store {
            inner: Arc::new(StoreInner {
                core,
                commands: tx,
                writer: parking_lot::Mutex::new(None),
                closing: AtomicBool::new(false),
                _backend: PhantomData,
            }),
        }
    }

    /// The open incarnation.
    pub fn incarnation(&self) -> Incarnation {
        self.inner.core.incarnation
    }

    /// A metrics snapshot.
    pub fn metrics(&self) -> Metrics {
        self.inner.core.metrics()
    }

    /// EXPERIMENT (T2.2): toggles overflow-chain prefetch on this
    /// store's page cache. Off by default; not a stability contract. Opt in
    /// only for genuinely cold, non-repeating large-value reads: prefetched
    /// links bypass the page cache (a re-read of a just-prefetched value is
    /// cold again — measured 12–100× slower than a warm re-read), and
    /// cancelling an enabled read mid-flight can transiently exceed the
    /// speculative device in-flight bound. See
    /// [`PageCache::set_chain_prefetch`](crate::cache::PageCache::set_chain_prefetch).
    #[doc(hidden)]
    pub fn set_chain_prefetch(&self, enabled: bool) {
        self.inner.core.cache.set_chain_prefetch(enabled);
    }

    /// EXPERIMENT (T2.2): chain-prefetch counter snapshot.
    #[doc(hidden)]
    pub fn chain_prefetch_stats(&self) -> crate::cache::ChainPrefetchStats {
        self.inner.core.cache.chain_prefetch_stats()
    }

    /// EXPERIMENT (T2.2): the page cache's counter snapshot, for per-arm
    /// physical-load accounting in the T2.2 benchmark.
    #[doc(hidden)]
    pub fn cache_metrics(&self) -> crate::cache::CacheMetrics {
        self.inner.core.cache.metrics()
    }

    /// Enters maintenance: the writer parks between commands and fresh admission
    /// fails fast with `Compacting`. Already-queued reservations stay charged.
    /// Returns `false` if the store is not `Running`.
    ///
    /// This flips the phase and returns; it does **not** wait for a writer
    /// acknowledgement. No command the writer receives after the flip is applied
    /// (`writer::defer_if_paused`), but one already dispatched — or a flush
    /// already running — can still complete after this returns. Note [`Store::compact`] and
    /// [`Store::checkpoint`] do **not** use this — they run as writer commands;
    /// this is the operator-controlled pause (e.g. for external backup).
    ///
    /// Dropping every handle while paused is safe: the parked writer
    /// observes the command channel close, applies whatever was admitted,
    /// flushes, exits, and releases the sidecar lock — see
    /// `tests_handle_drop_lifecycle`.
    pub fn begin_maintenance(&self) -> bool {
        self.inner.core.state.enter_maintenance()
    }

    /// Resumes from maintenance, waking parked writer and blocked admission.
    pub fn end_maintenance(&self) {
        self.inner.core.state.resume()
    }

    /// Runs a data-file compaction (M6 Phase D): rebuild a dense data file
    /// containing only the pages reachable from the committed root — dropping
    /// the superseded CoW ancestors Direct accretes and the orphan checkpoint
    /// extents StoreWal strands — and atomically promote it over the data path
    /// (rename + directory fsync under the sidecar lock). For StoreWal the
    /// rotation and header protocol are the checkpoint's own, so a compaction
    /// also *is* a checkpoint (records ≤ T are superseded; the WAL rotates).
    ///
    /// Runs **on the writer thread**, as a command: it takes effect after every
    /// write admitted before this call and before every write admitted after
    /// it. Reads never block; a [`Snapshot`] captured beforehand keeps reading
    /// the superseded file's inode until it drops — or until
    /// [`Options::max_retained_generations`] revokes it (M6 Phase E).
    ///
    /// [`CheckpointError::Unsupported`](crate::error::CheckpointError) if the
    /// store owns no data path (`create_with_io` / `open_over` — a bare device
    /// cannot be renamed over); [`CheckpointError::Io`] is a **clean abort**
    /// before the cutover (nothing changed; the store runs on).
    pub async fn compact(&self) -> Result<crate::compact::CompactReport, CheckpointError> {
        let (tx, rx) = oneshot::channel();
        // Decide and enqueue under one lock, exactly as `checkpoint` does: a
        // parked writer dequeues nothing, so a compact command that landed in
        // the queue during maintenance would wait on an operator.
        self.inner
            .core
            .state
            .with_running_intake(|decision| match decision {
                Ok(()) => {
                    if let Err(err) = self.inner.commands.try_send(Command::Compact { reply: tx }) {
                        let Command::Compact { reply } = err.into_inner() else {
                            unreachable!("sent a Compact")
                        };
                        let _ = reply.send(Err(self.inner.terminal_checkpoint_error()));
                    }
                }
                Err(WriteError::Compacting) => {
                    let _ = tx.send(Err(CheckpointError::Compacting));
                }
                Err(WriteError::Poisoned(r)) => {
                    let _ = tx.send(Err(CheckpointError::Poisoned(r)));
                }
                Err(_) => {
                    let _ = tx.send(Err(CheckpointError::Closed));
                }
            });
        rx.await
            .unwrap_or_else(|_| Err(self.inner.terminal_checkpoint_error()))
    }

    /// Reserves admission capacity, awaiting if the store is momentarily full.
    /// Fails fast on maintenance (`Compacting`) and terminal states.
    pub fn reserve_write(
        &self,
        cost: WriteCost,
    ) -> impl std::future::Future<Output = Result<crate::map::WritePermit<B>, WriteError>> + Send
    {
        let inner = Arc::clone(&self.inner);
        async move {
            // StoreWal: reserve WAL bytes (fail-fast `StoreFull`, invariant 18)
            // *before* the dirty budget. Held in `wal_res` across the awaiting
            // dirty reservation so a cancellation of that await drops and releases
            // both charges; attached on success so drop/reconcile stays unified.
            let wal_res = reserve_wal_bytes(&inner.core, &cost)?;
            let mut reservation = inner.core.budget.reserve(cost, &inner.core.state).await?;
            if let Some(w) = wal_res {
                reservation.attach_wal(w);
            }
            Ok(crate::map::WritePermit::new(inner, reservation, cost))
        }
    }

    /// Non-blocking reservation. `WriteError::Overloaded` means retry later.
    pub fn try_reserve_write(
        &self,
        cost: WriteCost,
    ) -> Result<crate::map::WritePermit<B>, WriteError> {
        self.inner.core.state.admission_gate()?;
        let wal_res = reserve_wal_bytes(&self.inner.core, &cost)?;
        let mut reservation = self.inner.core.budget.try_reserve(cost)?;
        if let Some(w) = wal_res {
            reservation.attach_wal(w);
        }
        Ok(crate::map::WritePermit::new(
            Arc::clone(&self.inner),
            reservation,
            cost,
        ))
    }

    /// Stops intake, drains accepted writes, and closes. Idempotent-ish: a
    /// second concurrent caller returns `AlreadyClosed`.
    pub async fn close(&self) -> Result<(), crate::error::CloseError> {
        use crate::error::CloseError;
        if self.inner.closing.swap(true, Ordering::SeqCst) {
            return Err(CloseError::AlreadyClosed);
        }
        // Shut intake and let the writer drain remaining commands, then finalize.
        self.inner.core.state.begin_close();
        // A read-only fixture store (`build_readonly`) has no writer to drive the
        // drain→`Closed` transition, so finalize it directly.
        if self.inner.writer.lock().is_none() {
            self.inner.core.state.finish_close();
        }
        // Wait until the writer reaches a terminal phase. Capture whether it was
        // a poison (a failed close-flush) rather than returning early, so the
        // writer is still joined and backend resources still released on the
        // poisoned path too: a store the operator closed — even one
        // that poisoned — must free its sidecar lock so the path can be reopened
        // for recovery.
        let poison = loop {
            use crate::state::Lifecycle;
            match self.inner.core.state.phase() {
                Lifecycle::Closed => break None,
                Lifecycle::Poisoned(r) => break Some(r),
                _ => {}
            }
            let listener = self.inner.core.state.listen();
            match self.inner.core.state.phase() {
                Lifecycle::Closed => break None,
                Lifecycle::Poisoned(r) => break Some(r),
                _ => {}
            }
            listener.await;
        };
        // Join the (now-finished) writer thread. It has already reached a
        // terminal phase (`Closed`, or `Poisoned` after `fail_drain`), so this
        // returns promptly. "Don't block the executor on join" is deferred to
        // the M1/M3 backend coordinator.
        if let Some(handle) = self.inner.writer.lock().take() {
            let _ = handle.join();
        }
        // Backend resources (the real-fs sidecar lock) were released by the
        // writer thread itself as its final act — on the clean and poisoned
        // paths alike — so after the join above the path is
        // free. A readonly fixture has neither writer nor lock.
        match poison {
            Some(r) => Err(CloseError::Poisoned(r)),
            None => Ok(()),
        }
    }
}

impl Store<Direct> {
    /// Creates a fresh StoreDirect over a caller-provided data device. The
    /// writer lays down the base header + slot B durably at startup. For a
    /// real-file store with the sidecar-lock / directory-fsync protocol, use
    /// [`create_path`](Self::create_path).
    pub fn create_with_io(
        data: Arc<dyn PageIo>,
        options: Options,
    ) -> Result<Store<Direct>, OpenError> {
        Self::create_with_io_hosted(data, None, options, None)
    }

    /// [`create_with_io`](Self::create_with_io) for a store that owns its data
    /// path: `data_host` lets it compact (M6 Phase D). Used by `create_path`
    /// and the in-memory test hosts.
    pub(crate) fn create_with_io_hosted(
        data: Arc<dyn PageIo>,
        data_host: Option<Arc<dyn crate::compact::DataFileHost>>,
        options: Options,
        namespace_lock: Option<std::fs::File>,
    ) -> Result<Store<Direct>, OpenError> {
        let store_uuid = crate::direct::generate_store_uuid();
        // Durable create: write + fdatasync the base header *before* returning, so
        // `durable_txid = 0` is truthful and an immediate device error surfaces
        // here rather than after acknowledgement.
        crate::direct::write_base_header_blocking(&data, store_uuid)?;
        let (file, alloc) = direct_generation(data, FIRST_DATA_INDEX, 0);
        Ok(Store::build_bound(
            file,
            alloc,
            BackendSetup::Direct {
                store_uuid,
                data_host,
                data_bytes: FIRST_DATA_INDEX * crate::page::PAGE_SIZE as u64,
                stale_slot: crate::direct::DirectSlot::B,
            },
            PageRef::EMPTY,
            0,
            0,
            &options,
            namespace_lock,
        ))
    }

    /// Reopens a StoreDirect over a device whose page 0 holds a valid direct
    /// header. Validates the header (any defect → `Corrupt`, never heuristic
    /// repair), seeds the allocator from the logical tail, and publishes the
    /// header's root. The visible root continues at the header's txid under a
    /// fresh incarnation (invariant 11; the persisted header field never
    /// regresses across opens).
    pub fn open_over(data: Arc<dyn PageIo>, options: Options) -> Result<Store<Direct>, OpenError> {
        Self::open_over_hosted(data, None, options, None)
    }

    /// [`open_over`](Self::open_over) for a store that owns its data path.
    pub(crate) fn open_over_hosted(
        data: Arc<dyn PageIo>,
        data_host: Option<Arc<dyn crate::compact::DataFileHost>>,
        options: Options,
        namespace_lock: Option<std::fs::File>,
    ) -> Result<Store<Direct>, OpenError> {
        // v4 (T1.1): a Direct store has TWO alternating header slots.
        // Read both (each may independently be torn by a crashed flush — a
        // failed slot read is slot damage, not an open error) and pick the
        // winner; the loser is the slot the next flush overwrites. A file too
        // short for both slots cannot be a v4 store.
        let size = futures_lite::future::block_on(data.size())?;
        if size < FIRST_DATA_INDEX * PAGE_SIZE as u64 {
            return Err(OpenError::Corrupt(
                "direct store shorter than its two header slots",
            ));
        }
        let slot_a = futures_lite::future::block_on(
            data.read_exact_at(crate::direct::DirectSlot::A.offset(), PAGE_SIZE),
        );
        let slot_b = futures_lite::future::block_on(
            data.read_exact_at(crate::direct::DirectSlot::B.offset(), PAGE_SIZE),
        );
        let (header, stale_slot) =
            crate::direct::pick_winning_header(slot_a.as_deref(), slot_b.as_deref())?;
        // `logical_tail` may legitimately exceed the device size: ids burned by
        // intra-batch superseded pages advance the persisted frontier without a
        // byte ever being written at their offsets (see `reopen_emptied_store`),
        // and the frontier must be honored in full or the reopened allocator
        // would re-hand-out burned ids. So a `tail <= size` open check
        // is unsound and deliberately absent: a truncated file fails
        // instead through what it can't serve — `verify_tree_on_open` (default)
        // or the first read's checksum — while a crafted-but-consistent large
        // tail merely yields a sparse file, bounded by the codec's headroom cap.
        // Ids live in the shifted space past the header's delta (M6 Phase D).
        let next_id = header.id_delta + header.logical_tail / PAGE_SIZE as u64;
        let (file, alloc) = direct_generation(Arc::clone(&data), next_id, header.id_delta);
        // Optional authoritative validation (gated by
        // `verify_tree_on_open`): structurally walk the root and confirm the
        // *actual* entry count matches the header before enabling writes. A
        // checksum-valid header whose count disagrees with the tree would
        // otherwise drive wrong `len()` and, worse, a premature empty collapse
        // that discards live keys. Any structural defect → `Corrupt`. This reads
        // every page once (O(store)); disable it for a fast header-only open of a
        // large trusted store (reads still validate pages lazily on demand).
        if options.verify_tree_on_open {
            let verified = futures_lite::future::block_on(async {
                let cache = make_cache(&options);
                crate::read::tree_verify(&cache, &file, header.root).await
            });
            match verified {
                Ok(report) if report.entry_count == header.entry_count => {}
                Ok(_) => {
                    return Err(OpenError::Corrupt(
                        "direct header entry count disagrees with tree",
                    ))
                }
                Err(crate::error::ReadError::Corrupt(what)) => {
                    return Err(OpenError::Corrupt(what))
                }
                Err(crate::error::ReadError::Io(e)) => return Err(OpenError::Io(e)),
                Err(_) => return Err(OpenError::Corrupt("root verification failed")),
            }
        }
        Ok(Store::build_bound(
            file,
            alloc,
            BackendSetup::Direct {
                store_uuid: header.store_uuid,
                data_host,
                data_bytes: header.logical_tail,
                stale_slot,
            },
            header.root,
            header.entry_count,
            header.header_txid,
            &options,
            namespace_lock,
        ))
    }

    /// Opens a store whose visible root is a prebuilt read-only page tree
    /// ([`TreeBuilder`](crate::read::TreeBuilder)). This is the M2 read-path
    /// fixture that lets `BTreeMap`/`Snapshot` traverse real pages without a
    /// writer; it does not run the durable open protocol.
    #[doc(hidden)]
    pub fn open_readonly_tree(built: crate::read::BuiltTree, options: Options) -> Store<Direct> {
        Store::build_readonly(built.file, built.root, built.entry_count, &options)
    }
}

impl Store<Wal> {
    /// Creates a fresh StoreWal over caller-provided data + WAL devices. Writes
    /// the seq-0 checkpoint data header and WAL segment 0's identity header (same
    /// `store_uuid`) and `fdatasync`s both before returning, so an immediate
    /// device error surfaces here and a crash right after create reopens to an
    /// empty store. `wal` **is** segment 0: a caller-supplied device cannot be
    /// rotated, so a store created this way cannot checkpoint. For a
    /// real-file store with the sidecar-lock / directory-fsync protocol — and
    /// checkpoint support — use [`create_path`](Self::create_path).
    pub fn create_with_io(
        data: Arc<dyn PageIo>,
        wal: Arc<dyn PageIo>,
        options: Options,
    ) -> Result<Store<Wal>, OpenError> {
        let store_uuid = crate::direct::generate_store_uuid();
        crate::wal::write_headers_blocking(&data, &wal, store_uuid)?;
        Ok(Store::finish_create(
            data, wal, store_uuid, None, None, options, None,
        ))
    }

    /// Creates a fresh StoreWal whose WAL segment 0 was minted by `file_set` (its
    /// identity header is already durable), handing the store the namespace so it
    /// **can** checkpoint. Used by the path-backed `create_path` and by the
    /// in-memory [`create_mem_store`](crate::fileset::create_mem_store).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_with_io_owned(
        data: Arc<dyn PageIo>,
        wal: Arc<dyn PageIo>,
        store_uuid: [u8; 16],
        file_set: Option<Arc<dyn crate::fileset::WalFileSet>>,
        data_host: Option<Arc<dyn crate::compact::DataFileHost>>,
        options: Options,
        namespace_lock: Option<std::fs::File>,
    ) -> Result<Store<Wal>, OpenError> {
        crate::wal::write_data_header_blocking(&data, store_uuid)?;
        Ok(Store::finish_create(
            data,
            wal,
            store_uuid,
            file_set,
            data_host,
            options,
            namespace_lock,
        ))
    }

    /// The shared tail of both create paths: both headers are durable by now, so
    /// this is pure in-memory construction over the empty seq-0 state.
    #[allow(clippy::too_many_arguments)]
    fn finish_create(
        data: Arc<dyn PageIo>,
        wal: Arc<dyn PageIo>,
        store_uuid: [u8; 16],
        file_set: Option<Arc<dyn crate::fileset::WalFileSet>>,
        data_host: Option<Arc<dyn crate::compact::DataFileHost>>,
        options: Options,
        namespace_lock: Option<std::fs::File>,
    ) -> Store<Wal> {
        let binding = crate::wal::WalBinding {
            data: Arc::clone(&data),
            store_uuid,
            file_set,
            data_host,
            segment_seq: 0,
            checkpoint_seq: 0,
            // Both header slots are laid down, so the first relocated page lands
            // at the first data page.
            data_frontier: FIRST_DATA_INDEX * PAGE_SIZE as u64,
        };
        let (file, alloc, locator) = wal_generation(data, wal, FIRST_DATA_INDEX);
        Store::build_bound(
            file,
            alloc,
            BackendSetup::Wal {
                locator,
                binding,
                append_offset: crate::wal::WAL_HEADER_LEN,
                chain_head: crate::wal::seed_checksum(store_uuid, 0),
                max_wal_bytes: options.max_wal_bytes,
                open_scan_ms: 0,
                open_scan_bytes: 0,
                physical_bytes: crate::wal::WAL_HEADER_LEN,
                checkpoint_txid: 0,
                // A fresh store's data file is its two header slots.
                data_bytes: FIRST_DATA_INDEX * PAGE_SIZE as u64,
            },
            PageRef::EMPTY,
            0,
            0,
            &options,
            namespace_lock,
        )
    }

    /// Runs a checkpoint: relocate the live page set into the data file, rotate to
    /// a fresh WAL segment, and unlink the retired one — bounding WAL bytes and
    /// reopen cost, which otherwise grow with every write ever made.
    ///
    /// The checkpoint runs **on the writer thread**, as a command: it takes effect
    /// after every write admitted before this call and before every write admitted
    /// after it, with no window in which a concurrent batch could be half-included.
    /// Reads never block; a [`Snapshot`] captured beforehand stays readable against
    /// the retired segment until it drops (invariant 1/8).
    ///
    /// [`CheckpointError::Unsupported`](crate::error::CheckpointError::Unsupported)
    /// if the store owns no segment namespace (`create_with_io` / `open_over` —
    /// see [`crate::fileset`]).
    pub async fn checkpoint(&self) -> Result<crate::wal::CheckpointReport, CheckpointError> {
        let (tx, rx) = oneshot::channel();
        // Decide and enqueue under one lock: a parked writer dequeues nothing, so a
        // checkpoint that landed in the queue during maintenance would wait on an
        // operator rather than on I/O. Checking the phase and *then* sending would
        // leave exactly that window open.
        self.inner
            .core
            .state
            .with_running_intake(|decision| match decision {
                Ok(()) => {
                    if let Err(err) = self
                        .inner
                        .commands
                        .try_send(Command::Checkpoint { reply: tx })
                    {
                        let Command::Checkpoint { reply } = err.into_inner() else {
                            unreachable!("sent a Checkpoint")
                        };
                        let _ = reply.send(Err(self.inner.terminal_checkpoint_error()));
                    }
                }
                Err(WriteError::Compacting) => {
                    let _ = tx.send(Err(CheckpointError::Compacting));
                }
                Err(WriteError::Poisoned(r)) => {
                    let _ = tx.send(Err(CheckpointError::Poisoned(r)));
                }
                Err(_) => {
                    let _ = tx.send(Err(CheckpointError::Closed));
                }
            });
        rx.await
            .unwrap_or_else(|_| Err(self.inner.terminal_checkpoint_error()))
    }

    /// Reopens a StoreWal over paired data + WAL devices, running crash recovery:
    /// validates both headers (UUID pairing), scans the WAL for the longest valid
    /// committed prefix, rebuilds the overlay/root, and appends at the accepted-
    /// prefix offset (never after a torn tail, never truncating). The visible root
    /// continues at the recovered txid under a **fresh incarnation** (invariant
    /// 11). Any header defect → `Corrupt`; a data/WAL UUID mismatch → `UuidMismatch`.
    pub fn open_over(
        data: Arc<dyn PageIo>,
        wal: Arc<dyn PageIo>,
        options: Options,
    ) -> Result<Store<Wal>, OpenError> {
        Store::<Wal>::open_owned(data, wal, None, None, options, None)
    }

    /// [`open_over`](Self::open_over) for a store that owns its segment namespace:
    /// `wal` must be the segment the data header names (the caller resolved it
    /// *from* that header), and `file_set` lets the reopened store checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open_owned(
        data: Arc<dyn PageIo>,
        wal: Arc<dyn PageIo>,
        file_set: Option<Arc<dyn crate::fileset::WalFileSet>>,
        data_host: Option<Arc<dyn crate::compact::DataFileHost>>,
        options: Options,
        namespace_lock: Option<std::fs::File>,
    ) -> Result<Store<Wal>, OpenError> {
        let scan_start = std::time::Instant::now();
        let rec = crate::wal::recover(&data, &wal)?;
        let open_scan_ms = scan_start.elapsed().as_millis() as u64;
        // The data file only grows within an open, so its current length is past
        // every extent either header slot references — including the orphans a
        // crashed checkpoint may have left. Relocation appends from there.
        let data_size = futures_lite::future::block_on(data.size())?;
        let binding = crate::wal::WalBinding {
            data: Arc::clone(&data),
            store_uuid: rec.store_uuid,
            file_set,
            data_host,
            segment_seq: rec.wal_segment_seq,
            checkpoint_seq: rec.checkpoint_seq,
            data_frontier: data_size
                .next_multiple_of(PAGE_SIZE as u64)
                .max(FIRST_DATA_INDEX * PAGE_SIZE as u64),
        };
        // Build the generation over the locator the recovery scan populated, so
        // recovered and checkpointed pages resolve immediately.
        let file = FileGeneration::wal(data, wal, rec.locator.clone());
        let alloc = Allocator::for_wal(rec.next_page_id);
        Ok(Store::build_bound(
            file,
            alloc,
            BackendSetup::Wal {
                locator: rec.locator,
                binding,
                append_offset: rec.append_offset,
                chain_head: rec.chain_head,
                max_wal_bytes: options.max_wal_bytes,
                open_scan_ms,
                // The recovered logical frontier (WAL header + accepted committed
                // records) — exactly the bytes recovery accepted, excluding any
                // rejected torn tail. `physical_bytes` keeps the full file size,
                // which may exceed this after a torn-tail overwrite.
                open_scan_bytes: rec.append_offset,
                physical_bytes: rec.scanned_bytes,
                checkpoint_txid: rec.checkpoint_txid,
                // The real EOF, *not* `binding.data_frontier` — that is this value
                // rounded up to a page for appends, and seeding the metric from it
                // would make an unaligned store report one size before a restart
                // and a larger one after.
                data_bytes: data_size,
            },
            rec.root,
            rec.entry_count,
            rec.last_txid,
            &options,
            namespace_lock,
        ))
    }
}

/// A snapshot of the store's applied state at capture time. It stays readable
/// across later cutovers and evictions (invariant 1) — up to the retention
/// bound: a snapshot does **not** hold the generation's files irrevocably (M6
/// Phase E). It holds a counted [`GenHold`] on the generation's revocable
/// holder, plus copies of the descriptor's plain fields; each read takes a
/// short-lived lease for its traversal. If the retention policy expires the
/// generation (`Options::max_retained_generations`), later reads return
/// [`SnapshotExpired`](crate::error::ReadError::SnapshotExpired) while
/// `version`/`len` (plain copies) keep answering.
pub struct Snapshot<B: Backend> {
    version: Version,
    root: PageRef,
    entry_count: u64,
    hold: GenHold,
    cache: Arc<crate::cache::PageCache>,
    _backend: PhantomData<B>,
}

impl<B: Backend> Clone for Snapshot<B> {
    fn clone(&self) -> Self {
        Snapshot {
            version: self.version,
            root: self.root,
            entry_count: self.entry_count,
            hold: self.hold.clone(),
            cache: Arc::clone(&self.cache),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Snapshot<B> {
    /// Captures the currently visible state. Load–hold–recheck: `try_hold`
    /// refuses a generation whose rebind race already released it, in which
    /// case the reload observes the *new* descriptor — so a capture can never
    /// bind to a generation it cannot read, and the loop terminates.
    pub(crate) fn capture(core: &Core) -> Self {
        loop {
            let d = core.root.load_full();
            if let Some(hold) = d.retain.try_hold() {
                return Snapshot {
                    version: d.version,
                    root: d.root,
                    entry_count: d.entry_count,
                    hold,
                    cache: Arc::clone(&core.cache),
                    _backend: PhantomData,
                };
            }
        }
    }

    /// The version captured by this snapshot.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Point read against the captured root.
    ///
    /// The future holds its own [`GenHold`] (taken synchronously at call), so it
    /// stays valid even if the `Snapshot` is dropped first. At first poll it
    /// acquires the read lease; a lease acquired even one instant before a
    /// revocation keeps its device for the whole traversal (invariant 8), and a
    /// lease refused yields `SnapshotExpired`.
    pub fn get(
        &self,
        key: Vec<u8>,
    ) -> impl std::future::Future<Output = Result<Option<Vec<u8>>, crate::error::ReadError>> + Send
    {
        let hold = self.hold.clone();
        let cache = Arc::clone(&self.cache);
        let root = self.root;
        async move {
            let lease = hold.gen().lease()?;
            let out = crate::read::tree_get(&cache, lease.file(), root, &key).await;
            drop(lease);
            out
        }
    }

    /// In-order range scan against the captured root, delivering each entry to
    /// `consumer` as borrowed `(key, value)` slices in ascending key order
    /// (overflow values reassembled first). The consumer returns
    /// [`ControlFlow::Continue`](std::ops::ControlFlow::Continue) to keep
    /// going or `Break` to end the scan early.
    ///
    /// Like [`get`](Self::get), the future holds its own [`GenHold`] and takes
    /// one read lease for the whole scan at first poll — a lease acquired even
    /// an instant before a revocation keeps its device for the entire
    /// traversal, and a lease refused yields `SnapshotExpired`.
    pub fn for_each_range<K, R, F>(
        &self,
        range: R,
        mut consumer: F,
    ) -> impl std::future::Future<Output = Result<crate::read::ScanSummary, crate::error::ReadError>>
           + Send
    where
        K: AsRef<[u8]>,
        R: std::ops::RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
    {
        let hold = self.hold.clone();
        let cache = Arc::clone(&self.cache);
        let root = self.root;
        let (start, end) = crate::read::own_bounds(&range);
        async move {
            let lease = hold.gen().lease()?;
            let out = crate::read::tree_scan_range(
                &cache,
                lease.file(),
                root,
                crate::read::slice_bound(&start),
                crate::read::slice_bound(&end),
                &mut consumer,
            )
            .await;
            drop(lease);
            out
        }
    }

    /// The descending mirror of [`for_each_range`](Self::for_each_range):
    /// same bounds semantics, same one-lease-for-the-whole-scan contract,
    /// entries delivered in **descending** key order.
    pub fn for_each_range_rev<K, R, F>(
        &self,
        range: R,
        mut consumer: F,
    ) -> impl std::future::Future<Output = Result<crate::read::ScanSummary, crate::error::ReadError>>
           + Send
    where
        K: AsRef<[u8]>,
        R: std::ops::RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
    {
        let hold = self.hold.clone();
        let cache = Arc::clone(&self.cache);
        let root = self.root;
        let (start, end) = crate::read::own_bounds(&range);
        async move {
            let lease = hold.gen().lease()?;
            let out = crate::read::tree_scan_range_rev(
                &cache,
                lease.file(),
                root,
                crate::read::slice_bound(&start),
                crate::read::slice_bound(&end),
                &mut consumer,
            )
            .await;
            drop(lease);
            out
        }
    }

    /// The entry with the smallest key, or `None` on an empty snapshot — a
    /// single edge descent, no scan. (Vocabulary aligned with
    /// mapdb-collections' `TreeMap::first_entry`.)
    pub fn first_entry(
        &self,
    ) -> impl std::future::Future<Output = Result<Option<crate::read::Entry>, crate::error::ReadError>>
           + Send {
        self.edge(false)
    }

    /// The entry with the largest key, or `None` on an empty snapshot.
    pub fn last_entry(
        &self,
    ) -> impl std::future::Future<Output = Result<Option<crate::read::Entry>, crate::error::ReadError>>
           + Send {
        self.edge(true)
    }

    /// The greatest entry with key `<= key` (NavigableMap `floor`).
    pub fn floor_entry(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl std::future::Future<Output = Result<Option<crate::read::Entry>, crate::error::ReadError>>
           + Send {
        self.nav(key.as_ref().to_vec(), true, false)
    }

    /// The greatest entry with key `< key` (NavigableMap `lower`).
    pub fn lower_entry(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl std::future::Future<Output = Result<Option<crate::read::Entry>, crate::error::ReadError>>
           + Send {
        self.nav(key.as_ref().to_vec(), true, true)
    }

    /// The least entry with key `>= key` (NavigableMap `ceiling`).
    pub fn ceiling_entry(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl std::future::Future<Output = Result<Option<crate::read::Entry>, crate::error::ReadError>>
           + Send {
        self.nav(key.as_ref().to_vec(), false, false)
    }

    /// The least entry with key `> key` (NavigableMap `higher`).
    pub fn higher_entry(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl std::future::Future<Output = Result<Option<crate::read::Entry>, crate::error::ReadError>>
           + Send {
        self.nav(key.as_ref().to_vec(), false, true)
    }

    fn edge(
        &self,
        last: bool,
    ) -> impl std::future::Future<Output = Result<Option<crate::read::Entry>, crate::error::ReadError>>
           + Send {
        let hold = self.hold.clone();
        let cache = Arc::clone(&self.cache);
        let root = self.root;
        async move {
            let lease = hold.gen().lease()?;
            let out = crate::read::tree_edge(&cache, lease.file(), root, last).await;
            drop(lease);
            out
        }
    }

    /// Shared navigation plumbing: one lease per lookup, exactly like
    /// [`get`](Self::get). `downward` routes to the floor/lower descent,
    /// upward to the ceiling/higher bounded scan; `strict` excludes equality.
    fn nav(
        &self,
        key: Vec<u8>,
        downward: bool,
        strict: bool,
    ) -> impl std::future::Future<Output = Result<Option<crate::read::Entry>, crate::error::ReadError>>
           + Send {
        let hold = self.hold.clone();
        let cache = Arc::clone(&self.cache);
        let root = self.root;
        async move {
            let lease = hold.gen().lease()?;
            let out = if downward {
                crate::read::tree_floor(&cache, lease.file(), root, &key, strict).await
            } else {
                crate::read::tree_ceiling(&cache, lease.file(), root, &key, strict).await
            };
            drop(lease);
            out
        }
    }

    /// Entry count at capture time. `len`/`is_empty` stay sync — plain copies,
    /// answering even after expiration.
    pub fn len(&self) -> u64 {
        self.entry_count
    }

    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Whether the retention policy has expired this snapshot's generation:
    /// reads now return `SnapshotExpired`. Point-in-time (a concurrent
    /// checkpoint can expire it right after `false`), so a reader must still
    /// handle the error; useful for fail-fast and metrics.
    pub fn is_expired(&self) -> bool {
        self.hold.gen().is_expired()
    }
}

/// A convenience for the writer to build the next descriptor over a freshly
/// built root. Kept here so the descriptor shape stays private to this module.
pub(crate) fn next_descriptor(
    current: &RootDescriptor,
    root: PageRef,
    entry_count: u64,
) -> RootDescriptor {
    RootDescriptor {
        version: current.version.next(),
        root,
        file: Arc::clone(&current.file),
        entry_count,
        // Same generation → same revocable holder (one `SnapshotGen` per
        // generation; only a rebind mints a new one).
        retain: Arc::clone(&current.retain),
    }
}

/// Reserves the worst-case WAL bytes for `cost` on a StoreWal, or `Ok(None)` for
/// a StoreDirect. `StoreFull` (never `Overloaded`) if the reserved frontier would
/// exceed `max_wal_bytes` — a pre-publication result (invariant 18).
pub(crate) fn reserve_wal_bytes(
    core: &Core,
    cost: &WriteCost,
) -> Result<Option<WalReservation>, WriteError> {
    match &core.wal {
        Some(meter) => Ok(Some(meter.reserve(cost.wal_worst_case_bytes)?)),
        None => Ok(None),
    }
}

/// Builds the page cache honoring `options.cache_capacity`.
fn make_cache(options: &Options) -> crate::cache::PageCache {
    let cap = crate::cache::selected_clean_capacity(options.cache_capacity);
    crate::cache::PageCache::with_capacity(cap)
}

// Silence "field never read" for the frozen backend hand-off shape that M4's WAL
// coordinator consumes.
#[allow(dead_code)]
fn _assert_pagebatch_shape(_: PageBatch) {}
