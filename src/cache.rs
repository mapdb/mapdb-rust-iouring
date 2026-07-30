//! The engine-owned page cache: state machine, cancellation-independent
//! single-flight, and a sharded `quick_cache` clean tier whose warm hits are
//! served lock-free (the earlier custom CLOCK evictor is deleted).
//!
//! Keyed by [`PageId`]. A page id is never reused and a published page is
//! immutable, so a given id always decodes to the same image and the cache is
//! **generation-independent** (it lives in the shared `Core`, while the
//! per-generation `PageLocator` that says *where* the id sits lives in the
//! [`FileGeneration`](crate::backend::FileGeneration)).
//!
//! States: `Loading` (one physical read in flight), `Clean`
//! (evictable image — lives in the `quick_cache` tier, never the map), `Dirty`
//! (writer-produced, pinned until the durable frontier releases it), `Failed`
//! (a cached `Corrupt` reason; deterministic corruption must not re-issue I/O).
//! The map holds only `Loading`/`Dirty`/`Failed`; a clean page's *only* resident
//! home is the quick tier, so cross-tier exclusivity holds by construction.
//!
//! ## Cancellation-independent single-flight
//!
//! There is no global executor, so the load is driven cooperatively by whichever
//! caller is polling, with the *one* device future handed between them under a
//! mutex. Each poll uses a `Waker` built over a shared [`Event`] (via
//! [`std::task::Wake`]), so a device completion notifies **every** waiter rather
//! than one waiter's task. Dropping any waiter — including the one that issued
//! the load — leaves the future in the slot for the next waiter to continue.
//! Exactly one physical read is ever created for a cold page, no matter how many
//! callers race or which ones cancel.

use crate::backend::FileGeneration;
use crate::error::ReadError;
use crate::io::PageIo;
use crate::page::{Checksum, Page, PageId, PageRef, PAGE_SIZE};
use event_listener::{Event, EventListener};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::task::{Context, Poll, Wake, Waker};

/// Bound on resident negative (`Failed`) verdicts (T1.4). Deliberately
/// a constant, not an `Options` knob: nothing legitimate tunes it — the count is
/// a corruption symptom, not a workload parameter. 1024 verdicts ≈ tens of KB
/// including the FIFO ring; anti-thrash holds up to that many
/// concurrently cached distinct verdicts, far beyond any honest workload, and an
/// evicted verdict is merely re-derived by a re-read (cold path, correctness-safe).
const NEGATIVE_CAPACITY: usize = 1024;

/// Minimum and fallback clean-tier capacity in pages (16 KiB each → 16 MiB).
///
/// [`PageCache::new`] may select a larger memory-aware capacity; this constant
/// remains the lower bound and the fallback when Linux memory availability
/// cannot be detected.
pub const DEFAULT_CLEAN_CAPACITY: usize = 1024;

/// The memory-aware default never assigns more than 512 MiB of clean images to
/// one cache. Dirty pages and cache metadata consume memory outside this bound.
const MAX_DEFAULT_CLEAN_CAPACITY: usize = 32 * 1024;
/// A default cache may consume at most one eighth of memory currently available
/// to the process. The cgroup limit, when detectable, constrains host-wide
/// `MemAvailable` so a small container does not inherit its host's budget.
const DEFAULT_CLEAN_MEMORY_DIVISOR: u64 = 8;
/// Linux cgroup v1 represents an unlimited memory controller with a page-aligned
/// value near `i64::MAX`; no practical finite process limit reaches this bound.
const CGROUP_V1_UNLIMITED_THRESHOLD: u64 = 1 << 60;

/// Resolves an explicit capacity or derives the bounded memory-aware default.
/// `Some` is returned byte-for-byte so [`PageCache::with_capacity`]'s existing
/// minimum-of-one handling remains the only normalization of explicit input.
pub(crate) fn selected_clean_capacity(explicit: Option<usize>) -> usize {
    match explicit {
        Some(capacity) => capacity,
        None => resolve_clean_capacity(None, detected_available_memory_bytes()),
    }
}

fn resolve_clean_capacity(explicit: Option<usize>, available_bytes: Option<u64>) -> usize {
    if let Some(capacity) = explicit {
        return capacity;
    }
    let Some(available_bytes) = available_bytes else {
        return DEFAULT_CLEAN_CAPACITY;
    };
    let pages = available_bytes / DEFAULT_CLEAN_MEMORY_DIVISOR / PAGE_SIZE as u64;
    pages.clamp(
        DEFAULT_CLEAN_CAPACITY as u64,
        MAX_DEFAULT_CLEAN_CAPACITY as u64,
    ) as usize
}

/// The tighter of host `MemAvailable` and remaining cgroup memory is the
/// process's effective availability. Missing or unlimited controllers impose no
/// additional bound.
fn detected_available_memory_bytes() -> Option<u64> {
    match (proc_mem_available_bytes(), cgroup_available_memory_bytes()) {
        (Some(host), Some(cgroup)) => Some(host.min(cgroup)),
        (Some(host), None) => Some(host),
        (None, Some(cgroup)) => Some(cgroup),
        (None, None) => None,
    }
}

fn proc_mem_available_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_mem_available_bytes(&meminfo)
}

fn parse_mem_available_bytes(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        let mut fields = line.split_whitespace();
        if fields.next() != Some("MemAvailable:") {
            continue;
        }
        let kib = fields.next()?.parse::<u64>().ok()?;
        if fields.next() != Some("kB") {
            return None;
        }
        return kib.checked_mul(1024);
    }
    None
}

/// Reads both cgroup layouts without depending on either being mounted. The
/// current group and every visible ancestor are considered because an ancestor
/// may impose the effective hard limit while the leaf says `max`.
fn cgroup_available_memory_bytes() -> Option<u64> {
    let memberships = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mut available = None;
    for line in memberships.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(hierarchy), Some(controllers), Some(relative)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let candidate = if hierarchy == "0" && controllers.is_empty() {
            cgroup_hierarchy_headroom(
                std::path::Path::new("/sys/fs/cgroup"),
                relative,
                "memory.max",
                "memory.current",
                false,
            )
        } else if controllers.split(',').any(|name| name == "memory") {
            cgroup_hierarchy_headroom(
                std::path::Path::new("/sys/fs/cgroup/memory"),
                relative,
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
                true,
            )
        } else {
            None
        };
        if let Some(candidate) = candidate {
            available = Some(
                available
                    .map(|previous: u64| previous.min(candidate))
                    .unwrap_or(candidate),
            );
        }
    }
    available
}

fn cgroup_hierarchy_headroom(
    root: &std::path::Path,
    relative: &str,
    limit_file: &str,
    usage_file: &str,
    v1: bool,
) -> Option<u64> {
    let mut directory = root.to_path_buf();
    for component in std::path::Path::new(relative.trim_start_matches('/')).components() {
        match component {
            std::path::Component::Normal(name) => directory.push(name),
            std::path::Component::CurDir => {}
            _ => return None,
        }
    }

    let mut available = None;
    loop {
        if let Some(candidate) = cgroup_headroom(&directory, limit_file, usage_file, v1) {
            available = Some(
                available
                    .map(|previous: u64| previous.min(candidate))
                    .unwrap_or(candidate),
            );
        }
        if directory.as_path() == root || !directory.pop() {
            break;
        }
    }
    available
}

fn cgroup_headroom(
    directory: &std::path::Path,
    limit_file: &str,
    usage_file: &str,
    v1: bool,
) -> Option<u64> {
    let raw_limit = std::fs::read_to_string(directory.join(limit_file)).ok()?;
    if raw_limit.trim() == "max" {
        return None;
    }
    let limit = raw_limit.trim().parse::<u64>().ok()?;
    if v1 && limit >= CGROUP_V1_UNLIMITED_THRESHOLD {
        return None;
    }
    let usage = std::fs::read_to_string(directory.join(usage_file))
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    Some(limit.saturating_sub(usage))
}

/// The engine-owned clean tier: a sharded `quick_cache` whose warm-hit path never
/// touches the map lock; `Inner.map` holds only `Loading`/`Dirty`/`Failed`.
/// **Every `quick_cache` *mutation* happens under the map mutex**: the
/// lock-free `get` is a read; inserts/removes ride the same transition lock as the
/// map so cross-tier state can never tear.
type QuickTier = quick_cache::sync::Cache<
    PageId,
    Arc<Page>,
    quick_cache::UnitWeighter,
    quick_cache::DefaultHashBuilder,
    EvictCounter,
>;

/// A `quick_cache` [`Lifecycle`](quick_cache::Lifecycle) that counts capacity
/// evictions into a shared atomic and **does nothing else** — its `on_evict` runs
/// inside a `quick_cache` shard write lock, so touching anything that could take the
/// map mutex would invert the map→shard lock order and deadlock. Only an
/// atomic is safe here.
///
/// Two `on_evict` subtleties this count depends on:
///   * `remove` bypasses `on_evict`, so tier invalidations (`insert_dirty`) do not
///     inflate this count;
///   * a value **replacement** (`insert` onto an already-resident key) *does* fire
///     `on_evict` for the displaced value. That would count a non-capacity event as
///     an eviction — but it is **unreachable while cross-tier exclusivity holds**
///     (qc.insert only ever runs for an id proven absent from qc under the map
///     lock), so it never happens. (Note: `quick_cache`'s doc caveat
///     "value replacement won't call this method" is on `before_evict`, *not*
///     `on_evict` — do not read it as a guarantee here.)
#[derive(Clone)]
struct EvictCounter {
    evictions: Arc<AtomicU64>,
}

impl quick_cache::Lifecycle<PageId, Arc<Page>> for EvictCounter {
    type RequestState = ();
    fn begin_request(&self) -> Self::RequestState {}
    fn on_evict(&self, _state: &mut (), _key: PageId, _val: Arc<Page>) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }
}

/// A boxed, `Send`, `'static` device read future. Owns its device `Arc`, so it
/// can be handed between drivers and outlive any single caller.
type LoadFuture = Pin<Box<dyn Future<Output = io::Result<Box<[u8]>>> + Send>>;

/// Observable cache state of one page id (for tests/metrics).
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum CacheState {
    Loading,
    Clean,
    Dirty,
    Failed,
}

/// A live cache slot (map-resident states only — a clean image lives in the
/// quick tier, never here).
enum Slot {
    /// A physical read is in flight; waiters share `LoadState`.
    Loading(Arc<LoadState>),
    /// A writer-produced image pinned until the durable frontier (M3).
    Dirty(Arc<Page>),
    /// A cached structural-corruption verdict, keyed by the bytes' **actual** crc.
    /// A re-get whose ref checksum equals `actual_crc` is the authoritative reader of
    /// these (structurally corrupt) bytes and gets `reason` without I/O; a re-get with a
    /// *different* checksum is — under the never-reuse invariant (one durable image, one
    /// true crc per id) — permanently a checksum mismatch and gets that verdict, also
    /// without I/O. Keying by crc (H7) is what lets a `Failed` verdict answer *every* ref
    /// correctly instead of poisoning a good ref to the same healthy-or-differently-bad id.
    Failed {
        actual_crc: Checksum,
        reason: &'static str,
    },
}

/// The reference-independent outcome of one physical read, shared by all single-flight
/// participants. The read is decoded **once** against the bytes' own crc (so the decode
/// does not depend on any participant's ref), then each participant projects this through
/// its own `PageRef` in [`project_shared`]. This is the H7 fix: the shared unit is the
/// physical read, never one participant's verdict.
#[derive(Clone)]
enum SharedRead {
    /// A full read that decoded to a structurally valid page (its `checksum()` is the
    /// bytes' actual crc). A participant whose ref checksum matches gets it; any other
    /// ref fails `validate_hit`.
    Decoded(Arc<Page>),
    /// A full read whose bytes are structurally corrupt. `actual_crc` is the bytes' crc;
    /// the 16 KiB image is dropped. An authoritative ref (`expected == actual_crc`) gets
    /// `reason`; any other ref gets a checksum mismatch.
    Malformed {
        actual_crc: Checksum,
        reason: &'static str,
    },
    /// A transient physical read fault (I/O error or short read). Reference-neutral:
    /// shared verbatim to every participant, cached by no one, retried by a later get.
    ReadErr(ReadError),
}

/// Projects a shared physical-read outcome through one participant's own `PageRef`,
/// restoring the H7 invariant that a participant only ever accepts a result that
/// validates against *its own* ref (`page_id` **and** `expected_checksum`) — exactly the
/// guarantee a clean/dirty hit gives via [`validate_hit`].
fn project_shared(shared: &SharedRead, page_ref: PageRef) -> Result<Arc<Page>, ReadError> {
    match shared {
        SharedRead::Decoded(page) => validate_hit(page.clone(), page_ref),
        SharedRead::Malformed { actual_crc, reason } => {
            if page_ref.expected_checksum == *actual_crc {
                Err(ReadError::Corrupt(reason))
            } else {
                Err(ReadError::Corrupt("page checksum mismatch"))
            }
        }
        SharedRead::ReadErr(e) => Err(e.clone()),
    }
}

/// Shared single-flight state for one in-flight load.
struct LoadState {
    /// The page this load resolves. Its `page_id` is the map key and the id
    /// `complete_load` decodes/validates against (a sweep or drain completing an orphan
    /// has no caller ref of its own); its `expected_checksum` is now used **only** for
    /// projecting the shared outcome back to the *installer* (H7 — the completion decodes
    /// reference-independently against the bytes' own crc, so no participant's checksum
    /// steers the shared verdict or the map transition). Immutable for the load's life.
    page_ref: PageRef,
    /// Notified on every driver step and on completion; all waiters listen here.
    event: Arc<Event>,
    slot: Mutex<LoadSlot>,
    /// Dedup flag for the pending-drive queue. `true` iff this load
    /// currently has a `Weak` node in [`PendingDrives::queue`]. An enqueue is a
    /// `false→true` CAS — only the winner pushes a node — so a wake storm / a
    /// spurious multi-wake future coalesces to **at most one** queue node per
    /// not-yet-drained load. `drive_pending` clears it **on pop, before polling**,
    /// so a wake landing during that poll re-enqueues a fresh edge (never lost).
    queued: AtomicBool,
}

struct LoadSlot {
    /// The single physical read future, `None` while a driver is polling it or
    /// once the load has resolved. Handed back on `Pending`.
    fut: Option<LoadFuture>,
    /// The resolved **reference-neutral** physical-read outcome, projected to every
    /// waiter through its own ref once present (H7). Was a single decoded `Result` — one
    /// participant's verdict — which is exactly the cross-ref defect this phase fixes.
    done: Option<SharedRead>,
}

/// A `Waker` that notifies a shared `Event` — decouples device completion from
/// any single caller's task so single-flight is cancellation-independent.
struct EventWaker {
    event: Arc<Event>,
    /// The cache-global load-activity signal. Notified alongside the per-load
    /// `event` on every completion so an idle writer wakes to drive even when the
    /// load is orphaned (nobody listens on `event`). See `PageCache::load_activity`.
    activity: Arc<Event>,
    /// The pending-drive scheduler, reachable from this waker so a
    /// device completion — including one delivered on the io_uring completion
    /// thread with no waiter — **enqueues this exact load** for the writer to
    /// drive, rather than forcing the writer to rescan the whole `Loading` set.
    pending: Arc<PendingDrives>,
    /// Weak back-reference to the load this waker belongs to, so the completion
    /// edge names the token to drive. **Weak** is mandatory: a strong ref would
    /// close the cycle `LoadState → slot.fut → this waker → LoadState` and leak
    /// every future that never completes.
    load: Weak<LoadState>,
}

impl Wake for EventWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        // Completion edge: **enqueue first** (leaf lock, released before any notify),
        // then notify — both the per-load `event` (real waiters) and the global
        // `activity` (idle writer). Enqueue-before-notify is load-bearing for the
        // *activity* notify specifically: it keeps the writer's listen-before-drain
        // airtight, so a notify consumed by a drain that then finds the queue empty
        // can never leave this edge parked. Ordering the enqueue ahead of *both*
        // notifies (rather than only `activity`) makes the invariant literal, so a
        // later edit cannot "tidy" it into a park-with-queued-edge hole.
        self.pending.enqueue(&self.load);
        self.event.notify(usize::MAX);
        self.activity.notify(usize::MAX);
    }
}

/// The pending-drive scheduler: a FIFO of loads with an unserved
/// drive edge (an install, or a device completion). The writer drains it in
/// bounded fair batches instead of rescanning the whole `Loading` map, so the
/// idle-drive work is proportional to **edges** (O(1) per load) rather than to
/// `installs × in-flight-set-size` (the old O(N²) full sweep). FIFO order is
/// inherently fair — no hash-order prefix starves the tail.
struct PendingDrives {
    /// `Weak` so a load removed from the map (`ReadErr` drop, or displaced by
    /// `insert_dirty`'s overwrite) is skipped on upgrade failure — the queue never
    /// keeps an orphan alive past its natural life. The map's `Slot::Loading` holds
    /// the strong `Arc` for exactly the window the drive matters.
    queue: Mutex<VecDeque<Weak<LoadState>>>,
}

impl PendingDrives {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    /// Enqueues `load` iff it is not already queued (a `false→true` CAS on its
    /// `queued` flag — only the winner pushes). Leaf lock: never taken while the
    /// map or any `ls.slot` lock is held, and nothing is taken under it.
    fn enqueue(&self, load: &Weak<LoadState>) {
        let Some(ls) = load.upgrade() else { return };
        if ls
            .queued
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.queue.lock().push_back(load.clone());
        }
    }
}

/// The outcome of one queued-load poll, so `drive_pending` can tell a served edge
/// (completed, or handed back on `Pending`) from a **`Busy`** one — the future was
/// held by a concurrent driver, so nothing was polled and the edge must be
/// re-enqueued (defense-in-depth).
enum PollOutcome {
    /// The load completed, or was polled to `Pending` and the future handed back.
    Served,
    /// `done == None` and `fut.take() == None`: a concurrent driver holds the
    /// future. Re-enqueue so the edge is not consumed as a no-op.
    Busy,
}

// H7 note (why there is no longer a `DrivePolicy`): the load is now decoded
// **reference-independently** against the bytes' own crc, so the map transition a
// completer makes (`Clean` for a valid page, `Slot::Failed { actual_crc, .. }` for a
// structurally corrupt one, drop for a transient read fault) is identical no matter which
// participant — real reader, idle/rebind sweep, or close drain — happens to complete it.
// The old G0 hazard was that a *maintenance* driver completing a corrupt-ref orphan
// could cache a sticky `Failed` **by page id**, poisoning a later good ref to the same
// healthy page. That cannot happen now: a healthy orphan decodes to `Clean` (every hit is
// re-validated against the requester's ref), and a genuinely corrupt image caches
// `Failed` keyed by its **actual crc**, which answers a differently-checksummed good ref
// with a plain checksum mismatch — never the structural verdict. So the sweep and drain
// safely cache, and the disposition no longer depends on the driver's role.

/// Point-in-time cache counters.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct CacheMetrics {
    /// Serviced from a resident Clean/Dirty image.
    pub hits: u64,
    /// Required a load (or joined one in flight).
    pub misses: u64,
    /// Physical reads issued (== distinct cold pages loaded).
    pub loads_issued: u64,
    /// Waiters that joined an already-in-flight load (single-flight merges).
    pub load_joins: u64,
    /// Loads that resolved to an error (corrupt or I/O).
    pub load_errors: u64,
    /// Clean pages evicted by the quick tier for capacity (tier invalidations via
    /// `insert_dirty` do not count — `remove` bypasses `on_evict`).
    pub evictions: u64,
    /// Resident clean pages.
    pub clean_resident: u64,
    /// Resident (pinned) dirty pages.
    pub dirty_resident: u64,
    /// Pending-drive queue depth right now: loads with an unserved
    /// drive edge awaiting the writer's bounded drain. A sustained nonzero is the
    /// trigger signal — in-flight cold loads accumulating faster than they drain.
    pub pending_drive_depth: u64,
    /// Cumulative idle/drain-path load polls: the writer's
    /// `drive_pending` + barrier-sweep `poll_load_once` invocations, never the reader
    /// path. The sweep-amplification metric — grows O(edges), not O(installs × N).
    pub drive_polls: u64,
    /// Probes answered from a resident `Failed` verdict, no I/O (T1.4).
    /// Deliberately **not** counted in `hits` (which means a resident *image*) nor
    /// `misses` (which means a load was required).
    pub negative_hits: u64,
    /// Resident `Failed` verdicts right now (gauge; bounded by
    /// [`NEGATIVE_CAPACITY`]).
    pub failed_resident: u64,
    /// `Failed` verdicts evicted by the negative-capacity bound (T1.4). Distinct
    /// from `evictions` (clean-tier capacity evictions).
    pub negative_evictions: u64,
}

/// Shard count for [`ShardedU64`]. Power of two, sized so a modest number of
/// reader threads rarely collide on a shard.
const HIT_SHARDS: usize = 16;

/// One cache-line-padded counter shard, so two threads bumping adjacent shards
/// never false-share a line.
#[derive(Default)]
#[repr(align(64))]
struct CounterShard(AtomicU64);

/// A sharded, padded event counter for the **lock-free warm-hit path**.
/// With the CLOCK tier gone, the hit
/// counter is the only cross-thread write left on that path; a single shared
/// `AtomicU64` would put every reader's `fetch_add` on one contended cache line —
/// re-introducing exactly the shared-write serialization the quick tier removed.
/// Each thread picks a fixed shard once (round-robin from a global atomic) and
/// bumps only it; a metrics read sums the shards. Relaxed everywhere: this is a
/// statistic, not a synchronization edge. Only the hit counter gets this
/// treatment — every other counter is written on paths that already take a lock.
#[derive(Default)]
struct ShardedU64 {
    shards: [CounterShard; HIT_SHARDS],
}

impl ShardedU64 {
    /// Adds one to the calling thread's shard (lock-free, no shared hot line).
    #[inline]
    fn incr(&self) {
        std::thread_local! {
            /// This thread's shard slot, assigned round-robin at first use.
            static SLOT: usize = {
                static NEXT: AtomicUsize = AtomicUsize::new(0);
                NEXT.fetch_add(1, Ordering::Relaxed) % HIT_SHARDS
            };
        }
        let slot = SLOT.with(|s| *s);
        self.shards[slot].0.fetch_add(1, Ordering::Relaxed);
    }

    /// The total across shards. Relaxed per-shard loads: coherent enough for a
    /// point-in-time metric, same contract the single atomic had.
    fn sum(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.0.load(Ordering::Relaxed))
            .sum()
    }
}

#[derive(Default)]
struct Counters {
    /// Sharded: bumped on the lock-free warm-hit path (see [`ShardedU64`]).
    hits: ShardedU64,
    misses: AtomicU64,
    loads_issued: AtomicU64,
    load_joins: AtomicU64,
    load_errors: AtomicU64,
    /// Probes answered by a resident `Failed` verdict (T1.4; under the map lock).
    negative_hits: AtomicU64,
    /// `Failed` verdicts evicted by [`NEGATIVE_CAPACITY`] (T1.4).
    negative_evictions: AtomicU64,
}

struct Inner {
    map: HashMap<PageId, Slot>,
    dirty_count: usize,
    /// FIFO of `Failed` installs, bounding negative residency (T1.4).
    /// May hold **stale** entries: an id whose verdict `insert_dirty` overwrote
    /// stays in the ring until popped (the pop's still-`Failed` check skips it),
    /// and an id that re-fails after such an overwrite appears twice — its older
    /// entry then evicts the newer verdict early, which only under-retains
    /// (bound preserved; a re-read re-derives the verdict — correctness-safe).
    failed_ring: VecDeque<PageId>,
    /// Resident `Failed` verdicts (gauge; every map mutation holds this lock).
    failed_count: usize,
}

/// The engine page cache.
pub struct PageCache {
    inner: Mutex<Inner>,
    counters: Counters,
    /// Notified whenever a load is **installed** (`begin`) or a device read
    /// **completes** (`EventWaker::wake`), so an idle writer can wake and run a
    /// non-blocking `sweep_parked_loads` to drive an orphaned single-flight load
    /// to terminal completion — releasing its superseded-generation device pin
    /// promptly instead of waiting for the next rebind or close (H5 idle drive).
    /// This is a store-global signal (not per-load): a waiter-less orphan notifies
    /// nobody on its own `ls.event`, so the idle bound needs a channel the writer
    /// listens on regardless of which load moved. `Arc` because `EventWaker`,
    /// woken from the io_uring completion thread, holds a clone.
    load_activity: Arc<Event>,
    /// The pending-drive scheduler. Holds `Weak<LoadState>` nodes for
    /// loads with an unserved drive edge (install or completion); the writer drains
    /// it in bounded fair batches (`drive_pending`) instead of rescanning the whole
    /// `Loading` map, making the idle-drive work proportional to *edges* rather than
    /// to `installs × in-flight-set-size`. `Arc` because every `EventWaker`
    /// (including one fired on the io_uring completion thread) holds a clone.
    pending: Arc<PendingDrives>,
    /// Cumulative idle/drain-path load polls. Bumped by
    /// every `poll_load_once` (the writer's idle drain and the barrier sweeps; never
    /// the reader path `drive_load`), so the aggregate-work bound is assertable on
    /// real instrumentation rather than a test-only counter, and the trigger
    /// (sweep-attributable writer work) is visible. Queue *depth* is read on demand
    /// from the scheduler in `metrics()`.
    drive_polls: AtomicU64,
    /// The clean tier (G3): warm hits skip `inner` entirely.
    clean: QuickTier,
    /// Capacity evictions from the tier's `Lifecycle`. Shares the atomic with the
    /// `EvictCounter`.
    quick_evictions: Arc<AtomicU64>,
    /// Chain-prefetch experiment state (T2.2): switch, global permit pool,
    /// counters. Speculation never enters the cache's page state.
    spec: ChainPrefetch,
}

impl PageCache {
    /// A cache with the bounded, memory-aware default clean-tier capacity.
    /// See [`crate::store::Options::cache_capacity`] for the sizing contract.
    pub fn new() -> Self {
        Self::with_capacity(selected_clean_capacity(None))
    }

    /// A cache whose clean tier holds `capacity` pages, up to the per-shard model
    /// below.
    ///
    /// **Capacity semantics are per-shard**:
    /// `quick_cache` divides `capacity` across its shards with ceiling division and
    /// evicts per shard, so the effective clean bound is **≤ `capacity + shards −
    /// 1`** (an overshoot of at most one page per shard; `shards` scales with core
    /// count, so ≤ ~31 pages / ~0.5 MiB over a 1024-page / 16 MiB default) and,
    /// under a skewed page-id distribution, the *observed* residency can also sit
    /// **below** `capacity` because one shard fills while others are empty. Tests
    /// that assert residency must use this per-shard model, not a global bound.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let quick_evictions = Arc::new(AtomicU64::new(0));
        let clean = quick_cache::sync::Cache::with(
            capacity,
            capacity as u64,
            quick_cache::UnitWeighter,
            quick_cache::DefaultHashBuilder::default(),
            EvictCounter {
                evictions: quick_evictions.clone(),
            },
        );
        PageCache {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                dirty_count: 0,
                failed_ring: VecDeque::new(),
                failed_count: 0,
            }),
            counters: Counters::default(),
            load_activity: Arc::new(Event::new()),
            pending: Arc::new(PendingDrives::new()),
            drive_polls: AtomicU64::new(0),
            clean,
            quick_evictions,
            spec: ChainPrefetch {
                enabled: AtomicBool::new(false),
                permits: AtomicUsize::new(SPEC_PERMITS),
                engaged_chains: AtomicU64::new(0),
                issued: AtomicU64::new(0),
                consumed: AtomicU64::new(0),
                unused: AtomicU64::new(0),
                redundant: AtomicU64::new(0),
                fallbacks: AtomicU64::new(0),
                permit_denials: AtomicU64::new(0),
            },
        }
    }

    /// Loads (or joins a load of, or hits) the page named by `page_ref` from the
    /// generation `file`. Cancellation-independent single-flight: N racing
    /// callers, and any pattern of dropped callers, still issue exactly one
    /// physical read for a cold page.
    ///
    /// The returned future is `Send` and owns everything it needs (it does not
    /// borrow `file`).
    pub fn get_page(
        &self,
        page_ref: PageRef,
        file: &FileGeneration,
    ) -> impl Future<Output = Result<Arc<Page>, ReadError>> + Send + '_ {
        let start = self.begin(page_ref, file);
        async move {
            match start {
                Begin::Ready(r) => r,
                Begin::Drive(ls) => self.drive_load(page_ref, ls).await,
            }
        }
    }

    /// Synchronous cache probe + single-load initiation.
    ///
    /// The physical-location lookup happens **outside** the map lock: a locator
    /// may take its own lock (e.g. the M4 WAL overlay), and calling it under the
    /// cache lock would fix a lock order that invites deadlock. The
    /// `Loading` entry is installed under a re-checked lock so a race still
    /// yields exactly one physical read.
    fn begin(&self, page_ref: PageRef, file: &FileGeneration) -> Begin {
        if let Some(b) = self.probe(page_ref) {
            return b;
        }
        // Cold: resolve the location and build the single read future with no
        // lock held.
        let location = match file.locator().locate(page_ref) {
            Some(loc) => loc,
            None => return Begin::Ready(Err(ReadError::Corrupt("page not in generation"))),
        };
        let dev = file.device(location.member);
        let fut = make_load_future(dev, location.offset, location.length as usize);
        let ls = Arc::new(LoadState {
            page_ref,
            event: Arc::new(Event::new()),
            slot: Mutex::new(LoadSlot {
                fut: Some(fut),
                done: None,
            }),
            queued: AtomicBool::new(false),
        });
        // Install under the lock, re-checking in case another caller raced us to
        // the same cold page (its `Loading` wins; our unused future is dropped
        // without ever being polled, so no second read is issued).
        let mut inner = self.inner.lock();
        if let Some(b) = self.probe_locked(&mut inner, page_ref) {
            return b;
        }
        inner
            .map
            .insert(page_ref.page_id, Slot::Loading(ls.clone()));
        self.counters.misses.fetch_add(1, Ordering::Relaxed);
        self.counters.loads_issued.fetch_add(1, Ordering::Relaxed);
        drop(inner);
        // Install edge: enqueue this load, *then* notify. A
        // never-polled orphan (its caller drops before/without ever polling) issues
        // no device read, so completion alone cannot signal it — the install is the
        // only edge, and it is durable queue state, so even a notify delivered to no
        // listener (writer busy) cannot lose it: the entry sits in the queue until
        // the next drain. Enqueue-before-notify keeps the writer's listen-before-
        // drain airtight. The leaf queue lock is taken with no map/slot lock
        // held (we dropped `inner` above).
        self.pending.enqueue(&Arc::downgrade(&ls));
        self.load_activity.notify(usize::MAX);
        Begin::Drive(ls)
    }

    /// A listener on the cache-global load-activity signal (H5 idle drive). The
    /// writer registers this on every idle/maintenance park, *before* sweeping, so
    /// a load installed or completed after the sweep wakes it to sweep again. Not
    /// gated on there being parked loads: the whole point is to catch a load that
    /// appears *after* the park, so the writer must be listening regardless.
    pub(crate) fn load_activity_listen(&self) -> EventListener {
        self.load_activity.listen()
    }

    /// Probes for a resident/loading entry. Checks the clean tier **first,
    /// without the map lock** — the whole point of G3 — and only falls to the map
    /// (Loading/Dirty/Failed) on a clean miss.
    fn probe(&self, page_ref: PageRef) -> Option<Begin> {
        if let Some(page) = self.clean.get(&page_ref.page_id) {
            self.counters.hits.incr();
            return Some(Begin::Ready(validate_hit(page, page_ref)));
        }
        let mut inner = self.inner.lock();
        self.probe_locked(&mut inner, page_ref)
    }

    /// Probes an already-locked map for `page_ref`. `None` means "cold — start a
    /// load". Updates hit/join counters.
    ///
    /// **Re-checks the clean tier under the map lock** before consulting the
    /// map: a `release_dirty` may have inserted the id into `quick_cache`
    /// between a caller's lock-free clean miss and its taking the lock, and without
    /// this re-probe that caller would install a redundant `Loading` and issue a
    /// physical read for a resident page. `quick_cache::get` takes only a shard
    /// read lock and releases it, so the map→shard order holds.
    fn probe_locked(&self, inner: &mut Inner, page_ref: PageRef) -> Option<Begin> {
        if let Some(page) = self.clean.get(&page_ref.page_id) {
            self.counters.hits.incr();
            return Some(Begin::Ready(validate_hit(page, page_ref)));
        }
        match inner.map.get_mut(&page_ref.page_id)? {
            Slot::Dirty(page) => {
                let page = page.clone();
                self.counters.hits.incr();
                Some(Begin::Ready(validate_hit(page, page_ref)))
            }
            Slot::Failed { actual_crc, reason } => {
                // T1.4: a negative answer is neither a `hit` (no image) nor a
                // `miss` (no load) — it gets its own counter (T1.4).
                self.counters.negative_hits.fetch_add(1, Ordering::Relaxed);
                // Answer *every* ref from the crc-keyed verdict, no I/O (H7): the
                // authoritative reader of these bytes (matching crc) gets the structural
                // `reason`; any other ref is — one durable image per never-reused id — a
                // permanent checksum mismatch. A differently-checksummed (e.g. forged)
                // probe therefore does not install a fresh `Loading` and cannot evict the
                // sticky failure.
                let verdict = if page_ref.expected_checksum == *actual_crc {
                    Err(ReadError::Corrupt(reason))
                } else {
                    Err(ReadError::Corrupt("page checksum mismatch"))
                };
                Some(Begin::Ready(verdict))
            }
            Slot::Loading(ls) => {
                self.counters.misses.fetch_add(1, Ordering::Relaxed);
                self.counters.load_joins.fetch_add(1, Ordering::Relaxed);
                Some(Begin::Drive(ls.clone()))
            }
        }
    }

    /// Cooperatively drives the shared load to completion and returns the outcome
    /// **projected through the caller's own `page_ref`** (H7). Any caller may run this;
    /// only one polls the device future at a time, and it is handed back on `Pending` so
    /// a dropped caller never strands the load. The completion itself is
    /// reference-independent (see [`complete_load`]) — this caller's ref affects only how
    /// the shared outcome is projected for *its* return value, never the shared outcome
    /// or the map transition. This is the fix for the cross-ref defect: a joiner used to
    /// receive the completer's decoded verdict verbatim (`done.clone()`), so a forged- or
    /// good-ref joiner could inherit a success/corruption decided against a *different*
    /// ref; now every participant re-validates against its own ref.
    async fn drive_load(
        &self,
        page_ref: PageRef,
        ls: Arc<LoadState>,
    ) -> Result<Arc<Page>, ReadError> {
        loop {
            {
                let slot = ls.slot.lock();
                if let Some(shared) = &slot.done {
                    return project_shared(shared, page_ref);
                }
            }
            // Register for a wakeup *before* attempting to drive, so a completion
            // that lands while we poll is never lost.
            let listener = ls.event.listen();
            {
                let slot = ls.slot.lock();
                if let Some(shared) = &slot.done {
                    return project_shared(shared, page_ref);
                }
            }
            let taken = ls.slot.lock().fut.take();
            if let Some(mut fut) = taken {
                let waker = Waker::from(Arc::new(EventWaker {
                    event: ls.event.clone(),
                    activity: self.load_activity.clone(),
                    pending: self.pending.clone(),
                    load: Arc::downgrade(&ls),
                }));
                let mut cx = Context::from_waker(&waker);
                match fut.as_mut().poll(&mut cx) {
                    Poll::Ready(io_res) => {
                        // Decode/CRC and the map transition run inside `complete_load`
                        // (which takes `inner`); we then publish the shared outcome under
                        // `ls.slot` — never holding `ls.slot` across `inner`.
                        let shared = self.complete_load(&ls, io_res);
                        {
                            let mut slot = ls.slot.lock();
                            slot.done = Some(shared.clone());
                            slot.fut = None;
                        }
                        ls.event.notify(usize::MAX);
                        return project_shared(&shared, page_ref);
                    }
                    Poll::Pending => {
                        // Hand the single future back for the next driver — deliberately
                        // with NO notify (T3.1). The proof is
                        // local to this function: `listener` was registered *before* the
                        // take, and the hand-back is synchronous, so a completion notify
                        // landing anywhere in the take/poll/hand-back window makes the
                        // `listener.await` below return immediately and this same task
                        // re-loops to terminal completion — within one outer poll, with
                        // no suspension and hence no cancellation window.
                        // Every joiner is covered the same way: it registers its
                        // listener before it can observe `fut == None`. Do NOT "fix"
                        // this by notifying here: this task owns a registered listener
                        // and loops, so a notify would wake *itself* after every
                        // Pending and spin a genuinely gated read into a livelock.
                        // (`event_listener` drop-forwarding and the pending-drive
                        // completion edge are additional defenses, not the proof.)
                        // Pinned by the `*_handback_window_*` / `*_same_task_poll`
                        // tests; contrast `poll_load_once`, which registers no
                        // listener, does not loop, and therefore MUST notify.
                        ls.slot.lock().fut = Some(fut);
                    }
                }
            }
            listener.await;
        }
    }

    /// Completes a physical read into a **reference-neutral** [`SharedRead`] and
    /// transitions the map entry out of `Loading`. The bytes are decoded **once against
    /// their own crc** (`decode_checked`), so both the shared outcome and the map
    /// transition are identical no matter which participant — real reader, idle/rebind
    /// sweep, or close drain — completes the load. That is why H7 needs no `DrivePolicy`:
    /// a healthy orphan always caches `Clean` (every hit is re-validated per ref) and a
    /// genuinely corrupt image always caches `Slot::Failed { actual_crc, .. }` (which
    /// answers a differently-checksummed good ref with a plain mismatch, never the
    /// structural verdict), regardless of the completer's role or ref.
    ///
    /// The slot is mutated **only** if the map still holds *this exact* load token
    /// (`Arc::ptr_eq` on the id frozen in `ls`) — identity, not slot shape. An orphaned
    /// or superseded load hands its outcome to its own waiters (via `drive_load` /
    /// `poll_load_once` publishing into `ls.slot.done`) but mutates no map entry: a
    /// racing writer's `Dirty` pin (M3) must win over a stale disk read, and a newer
    /// `Loading` for the same id must not be clobbered. (Pre-H7 this was `finish_load`;
    /// the `None` / `Some(Loading(other))` orphan reasoning is unchanged — only the
    /// disposition moved from id-keyed to crc-keyed `Failed`.)
    fn complete_load(&self, ls: &Arc<LoadState>, io_res: io::Result<Box<[u8]>>) -> SharedRead {
        let id = ls.page_ref.page_id;
        let shared = match io_res {
            Ok(bytes) => {
                let arc: Arc<[u8]> = Vec::from(bytes).into();
                let actual_crc = Checksum::crc32c(&arc);
                // Decode against the bytes' OWN crc: the checksum gate passes by
                // construction (`expected == actual_crc`), so only structural validity is
                // in question — a reference-independent verdict. The embedded page-id
                // check still runs (`id`), catching a misdirected image of another page.
                let self_ref = PageRef {
                    page_id: id,
                    expected_checksum: actual_crc,
                };
                match Page::decode_checked(arc, self_ref, actual_crc) {
                    Ok(page) => SharedRead::Decoded(Arc::new(page)),
                    Err(ReadError::Corrupt(reason)) => SharedRead::Malformed { actual_crc, reason },
                    // `decode_checked` only ever returns `Corrupt` past the (here trivial)
                    // checksum gate; any other error would be non-structural — treat as a
                    // transient read fault rather than cache a verdict for it.
                    Err(e) => SharedRead::ReadErr(e),
                }
            }
            // A short read (image not fully present — e.g. a corrupt ref into an
            // allocated-but-unflushed id: `alloc_id` advances the locator tail at
            // allocation, the bytes land at flush, `direct.rs`) is treated as a
            // **transient** fault and cached by no one. Caching a verdict for it by id
            // would poison a later good ref once the writer flushes those very bytes
            // (the H7 decision deliberately chose transient-retry over the pre-H7
            // sticky-`Failed`, which removes that latent poison). A permanently-truncated
            // corrupt ref simply re-reads on each traversal — a bounded, crafted-input
            // perf cost, not a correctness issue.
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                SharedRead::ReadErr(ReadError::Corrupt("page image truncated / not present"))
            }
            Err(e) => SharedRead::ReadErr(ReadError::from(e)),
        };

        let mut inner = self.inner.lock();
        let still_loading = matches!(
            inner.map.get(&id),
            Some(Slot::Loading(cur)) if Arc::ptr_eq(cur, ls)
        );
        if still_loading {
            match &shared {
                SharedRead::Decoded(page) => {
                    self.insert_clean_locked(&mut inner, id, page.clone());
                }
                SharedRead::Malformed { actual_crc, reason } => {
                    // Sticky, crc-keyed. Note the one non-short-read pre-flush case
                    // (H7): a corrupt ref into an allocated-but-unflushed id
                    // whose extent reads back *full-length garbage* (e.g. a preallocated /
                    // zeroed region) lands here, not in the transient short-read arm, and
                    // caches `Failed { crc(garbage) }`. That is still not a poison — no
                    // good ref to `id` can exist before the writer's `insert_dirty`, which
                    // **unconditionally overwrites** any prior slot (incl. `Failed`) with
                    // the real `Dirty` image, and a stale completion racing in after that
                    // fails the `Arc::ptr_eq` guard above. The repair therefore rests on
                    // `insert_dirty`'s overwrite contract; a future change that made it
                    // conditional would have to re-examine this.
                    inner.map.insert(
                        id,
                        Slot::Failed {
                            actual_crc: *actual_crc,
                            reason,
                        },
                    );
                    // T1.4: account the install and enforce the
                    // negative-capacity bound — the one growth path that was
                    // unbounded. Under the map lock, like every map mutation.
                    inner.failed_count += 1;
                    inner.failed_ring.push_back(id);
                    while inner.failed_ring.len() > NEGATIVE_CAPACITY {
                        let Some(old) = inner.failed_ring.pop_front() else {
                            break;
                        };
                        if matches!(inner.map.get(&old), Some(Slot::Failed { .. })) {
                            inner.map.remove(&old);
                            inner.failed_count -= 1;
                            self.counters
                                .negative_evictions
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        // else: a stale ring entry (`insert_dirty` overwrote the
                        // verdict since) — dropped, which is also what keeps
                        // stale entries from accumulating.
                    }
                }
                SharedRead::ReadErr(_) => {
                    inner.map.remove(&id);
                }
            }
        }
        // One `load_errors` per **physical read** that failed to yield a usable page
        // (structurally corrupt bytes or a transient fault) — counted here, once, by the
        // sole completer. A participant that later fails its *own* checksum projection
        // against a `Decoded` page (a forged/stale ref) is a reference mismatch, not an
        // I/O error, and never touches this counter (H7).
        if matches!(
            shared,
            SharedRead::Malformed { .. } | SharedRead::ReadErr(_)
        ) {
            self.counters.load_errors.fetch_add(1, Ordering::Relaxed);
        }
        shared
    }

    /// Polls one parked load a single step toward completion (the writer's idle
    /// drain and the barrier sweeps; never the reader path `drive_load`). Reuses the
    /// exact `drive_load` take/poll/hand-back protocol — the shared `EventWaker`, the
    /// single physical future, and `complete_load`'s exact-token check — so it never
    /// issues a second read and never clobbers a newer load. On `Ready` it publishes
    /// the result (any real waiter is woken via the shared `Event`); on `Pending` it
    /// hands the future back for the next driver. Takes no lock across the poll.
    ///
    /// Returns [`PollOutcome::Busy`] iff the load is not done **and** a concurrent
    /// driver holds the future (`fut.take() == None`) — nothing was polled, so the
    /// caller ([`drive_pending`]) must not consume the drive edge as a no-op. (The
    /// `Busy` case is provably covered by the concurrent driver's own pre-registered
    /// listener — but the queue path re-enqueues defensively so the
    /// writer never relies on that cross-component timing.)
    fn poll_load_once(&self, ls: &Arc<LoadState>) -> PollOutcome {
        self.drive_polls.fetch_add(1, Ordering::Relaxed);
        {
            let slot = ls.slot.lock();
            if slot.done.is_some() {
                return PollOutcome::Served;
            }
        }
        let taken = ls.slot.lock().fut.take();
        let Some(mut fut) = taken else {
            return PollOutcome::Busy;
        };
        let waker = Waker::from(Arc::new(EventWaker {
            event: ls.event.clone(),
            activity: self.load_activity.clone(),
            pending: self.pending.clone(),
            load: Arc::downgrade(ls),
        }));
        let mut cx = Context::from_waker(&waker);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(io_res) => {
                // Reference-independent completion (H7): the decode is against the
                // bytes' own crc, so the sweep caches exactly what a real reader
                // would (Clean for a healthy orphan, crc-keyed Failed for a corrupt
                // image) — no poison, no `DrivePolicy` needed.
                let shared = self.complete_load(ls, io_res);
                {
                    let mut slot = ls.slot.lock();
                    slot.done = Some(shared);
                    slot.fut = None;
                }
                ls.event.notify(usize::MAX);
            }
            Poll::Pending => {
                ls.slot.lock().fut = Some(fut);
                // MUST notify after handing the future back (H5):
                // unlike `drive_load`, the sweep registered no listener and does
                // not loop, so without this a reader that took the slot lock in
                // the window between our poll and this hand-back — got `None`,
                // registered its listener, and awaits — would never be woken if
                // the device's own completion notify fired into that gap (an
                // `Event` notify with no listener is lost). A reader that
                // observed `None` necessarily holds the slot lock before this
                // hand-back, so its listener is registered before this notify
                // and cannot miss it; then it re-drives the handed-back future.
                ls.event.notify(usize::MAX);
            }
        }
        PollOutcome::Served
    }

    /// Drives up to `budget` queued loads one poll each, and returns
    /// whether the queue **still** has entries — so the writer re-loops (re-checking
    /// commands *first*) rather than parking while work remains. This replaces the
    /// full-`Loading`-set rescan (`sweep_parked_loads`) on the writer's idle/drain
    /// hot path: work is proportional to *drive edges* (install + completion, O(1)
    /// per load) instead of `installs × in-flight-set-size` (the old O(N²)). FIFO is
    /// inherently fair — no hash-order prefix starves the tail.
    ///
    /// The `queued` flag is cleared **on pop, before** [`poll_load_once`], so a wake
    /// landing during that poll re-enqueues a fresh edge. A `Busy` outcome (the
    /// future was held by a concurrent driver) is **re-enqueued** so the edge is not
    /// lost; it also forces the `true` return, so the writer re-loops and retries
    /// after the concurrent (non-blocking, µs-long) poll finishes. The queue lock is
    /// a leaf: `pop_front` releases it before the upgrade/poll, which hold no lock.
    pub(crate) fn drive_pending(&self, budget: usize) -> bool {
        let mut progressed = false;
        let mut any_busy = false;
        for _ in 0..budget {
            let node = { self.pending.queue.lock().pop_front() };
            let Some(weak) = node else {
                return false;
            };
            // Stale/displaced (`ReadErr` drop or `insert_dirty` overwrite): the last
            // strong `Arc` is gone, so the load reached terminal or its device pin was
            // released by the drop itself — nothing to drive. Draining a
            // dead node IS progress (the queue shrank).
            let Some(ls) = weak.upgrade() else {
                progressed = true;
                continue;
            };
            // Clear BEFORE polling so a concurrent wake re-enqueues a fresh edge.
            ls.queued.store(false, Ordering::Release);
            match self.poll_load_once(&ls) {
                PollOutcome::Served => progressed = true,
                PollOutcome::Busy => {
                    // A concurrent driver holds the future: re-enqueue the edge so it
                    // is not consumed as a no-op (defense-in-depth).
                    any_busy = true;
                    self.pending.enqueue(&weak);
                }
            }
        }
        // If the whole batch made **no progress** — every popped node was `Busy` — do
        // not force the writer to re-loop: that would hot-spin on the same nodes, and
        // a *torn* load (a reader that panicked between `fut.take()` and its hand-back,
        // leaving `done==None, fut==None` forever) would spin the writer at 100% CPU.
        // Park instead — safe by the Busy-edge proof:
        // a live holder either completes the load or hands the future back,
        // and the ensuing device completion re-enqueues + notifies `load_activity`,
        // waking the writer. A torn load then rests as a bounded queued residue (driven
        // at the next barrier/close) — the graceful degradation the pre-queue sweep had.
        if any_busy && !progressed {
            return false;
        }
        !self.pending.queue.lock().is_empty()
    }

    /// All `Loading` tokens currently in the map, snapshotted under the map lock
    /// and returned *without* touching any `ls.slot` (so the map→slot lock order is
    /// never inverted).
    fn loading_tokens(&self) -> Vec<Arc<LoadState>> {
        let inner = self.inner.lock();
        inner
            .map
            .values()
            .filter_map(|slot| match slot {
                Slot::Loading(ls) => Some(Arc::clone(ls)),
                _ => None,
            })
            .collect()
    }

    /// Full-scan sweep of *all* parked loads (H5). This is the
    /// **barrier backstop**, not the idle hot path: the writer's idle/maintenance
    /// drive now goes through the bounded `drive_pending` queue, and this full scan
    /// runs only at each **rebind** (checkpoint/compaction, once per maintenance op,
    /// at a drained flushed barrier). It guarantees eventual completeness — every
    /// orphaned single-flight load (its sole waiter dropped) is driven even if some
    /// queue edge were ever missed — by polling every `Loading` slot one step. One
    /// poll each drives the common case (the device already completed the read) to
    /// terminal completion, releasing its (possibly superseded) device pin; a
    /// still-in-flight or still-wanted load is a harmless no-op or an early service.
    /// Driving all loads is safe (the exact-token protocol makes helping a live load
    /// a no-op); the `Busy` outcome is ignored here — a concurrent driver owns that
    /// future and will complete it. The set is empty-dirty at the flushed barrier and
    /// small (bounded by reader concurrency × cold-miss rate). Runs on the writer
    /// stack holding no lock.
    pub(crate) fn sweep_parked_loads(&self) {
        for ls in self.loading_tokens() {
            let _ = self.poll_load_once(&ls);
        }
    }

    /// Drives every load that exists **now** to terminal completion, blocking (H5
    /// close/teardown path). Snapshots the current `Loading` tokens once and drives
    /// each to Ready, so every load in flight at teardown — and, for an io_uring
    /// backend, its kernel op — is resolved before the store finishes closing.
    ///
    /// Scope, stated honestly (H5): the read path is **not**
    /// admission-gated, so a reader racing close can install a *fresh* `Loading`
    /// slot after this snapshot and then drop it, leaving one orphan past close.
    /// That residue is memory-safe (its own device `Arc` keeps its fd alive until
    /// it completes; compio owns the buffer until the terminal CQE) and stays
    /// visible in `retained_segment_bytes` — it is bounded, not a leak of the
    /// unbounded-history kind H5 closes. Snapshotting once (rather than re-scanning)
    /// bounds the work and cannot loop forever on such a late install.
    pub(crate) async fn drain_parked_loads(&self) {
        for ls in self.loading_tokens() {
            // The completion is reference-independent (H7 `complete_load`), so draining
            // at close caches exactly what a real reader would — no corrupt-ref-orphan
            // poison to guard against, hence no policy argument. The result is discarded
            // (we only need terminal completion so the device/kernel op resolves before
            // close); we project it through the installer's frozen ref only because
            // `drive_load` returns a projected result.
            let _ = self.drive_load(ls.page_ref, ls).await;
        }
    }

    fn insert_clean_locked(&self, inner: &mut Inner, id: PageId, page: Arc<Page>) {
        // The clean image lives in `quick_cache`, never the map. This is the
        // Loading→Clean (complete_load) or Dirty→Clean (release_dirty) transition,
        // both under the map lock. Remove the map's Loading/Dirty entry, then insert
        // into qc. **The order is irrelevant to correctness only because both writes
        // are under the map lock**: a lock-free prober that misses qc must take the
        // map lock and re-probe (M3), so it never observes the transient neither-tier
        // state. (`metrics()` reads `qc.len()` under the same lock, so it can't see it
        // either.) A future refactor must not move a qc write outside this lock.
        //
        // Cross-tier exclusivity means qc never holds `id` here; the guard costs
        // a shard lookup, so keep it debug-only (`debug_assert!`'s argument
        // still evaluates in release, so gate the `qc.get` itself).
        #[cfg(debug_assertions)]
        debug_assert_immutable(self.clean.get(&id).as_ref(), &page);
        if let Some(prev) = inner.map.remove(&id) {
            if matches!(prev, Slot::Dirty(_)) {
                inner.dirty_count -= 1;
            }
        }
        // Insert may evict; `on_evict` only bumps an atomic (no map lock).
        self.clean.insert(id, page);
    }

    // -- Dirty tier (writer-facing; wired at M3, unit-tested now) -------------

    /// Inserts a writer-produced page as pinned `Dirty`. Overwrites any resident
    /// state for the id.
    pub fn insert_dirty(&self, page: Arc<Page>) {
        let mut inner = self.inner.lock();
        let id = page.id();
        // A resident clean image lives in qc, not the map, and `insert_dirty`'s
        // "overwrites any resident state" contract must cover it — otherwise a
        // stale clean image would win every later lock-free probe until qc
        // happened to evict it. `remove` returns the image and does
        // **not** route through `on_evict`, so this invalidation cannot inflate the
        // eviction metric. Under the map lock, so it cannot race a transition.
        let prev = self.clean.remove(&id);
        debug_assert_immutable(prev.as_ref().map(|(_, p)| p), &page);
        if !matches!(inner.map.get(&id), Some(Slot::Dirty(_))) {
            inner.dirty_count += 1;
        }
        // T1.4: the documented overwrite of a stale `Failed` verdict (relied on
        // by the H7 pre-flush-garbage note in `complete_load`) now also
        // maintains the gauge; the verdict's ring entry goes stale and is
        // skipped when popped.
        if matches!(inner.map.get(&id), Some(Slot::Failed { .. })) {
            inner.failed_count -= 1;
        }
        // Any resident image for an id must be byte-identical to a new one —
        // ids are never reused and pages are immutable (`cache.rs` module docs). A
        // future path that violates this fails loudly here instead of serving the
        // wrong one of two images.
        if let Some(Slot::Dirty(prev)) = inner.map.get(&id) {
            debug_assert_eq!(
                prev.checksum(),
                page.checksum(),
                "page id {id:?} resident with different bytes — never-reuse invariant violated"
            );
        }
        inner.map.insert(id, Slot::Dirty(page));
    }

    /// Releases a pinned dirty page to the clean tier once its write is durable.
    /// No-op if the id is not currently `Dirty`.
    pub fn release_dirty(&self, id: PageId) {
        let mut inner = self.inner.lock();
        if let Some(Slot::Dirty(page)) = inner.map.get(&id) {
            // `insert_clean_locked` accounts for the dirty→clean transition
            // (it decrements `dirty_count` when the prior slot was `Dirty`).
            let page = page.clone();
            self.insert_clean_locked(&mut inner, id, page);
        }
    }

    // -- Introspection --------------------------------------------------------

    /// The cache state of `id`, or `None` if absent. A clean image lives in qc,
    /// not the map, so it is checked there.
    pub fn state_of(&self, id: PageId) -> Option<CacheState> {
        let inner = self.inner.lock();
        if let Some(state) = inner.map.get(&id).map(|slot| match slot {
            Slot::Loading(_) => CacheState::Loading,
            Slot::Dirty(_) => CacheState::Dirty,
            Slot::Failed { .. } => CacheState::Failed,
        }) {
            return Some(state);
        }
        if self.clean.get(&id).is_some() {
            return Some(CacheState::Clean);
        }
        None
    }

    /// A snapshot of the counters plus resident tallies.
    ///
    /// **Clean and dirty residency are read under a single `inner` hold** so the
    /// snapshot is coherent: every membership mutation (both tiers) takes `inner`,
    /// so a concurrent Dirty↔Clean transition cannot land *between* the two tallies
    /// and make `clean_resident + dirty_resident` double-count or lose a page.
    /// `qc.len()` runs while `inner` is held — map→shard order, the
    /// same as every other under-lock qc access, so no deadlock.
    pub fn metrics(&self) -> CacheMetrics {
        let inner = self.inner.lock();
        let dirty = inner.dirty_count as u64;
        let failed = inner.failed_count as u64;
        // Clean residency is `qc.len()`; evictions come from the tier
        // that owns eviction — the shared `on_evict` atomic.
        let clean = self.clean.len() as u64;
        let evictions = self.quick_evictions.load(Ordering::Relaxed);
        drop(inner);
        CacheMetrics {
            hits: self.counters.hits.sum(),
            misses: self.counters.misses.load(Ordering::Relaxed),
            loads_issued: self.counters.loads_issued.load(Ordering::Relaxed),
            load_joins: self.counters.load_joins.load(Ordering::Relaxed),
            load_errors: self.counters.load_errors.load(Ordering::Relaxed),
            evictions,
            clean_resident: clean,
            dirty_resident: dirty,
            pending_drive_depth: self.pending.queue.lock().len() as u64,
            drive_polls: self.drive_polls.load(Ordering::Relaxed),
            negative_hits: self.counters.negative_hits.load(Ordering::Relaxed),
            failed_resident: failed,
            negative_evictions: self.counters.negative_evictions.load(Ordering::Relaxed),
        }
    }
}

impl Default for PageCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Chain-prefetch experiment (T2.2).
// ---------------------------------------------------------------------------

/// Store-wide state for the overflow-chain prefetch **experiment** (T2.2).
///
/// Speculation itself never touches the page cache — no `Loading` slot, no
/// clean-tier insertion (the no-shadowing/no-pollution property is structural).
/// The cache merely *hosts* the experiment's store-wide pieces, because it is
/// the one read-side object every traversal already holds: the off-by-default
/// switch, the global speculative-read permit budget, and the counters the
/// budget accounting is judged by.
pub(crate) struct ChainPrefetch {
    enabled: AtomicBool,
    /// Global nonblocking permit pool bounding speculative reads in flight
    /// across *all* chains: `get_many`
    /// bounds neither key count nor chain count, so a per-chain window alone
    /// leaves `N × W` caller-controlled. No permit → the window shrinks and the
    /// chain continues on demand reads; speculation never waits.
    permits: AtomicUsize,
    pub(crate) engaged_chains: AtomicU64,
    pub(crate) issued: AtomicU64,
    pub(crate) consumed: AtomicU64,
    pub(crate) unused: AtomicU64,
    pub(crate) redundant: AtomicU64,
    pub(crate) fallbacks: AtomicU64,
    pub(crate) permit_denials: AtomicU64,
}

/// Store-wide cap on speculative chain reads in flight (64 × 16 KiB = 1 MiB of
/// private buffers). Pre-declared experiment constant, not a tuned value.
pub(crate) const SPEC_PERMITS: usize = 64;

/// One unit of the speculative bound. Owned by the read's **slot** for the
/// slot's whole life — the in-flight read, the completed-but-unconsumed
/// private buffer, and the drain set — so the pool bounds private *buffers*,
/// not merely submitted reads.
/// Releases when the slot drops: at consumption, at discard, at terminal
/// drain completion, or on caller cancellation. The drop-release on
/// cancellation is the experiment-grade caveat: a
/// real backend's device read may still be completing pool-side after the
/// future is dropped, so the *device in-flight* bound can transiently
/// overshoot by the number of mid-cancellation reads. A
/// cancellation-independent owner is a named precondition for default-on.
pub(crate) struct SpecPermit<'a>(&'a AtomicUsize);

impl Drop for SpecPermit<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Release);
    }
}

/// Counter snapshot for the chain-prefetch experiment. All fields cumulative;
/// benches/tests scope deltas per sample by snapshotting before and after.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChainPrefetchStats {
    /// Chains that engaged speculation (passed switch, capability, length, and
    /// cold-head checks).
    pub engaged_chains: u64,
    /// Speculative device reads issued.
    pub issued: u64,
    /// Speculative pages consumed after full authoritative validation
    /// (location equality and full-`PageRef` identity).
    pub consumed: u64,
    /// Speculative reads issued but never consumed: drained after a
    /// misprediction/error, superseded by a resident page, or still unconsumed
    /// when the chain finished. (Reads dropped by caller cancellation are not
    /// counted — stats are best-effort under cancellation.)
    pub unused: u64,
    /// Positions where the cache already held (or was loading) the
    /// authoritative page when it became consumable: the private image was
    /// discarded in the cache's favor and the read counted as redundant work.
    pub redundant: u64,
    /// Chains that abandoned speculation because a speculative read **failed
    /// validation** (location mismatch, ref mismatch, read/decode failure) and
    /// finished on demand reads. Deliberately narrow: a chain that merely
    /// *stops replenishing* — capability refusal mid-chain, permit denial, a
    /// redundant hit — is not a fallback (nothing mispredicted); those show up
    /// as `permit_denials`/`redundant` and a shortfall in `issued`.
    pub fallbacks: u64,
    /// Window issues skipped because the global permit pool was empty.
    pub permit_denials: u64,
}

impl PageCache {
    /// EXPERIMENT (T2.2): enables/disables overflow-chain prefetch on this
    /// store's read path. **Off by default.** Not a stability contract — the
    /// experiment's ship decision may remove or replace it.
    ///
    /// Opt in only for **genuinely cold, non-repeating large-value reads**
    /// (measured: 1.5–5.3× on cold ≥4-link chains). Two costs to
    /// know about:
    /// - prefetched links do **not** enter the page cache, so re-reading a
    ///   just-prefetched value is cold again (measured 12–100× slower than the
    ///   demand path's warm re-read) — repeat-readers should leave this off;
    /// - cancelling an enabled chain read mid-flight can transiently exceed
    ///   the speculative device in-flight bound on real backends (permits
    ///   release when the dropped futures do, but a pool-side read may still
    ///   be completing) — the reason this is not default-on.
    #[doc(hidden)]
    pub fn set_chain_prefetch(&self, enabled: bool) {
        self.spec.enabled.store(enabled, Ordering::Release);
    }

    pub(crate) fn chain_prefetch_enabled(&self) -> bool {
        self.spec.enabled.load(Ordering::Acquire)
    }

    /// Snapshot of the chain-prefetch experiment counters.
    #[doc(hidden)]
    pub fn chain_prefetch_stats(&self) -> ChainPrefetchStats {
        ChainPrefetchStats {
            engaged_chains: self.spec.engaged_chains.load(Ordering::Relaxed),
            issued: self.spec.issued.load(Ordering::Relaxed),
            consumed: self.spec.consumed.load(Ordering::Relaxed),
            unused: self.spec.unused.load(Ordering::Relaxed),
            redundant: self.spec.redundant.load(Ordering::Relaxed),
            fallbacks: self.spec.fallbacks.load(Ordering::Relaxed),
            permit_denials: self.spec.permit_denials.load(Ordering::Relaxed),
        }
    }

    /// Permits currently available (tests assert full recovery).
    #[doc(hidden)]
    pub fn chain_prefetch_permits(&self) -> usize {
        self.spec.permits.load(Ordering::Acquire)
    }

    pub(crate) fn spec(&self) -> &ChainPrefetch {
        &self.spec
    }

    /// Nonblocking permit acquisition; `None` means the window must shrink.
    pub(crate) fn spec_try_permit(&self) -> Option<SpecPermit<'_>> {
        let permits = &self.spec.permits;
        let mut cur = permits.load(Ordering::Acquire);
        loop {
            if cur == 0 {
                self.spec.permit_denials.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            match permits.compare_exchange_weak(cur, cur - 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(SpecPermit(permits)),
                Err(seen) => cur = seen,
            }
        }
    }
}

enum Begin {
    Ready(Result<Arc<Page>, ReadError>),
    Drive(Arc<LoadState>),
}

/// Debug-only guard that a page about to displace a resident image for the same id
/// carries the **same bytes**. Ids are never reused and pages are
/// immutable, so this always holds; a future in-place/id-reuse path that breaks it
/// fails loudly here rather than silently serving the wrong one of two images.
fn debug_assert_immutable(resident: Option<&Arc<Page>>, incoming: &Arc<Page>) {
    if let Some(prev) = resident {
        debug_assert_eq!(
            prev.checksum(),
            incoming.checksum(),
            "page id {:?} resident with different bytes — never-reuse invariant violated",
            incoming.id()
        );
    }
}

/// Validates a cache-hit image against the parent's expected checksum. A
/// mismatch means a corrupt parent reference points at a real (good) page — a
/// structural defect surfaced even on the hit path.
fn validate_hit(page: Arc<Page>, page_ref: PageRef) -> Result<Arc<Page>, ReadError> {
    if page.checksum() != page_ref.expected_checksum {
        return Err(ReadError::Corrupt(
            "cached page checksum mismatch with reference",
        ));
    }
    Ok(page)
}

/// Builds a `'static` device read future that owns its device `Arc`.
fn make_load_future(dev: Arc<dyn PageIo>, offset: u64, len: usize) -> LoadFuture {
    Box::pin(async move { dev.read_exact_at(offset, len).await })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DataFileLocator, FileGeneration};
    use crate::io::{BufResult, IoFuture};
    use crate::page::{encode_leaf_inline, PageKind, PAGE_SIZE};
    use futures_lite::future::block_on;
    use std::collections::HashMap as Map;
    use std::sync::atomic::AtomicBool;

    fn available_bytes_for_pages(pages: usize) -> u64 {
        pages as u64 * PAGE_SIZE as u64 * DEFAULT_CLEAN_MEMORY_DIVISOR
    }

    #[test]
    fn default_capacity_derivation_has_floor_fallback_and_ceiling() {
        assert_eq!(
            resolve_clean_capacity(None, None),
            DEFAULT_CLEAN_CAPACITY,
            "failed detection uses the 16-MiB fallback"
        );
        assert_eq!(
            resolve_clean_capacity(None, Some(0)),
            DEFAULT_CLEAN_CAPACITY,
            "small-host derivation has the 16-MiB floor"
        );
        assert_eq!(
            resolve_clean_capacity(None, Some(available_bytes_for_pages(8192))),
            8192,
            "an in-range budget is preserved"
        );
        assert_eq!(
            resolve_clean_capacity(None, Some(u64::MAX)),
            MAX_DEFAULT_CLEAN_CAPACITY,
            "large-host derivation has the 512-MiB ceiling"
        );
    }

    #[test]
    fn explicit_capacity_wins_without_host_dependent_clamping() {
        assert_eq!(selected_clean_capacity(Some(0)), 0);
        assert_eq!(resolve_clean_capacity(Some(0), Some(u64::MAX)), 0);
        assert_eq!(
            resolve_clean_capacity(Some(MAX_DEFAULT_CLEAN_CAPACITY * 2), Some(0)),
            MAX_DEFAULT_CLEAN_CAPACITY * 2
        );
    }

    #[test]
    fn mem_available_parser_requires_kib_and_checks_overflow() {
        assert_eq!(
            parse_mem_available_bytes("MemTotal: 9 kB\nMemAvailable: 1234 kB\n"),
            Some(1234 * 1024)
        );
        assert_eq!(parse_mem_available_bytes("MemAvailable: 1234 MB\n"), None);
        assert_eq!(parse_mem_available_bytes("MemFree: 1234 kB\n"), None);
        assert_eq!(
            parse_mem_available_bytes(&format!("MemAvailable: {} kB\n", u64::MAX)),
            None
        );
    }

    // A programmable device: serves seeded page images at recorded offsets,
    // counts physical `read_exact_at` invocations, and can gate reads so many
    // waiters genuinely race one in-flight load.
    struct ProgIo {
        pages: Mutex<Map<u64, Vec<u8>>>,
        reads: AtomicU64,
        gate: Arc<Gate>,
    }

    struct Gate {
        open: AtomicBool,
        wakers: Mutex<Vec<Waker>>,
    }

    impl Gate {
        fn open(&self) {
            self.open.store(true, Ordering::Release);
            for w in self.wakers.lock().drain(..) {
                w.wake();
            }
        }
    }

    impl ProgIo {
        fn new(gated: bool) -> Arc<Self> {
            Arc::new(ProgIo {
                pages: Mutex::new(Map::new()),
                reads: AtomicU64::new(0),
                gate: Arc::new(Gate {
                    open: AtomicBool::new(!gated),
                    wakers: Mutex::new(Vec::new()),
                }),
            })
        }
        fn seed(&self, offset: u64, bytes: Vec<u8>) {
            self.pages.lock().insert(offset, bytes);
        }
        fn read_count(&self) -> u64 {
            self.reads.load(Ordering::Relaxed)
        }
    }

    // A future that resolves once the gate is open, re-registering its waker on
    // each pending poll (so an EventWaker hand-off keeps working).
    struct GateWait {
        gate: Arc<Gate>,
    }
    impl Future for GateWait {
        type Output = ();
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.gate.open.load(Ordering::Acquire) {
                Poll::Ready(())
            } else {
                self.gate.wakers.lock().push(cx.waker().clone());
                Poll::Pending
            }
        }
    }

    impl PageIo for ProgIo {
        fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let gate = self.gate.clone();
            Box::pin(async move {
                GateWait { gate }.await;
                match self.pages.lock().get(&offset) {
                    Some(bytes) if bytes.len() >= len => {
                        Ok(bytes[..len].to_vec().into_boxed_slice())
                    }
                    Some(_) => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short")),
                    None => Err(io::Error::from_raw_os_error(5)),
                }
            })
        }
        fn write_all_at(&self, _o: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult> {
            Box::pin(async move {
                BufResult {
                    result: Ok(()),
                    buffer,
                }
            })
        }
        fn datasync(&self) -> IoFuture<'_, io::Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn size(&self) -> IoFuture<'_, io::Result<u64>> {
            Box::pin(async { Ok(0) })
        }
        fn shutdown(&self) -> IoFuture<'_, io::Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn leaf_at(dev: &ProgIo, loc: &DataFileLocator, id: u64, offset: u64) -> PageRef {
        let page = encode_leaf_inline(
            PageId(id),
            1,
            &[(b"k".to_vec(), format!("v{id}").into_bytes())],
        );
        let r = page.page_ref();
        dev.seed(offset, page.bytes().to_vec());
        loc.record(id, offset);
        r
    }

    fn file_with(dev: Arc<ProgIo>, loc: DataFileLocator) -> FileGeneration {
        FileGeneration::direct(dev as Arc<dyn PageIo>, Arc::new(loc))
    }

    // A device that **snapshots the addressed bytes at invocation** and delays the
    // completion behind a gate. `ProgIo` samples its map *after* the gate, which
    // cannot express the one interleaving that matters here: a read issued against
    // a not-yet-written region whose completion lands *after* the writer's flush
    // has filled it. This device can, so the load observes the pre-flush bytes
    // while the store goes on to hold a perfectly good page at that id.
    struct LateIo {
        pages: Mutex<Map<u64, Vec<u8>>>,
        reads: AtomicU64,
        gate: Arc<Gate>,
    }

    impl LateIo {
        fn new() -> Arc<Self> {
            Arc::new(LateIo {
                pages: Mutex::new(Map::new()),
                reads: AtomicU64::new(0),
                gate: Arc::new(Gate {
                    open: AtomicBool::new(false),
                    wakers: Mutex::new(Vec::new()),
                }),
            })
        }
        fn seed(&self, offset: u64, bytes: Vec<u8>) {
            self.pages.lock().insert(offset, bytes);
        }
        fn read_count(&self) -> u64 {
            self.reads.load(Ordering::Acquire)
        }
    }

    impl PageIo for LateIo {
        fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
            // Snapshot first, then publish the count: a test that observes
            // `read_count() == 1` has a guaranteed-taken snapshot, so re-seeding
            // afterwards cannot race into this read's result.
            let snap = self.pages.lock().get(&offset).cloned();
            self.reads.fetch_add(1, Ordering::Release);
            let gate = self.gate.clone();
            Box::pin(async move {
                GateWait { gate }.await;
                match snap {
                    Some(bytes) if bytes.len() >= len => {
                        Ok(bytes[..len].to_vec().into_boxed_slice())
                    }
                    Some(_) => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short")),
                    None => Err(io::Error::from_raw_os_error(5)),
                }
            })
        }
        fn write_all_at(&self, _o: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult> {
            Box::pin(async move {
                BufResult {
                    result: Ok(()),
                    buffer,
                }
            })
        }
        fn datasync(&self) -> IoFuture<'_, io::Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn size(&self) -> IoFuture<'_, io::Result<u64>> {
            Box::pin(async { Ok(0) })
        }
        fn shutdown(&self) -> IoFuture<'_, io::Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    /// An orphaned load must not install a verdict for an id it no longer owns.
    ///
    /// Regression for a defect found in M6 Phase G:
    /// `complete_load`'s (then `finish_load`'s) old `still_loading` test accepted `Some(Loading(_)) | None`,
    /// and the `None` arm let a *stale* load durably poison a **good** page:
    ///
    ///   1. A corrupt-but-checksum-valid parent ref names id 7, freshly allocated
    ///      but not yet flushed. The ref resolves (`alloc_id` advances the locator
    ///      tail at allocation; the bytes land at flush), so a reader installs
    ///      `Loading(7)` and issues a read of the unwritten region.
    ///   2. The writer pins the real image: `insert_dirty(7)` overwrites the
    ///      `Loading` slot, orphaning the in-flight load (its driver lives on).
    ///   3. The flush is durable; `release_dirty(7)` moves 7 to `Clean`.
    ///   4. The clean tier evicts 7 under pressure → **neither tier** holds 7.
    ///   5. The orphaned load completes `Corrupt` and — pre-fix — read `None` as
    ///      "still loading" and installed a **sticky `Failed(7)`**.
    ///
    /// From then on every reader of the good, durable page 7 got a false `Corrupt`
    /// with no re-read, and the first write-path read poisoned the store.
    #[test]
    fn an_orphaned_load_cannot_install_a_verdict_for_a_page_it_no_longer_owns() {
        let dev = LateIo::new();
        let loc = DataFileLocator::new();
        loc.record(7, 0);
        // Id 7's extent is allocated but unwritten: a read of it comes back short.
        dev.seed(0, vec![0u8; 8]);

        let good = encode_leaf_inline(PageId(7), 1, &[(b"k".to_vec(), b"v7".to_vec())]);
        let r7 = good.page_ref();
        let cache = Arc::new(PageCache::with_capacity(1));
        let file = Arc::new(FileGeneration::direct(
            dev.clone() as Arc<dyn PageIo>,
            Arc::new(loc),
        ));

        // (1) The reader becomes the load's driver and issues the physical read.
        let reader = {
            let (cache, file) = (cache.clone(), file.clone());
            std::thread::spawn(move || block_on(cache.get_page(r7, &file)))
        };
        while dev.read_count() == 0 {
            std::thread::yield_now();
        }

        // (2)+(3) The writer pins the real image and the flush makes it durable.
        cache.insert_dirty(Arc::new(good.clone()));
        cache.release_dirty(PageId(7));
        assert_eq!(cache.state_of(PageId(7)), Some(CacheState::Clean));

        // (4) Churn other pages through the one-page clean tier until 7 is evicted
        // — no device I/O. (quick_cache evicts per shard, so *which* insert
        // displaces 7 depends on shard hashing; churn distinct ids until the
        // eviction lands, bounded so a regression fails rather than hangs.)
        let mut next_id = 8u64;
        while cache.state_of(PageId(7)) == Some(CacheState::Clean) {
            let other = encode_leaf_inline(PageId(next_id), 1, &[(b"k".to_vec(), b"v8".to_vec())]);
            cache.insert_dirty(Arc::new(other));
            cache.release_dirty(PageId(next_id));
            next_id += 1;
            assert!(next_id < 10_000, "clean-tier pressure never evicted page 7");
        }
        assert_eq!(
            cache.state_of(PageId(7)),
            None,
            "precondition: 7 is evicted, so neither tier holds it"
        );

        // (5) Now let the orphaned load complete. Its own read genuinely saw the
        // unwritten region, so *its* caller correctly gets Corrupt...
        dev.gate.open();
        assert!(
            matches!(reader.join().unwrap(), Err(ReadError::Corrupt(_))),
            "the orphaned load's own waiter still gets its result"
        );
        // ...but it must not leave a verdict behind for an id it no longer owns.
        assert_eq!(
            cache.state_of(PageId(7)),
            None,
            "a stale load must not install Failed for a page it no longer owns"
        );

        // The harm, stated as behaviour: the flush has landed, so page 7 is good on
        // disk. A fresh reader must be able to read it. Pre-fix this returned the
        // sticky Corrupt verdict without ever touching the device.
        dev.seed(0, good.bytes().to_vec());
        let page = block_on(cache.get_page(r7, &file)).expect("good durable page reads back");
        assert_eq!(page.id(), PageId(7));
    }

    struct Noop;
    impl Wake for Noop {
        fn wake(self: Arc<Self>) {}
    }

    #[test]
    fn load_then_hit_reads_once() {
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 7, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        let p = block_on(cache.get_page(r, &file)).unwrap();
        assert_eq!(p.kind(), PageKind::Leaf);
        assert_eq!(
            p.leaf_get(b"k"),
            Some(crate::page::LeafValue::Inline(&b"v7"[..]))
        );
        assert_eq!(cache.state_of(PageId(7)), Some(CacheState::Clean));
        // Second get is a pure hit.
        let _ = block_on(cache.get_page(r, &file)).unwrap();
        assert_eq!(dev.read_count(), 1);
        let m = cache.metrics();
        assert_eq!(m.hits, 1);
        assert_eq!(m.loads_issued, 1);
        assert_eq!(m.clean_resident, 1);
    }

    #[test]
    fn single_flight_dedups_racing_waiters() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 3, 0);
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let (cache, file, r) = (cache.clone(), file.clone(), r);
            handles.push(std::thread::spawn(move || {
                block_on(cache.get_page(r, &file))
                    .unwrap()
                    .leaf_get(b"k")
                    .unwrap()
                    .to_owned_value()
            }));
        }
        // Give the threads time to register as joiners, then open the gate.
        std::thread::sleep(std::time::Duration::from_millis(50));
        dev.gate.open();
        for h in handles {
            assert_eq!(
                h.join().unwrap(),
                crate::page::EntryValue::Inline(b"v3".to_vec())
            );
        }
        assert_eq!(
            dev.read_count(),
            1,
            "exactly one physical read for a cold page"
        );
        assert!(cache.metrics().load_joins >= 1);
    }

    #[test]
    fn single_flight_survives_first_waiter_drop() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        // First waiter: poll once so it becomes the driver and issues the single
        // physical read, then drop it while the gate is still closed.
        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut f0 = Box::pin(cache.get_page(r, &file));
        assert!(matches!(f0.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(dev.read_count(), 1, "driver issued the read");
        drop(f0); // the load future is safely parked in the Loading slot

        // Other waiters join the same in-flight load.
        let mut handles = Vec::new();
        for _ in 0..3 {
            let (cache, file, r) = (cache.clone(), file.clone(), r);
            handles.push(std::thread::spawn(move || {
                block_on(cache.get_page(r, &file))
                    .unwrap()
                    .leaf_get(b"k")
                    .unwrap()
                    .to_owned_value()
            }));
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
        dev.gate.open();
        for h in handles {
            assert_eq!(
                h.join().unwrap(),
                crate::page::EntryValue::Inline(b"v5".to_vec())
            );
        }
        assert_eq!(
            dev.read_count(),
            1,
            "still exactly one read after first drop"
        );
    }

    /// H5 — the maintenance sweep drives an orphaned load (its sole waiter dropped
    /// mid-load, no one left to drive it) to terminal completion, transitioning
    /// its `Loading` slot out so the parked device clone releases. Without the
    /// sweep the slot would sit `Loading` — pinning its (possibly superseded)
    /// device — until another reader happened by.
    #[test]
    fn sweep_drives_an_orphaned_load_to_completion() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        // Orphan it: poll once to become the driver and issue the read, then drop
        // the sole waiter while the gate is closed — nothing is left to drive it.
        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut f0 = Box::pin(cache.get_page(r, &file));
        assert!(matches!(f0.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(dev.read_count(), 1, "driver issued the read");
        drop(f0);
        assert_eq!(
            cache.state_of(PageId(5)),
            Some(CacheState::Loading),
            "orphaned load is parked with no waiter"
        );

        // The device read completes, but there is no waiter to publish it. A
        // sweep drives the parked load to terminal completion.
        dev.gate.open();
        cache.sweep_parked_loads();
        assert_eq!(
            cache.state_of(PageId(5)),
            Some(CacheState::Clean),
            "the sweep drove the orphan to Clean, releasing its device pin"
        );
        assert_eq!(
            dev.read_count(),
            1,
            "the sweep issued no second physical read"
        );
        // The page is now served from the resident clean image.
        let page = block_on(cache.get_page(r, &file)).unwrap();
        assert_eq!(
            page.leaf_get(b"k").unwrap().to_owned_value(),
            crate::page::EntryValue::Inline(b"v5".to_vec())
        );
    }

    /// H7 (supersedes the H5 drop-the-slot behaviour) — the sweep completes an
    /// orphaned load carrying a **corrupt (wrong-checksum) ref over good bytes**
    /// *reference-independently*: it decodes against the bytes' own crc and caches the
    /// healthy page as `Clean`, rather than dropping the slot. No poison results — the
    /// crc-keyed `Clean` image is re-validated per ref on every hit, so the corrupt ref
    /// still gets a checksum mismatch while a good ref reads the page. This is strictly
    /// better than the pre-H7 drop (which forced a re-read) and needs no `DrivePolicy`.
    #[test]
    fn sweep_of_a_corrupt_ref_orphan_caches_clean_without_poison() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let good = leaf_at(&dev, &loc, 5, 0);
        // A ref to the same id/offset but with the wrong expected checksum — the
        // page bytes on disk are perfectly good.
        let corrupt = PageRef {
            page_id: good.page_id,
            expected_checksum: crate::page::Checksum(good.expected_checksum.0 ^ 0xFFFF_FFFF),
        };
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        // Orphan a load carrying the corrupt ref.
        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut f0 = Box::pin(cache.get_page(corrupt, &file));
        assert!(matches!(f0.as_mut().poll(&mut cx), Poll::Pending));
        drop(f0);

        dev.gate.open();
        cache.sweep_parked_loads();
        assert_eq!(
            cache.state_of(PageId(5)),
            Some(CacheState::Clean),
            "the sweep decoded ref-independently and cached the healthy page as Clean"
        );

        // No poison: the corrupt ref still fails its own checksum (served from Clean),
        // and a good-ref read succeeds — both without any new physical read.
        let before = dev.read_count();
        assert!(matches!(
            block_on(cache.get_page(corrupt, &file)),
            Err(ReadError::Corrupt(_))
        ));
        let page = block_on(cache.get_page(good, &file)).expect("good ref reads the healthy page");
        assert_eq!(
            page.leaf_get(b"k").unwrap().to_owned_value(),
            crate::page::EntryValue::Inline(b"v5".to_vec())
        );
        assert_eq!(
            dev.read_count(),
            before,
            "both served from the cached Clean image, no re-read"
        );
    }

    /// H5 idle drive — installing a `Loading` slot notifies the cache-global
    /// load-activity signal, so an idle writer wakes to sweep even for an orphan
    /// that is never polled (and thus issues no device read whose completion could
    /// signal it). The install is the only edge for that case.
    #[test]
    fn installing_a_load_notifies_idle_writer() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        // Listen *before* the install, exactly as the writer's listen-before-sweep.
        let mut activity = Box::pin(cache.load_activity_listen());
        // `get_page` runs `begin` (the install) synchronously at call time, before
        // the returned future is ever polled — so a never-polled orphan still fires.
        let _f0 = Box::pin(cache.get_page(r, &file));
        assert!(
            matches!(activity.as_mut().poll(&mut cx), Poll::Ready(())),
            "installing a Loading slot notifies the idle writer to sweep"
        );
    }

    /// H5 idle drive — a device completion notifies the cache-global load-activity
    /// signal (via `EventWaker`), so an idle writer wakes to re-poll a handed-back
    /// orphan whose read finished with no waiter listening on its own `ls.event`.
    #[test]
    fn a_completion_notifies_idle_writer() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        // Orphan a load: poll once to issue the read, then drop the sole waiter.
        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut f0 = Box::pin(cache.get_page(r, &file));
        assert!(matches!(f0.as_mut().poll(&mut cx), Poll::Pending));
        drop(f0);

        // Listen, then let the device complete — the completion must wake us.
        let mut activity = Box::pin(cache.load_activity_listen());
        dev.gate.open();
        assert!(
            matches!(activity.as_mut().poll(&mut cx), Poll::Ready(())),
            "a device completion notifies the idle writer to re-poll the orphan"
        );
    }

    // === The bounded, edge-targeted pending-drive queue ===

    fn depth(cache: &PageCache) -> usize {
        cache.pending.queue.lock().len()
    }

    /// A **never-polled** orphan (its caller drops before the first
    /// poll, so it issues no device read whose completion could signal it) is driven
    /// to terminal completion by `drive_pending` over its **install** edge alone,
    /// exactly as the old full sweep did — but without scanning the map.
    #[test]
    fn drive_pending_drives_a_never_polled_orphan() {
        let dev = ProgIo::new(false); // ungated: the read completes on first poll
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        // `begin` (install) runs synchronously at call time and enqueues the edge;
        // dropping the never-polled future orphans the load.
        let f0 = Box::pin(cache.get_page(r, &file));
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Loading));
        assert_eq!(depth(&cache), 1, "install enqueued exactly one drive edge");
        drop(f0);

        assert!(
            !cache.drive_pending(64),
            "one batch drains the single edge to empty"
        );
        assert_eq!(
            cache.state_of(PageId(5)),
            Some(CacheState::Clean),
            "drive_pending drove the never-polled orphan to Clean"
        );
        assert_eq!(depth(&cache), 0, "queue drained");
        assert_eq!(dev.read_count(), 1, "no second physical read");
    }

    /// A completion edge (the device finishing with no waiter) is what
    /// drives a **handed-back** orphan: poll once to issue the gated read and drop
    /// the sole waiter, open the gate (the `EventWaker` enqueues the completion
    /// edge), then `drive_pending` completes it.
    #[test]
    fn drive_pending_completes_a_handed_back_orphan_on_the_completion_edge() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut f0 = Box::pin(cache.get_page(r, &file));
        assert!(matches!(f0.as_mut().poll(&mut cx), Poll::Pending));
        // Drain the install edge while still gated: the load stays Loading (Pending),
        // the edge is served (not re-enqueued), and the future is handed back.
        assert!(!cache.drive_pending(64));
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Loading));
        drop(f0); // orphan it: no waiter left

        // The device completes -> EventWaker enqueues the completion edge.
        dev.gate.open();
        assert_eq!(depth(&cache), 1, "completion enqueued the drive edge");
        assert!(!cache.drive_pending(64));
        assert_eq!(
            cache.state_of(PageId(5)),
            Some(CacheState::Clean),
            "the completion edge drove the orphan to Clean"
        );
        assert_eq!(dev.read_count(), 1);
    }

    // === T3.1: the disputed drive_load hand-back notify, settled ===
    //
    // v1 proposed a notify after `drive_load`'s `Pending` hand-back (the site above
    // `listener.await`); but the pre-registered retained
    // listener plus the EventWaker's pending-queue push already close the window.
    // Five tests pin down the disputed interleaving — "joiner sees `fut == None`;
    // completion notify fires pre-hand-back" — in two kinds:
    //
    // - Four STATE-MACHINE tests (privileged: the test plays the mid-poll driver by
    //   taking the future out of `ls.slot`, holding the window open across staged
    //   steps): the parked joiner completing across a silent hand-back; the joiner
    //   that spent its notification being rescued by the driver's retained
    //   listener; drop-forwarding of a dropped driver's notification; and the
    //   pending-queue completion edge as a genuine 0->1 backstop.
    // - One PRODUCTION-PATH test (`silent_handback_re_drives_within_the_same_task_
    //   poll`, no slot surgery): a staged device future opens the window inside the
    //   driver's real `fut.poll()`, and the real hand-back runs.
    //
    // All pass against the notify-less hand-back, so the notify is redundant (and
    // would self-wake — see the hand-back comment); if a refactor breaks the
    // covering invariants (the listener registration spanning the device poll in
    // `drive_load`, or enqueue-before-notify in `EventWaker::wake`), these fail.

    /// Takes the single physical future out of the load slot, emulating a driver
    /// that is *inside* `fut.poll()` — the state a joiner observes as `fut == None`.
    fn take_fut(ls: &Arc<LoadState>) -> LoadFuture {
        ls.slot
            .lock()
            .fut
            .take()
            .expect("slot holds the handed-back future")
    }

    /// T3.1 test 1 — a joiner that parked in the hand-back window (saw
    /// `fut == None`) completes on its own after a **silent** hand-back, because
    /// its listener was registered before the completion notify fired.
    #[test]
    fn joiner_parked_in_the_handback_window_completes_without_a_handback_notify() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);

        // Reader A drives once: issues the gated read, hands the future back, parks.
        let mut fa = Box::pin(cache.get_page(r, &file));
        assert!(matches!(fa.as_mut().poll(&mut cx), Poll::Pending));

        // The test becomes the mid-poll driver: the future leaves the slot.
        let ls = cache.loading_tokens().pop().expect("one Loading token");
        let fut = take_fut(&ls);

        // Joiner B arrives inside the window: done=None, listener registered,
        // `fut == None` observed — parks.
        let mut fb = Box::pin(cache.get_page(r, &file));
        assert!(matches!(fb.as_mut().poll(&mut cx), Poll::Pending));

        // The device completes *pre-hand-back*: EventWaker enqueues the completion
        // edge, then notifies — while the future is still held by the "driver".
        dev.gate.open();

        // The disputed hand-back: give the future back with NO notify.
        ls.slot.lock().fut = Some(fut);

        // B's pre-registered listener buffered the notify: it re-loops, takes the
        // future, and completes the load itself — no hand-back notify needed.
        match fb.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(p)) => assert_eq!(p.id(), PageId(5)),
            other => panic!("joiner must complete after a silent hand-back: {other:?}"),
        }
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Clean));
        assert_eq!(dev.read_count(), 1, "single flight held through the window");
        match fa.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(p)) => assert_eq!(p.id(), PageId(5)),
            other => panic!("driver joins the published outcome: {other:?}"),
        }
    }

    /// T3.1 test 2 — a joiner that *spends* its buffered notification inside the
    /// window (wakes, still sees `fut == None`, re-parks on a fresh listener) WOULD
    /// strand alone — and is rescued by the driver's retained listener. In the real
    /// interleaving the fut-holder and the rescuer are the same task, so the rescuer
    /// is present by construction (the re-loop happens inside the same task
    /// poll, with no suspension point between hand-back and re-drive).
    #[test]
    fn joiner_that_spent_its_notification_in_the_window_is_rescued_by_the_driver() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);

        let mut fa = Box::pin(cache.get_page(r, &file));
        assert!(matches!(fa.as_mut().poll(&mut cx), Poll::Pending));
        let ls = cache.loading_tokens().pop().expect("one Loading token");
        let fut = take_fut(&ls);
        let mut fb = Box::pin(cache.get_page(r, &file));
        assert!(matches!(fb.as_mut().poll(&mut cx), Poll::Pending));
        dev.gate.open();

        // B wakes on the buffered notify *before* the hand-back, finds the future
        // still gone, and re-parks on a fresh, unnotified listener: its one
        // notification is spent.
        assert!(matches!(fb.as_mut().poll(&mut cx), Poll::Pending));

        // Silent hand-back. B alone is now stranded — v1's intuition, pinned:
        ls.slot.lock().fut = Some(fut);
        assert!(
            matches!(fb.as_mut().poll(&mut cx), Poll::Pending),
            "a joiner whose notification was spent in the window cannot proceed alone"
        );

        // The driver's listener — registered before the window — buffered the same
        // notify; it re-drives, completes, and publishes with a notify.
        match fa.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(p)) => assert_eq!(p.id(), PageId(5)),
            other => panic!("driver's retained listener must rescue the load: {other:?}"),
        }
        match fb.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(p)) => assert_eq!(p.id(), PageId(5)),
            other => panic!("joiner wakes on the completion publish: {other:?}"),
        }
        assert_eq!(dev.read_count(), 1);
    }

    /// T3.1 test 3 — the cancellation variant: the driver's task is dropped after
    /// the silent hand-back, taking its buffered notification with it. The joiner
    /// STILL completes on its own: `event_listener` **forwards** a notified-but-
    /// unconsumed listener's notification to the next parked listener when it is
    /// dropped (guaranteed delivery) — a third covering mechanism, beyond the
    /// retained listener (test 2) and the pending-edge writer backstop (which this
    /// test pins as intact-but-unneeded defense-in-depth). Discovered by this test:
    /// the first draft asserted the joiner strands and was refuted by the run.
    #[test]
    fn a_dropped_drivers_buffered_notification_is_inherited_by_the_joiner() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);

        let mut fa = Box::pin(cache.get_page(r, &file));
        assert!(matches!(fa.as_mut().poll(&mut cx), Poll::Pending));
        let ls = cache.loading_tokens().pop().expect("one Loading token");
        let fut = take_fut(&ls);
        let mut fb = Box::pin(cache.get_page(r, &file));
        assert!(matches!(fb.as_mut().poll(&mut cx), Poll::Pending));
        dev.gate.open();
        // B spends its notification inside the window, then the hand-back is silent
        // and the driver's task — with its buffered notification — is dropped.
        assert!(matches!(fb.as_mut().poll(&mut cx), Poll::Pending));
        ls.slot.lock().fut = Some(fut);
        drop(fa);

        // The drop forwarded A's unconsumed notification to B's parked listener:
        // B re-drives and completes the load with no external help.
        match fb.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(p)) => assert_eq!(p.id(), PageId(5)),
            other => panic!("the joiner inherits the dropped driver's notification: {other:?}"),
        }
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Clean));
        assert_eq!(dev.read_count(), 1);

        // One queued drive edge remains — the never-drained INSTALL edge, into
        // which the completion enqueue coalesced (`queued` was still true;
        // this does NOT pin enqueue-before-notify; the
        // dedicated zero-to-one test below does). Drained as a harmless no-op.
        assert_eq!(
            depth(&cache),
            1,
            "the coalesced install edge is still queued"
        );
        assert!(!cache.drive_pending(64), "one batch drains the stale edge");
        assert_eq!(
            dev.read_count(),
            1,
            "the drained edge issues no second read"
        );
    }

    /// T3.1 test 4 — the queue oracle, corrected: the
    /// install edge is drained FIRST (clearing `queued`), so the later gate-open
    /// provably enqueues the **completion** edge as a zero-to-one depth transition —
    /// and that edge alone rescues a joiner after every listener notification has
    /// been spent or dropped. The stranded state is constructed by exhausting B's
    /// notifications and dropping A's; it is unreachable in
    /// production `drive_load` control flow (the fut-holder re-drives within its own
    /// poll), but the writer backstop must hold regardless of reachability.
    #[test]
    fn completion_edge_enqueues_zero_to_one_and_rescues_a_fully_stranded_joiner() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);

        // A drives once (issues the gated read, hands back, parks), then the
        // writer drains the INSTALL edge while still gated: depth 0, `queued`
        // false. (The drain's own hand-back notify buffers into A's listener.)
        let mut fa = Box::pin(cache.get_page(r, &file));
        assert!(matches!(fa.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(depth(&cache), 1, "install edge");
        assert!(!cache.drive_pending(64), "install edge drained while gated");
        assert_eq!(depth(&cache), 0, "no queued edge before the completion");

        // Window open: the test holds the future; B joins and parks.
        let ls = cache.loading_tokens().pop().expect("one Loading token");
        let fut = take_fut(&ls);
        let mut fb = Box::pin(cache.get_page(r, &file));
        assert!(matches!(fb.as_mut().poll(&mut cx), Poll::Pending));

        // The completion fires pre-hand-back: depth transitions 0 -> 1. THIS pins
        // the completion enqueue (enqueue-before-notify), not a coalesced install.
        dev.gate.open();
        assert_eq!(depth(&cache), 1, "the completion edge enqueued zero-to-one");

        // Exhaust every notification: B spends its own inside the window; dropping
        // A forwards A's (drop-forwarding) and B spends that one too.
        assert!(matches!(fb.as_mut().poll(&mut cx), Poll::Pending));
        drop(fa);
        assert!(matches!(fb.as_mut().poll(&mut cx), Poll::Pending));

        // Silent hand-back into a world with zero notified listeners: B is
        // genuinely stranded (unreachable in production, constructed here).
        ls.slot.lock().fut = Some(fut);
        assert!(matches!(fb.as_mut().poll(&mut cx), Poll::Pending));

        // The completion edge is the backstop: the writer drives, publishes, wakes B.
        assert!(
            !cache.drive_pending(64),
            "one batch serves the completion edge"
        );
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Clean));
        match fb.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(p)) => assert_eq!(p.id(), PageId(5)),
            other => panic!("the queue backstop wakes the stranded joiner: {other:?}"),
        }
        assert_eq!(dev.read_count(), 1);
    }

    /// A device whose read future runs a staged callback (receiving the real
    /// `EventWaker`) on its FIRST poll and returns `Pending`; the second poll
    /// completes. This drives the **production** hand-back path through the exact
    /// disputed window — no privileged slot surgery.
    struct StagedIo {
        bytes: Mutex<Vec<u8>>,
        polls: AtomicU64,
        reads: AtomicU64,
        #[allow(clippy::type_complexity)]
        stage: Mutex<Option<Box<dyn FnMut(&Waker) + Send>>>,
    }

    impl StagedIo {
        fn new(bytes: Vec<u8>) -> Arc<Self> {
            Arc::new(StagedIo {
                bytes: Mutex::new(bytes),
                polls: AtomicU64::new(0),
                reads: AtomicU64::new(0),
                stage: Mutex::new(None),
            })
        }
    }

    struct StagedRead<'a> {
        io: &'a StagedIo,
        len: usize,
    }
    impl Future for StagedRead<'_> {
        type Output = io::Result<Box<[u8]>>;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.io.polls.fetch_add(1, Ordering::SeqCst) == 0 {
                if let Some(mut cb) = self.io.stage.lock().take() {
                    cb(cx.waker());
                }
                Poll::Pending
            } else {
                let b = self.io.bytes.lock();
                Poll::Ready(Ok(b[..self.len].to_vec().into_boxed_slice()))
            }
        }
    }

    impl PageIo for StagedIo {
        fn read_exact_at(&self, _offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            Box::pin(StagedRead { io: self, len })
        }
        fn write_all_at(&self, _o: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult> {
            Box::pin(async move {
                BufResult {
                    result: Ok(()),
                    buffer,
                }
            })
        }
        fn datasync(&self) -> IoFuture<'_, io::Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn size(&self) -> IoFuture<'_, io::Result<u64>> {
            Box::pin(async { Ok(0) })
        }
        fn shutdown(&self) -> IoFuture<'_, io::Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    /// T3.1 test 5 — the disputed window on the PRODUCTION hand-back path:
    /// while the driver is genuinely inside `fut.poll()`, the
    /// staged callback has the joiner observe `fut == None` and park, fires the
    /// real `EventWaker` (completion notify pre-hand-back), and lets the joiner
    /// spend that notification inside the window. The future then returns
    /// `Pending`, `drive_load` executes its real silent hand-back — and
    /// the driver's retained listener re-drives to terminal
    /// completion **within the same outer poll**: the driver's FIRST outer poll
    /// returns `Ready`. Adding a notify at the hand-back could not change this
    /// test's outcome; teeth-checked the other way — registering the listener only
    /// at the await point (so it no longer spans the device poll) fails this test.
    #[test]
    fn silent_handback_re_drives_within_the_same_task_poll() {
        let page = encode_leaf_inline(PageId(5), 1, &[(b"k".to_vec(), b"v5".to_vec())]);
        let r = page.page_ref();
        let dev = StagedIo::new(page.bytes().to_vec());
        let loc = DataFileLocator::new();
        loc.record(5, 0);
        // Leaked to `'static` so the joiner future can live inside the staged
        // callback (test-scoped leak).
        let cache: &'static PageCache = Box::leak(Box::new(PageCache::new()));
        let file: &'static FileGeneration = Box::leak(Box::new(FileGeneration::direct(
            dev.clone() as Arc<dyn PageIo>,
            Arc::new(loc),
        )));

        // A's creation installs the load; B's creation joins it. B is parked in a
        // shared slot the staged callback can reach.
        let mut fa = Box::pin(cache.get_page(r, file));
        type Joiner = Pin<Box<dyn Future<Output = Result<Arc<Page>, ReadError>> + Send>>;
        let fb: Arc<Mutex<Option<Joiner>>> =
            Arc::new(Mutex::new(Some(Box::pin(cache.get_page(r, file)))));

        let fb_in_window = Arc::clone(&fb);
        *dev.stage.lock() = Some(Box::new(move |event_waker: &Waker| {
            let noop = Waker::from(Arc::new(Noop));
            let mut bcx = Context::from_waker(&noop);
            let mut guard = fb_in_window.lock();
            let b = guard.as_mut().expect("joiner staged");
            // The joiner observes `fut == None` (the driver is mid-poll) and parks.
            assert!(matches!(b.as_mut().poll(&mut bcx), Poll::Pending));
            // The completion notify fires pre-hand-back, on the real EventWaker.
            event_waker.wake_by_ref();
            // The joiner spends its buffered notification inside the window and
            // re-parks on a fresh, unnotified listener.
            assert!(matches!(b.as_mut().poll(&mut bcx), Poll::Pending));
        }));

        // The driver's FIRST outer poll: staged Pending -> real silent hand-back ->
        // retained listener already notified -> re-loop -> second device poll is
        // Ready -> complete_load -> publish. All within this one poll.
        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        match fa.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(p)) => assert_eq!(p.id(), PageId(5)),
            other => panic!("the driver must re-drive within its own poll: {other:?}"),
        }
        assert!(dev.stage.lock().is_none(), "the staged window ran");
        assert_eq!(
            dev.polls.load(Ordering::SeqCst),
            2,
            "Pending once, Ready once"
        );
        assert_eq!(dev.reads.load(Ordering::Relaxed), 1, "single flight");

        // The publish notify wakes the joiner spent inside the window.
        let mut guard = fb.lock();
        let b = guard.as_mut().expect("joiner still staged");
        match b.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(p)) => assert_eq!(p.id(), PageId(5)),
            other => panic!("the joiner wakes on the completion publish: {other:?}"),
        }
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Clean));
    }

    /// `drive_pending` polls **at most `budget`** loads per call and
    /// reports `more` so the writer re-loops, and FIFO order eventually services
    /// *every* token (no hash-order starvation). Pins the latency bound (T1) and
    /// fairness (T2).
    #[test]
    fn drive_pending_is_bounded_and_services_every_token() {
        let dev = ProgIo::new(false); // ungated: each drives to Clean in one poll
        let loc = DataFileLocator::new();
        let cache = PageCache::new();
        let k = 50u64;
        let mut refs = Vec::new();
        for i in 1..=k {
            refs.push(leaf_at(&dev, &loc, i, i * PAGE_SIZE as u64));
        }
        let file = file_with(dev.clone(), loc);

        // Install K never-polled orphans.
        for r in &refs {
            let f = Box::pin(cache.get_page(*r, &file));
            drop(f);
        }
        assert_eq!(depth(&cache), k as usize, "one edge per install");

        // A bounded batch polls at most `budget` and reports more remain.
        let budget = 8usize;
        let before = cache.drive_polls.load(Ordering::Relaxed);
        assert!(
            cache.drive_pending(budget),
            "more work remains after one bounded batch"
        );
        let polled = cache.drive_polls.load(Ordering::Relaxed) - before;
        assert_eq!(
            polled, budget as u64,
            "polled exactly `budget` loads, not the whole set"
        );

        // Drain the rest; every token is eventually serviced (FIFO fairness).
        while cache.drive_pending(budget) {}
        for i in 1..=k {
            assert_eq!(
                cache.state_of(PageId(i)),
                Some(CacheState::Clean),
                "token {i} was serviced"
            );
        }
        assert_eq!(depth(&cache), 0);
    }

    /// The headline anti-regression — under N separated slow cold
    /// loads, the writer's aggregate drive work is **O(N)**, not the old full sweep's
    /// **O(N²)**. Model the writer reacting to each install with a bounded drain while
    /// the device stays gated (loads accumulate as `Loading`): the edge-targeted queue
    /// polls ~one load per install (the new edge), whereas a full `Loading`-set rescan
    /// would pay Σ(1..N) = N(N+1)/2. A regression to the full sweep fails this bound.
    #[test]
    fn pending_drive_aggregate_is_linear_not_quadratic() {
        let dev = ProgIo::new(true); // GATED: never-completing reads accumulate as Loading
        let loc = DataFileLocator::new();
        let cache = PageCache::new();
        let n = 200u64;
        let mut refs = Vec::new();
        for i in 1..=n {
            refs.push(leaf_at(&dev, &loc, i, i * PAGE_SIZE as u64));
        }
        let file = file_with(dev.clone(), loc);

        // Install each orphan, then the writer reacts with a bounded drain — exactly
        // the loop the real writer runs on each idle/activity wake.
        let mut futs = Vec::new();
        for r in &refs {
            futs.push(Box::pin(cache.get_page(*r, &file))); // never polled -> orphan
                                                            // The new edge (and only it) is in the queue; drain it. The load polls
                                                            // Pending (gated) and is handed back, NOT re-enqueued.
            while cache.drive_pending(64) {}
        }
        assert_eq!(cache.state_of(PageId(1)), Some(CacheState::Loading));
        let polls = cache.drive_polls.load(Ordering::Relaxed);
        assert!(
            polls <= 4 * n,
            "aggregate drive work must be linear: {polls} polls for {n} installs \
             (a full-sweep regression would pay ~{}); linear cap {}",
            n * (n + 1) / 2,
            4 * n
        );

        // Completeness still holds: open the gate, drain, every token reaches Clean.
        dev.gate.open();
        while cache.drive_pending(64) {}
        for i in 1..=n {
            assert_eq!(
                cache.state_of(PageId(i)),
                Some(CacheState::Clean),
                "token {i} serviced"
            );
        }
        drop(futs);
    }

    /// The `queued` dedup flag coalesces duplicate/spurious edges:
    /// re-enqueuing the same load while it is already queued adds no second node, so
    /// a wake storm cannot grow the queue past one node per not-yet-drained load.
    #[test]
    fn dedup_flag_coalesces_duplicate_edges() {
        let dev = ProgIo::new(true); // gated: stays Loading so the token remains alive
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        let f0 = Box::pin(cache.get_page(r, &file)); // install -> one edge
        assert_eq!(depth(&cache), 1);
        let ls = Arc::downgrade(&cache.loading_tokens()[0]);
        // Many spurious re-enqueue attempts while already queued: all coalesce.
        for _ in 0..100 {
            cache.pending.enqueue(&ls);
        }
        assert_eq!(
            depth(&cache),
            1,
            "the dedup flag coalesced 100 edges to one node"
        );
        drop(f0);
    }

    /// The defensive `Busy` branch — when `drive_pending` pops a
    /// drive edge while a **concurrent driver holds the future** (`fut.take() ==
    /// None`), it must **re-enqueue** the edge (not consume it as a no-op) so the
    /// drive is not lost, and — since the whole batch made no progress — it must
    /// **park** (return `false`), not force a re-loop that would hot-spin on the held
    /// (or torn) node. Deterministically stage the
    /// Busy state by taking the future out of the slot, then restore it and confirm
    /// the re-enqueued edge completes the load.
    #[test]
    fn busy_edge_is_reenqueued_not_lost() {
        let dev = ProgIo::new(false); // ungated: completes once the fut is restored + polled
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        // Install (enqueues the install edge), then grab the token and simulate a
        // concurrent driver by taking the future out of the slot.
        let f0 = Box::pin(cache.get_page(r, &file));
        let ls = cache.loading_tokens().pop().expect("one loading token");
        let held = ls.slot.lock().fut.take().expect("future present");
        drop(f0); // orphan; the edge is still queued

        // drive_pending pops the edge, finds the future absent (Busy), and re-enqueues
        // — but the all-Busy batch made no progress, so it parks (no hot spin).
        assert!(
            !cache.drive_pending(64),
            "an all-Busy batch parks, it does not spin"
        );
        assert_eq!(
            depth(&cache),
            1,
            "the edge was re-enqueued, not consumed as a no-op"
        );
        assert_eq!(
            cache.state_of(PageId(5)),
            Some(CacheState::Loading),
            "still not driven"
        );

        // The concurrent driver hands the future back; the next drain completes it.
        ls.slot.lock().fut = Some(held);
        assert!(!cache.drive_pending(64));
        assert_eq!(
            cache.state_of(PageId(5)),
            Some(CacheState::Clean),
            "the re-enqueued edge drove the load once the future returned"
        );
        assert_eq!(depth(&cache), 0);
    }

    /// A **torn** load (a driver that
    /// vanished between `fut.take()` and its hand-back, e.g. via a panic-unwind,
    /// leaving `done == None, fut == None` forever) must **not** hot-spin the writer.
    /// `drive_pending` re-enqueues the Busy edge but, having made no progress, returns
    /// `false` so the writer parks instead of re-looping — the graceful degradation
    /// the pre-queue full sweep had. A regression that forces `more` on an all-Busy
    /// batch spins forever here (the load never becomes drivable).
    #[test]
    fn a_torn_load_does_not_spin_the_drive() {
        let dev = ProgIo::new(true); // gated: the read never completes on its own
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        // Orphan a load, then permanently tear it: take the future and never return it
        // (models a driver that unwound mid-poll).
        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut f0 = Box::pin(cache.get_page(r, &file));
        assert!(matches!(f0.as_mut().poll(&mut cx), Poll::Pending));
        let _torn = cache.loading_tokens().pop().unwrap().slot.lock().fut.take();
        drop(f0);

        // Every drive is a no-progress all-Busy batch → parks (false), never spins.
        for _ in 0..5 {
            assert!(
                !cache.drive_pending(64),
                "a torn load must never force a re-loop"
            );
        }
        assert_eq!(
            depth(&cache),
            1,
            "the torn load rests as a bounded queued residue"
        );
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Loading));
    }

    /// H7 (supersedes the H5 drop behaviour for the close/poison drain) — the drain
    /// completes a corrupt-ref-over-good-bytes orphan *reference-independently* and
    /// caches the healthy page as `Clean`; no poison, because every hit re-validates
    /// against the requester's ref. Same property as the sweep, through
    /// `drain_parked_loads`, and needing no `DrivePolicy`.
    #[test]
    fn drain_of_a_corrupt_ref_orphan_caches_clean_without_poison() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let good = leaf_at(&dev, &loc, 5, 0);
        let corrupt = PageRef {
            page_id: good.page_id,
            expected_checksum: crate::page::Checksum(good.expected_checksum.0 ^ 0xFFFF_FFFF),
        };
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut f0 = Box::pin(cache.get_page(corrupt, &file));
        assert!(matches!(f0.as_mut().poll(&mut cx), Poll::Pending));
        drop(f0);

        dev.gate.open();
        block_on(cache.drain_parked_loads());
        assert_eq!(
            cache.state_of(PageId(5)),
            Some(CacheState::Clean),
            "the drain decoded ref-independently and cached the healthy page as Clean"
        );

        // No poison: good ref reads the page, corrupt ref still mismatches — no re-read.
        let before = dev.read_count();
        assert!(matches!(
            block_on(cache.get_page(corrupt, &file)),
            Err(ReadError::Corrupt(_))
        ));
        let page = block_on(cache.get_page(good, &file)).expect("good ref reads the healthy page");
        assert_eq!(
            page.leaf_get(b"k").unwrap().to_owned_value(),
            crate::page::EntryValue::Inline(b"v5".to_vec())
        );
        assert_eq!(
            dev.read_count(),
            before,
            "both served from the cached Clean image, no re-read"
        );
    }

    // ---- H7 cross-ref single-flight authority ----------------------------------------
    //
    // The single-flight join merges by `page_id`; before H7 a joiner returned the shared
    // decoded verdict verbatim, so a joiner with a *different* `expected_checksum` could
    // inherit a success/corruption decided against another ref. These tests pin that each
    // participant now validates against its **own** ref, across the install/join orderings
    // and every completion path (real reader, idle sweep). Each would fail pre-H7.

    /// Good-ref installer + wrong-ref joiner, completed by the **wrong-ref reader**: the
    /// wrong ref must get its own checksum mismatch (not the good verdict), the good ref
    /// must get `Ok`, and there is exactly one physical read. Pre-H7 the wrong-ref joiner
    /// returned the completer's `Ok(page)` — a false success across a checksum boundary.
    #[test]
    fn wrong_ref_join_gets_own_verdict_not_the_good_one() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let good = leaf_at(&dev, &loc, 5, 0);
        let corrupt = PageRef {
            page_id: good.page_id,
            expected_checksum: crate::page::Checksum(good.expected_checksum.0 ^ 0xFFFF_FFFF),
        };
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        // Good ref INSTALLS the load (begin runs at construction) and stays alive as a
        // waiter (poll once → Pending, issues the single read).
        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut good_fut = Box::pin(cache.get_page(good, &file));
        assert!(matches!(good_fut.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(dev.read_count(), 1);

        // Wrong ref JOINS the same in-flight load and (after the gate opens) completes it.
        let corrupt_reader = {
            let (c, f) = (cache.clone(), file.clone());
            std::thread::spawn(move || block_on(c.get_page(corrupt, &f)))
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        dev.gate.open();
        let corrupt_res = corrupt_reader.join().unwrap();
        assert!(
            matches!(corrupt_res, Err(ReadError::Corrupt(_))),
            "the wrong-ref joiner gets its OWN checksum mismatch, not the good Ok"
        );

        // The good installer, polled to completion, gets the page (projected to its ref).
        match good_fut.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(page)) => assert_eq!(
                page.leaf_get(b"k").unwrap().to_owned_value(),
                crate::page::EntryValue::Inline(b"v5".to_vec())
            ),
            other => panic!("good ref must read Ok, got {other:?}"),
        }
        assert_eq!(
            dev.read_count(),
            1,
            "single-flight: one physical read for both refs"
        );
        assert_eq!(
            cache.metrics().load_errors,
            0,
            "a successful physical read + a per-ref checksum mismatch is NOT a load error"
        );
        assert!(
            cache.metrics().load_joins >= 1,
            "the wrong ref joined the in-flight load"
        );
    }

    /// Wrong-ref installer + good-ref joiner, completed by the **good-ref reader**: the
    /// good ref must decode successfully (not inherit the installer's corruption), the
    /// wrong installer gets its own mismatch, and the healthy page is cached `Clean`
    /// (not dropped). Pre-H7 the good joiner inherited the installer's `Err(Corrupt)` —
    /// a false corruption across a checksum boundary.
    #[test]
    fn good_ref_join_over_a_wrong_ref_install_decodes_successfully() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let good = leaf_at(&dev, &loc, 5, 0);
        let corrupt = PageRef {
            page_id: good.page_id,
            expected_checksum: crate::page::Checksum(good.expected_checksum.0 ^ 0xFFFF_FFFF),
        };
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        // Wrong ref INSTALLS the load and stays alive as a waiter.
        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut corrupt_fut = Box::pin(cache.get_page(corrupt, &file));
        assert!(matches!(corrupt_fut.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(dev.read_count(), 1);

        // Good ref JOINS and completes the load.
        let good_reader = {
            let (c, f) = (cache.clone(), file.clone());
            std::thread::spawn(move || block_on(c.get_page(good, &f)))
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        dev.gate.open();
        let good_res = good_reader.join().unwrap();
        let page = good_res.expect("the good joiner decodes the healthy page with its own ref");
        assert_eq!(
            page.leaf_get(b"k").unwrap().to_owned_value(),
            crate::page::EntryValue::Inline(b"v5".to_vec())
        );

        // The wrong installer, polled to completion, gets its own mismatch.
        assert!(matches!(
            corrupt_fut.as_mut().poll(&mut cx),
            Poll::Ready(Err(ReadError::Corrupt(_)))
        ));
        assert_eq!(
            cache.state_of(PageId(5)),
            Some(CacheState::Clean),
            "the healthy page is cached Clean, not dropped, even though the installer's ref was wrong"
        );
        assert_eq!(dev.read_count(), 1, "single-flight: one physical read");
    }

    /// Same cross-ref orderings, but the load is completed by a **maintenance sweep**
    /// (the idle-drive path) while both the good installer and the wrong joiner are
    /// parked. Each must still get its own per-ref verdict from the sweep-published
    /// shared outcome.
    #[test]
    fn a_sweep_completion_still_yields_per_ref_verdicts() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        let good = leaf_at(&dev, &loc, 5, 0);
        let corrupt = PageRef {
            page_id: good.page_id,
            expected_checksum: crate::page::Checksum(good.expected_checksum.0 ^ 0xFFFF_FFFF),
        };
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        // Good installs, wrong joins; both poll once to Pending and PARK (neither drives).
        let mut good_fut = Box::pin(cache.get_page(good, &file));
        assert!(matches!(good_fut.as_mut().poll(&mut cx), Poll::Pending));
        let mut corrupt_fut = Box::pin(cache.get_page(corrupt, &file));
        assert!(matches!(corrupt_fut.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(dev.read_count(), 1, "one read issued, now parked");

        // The sweep (maintenance) completes the parked load.
        dev.gate.open();
        cache.sweep_parked_loads();
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Clean));

        // Each parked participant now projects the sweep's shared outcome through its ref.
        match good_fut.as_mut().poll(&mut cx) {
            Poll::Ready(Ok(page)) => assert_eq!(
                page.leaf_get(b"k").unwrap().to_owned_value(),
                crate::page::EntryValue::Inline(b"v5".to_vec())
            ),
            other => panic!("good ref must read Ok after the sweep, got {other:?}"),
        }
        assert!(matches!(
            corrupt_fut.as_mut().poll(&mut cx),
            Poll::Ready(Err(ReadError::Corrupt(_)))
        ));
        assert_eq!(dev.read_count(), 1, "the sweep issued no second read");
    }

    /// A short read (truncation — an allocated-but-unflushed id) is transient: it caches
    /// no verdict, so a later good ref to the same id — once the writer has flushed the
    /// real bytes — reads successfully rather than inheriting a poisoned sticky failure.
    /// This is the H7 choice (transient over sticky) that removes the latent
    /// truncation poison.
    #[test]
    fn a_truncated_read_does_not_poison_a_later_good_ref() {
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        let page = encode_leaf_inline(PageId(7), 1, &[(b"k".to_vec(), b"v7".to_vec())]);
        let good = page.page_ref();
        // Seed a TRUNCATED image (fewer than a page) → the device short-reads.
        dev.seed(0, page.bytes()[..100].to_vec());
        loc.record(7, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        assert!(matches!(
            block_on(cache.get_page(good, &file)),
            Err(ReadError::Corrupt(_))
        ));
        assert_eq!(
            cache.state_of(PageId(7)),
            None,
            "a short read is transient — cached by no one"
        );

        // The writer flushes the real bytes; a later good read must succeed (no poison).
        dev.seed(0, page.bytes().to_vec());
        let got =
            block_on(cache.get_page(good, &file)).expect("good ref reads the now-flushed page");
        assert_eq!(
            got.leaf_get(b"k").unwrap().to_owned_value(),
            crate::page::EntryValue::Inline(b"v7".to_vec())
        );
        assert_eq!(
            dev.read_count(),
            2,
            "re-read after the transient short read"
        );
    }

    /// Two joiners of the SAME in-flight read of a **structurally malformed** page, with
    /// different checksums, must each get their own projection of the shared `Malformed`
    /// outcome: the authoritative-crc ref gets the structural reason, the other gets a
    /// plain checksum mismatch — from one physical read, counted once in `load_errors`.
    /// This pins the `project_shared` `Malformed` branch on the in-flight join path (the
    /// cross-ref tests above exercise only `Decoded`).
    #[test]
    fn malformed_join_projects_per_ref() {
        let dev = ProgIo::new(true); // gated
        let loc = DataFileLocator::new();
        // Structurally malformed bytes (broken magic) with a known actual crc `c`.
        let page = encode_leaf_inline(PageId(9), 1, &[(b"k".to_vec(), b"v9".to_vec())]);
        let mut bad = page.bytes().to_vec();
        bad[0] ^= 0xFF; // break the magic
        let c = crate::page::Checksum::crc32c(&bad);
        dev.seed(0, bad);
        loc.record(9, 0);
        let auth = PageRef {
            page_id: PageId(9),
            expected_checksum: c,
        }; // authoritative for the malformed bytes
        let other = PageRef {
            page_id: PageId(9),
            expected_checksum: crate::page::Checksum(c.0 ^ 1),
        };
        let cache = Arc::new(PageCache::new());
        let file = Arc::new(file_with(dev.clone(), loc));

        // `auth` installs the load and parks; `other` joins and completes it.
        let waker = Waker::from(Arc::new(Noop));
        let mut cx = Context::from_waker(&waker);
        let mut auth_fut = Box::pin(cache.get_page(auth, &file));
        assert!(matches!(auth_fut.as_mut().poll(&mut cx), Poll::Pending));
        let other_reader = {
            let (c, f) = (cache.clone(), file.clone());
            std::thread::spawn(move || block_on(c.get_page(other, &f)))
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        dev.gate.open();

        // The joiner's ref does not match the bytes' crc → plain checksum mismatch.
        assert!(matches!(
            other_reader.join().unwrap(),
            Err(ReadError::Corrupt("page checksum mismatch"))
        ));
        // The authoritative ref projects the SAME shared `Malformed` to the structural
        // reason — not a checksum mismatch.
        assert!(matches!(
            auth_fut.as_mut().poll(&mut cx),
            Poll::Ready(Err(ReadError::Corrupt("bad page magic")))
        ));
        assert_eq!(dev.read_count(), 1, "one physical read for both joiners");
        assert_eq!(
            cache.metrics().load_errors,
            1,
            "one malformed physical read counts once, regardless of joiner count"
        );
    }

    /// H5 — a reader that takes the slot lock in the window
    /// between the sweep's `Pending` poll and its future hand-back observes
    /// `fut == None`, registers its listener, and awaits; without the sweep's
    /// notify-after-hand-back it could miss the device's (lost) completion wake and
    /// hang. Stress the sweep-vs-reader race; a lost wakeup shows as the reader
    /// never delivering, which the receive timeout converts into a clean failure
    /// rather than a hung suite.
    #[test]
    fn sweep_never_strands_a_concurrent_reader() {
        use std::sync::mpsc;
        for _ in 0..300 {
            let dev = ProgIo::new(true); // gated
            let loc = DataFileLocator::new();
            let r = leaf_at(&dev, &loc, 5, 0);
            let cache = Arc::new(PageCache::new());
            let file = Arc::new(file_with(dev.clone(), loc));

            // Install a parked load (driver dropped mid-load).
            let waker = Waker::from(Arc::new(Noop));
            let mut cx = Context::from_waker(&waker);
            let mut f0 = Box::pin(cache.get_page(r, &file));
            assert!(matches!(f0.as_mut().poll(&mut cx), Poll::Pending));
            drop(f0);

            // A reader joins the in-flight load; a sweeper races it; the gate opens.
            let (tx, rx) = mpsc::channel();
            let reader = {
                let (cache, file) = (cache.clone(), file.clone());
                std::thread::spawn(move || {
                    let v = block_on(cache.get_page(r, &file)).unwrap();
                    tx.send(v.leaf_get(b"k").unwrap().to_owned_value()).unwrap();
                })
            };
            let sweeper = {
                let cache = cache.clone();
                std::thread::spawn(move || {
                    for _ in 0..50 {
                        cache.sweep_parked_loads();
                    }
                })
            };
            dev.gate.open();
            let got = rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("reader delivered — a lost wakeup would time out here");
            assert_eq!(got, crate::page::EntryValue::Inline(b"v5".to_vec()));
            reader.join().unwrap();
            sweeper.join().unwrap();
        }
    }

    /// T1.4: the negative population is bounded at
    /// `NEGATIVE_CAPACITY`; the oldest verdict is evicted (FIFO), an evicted id
    /// re-reads (cold path), a resident one answers without I/O via
    /// `negative_hits` — and none of it touches `hits`/`misses`.
    #[test]
    fn negative_population_is_bounded_and_evicted_verdicts_reread() {
        const EXTRA: usize = 5;
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        let cache = PageCache::new();

        // NEGATIVE_CAPACITY + EXTRA distinct structurally-corrupt pages, each
        // with an authoritative (crc-of-the-bad-bytes) ref.
        let mut refs = Vec::new();
        for i in 0..(NEGATIVE_CAPACITY + EXTRA) {
            let id = PageId(1000 + i as u64);
            let page = encode_leaf_inline(id, 1, &[(b"k".to_vec(), b"v".to_vec())]);
            let mut bad = page.bytes().to_vec();
            bad[0] ^= 0xFF;
            let r = PageRef {
                page_id: id,
                expected_checksum: crate::page::Checksum::crc32c(&bad),
            };
            let off = (i * crate::page::PAGE_SIZE) as u64;
            dev.seed(off, bad);
            loc.record(id.as_u64(), off);
            refs.push(r);
        }
        let file = file_with(dev.clone(), loc);
        for r in &refs {
            let err = block_on(cache.get_page(*r, &file)).unwrap_err();
            assert!(matches!(err, ReadError::Corrupt(_)));
        }

        let m = cache.metrics();
        assert_eq!(
            m.failed_resident, NEGATIVE_CAPACITY as u64,
            "negative residency must be bounded"
        );
        assert_eq!(m.negative_evictions, EXTRA as u64, "FIFO evictions counted");
        // The oldest EXTRA ids were evicted; the newest are resident.
        assert_eq!(
            cache.state_of(refs[0].page_id),
            None,
            "oldest verdict evicted"
        );
        assert_eq!(
            cache.state_of(refs[NEGATIVE_CAPACITY + EXTRA - 1].page_id),
            Some(CacheState::Failed)
        );

        // (b) An evicted verdict re-reads: a fresh physical load, not a map answer.
        let reads_before = dev.read_count();
        let err = block_on(cache.get_page(refs[0], &file)).unwrap_err();
        assert!(matches!(err, ReadError::Corrupt(_)));
        assert_eq!(
            dev.read_count(),
            reads_before + 1,
            "an evicted verdict must be re-derived by a real read"
        );

        // (c) A resident verdict answers without I/O and counts ONLY negative_hits.
        let before = cache.metrics();
        let reads_before = dev.read_count();
        let last = refs[NEGATIVE_CAPACITY + EXTRA - 1];
        let err = block_on(cache.get_page(last, &file)).unwrap_err();
        assert!(matches!(err, ReadError::Corrupt(_)));
        let after = cache.metrics();
        assert_eq!(dev.read_count(), reads_before, "resident verdict is no-I/O");
        assert_eq!(after.negative_hits, before.negative_hits + 1);
        assert_eq!(after.hits, before.hits, "a negative answer is not a hit");
        assert_eq!(after.misses, before.misses, "…nor a miss");
    }

    /// T1.4: `insert_dirty`'s documented overwrite of a stale `Failed` verdict
    /// (the H7 pre-flush-garbage repair path) maintains the gauge, and
    /// the verdict's stale ring entry is skipped when later popped.
    #[test]
    fn insert_dirty_over_failed_maintains_the_gauge() {
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        let cache = PageCache::new();
        let id = PageId(77);
        let page = encode_leaf_inline(id, 1, &[(b"k".to_vec(), b"v".to_vec())]);
        let mut bad = page.bytes().to_vec();
        bad[0] ^= 0xFF;
        let r = PageRef {
            page_id: id,
            expected_checksum: crate::page::Checksum::crc32c(&bad),
        };
        dev.seed(0, bad);
        loc.record(id.as_u64(), 0);
        let file = file_with(dev.clone(), loc);
        block_on(cache.get_page(r, &file)).unwrap_err();
        assert_eq!(cache.metrics().failed_resident, 1);

        cache.insert_dirty(Arc::new(page.clone()));
        let m = cache.metrics();
        assert_eq!(m.failed_resident, 0, "overwrite decrements the gauge");
        assert_eq!(cache.state_of(id), Some(CacheState::Dirty));
        // The real image now answers the good ref.
        let good = PageRef {
            page_id: id,
            expected_checksum: page.checksum(),
        };
        let got = block_on(cache.get_page(good, &file)).unwrap();
        assert_eq!(got.checksum(), page.checksum());
    }

    #[test]
    fn corrupt_page_is_cached_failed() {
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        // Structurally corrupt the page (break its magic, byte 0) and make the ref
        // AUTHORITATIVE for the corrupt bytes: `expected_checksum = crc32c(bad)`. The
        // checksum gate passes, so the requester is the true owner of these bytes and
        // hits a genuine *structural* fault → a crc-keyed sticky `Failed` (H7). (Pre-H7
        // this test kept the good checksum and relied on a *checksum* failure; under H7
        // that is a structurally-valid page cached `Clean` and rejected per-ref, so the
        // test must exercise real structural corruption to reach `Failed`.)
        let page = encode_leaf_inline(PageId(9), 1, &[(b"k".to_vec(), b"v".to_vec())]);
        let mut bad = page.bytes().to_vec();
        bad[0] ^= 0xFF; // break the magic → structural decode failure
        let actual = crate::page::Checksum::crc32c(&bad);
        let r = PageRef {
            page_id: PageId(9),
            expected_checksum: actual,
        };
        dev.seed(0, bad);
        loc.record(9, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        let err = block_on(cache.get_page(r, &file)).unwrap_err();
        assert!(matches!(err, ReadError::Corrupt(_)));
        assert_eq!(cache.state_of(PageId(9)), Some(CacheState::Failed));
        // Repeat get is served from the crc-keyed Failed verdict; no new read.
        let err2 = block_on(cache.get_page(r, &file)).unwrap_err();
        assert!(matches!(err2, ReadError::Corrupt(_)));
        assert_eq!(dev.read_count(), 1, "corrupt verdict is cached");
        // A *different*-checksum ref to the same id gets a derived checksum mismatch from
        // the crc-keyed Failed entry — still no re-read, and not the structural reason.
        let other = PageRef {
            page_id: PageId(9),
            expected_checksum: crate::page::Checksum(actual.0 ^ 1),
        };
        let err3 = block_on(cache.get_page(other, &file)).unwrap_err();
        assert!(matches!(err3, ReadError::Corrupt("page checksum mismatch")));
        assert_eq!(
            dev.read_count(),
            1,
            "differently-checksummed ref answered without I/O"
        );
    }

    #[test]
    fn io_error_is_not_cached() {
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        // Record an id but seed no bytes → device returns EIO.
        let r = PageRef {
            page_id: PageId(4),
            expected_checksum: crate::page::Checksum(123),
        };
        loc.record(4, 0);
        let cache = PageCache::new();
        let file = file_with(dev.clone(), loc);

        assert!(matches!(
            block_on(cache.get_page(r, &file)),
            Err(ReadError::Io(_))
        ));
        assert_eq!(
            cache.state_of(PageId(4)),
            None,
            "transient error not cached"
        );
        let _ = block_on(cache.get_page(r, &file));
        assert_eq!(dev.read_count(), 2, "each get re-issues after an I/O error");
    }

    /// Eviction bounds the clean tier on the **load path** (`get_page` →
    /// `complete_load` → qc). The bound is per-shard (see `with_capacity`):
    /// residency stays within `capacity + shards`, and enough distinct pages are
    /// loaded that eviction demonstrably ran.
    #[test]
    fn eviction_bounds_clean_residency() {
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        let mut refs = Vec::new();
        for i in 1..=100u64 {
            refs.push(leaf_at(&dev, &loc, i, (i - 1) * PAGE_SIZE_U64));
        }
        let cap = 4usize;
        let cache = PageCache::with_capacity(cap);
        let file = file_with(dev.clone(), loc);
        for r in &refs {
            let _ = block_on(cache.get_page(*r, &file)).unwrap();
        }
        let m = cache.metrics();
        let shards = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        assert!(
            m.clean_resident <= (cap + shards) as u64,
            "clean tier bounded by capacity + per-shard overshoot: {}",
            m.clean_resident
        );
        assert!(m.evictions > 0, "evicted the overflow: {}", m.evictions);
    }

    #[test]
    fn dirty_pages_are_pinned_against_eviction() {
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        let cache = PageCache::with_capacity(2);
        // Pin a dirty page.
        let dirty = Arc::new(
            Page::decode(
                encode_leaf_inline(PageId(100), 1, &[(b"d".to_vec(), b"v".to_vec())])
                    .bytes()
                    .to_vec()
                    .into(),
                encode_leaf_inline(PageId(100), 1, &[(b"d".to_vec(), b"v".to_vec())]).page_ref(),
            )
            .unwrap(),
        );
        cache.insert_dirty(dirty);
        assert_eq!(cache.state_of(PageId(100)), Some(CacheState::Dirty));
        // Now churn clean pages past capacity; the dirty page must remain.
        let mut refs = Vec::new();
        for i in 1..=6u64 {
            refs.push(leaf_at(&dev, &loc, i, (i - 1) * PAGE_SIZE_U64));
        }
        let file = file_with(dev.clone(), loc);
        for r in &refs {
            let _ = block_on(cache.get_page(*r, &file)).unwrap();
        }
        assert_eq!(
            cache.state_of(PageId(100)),
            Some(CacheState::Dirty),
            "dirty page survives clean-tier churn"
        );
        // Release it to clean; it now counts against and can be evicted.
        cache.release_dirty(PageId(100));
        assert_eq!(cache.state_of(PageId(100)), Some(CacheState::Clean));
    }

    const PAGE_SIZE_U64: u64 = crate::page::PAGE_SIZE as u64;

    // ---- M6 Phase G / G3: the quick_cache clean tier ------------------------
    // Barrier tests for exactly the cross-tier transition races: with the clean
    // image outside the map, cross-tier state must
    // never tear. They reuse this module's helpers (ProgIo/LateIo/leaf_at/
    // file_with), which is why they live here.

    fn quick(cap: usize) -> Arc<PageCache> {
        Arc::new(PageCache::with_capacity(cap))
    }

    fn qpage(id: u64) -> Arc<Page> {
        Arc::new(encode_leaf_inline(
            PageId(id),
            1,
            &[(b"k".to_vec(), format!("v{id}").into_bytes())],
        ))
    }

    /// A cold load lands in `quick_cache`, and the next get is a **lock-free** hit
    /// served from the clean tier — the whole point of the two-tier design.
    #[test]
    fn quick_load_populates_tier_and_hits() {
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = quick(16);
        let file = file_with(dev.clone(), loc);

        let first = block_on(cache.get_page(r, &file)).unwrap();
        assert_eq!(first.id(), PageId(5));
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Clean));
        assert_eq!(cache.metrics().clean_resident, 1, "clean lives in qc");

        let before = dev.read_count();
        let again = block_on(cache.get_page(r, &file)).unwrap();
        assert_eq!(again.id(), PageId(5));
        assert_eq!(dev.read_count(), before, "warm hit issues no read");
        assert!(cache.metrics().hits >= 1);
    }

    /// `insert_dirty` must invalidate the clean tier: after pinning an id
    /// Dirty it is no longer clean-resident; `release_dirty` returns it to qc.
    #[test]
    fn quick_dirty_invalidates_then_release_repopulates() {
        let cache = quick(16);
        let p = qpage(9);
        cache.insert_dirty(p.clone());
        cache.release_dirty(PageId(9));
        assert_eq!(cache.state_of(PageId(9)), Some(CacheState::Clean));
        assert_eq!(cache.metrics().clean_resident, 1);

        cache.insert_dirty(p.clone());
        assert_eq!(cache.state_of(PageId(9)), Some(CacheState::Dirty));
        assert_eq!(cache.metrics().clean_resident, 0, "qc invalidated on dirty");
        assert_eq!(cache.metrics().dirty_resident, 1);
        assert_eq!(
            cache.metrics().evictions,
            0,
            "invalidation is not an eviction"
        );

        cache.release_dirty(PageId(9));
        assert_eq!(cache.state_of(PageId(9)), Some(CacheState::Clean));
        assert_eq!(cache.metrics().clean_resident, 1);
        assert_eq!(cache.metrics().dirty_resident, 0);
    }

    /// A pinned Dirty image cannot be evicted by clean-tier pressure — it lives in
    /// the map, which `quick_cache` cannot touch.
    #[test]
    fn quick_dirty_pin_survives_pressure() {
        let cache = quick(4);
        cache.insert_dirty(qpage(100));
        for id in 200..264u64 {
            cache.insert_dirty(qpage(id));
            cache.release_dirty(PageId(id));
        }
        assert_eq!(
            cache.state_of(PageId(100)),
            Some(CacheState::Dirty),
            "pinned dirty page is never evicted by clean-tier churn"
        );
        assert!(
            cache.metrics().clean_resident <= 4,
            "clean tier stays bounded"
        );
        assert!(
            cache.metrics().evictions > 0,
            "eviction metric is live under qc"
        );
        cache.release_dirty(PageId(100));
        assert_eq!(cache.state_of(PageId(100)), Some(CacheState::Clean));
    }

    /// The G0 orphaned-load interleaving under the **Quick** tier — where the map's
    /// `None` state is the *normal* steady state of every clean page, i.e. the
    /// configuration that would make the pre-fix bug permanent.
    /// The exact-token fix must hold here too.
    #[test]
    fn quick_orphaned_load_installs_no_verdict() {
        let dev = LateIo::new();
        let loc = DataFileLocator::new();
        loc.record(7, 0);
        dev.seed(0, vec![0u8; 8]);

        let good = encode_leaf_inline(PageId(7), 1, &[(b"k".to_vec(), b"v7".to_vec())]);
        let r7 = good.page_ref();
        let cache = quick(16);
        let file = Arc::new(FileGeneration::direct(
            dev.clone() as Arc<dyn PageIo>,
            Arc::new(loc),
        ));

        let reader = {
            let (cache, file) = (cache.clone(), file.clone());
            std::thread::spawn(move || block_on(cache.get_page(r7, &file)))
        };
        while dev.read_count() == 0 {
            std::thread::yield_now();
        }

        cache.insert_dirty(Arc::new(good.clone()));
        cache.release_dirty(PageId(7));
        assert_eq!(cache.state_of(PageId(7)), Some(CacheState::Clean));

        dev.gate.open();
        assert!(
            matches!(reader.join().unwrap(), Err(ReadError::Corrupt(_))),
            "orphaned load's own waiter still gets its result"
        );
        assert_eq!(
            cache.state_of(PageId(7)),
            Some(CacheState::Clean),
            "the good page stays clean; no sticky Failed under Quick"
        );
        let page = block_on(cache.get_page(r7, &file)).expect("good page still reads");
        assert_eq!(page.id(), PageId(7));
    }

    /// A corrupt page caches a `Failed` verdict (in the map), served without a
    /// clean-tier hit shadowing it (post-fix, `Failed` never coexists with
    /// `Clean` for one id).
    #[test]
    fn quick_corrupt_is_failed_not_shadowed() {
        let dev = ProgIo::new(false);
        let loc = DataFileLocator::new();
        let good = encode_leaf_inline(PageId(9), 1, &[(b"k".to_vec(), b"v".to_vec())]);
        // Structural corruption with an AUTHORITATIVE ref (crc of the corrupt bytes), so
        // the requester owns these bytes and hits a real structural fault → crc-keyed
        // `Failed` in the map, never a `Clean` shadow (H7; see corrupt_page_is_cached_failed).
        let mut bad = good.bytes().to_vec();
        bad[0] ^= 0xFF; // break the magic
        let r = PageRef {
            page_id: PageId(9),
            expected_checksum: crate::page::Checksum::crc32c(&bad),
        };
        dev.seed(0, bad);
        loc.record(9, 0);
        let cache = quick(16);
        let file = file_with(dev.clone(), loc);

        assert!(matches!(
            block_on(cache.get_page(r, &file)),
            Err(ReadError::Corrupt(_))
        ));
        assert_eq!(cache.state_of(PageId(9)), Some(CacheState::Failed));
        assert_eq!(cache.metrics().clean_resident, 0);
        let before = dev.read_count();
        assert!(matches!(
            block_on(cache.get_page(r, &file)),
            Err(ReadError::Corrupt(_))
        ));
        assert_eq!(
            dev.read_count(),
            before,
            "failed verdict served without re-read"
        );
    }

    /// Single-flight is tier-independent (it lives in `drive_load`): under Quick, N
    /// racing waiters over a gated cold page still issue exactly one physical read.
    #[test]
    fn quick_single_flight_holds() {
        let dev = ProgIo::new(true);
        let loc = DataFileLocator::new();
        let r = leaf_at(&dev, &loc, 5, 0);
        let cache = quick(16);
        let file = Arc::new(file_with(dev.clone(), loc));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let (cache, file, r) = (cache.clone(), file.clone(), r);
            handles.push(std::thread::spawn(move || {
                block_on(cache.get_page(r, &file)).unwrap().id()
            }));
        }
        // Establish that all four calls are genuinely racing ONE in-flight load
        // before opening the gate (a bare sleep can let one thread drive
        // the load to completion while the others hit the clean tier, so read_count
        // == 1 would pass without a real race). One thread installs `Loading`; the
        // other three join it, bumping `load_joins`. Spin until all three have.
        while cache.metrics().load_joins < 3 {
            std::thread::yield_now();
        }
        dev.gate.open();
        for h in handles {
            assert_eq!(h.join().unwrap(), PageId(5));
        }
        assert_eq!(dev.read_count(), 1, "exactly one physical read under Quick");
        assert_eq!(
            cache.metrics().load_joins,
            3,
            "all 3 non-driver waiters joined"
        );
        assert_eq!(cache.state_of(PageId(5)), Some(CacheState::Clean));
    }

    /// `Quick`'s clean bound is per-shard, so `capacity` is honoured only up to a
    /// bounded overshoot of `shards − 1`. At a **non-power-of-two**
    /// capacity (65) — the case the other tests' 4/8/16/512 caps conceal — residency
    /// must stay within `capacity + shards`, never grow unbounded. (On a tiny cap the
    /// vendored cache collapses to few shards, so in practice this stays at/near 65;
    /// the assertion pins the *contract*, not an exact number.)
    #[test]
    fn quick_capacity_overshoot_is_bounded() {
        let cap = 65usize;
        let cache = quick(cap);
        let shards = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        for id in 1..2000u64 {
            cache.insert_dirty(qpage(id));
            cache.release_dirty(PageId(id));
        }
        let resident = cache.metrics().clean_resident as usize;
        assert!(
            resident <= cap + shards,
            "clean residency {resident} exceeds cap {cap} + shards {shards} — bound violated"
        );
        assert!(
            cache.metrics().evictions > 0,
            "eviction ran under a non-divisible cap"
        );
    }
}
