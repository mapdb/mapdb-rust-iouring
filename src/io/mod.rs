//! The I/O boundary trait and a deterministic in-memory fake.
//!
//! [`PageIo`] is the single seam the engine uses for durable bytes. M1 decides
//! its production implementation (positional syscall fallback vs. Compio
//! proactor vs. the custom lifecycle prototype); M0 ships only [`FakeIo`], an
//! owned-buffer in-memory device whose futures resolve deterministically.
//!
//! The boundary is intentionally **object-safe** (boxed `Send` futures) so a
//! [`FileGeneration`](crate::backend::FileGeneration) can hold `Arc<dyn PageIo>`
//! members and the production boundary can be swapped at M1 without generics
//! threading through the whole engine.
//!
//! Ownership contract (mirrors the prototype's hard-won lesson and
//! invariant 8): a submitted buffer is **owned by the device** until the
//! returned future resolves. `write_all_at` therefore takes an owned `Box<[u8]>`
//! and hands it back in [`BufResult`]; dropping the caller's future must not (in
//! a real ring) free a buffer the kernel still owns. `FakeIo` completes inline,
//! so this is trivially upheld here, but the *signature* is what the engine is
//! built against.

use std::future::Future;
use std::io;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

mod aligned;
mod syscall;
pub use syscall::SyscallIo;

#[cfg(feature = "io-uring")]
mod uring;
#[cfg(feature = "io-uring")]
#[doc(hidden)]
pub use uring::bench as uring_bench;
#[cfg(feature = "io-uring")]
pub use uring::CompioIo;

#[cfg(test)]
mod device_boundary_tests;

/// Whether io_uring can actually be set up on this host. The `io-uring` feature
/// may compile while a seccomp'd container or an old/locked-down kernel blocks
/// `io_uring_setup` at runtime. Path-backed io_uring tests call this
/// and **skip** rather than panic on `CompioIo::open`. Test-only.
#[cfg(all(test, feature = "io-uring"))]
pub(crate) fn io_uring_available() -> bool {
    let p = std::env::temp_dir().join(format!("mapdb-uring-probe-{}.tmp", std::process::id()));
    let ok = match CompioIo::open(&p) {
        Ok(dev) => {
            let _ = futures_lite::future::block_on(dev.shutdown());
            true
        }
        Err(_) => false,
    };
    let _ = std::fs::remove_file(&p);
    ok
}

/// T3.5 ring-required canary: CI's dedicated ring
/// runner sets `MAPDB_REQUIRE_URING=1`, which turns "io_uring unavailable" from
/// the per-test skip into a hard failure — and on an available ring the canary
/// positively asserts a full Compio round trip (write + datasync + read through
/// a real ring) rather than the ignorable absence of skip text.
#[cfg(all(test, feature = "io-uring"))]
mod ring_canary {
    use super::PageIo as _;

    #[test]
    fn ring_required_canary_round_trips_io_uring() {
        let required = std::env::var_os("MAPDB_REQUIRE_URING").is_some();
        if !super::io_uring_available() {
            assert!(
                !required,
                "MAPDB_REQUIRE_URING is set but io_uring cannot be exercised on this runner"
            );
            eprintln!("skip: io_uring unavailable on this host");
            return;
        }
        let p = std::env::temp_dir().join(format!("mapdb-ring-canary-{}.tmp", std::process::id()));
        let dev = super::CompioIo::open(&p).expect("ring builds on an available host");
        futures_lite::future::block_on(async {
            let payload = vec![0xC7u8; 4096].into_boxed_slice();
            let w = dev.write_all_at(0, payload).await;
            w.result.expect("ring write completes");
            dev.datasync().await.expect("ring datasync completes");
            let bytes = dev
                .read_exact_at(0, 4096)
                .await
                .expect("ring read completes");
            assert!(
                bytes.len() == 4096 && bytes.iter().all(|&b| b == 0xC7),
                "round-tripped bytes match"
            );
        });
        let _ = futures_lite::future::block_on(dev.shutdown());
        let _ = std::fs::remove_file(&p);
        eprintln!("ring canary: io_uring round trip OK");
    }
}

/// Which durable-byte backend a path-backed store mints its devices with (M7-0).
/// `Syscall` is the default and was the only production backend
/// through M6 (`CompioIo` was reachable only from benchmarks — G1). The
/// selector wires `CompioIo` into the production `create_path`/`open_path` path so
/// the io_uring backend can be A/B'd *with checkpoint/compaction intact*, which is
/// G1's #1 reopen precondition.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum IoBackend {
    /// Positional `pread`/`pwrite`/`fdatasync` on a worker pool ([`SyscallIo`]).
    /// Always available; the production default. M7-E closed **retaining** it:
    /// the backend comparison ended UNRESOLVED on a non-quiet host, so the
    /// incumbent stays until a pre-registered re-open condition is met.
    #[default]
    Syscall,
    /// io_uring via Compio's proactor ([`CompioIo`]). Requires the `io-uring`
    /// feature — without it this variant does not exist, so selecting it is a
    /// compile error, not a runtime one. With the feature but on a host where
    /// io_uring is unavailable/blocked, the store is rejected *at open* (the ring
    /// fails to build in `CompioIo::open`).
    #[cfg(feature = "io-uring")]
    Uring,
}

/// A stable, coarse identifier of a concrete [`PageIo`]'s kind — for tests and
/// metrics that must assert *which* backend a path-backed store actually minted
/// at each of its device sites (data file, rotated WAL segments, compaction temp).
/// Not a stable public API; exists so the M7-0 suite-under-Uring and
/// retention-across-rebind tests can catch silent measurement mislabeling —
/// a "Uring" store whose rotated segments are secretly syscall.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IoKind {
    Syscall,
    #[cfg(feature = "io-uring")]
    Uring,
    Fake,
    /// Any other `PageIo` (test doubles); the default.
    Other,
}

/// What a path-backed device is for — decides whether `direct_io` (O_DIRECT)
/// applies to it (M7-A). `Data` and `CompactTemp` are page-granular and
/// O_DIRECT-eligible; `WalSegment` is a byte-granular record stream (wal.rs) and is
/// **always** buffered, since O_DIRECT cannot apply to unaligned record appends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FileRole {
    /// The data file (checkpointed page images + manifest) of a Direct or Wal store.
    Data,
    /// A WAL record segment — byte-granular, never O_DIRECT.
    WalSegment,
    /// A compaction temp file — page-granular like the data file.
    CompactTemp,
}

impl FileRole {
    /// Whether O_DIRECT may apply to this role when `direct_io` is requested.
    fn allows_direct(self) -> bool {
        matches!(self, FileRole::Data | FileRole::CompactTemp)
    }
}

/// The bench-facing mirror of the crate-internal [`FileRole`] (M7-E).
/// Public so an out-of-crate benchmark's device wrapper can tell which
/// site it is decorating — the data file, a rotated WAL segment, or a compaction
/// temp — and keep **per-role** counts. A blind total can certify unequal
/// durability work when it merely *moved between files* (the WAL segment is
/// buffered even in a direct arm), so equal-I/O must be checked per role. Only
/// built under `bench-instrument`; the shipping public API is unchanged without it.
#[cfg(feature = "bench-instrument")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IoRole {
    /// The data file (checkpointed page images + manifest).
    Data,
    /// A WAL record segment — byte-granular, always buffered.
    WalSegment,
    /// A compaction temp file.
    CompactTemp,
}

#[cfg(feature = "bench-instrument")]
impl From<FileRole> for IoRole {
    fn from(r: FileRole) -> Self {
        match r {
            FileRole::Data => IoRole::Data,
            FileRole::WalSegment => IoRole::WalSegment,
            FileRole::CompactTemp => IoRole::CompactTemp,
        }
    }
}

/// A bench-only device decorator (M7-E): given a freshly-minted
/// device and its [`IoRole`], returns a wrapping [`PageIo`] (a counting/tracing
/// shim). Installed via [`Options::device_wrapper`](crate::store::Options) and
/// applied by [`DeviceFactory`] to **every** device the production path mints,
/// **including those minted after open** (rotated WAL segments, compaction temps),
/// because the factory persists in `FsWalFileSet`/`FsDataHost`. This is how the
/// M7-E harness instruments the *production* `create_path`/`open_path` (which mint
/// their device internally) rather than falling back to the caller-supplied-device
/// path, which has no checkpoint. Only built under `bench-instrument`.
#[cfg(feature = "bench-instrument")]
pub type DeviceWrapper = Arc<dyn Fn(Arc<dyn PageIo>, IoRole) -> Arc<dyn PageIo> + Send + Sync>;

/// Mints path-backed devices of the selected [`IoBackend`] and `direct_io` mode.
/// Held (cloned) by the three FS namespace owners — the `create_path`/`open_path`
/// constructors, `FsWalFileSet` (every rotated WAL segment), and `FsDataHost` (the
/// compaction temp) — so a device minted *after* `create_path` returns is still the
/// selected backend/mode. Cheap to clone. `direct_io`
/// is honored per [`FileRole`]: the data file and compaction temp open O_DIRECT,
/// the WAL segment never does (M7-A). Under `bench-instrument` it also
/// carries an optional [`DeviceWrapper`] applied to every minted device (M7-E).
#[derive(Clone)]
pub(crate) struct DeviceFactory {
    backend: IoBackend,
    direct_io: bool,
    /// M7-E bench-only decorator. `None` on the shipping path, so a
    /// plain build mints devices exactly as before. Not `Debug`/`Copy`, which is why
    /// this struct dropped both derives.
    #[cfg(feature = "bench-instrument")]
    wrap: Option<DeviceWrapper>,
}

impl DeviceFactory {
    /// The factory for a store opened with `backend` and `direct_io`.
    pub(crate) fn new(backend: IoBackend, direct_io: bool) -> Self {
        DeviceFactory {
            backend,
            direct_io,
            #[cfg(feature = "bench-instrument")]
            wrap: None,
        }
    }

    /// Install a bench-only device decorator (M7-E). Threaded from
    /// [`Options::device_wrapper`](crate::store::Options) at the four mint sites so
    /// every device — data file, rotated WAL segments, compaction temps — is wrapped.
    #[cfg(feature = "bench-instrument")]
    pub(crate) fn with_wrapper(mut self, wrap: Option<DeviceWrapper>) -> Self {
        self.wrap = wrap;
        self
    }

    /// Applies the bench-only wrapper (if any) to a freshly-minted device. A no-op
    /// on the shipping path (and without the feature).
    fn wrapped(&self, dev: Arc<dyn PageIo>, _role: FileRole) -> Arc<dyn PageIo> {
        #[cfg(feature = "bench-instrument")]
        if let Some(w) = &self.wrap {
            return w(dev, _role.into());
        }
        dev
    }

    /// Whether this factory opens `role` with O_DIRECT.
    fn direct_for(&self, role: FileRole) -> bool {
        self.direct_io && role.allows_direct()
    }

    /// Opens `path` for I/O in `role`, **creating it if absent** (mirrors
    /// `SyscallIo::open` / `CompioIo::open`). The file's namespace prep — truncate,
    /// `create_new` — is the caller's explicit `OpenOptions` step before this,
    /// exactly as it was pre-M7; this only mints the byte device over an
    /// already-arranged path.
    pub(crate) fn open(&self, path: &Path, role: FileRole) -> io::Result<Arc<dyn PageIo>> {
        let direct = self.direct_for(role);
        let dev: Arc<dyn PageIo> = match self.backend {
            IoBackend::Syscall => Arc::new(SyscallIo::open_direct(path, direct)?),
            #[cfg(feature = "io-uring")]
            IoBackend::Uring => Arc::new(CompioIo::open_direct(path, direct)?),
        };
        Ok(self.wrapped(dev, role))
    }

    /// Opens an **existing** `path` for I/O in `role`, never creating it (mirrors
    /// `SyscallIo::open_existing`). Preserves the guarantee: a file deleted
    /// after its existence check reports `NotFound`, never a silent junk store.
    pub(crate) fn open_existing(&self, path: &Path, role: FileRole) -> io::Result<Arc<dyn PageIo>> {
        let direct = self.direct_for(role);
        let dev: Arc<dyn PageIo> = match self.backend {
            IoBackend::Syscall => Arc::new(SyscallIo::open_existing_direct(path, direct)?),
            #[cfg(feature = "io-uring")]
            IoBackend::Uring => Arc::new(CompioIo::open_existing_direct(path, direct)?),
        };
        Ok(self.wrapped(dev, role))
    }
}

/// A device-owned future. Boxed and `Send` so the boundary is object-safe.
pub type IoFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A write result that returns the owned buffer regardless of outcome.
pub struct BufResult {
    pub result: io::Result<()>,
    pub buffer: Box<[u8]>,
}

/// One durable byte device (one member of a [`FileGeneration`]).
///
/// All methods are positional; the engine never assumes `page_id * page_size`
/// — offsets come from the backend's `PageLocator`.
pub trait PageIo: Send + Sync + 'static {
    /// Reads exactly `len` bytes at `offset` into a fresh owned buffer. A short
    /// read before `len` is an error, not a partial success.
    fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>>;

    /// Writes the entire owned buffer at `offset`, returning it on completion.
    ///
    /// **Eager-submission contract (M7-B):** the write is enqueued to
    /// the device *before* the returned future is first polled — the buffer is
    /// moved onto the backend's submission channel (`Job`/`Req`) in this method's
    /// synchronous prologue, not on first poll. Two consequences callers rely on:
    /// (1) creating N write futures without awaiting them puts N ops in flight, so
    /// a bounded fan-out gets device concurrency from *construction*, not from
    /// concurrent polling; and (2) the returned future owns only a completion
    /// receiver — **dropping it never frees a buffer the device (kernel/worker)
    /// still owns**, since that buffer lives in the driver, which completes-or-
    /// leaks it per its own teardown discipline. A dropped-but-submitted write
    /// therefore still executes; its result is simply lost. Callers that must know
    /// a write succeeded (e.g. before a durability barrier) MUST await and observe
    /// every submitted write's `BufResult` — see `DirectCoordinator::flush_all`.
    fn write_all_at(&self, offset: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult>;

    /// `fdatasync`. Resolves only after the device reports the sync durable.
    fn datasync(&self) -> IoFuture<'_, io::Result<()>>;

    /// Current logical size in bytes.
    fn size(&self) -> IoFuture<'_, io::Result<u64>>;

    /// Deterministic teardown. After this resolves the device rejects further
    /// submissions. Idempotent.
    fn shutdown(&self) -> IoFuture<'_, io::Result<()>>;

    /// A coarse identifier of this device's concrete kind (M7-0). Defaults to
    /// [`IoKind::Other`]; the production backends and `FakeIo` override it so
    /// tests can assert what a path-backed store minted at each device site.
    fn kind(&self) -> IoKind {
        IoKind::Other
    }
}

/// A deterministic, in-memory [`PageIo`] for M0 and unit tests.
///
/// Backed by a `Vec<u8>` that grows to cover the highest written offset. Every
/// operation completes inline (no scheduler), which is exactly what M0 needs to
/// exercise the actor/permit/poison state machine without real I/O. Fault
/// injection ([`Faults`]) lets later tests drive short reads/writes, EINTR-style
/// retries, and sync failures deterministically; M0 defaults to no faults.
pub struct FakeIo {
    inner: parking_lot::Mutex<FakeInner>,
    /// Test-only sync barrier: lets a test hold an in-flight `datasync` parked
    /// and release it deterministically, so the M6 mid-sync abandonment /
    /// completion-frontier tests can pin the exact moment a durability barrier is
    /// between "writes landed" and "sync durable" (M5 test debt).
    /// Inert (zero overhead, resolves inline) until [`FakeIo::arm_sync_gate`].
    gate: SyncGate,
    /// Test-only write barrier (M7-B): holds in-flight `write_all_at` completions
    /// parked so a test can observe the caller's fan-out depth mid-flush. Inert
    /// until [`FakeIo::arm_write_gate`]. Shares [`SyncGate`]'s claim/park/release
    /// mechanism; distinct instance so writes gate independently of syncs.
    write_gate: SyncGate,
    /// Count of `write_all_at` *submissions* (M7-B): incremented in the sync
    /// prologue, before the returned future is polled — models eager submission,
    /// and lets a test assert exactly how many writes a flush issued (e.g. one
    /// bounded chunk's worth) before any completes.
    writes_submitted: AtomicUsize,
    /// Writes currently submitted-but-not-yet-completed (M7-B). Incremented at
    /// submission, decremented at terminal completion.
    writes_in_flight: AtomicUsize,
    /// High-water mark of [`writes_in_flight`](Self::writes_in_flight) — the peak
    /// caller-visible write concurrency (M7-B fan-out observability; feeds the
    /// M7-E device-overlap bench).
    writes_peak: AtomicUsize,
    /// Test-only read barrier (T2.1): holds in-flight `read_exact_at`
    /// completions parked so a test can observe a reader's fan-out depth — the
    /// concurrent-`get_many` acceptance oracle ("wait until the expected reads
    /// are *parked*, then release"). Inert until [`FakeIo::arm_read_gate`].
    read_gate: SyncGate,
    /// Count of `read_exact_at` *submissions* (sync prologue, before the future
    /// is polled), mirroring [`writes_submitted`](Self::writes_submitted).
    reads_submitted: AtomicUsize,
    /// Reads currently submitted-but-not-yet-completed.
    reads_in_flight: AtomicUsize,
    /// High-water mark of [`reads_in_flight`](Self::reads_in_flight) — the peak
    /// caller-visible read concurrency (T2.1's "expected peak = distinct child
    /// pages at that level" oracle).
    reads_peak: AtomicUsize,
}

/// A deterministic hold-and-release barrier over `datasync` completions.
///
/// When armed to hold `n` syncs, the next `n` `datasync` calls each claim a slot
/// (in call order) and park before folding the durable image, until an equal
/// number of [`SyncGate::release`]s are granted. Unarmed, `claim` returns `None`
/// and `datasync` behaves exactly as before (inline). All test-only.
struct SyncGate {
    /// Woken on every release so parked syncs re-check their slot.
    event: event_listener::Event,
    /// Total slots to trap (each armed sync consumes one, in claim order).
    arm: AtomicUsize,
    /// Slots claimed so far — a sync parks on slot `i` iff `i < arm`.
    claimed: AtomicUsize,
    /// Slots released so far — the sync on slot `i` proceeds once `released > i`.
    released: AtomicUsize,
    /// Syncs currently parked at the barrier (test observation).
    parked: AtomicUsize,
}

impl SyncGate {
    fn new() -> Self {
        SyncGate {
            event: event_listener::Event::new(),
            arm: AtomicUsize::new(0),
            claimed: AtomicUsize::new(0),
            released: AtomicUsize::new(0),
            parked: AtomicUsize::new(0),
        }
    }

    /// Claims the next gate slot if any hold remains, returning its index. A
    /// claimed slot must [`wait`](Self::wait) before its sync proceeds.
    fn claim(&self) -> Option<usize> {
        loop {
            let c = self.claimed.load(Ordering::Acquire);
            if c >= self.arm.load(Ordering::Acquire) {
                return None;
            }
            if self
                .claimed
                .compare_exchange(c, c + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(c);
            }
        }
    }

    /// Parks until slot `i` is released. Increments the parked count while
    /// waiting so a test can observe the barrier is occupied. The decrement
    /// rides a drop guard, so a **cancelled** parked op (its future dropped
    /// mid-wait) still leaves `parked` accurate — only the op's own
    /// in-flight counter strands, which is the documented divergence the
    /// drain-evidence tests rely on (T2.1).
    async fn wait(&self, i: usize) {
        struct ParkGuard<'a>(&'a AtomicUsize);
        impl Drop for ParkGuard<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        self.parked.fetch_add(1, Ordering::AcqRel);
        let _guard = ParkGuard(&self.parked);
        loop {
            if self.released.load(Ordering::Acquire) > i {
                break;
            }
            let listener = self.event.listen();
            if self.released.load(Ordering::Acquire) > i {
                break;
            }
            listener.await;
        }
    }
}

struct FakeInner {
    data: Vec<u8>,
    faults: Faults,
    shut: bool,
    /// Set once a datasync has failed, to model a poisoned device.
    sync_failed: bool,
    /// Count of successful `datasync`s (test observation of group commit).
    syncs: u64,
    /// The durable image as of the last successful `datasync` — what a crash
    /// would preserve verbatim outside any in-flight (unsynced) write range
    /// (powersafe overwrite). `data` is the live view a running
    /// store reads; `synced` is what survives a crash before the next sync.
    synced: Vec<u8>,
    /// Byte ranges written since the last successful `datasync`, in write
    /// *completion* (await) order — the "non-durable written ranges" a crash may
    /// persist as arbitrary subsets/torn/garbage. Since M7-B the apply happens when
    /// a write future is awaited, so a fanned-out flush records ranges in the order
    /// its awaits resolve, not submission order (irrelevant to the crash model,
    /// which treats the whole set as an arbitrary-subset hazard). Cleared on `datasync`.
    unsynced: Vec<(u64, usize)>,
}

/// Deterministic fault knobs. All default to "no fault".
#[derive(Clone, Default)]
pub struct Faults {
    /// If set, the next `datasync` fails with this errno, then clears.
    pub fail_next_sync: Option<i32>,
    /// If set, the next `size` probe fails with this errno, then clears
    /// (Phase-H debt: the compaction pre-temp size probe's abort arm).
    pub fail_next_size: Option<i32>,
    /// If set, reads/writes to offsets `>= this` fail with EIO.
    pub fail_beyond_offset: Option<u64>,
    /// If set, every `write_all_at` whose **submission index** (0-based, since the
    /// last [`FakeIo::reset_write_stats`]) is `>= this` fails with EIO at
    /// completion (M7-B). Order-based rather than offset-based, so a test can land
    /// exactly the first *n* writes of a fan-out and fail the rest regardless of
    /// which offsets the CoW builder assigned them.
    pub fail_writes_from: Option<usize>,
    /// If set, every `read_exact_at` whose **submission index** (0-based, since
    /// the last [`FakeIo::reset_read_stats`]) is `>= this` fails with EIO at
    /// completion (T2.1) — the read mirror of `fail_writes_from`.
    pub fail_reads_from: Option<usize>,
    /// Reads whose exact `offset` is listed here fail with EIO at completion
    /// (T2.1). Targeted, unlike the range knobs: lets a test EIO one specific
    /// child page of a fan-out while its siblings (at offsets on either side)
    /// succeed or fail differently.
    pub fail_read_offsets: Vec<u64>,
    /// Reads whose exact **submission index** (0-based, since the last
    /// [`FakeIo::reset_read_stats`]) is listed here fail with EIO at completion
    /// (T2.2). Single-shot per listed index, unlike the `from` range knob: a
    /// later read of the *same offset* succeeds — which is what lets a test fail
    /// one speculative chain read and prove the demand fallback re-reads the
    /// same bytes successfully.
    pub fail_read_orders: Vec<usize>,
}

impl FakeIo {
    /// A fresh device with `len` zero bytes and no faults.
    pub fn new(len: usize) -> Self {
        FakeIo::from_vec(vec![0u8; len])
    }

    /// A device pre-loaded with `bytes` (durable). Used to reopen a store over a
    /// crash image built by [`FakeIo::crash_image`].
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        FakeIo {
            inner: parking_lot::Mutex::new(FakeInner {
                synced: bytes.clone(),
                data: bytes,
                faults: Faults::default(),
                shut: false,
                sync_failed: false,
                syncs: 0,
                unsynced: Vec::new(),
            }),
            gate: SyncGate::new(),
            write_gate: SyncGate::new(),
            writes_submitted: AtomicUsize::new(0),
            writes_in_flight: AtomicUsize::new(0),
            writes_peak: AtomicUsize::new(0),
            read_gate: SyncGate::new(),
            reads_submitted: AtomicUsize::new(0),
            reads_in_flight: AtomicUsize::new(0),
            reads_peak: AtomicUsize::new(0),
        }
    }

    /// Number of successful `datasync`s so far (for group-commit observation).
    pub fn sync_count(&self) -> u64 {
        self.inner.lock().syncs
    }

    /// Installs fault knobs. Takes effect on subsequent operations.
    pub fn set_faults(&self, faults: Faults) {
        self.inner.lock().faults = faults;
    }

    /// Arms the sync barrier to hold the next `n` `datasync` calls: each parks
    /// (its future stays `Pending`) after claiming a slot but *before* making the
    /// write durable, until an equal number of [`release_syncs`](Self::release_syncs)
    /// is granted. Additive across calls. Test-only.
    pub fn arm_sync_gate(&self, n: usize) {
        self.gate.arm.fetch_add(n, Ordering::AcqRel);
    }

    /// Releases up to `n` parked (or not-yet-arrived) gated syncs, in slot order,
    /// and wakes any waiters. Test-only.
    pub fn release_syncs(&self, n: usize) {
        self.gate.released.fetch_add(n, Ordering::AcqRel);
        self.gate.event.notify(usize::MAX);
    }

    /// Number of `datasync` calls currently parked at the barrier (test
    /// observation of a mid-sync durability barrier).
    pub fn parked_sync_count(&self) -> usize {
        self.gate.parked.load(Ordering::Acquire)
    }

    /// Arms the write barrier to hold the next `n` `write_all_at` completions:
    /// each claims a slot (in submission order) after the op is counted in-flight
    /// but *before* it applies its bytes or resolves, until an equal number of
    /// [`release_writes`](Self::release_writes) is granted. Additive. Test-only
    /// (M7-B): lets a test freeze a bounded fan-out mid-flight and read its depth.
    ///
    /// Footgun: an armed-but-unreleased slot parks *every* later write, including
    /// the header write and any flush during `close()`. A test must
    /// [`release_writes`](Self::release_writes) at least as many as it will ever
    /// submit (release generously) before closing, or the store wedges.
    pub fn arm_write_gate(&self, n: usize) {
        self.write_gate.arm.fetch_add(n, Ordering::AcqRel);
    }

    /// Releases up to `n` parked (or not-yet-arrived) gated write completions, in
    /// slot order, and wakes any waiters. Test-only (M7-B).
    pub fn release_writes(&self, n: usize) {
        self.write_gate.released.fetch_add(n, Ordering::AcqRel);
        self.write_gate.event.notify(usize::MAX);
    }

    /// Number of write completions currently parked at the write barrier (M7-B).
    pub fn parked_write_count(&self) -> usize {
        self.write_gate.parked.load(Ordering::Acquire)
    }

    /// Total `write_all_at` submissions so far (M7-B). Counts a write the instant
    /// its op is enqueued (sync prologue), before its future is polled — so a test
    /// can assert a flush issued exactly one bounded chunk before any completed.
    pub fn writes_submitted(&self) -> usize {
        self.writes_submitted.load(Ordering::Acquire)
    }

    /// Writes currently submitted-but-not-completed (M7-B).
    pub fn writes_in_flight(&self) -> usize {
        self.writes_in_flight.load(Ordering::Acquire)
    }

    /// Peak concurrent in-flight writes observed (M7-B fan-out depth; feeds the
    /// M7-E device-overlap bench). Reset with
    /// [`reset_write_stats`](Self::reset_write_stats).
    pub fn writes_peak(&self) -> usize {
        self.writes_peak.load(Ordering::Acquire)
    }

    /// Resets the submitted/peak write counters (M7-B). In-flight is left alone
    /// (it reflects live ops). Test-only, e.g. to isolate one flush's fan-out.
    pub fn reset_write_stats(&self) {
        self.writes_submitted.store(0, Ordering::Release);
        self.writes_peak.store(0, Ordering::Release);
    }

    /// Arms the read barrier to hold the next `n` `read_exact_at` completions:
    /// each claims a slot (in submission order) after the op is counted in-flight
    /// but *before* it evaluates faults or copies bytes, until an equal number of
    /// [`release_reads`](Self::release_reads) is granted. Additive. Test-only
    /// (T2.1): lets a test freeze a reader's fan-out mid-flight, observe its
    /// depth, and install completion-time faults while reads are parked.
    ///
    /// Same footgun as the write gate: an armed-but-unreleased slot parks *every*
    /// later read. Release at least as many as will ever be submitted.
    pub fn arm_read_gate(&self, n: usize) {
        self.read_gate.arm.fetch_add(n, Ordering::AcqRel);
    }

    /// Releases up to `n` parked (or not-yet-arrived) gated read completions, in
    /// slot order, and wakes any waiters. Test-only (T2.1).
    pub fn release_reads(&self, n: usize) {
        self.read_gate.released.fetch_add(n, Ordering::AcqRel);
        self.read_gate.event.notify(usize::MAX);
    }

    /// Number of read completions currently parked at the read barrier (T2.1).
    pub fn parked_read_count(&self) -> usize {
        self.read_gate.parked.load(Ordering::Acquire)
    }

    /// Total `read_exact_at` submissions so far (T2.1). Counts a read the instant
    /// its op is counted (sync prologue), before its future is polled.
    pub fn reads_submitted(&self) -> usize {
        self.reads_submitted.load(Ordering::Acquire)
    }

    /// Reads currently submitted-but-not-completed (T2.1). With no writer thread
    /// to backstop orphaned cache loads (e.g. under `TreeReader`), a zero here
    /// after an errored multi-get is evidence the reader *drained* every started
    /// child rather than dropping one mid-flight.
    pub fn reads_in_flight(&self) -> usize {
        self.reads_in_flight.load(Ordering::Acquire)
    }

    /// Peak concurrent in-flight reads observed (T2.1's fan-out oracle). Reset
    /// with [`reset_read_stats`](Self::reset_read_stats).
    pub fn reads_peak(&self) -> usize {
        self.reads_peak.load(Ordering::Acquire)
    }

    /// Resets the submitted/peak read counters (T2.1). Only meaningful — and
    /// only permitted — while no read is in flight: `fail_reads_from` indices
    /// are relative to this reset, and re-basing them under a live read would
    /// make the fault's target ambiguous.
    pub fn reset_read_stats(&self) {
        assert_eq!(
            self.reads_in_flight.load(Ordering::Acquire),
            0,
            "reset_read_stats requires no in-flight reads"
        );
        self.reads_submitted.store(0, Ordering::Release);
        self.reads_peak.store(0, Ordering::Release);
    }

    /// Applies a successful write into the live image and records its range as
    /// non-durable until the next `datasync`. Shared by the gated write path.
    fn apply_write(inner: &mut FakeInner, offset: u64, buffer: &[u8]) {
        let start = offset as usize;
        let end = start.saturating_add(buffer.len());
        if end > inner.data.len() {
            inner.data.resize(end, 0);
        }
        inner.data[start..end].copy_from_slice(buffer);
        inner.unsynced.push((offset, buffer.len()));
    }

    /// The synchronous `datasync` body (device checks + durable-image fold),
    /// shared by the gated and ungated paths. Returns once the write is durable.
    fn sync_now(&self) -> io::Result<()> {
        let mut inner = self.inner.lock();
        if inner.shut {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "device shut"));
        }
        if inner.sync_failed {
            // A failed sync poisons the device permanently (never retried into
            // success): mirror invariant 17 at the device level.
            return Err(io::Error::from_raw_os_error(libc_eio()));
        }
        if let Some(errno) = inner.faults.fail_next_sync.take() {
            inner.sync_failed = true;
            return Err(io::Error::from_raw_os_error(errno));
        }
        // Durable: fold the writes since the last sync into the durable image.
        // Equivalent to `synced = data.clone()` (after the previous sync
        // `synced == data`, and `unsynced` records exactly the ranges that have
        // diverged since), but O(bytes written since last sync) rather than
        // O(file). A full clone here makes any sync-per-op workload on an
        // append-only growing file quadratic in wall-clock, which silently
        // contaminated the M5 write benchmark's timing.
        let g = &mut *inner;
        if g.synced.len() < g.data.len() {
            g.synced.resize(g.data.len(), 0);
        }
        let ranges = std::mem::take(&mut g.unsynced);
        for (off, len) in ranges {
            let (s, e) = (off as usize, off as usize + len);
            g.synced[s..e].copy_from_slice(&g.data[s..e]);
        }
        g.syncs += 1;
        Ok(())
    }

    /// A copy of the current (live) backing bytes (test inspection).
    pub fn snapshot_bytes(&self) -> Vec<u8> {
        self.inner.lock().data.clone()
    }

    /// The durable image as of the last successful `datasync`. Recovery over this
    /// must recover every acknowledged commit (each is synced before its ack).
    pub fn synced_bytes(&self) -> Vec<u8> {
        self.inner.lock().synced.clone()
    }

    /// Byte ranges written since the last successful `datasync` (submission
    /// order) — the non-durable region a crash may persist arbitrarily.
    pub fn unsynced_ranges(&self) -> Vec<(u64, usize)> {
        self.inner.lock().unsynced.clone()
    }

    /// Builds a **crash image** in application write-range terms: start from
    /// the durable (`synced`) bytes, then for each
    /// `blast_radius`-byte granule that lies inside an unsynced write range, apply
    /// `decide(granule_index)` — [`Persist::Old`] leaves the durable byte,
    /// [`Persist::New`] writes the live (just-written) byte, [`Persist::Garble`]
    /// fills with garbage. Every byte **outside** the union of unsynced write
    /// ranges is preserved verbatim (powersafe overwrite). The image is truncated
    /// to the highest actually-persisted granule end (a crash that persisted no
    /// trailing bytes yields a shorter file, which recovery reads by size). The
    /// caller's `decide` closure is the recorded, replayable seed.
    ///
    /// Model limitation: `New` reflects the *final* live bytes, so two unsynced
    /// writes that overlap the same granule cannot be decided independently. The
    /// WAL is append-only (each record occupies a distinct offset range), so this
    /// never arises for the recovery scenarios under test.
    pub fn crash_image(&self, blast_radius: usize, decide: impl Fn(usize) -> Persist) -> Vec<u8> {
        let inner = self.inner.lock();
        let br = blast_radius.max(1);
        let mut img = inner.synced.clone();
        // Highest byte the durable image already covers; anything beyond exists
        // only if a New/Garble granule persists it.
        let mut persisted_end = img.len();
        for &(off, len) in &inner.unsynced {
            let start = off as usize;
            let end = start + len;
            let mut g = start / br;
            loop {
                let gstart = (g * br).max(start);
                let gend = ((g + 1) * br).min(end);
                if gstart >= end {
                    break;
                }
                match decide(g) {
                    Persist::Old => {}
                    Persist::New => {
                        if img.len() < gend {
                            img.resize(gend, 0);
                        }
                        img[gstart..gend].copy_from_slice(&inner.data[gstart..gend]);
                        persisted_end = persisted_end.max(gend);
                    }
                    Persist::Garble => {
                        if img.len() < gend {
                            img.resize(gend, 0);
                        }
                        for b in &mut img[gstart..gend] {
                            *b = 0xA5;
                        }
                        persisted_end = persisted_end.max(gend);
                    }
                }
                g += 1;
            }
        }
        img.truncate(persisted_end.max(inner.synced.len()));
        img
    }
}

/// How one blast-radius granule of an unsynced write range survives a crash
/// ([`FakeIo::crash_image`]). Deterministic — the caller chooses per granule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Persist {
    /// The durable (pre-write) byte survives — the write did not reach stable
    /// storage for this granule.
    Old,
    /// The just-written (live) byte survives — the write reached stable storage.
    New,
    /// Torn/garbage bytes (neither old nor new).
    Garble,
}

fn ready<T: Send + 'static>(v: T) -> IoFuture<'static, T> {
    Box::pin(std::future::ready(v))
}

impl PageIo for FakeIo {
    // T2.1: mirrors `write_all_at`'s shape — shut is terminal-at-submit (never
    // counted in flight); the submission is counted and a read-gate slot claimed
    // in the sync prologue; faults are evaluated and bytes copied at COMPLETION,
    // after the gate releases. Completion-time faults are load-bearing for the
    // acceptance tests: a test can park a fan-out at the gate, *then* install
    // `fail_beyond_offset`/`fail_reads_from`, and choose per-child outcomes.
    // The same caller-side divergence note as `write_all_at` applies: a dropped
    // read future never completes and permanently strands `reads_in_flight` —
    // which is exactly what makes `reads_in_flight()==0` after an errored
    // multi-get evidence of drain-all (no writer backstop exists under
    // `TreeReader` to finish an abandoned cache load). `parked_read_count` does
    // NOT share that divergence: a cancelled parked read un-parks via
    // `SyncGate::wait`'s drop guard (its claimed slot still absorbs one later
    // release, as any slot does). Regression: `dropped_parked_read_unparks_...`.
    fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
        if self.inner.lock().shut {
            return ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "device shut",
            )));
        }
        let submit_idx = self.reads_submitted.fetch_add(1, Ordering::AcqRel);
        let depth = self.reads_in_flight.fetch_add(1, Ordering::AcqRel) + 1;
        self.reads_peak.fetch_max(depth, Ordering::AcqRel);
        let slot = self.read_gate.claim();
        Box::pin(async move {
            if let Some(i) = slot {
                self.read_gate.wait(i).await;
            }
            let result = {
                let inner = self.inner.lock();
                let fail_by_offset = inner.faults.fail_beyond_offset.is_some_and(|b| offset >= b);
                let fail_by_order = inner
                    .faults
                    .fail_reads_from
                    .is_some_and(|n| submit_idx >= n);
                let fail_by_target = inner.faults.fail_read_offsets.contains(&offset);
                let fail_by_exact_order = inner.faults.fail_read_orders.contains(&submit_idx);
                let start = offset as usize;
                let end = start.saturating_add(len);
                if fail_by_offset || fail_by_order || fail_by_target || fail_by_exact_order {
                    Err(io::Error::from_raw_os_error(libc_eio()))
                } else if end > inner.data.len() {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "read past end of fake device",
                    ))
                } else {
                    Ok(inner.data[start..end].to_vec().into_boxed_slice())
                }
            };
            self.reads_in_flight.fetch_sub(1, Ordering::AcqRel);
            result
        })
    }

    // NOTE (M7-B): this fake **diverges from the `PageIo::write_all_at`
    // eager-submission contract**. The real backends move the buffer onto a
    // channel in the sync prologue, so a submitted write completes even if its
    // caller future is dropped; here the apply/fault happens *inside* the returned
    // future, so a dropped-or-never-polled write never executes — it never touches
    // `data`/`unsynced` and permanently strands `writes_in_flight`. Two
    // consequences: (1) `writes_submitted`/`writes_in_flight`/`writes_peak` measure
    // caller-side submission and *consumption* of the future, not device-visible
    // overlap — the peak equals the fan-out depth only because `flush_all` builds a
    // whole chunk before awaiting, and the drain-all test relies precisely on this
    // divergence (a dropped future leaves in-flight > 0); (2) `crash_image` /
    // `synced_bytes` only reflect writes that were actually awaited. Do not model a
    // crash-consistency test on FakeIo as if an un-awaited write could have reached
    // media — on a real device it can.
    fn write_all_at(&self, offset: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult> {
        // Terminal-at-submit: a shut device rejects before the op is ever counted
        // in flight, mirroring the real backends' `terminal()`/closed-channel
        // check (no queued op, buffer handed straight back).
        if self.inner.lock().shut {
            return ready(BufResult {
                result: Err(io::Error::new(io::ErrorKind::BrokenPipe, "device shut")),
                buffer,
            });
        }
        // Count the submission (sync prologue, before the future is polled) so a
        // test sees the caller's fan-out depth even while every completion is
        // parked. Claim a write-gate slot here too, so slot order == submission
        // order regardless of the order completions are later awaited (M7-B).
        let submit_idx = self.writes_submitted.fetch_add(1, Ordering::AcqRel);
        let depth = self.writes_in_flight.fetch_add(1, Ordering::AcqRel) + 1;
        self.writes_peak.fetch_max(depth, Ordering::AcqRel);
        let slot = self.write_gate.claim();
        Box::pin(async move {
            if let Some(i) = slot {
                self.write_gate.wait(i).await;
            }
            let result = {
                let mut inner = self.inner.lock();
                // A device-side I/O error surfaces at completion, not submit — by
                // offset (`fail_beyond_offset`) or by submission order
                // (`fail_writes_from`). Either failing knob leaves `data`/`unsynced`
                // untouched for this write.
                let fail_by_offset = inner.faults.fail_beyond_offset.is_some_and(|b| offset >= b);
                let fail_by_order = inner
                    .faults
                    .fail_writes_from
                    .is_some_and(|n| submit_idx >= n);
                if fail_by_offset || fail_by_order {
                    Err(io::Error::from_raw_os_error(libc_eio()))
                } else {
                    Self::apply_write(&mut inner, offset, &buffer);
                    Ok(())
                }
            };
            self.writes_in_flight.fetch_sub(1, Ordering::AcqRel);
            BufResult { result, buffer }
        })
    }

    fn datasync(&self) -> IoFuture<'_, io::Result<()>> {
        // Claim a gate slot (inline, no-op unless the barrier is armed) so call
        // order fixes slot order, then park on it before folding. An unarmed gate
        // resolves on first poll, exactly like the previous inline body.
        let slot = self.gate.claim();
        Box::pin(async move {
            if let Some(i) = slot {
                self.gate.wait(i).await;
            }
            self.sync_now()
        })
    }

    fn size(&self) -> IoFuture<'_, io::Result<u64>> {
        let mut inner = self.inner.lock();
        if let Some(errno) = inner.faults.fail_next_size.take() {
            return ready(Err(io::Error::from_raw_os_error(errno)));
        }
        ready(Ok(inner.data.len() as u64))
    }

    fn shutdown(&self) -> IoFuture<'_, io::Result<()>> {
        let mut inner = self.inner.lock();
        inner.shut = true;
        ready(Ok(()))
    }

    fn kind(&self) -> IoKind {
        IoKind::Fake
    }
}

/// EIO errno without pulling in the `libc` crate for one constant.
fn libc_eio() -> i32 {
    5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_roundtrips() {
        futures_lite::future::block_on(async {
            let dev = FakeIo::new(0);
            let w = dev
                .write_all_at(4096, vec![7u8; 100].into_boxed_slice())
                .await;
            w.result.unwrap();
            let r = dev.read_exact_at(4096, 100).await.unwrap();
            assert!(r.iter().all(|b| *b == 7));
        });
    }

    #[test]
    fn factory_mints_selected_backend_and_open_existing_refuses_missing() {
        // M7-0: the DeviceFactory mints the selected backend (asserted via kind())
        // and open_existing preserves the guarantee (a missing file is NotFound,
        // never a silently-created junk device).
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("present.data");
        let missing = dir.path().join("absent.data");

        let f = DeviceFactory::new(IoBackend::Syscall, false);
        // open() creates if absent, and it is a Syscall device.
        let dev = f.open(&existing, FileRole::Data).unwrap();
        assert_eq!(dev.kind(), IoKind::Syscall);
        futures_lite::future::block_on(dev.shutdown()).unwrap();
        // open_existing() on a present file succeeds; on a missing one → NotFound.
        assert_eq!(
            f.open_existing(&existing, FileRole::Data).unwrap().kind(),
            IoKind::Syscall
        );
        match f.open_existing(&missing, FileRole::Data) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::NotFound),
            Ok(_) => panic!("open_existing on a missing file must fail"),
        }
        assert!(!missing.exists(), "open_existing must not create the file");
    }

    #[test]
    fn direct_factory_honors_role_and_wal_stays_buffered() {
        // M7-A: a direct_io factory opens Data + CompactTemp O_DIRECT but never a
        // WalSegment (byte-granular records cannot be O_DIRECT). We can't observe the
        // O_DIRECT flag through the trait, but the role gate is a pure predicate.
        let f = DeviceFactory::new(IoBackend::Syscall, true);
        assert!(f.direct_for(FileRole::Data));
        assert!(f.direct_for(FileRole::CompactTemp));
        assert!(!f.direct_for(FileRole::WalSegment));
        // A non-direct factory never opens anything direct.
        let g = DeviceFactory::new(IoBackend::Syscall, false);
        assert!(!g.direct_for(FileRole::Data));
    }

    #[cfg(feature = "io-uring")]
    #[test]
    fn factory_mints_uring_backend() {
        // M7-0: under the io-uring feature the factory mints CompioIo, and its
        // open_existing keeps the missing-file guarantee too (CompioIo::open_existing).
        if !super::io_uring_available() {
            eprintln!("skip: io_uring unavailable on this host");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let existing = dir.path().join("present.data");
        let missing = dir.path().join("absent.data");
        let f = DeviceFactory::new(IoBackend::Uring, false);
        let dev = f.open(&existing, FileRole::Data).unwrap();
        assert_eq!(dev.kind(), IoKind::Uring);
        futures_lite::future::block_on(dev.shutdown()).unwrap();
        assert_eq!(
            f.open_existing(&existing, FileRole::Data).unwrap().kind(),
            IoKind::Uring
        );
        match f.open_existing(&missing, FileRole::Data) {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::NotFound),
            Ok(_) => panic!("CompioIo::open_existing on a missing file must fail"),
        }
        assert!(
            !missing.exists(),
            "CompioIo::open_existing must not create the file"
        );
        // M7-A commit 2: Uring + direct_io opens O_DIRECT (where the fs honors it).
        // The temp dir may be tmpfs (no O_DIRECT), so only assert success when it opens.
        let d = DeviceFactory::new(IoBackend::Uring, true);
        let direct = dir.path().join("direct.data");
        if let Ok(dev) = d.open(&direct, FileRole::Data) {
            assert_eq!(dev.kind(), IoKind::Uring);
            futures_lite::future::block_on(dev.shutdown()).unwrap();
        }
        // A WalSegment never gets O_DIRECT even under a direct factory, so it opens
        // on any filesystem (buffered).
        let seg = dir.path().join("seg.wal");
        assert_eq!(
            d.open(&seg, FileRole::WalSegment).unwrap().kind(),
            IoKind::Uring
        );
    }

    #[test]
    fn sync_gate_holds_then_releases() {
        // An armed gate parks a datasync in-flight (write landed, not yet
        // durable) until released; an unarmed gate resolves inline.
        let dev = std::sync::Arc::new(FakeIo::new(0));
        futures_lite::future::block_on(async {
            dev.write_all_at(0, vec![9u8; 16].into_boxed_slice())
                .await
                .result
                .unwrap();
        });
        dev.arm_sync_gate(1);

        // Drive the gated sync on its own thread; it must not complete while held.
        let d = std::sync::Arc::clone(&dev);
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d2 = std::sync::Arc::clone(&done);
        let h = std::thread::spawn(move || {
            futures_lite::future::block_on(d.datasync()).unwrap();
            d2.store(true, std::sync::atomic::Ordering::Release);
        });

        // Spin until it parks; it stays parked (durable image not yet folded).
        while dev.parked_sync_count() == 0 {
            std::thread::yield_now();
        }
        assert!(!done.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(dev.sync_count(), 0, "gated sync must not have folded yet");

        // Release: it completes and the write is now durable.
        dev.release_syncs(1);
        h.join().unwrap();
        assert!(done.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(dev.sync_count(), 1);
        assert_eq!(dev.synced_bytes()[..16], [9u8; 16]);

        // Gate is spent: a further sync is not held.
        futures_lite::future::block_on(dev.datasync()).unwrap();
        assert_eq!(dev.sync_count(), 2);
    }

    #[test]
    fn dropped_parked_read_unparks_but_stays_in_flight() {
        // T2.1: cancelling a gated read must repair
        // `parked_read_count` (drop guard) while `reads_in_flight` deliberately
        // strands — the documented caller-side divergence the drain-evidence
        // tests rely on.
        let dev = FakeIo::new(4096);
        dev.arm_read_gate(1);
        // Poll once (parks at the gate), then drop the future.
        assert!(
            futures_lite::future::block_on(futures_lite::future::poll_once(
                dev.read_exact_at(0, 16)
            ))
            .is_none(),
            "armed gate must hold the read pending"
        );
        assert_eq!(dev.parked_read_count(), 0, "drop guard un-parks");
        assert_eq!(dev.reads_in_flight(), 1, "cancelled op strands in-flight");
        // The dead claimed slot absorbs one release; a later read is unaffected.
        dev.release_reads(1);
        futures_lite::future::block_on(dev.read_exact_at(0, 16)).unwrap();
        assert_eq!(dev.reads_in_flight(), 1, "only the cancelled op remains");
    }

    #[test]
    fn read_gate_holds_counts_and_faults_at_completion() {
        // T2.1: an armed read gate parks reads in submission order while the
        // in-flight/peak counters expose the fan-out depth; faults installed
        // WHILE PARKED take effect at completion; release order frees slots in
        // submission order.
        let dev = std::sync::Arc::new(FakeIo::new(0));
        futures_lite::future::block_on(async {
            dev.write_all_at(0, vec![1u8; 4096].into_boxed_slice())
                .await
                .result
                .unwrap();
            dev.write_all_at(4096, vec![2u8; 4096].into_boxed_slice())
                .await
                .result
                .unwrap();
        });
        dev.reset_read_stats();
        dev.arm_read_gate(2);

        // Two concurrent reads on scratch threads; both park.
        let mut handles = Vec::new();
        for off in [0u64, 4096] {
            let d = std::sync::Arc::clone(&dev);
            handles.push(std::thread::spawn(move || {
                futures_lite::future::block_on(d.read_exact_at(off, 4096))
            }));
        }
        // Bounded join helper: joins only a finished thread; a thread still
        // wedged after the deadline is leaked so the failure is a panic, never
        // a suite hang.
        let join_bounded = |h: std::thread::JoinHandle<io::Result<Box<[u8]>>>, what: &str| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !h.is_finished() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "scratch read thread wedged: {what}"
                );
                std::thread::yield_now();
            }
            h.join().unwrap()
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while dev.parked_read_count() < 2 {
            if std::time::Instant::now() >= deadline {
                // Free the scratch threads before failing (bounded cleanup).
                dev.release_reads(usize::MAX / 2);
                for h in handles {
                    let _ = join_bounded(h, "cleanup after park timeout");
                }
                panic!("reads never parked");
            }
            std::thread::yield_now();
        }
        assert_eq!(dev.reads_submitted(), 2);
        assert_eq!(dev.reads_in_flight(), 2);
        assert_eq!(dev.reads_peak(), 2);

        // Install a completion-time fault while both are parked: the read at
        // offset 4096 fails, the one at 0 succeeds.
        dev.set_faults(Faults {
            fail_beyond_offset: Some(4096),
            ..Default::default()
        });
        dev.release_reads(2);
        let results: Vec<_> = handles
            .into_iter()
            .map(|h| join_bounded(h, "released reads"))
            .collect();
        let ok: Vec<bool> = results.iter().map(|r| r.is_ok()).collect();
        assert_eq!(ok.iter().filter(|b| **b).count(), 1, "exactly one succeeds");
        assert_eq!(dev.reads_in_flight(), 0, "all reads completed");
        assert!(
            results
                .iter()
                .flatten()
                .all(|b| b.iter().all(|x| *x == 1u8)),
            "the surviving read is the offset-0 image"
        );

        // Gate spent: further reads resolve without parking; order-based knob.
        dev.reset_read_stats();
        dev.set_faults(Faults {
            fail_reads_from: Some(1),
            ..Default::default()
        });
        futures_lite::future::block_on(async {
            assert!(dev.read_exact_at(0, 16).await.is_ok());
            assert!(dev.read_exact_at(0, 16).await.is_err());
        });
    }

    #[test]
    fn sync_fault_poisons_device() {
        futures_lite::future::block_on(async {
            let dev = FakeIo::new(0);
            dev.set_faults(Faults {
                fail_next_sync: Some(libc_eio()),
                ..Default::default()
            });
            assert!(dev.datasync().await.is_err());
            // Never retried into success.
            assert!(dev.datasync().await.is_err());
        });
    }
}
