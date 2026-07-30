//! M7-E — **"the decision"**: re-run the G1 io_uring-default A/B on the
//! **production** `create_path`, and adopt/reject each performance knob (io_uring
//! default, `direct_io`, flush fan-out, SQPOLL) **with data**, on the real path
//! that mints its own device — not the caller-supplied-device proxy G1 used, which
//! has no checkpoint.
//!
//! ## Why this bench is a rewrite, not an edit of `io_default.rs`
//!
//! The G1 bench (`benches/io_default.rs`) carries several Phase-G harness defects,
//! each load-bearing here:
//!
//!   * **clock-before-spawn** (`io_default.rs:288`) — the same G3 defect that
//!     put a ±66% floor into `clean_tier` before its start barrier. Fixed here with
//!     a park-until-released [`Barrier`] and a window sized so the *fastest* arm of
//!     every cell runs ≥ ~1 s.
//!   * **the 4-RMW `CountIo`** — four shared read-modify-writes per read (incl. a
//!     `fetch_max` watermark) that bounce a cache line on the very fast path being
//!     measured. The watermark is *struck entirely* (eager submission makes an
//!     in-flight watermark blind to filesystem serialization); the
//!     surviving counters are **per-thread striped** (a relaxed add per metric —
//!     op count + byte count — on the
//!     thread's own cache line) and **role-tagged** (data / WAL-segment /
//!     compaction-temp), and a wrapper-on/off control bounds their residual cost.
//!   * **the 2% unequal-I/O tolerance** (`io_default.rs:431`) — replaced by an
//!     **exact** per-role equal-demand guard, made legitimate by a **compulsory-miss
//!     disjoint-partition** trace: every reader replays its own key partition, no
//!     key touched twice by anyone, and keys are spaced past the leaf fan-out so
//!     each touched leaf is read exactly once — so the device read count is fixed by
//!     construction, independent of eviction order and thread interleaving.
//!   * **the no-checkpoint proxy path** — G1 used `open_over`, which has no host, so
//!     it can fire no checkpoint/compaction. This bench drives `create_path`, whose
//!     factory the M7-E seam (`Options::device_wrapper`, `bench-instrument`) lets us
//!     wrap per role, and a `Store::<Wal>` cell fires a real checkpoint (sweep 2).
//!
//! ## Honesty is the score
//!
//! A closed, scoped *negative* is a success; the failure mode is shipping a number
//! that is a measurement artifact. So the apparatus is validated *before* any
//! decision data is trusted (`main` runs the validation suite), and — once
//! reviewed — the harness/trace/metrics are **frozen** before the sweeps run.
//!
//! ```sh
//! cargo bench --bench perf_knobs --features io-uring,bench-instrument
//! ```

use mapdb_uring::backend::Direct;
use mapdb_uring::batch::{ApplyOutcome, WriteBatch};
use mapdb_uring::io::{BufResult, IoBackend, IoFuture, IoKind, IoRole, PageIo};
use mapdb_uring::store::{Options, Store};
use mapdb_uring::BTreeMap;
use std::cell::Cell;
use std::io;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Instant;

use futures_lite::future::block_on;

// ===========================================================================
// Environment
// ===========================================================================

fn fixture_root() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("MAPDB_BENCH_DIR") {
        return std::path::PathBuf::from(d);
    }
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bench-fixtures");
    std::fs::create_dir_all(&p).expect("create fixture root");
    p
}

fn fs_kind(path: &std::path::Path) -> &'static str {
    use std::ffi::CString;
    let c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
    // SAFETY: `c` is a valid NUL-terminated path; `st` is filled by statfs.
    unsafe {
        let mut st: libc::statfs = std::mem::zeroed();
        if libc::statfs(c.as_ptr(), &mut st) != 0 {
            return "unknown";
        }
        match st.f_type {
            0x0102_1994 => "tmpfs (RAM — NOT a device!)",
            0x9123_683E => "btrfs",
            0xEF53 => "ext2/3/4",
            0x5846_5342 => "xfs",
            _ => "other (disk-backed)",
        }
    }
}

/// `/proc/meminfo` `MemAvailable` in bytes (0 if unreadable). The sustained
/// buffered-cold fixture (sweep 2b) is sized ~1.3–1.5× this so a buffered arm
/// stays device-bound instead of self-warming.
fn mem_available_bytes() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            // "MemAvailable:   12345678 kB"
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
            return kb * 1024;
        }
    }
    0
}

// ===========================================================================
// Coldness proof — btrfs silently buffers a misaligned "O_DIRECT" op,
// so coldness/directness is proven with mincore, never asserted.
// ===========================================================================

/// `fsync` + `posix_fadvise(DONTNEED)` on `path`. Eviction is *proven* afterward by
/// `touched_residency` on the specific pages the trace touches, so this no
/// longer computes a wasteful whole-file residency.
fn evict(path: &std::path::Path) {
    use std::os::unix::io::AsRawFd;
    let Ok(f) = std::fs::File::open(path) else {
        return;
    };
    // SAFETY: `fd` is live for the duration of both calls.
    unsafe {
        libc::fsync(f.as_raw_fd());
        libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

/// Resident fraction of ONLY the `touched` data offsets (16 KiB pages), not the
/// whole file. This is the honest cold/direct proof: the frozen trace touches
/// ≤ `readers × gets` leaves, so whole-file residency dilutes a fully-buffered arm
/// below the 5% threshold on a large fixture. mmaps the file once, one `mincore`,
/// then folds over the system pages backing each touched 16 KiB page. Returns
/// `(resident_fraction, touched_page_count)`; fraction 1.0 (fail-closed) if the
/// file can't be mapped/probed.
fn touched_residency(path: &std::path::Path, touched: &[u64]) -> (f64, u64) {
    use std::os::unix::io::AsRawFd;
    if touched.is_empty() {
        return (0.0, 0);
    }
    let Ok(f) = std::fs::File::open(path) else {
        return (1.0, touched.len() as u64);
    };
    let len = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
    if len == 0 {
        return (1.0, touched.len() as u64);
    }
    let sys = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    let n = len.div_ceil(sys);
    // SAFETY: read-only map of the whole file; `v` sized to the page count as
    // mincore requires; unmapped before return.
    unsafe {
        let addr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            f.as_raw_fd(),
            0,
        );
        if addr == libc::MAP_FAILED {
            return (1.0, touched.len() as u64);
        }
        let mut v = vec![0u8; n];
        let rc = libc::mincore(addr, len, v.as_mut_ptr());
        let (mut probed, mut resident) = (0u64, 0u64);
        if rc == 0 {
            for &off in touched {
                // Every system page backing this 16 KiB data page.
                let mut p = off as usize;
                while p < off as usize + PAGE_SIZE as usize && p < len {
                    let idx = p / sys;
                    if idx < n {
                        probed += 1;
                        if v[idx] & 1 == 1 {
                            resident += 1;
                        }
                    }
                    p += sys;
                }
            }
        } else {
            probed = touched.len() as u64;
            resident = probed; // mincore failed ⇒ fail-closed (treat as resident)
        }
        libc::munmap(addr, len);
        let frac = if probed == 0 {
            0.0
        } else {
            resident as f64 / probed as f64
        };
        (frac, touched.len() as u64)
    }
}

/// A Data-role wrapper that records every read offset (for `probe_touched_offsets`).
struct OffsetRec {
    inner: Arc<dyn PageIo>,
    offs: Arc<std::sync::Mutex<std::collections::BTreeSet<u64>>>,
}
impl PageIo for OffsetRec {
    fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
        self.offs.lock().unwrap().insert(offset);
        self.inner.read_exact_at(offset, len)
    }
    fn write_all_at(&self, offset: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult> {
        self.inner.write_all_at(offset, buffer)
    }
    fn datasync(&self) -> IoFuture<'_, io::Result<()>> {
        self.inner.datasync()
    }
    fn size(&self) -> IoFuture<'_, io::Result<u64>> {
        self.inner.size()
    }
    fn shutdown(&self) -> IoFuture<'_, io::Result<()>> {
        self.inner.shutdown()
    }
    fn kind(&self) -> IoKind {
        self.inner.kind()
    }
}

// ===========================================================================
// Instrument — per-thread striped, role-tagged counters + an eager-submission-
// preserving device wrapper. Replaces the 4-RMW `CountIo`.
// ===========================================================================

/// Enough stripes that concurrent readers rarely collide; a collision only adds
/// contention, never miscounts (the total is a sum over all stripes).
const STRIPES: usize = 32;

thread_local! {
    /// This thread's stripe index, assigned round-robin on first use. Fresh per
    /// spawned thread (threads are spawned per run and exit), so a run's counts are
    /// never contaminated by a prior run's stripe assignment.
    static STRIPE: Cell<usize> = const { Cell::new(usize::MAX) };
}
static NEXT_STRIPE: AtomicUsize = AtomicUsize::new(0);

fn my_stripe() -> usize {
    STRIPE.with(|s| {
        let cur = s.get();
        if cur != usize::MAX {
            return cur;
        }
        let n = NEXT_STRIPE.fetch_add(1, Ordering::Relaxed) % STRIPES;
        s.set(n);
        n
    })
}

/// A cache-line-padded atomic so adjacent stripes never share a line — the exact
/// false-sharing the 4-RMW `CountIo` suffered on the measured fast path.
#[repr(align(128))]
struct Padded(AtomicU64);

/// One striped counter: `add` hits only the calling thread's stripe (uncontended);
/// `sum`/`reset` fold across all stripes.
struct Striped {
    cells: [Padded; STRIPES],
}
impl Striped {
    fn new() -> Self {
        Striped {
            cells: std::array::from_fn(|_| Padded(AtomicU64::new(0))),
        }
    }
    #[inline]
    fn add(&self, v: u64) {
        self.cells[my_stripe()].0.fetch_add(v, Ordering::Relaxed);
    }
    fn sum(&self) -> u64 {
        self.cells.iter().map(|c| c.0.load(Ordering::Relaxed)).sum()
    }
    fn reset(&self) {
        for c in &self.cells {
            c.0.store(0, Ordering::Relaxed);
        }
    }
}

/// The five equal-demand quantities for one file role. Reads and writes are
/// counted **and** summed in bytes: a WAL segment is byte-granular, so its byte
/// total can move even when its op count matches.
struct RoleCtr {
    read_ops: Striped,
    read_bytes: Striped,
    write_ops: Striped,
    write_bytes: Striped,
    syncs: Striped,
}
impl RoleCtr {
    fn new() -> Self {
        RoleCtr {
            read_ops: Striped::new(),
            read_bytes: Striped::new(),
            write_ops: Striped::new(),
            write_bytes: Striped::new(),
            syncs: Striped::new(),
        }
    }
    fn reset(&self) {
        self.read_ops.reset();
        self.read_bytes.reset();
        self.write_ops.reset();
        self.write_bytes.reset();
        self.syncs.reset();
    }
    fn snapshot(&self) -> RoleTotals {
        RoleTotals {
            read_ops: self.read_ops.sum(),
            read_bytes: self.read_bytes.sum(),
            write_ops: self.write_ops.sum(),
            write_bytes: self.write_bytes.sum(),
            syncs: self.syncs.sum(),
        }
    }
}

/// Per-role counters for one run. Shared (an `Arc`) with the device wrapper the
/// factory installs; the bench retains its own clone to read (the wrapper must
/// be a shared collector, not a bare fn).
struct RunCounters {
    data: RoleCtr,
    wal: RoleCtr,
    temp: RoleCtr,
    /// Per-role device MINT count (wrapper invocations) — [Data, WalSegment,
    /// CompactTemp]. Proves the seam wraps devices minted AFTER open.
    mints: [AtomicU64; 3],
}
impl RunCounters {
    fn new() -> Arc<Self> {
        Arc::new(RunCounters {
            data: RoleCtr::new(),
            wal: RoleCtr::new(),
            temp: RoleCtr::new(),
            mints: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        })
    }
    fn note_mint(&self, r: IoRole) {
        self.mints[r as usize].fetch_add(1, Ordering::Relaxed);
    }
    fn mints(&self, r: IoRole) -> u64 {
        self.mints[r as usize].load(Ordering::Relaxed)
    }
    fn role(&self, r: IoRole) -> &RoleCtr {
        match r {
            IoRole::Data => &self.data,
            IoRole::WalSegment => &self.wal,
            IoRole::CompactTemp => &self.temp,
        }
    }
    fn reset(&self) {
        self.data.reset();
        self.wal.reset();
        self.temp.reset();
    }
    fn snapshot(&self) -> Totals {
        Totals {
            data: self.data.snapshot(),
            wal: self.wal.snapshot(),
            temp: self.temp.snapshot(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct RoleTotals {
    read_ops: u64,
    read_bytes: u64,
    write_ops: u64,
    write_bytes: u64,
    syncs: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Totals {
    data: RoleTotals,
    wal: RoleTotals,
    temp: RoleTotals,
}

/// The device wrapper the factory installs on every minted device. Counts in the
/// **synchronous prologue** and forwards the inner future untouched — so the M7-B
/// eager-submission contract survives: N unawaited `write_all_at` calls still put
/// N ops in flight from construction. A wrapper that did
/// `Box::pin(async move { inner.write_all_at().await })` would serialize the
/// fan-out and benchmark a serial artifact.
struct CountWrap {
    inner: Arc<dyn PageIo>,
    counters: Arc<RunCounters>,
    role: IoRole,
}

impl PageIo for CountWrap {
    fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
        let c = self.counters.role(self.role);
        c.read_ops.add(1);
        c.read_bytes.add(len as u64);
        self.inner.read_exact_at(offset, len)
    }
    fn write_all_at(&self, offset: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult> {
        let c = self.counters.role(self.role);
        c.write_ops.add(1);
        c.write_bytes.add(buffer.len() as u64);
        // Forward in the prologue — eager submission preserved.
        self.inner.write_all_at(offset, buffer)
    }
    fn datasync(&self) -> IoFuture<'_, io::Result<()>> {
        self.counters.role(self.role).syncs.add(1);
        self.inner.datasync()
    }
    fn size(&self) -> IoFuture<'_, io::Result<u64>> {
        self.inner.size()
    }
    fn shutdown(&self) -> IoFuture<'_, io::Result<()>> {
        self.inner.shutdown()
    }
    /// Forward the concrete kind: a non-forwarding wrapper would relabel a
    /// "Uring arm whose rotated segments are secretly syscall" — the exact mislabel
    /// `IoKind` exists to catch.
    fn kind(&self) -> IoKind {
        self.inner.kind()
    }
}

/// Builds the `DeviceWrapper` closure the factory calls per minted device. Returns
/// the closure plus the shared counters the bench reads. `None` counters ⇒ the
/// wrapper-off control (no wrapper installed at all — that is the caller's job).
fn counting_wrapper(counters: Arc<RunCounters>) -> mapdb_uring::io::DeviceWrapper {
    Arc::new(
        move |dev: Arc<dyn PageIo>, role: IoRole| -> Arc<dyn PageIo> {
            counters.note_mint(role); // count wrapper invocations per role
            Arc::new(CountWrap {
                inner: dev,
                counters: Arc::clone(&counters),
                role,
            })
        },
    )
}

// ===========================================================================
// Fixture + tree-shape probe
// ===========================================================================

const VAL_LEN: usize = 64;

fn key_bytes(i: u64) -> Vec<u8> {
    format!("key{i:012}").into_bytes()
}

/// A built Direct fixture and its measured tree shape.
struct Fixture {
    /// Held only to keep the tempdir (and its files) alive for the fixture's
    /// lifetime; never read directly.
    #[allow(dead_code)]
    dir: tempfile::TempDir,
    path: std::path::PathBuf,
    entries: u64,
    /// Leaf fan-out F (entries per leaf), measured. Keys spaced ≥ F land in
    /// distinct leaves — the compulsory-miss precondition.
    leaf_fanout: u64,
}
impl Fixture {
    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

/// Builds a Direct store of `entries` sequential keys via the **production**
/// `create_path` (Syscall backend, buffered — the file bytes are backend-agnostic,
/// and this is the substrate every arm reopens). Then measures the leaf fan-out.
fn build_direct_fixture(root: &std::path::Path, entries: u64) -> Fixture {
    let dir = tempfile::tempdir_in(root).expect("tempdir");
    let path = dir.path().join("m7e.db");
    let store = Store::<Direct>::create_path(&path, plain_opts(IoBackend::Syscall, false))
        .expect("create fixture");
    let map = BTreeMap::over(store);
    block_on(async {
        let mut i = 0u64;
        while i < entries {
            let mut b = WriteBatch::new();
            let end = (i + 1000).min(entries);
            for k in i..end {
                b = b.insert(key_bytes(k), vec![(k & 0xff) as u8; VAL_LEN]);
            }
            let out = map.apply(b).await.unwrap();
            assert!(
                matches!(out, ApplyOutcome::Applied { .. }),
                "fixture insert applied"
            );
            i = end;
        }
        map.close().await.expect("close fixture");
    });
    let leaf_fanout = probe_leaf_fanout(&path, entries);
    Fixture {
        dir,
        path,
        entries,
        leaf_fanout,
    }
}

/// Options with no wrapper and the given backend/direct mode. Verify-on-open is
/// off (the fixture is trusted; a full-tree walk of a >RAM store would dominate).
fn plain_opts(backend: IoBackend, direct: bool) -> Options {
    Options {
        io_backend: backend,
        direct_io: direct,
        verify_tree_on_open: false,
        ..Options::default()
    }
}

/// Measures the leaf fan-out F: with a huge cache (nothing evicts), `get(key 0)`
/// warms the root path; then the first index `i ≥ 1` whose `get` triggers a *new*
/// Data read has crossed into leaf 1, so `i == F`. Robust to depth: only leaf
/// crossings add reads once the upper path is cached.
fn probe_leaf_fanout(path: &std::path::Path, entries: u64) -> u64 {
    let counters = RunCounters::new();
    let opts = Options {
        cache_capacity: Some(1 << 20), // huge: no eviction during the probe
        device_wrapper: Some(counting_wrapper(Arc::clone(&counters))),
        ..plain_opts(IoBackend::Syscall, false)
    };
    let store = Store::<Direct>::open_path(path, opts).expect("open for probe");
    let map = BTreeMap::over(store);
    let f = block_on(async {
        map.get(key_bytes(0)).await.unwrap();
        let base = counters.data.read_ops.sum();
        let mut i = 1u64;
        while i < entries {
            map.get(key_bytes(i)).await.unwrap();
            if counters.data.read_ops.sum() > base {
                return i;
            }
            i += 1;
        }
        entries.max(1)
    });
    block_on(map.close()).ok();
    f
}

// ===========================================================================
// Compulsory-miss disjoint-partition trace (equal-demand primary)
// ===========================================================================

/// A frozen per-thread key trace: reader `t` replays `keys[t]`, disjoint from
/// every other reader's, spaced past the leaf fan-out so no leaf is touched twice
/// by anyone. Every leaf read is a compulsory miss ⇒ the device read count is
/// fixed by construction, independent of eviction order and interleaving.
struct Trace {
    keys: Vec<Vec<u64>>, // [reader][get] = key index
    stride: u64,
    total_gets: u64,
}

/// A frozen, deterministic PRNG (splitmix64) — used to shuffle the trace so it is
/// identical every run (freeze discipline) yet not sequential on disk.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Builds the trace for `readers` threads × `gets_per_thread` gets. The distinct
/// leaf-index set `{k*stride : k in 0..readers*gets}` (all distinct multiples of
/// `stride ≥ F`, so every touched leaf is distinct and single-touch) is **shuffled
/// with a frozen seed** and then partitioned into readers — so each reader touches
/// a scattered subset in scattered order (an ascending run would give the
/// buffered-cold arm kernel readahead the O_DIRECT arm cannot get, deciding sweep
/// 2b by trace order rather than by direct_io). The shuffle preserves every
/// equal-demand property (still a permutation of the same distinct single-touch
/// leaves) while removing the sequential-readahead confound.
///
/// Panics if the span would exceed the fixture — the caller must size the fixture
/// to the trace, never silently truncate coverage.
fn build_trace(fixture: &Fixture, readers: usize, gets_per_thread: u64) -> Trace {
    let stride = fixture.leaf_fanout.max(1);
    let total_gets = readers as u64 * gets_per_thread;
    let span = total_gets.saturating_mul(stride);
    assert!(
        span <= fixture.entries,
        "trace span {span} (= {readers}×{gets_per_thread}×stride {stride}) exceeds fixture \
         entries {}; grow the fixture or shrink the trace",
        fixture.entries
    );
    let mut all: Vec<u64> = (0..total_gets).map(|k| k * stride).collect();
    // Frozen Fisher-Yates: fixed seed ⇒ identical trace every run.
    let mut rng = 0x5DEE_CE5E_D000_07E5_u64 ^ total_gets.wrapping_mul(0x0000_0100_0000_01B3);
    for i in (1..all.len()).rev() {
        let j = (splitmix64(&mut rng) % (i as u64 + 1)) as usize;
        all.swap(i, j);
    }
    let mut keys = Vec::with_capacity(readers);
    for t in 0..readers {
        let start = t * gets_per_thread as usize;
        keys.push(all[start..start + gets_per_thread as usize].to_vec());
    }
    Trace {
        keys,
        stride,
        total_gets,
    }
}

/// The measurement cache size (pages). Because the trace is **single-touch**, a
/// leaf misses on its first-and-only touch regardless of cache size, so cache size
/// does **not** affect the leaf read count — it only needs to be large enough to
/// keep the whole upper tree pinned so no branch is evicted and re-read (which
/// would make the count non-deterministic under concurrent interleaving). Sizing
/// it to hold every touched page + margin pins the upper tree unconditionally,
/// while single-touch leaves still each force exactly one physical read (they are
/// never preloaded). Confirmed by the equal-demand guard: reads == upper + gets.
fn measurement_cache(upper_pages: u64, trace: &Trace) -> usize {
    (upper_pages + trace.total_gets + 64) as usize
}

/// Measures the trace's upper-tree page count: replay every index once, single
/// thread, huge cache (branches stay resident, leaves each read once). Total Data
/// reads = upper_pages + total_gets, so `upper_pages = reads − total_gets`.
fn probe_upper_pages(fixture: &Fixture, trace: &Trace) -> u64 {
    let counters = RunCounters::new();
    let opts = Options {
        cache_capacity: Some(1 << 20),
        device_wrapper: Some(counting_wrapper(Arc::clone(&counters))),
        ..plain_opts(IoBackend::Syscall, false)
    };
    let store = Store::<Direct>::open_path(fixture.path(), opts).expect("open for upper probe");
    let map = BTreeMap::over(store);
    block_on(async {
        for ks in &trace.keys {
            for &k in ks {
                map.get(key_bytes(k)).await.unwrap();
            }
        }
    });
    let reads = counters.data.read_ops.sum();
    block_on(map.close()).ok();
    reads.saturating_sub(trace.total_gets)
}

/// The exact set of Data file offsets the frozen trace touches (leaf + upper pages),
/// recorded by replaying it once with an offset-recording wrapper. These are the
/// pages the cold/direct guards mincore — the trace working set, not the whole
/// (often sparse) file.
fn probe_touched_offsets(fixture: &Fixture, trace: &Trace) -> Vec<u64> {
    let offs = Arc::new(std::sync::Mutex::new(
        std::collections::BTreeSet::<u64>::new(),
    ));
    let o2 = Arc::clone(&offs);
    let wrapper: mapdb_uring::io::DeviceWrapper =
        Arc::new(move |dev: Arc<dyn PageIo>, role: IoRole| {
            if role == IoRole::Data {
                Arc::new(OffsetRec {
                    inner: dev,
                    offs: Arc::clone(&o2),
                }) as Arc<dyn PageIo>
            } else {
                dev
            }
        });
    let opts = Options {
        cache_capacity: Some(1 << 20),
        device_wrapper: Some(wrapper),
        ..plain_opts(IoBackend::Syscall, false)
    };
    let store = Store::<Direct>::open_path(fixture.path(), opts).expect("open for offset probe");
    let map = BTreeMap::over(store);
    block_on(async {
        for ks in &trace.keys {
            for &k in ks {
                map.get(key_bytes(k)).await.unwrap();
            }
        }
    });
    // Clone out of the mutex — the wrapper closure + per-device shims still hold
    // Arc clones while `store`/`map` are in scope, so try_unwrap would fail.
    let out: Vec<u64> = offs.lock().unwrap().iter().copied().collect();
    block_on(map.close()).ok();
    out
}

/// Warms the exact touched pages into the OS page cache by replaying the trace once
/// through a separate BUFFERED opener, then closes it. Establishes a controlled
/// warm cache state before every warm measured arm so one arm cannot inherit the
/// previous arm's population — the measured store then opens with an empty engine
/// tier but a known-warm kernel cache.
fn warm_pass(fixture: &Fixture, trace: &Trace) {
    let opts = Options {
        cache_capacity: Some(1 << 20),
        ..plain_opts(IoBackend::Syscall, false)
    };
    let store = Store::<Direct>::open_path(fixture.path(), opts).expect("open for warm pass");
    let map = BTreeMap::over(store);
    block_on(async {
        for ks in &trace.keys {
            for &k in ks {
                std::hint::black_box(map.get(key_bytes(k)).await.unwrap());
            }
        }
    });
    block_on(map.close()).ok();
}

// ===========================================================================
// One measured read arm — start barrier, per-run p95, role-tagged demand
// ===========================================================================

struct ArmResult {
    ops_per_s: f64,
    /// p95 computed **within this run** (compare paired run-level p95s;
    /// pooling across runs manufactures false replicates). Consumed by the
    /// run_ab_cell p95 regression guard.
    p95_us: f64,
    elapsed_s: f64,
    totals: Totals,
    /// Residency of the TOUCHED trace pages (not the whole file) right after
    /// this arm's pre-run evict, and again right after join. Cold arms only (`None`
    /// for warm). The guard compares this arm's own pre/post: an O_DIRECT arm whose
    /// touched-residency grew silently buffered (`[!! NOT DIRECT]`); a buffered arm
    /// whose post-evict pre-residency is not ~0 did not evict (`[!! NOT COLD]`).
    /// Sampled inside the arm so the *next* arm's evict cannot erase the evidence.
    pre_residency: Option<f64>,
    post_residency: Option<f64>,
}

/// Replays the frozen `trace` through `backend`/`direct` over the fixture with a
/// measurement cache sized to `cache_capacity`. `wrap` decides whether the
/// counting wrapper is installed (the wrapper-on/off perturbation control).
///
/// **Start barrier:** all readers spawn parked on a [`Barrier`]; the clock starts
/// the instant the barrier releases and stops after join — spawn/join are excluded
/// (fixes `io_default.rs:288`).
#[allow(clippy::too_many_arguments)]
fn run_read_arm(
    backend: IoBackend,
    direct: bool,
    fixture: &Fixture,
    trace: &Trace,
    touched: &[u64],
    cache_capacity: usize,
    wrap: bool,
    cold: bool,
) -> ArmResult {
    // Cold arms: evict, then measure the residency of ONLY the touched pages.
    // Warm arms: WARM the exact touched pages through a separate buffered opener
    // first — so every measured warm arm starts from the SAME OS-cache state
    // and cannot inherit the previous arm's population; then prove they're resident.
    let pre_residency = if cold {
        evict(fixture.path());
        Some(touched_residency(fixture.path(), touched).0)
    } else {
        warm_pass(fixture, trace);
        (!touched.is_empty()).then(|| touched_residency(fixture.path(), touched).0)
    };
    let counters = RunCounters::new();
    let opts = Options {
        cache_capacity: Some(cache_capacity),
        device_wrapper: if wrap {
            Some(counting_wrapper(Arc::clone(&counters)))
        } else {
            None
        },
        ..plain_opts(backend, direct)
    };
    let store = Store::<Direct>::open_path(fixture.path(), opts).expect("open arm");
    let map = BTreeMap::over(store);
    counters.reset();

    let readers = trace.keys.len();
    let barrier = Arc::new(Barrier::new(readers + 1));
    let mut handles = Vec::with_capacity(readers);
    for t in 0..readers {
        let map = map.clone();
        let barrier = Arc::clone(&barrier);
        let ks = trace.keys[t].clone();
        handles.push(std::thread::spawn(move || {
            let mut lat = Vec::with_capacity(ks.len());
            barrier.wait(); // park until the clock starts
            block_on(async {
                for &k in &ks {
                    let t0 = Instant::now();
                    std::hint::black_box(map.get(key_bytes(k)).await.unwrap());
                    lat.push(t0.elapsed().as_secs_f64() * 1e6);
                }
            });
            lat
        }));
    }
    barrier.wait();
    let start = Instant::now();
    let mut all: Vec<f64> = Vec::new();
    for h in handles {
        all.extend(h.join().unwrap());
    }
    let elapsed = start.elapsed().as_secs_f64();
    // Post-run touched residency, sampled BEFORE close and before any next arm can
    // evict/populate. Store still open, but the reads are done — the page cache
    // now reflects exactly this arm's I/O.
    let post_residency = if cold {
        Some(touched_residency(fixture.path(), touched).0)
    } else {
        None
    };

    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Nearest-rank p95 (1-indexed rank ⌈0.95n⌉ → 0-indexed rank−1) — the frozen
    // estimator.
    let p95 = if all.is_empty() {
        0.0
    } else {
        let rank = ((all.len() as f64 * 0.95).ceil() as usize).clamp(1, all.len());
        all[rank - 1]
    };
    let totals = counters.snapshot();
    block_on(map.close()).ok();
    ArmResult {
        ops_per_s: trace.total_gets as f64 / elapsed,
        p95_us: p95,
        elapsed_s: elapsed,
        totals,
        pre_residency,
        post_residency,
    }
}

// ===========================================================================
// Equal-demand guard + metric-class gates
// ===========================================================================

/// The equal-demand verdict for one A/B pair. The gate is layered:
///
///  1. **Exact arm equality** `a.data == b.data` — the anti-cheat: a backend that
///     "wins" by reading fewer pages fails here. This is the real gate; a pair
///     that fails it is *unequal work* and is **dropped and logged**, never
///     displayed (drops G1's 2% tolerance at `io_default.rs:431`).
///  2. **Compulsory-miss structure** `total_gets ≤ read_ops ≤ total_gets +
///     upper_slack` — confirms the trace shape held: every leaf read exactly once
///     (leaves are single-touch, so a leaf read can neither be deduped nor
///     re-issued — the count *cannot* drift on the leaves), plus the small,
///     bounded **shared** upper tree whose read count varies by ±a few pages with
///     single-flight timing (the 4 barrier-released readers collapse a shared root
///     miss to one physical read; a serial probe would not). The slack covers only
///     that upper-tree jitter — it can hide no leaf discrepancy, because leaf
///     re-reads are structurally impossible.
///  3. **No stray durability work** — read-only Direct: zero writes/syncs, WAL and
///     compaction-temp roles untouched.
///
/// The predeclared invalid-pair budget (a bounded fraction of pairs may fail gate
/// 1 to single-flight timing) is enforced by the caller; past the budget the cell
/// is UNRESOLVED, never a retry-until-equal that conditions the sample on timing.
struct DemandVerdict {
    arms_equal: bool,
    structural: bool,
    no_stray: bool,
}
impl DemandVerdict {
    fn valid(&self) -> bool {
        self.arms_equal && self.structural && self.no_stray
    }
}

fn read_demand_verdict(a: &Totals, b: &Totals, upper_pages: u64, trace: &Trace) -> DemandVerdict {
    // Upper-tree jitter bound: each of the (≤readers) descents can, in the worst
    // case of no single-flight dedup, read each upper page once — so the shared
    // upper reads sit in [~1, upper_pages × readers]. A generous slack that still
    // cannot mask a single leaf discrepancy (leaves are single-touch).
    let slack = upper_pages * trace.keys.len() as u64 + 16;
    let structural_ok = |t: &Totals| {
        t.data.read_ops >= trace.total_gets && t.data.read_ops <= trace.total_gets + slack
    };
    let no_stray = |t: &Totals| {
        t.data.write_ops == 0
            && t.data.syncs == 0
            && t.wal == RoleTotals::default()
            && t.temp == RoleTotals::default()
    };
    DemandVerdict {
        arms_equal: a.data == b.data,
        structural: structural_ok(a) && structural_ok(b),
        no_stray: no_stray(a) && no_stray(b),
    }
}

/// Relative spread of a list of samples: (max − min) / median. Used to size the
/// A/A noise floor a real delta must clear by ≥2×.
fn rel_spread(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut v = samples.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = v[v.len() / 2];
    if med == 0.0 {
        return 0.0;
    }
    (v[v.len() - 1] - v[0]) / med
}

fn median(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v[v.len() / 2]
}

// ===========================================================================
// Validation suite — run BEFORE any decision data, then freeze
// ===========================================================================

/// Everything that must hold for a sweep's numbers to mean anything. Each check
/// prints PASS/FAIL; a FAIL means the apparatus, not the backend, is being
/// measured — the sweep must not be trusted until it is green.
fn run_validation(fixture: &Fixture, root: &std::path::Path) -> bool {
    println!("\n===== VALIDATION SUITE (must pass before any decision data) =====\n");
    const DEMAND_PAIRS: usize = 6;
    let mut ok = true;

    // Readers/gets sized so the fastest (warm) arm still runs a while; the cold
    // arm runs far longer. Kept well within the fixture span.
    let readers = 4usize;
    let max_gets = fixture.entries / (fixture.leaf_fanout.max(1) * readers as u64);
    let gets = max_gets.clamp(1, 2000);
    let trace = build_trace(fixture, readers, gets);
    let upper = probe_upper_pages(fixture, &trace);
    let cache_cap = measurement_cache(upper, &trace);
    println!(
        "  fixture: {} entries, leaf fan-out F={}, upper-tree pages={}, trace stride={}",
        fixture.entries, fixture.leaf_fanout, upper, trace.stride
    );
    println!(
        "  trace: {readers} readers × {gets} gets = {} gets; measurement cache={} pages\n\
         \x20 compulsory-miss structure ⇒ Data reads/arm ∈ [gets({}), gets+upper-slack]",
        trace.total_gets, cache_cap, trace.total_gets,
    );

    // --- Check 1: equal-demand holds by construction (warm; both backends) -----
    // Run several pairs and measure the VALID-PAIR RATE (pairs where the two arms
    // did exactly equal demand). This is the empirical basis for the invalid-pair
    // budget: the upper tree is shared, so single-flight timing can make one pair's
    // arms differ by a page — those pairs are dropped, and the rate tells us how
    // many we must run to net enough valid pairs.
    println!("\n  [1] equal-demand — exact arm equality, {DEMAND_PAIRS} pairs:");
    let mut valid = 0usize;
    let mut last = None;
    for k in 0..DEMAND_PAIRS {
        let a = run_read_arm(
            IoBackend::Syscall,
            false,
            fixture,
            &trace,
            &[],
            cache_cap,
            true,
            false,
        );
        let b = run_read_arm(
            IoBackend::Uring,
            false,
            fixture,
            &trace,
            &[],
            cache_cap,
            true,
            false,
        );
        let v = read_demand_verdict(&a.totals, &b.totals, upper, &trace);
        println!(
            "      pair {k}: Syscall reads={} Uring reads={}  arms_equal={} structural={} no_stray={} => {}",
            a.totals.data.read_ops,
            b.totals.data.read_ops,
            v.arms_equal,
            v.structural,
            v.no_stray,
            if v.valid() { "VALID" } else { "drop" },
        );
        if v.valid() {
            valid += 1;
        }
        last = Some((a, b));
    }
    let (s, c) = last.unwrap();
    println!("      valid-pair rate: {valid}/{DEMAND_PAIRS}");
    report(
        &mut ok,
        valid * 2 >= DEMAND_PAIRS,
        "≥ half the pairs did exactly equal demand (sets the invalid-pair budget)",
    );

    // --- Check 2: wrapper-on/off perturbation control (Syscall) ----------------
    // Same backend/trace, wrapper installed vs not: bounds the instrument's cost.
    println!("\n  [2] wrapper-on/off perturbation (Syscall, {PAIRS} pairs):");
    const PAIRS: usize = 5;
    let (mut on, mut off) = (Vec::new(), Vec::new());
    for _ in 0..PAIRS {
        on.push(
            run_read_arm(
                IoBackend::Syscall,
                false,
                fixture,
                &trace,
                &[],
                cache_cap,
                true,
                false,
            )
            .ops_per_s,
        );
        off.push(
            run_read_arm(
                IoBackend::Syscall,
                false,
                fixture,
                &trace,
                &[],
                cache_cap,
                false,
                false,
            )
            .ops_per_s,
        );
    }
    let pert = (median(&on) - median(&off)) / median(&off) * 100.0;
    println!(
        "      wrapper-on median={:.0} ops/s  wrapper-off median={:.0} ops/s  perturbation={:+.1}%",
        median(&on),
        median(&off),
        pert,
    );
    report(
        &mut ok,
        pert.abs() < 10.0,
        "wrapper perturbation within ±10% (else the counter distorts the fast path)",
    );

    // --- Check 3: wrapped A/A noise floor (Syscall vs Syscall) -----------------
    // Both arms identical: the delta distribution is pure host/run noise — the
    // floor every real verdict must clear by ≥2×.
    println!("\n  [3] wrapped A/A noise floor (Syscall vs Syscall, {PAIRS} pairs):");
    let mut aa = Vec::new();
    for _ in 0..PAIRS {
        let a = run_read_arm(
            IoBackend::Syscall,
            false,
            fixture,
            &trace,
            &[],
            cache_cap,
            true,
            false,
        )
        .ops_per_s;
        let b = run_read_arm(
            IoBackend::Syscall,
            false,
            fixture,
            &trace,
            &[],
            cache_cap,
            true,
            false,
        )
        .ops_per_s;
        aa.push((b - a) / a * 100.0);
    }
    // The A/A floor is the spread of the A/A deltas themselves — NOT
    // rel_spread(&on) (check 2's wrapper-on samples, a different experiment).
    let aa_spread = aa.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - aa.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "      A/A deltas: [{:+.1}% .. {:+.1}%]  spread ≈ {:.1}%",
        aa.iter().cloned().fold(f64::INFINITY, f64::min),
        aa.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        aa_spread,
    );
    // Not a hard fail — a wide floor is a fact about the host, reported so a later
    // verdict's gate is set relative to it. Flag an alarming floor.
    report(
        &mut ok,
        aa_spread < 50.0,
        "A/A noise floor < 50% (a wider floor may moot fine deltas — quiet the host)",
    );

    // --- Check 4: window length (INFORMATIONAL) --------------------------------
    // The ≥1 s window is enforced PER SWEEP (each sweep sizes its own fixture +
    // gets so its fastest arm clears 1 s). Here it is reported so
    // the sweep sizing can be calibrated; a short validation window is expected
    // (the validation fixture is deliberately small) and does not fail validation.
    println!("\n  [4] window length (informational — per-sweep sizing enforces ≥1 s):");
    let per_get_us = s.elapsed_s * 1e6 / trace.total_gets as f64;
    println!(
        "      warm Syscall elapsed={:.3}s  warm Uring elapsed={:.3}s  (~{:.2} µs/get warm)",
        s.elapsed_s, c.elapsed_s, per_get_us
    );
    let gets_for_1s = (1.0 / (s.elapsed_s / trace.total_gets as f64)).ceil() as u64;
    println!(
        "      ⇒ a warm sweep needs ≈{} gets total ({} per reader @ {readers} readers) for a 1 s window",
        gets_for_1s,
        gets_for_1s / readers as u64,
    );

    // --- Check 5: future-device-mint (WAL rotation + compaction wrapped) -------
    // The most production-specific part of the seam: devices minted AFTER open.
    println!("\n  [5] future-device-mint (Wal checkpoint + Direct compaction wrapped):");
    let mint_ok = check_future_device_mint(root);
    report(
        &mut ok,
        mint_ok,
        "post-open mints (WAL segment, compaction temp) are wrapped",
    );

    println!(
        "\n===== VALIDATION {} =====",
        if ok {
            "PASSED — apparatus may be frozen"
        } else {
            "FAILED — do not trust any sweep"
        }
    );
    ok
}

/// Forces a WAL rotation (checkpoint) and a compaction on wrapped stores and
/// asserts the collector saw the newly-minted role-tagged devices.
/// Mirrors the crate-internal `tests_checkpoint_compaction` seam test at bench scope.
fn check_future_device_mint(root: &std::path::Path) -> bool {
    let dir = tempfile::tempdir_in(root).expect("tempdir");
    // WAL store: writing + checkpoint rotates a segment (a post-open WalSegment
    // mint) and its data host may compact.
    let wpath = dir.path().join("mint-wal.db");
    let wc = RunCounters::new();
    let wopts = Options {
        device_wrapper: Some(counting_wrapper(Arc::clone(&wc))),
        ..plain_opts(IoBackend::Syscall, false)
    };
    let store = Store::<Wal_>::create_path(&wpath, wopts).expect("create wal");
    let map = BTreeMap::over(store);
    // Snapshot WAL-segment mints right after open+seed but BEFORE the checkpoint —
    // the create-time segment is already counted here, so only a checkpoint
    // ROTATION (a genuine post-open mint) can raise it (the old
    // writes>0||reads>0 was satisfied by the create-time segment).
    let wal_mints_before = block_on(async {
        for pass in 0..2u64 {
            let mut b = WriteBatch::new();
            for k in 0..200u64 {
                b = b.insert(key_bytes(pass * 1000 + k), vec![1u8; VAL_LEN]);
            }
            map.apply(b).await.unwrap();
        }
        let before = wc.mints(IoRole::WalSegment);
        map.store().checkpoint().await.ok();
        map.close().await.ok();
        before
    });
    let wal_seg = wc.mints(IoRole::WalSegment) > wal_mints_before;

    // Direct store: a compaction mints a CompactTemp (post-open).
    let dpath = dir.path().join("mint-direct.db");
    let dc = RunCounters::new();
    let dopts = Options {
        device_wrapper: Some(counting_wrapper(Arc::clone(&dc))),
        ..plain_opts(IoBackend::Syscall, false)
    };
    let store = Store::<Direct>::create_path(&dpath, dopts).expect("create direct");
    let map = BTreeMap::over(store);
    block_on(async {
        for k in 0..2000u64 {
            let b = WriteBatch::new().insert(key_bytes(k), vec![1u8; VAL_LEN]);
            map.apply(b).await.unwrap();
        }
        map.store().compact().await.ok();
        map.close().await.ok();
    });
    // A CompactTemp is only ever minted by a compaction (never at open), so any
    // mint proves the post-open path is wrapped.
    let temp = dc.mints(IoRole::CompactTemp) > 0;

    println!(
        "      WAL-segment mints: {} (before checkpoint {})  CompactTemp mints: {}",
        wc.mints(IoRole::WalSegment),
        wal_mints_before,
        dc.mints(IoRole::CompactTemp),
    );
    wal_seg && temp
}

fn report(ok: &mut bool, pass: bool, what: &str) {
    println!(
        "      {} {what}",
        if pass { "PASS" } else { "*** FAIL ***" }
    );
    *ok &= pass;
}

// The Wal backend type, aliased to keep the one import obviously bench-facing.
use mapdb_uring::backend::Wal as Wal_;

// ===========================================================================
// Generic paired A/B cell — the engine under sweeps 1, 2b, 3 (read-driven)
// ===========================================================================

/// One arm configuration: a backend and whether the data file opens O_DIRECT.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Arm {
    backend: IoBackend,
    direct: bool,
}
impl Arm {
    fn label(self) -> &'static str {
        match (self.backend, self.direct) {
            (IoBackend::Syscall, false) => "syscall",
            (IoBackend::Syscall, true) => "syscall+O_DIRECT",
            (IoBackend::Uring, false) => "uring",
            (IoBackend::Uring, true) => "uring+O_DIRECT",
        }
    }
}

/// The per-cell A/A noise floor: `pairs` `a`-vs-`a` throughput deltas (same config,
/// same trace, both wrapped). Its spread is the floor a real delta must clear by
/// ≥2×. Returned as a percentage (max delta − min delta).
#[allow(clippy::too_many_arguments)]
fn aa_floor_pct(
    fixture: &Fixture,
    trace: &Trace,
    touched: &[u64],
    cache_cap: usize,
    a: Arm,
    cold: bool,
    pairs: usize,
) -> f64 {
    let mut d = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        let x = run_read_arm(
            a.backend, a.direct, fixture, trace, touched, cache_cap, true, cold,
        );
        let y = run_read_arm(
            a.backend, a.direct, fixture, trace, touched, cache_cap, true, cold,
        );
        d.push((y.ops_per_s - x.ops_per_s) / x.ops_per_s * 100.0);
    }
    d.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - d.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// One cell's outcome over its valid pairs. `a_*`/`b_*` are the two arms; the
/// delta and verdict are always **b relative to a** (positive ⇒ b faster).
struct CellVerdict {
    valid: usize,
    total: usize,
    d_median: f64,
    d_lo: f64,
    d_hi: f64,
    a_p95: f64,
    b_p95: f64,
    a_ops: f64,
    b_ops: f64,
    aa_floor: f64,
    fastest_elapsed: f64,
    not_direct: bool,
    not_cold: bool,
    not_warm: bool,
    /// True only for a window-qualified, demand-qualified, within-budget,
    /// non-straddling ADOPT/REGRESS/TIE cell — the only kind a downstream decision
    /// (the SQPOLL go/no-go) may consume.
    resolvable: bool,
    verdict: &'static str,
}

/// Runs `pairs` `a`-vs-`b` pairs for one cell, alternating arm order, dropping
/// unequal-demand pairs (invalid budget). In a `cold` regime each arm evicts first
/// (run_read_arm), and two mincore guards apply **per arm**:
/// an arm with `direct` whose residency grew from its post-evict baseline silently
/// buffered → `[!! NOT DIRECT]`; a **buffered** arm whose post-evict residency is
/// already high means the fixture fits in RAM (evict was a no-op) → `[!! NOT COLD]`
/// (the >RAM buffered-cold arm sweep 2b needs is invalid on a too-small fixture).
/// Predeclared primary = throughput (b vs a); p95 secondary. Verdict against the
/// ≥10% AND ≥2× A/A-floor bar, noise-straddle checked first.
#[allow(clippy::too_many_arguments)]
fn run_ab_cell(
    fixture: &Fixture,
    cold: bool,
    a: Arm,
    b: Arm,
    readers: usize,
    gets_per_thread: u64,
    pairs: usize,
    invalid_budget: usize,
) -> CellVerdict {
    let max_gets = fixture.entries / (readers as u64 * fixture.leaf_fanout.max(1));
    let gets = gets_per_thread.min(max_gets).max(1);
    let trace = build_trace(fixture, readers, gets);
    let touched = probe_touched_offsets(fixture, &trace);
    let upper = probe_upper_pages(fixture, &trace);
    let cache_cap = measurement_cache(upper, &trace);
    let aa_floor = aa_floor_pct(fixture, &trace, &touched, cache_cap, a, cold, pairs);

    let mut deltas = Vec::new();
    let (mut ap, mut bp) = (Vec::new(), Vec::new());
    let (mut ao, mut bo) = (Vec::new(), Vec::new());
    let mut invalid = 0usize;
    let mut not_direct = false;
    let mut not_cold = false;
    let mut not_warm = false;
    let mut fastest = f64::INFINITY;

    // A buffered-cold arm is only genuinely device-bound if its TOUCHED working set
    // (not the sparse file) can't fit in the page cache. `MemAvailable` must be
    // readable (else we cannot prove it — NOT-RUN, treated as not-cold).
    let mem = mem_available_bytes();
    let working_set_bytes = touched.len() as u64 * PAGE_SIZE;
    let cold_workingset_ok = mem > 0 && working_set_bytes as f64 > 1.2 * mem as f64;
    let run = |arm: Arm| {
        run_read_arm(
            arm.backend,
            arm.direct,
            fixture,
            &trace,
            &touched,
            cache_cap,
            true,
            cold,
        )
    };
    // Per-arm guard on the arm's OWN residency (sampled inside run_read_arm, so the
    // next arm's evict cannot erase the evidence). Warm arms prove the warmup
    // established residency; cold arms prove coldness/directness.
    let guard = |arm: Arm, res: &ArmResult, nd: &mut bool, nc: &mut bool, nw: &mut bool| -> bool {
        if !cold {
            // Warm: the touched pages must be resident after the warmup. A low
            // post-warmup residency means the warm state was not established.
            if let Some(pre) = res.pre_residency {
                if pre < 0.9 {
                    *nw = true;
                    return false;
                }
            }
            return true;
        }
        let pre = res.pre_residency.unwrap_or(1.0);
        let post = res.post_residency.unwrap_or(1.0);
        // Fail-closed: a cold arm whose evict left the touched pages resident did
        // not run cold (fadvise no-op / already-resident / unreadable file).
        if pre > 0.05 {
            *nc = true;
            return false;
        }
        if arm.direct {
            // O_DIRECT must not populate the page cache: touched-residency growth
            // means it silently buffered (btrfs STATX_DIOALIGN absent).
            if post - pre > 0.05 {
                *nd = true;
                return false;
            }
        } else if !cold_workingset_ok {
            // Buffered arm whose touched working set fits in RAM — not sustained-cold.
            *nc = true;
            return false;
        }
        true
    };

    for k in 0..pairs {
        let (ra, rb) = if k % 2 == 0 {
            let ra = run(a);
            let rb = run(b);
            (ra, rb)
        } else {
            let rb = run(b);
            let ra = run(a);
            (ra, rb)
        };
        fastest = fastest.min(ra.elapsed_s).min(rb.elapsed_s);

        if !guard(a, &ra, &mut not_direct, &mut not_cold, &mut not_warm)
            || !guard(b, &rb, &mut not_direct, &mut not_cold, &mut not_warm)
        {
            invalid += 1;
            continue;
        }
        if !read_demand_verdict(&ra.totals, &rb.totals, upper, &trace).valid() {
            invalid += 1;
            continue;
        }
        deltas.push((rb.ops_per_s - ra.ops_per_s) / ra.ops_per_s * 100.0);
        ao.push(ra.ops_per_s);
        bo.push(rb.ops_per_s);
        ap.push(ra.p95_us);
        bp.push(rb.p95_us);
    }

    let valid = deltas.len();
    // Sort UNCONDITIONALLY so d_median/d_lo/d_hi are order statistics on EVERY path
    // (the early-UNRESOLVED path used to read them off chronological order).
    deltas.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let (a_p95, b_p95) = (
        if ap.is_empty() { 0.0 } else { median(&ap) },
        if bp.is_empty() { 0.0 } else { median(&bp) },
    );
    // p95 is LOWER-is-better: b regressed if its p95 is >5% worse than a's.
    let p95_regressed = a_p95 > 0.0 && (b_p95 - a_p95) / a_p95 > 0.05;
    let floor = (2.0 * aa_floor).max(10.0);
    // Window gate FIRST: a sub-1 s window cannot resolve anything.
    let verdict = if fastest < 1.0 {
        "UNRESOLVED (short window)"
    } else if not_direct {
        "NOT-DIRECT"
    } else if not_cold {
        "NOT-COLD"
    } else if not_warm {
        "NOT-WARM"
    } else if invalid > invalid_budget || valid < 2 {
        "UNRESOLVED"
    } else {
        let (d_lo, d_hi) = (deltas[0], deltas[deltas.len() - 1]);
        if d_hi - d_lo > floor {
            "UNRESOLVED (noise-straddle)"
        } else if d_lo >= floor {
            // Throughput win — but block adoption on a >5% p95 regression.
            if p95_regressed {
                "TIE (throughput win, p95 regressed >5% — retain)"
            } else {
                "ADOPT-b"
            }
        } else if d_hi <= -floor {
            // Real regression: whole spread clears the noise floor (not −5%).
            "REGRESS-b"
        } else {
            "TIE"
        }
    };
    // Resolvable = a real, qualified comparison the go/no-go may consume.
    let resolvable = matches!(verdict, "ADOPT-b" | "REGRESS-b" | "TIE");

    CellVerdict {
        valid,
        total: valid + invalid,
        d_median: if deltas.is_empty() {
            0.0
        } else {
            deltas[deltas.len() / 2]
        },
        d_lo: deltas.first().copied().unwrap_or(0.0),
        d_hi: deltas.last().copied().unwrap_or(0.0),
        a_p95,
        b_p95,
        a_ops: if ao.is_empty() { 0.0 } else { median(&ao) },
        b_ops: if bo.is_empty() { 0.0 } else { median(&bo) },
        aa_floor,
        fastest_elapsed: fastest,
        not_direct,
        not_cold,
        not_warm,
        resolvable,
        verdict,
    }
}

/// Renders one cell row + any guard flags.
fn print_cell(readers: usize, a: Arm, b: Arm, cv: &CellVerdict) {
    println!(
        "{readers:>7} {:>11.0} {:>11.0} {:>+10.1} {:>10.1} {:>9.1} {:>4.0}/{:<4.0} {:>2}/{:<3} {:>11}",
        cv.a_ops,
        cv.b_ops,
        cv.d_median,
        cv.d_hi - cv.d_lo,
        cv.aa_floor,
        cv.a_p95,
        cv.b_p95,
        cv.valid,
        cv.total,
        cv.verdict,
    );
    let _ = (a, b);
    if cv.fastest_elapsed < 1.0 {
        println!(
            "        [!! WINDOW {:.3}s < 1s] raise gets_per_thread (needs a larger fixture)",
            cv.fastest_elapsed
        );
    }
    if cv.not_direct {
        println!("        [!! NOT DIRECT] an O_DIRECT arm's touched pages grew resident (silently buffered) — excluded");
    }
    if cv.not_cold {
        println!("        [!! NOT COLD] evict left touched pages resident, or the touched working set fits in RAM — excluded");
    }
    if cv.not_warm {
        println!(
            "        [!! NOT WARM] the warm-up did not leave the touched pages resident — excluded"
        );
    }
}

fn cell_header(a_label: &str, b_label: &str) {
    println!(
        "{:>7} {:>11} {:>11} {:>10} {:>10} {:>9} {:>8} {:>6} {:>11}",
        "readers", a_label, b_label, "d_med%", "spread%", "aa%", "p95 a/b", "valid", "verdict"
    );
}

// ===========================================================================
// Sweep 1 — the backend decision (G1 proper), on the production create_path
// ===========================================================================

/// The predeclared PRIMARY deployment concurrency for sweep 1. Exactly ONE
/// reader count carries the adoption verdict per regime; the others are diagnostic
/// / interaction characterization, NEVER independent adoption endpoints (that
/// would be an outcome-shopping hatch). Overridable so the operator can
/// pin the real deployment concurrency before the frozen run; the rest of the
/// curve is a pre-registered extension check only.
fn primary_readers() -> usize {
    std::env::var("MAPDB_M7E_PRIMARY_READERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
}

fn sweep1_backend(fixture: &Fixture, gets_per_thread: u64, pairs: usize) -> Vec<Sweep1Cell> {
    let primary = primary_readers();
    println!("\n\n################ SWEEP 1 — BACKEND (G1 proper) ################");
    println!(
        "Primary metric: THROUGHPUT at the PREDECLARED primary concurrency = {primary} readers.\n\
         a=syscall, b=uring; ADOPT-b ⇒ adopt Uring as default. Gate: ≥10% AND ≥2× the\n\
         A/A noise floor, no >5% p95 regression; else RETAIN Syscall (a closed rejection, not a\n\
         defer). The other reader rows are DIAGNOSTIC characterization, not adoption endpoints.\n\
         Warm and cold-O_DIRECT are SEPARATE verdicts, never blended; a cold-only win is a\n\
         conditional recommendation, never a default switch.\n"
    );
    let budget = pairs / 3; // predeclared: ≤ ⅓ of pairs may be dropped
                            // Warm-buffered: both arms buffered. Cold: both arms O_DIRECT.
    let regimes = [
        ("WARM-buffered (dispatch/copy/scheduling)", false, false),
        ("COLD-O_DIRECT (physical read per leaf)", true, true),
    ];
    let mut cells = Vec::new();
    for (label, cold, direct) in regimes {
        let a = Arm {
            backend: IoBackend::Syscall,
            direct,
        };
        let b = Arm {
            backend: IoBackend::Uring,
            direct,
        };
        println!("\n== {label} ==");
        cell_header(a.label(), b.label());
        for readers in [1usize, 2, 4, 8] {
            let cv = run_ab_cell(fixture, cold, a, b, readers, gets_per_thread, pairs, budget);
            print_cell(readers, a, b, &cv);
            if readers == primary {
                println!(
                    "        ↑ PRIMARY concurrency — this row carries the {label} verdict: {}",
                    cv.verdict
                );
                // Only the primary-concurrency cell feeds the go/no-go — and
                // only if it is itself a resolvable comparison.
                cells.push(Sweep1Cell {
                    regime: label,
                    readers,
                    d_median: cv.d_median,
                    valid: cv.resolvable,
                });
            }
        }
    }
    cells
}

/// A sweep-1 PRIMARY-concurrency cell's headline: the median uring-vs-syscall
/// throughput delta and whether it was a resolvable comparison (feeds the SQPOLL
/// go/no-go). Only the primary concurrency is recorded.
struct Sweep1Cell {
    regime: &'static str,
    readers: usize,
    d_median: f64,
    valid: bool,
}

// ===========================================================================
// Sweep 4 — SQPOLL go/no-go (M7-C). Pre-registered BEFORE any
// sweep-1 number: SQPOLL offloads submission-syscall cost, so it can plausibly
// help ONLY where Uring is already competitive. If Uring is decisively worse in
// EVERY required regime, close SQPOLL as NOT APPLICABLE (parent backend not
// competitive) — a legitimate closure for an internal, near-zero-EV knob, NOT a
// relabelled missing measurement. The actual SQPOLL device plumbing (a bench-only
// config threaded through DeviceFactory → CompioIo → build_proactor) is built ONLY
// if this GO fires on the FROZEN run — like the preallocate fork, it is its own
// sub-step, never built speculatively.
// ===========================================================================

fn sweep4_sqpoll_gonogo(cells: &[Sweep1Cell]) {
    println!("\n\n############ SWEEP 4 — SQPOLL go/no-go (pre-registered) ############");
    // The plausible submission-cost band: SQPOLL might recover a MODEST Uring loss
    // (a few %..~15%) by offloading the submission syscall, but not a large one.
    const BAND: f64 = -15.0;
    let resolvable: Vec<&Sweep1Cell> = cells.iter().filter(|c| c.valid).collect();
    let competitive: Vec<&Sweep1Cell> = resolvable
        .iter()
        .copied()
        .filter(|c| c.d_median > BAND)
        .collect();
    if resolvable.is_empty() {
        println!(
            "  No resolvable sweep-1 cell (all UNRESOLVED / guarded-out). SQPOLL go/no-go\n\
             cannot be evaluated on this run — NOT RUN (re-evaluate on a frozen, ≥1 s-window\n\
             run). This is a not-run, never a loss."
        );
        return;
    }
    println!(
        "  Band: Uring within {BAND:.0}% of Syscall in SOME required regime ⇒ submission\n\
         offload could plausibly matter. Resolvable cells: {}; within-band: {}.",
        resolvable.len(),
        competitive.len()
    );
    if competitive.is_empty() {
        println!(
            "  => NO-GO — CLOSE SQPOLL as NOT APPLICABLE (parent backend not competitive in any\n\
             required regime). Uring's best resolvable delta is {:.1}%, below the {BAND:.0}% band.\n\
             The knob is deleted-after-the-decision; do not build its plumbing.",
            resolvable.iter().map(|c| c.d_median).fold(f64::NEG_INFINITY, f64::max)
        );
    } else {
        println!("  => GO — Uring is competitive in:");
        for c in &competitive {
            println!(
                "       {} @ {} readers: Δ {:+.1}%",
                c.regime, c.readers, c.d_median
            );
        }
        println!(
            "     Build the bench-only SQPOLL DeviceFactory config (its own sub-step: thread\n\
             through CompioIo::open_with → driver_main → build_proactor, uring.rs:557), then\n\
             run the SQPOLL cell off vs on for the representative regime. Runtime-probe;\n\
             permission/kernel skips are NOT RUN, never losses."
        );
    }
}

// ===========================================================================
// Sweep 2b — the direct_io PERFORMANCE verdict (USER: full perf verdict).
// Buffered-cold vs direct-cold at equal engine demand, per backend, on a
// SUSTAINED-COLD (>RAM) fixture so the buffered arm stays device-bound.
// NOT blended with warm.
// ===========================================================================

fn sweep2b_direct_io(fixture: &Fixture, gets_per_thread: u64, pairs: usize) {
    println!("\n\n############ SWEEP 2b — direct_io PERF (full verdict) ############");
    println!(
        "Primary metric: THROUGHPUT (predeclared). a=buffered-cold, b=direct-cold; ADOPT-b ⇒\n\
         direct_io wins the cold read. Requires a >RAM fixture so the buffered arm is truly\n\
         device-bound (else [!! NOT COLD]); the direct arm is mincore-checked ([!! NOT\n\
         DIRECT]). Run per backend — direct_io × backend is a named interaction.\n"
    );
    let fixture_gib = fixture_bytes(fixture) as f64 / (1u64 << 30) as f64;
    let mem_gib = mem_available_bytes() as f64 / (1u64 << 30) as f64;
    println!(
        "fixture file ≈ {fixture_gib:.1} GiB (sparse), MemAvailable ≈ {mem_gib:.1} GiB. The\n\
         authoritative cold proof is PER CELL: the touched WORKING SET must exceed 1.2×\n\
         MemAvailable (not the sparse file size), else the cell self-flags [!! NOT COLD].\n"
    );
    let budget = pairs / 3;
    for backend in [IoBackend::Syscall, IoBackend::Uring] {
        let a = Arm {
            backend,
            direct: false,
        };
        let b = Arm {
            backend,
            direct: true,
        };
        println!("\n== {} : buffered-cold vs direct-cold ==", a.label());
        cell_header(a.label(), b.label());
        for readers in [1usize, 2, 4, 8] {
            let cv = run_ab_cell(fixture, true, a, b, readers, gets_per_thread, pairs, budget);
            print_cell(readers, a, b, &cv);
        }
    }
}

/// The fixture's data-file size in bytes (for the >RAM check).
fn fixture_bytes(fixture: &Fixture) -> u64 {
    std::fs::metadata(fixture.path())
        .map(|m| m.len())
        .unwrap_or(0)
}

// ===========================================================================
// Sweep 2 — checkpoint confirmation. Discharges G1's "production path
// WITH checkpoint" phrase, which the read-only Direct sweeps only vacuously
// satisfy (Direct has no checkpoint — StoreWal does). A Store::<Wal> seeds a
// live set, fires a REAL checkpoint (Metrics.checkpoints > 0), and reads run
// concurrently so read-during-checkpoint latency is observed.
// ===========================================================================

struct CheckpointArm {
    checkpoint_wall_s: f64,
    ok: bool,
    read_p95_us: f64,
    /// Number of reader gets whose interval OVERLAPPED the checkpoint window —
    /// the only reads whose latency reflects checkpoint contention.
    overlap_reads: u64,
    /// Per-role I/O with counters reset AFTER seeding — the checkpoint's durability
    /// work + the reader's (fixed-length) reads. The equal-work gate.
    totals: Totals,
    checkpoints: u64,
    compactions: u64,
}

/// Seeds a fresh `Store::<Wal>` (counting wrapper installed), then times one
/// checkpoint while a **barrier-synchronized** background reader replays a
/// FIXED-LENGTH trace. The reader records each get's interval; only gets
/// overlapping the checkpoint window feed the p95, so the sample does not depend on
/// how long the checkpoint took. A fixed reader trace + a deterministic checkpoint
/// ⇒ equal per-role durability work across arms (asserted by the caller).
fn run_checkpoint_arm(
    root: &std::path::Path,
    backend: IoBackend,
    seed_entries: u64,
    reader_gets: u64,
) -> CheckpointArm {
    let dir = tempfile::tempdir_in(root).expect("tempdir");
    let path = dir.path().join("ckpt.db");
    let counters = RunCounters::new();
    let opts = Options {
        device_wrapper: Some(counting_wrapper(Arc::clone(&counters))),
        ..plain_opts(backend, false)
    };
    let store = Store::<Wal_>::create_path(&path, opts).expect("create wal");
    let map = BTreeMap::over(store);

    // Seed a nontrivial live set so the checkpoint relocates real work (≥10 ms,
    // the WALL metric-class floor).
    block_on(async {
        let mut i = 0u64;
        while i < seed_entries {
            let mut b = WriteBatch::new();
            let end = (i + 1000).min(seed_entries);
            for k in i..end {
                b = b.insert(key_bytes(k), vec![(k & 0xff) as u8; VAL_LEN]);
            }
            map.apply(b).await.unwrap();
            i = end;
        }
    });
    counters.reset(); // measure only the checkpoint + the concurrent reader

    // Barrier-synchronized reader: parks, releases together with the checkpoint
    // clock, then replays a FIXED-LENGTH trace recording each get's interval.
    let barrier = Arc::new(Barrier::new(2));
    let reader = {
        let map = map.clone();
        let barrier = Arc::clone(&barrier);
        std::thread::spawn(move || {
            let mut ivs: Vec<(Instant, Instant)> = Vec::with_capacity(reader_gets as usize);
            barrier.wait();
            block_on(async {
                for i in 0..reader_gets {
                    let k = i.wrapping_mul(2_654_435_761) % seed_entries;
                    let t0 = Instant::now();
                    std::hint::black_box(map.get(key_bytes(k)).await.unwrap());
                    ivs.push((t0, Instant::now()));
                }
            });
            ivs
        })
    };

    barrier.wait();
    let ckpt_start = Instant::now();
    let ok = block_on(map.store().checkpoint()).is_ok();
    let ckpt_end = Instant::now();
    let wall = (ckpt_end - ckpt_start).as_secs_f64();
    let ivs = reader.join().unwrap();

    let totals = counters.snapshot();
    let m = map.metrics();
    block_on(map.close()).ok();

    // Classify: keep only reads whose interval overlapped the checkpoint window.
    let mut lat: Vec<f64> = ivs
        .iter()
        .filter(|(s, e)| *s < ckpt_end && *e > ckpt_start)
        .map(|(s, e)| (*e - *s).as_secs_f64() * 1e6)
        .collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = if lat.is_empty() {
        0.0
    } else {
        let rank = ((lat.len() as f64 * 0.95).ceil() as usize).clamp(1, lat.len());
        lat[rank - 1]
    };
    CheckpointArm {
        checkpoint_wall_s: wall,
        ok,
        read_p95_us: p95,
        overlap_reads: lat.len() as u64,
        totals,
        checkpoints: m.checkpoints,
        compactions: m.compactions,
    }
}

/// The checkpoint's per-role DURABILITY work (writes/bytes/syncs across data +
/// WAL). The equal-work gate compares this exactly across arms — reads are
/// contention-dependent (single-flight vs the checkpoint) so are not gated.
fn ckpt_durability(t: &Totals) -> (u64, u64, u64, u64, u64, u64) {
    (
        t.data.write_ops,
        t.data.write_bytes,
        t.data.syncs,
        t.wal.write_ops,
        t.wal.write_bytes,
        t.wal.syncs,
    )
}

fn sweep2_checkpoint(root: &std::path::Path, seed_entries: u64, pairs: usize) {
    println!("\n\n########## SWEEP 2 — CHECKPOINT CONFIRMATION (Wal) ##########");
    println!(
        "Discharges G1's 'production path WITH checkpoint' (Direct has none — Wal does).\n\
         Primary = CHECKPOINT WALL (a wall metric: gate ≥10% AND ≥10 ms AND ≥2× A/A);\n\
         secondary = read-during-checkpoint p95 (only reads OVERLAPPING the checkpoint\n\
         window, from a barrier-synced fixed-length reader). A pair is valid only if\n\
         both checkpoints returned Ok, per-role durability work (writes/bytes/syncs on\n\
         data+WAL) is EXACTLY equal, checkpoints+compactions match, and each arm saw a\n\
         minimum overlap sample; an invalid-pair budget bounds drops.\n"
    );
    // A fixed reader trace long enough to run past the checkpoint so a good slice
    // overlaps its window; equal count both arms.
    let reader_gets = seed_entries.saturating_mul(4).max(50_000);
    const MIN_OVERLAP: u64 = 30;
    let budget = pairs / 3;
    println!(
        "seeding {seed_entries} entries/arm; reader trace {reader_gets} gets; {pairs} pairs (syscall=a, uring=b)\n\
         {:>6} {:>13} {:>13} {:>10} {:>12} {:>11} {:>14}",
        "pair", "syscall wall", "uring wall", "d_wall%", "p95 s/c us", "overlap s/c", "valid"
    );
    let mut wall_deltas = Vec::new();
    let mut aa = Vec::new();
    let (mut sw, mut cw) = (Vec::new(), Vec::new());
    let mut invalid = 0usize;
    for k in 0..pairs {
        // A/A wall floor: syscall vs syscall.
        let x = run_checkpoint_arm(root, IoBackend::Syscall, seed_entries, reader_gets);
        let y = run_checkpoint_arm(root, IoBackend::Syscall, seed_entries, reader_gets);
        if x.checkpoint_wall_s > 0.0 {
            aa.push((y.checkpoint_wall_s - x.checkpoint_wall_s) / x.checkpoint_wall_s * 100.0);
        }
        // The A/B pair (alternate order).
        let (s, c) = if k % 2 == 0 {
            let s = run_checkpoint_arm(root, IoBackend::Syscall, seed_entries, reader_gets);
            let c = run_checkpoint_arm(root, IoBackend::Uring, seed_entries, reader_gets);
            (s, c)
        } else {
            let c = run_checkpoint_arm(root, IoBackend::Uring, seed_entries, reader_gets);
            let s = run_checkpoint_arm(root, IoBackend::Syscall, seed_entries, reader_gets);
            (s, c)
        };
        // Equal-work gate: both Ok, exact per-role durability equality, equal
        // maintenance metrics, a real checkpoint, a minimum overlap sample each.
        let equal_work = s.ok
            && c.ok
            && s.checkpoints == c.checkpoints
            && s.compactions == c.compactions
            && s.checkpoints > 0
            && ckpt_durability(&s.totals) == ckpt_durability(&c.totals)
            && s.overlap_reads >= MIN_OVERLAP
            && c.overlap_reads >= MIN_OVERLAP;
        let valid = equal_work && s.checkpoint_wall_s > 0.0;
        if valid {
            wall_deltas
                .push((c.checkpoint_wall_s - s.checkpoint_wall_s) / s.checkpoint_wall_s * 100.0);
            sw.push(s.checkpoint_wall_s);
            cw.push(c.checkpoint_wall_s);
        } else {
            invalid += 1;
        }
        println!(
            "{k:>6} {:>12.1}ms {:>12.1}ms {:>+9.1} {:>5.0}/{:<6.0} {:>5}/{:<5} {:>14}",
            s.checkpoint_wall_s * 1e3,
            c.checkpoint_wall_s * 1e3,
            if s.checkpoint_wall_s > 0.0 {
                (c.checkpoint_wall_s - s.checkpoint_wall_s) / s.checkpoint_wall_s * 100.0
            } else {
                0.0
            },
            s.read_p95_us,
            c.read_p95_us,
            s.overlap_reads,
            c.overlap_reads,
            if valid { "VALID" } else { "drop(unequal)" },
        );
    }
    if invalid > budget {
        println!("\n  verdict: UNRESOLVED ({invalid}/{pairs} pairs dropped, over budget {budget})");
        return;
    }

    // Wall metric-class verdict: ≥10% AND ≥10 ms AND ≥2× A/A spread.
    let aa_spread = if aa.is_empty() {
        0.0
    } else {
        aa.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - aa.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    if wall_deltas.len() < 2 {
        println!("\n  verdict: UNRESOLVED (too few valid pairs — checkpoint did equal work?)");
        return;
    }
    wall_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let d_med = wall_deltas[wall_deltas.len() / 2];
    let (d_lo, d_hi) = (wall_deltas[0], wall_deltas[wall_deltas.len() - 1]);
    let s_wall_ms = median(&sw) * 1e3;
    let c_wall_ms = median(&cw) * 1e3;
    let abs_ms = (c_wall_ms - s_wall_ms).abs();
    let floor = (2.0 * aa_spread).max(10.0);
    // Wall time is LOWER-is-better and Δ = (uring − syscall)/syscall, so uring is
    // FASTER when Δ is negative. Adopt on a whole spread below −floor; call a
    // regression on a whole spread above +5% (this sign was once reversed).
    let verdict = if d_hi - d_lo > floor {
        "UNRESOLVED (noise-straddle)"
    } else if abs_ms < 10.0 {
        "TIE (below 10 ms wall floor)"
    } else if d_hi <= -floor {
        "ADOPT-uring (faster checkpoint)"
    } else if d_lo >= 5.0 {
        "REGRESS-uring (slower checkpoint) — retain syscall"
    } else {
        "TIE"
    };
    println!(
        "\n  checkpoint wall: syscall {s_wall_ms:.1} ms / uring {c_wall_ms:.1} ms  \
         (median Δ {d_med:+.1}%, |Δ| {abs_ms:.1} ms, A/A spread {aa_spread:.1}%)\n  verdict: {verdict}"
    );
}

// ===========================================================================
// Sweep 3 — flush fan-out (watermark replacement + causal set_len arm).
//
// The M7-B in-flight watermark is BLIND: eager submission makes in-flight ≈ cap
// the instant flush_all `.collect()`s a chunk, whether the device parallelizes OR
// the btrfs inode lock serializes. This sweep replaces the watermark with
// per-write completion TIMESTAMPS + a cap-scaling wall curve, and attributes any
// serialization to file EXTENSION with a causal set_len control arm.
// ===========================================================================

const PAGE_SIZE: u64 = 16 * 1024; // page.rs:18 (crate-internal; stable since M0)

/// Records each `write_all_at`'s offset + issue/completion instants. The mutex is
/// taken only to push a finished record, never held across an await (the
/// dependent_depth.rs TraceIo discipline).
struct WriteTrace {
    evs: std::sync::Mutex<Vec<(u64, u64, Instant, Instant)>>, // (offset, bytes, issue, done)
    syncs: AtomicU64,
}
impl WriteTrace {
    fn new() -> Arc<Self> {
        Arc::new(WriteTrace {
            evs: std::sync::Mutex::new(Vec::new()),
            syncs: AtomicU64::new(0),
        })
    }
    fn reset(&self) {
        self.evs.lock().unwrap().clear();
        self.syncs.store(0, Ordering::Relaxed);
    }
    fn snapshot(&self) -> Vec<(u64, u64, Instant, Instant)> {
        self.evs.lock().unwrap().clone()
    }
}

/// How the measured flush's target range is pre-arranged — the causal control that
/// attributes fan-out serialization to file growth. `None` = the
/// writes EXTEND the file; `Fallocate` = `fallocate(mode 0)` really allocates the
/// extents first (mirrors the *preallocate remedy* under evaluation); `SetLen` =
/// `ftruncate` only moves `i_size` and leaves a sparse hole (an i_size-only
/// discriminator — its writes are non-extending but still allocate extents).
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // SetLen is the documented i_size-only discriminator (not in the default loop)
enum Presize {
    None,
    SetLen,
    Fallocate,
}
impl Presize {
    fn label(self) -> &'static str {
        match self {
            Presize::None => "extending",
            Presize::SetLen => "set_len (i_size only, sparse)",
            Presize::Fallocate => "fallocate (real extents)",
        }
    }
}

/// Wraps the **Data** device to trace writes; other roles forward untraced. Like
/// `CountWrap`, it submits the inner write in its SYNCHRONOUS prologue and wraps
/// only the completion — so eager submission survives and the fan-out it measures
/// is not serialized by the instrument.
struct TraceWrap {
    inner: Arc<dyn PageIo>,
    trace: Arc<WriteTrace>,
}
impl PageIo for TraceWrap {
    fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
        self.inner.read_exact_at(offset, len)
    }
    fn write_all_at(&self, offset: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult> {
        let issue = Instant::now();
        let bytes = buffer.len() as u64;
        let fut = self.inner.write_all_at(offset, buffer); // eager submit in prologue
        let trace = Arc::clone(&self.trace);
        Box::pin(async move {
            let r = fut.await;
            trace
                .evs
                .lock()
                .unwrap()
                .push((offset, bytes, issue, Instant::now()));
            r
        })
    }
    fn datasync(&self) -> IoFuture<'_, io::Result<()>> {
        self.trace.syncs.fetch_add(1, Ordering::Relaxed);
        self.inner.datasync()
    }
    fn size(&self) -> IoFuture<'_, io::Result<u64>> {
        self.inner.size()
    }
    fn shutdown(&self) -> IoFuture<'_, io::Result<()>> {
        self.inner.shutdown()
    }
    fn kind(&self) -> IoKind {
        self.inner.kind()
    }
}

fn tracing_wrapper(trace: Arc<WriteTrace>) -> mapdb_uring::io::DeviceWrapper {
    Arc::new(
        move |dev: Arc<dyn PageIo>, role: IoRole| -> Arc<dyn PageIo> {
            if role == IoRole::Data {
                Arc::new(TraceWrap {
                    inner: dev,
                    trace: Arc::clone(&trace),
                })
            } else {
                dev
            }
        },
    )
}

struct FlushArm {
    makespan_ms: f64,
    /// Page writes in the measured flush (excludes the header at offset 0).
    n_page_writes: usize,
    write_bytes: u64,
    syncs: u64,
    /// O_DIRECT arm only: the written pages must NOT be resident afterward (they
    /// were freshly appended, so pre-residency is ~0). Resident ⇒ silently buffered.
    not_direct: bool,
}
// NOTE: a per-write completion-SPACING metric is deliberately NOT
// reported. `flush_all` awaits the eager submissions in input order, so a wrapper's
// `done` timestamp is the await-side poll instant, not device completion — a
// serialized flush whose first write finishes last would collapse the later stamps
// together and read as "parallel". The load-bearing, un-foolable signal is the
// cap-scaling MAKESPAN curve below.

/// Seeds a Direct store, pre-arranges the target range per `presize` (the causal
/// control), then times ONE flush of `dirty_keys` distinct-leaf overwrites at
/// fan-out `cap`, tracing every write. Records the equal-work signature
/// (writes/bytes/syncs) and the O_DIRECT directness check.
#[allow(clippy::too_many_arguments)]
fn run_flush_arm(
    root: &std::path::Path,
    backend: IoBackend,
    direct: bool,
    cap: usize,
    dirty_keys: u64,
    leaf_fanout: u64,
    presize: Presize,
) -> FlushArm {
    use std::os::unix::io::AsRawFd;
    let dir = tempfile::tempdir_in(root).expect("tempdir");
    let path = dir.path().join("flush.db");
    let trace = WriteTrace::new();
    let mut opts = Options {
        fanout_cap: Some(cap),
        device_wrapper: Some(tracing_wrapper(Arc::clone(&trace))),
        ..plain_opts(backend, direct)
    };
    opts.limits.max_dirty_pages = 1 << 20;
    let store = Store::<Direct>::create_path(&path, opts).expect("create flush store");
    let map = BTreeMap::over(store);

    let seed = (dirty_keys + 4) * leaf_fanout;
    let seed_v = block_on(async {
        let mut v = None;
        let mut i = 0u64;
        while i < seed {
            let mut b = WriteBatch::new();
            let end = (i + 2000).min(seed);
            for k in i..end {
                b = b.insert(key_bytes(k), vec![(k & 0xff) as u8; VAL_LEN]);
            }
            if let ApplyOutcome::Applied { version, .. } = map.apply(b).await.unwrap() {
                v = Some(version);
            }
            i = end;
        }
        v.unwrap()
    });
    block_on(map.flush(seed_v)).ok();

    // Pre-arrange the target range past the allocation frontier so the measured
    // writes are non-extending. set_len only moves i_size (sparse); fallocate really
    // allocates the extents — the control that mirrors the preallocate remedy.
    if presize != Presize::None {
        let cur = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let extra = (dirty_keys + 128) * PAGE_SIZE * 2;
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for presize");
        match presize {
            Presize::SetLen => {
                f.set_len(cur + extra).expect("set_len");
            }
            Presize::Fallocate => {
                // SAFETY: `fd` is a live writable fd; mode 0 = allocate real extents.
                let rc = unsafe {
                    libc::fallocate(f.as_raw_fd(), 0, cur as libc::off_t, extra as libc::off_t)
                };
                assert_eq!(rc, 0, "fallocate failed: {}", io::Error::last_os_error());
            }
            Presize::None => unreachable!(),
        }
    }

    trace.reset();
    let v = block_on(async {
        let mut b = WriteBatch::new();
        for j in 0..dirty_keys {
            let k = j * leaf_fanout;
            b = b.insert(key_bytes(k), vec![0xAB; VAL_LEN]);
        }
        match map.apply(b).await.unwrap() {
            ApplyOutcome::Applied { version, .. } => version,
            other => panic!("measured overwrite not applied: {other:?}"),
        }
    });
    block_on(map.flush(v)).expect("measured flush");

    let syncs = trace.syncs.load(Ordering::Relaxed);
    let evs: Vec<_> = trace
        .snapshot()
        .into_iter()
        .filter(|(off, ..)| *off != 0)
        .collect();
    // O_DIRECT directness: the freshly-written page offsets must not be resident.
    let not_direct = if direct {
        let offs: Vec<u64> = evs.iter().map(|(off, ..)| *off).collect();
        touched_residency(&path, &offs).0 > 0.05
    } else {
        false
    };
    block_on(map.close()).ok();

    if evs.is_empty() {
        return FlushArm {
            makespan_ms: 0.0,
            n_page_writes: 0,
            write_bytes: 0,
            syncs,
            not_direct,
        };
    }
    let write_bytes = evs.iter().map(|(_, b, ..)| *b).sum();
    let first_issue = evs.iter().map(|(_, _, i, _)| *i).min().unwrap();
    let last_done = evs.iter().map(|(_, _, _, d)| *d).max().unwrap();
    let makespan_ms = (last_done - first_issue).as_secs_f64() * 1e3;
    FlushArm {
        makespan_ms,
        n_page_writes: evs.len(),
        write_bytes,
        syncs,
        not_direct,
    }
}

const SHIPPED_CAP: usize = 64; // M7-B's FLUSH_FANOUT_CAP

/// Runs one (backend, direct, presize) cap-curve: `pairs` repeats per cap, median
/// makespan, an A/A noise floor (repeated SHIPPED_CAP), an exact equal-work check
/// (writes/bytes/syncs constant across caps), and the O_DIRECT directness guard.
/// One curve's result.
struct FlushCurve {
    curve: Vec<(usize, f64)>, // (cap, median makespan ms)
    aa_floor: f64,
    speedup: f64, // makespan(cap=1) / makespan(SHIPPED_CAP)
    equal_work: bool,
    not_direct: bool,
    wrote_all: bool, // every arm wrote >0 pages
}
impl FlushCurve {
    /// Whether this curve is trustworthy — the ONLY kind the trigger may consume (M1).
    fn valid(&self, direct: bool) -> bool {
        self.wrote_all && self.equal_work && !(direct && self.not_direct)
    }
}

#[allow(clippy::too_many_arguments)]
fn flush_curve(
    root: &std::path::Path,
    backend: IoBackend,
    direct: bool,
    presize: Presize,
    dirty: u64,
    leaf_fanout: u64,
    caps: &[usize],
    pairs: usize,
) -> FlushCurve {
    // A/A floor: SHIPPED_CAP repeated, relative spread of makespan.
    let aa: Vec<f64> = (0..pairs.max(2))
        .map(|_| {
            run_flush_arm(
                root,
                backend,
                direct,
                SHIPPED_CAP,
                dirty,
                leaf_fanout,
                presize,
            )
            .makespan_ms
        })
        .collect();
    let aa_floor = rel_spread(&aa);

    let mut curve = Vec::new();
    let mut sig: Option<(usize, u64, u64)> = None;
    let mut equal_work = true;
    let mut not_direct = false;
    let mut wrote_all = true;
    for &cap in caps {
        let mut ms = Vec::new();
        for _ in 0..pairs {
            let a = run_flush_arm(root, backend, direct, cap, dirty, leaf_fanout, presize);
            ms.push(a.makespan_ms);
            not_direct |= a.not_direct;
            wrote_all &= a.n_page_writes > 0; // a zero-write flush is not a datapoint
                                              // Equal work checked on EVERY run, not just the last per cap: the
                                              // same batch must write the same pages/bytes/syncs regardless of cap.
            let s = (a.n_page_writes, a.write_bytes, a.syncs);
            match sig {
                None => sig = Some(s),
                Some(prev) if prev != s => equal_work = false,
                _ => {}
            }
        }
        curve.push((cap, median(&ms)));
    }
    let base = curve.first().map(|c| c.1).unwrap_or(0.0);
    let at_shipped = curve
        .iter()
        .find(|c| c.0 == SHIPPED_CAP)
        .map(|c| c.1)
        .unwrap_or(base);
    let speedup = base / at_shipped.max(1e-9);
    FlushCurve {
        curve,
        aa_floor,
        speedup,
        equal_work,
        not_direct,
        wrote_all,
    }
}

fn sweep3_fanout(root: &std::path::Path, leaf_fanout: u64, pairs: usize) {
    println!("\n\n############### SWEEP 3 — FLUSH FAN-OUT ###############");
    println!(
        "Does fan-out (cap>1) give parallel device service, or is it serialized? Primary =\n\
         the cap-scaling MAKESPAN curve (the in-flight watermark is blind; the\n\
         per-write completion gap is await-order-biased and DIRECTIONAL ONLY).\n\
         Each curve carries an A/A noise floor, an EXACT equal-work check (same writes/\n\
         bytes/syncs at every cap), and — for O_DIRECT — a written-range mincore directness\n\
         guard ([!! NOT DIRECT]). Scoped verdict: does the shipped cap={SHIPPED_CAP} beat\n\
         cap=1 beyond 2× the A/A floor? Causal control: the EXTENDING curve vs the\n\
         FALLOCATE (real-extent) curve — the trigger mirrors the preallocate remedy.\n\
         read-during-flush p99: OUT-OF-SCOPE this run (a single ~ms flush gives no\n\
         stable overlap window; recorded rather than silently dropped).\n"
    );
    let dirty: u64 = 128;
    let caps = [1usize, 8, 16, 32, 64, 128];
    // speedup@SHIPPED per (uring?, direct?, fallocate?) for the trigger.
    // (uring?, direct?, fallocate?) -> (speedup@SHIPPED, curve_valid, aa_floor)
    let mut sp: std::collections::HashMap<(bool, bool, bool), (f64, bool, f64)> =
        Default::default();

    for backend in [IoBackend::Syscall, IoBackend::Uring] {
        let bname = if backend == IoBackend::Syscall {
            "syscall"
        } else {
            "uring"
        };
        for direct in [false, true] {
            let dname = if direct { "O_DIRECT" } else { "buffered" };
            for presize in [Presize::None, Presize::Fallocate] {
                let fc = flush_curve(
                    root,
                    backend,
                    direct,
                    presize,
                    dirty,
                    leaf_fanout,
                    &caps,
                    pairs,
                );
                println!(
                    "\n== {bname} / {dname} / {} — flush makespan vs cap ==",
                    presize.label()
                );
                println!("{:>6} {:>13} {:>14}", "cap", "makespan ms", "speedup/cap1");
                let base = fc.curve[0].1;
                for (cap, ms) in &fc.curve {
                    println!("{cap:>6} {ms:>13.2} {:>13.2}x", base / ms.max(1e-9));
                }
                // The signal bar clears BOTH the 1.5× threshold AND 2× the measured
                // A/A floor: on a noisy host 1.5× may itself be noise.
                let bar = (1.0 + 2.0 * fc.aa_floor).max(1.5);
                let verdict = if !fc.wrote_all {
                    "NOT-RUN (a flush wrote zero pages — store broken? — excluded)"
                } else if !fc.equal_work {
                    "UNEQUAL-WORK (cap changed the durability work — excluded)"
                } else if direct && fc.not_direct {
                    "NOT-DIRECT (O_DIRECT arm silently buffered — excluded)"
                } else if fc.speedup >= bar {
                    "fan-out HELPS (cap>1 beats cap=1 beyond noise)"
                } else if fc.speedup <= 1.0 / bar {
                    "fan-out HURTS (cap>1 slower than cap=1 beyond noise — retain cap=1)"
                } else {
                    "no measurable fan-out effect (within noise)"
                };
                println!(
                    "  A/A floor {:.1}%, speedup@cap{SHIPPED_CAP} = {:.2}× (bar {bar:.2}×) => {verdict}",
                    fc.aa_floor * 100.0,
                    fc.speedup,
                );
                // Store (speedup, valid, aa_floor) — the trigger consumes ONLY valid
                // curves (M1: an excluded curve must never drive the fork decision).
                sp.insert(
                    (
                        backend == IoBackend::Uring,
                        direct,
                        presize == Presize::Fallocate,
                    ),
                    (fc.speedup, fc.valid(direct), fc.aa_floor),
                );
            }
        }
    }

    // Pre-registered preallocate-fork trigger on the uring O_DIRECT arm, on the
    // FALLOCATE control. Consumes ONLY valid control curves (M1); the bar
    // clears the measured A/A floor. Both flat ⇒ the bottleneck is not extent
    // allocation; do NOT build preallocation.
    println!("\n== preallocate-fork trigger (uring / O_DIRECT; fallocate control) ==");
    let ext = sp.get(&(true, true, false)).copied();
    let fal = sp.get(&(true, true, true)).copied();
    match (ext, fal) {
        (Some((ext_sp, true, ext_aa)), Some((fal_sp, true, fal_aa))) => {
            let bar = (1.0 + 2.0 * ext_aa.max(fal_aa)).max(1.5);
            println!(
                "  extending speedup@cap{SHIPPED_CAP} = {ext_sp:.2}× ; fallocate = {fal_sp:.2}×  (bar {bar:.2}×)"
            );
            if ext_sp < bar && fal_sp >= bar {
                println!(
                    "  => OPEN the preallocate fork: extending is flat but real-extent preallocation\n\
                     \x20    unlocks fan-out (a separate follow-up round)."
                );
            } else if ext_sp >= bar {
                println!("  => fan-out ALREADY parallelizes while extending — preallocation is unnecessary.");
            } else {
                println!("  => BOTH arms flat — the bottleneck is NOT extent allocation; do NOT build preallocation.");
            }
        }
        _ => println!(
            "  => trigger NOT-RUN: a uring/O_DIRECT control curve was excluded (NOT-DIRECT /\n\
             \x20    UNEQUAL-WORK / zero-write) — the causal comparison has no valid data (M1)."
        ),
    }
}

/// The auditable environment/commit record printed before any decision data (quiet-host
/// protocol). Best-effort — missing pieces print "?" rather
/// than aborting; the point is that a published number carries the machine + code
/// identity it was measured on.
fn print_environment(root: &std::path::Path) {
    let read = |p: &str| {
        std::fs::read_to_string(p)
            .ok()
            .map(|s| s.trim().to_string())
    };
    let commit = std::process::Command::new("git")
        .args([
            "-C",
            env!("CARGO_MANIFEST_DIR"),
            "rev-parse",
            "--short",
            "HEAD",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "?".into());
    let dirty = std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "status", "--porcelain"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    let gov =
        read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").unwrap_or_else(|| "?".into());
    let load = read("/proc/loadavg").unwrap_or_else(|| "?".into());
    println!("===== ENVIRONMENT / COMMIT RECORD =====");
    println!(
        "  commit:        {commit}{}",
        if dirty { " (+dirty tree)" } else { "" }
    );
    println!("  filesystem:    {} ({})", root.display(), fs_kind(root));
    println!(
        "  MemAvailable:  {:.1} GiB",
        mem_available_bytes() as f64 / (1u64 << 30) as f64
    );
    println!("  cpu governor:  {gov}");
    println!("  loadavg:       {load}");
    println!(
        "  run knobs:     ENTRIES={} GETS={} PAIRS={} PRIMARY_READERS={} CKPT_SEED={}",
        std::env::var("MAPDB_M7E_ENTRIES").unwrap_or_else(|_| "default".into()),
        std::env::var("MAPDB_M7E_GETS").unwrap_or_else(|_| "auto".into()),
        std::env::var("MAPDB_M7E_PAIRS").unwrap_or_else(|_| "6".into()),
        std::env::var("MAPDB_M7E_PRIMARY_READERS").unwrap_or_else(|_| "8".into()),
        std::env::var("MAPDB_M7E_CKPT_SEED").unwrap_or_else(|_| "50000".into()),
    );
    println!(
        "  NOTE: quiet the host (taskset/cpuset pin, performance governor, low loadavg)\n\
         \x20 before trusting a decision run — the harness records but does not enforce this.\n"
    );
}

// ===========================================================================
// main — validation first (the sweeps are added and frozen after review)
// ===========================================================================

fn main() {
    println!("perf_knobs — M7-E \"the decision\" (production-path knob adopt/reject)\n");
    let root = fixture_root();
    let kind = fs_kind(&root);
    println!("fixture filesystem: {} ({kind})", root.display());
    if kind.starts_with("tmpfs") {
        println!(
            "  !! tmpfs — a store has no device under it here; this would be memcpy vs memcpy."
        );
        println!("  !! Set MAPDB_BENCH_DIR to a real disk-backed path. Aborting.");
        return;
    }
    let mem = mem_available_bytes();
    println!(
        "MemAvailable: {:.1} GiB (sweep 2b's sustained-cold fixture targets ~1.3–1.5× this)\n",
        mem as f64 / (1u64 << 30) as f64
    );

    // A modest fixture is enough to validate the apparatus mechanics; the decision
    // sweeps size their own fixtures (warm/cold-O_DIRECT, and sweep 2b's >RAM).
    // Overridable so a quick smoke can shrink it.
    let validation_entries: u64 = std::env::var("MAPDB_M7E_ENTRIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    println!("building validation fixture ({validation_entries} entries)...");
    let fixture = build_direct_fixture(&root, validation_entries);

    let ok = run_validation(&fixture, &root);

    // Decision data runs ONLY when explicitly asked (MAPDB_M7E_RUN=1) and ONLY
    // after validation passes — the freeze/review discipline says numbers are
    // trusted only after the apparatus is reviewed and frozen.
    // A default run is validation-only; a smoke run of the sweeps (short window,
    // meaningless numbers) is for exercising the code path, not for the verdict.
    if std::env::var("MAPDB_M7E_RUN").as_deref() == Ok("1") {
        print_environment(&root);
        if !ok {
            println!("\n!! validation FAILED — refusing to run decision sweeps.");
            return;
        }
        // gets_per_thread: on a real run, sized (with a >RAM fixture) so the
        // fastest arm clears 1 s; overridable. The sweep flags any short window.
        let gets: u64 = std::env::var("MAPDB_M7E_GETS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                (fixture.entries / (fixture.leaf_fanout.max(1) * 8)).clamp(1, 20_000)
            });
        let pairs: usize = std::env::var("MAPDB_M7E_PAIRS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);
        // Range-validate the run knobs so a bad value fails fast with a diagnostic
        // rather than a mid-sweep panic or a meaningless A/A spread.
        if gets == 0 || pairs < 2 || fixture.entries < gets * fixture.leaf_fanout.max(1) {
            println!(
                "!! invalid run config: gets={gets} (need ≥1), pairs={pairs} (need ≥2 for an A/A\n\
                 spread), fixture entries {} (need ≥ gets×F={}). Aborting.",
                fixture.entries,
                gets * fixture.leaf_fanout.max(1)
            );
            return;
        }
        println!(
            "\n(MAPDB_M7E_RUN=1 — running decision sweeps: gets/thread={gets}, pairs={pairs})"
        );
        let sweep1 = sweep1_backend(&fixture, gets, pairs);
        // Sweep 2 (checkpoint) builds its own fresh Wal stores; seed size is a free
        // parameter (enough to push the checkpoint past the 10 ms wall floor).
        let seed: u64 = std::env::var("MAPDB_M7E_CKPT_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000);
        sweep2_checkpoint(&root, seed, pairs);
        // Sweep 2b (direct_io perf) needs a >RAM fixture to be conclusive; on a
        // small fixture it correctly self-flags [!! NOT COLD]. For a real verdict
        // size MAPDB_M7E_ENTRIES so the data file exceeds ~1.3× MemAvailable.
        sweep2b_direct_io(&fixture, gets, pairs);
        // Sweep 3 (fan-out) builds its own small Direct stores per cap cell.
        sweep3_fanout(&root, fixture.leaf_fanout, pairs);
        // Sweep 4 = the SQPOLL go/no-go, evaluated on sweep 1's resolvable cells.
        // The plumbing is built only if GO fires on a frozen run (its own sub-step).
        sweep4_sqpoll_gonogo(&sweep1);
    } else {
        println!(
            "\n(Validation-only run. Set MAPDB_M7E_RUN=1 to run the decision sweeps — after the \n\
             apparatus is reviewed + frozen.)"
        );
    }
    // Keep the fixture alive until here.
    drop(fixture);
}
