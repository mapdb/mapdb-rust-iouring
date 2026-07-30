//! M6 Phase G — **G1 exit decision**: should `CompioIo` (io_uring) be the default
//! `PageIo`? `harness = false`; `required-features = ["io-uring"]`.
//!
//! ## Why this bench exists
//!
//! The M1 decision selected Compio but kept `SyscallIo` as the default
//! behind an **M3 re-decision gate**. M5 did not resolve it: `store_bench`'s only
//! real-file axis used `SyscallIo`, and the M1 `io_boundary` bench is a *device*-
//! boundary microbench, not a store-path one — so `--features io-uring` did not
//! change a single M5 store number. The question was left open, to be carried
//! into M6 with a dedicated store-path Compio run. This is that run.
//!
//! ## The experiment
//!
//! `Store::<Direct>::open_over(dev, opts)` takes the device, so this is a
//! **controlled swap**: identical store code, identical fixture file, identical key
//! trace, identical cache capacity — only the `Arc<dyn PageIo>` differs. Anything
//! that differs between the two arms is the device implementation.
//!
//! ## What G2's trace already settled (and why this bench is read-driven)
//!
//! `benches/dependent_depth.rs` measured **achieved** device queue depth:
//!
//!   * **write path: max READ in-flight = 1 at every client count** — every batch
//!     funnels through the single writer actor thread, so concurrent clients' write
//!     descents never have more than one *read* outstanding. NOTE the scope: that
//!     trace instrumented `read_exact_at` only. Flush *writes* are serialised by
//!     `flush_all` awaiting each one (`direct.rs:498-506`) — an implementation
//!     CHOICE over independently-addressable writes, not a law. So "a proactor has
//!     nothing to work with on writes" is true of the engine as written, NOT by
//!     construction;
//!   * **read path: max in-flight = 1/2/4/8 for 1/2/4/8 readers** — reads run outside
//!     the actor, so concurrent descents do expose independent reads.
//!
//! So the only place io_uring can pay off at this call shape is **concurrent reads**,
//! and that is the axis here. Concretely:
//! a proactor cannot shorten one dependent descent, but it may serve N of them with
//! fewer handoffs. Configured threads are not achieved depth — so this bench
//! **measures** in-flight rather than assuming it.
//!
//! ## Instrument discipline
//!
//!   * **Cache regime is named, never blended.** Warm-buffered answers "does the
//!     backend win when the fixture is in the page cache" (dispatch/copy/scheduling);
//!     cold-buffered answers the first-read question. A small engine cache forces
//!     calls into `PageIo`; it does **not** force physical I/O.
//!   * **Coldness is proven, not asserted** — `posix_fadvise(DONTNEED)` + a `mincore`
//!     residency probe; a row that is still resident is labelled `[!! NOT COLD]`.
//!   * **Paired and order-randomised.** Each (regime, concurrency) cell runs both
//!     arms alternately, A/B then B/A, so drift or thermal effects cannot favour
//!     whichever arm ran first.
//!   * **Equal work is verified**, not assumed: both arms replay the same key trace
//!     against the same file and must report the same device read count. A backend
//!     that "wins" by doing less I/O is not a win.
//!   * **Fixtures on a real filesystem** — `/tmp` is tmpfs here, where a store has no
//!     device under it and this comparison would be memcpy vs memcpy.
//!
//! ```sh
//! cargo bench --bench io_default --features io-uring
//! ```

use mapdb_uring::backend::Direct;
use mapdb_uring::batch::{ApplyOutcome, WriteBatch};
use mapdb_uring::io::{BufResult, CompioIo, IoFuture, PageIo, SyscallIo};
use mapdb_uring::store::{Options, Store};
use mapdb_uring::BTreeMap;
use std::io;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use futures_lite::future::block_on;

const N_ENTRIES: usize = 20_000;
const VAL_LEN: usize = 64;
const GETS_PER_THREAD: usize = 300;

/// Counts device **reads** and tracks the read in-flight watermark, so each arm
/// reports achieved read queue depth and its actual read count. Writes and syncs are
/// forwarded untraced — so nothing here says anything about write queue depth.
///
/// **Instrument caveat (unfixed).** This performs FOUR shared
/// read-modify-writes per read (`reads` +, `inflight` +, `max_inflight` fetch_max,
/// `inflight` -), not "one atomic pair", and those adjacent counters can bounce cache
/// lines on exactly the warm fast path whose backend overhead is being measured. A
/// zero-overhead control run (or sharded/per-thread counters) is required before the
/// concurrent deltas can be attributed to the backend.
struct CountIo {
    inner: Arc<dyn PageIo>,
    reads: AtomicU64,
    inflight: AtomicUsize,
    max_inflight: AtomicUsize,
}

impl CountIo {
    fn new(inner: Arc<dyn PageIo>) -> Arc<Self> {
        Arc::new(CountIo {
            inner,
            reads: AtomicU64::new(0),
            inflight: AtomicUsize::new(0),
            max_inflight: AtomicUsize::new(0),
        })
    }
    fn reset(&self) {
        self.reads.store(0, Ordering::Relaxed);
        self.max_inflight.store(0, Ordering::Relaxed);
    }
    fn stats(&self) -> (u64, usize) {
        (
            self.reads.load(Ordering::Relaxed),
            self.max_inflight.load(Ordering::Relaxed),
        )
    }
}

impl PageIo for CountIo {
    fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let cur = self.inflight.fetch_add(1, Ordering::AcqRel) + 1;
        self.max_inflight.fetch_max(cur, Ordering::AcqRel);
        Box::pin(async move {
            let r = self.inner.read_exact_at(offset, len).await;
            self.inflight.fetch_sub(1, Ordering::AcqRel);
            r
        })
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
}

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

/// `fsync` + `posix_fadvise(DONTNEED)` on the fixture, then prove it with `mincore`.
/// Returns the residency fraction actually achieved (0.0 = fully cold).
fn evict(path: &std::path::Path) -> f64 {
    use std::os::unix::io::AsRawFd;
    let f = std::fs::File::open(path).expect("open for evict");
    let len = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
    // SAFETY: `fd` is live for the duration of both calls.
    unsafe {
        libc::fsync(f.as_raw_fd());
        libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
    if len == 0 {
        return 0.0;
    }
    let pg = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    let n = len.div_ceil(pg);
    // SAFETY: map read-only, probe residency, unmap. `vec` is sized to the
    // mapping's page count as mincore requires.
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
            return 1.0;
        }
        let mut v = vec![0u8; n];
        let rc = libc::mincore(addr, len, v.as_mut_ptr());
        let res = if rc == 0 {
            v.iter().filter(|b| *b & 1 == 1).count()
        } else {
            n
        };
        libc::munmap(addr, len);
        res as f64 / n as f64
    }
}

fn build_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("g1.db");
    let dev: Arc<dyn PageIo> = Arc::new(SyscallIo::open(&path).expect("open"));
    let store = Store::<Direct>::create_with_io(dev, Options::default()).expect("create");
    let map = BTreeMap::over(store);
    block_on(async {
        for chunk in (0..N_ENTRIES).collect::<Vec<_>>().chunks(500) {
            let mut b = WriteBatch::new();
            for &i in chunk {
                b = b.insert(
                    format!("key{i:08}").into_bytes(),
                    vec![(i & 0xff) as u8; VAL_LEN],
                );
            }
            let out = map.apply(b).await.unwrap();
            assert!(matches!(out, ApplyOutcome::Applied { .. }));
        }
        map.close().await.expect("close");
    });
    path
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Syscall,
    Compio,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Syscall => "SyscallIo",
            Arm::Compio => "CompioIo",
        }
    }
    /// Opens an existing fixture (read arms).
    fn open(self, path: &std::path::Path) -> Arc<dyn PageIo> {
        match self {
            Arm::Syscall => Arc::new(SyscallIo::open_existing(path).expect("syscall open")),
            Arm::Compio => Arc::new(CompioIo::open(path).expect("compio open")),
        }
    }
    /// Creates a fresh file (write arm). `CompioIo::open` already creates.
    fn create(self, path: &std::path::Path) -> Arc<dyn PageIo> {
        match self {
            Arm::Syscall => Arc::new(SyscallIo::open(path).expect("syscall create")),
            Arm::Compio => Arc::new(CompioIo::open(path).expect("compio create")),
        }
    }
}

struct Run {
    ops_per_s: f64,
    p95_us: f64,
    reads: u64,
    max_inflight: usize,
}

/// One measured cell: `readers` concurrent point lookups replaying a fixed key
/// trace against `path` through `arm`'s device.
fn run_arm(arm: Arm, path: &std::path::Path, readers: usize, cold: bool) -> Run {
    if cold {
        evict(path);
    }
    let dev = CountIo::new(arm.open(path));
    let opts = Options {
        // Small clean tier: descents reach `PageIo`. (Forces calls into the
        // boundary — NOT necessarily physical I/O; that is what the regime names.)
        cache_capacity: Some(64),
        verify_tree_on_open: false,
        ..Options::default()
    };
    let store =
        Store::<Direct>::open_over(dev.clone() as Arc<dyn PageIo>, opts).expect("open_over");
    let map = BTreeMap::over(store);
    dev.reset();

    let start = Instant::now();
    let mut handles = Vec::new();
    for t in 0..readers {
        let map = map.clone();
        handles.push(std::thread::spawn(move || {
            let mut lat = Vec::with_capacity(GETS_PER_THREAD);
            block_on(async {
                for i in 0..GETS_PER_THREAD {
                    // Deterministic, identical across arms.
                    let k = format!("key{:08}", (t * 7919 + i * 104_729) % N_ENTRIES);
                    let t0 = Instant::now();
                    std::hint::black_box(map.get(k.into_bytes()).await.unwrap());
                    lat.push(t0.elapsed().as_secs_f64() * 1e6);
                }
            });
            lat
        }));
    }
    let mut all: Vec<f64> = Vec::new();
    for h in handles {
        all.extend(h.join().unwrap());
    }
    let elapsed = start.elapsed().as_secs_f64();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = all[((all.len() as f64 * 0.95) as usize).min(all.len() - 1)];
    let (reads, maxif) = dev.stats();
    block_on(map.close()).ok();
    Run {
        ops_per_s: (readers * GETS_PER_THREAD) as f64 / elapsed,
        p95_us: p95,
        reads,
        max_inflight: maxif,
    }
}

/// The regression arm the gate requires: a write/flush-heavy workload. No io_uring
/// *win* is expected — not "by construction", but because `flush_all` serialises its
/// page writes itself (`direct.rs:498-506`), so the device is never offered a write
/// queue. The question is only whether the proactor's hand-off COSTS anything.
///
/// **This is n=2 per arm, unpaired and not order-balanced** — below this bench's own
/// 5-pair standard — and records no read/write/byte/sync counts, so it cannot prove
/// identical durability work. It is consistent with no regression; it is NOT a passed
/// regression check.
fn run_write_arm(arm: Arm, dir: &std::path::Path, batches: usize) -> f64 {
    let path = dir.join(format!("w-{}.db", arm.name()));
    let _ = std::fs::remove_file(&path);
    let dev = arm.create(&path);
    let store = Store::<Direct>::create_with_io(dev, Options::default()).expect("create");
    let map = BTreeMap::over(store);
    let start = Instant::now();
    block_on(async {
        for i in 0..batches {
            let mut b = WriteBatch::new();
            for j in 0..32u32 {
                b = b.insert(
                    format!("w{i:06}{j:04}").into_bytes(),
                    vec![(i & 0xff) as u8; VAL_LEN],
                );
            }
            map.apply(b).await.unwrap();
        }
        map.close().await.expect("close");
    });
    let secs = start.elapsed().as_secs_f64();
    let _ = std::fs::remove_file(&path);
    (batches * 32) as f64 / secs
}

fn main() {
    println!("io_default — M6 Phase G / G1 exit decision (should io_uring be the default?)\n");
    let root = fixture_root();
    println!(
        "fixture filesystem: {} ({})",
        root.display(),
        fs_kind(&root)
    );
    if fs_kind(&root).starts_with("tmpfs") {
        println!(
            "  !! WARNING: tmpfs — this would compare memcpy with memcpy. Set MAPDB_BENCH_DIR."
        );
        return;
    }
    let dir = tempfile::tempdir_in(&root).expect("tempdir");
    println!("\nbuilding fixture ({N_ENTRIES} entries)...");
    let path = build_fixture(dir.path());

    println!(
        "\nG2's trace already bounds this: achieved device queue depth is 1 on the WRITE\n\
         path at every client count (the writer actor is the ceiling), and scales with\n\
         readers. So concurrent reads are the only axis where a proactor can be fed —\n\
         a write/commit-heavy row would be a depth-1 row by construction.\n"
    );

    for cold in [false, true] {
        let regime = if cold {
            "FIRST-TOUCH-COLD (evicted per run, then self-warms — see caveat)"
        } else {
            "WARM-buffered (fixture in page cache; dispatch/copy/scheduling question)"
        };
        if cold {
            let resid = evict(&path);
            println!(
                "\n  [cold check] fixture residency after evict: {:.1}%",
                resid * 100.0
            );
            if resid > 0.05 {
                println!("  [!! NOT COLD] pages still resident — treat this block as warm");
            }
            println!(
                "  CAVEAT (read before using these rows): eviction makes the\n  \
                 FIRST touch of each page cold, but this fixture has only ~186 distinct\n  \
                 pages and the working set is far smaller than RAM, so each run re-warms\n  \
                 itself within its first pass. These rows are therefore\n  \
                 'first-touch-cold, then warm' — NOT a sustained cold-device workload.\n  \
                 They are consequently dominated by run-to-run device noise: observed\n  \
                 p95 for the same cell has ranged 102us..1021us across runs (6-10x).\n  \
                 A real cold regime needs a working set > page cache, or O_DIRECT.\n  \
                 Do not read a win OR a regression out of this block."
            );
        }
        println!("\n== {regime} ==");
        println!(
            "{:>8} {:>12} {:>13} {:>11} {:>10} {:>9} {:>8}  {:>8}",
            "readers", "arm", "ops/s", "p95 us", "reads", "max i/f", "d ops/s", "verdict"
        );
        for readers in [1usize, 2, 4, 8] {
            // PAIRS of (A then B) and (B then A), repeated. A single pair cannot
            // resolve a 10% gate on this host: the same cell's delta has been
            // observed swinging from +81% to -45% across runs. Report the SPREAD,
            // not a mean that hides it.
            const PAIRS: usize = 5;
            let mut deltas = Vec::new();
            let mut unequal = 0usize;
            let (mut s_ops_l, mut c_ops_l) = (Vec::new(), Vec::new());
            let (mut s_p95_l, mut c_p95_l) = (Vec::new(), Vec::new());
            let (mut s_reads, mut c_reads, mut mif) = (0u64, 0u64, 0usize);
            for k in 0..PAIRS {
                // Alternate order every pair so drift cannot favour either arm.
                let (a, b) = if k % 2 == 0 {
                    let a = run_arm(Arm::Syscall, &path, readers, cold);
                    let b = run_arm(Arm::Compio, &path, readers, cold);
                    (a, b)
                } else {
                    let b = run_arm(Arm::Compio, &path, readers, cold);
                    let a = run_arm(Arm::Syscall, &path, readers, cold);
                    (a, b)
                };
                let rmax = a.reads.max(b.reads) as f64;
                let rmin = a.reads.min(b.reads) as f64;
                if rmax > 0.0 && (rmax - rmin) / rmax > 0.02 {
                    unequal += 1;
                }
                deltas.push((b.ops_per_s - a.ops_per_s) / a.ops_per_s * 100.0);
                s_ops_l.push(a.ops_per_s);
                c_ops_l.push(b.ops_per_s);
                s_p95_l.push(a.p95_us);
                c_p95_l.push(b.p95_us);
                s_reads = a.reads;
                c_reads = b.reads;
                mif = a.max_inflight.max(b.max_inflight);
            }
            let med = |v: &mut Vec<f64>| {
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
                v[v.len() / 2]
            };
            deltas.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let d_med = deltas[deltas.len() / 2];
            let (d_lo, d_hi) = (deltas[0], deltas[deltas.len() - 1]);
            let s_ops = med(&mut s_ops_l);
            let c_ops = med(&mut c_ops_l);
            let s_p95 = med(&mut s_p95_l);
            let c_p95 = med(&mut c_p95_l);

            // A verdict is only meaningful if the whole spread agrees. If the
            // range straddles the gate, the honest answer is "unresolved", not
            // whichever number the median happened to land on.
            let verdict = if unequal > 0 {
                "UNEQUAL"
            } else if d_lo >= 10.0 {
                "win"
            } else if d_hi <= -5.0 {
                "regress"
            } else if d_hi - d_lo > 15.0 {
                "NOISE"
            } else {
                "tie"
            };
            println!(
                "{readers:>8} {:>12} {s_ops:>13.0} {s_p95:>11.1} {s_reads:>10} {mif:>9} {:>9} {:>8}",
                Arm::Syscall.name(),
                "—",
                ""
            );
            println!(
                "{:>8} {:>12} {c_ops:>13.0} {c_p95:>11.1} {c_reads:>10} {mif:>9} {d_med:>8.1}% {verdict:>8}",
                "",
                Arm::Compio.name(),
            );
            println!(
                "{:>21}delta over {PAIRS} pairs: median {d_med:+.1}%, range [{d_lo:+.1}%, {d_hi:+.1}%]{}",
                "",
                if unequal > 0 {
                    format!("  ({unequal}/{PAIRS} pairs did unequal I/O)")
                } else {
                    String::new()
                }
            );
        }
    }

    // The regression arm: writes are depth-1 through the actor (G2), so this asks
    // only whether the proactor COSTS anything where it cannot help.
    println!("\n== write/flush-heavy regression arm (depth-1 by construction — G2) ==");
    println!("{:>12} {:>14} {:>10}", "arm", "ops/s", "delta");
    let sw = (run_write_arm(Arm::Syscall, dir.path(), 300)
        + run_write_arm(Arm::Syscall, dir.path(), 300))
        / 2.0;
    let cw = (run_write_arm(Arm::Compio, dir.path(), 300)
        + run_write_arm(Arm::Compio, dir.path(), 300))
        / 2.0;
    let wd = (cw - sw) / sw * 100.0;
    println!("{:>12} {sw:>14.0} {:>10}", Arm::Syscall.name(), "—");
    println!("{:>12} {cw:>14.0} {wd:>9.1}%", Arm::Compio.name());
    println!(
        "  {}",
        if wd <= -5.0 {
            "REGRESSION >5% — the gate's regression arm FAILS"
        } else {
            "within the gate's 5% regression allowance"
        }
    );

    println!(
        "\n== gate (G1) ==\n  \
         Adopt Compio as the DEFAULT only if paired runs show >=10% throughput or p95\n  \
         improvement in the primary workload at its normal concurrency, with no >5%\n  \
         regression in any required low-concurrency / commit-heavy / checkpoint\n  \
         workload, and identical I/O + durability counts. Otherwise RETAIN SyscallIo\n  \
         as the M6 default — that is a closed rejection of the default switch, not a\n  \
         defer.\n\n  \
         Scope limit, stated rather than buried: this is the\n  \
         caller-supplied-device path, which gets NO checkpoint\n  \
         (CheckpointError::Unsupported). It is a controlled causal A/B for read\n  \
         mechanics; it is not a production-path confirmation, and a default switch\n  \
         would need one."
    );
}
