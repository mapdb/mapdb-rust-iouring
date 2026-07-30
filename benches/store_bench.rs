//! M5 store-path benchmarks. `harness = false`: prints an
//! informational table, not a criterion harness.
//!
//! Axes:
//!   * concurrent point reads (1/2/4/8 threads, warm cache);
//!   * `get_many` naïve vs grouped shared-descent across batch sizes;
//!   * writes-visible latency: bulk / batched throughput + single-op **endpoint**
//!     latency (ascending vs interior);
//!   * Direct flush latency;
//!   * WAL group commit (asserts commits coalesce into fewer fdatasyncs);
//!   * cache first-fill vs warm (labelled honestly — not a controlled miss ratio);
//!   * file OS-cache-hot first touch vs engine-cache-hot (labelled honestly — not
//!     cold device I/O; that is the `io_boundary` bench's job).
//!
//! All in-memory axes use `FakeIo` (deterministic, no disk variance); the file
//! axis uses a real temp file so the engine-cache layer is visible over a syscall.
//! These are single-shot informational numbers (no CI/warm-up statistics); reads
//! are `black_box`ed so results are not elided.
//!
//! ```sh
//! cargo bench --bench store_bench
//! ```

use mapdb_uring::backend::{Direct, Wal};
use mapdb_uring::batch::{ApplyOutcome, WriteBatch};
use mapdb_uring::io::{FakeIo, PageIo, SyscallIo};
use mapdb_uring::read::{TreeBuilder, TreeReader};
use mapdb_uring::store::{Options, Store};
use mapdb_uring::{BTreeMap, PageCache};
use std::sync::Arc;
use std::time::Instant;

use futures_lite::future::block_on;

const N_ENTRIES: usize = 50_000;
const VAL_LEN: usize = 64;

/// A tiny deterministic xorshift RNG so the access trace is reproducible without
/// pulling a rand dependency into the benches.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The fixture entries: strictly ascending, unique keys `key{:08}`.
fn entries() -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..N_ENTRIES)
        .map(|i| {
            (
                format!("key{i:08}").into_bytes(),
                vec![(i & 0xff) as u8; VAL_LEN],
            )
        })
        .collect()
}

fn builder() -> TreeBuilder {
    TreeBuilder {
        max_fanout: 0,
        max_leaf_entries: 0,
        first_page_id: 1,
    }
}

/// Build a read-only tree on a device, returning a warmed `TreeReader` (every key
/// read once so the cache is hot) plus the raw keys.
fn warm_reader(dev: Arc<dyn PageIo>) -> (Arc<TreeReader>, Vec<Vec<u8>>) {
    let es = entries();
    let built = builder().build(Arc::clone(&dev), &es).expect("build tree");
    let reader = Arc::new(TreeReader::new(built));
    let keys: Vec<Vec<u8>> = es.iter().map(|(k, _)| k.clone()).collect();
    block_on(async {
        for k in &keys {
            reader.get(k.clone()).await.unwrap();
        }
    });
    (reader, keys)
}

fn main() {
    println!("store_bench — {N_ENTRIES} entries, {VAL_LEN}B values\n");
    let sections: &[(&str, fn())] = &[
        ("concurrent_reads", concurrent_reads),
        ("get_many", get_many_naive_vs_grouped),
        ("get_many_cold", get_many_cold_crosspage),
        ("chain_prefetch", chain_prefetch_cold),
        ("writes_visible", writes_visible),
        ("direct_flush", direct_flush),
        ("wal_group_commit", wal_group_commit),
        ("cache_hit_vs_miss", cache_hit_vs_miss),
        ("cold_vs_hot_file", cold_vs_hot_file),
    ];
    // An optional CLI filter (`cargo bench --bench store_bench -- get_many_cold`)
    // runs only matching sections.
    let filter: Vec<String> = std::env::args().skip(1).collect();
    for (name, f) in sections {
        if !filter.is_empty() && !filter.iter().any(|a| name.contains(a.as_str())) {
            continue;
        }
        let t = Instant::now();
        f();
        // Flush so the table is visible even when stdout is block-buffered to a
        // file, and note the section wall-clock on stderr (unbuffered).
        use std::io::Write;
        let _ = std::io::stdout().flush();
        eprintln!("[{name} done in {:.1}s]", t.elapsed().as_secs_f64());
    }
}

// --- concurrent point reads ------------------------------------------------

fn concurrent_reads() {
    println!("== concurrent point reads (warm cache) ==");
    let (reader, keys) = warm_reader(Arc::new(FakeIo::new(0)));
    let keys = Arc::new(keys);
    println!("{:>8} {:>16} {:>14}", "threads", "reads/s", "ns/read");
    for threads in [1usize, 2, 4, 8] {
        let per = 200_000 / threads;
        let start = Instant::now();
        let mut handles = Vec::new();
        for t in 0..threads {
            let reader = Arc::clone(&reader);
            let keys = Arc::clone(&keys);
            handles.push(std::thread::spawn(move || {
                let mut rng = Rng(0x9E3779B97F4A7C15 ^ (t as u64 + 1));
                block_on(async {
                    for _ in 0..per {
                        let k = keys[rng.below(keys.len())].clone();
                        std::hint::black_box(reader.get(k).await.unwrap());
                    }
                });
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let total = per * threads;
        let elapsed = start.elapsed();
        let per_s = total as f64 / elapsed.as_secs_f64();
        println!(
            "{threads:>8} {:>16.0} {:>14.0}",
            per_s,
            elapsed.as_nanos() as f64 / total as f64
        );
    }
    println!();
}

// --- get_many naïve vs grouped ---------------------------------------------

fn get_many_naive_vs_grouped() {
    println!("== get_many: naïve independent descents vs grouped shared descent ==");
    let (reader, keys) = warm_reader(Arc::new(FakeIo::new(0)));
    println!(
        "{:>10} {:>16} {:>16}",
        "batch", "naive keys/s", "grouped keys/s"
    );
    for batch in [1usize, 8, 64, 512] {
        let rounds = (200_000 / batch).max(1);
        let mut rng = Rng(0xDEADBEEF);
        // Pre-pick the batches so both variants read the identical key sets.
        let sets: Vec<Vec<Vec<u8>>> = (0..rounds)
            .map(|_| {
                (0..batch)
                    .map(|_| keys[rng.below(keys.len())].clone())
                    .collect()
            })
            .collect();

        let naive = time_keys(&sets, |ks| block_on(reader.get_many(ks)).unwrap());
        let grouped = time_keys(&sets, |ks| block_on(reader.get_many_grouped(ks)).unwrap());
        println!("{batch:>10} {naive:>16.0} {grouped:>16.0}");
    }
    println!();
}

// --- get_many COLD cross-page (T2.1 decision gate) --------------------------

/// T2.1: the reopen case — **cold, cross-page batches** — measured as a
/// paired experiment.
///
/// Design:
/// * fixture: a production-shaped tree (page-capacity fanout, ≥512 leaves) on a
///   real file under `MAPDB_BENCH_DIR` (default `target/bench-fixtures`), built
///   once outside every timed region; the filesystem is reported via statfs and
///   tmpfs is flagged;
/// * cells: buffered (labelled engine-cold/OS-hot) and O_DIRECT when the
///   filesystem admits it. O_DIRECT establishes **page-cache bypass** for the
///   timed reads — NOT storage-device-cache coldness (the drive/controller
///   cache is uncontrolled, and the file was recently written);
/// * per round one frozen key set; both arms run the identical set, each on its
///   own fresh default-capacity `PageCache` (production-realistic — eviction
///   behavior may differ per arm and is *measured*, not assumed: physical
///   loads are recorded per arm per sample); arm order alternates per round;
///   results cross-validated; each trace's distinct-page reach is measured
///   independently with an ample-cache untimed pass;
/// * controls: clustered cold batches at every candidate threshold size
///   (seeded by `MAPDB_BENCH_SEED` like every other trace), warm per-arm
///   ample caches asserting zero timed loads (isolates driver overhead), and
///   large-batch (4096) rows;
/// * paired reporting: per-round naive/grouped latency ratios as
///   min/median/max, plus per-arm median latency, throughput, and loads.
///
/// Predeclared adoption rule (T2.1 routing): route production `get_many` to the
/// concurrent grouped form for a batch size only if grouped beats naive by
/// ≥ 20% (the G3 noise floor) in a cold cross-page cell at that size while no
/// control cell (warm, clustered, buffered) regresses by more than 10%.
fn get_many_cold_crosspage() {
    use mapdb_uring::backend::FileGeneration;
    use mapdb_uring::read::BuiltTree;

    println!("== get_many COLD cross-page: naive vs grouped-concurrent (T2.1 gate) ==");
    let dir =
        std::env::var("MAPDB_BENCH_DIR").unwrap_or_else(|_| "target/bench-fixtures".to_string());
    std::fs::create_dir_all(&dir).unwrap();
    let path = std::path::PathBuf::from(&dir).join(format!("t21-cold-{}.data", std::process::id()));
    let _ = std::fs::remove_file(&path);
    println!("fixture: {} ({})", path.display(), fs_kind(&dir));

    // Build a production-shaped tree: page-capacity fanout, 200k entries.
    let n = 200_000usize;
    let data: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
        .map(|i| {
            (
                format!("key{i:08}").into_bytes(),
                vec![(i & 0xff) as u8; VAL_LEN],
            )
        })
        .collect();
    let build_dev = Arc::new(SyscallIo::open(&path).unwrap());
    let built = builder()
        .build(build_dev.clone() as Arc<dyn PageIo>, &data)
        .unwrap();
    let root = built.root;
    let entry_count = built.entry_count;
    let locator = Arc::clone(built.file.locator());
    drop(built);
    block_on(build_dev.shutdown()).unwrap();
    let file_bytes = std::fs::metadata(&path).unwrap().len();

    let keys: Vec<Vec<u8>> = data.iter().map(|(k, _)| k.clone()).collect();
    drop(data);

    for (mode, direct) in [("buffered(OS-hot)", false), ("O_DIRECT", true)] {
        let dev: Arc<dyn PageIo> = match SyscallIo::open_existing_direct(&path, direct) {
            Ok(d) => Arc::new(d),
            Err(e) => {
                println!("[{mode}: unavailable on this filesystem ({e}); cell skipped]");
                continue;
            }
        };
        let fresh_reader = || {
            TreeReader::new(BuiltTree {
                file: FileGeneration::direct(Arc::clone(&dev), Arc::clone(&locator)),
                root,
                entry_count,
            })
        };

        // Tree shape, measured: a single-key descent on a fresh cache loads
        // exactly `height` pages.
        let probe = fresh_reader();
        block_on(probe.get(keys[0].clone())).unwrap().unwrap();
        let height = probe.metrics().loads_issued;
        let leaf_estimate = (file_bytes as usize / 16384).saturating_sub(10);
        println!(
            "[{mode}] {entry_count} entries, {file_bytes} B file, height {height} \
             (leaves ≈ {leaf_estimate})"
        );
        println!(
            "{:>18} {:>6} {:>10} {:>10} {:>10} {:>10} {:>22} {:>16} {:>16} {:>8}",
            "cell",
            "batch",
            "naive ms",
            "grpd ms",
            "naive k/s",
            "grpd k/s",
            "ratio min/med/max",
            "loads-n lo/md/hi",
            "loads-g lo/md/hi",
            "distinct"
        );

        // An ample-cache reader for the untimed distinct-page validation pass.
        let ample_reader = || {
            TreeReader::with_cache(
                BuiltTree {
                    file: FileGeneration::direct(Arc::clone(&dev), Arc::clone(&locator)),
                    root,
                    entry_count,
                },
                Arc::new(PageCache::with_capacity(16 * 1024)),
            )
        };

        // Cold cross-page cells (uniform random keys → spread across leaves).
        // `MAPDB_BENCH_SEED` perturbs every frozen trace — cross-page,
        // clustered, and warm alike — so a nonzero seed is a genuine held-out
        // validation of the whole rule.
        let seed: u64 = std::env::var("MAPDB_BENCH_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let perturb = seed.wrapping_mul(0x9E3779B97F4A7C15);
        for (batch, rounds) in [
            (8usize, 40usize),
            (64, 24),
            (512, 12),
            (1024, 12),
            (2048, 12),
            (4096, 8),
        ] {
            let mut rng = Rng(0xC01D ^ batch as u64 ^ perturb);
            let sets: Vec<Vec<Vec<u8>>> = (0..rounds)
                .map(|_| {
                    (0..batch)
                        .map(|_| keys[rng.below(keys.len())].clone())
                        .collect()
                })
                .collect();
            let cell = paired_cold_cell(&sets, &fresh_reader, &ample_reader);
            // Trace validation: EVERY frozen cross-page trace must genuinely
            // spread (not just the cell median — a cell must not pass while
            // one of its traces fails). The analytic distinct-leaf expectation
            // for uniform sampling is L·(1-(1-1/L)^B); require ≥ 60% of it.
            let l = leaf_estimate as f64;
            let expect_leaves = l * (1.0 - (1.0 - 1.0 / l).powi(batch as i32));
            for (i, d) in cell.distinct.iter().enumerate() {
                assert!(
                    *d >= 0.6 * expect_leaves,
                    "cross-page trace {i} reached {d} distinct pages, expected ≥ {:.0}",
                    0.6 * expect_leaves
                );
            }
            print_cell("cold cross-page", batch, &cell);
        }

        // Control: clustered cold batches (one contiguous key run → few
        // leaves) at every candidate threshold size, same held-out seeding —
        // the adoption rule's no-regression check must run at the size it
        // certifies.
        for (batch, rounds) in [(64usize, 24usize), (512, 12), (2048, 12), (4096, 8)] {
            let mut rng = Rng(0xC1 ^ batch as u64 ^ perturb);
            let sets: Vec<Vec<Vec<u8>>> = (0..rounds)
                .map(|_| {
                    let start = rng.below(keys.len() - batch);
                    keys[start..start + batch].to_vec()
                })
                .collect();
            let cell = paired_cold_cell(&sets, &fresh_reader, &ample_reader);
            print_cell("cold clustered", batch, &cell);
        }

        // Control: warm caches (per-arm, primed identically outside the timed
        // region) — no I/O, so this isolates pure driver overhead. The default
        // clean-tier capacity (1024) is smaller than this tree's ~1100-page
        // working set, so a *genuinely* warm cell needs an ample cache; the
        // loads column asserts the timed region issued zero physical reads.
        for batch in [64usize, 512, 1024, 2048, 4096] {
            let mut rng = Rng(0x3A21 ^ batch as u64 ^ perturb);
            let rounds = if batch <= 2048 { 24 } else { 12 };
            let sets: Vec<Vec<Vec<u8>>> = (0..rounds)
                .map(|_| {
                    (0..batch)
                        .map(|_| keys[rng.below(keys.len())].clone())
                        .collect()
                })
                .collect();
            let roomy_reader = || {
                TreeReader::with_cache(
                    BuiltTree {
                        file: FileGeneration::direct(Arc::clone(&dev), Arc::clone(&locator)),
                        root,
                        entry_count,
                    },
                    Arc::new(PageCache::with_capacity(16 * 1024)),
                )
            };
            let naive_reader = roomy_reader();
            let grouped_reader = roomy_reader();
            for s in &sets {
                std::hint::black_box(block_on(naive_reader.get_many(s.clone())).unwrap());
                std::hint::black_box(block_on(grouped_reader.get_many_grouped(s.clone())).unwrap());
            }
            let primed_loads =
                naive_reader.metrics().loads_issued + grouped_reader.metrics().loads_issued;
            let mut cell = Cell::default();
            for (i, s) in sets.iter().enumerate() {
                let (a, b) = if i % 2 == 0 {
                    ("naive", "grouped")
                } else {
                    ("grouped", "naive")
                };
                for arm in [a, b] {
                    let t = Instant::now();
                    let res = if arm == "naive" {
                        block_on(naive_reader.get_many(s.clone())).unwrap()
                    } else {
                        block_on(grouped_reader.get_many_grouped(s.clone())).unwrap()
                    };
                    let ns = t.elapsed().as_nanos() as f64;
                    std::hint::black_box(&res);
                    if arm == "naive" {
                        cell.naive_ns.push(ns);
                    } else {
                        cell.grouped_ns.push(ns);
                    }
                }
            }
            // Assert the cell was genuinely warm: no physical loads while timed.
            let timed_loads = naive_reader.metrics().loads_issued
                + grouped_reader.metrics().loads_issued
                - primed_loads;
            assert_eq!(timed_loads, 0, "warm control issued physical reads");
            print_cell("warm (driver ohd)", batch, &cell);
        }
        block_on(dev.shutdown()).unwrap();
    }
    let _ = std::fs::remove_file(&path);
    println!();
}

// --- T2.2: chain-prefetch cold large-value gets -----------------------------

/// Records each physical read's (offset, issue, completion) so the
/// "issues ≥1 device-latency before demand" requirement is answered by a
/// **trace of the named workload**, not inferred from end-to-end timing.
/// The inner call runs in the prologue so the wrapper preserves the
/// backend's eager-submission behavior.
struct TraceIo {
    inner: Arc<dyn PageIo>,
    log: std::sync::Mutex<Vec<(u64, Instant, Instant)>>,
}

impl TraceIo {
    fn take(&self) -> Vec<(u64, Instant, Instant)> {
        std::mem::take(&mut self.log.lock().unwrap())
    }
}

impl PageIo for TraceIo {
    fn read_exact_at(
        &self,
        offset: u64,
        len: usize,
    ) -> mapdb_uring::io::IoFuture<'_, std::io::Result<Box<[u8]>>> {
        let issue = Instant::now();
        let fut = self.inner.read_exact_at(offset, len);
        Box::pin(async move {
            let r = fut.await;
            self.log
                .lock()
                .unwrap()
                .push((offset, issue, Instant::now()));
            r
        })
    }
    fn write_all_at(
        &self,
        offset: u64,
        buffer: Box<[u8]>,
    ) -> mapdb_uring::io::IoFuture<'_, mapdb_uring::io::BufResult> {
        self.inner.write_all_at(offset, buffer)
    }
    fn datasync(&self) -> mapdb_uring::io::IoFuture<'_, std::io::Result<()>> {
        self.inner.datasync()
    }
    fn size(&self) -> mapdb_uring::io::IoFuture<'_, std::io::Result<u64>> {
        self.inner.size()
    }
    fn shutdown(&self) -> mapdb_uring::io::IoFuture<'_, std::io::Result<()>> {
        self.inner.shutdown()
    }
    fn kind(&self) -> mapdb_uring::io::IoKind {
        self.inner.kind()
    }
}

/// Opens with a brief retry on `AlreadyOpen`: the previous sample's `close()`
/// has completed, but the sidecar-lock release can lag it by a scheduler beat.
fn open_direct_retry(path: &std::path::Path, opts: &dyn Fn() -> Options) -> Store<Direct> {
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match Store::<Direct>::open_path(path, opts()) {
            Ok(s) => return s,
            Err(mapdb_uring::error::OpenError::AlreadyOpen) if Instant::now() < deadline => {
                std::thread::yield_now();
            }
            Err(e) => panic!("open failed: {e:?}"),
        }
    }
}

/// [`open_direct_retry`] for the WAL fixture.
fn open_wal_retry(path: &std::path::Path, opts: &dyn Fn() -> Options) -> Store<Wal> {
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match Store::<Wal>::open_path(path, opts()) {
            Ok(s) => return s,
            Err(mapdb_uring::error::OpenError::AlreadyOpen) if Instant::now() < deadline => {
                std::thread::yield_now();
            }
            Err(e) => panic!("open failed: {e:?}"),
        }
    }
}

/// A deterministic value spanning exactly `links` chain links.
fn chain_value(links: usize, seed: u8) -> Vec<u8> {
    let len = mapdb_uring::page::overflow_payload_capacity() * links - 7;
    (0..len)
        .map(|i| seed.wrapping_add((i % 251) as u8))
        .collect()
}

/// The T2.2 experiment gate: paired cold large-value point gets on production
/// path-backed stores, prefetch off vs on, per-sample fresh store opens (cold
/// engine cache every sample). O_DIRECT rows are **page-cache-bypassed**, NOT
/// device-cold (storage cache uncontrolled; the file was recently written).
/// Buffered rows are OS-hot. Every prefetch-arm sample asserts the exact
/// counter contract — engaged=1, consumed=links−1, unused=0, redundant=0,
/// fallback=0 — so the ≤1% pre-declared amplification budget is enforced at 0%
/// rather than reported loosely.
///
/// Pre-declared adoption rule: prefetch may ship for a
/// layout/io-mode/size cell class only if its cold cell wins by ≥ 1.5× on
/// median paired ratio with the amplification contract intact AND no control
/// cell (small-value, warm, reread-after-cold noted separately) regresses more
/// than 5%, on both the decision trace and a held-out `MAPDB_BENCH_SEED` trace.
/// A StoreDirect win cannot enable WAL; a win under O_DIRECT alone cannot
/// enable buffered. Default-on additionally requires
/// cancellation-independent speculative ownership — a named precondition
/// that holds regardless of the numbers here.
fn chain_prefetch_cold() {
    use mapdb_uring::map::BTreeMap;

    println!("== chain prefetch COLD large-value gets: off vs on (T2.2 gate) ==");
    let dir =
        std::env::var("MAPDB_BENCH_DIR").unwrap_or_else(|_| "target/bench-fixtures".to_string());
    std::fs::create_dir_all(&dir).unwrap();
    let seed: u64 = std::env::var("MAPDB_BENCH_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let perturb = seed.wrapping_mul(0x9E3779B97F4A7C15);
    println!("fixture dir: {dir} ({}), seed {seed}", fs_kind(&dir));

    let nvals = 8usize; // distinct values per size (each open is a fresh cache)
    let sizes: &[(usize, usize)] = &[(4usize, 24usize), (16, 16), (64, 12)];
    let vkey = |links: usize, i: usize| format!("v{links:02}k{i:03}").into_bytes();

    let reopen_opts = |direct: bool| Options {
        direct_io: direct,
        verify_tree_on_open: false, // per-sample reopens: no full-tree walk
        ..Options::default()
    };

    // ---- Direct fixture: one path-backed store holding every size + controls.
    let dpath = std::path::PathBuf::from(&dir).join(format!("t22-direct-{}", std::process::id()));
    {
        let store = Store::<Direct>::create_path(&dpath, Options::default()).unwrap();
        let map = BTreeMap::over(store);
        for &(links, _) in sizes {
            for i in 0..nvals {
                let b = WriteBatch::new().insert(vkey(links, i), chain_value(links, i as u8));
                block_on(map.apply(b)).unwrap();
            }
        }
        for i in 0..nvals {
            // Small-value control: 2 links, below the engagement threshold.
            let b = WriteBatch::new().insert(vkey(2, i), chain_value(2, i as u8));
            block_on(map.apply(b)).unwrap();
        }
        block_on(map.close()).unwrap();
    }

    // One paired cold sample: fresh store open per arm, cold cache, one get.
    let cold_sample = |path: &std::path::Path,
                       opts: &dyn Fn() -> Options,
                       key: &[u8],
                       links: usize,
                       prefetch: bool|
     -> (f64, f64) {
        let store = open_direct_retry(path, opts);
        let map = BTreeMap::over(store);
        map.set_chain_prefetch(prefetch);
        let t = Instant::now();
        let v = block_on(map.get(key.to_vec())).unwrap().unwrap();
        let ns = t.elapsed().as_nanos() as f64;
        std::hint::black_box(&v);
        assert_eq!(v.len(), chain_value(links, 0).len());
        let s = map.chain_prefetch_stats();
        if prefetch && links >= 4 {
            // The amplification contract, enforced exactly (see doc above).
            assert_eq!(s.engaged_chains, 1, "cold chain must engage");
            assert_eq!(s.consumed, links as u64 - 1);
            assert_eq!((s.unused, s.redundant, s.fallbacks), (0, 0, 0));
        } else {
            assert_eq!(s.engaged_chains, 0, "must not engage");
        }
        // TOTAL physical reads: cache-issued loads + speculative device reads
        // (private reads bypass the cache counter — omitting them would
        // understate the on-arm's device I/O). Fresh store per sample, so
        // absolutes are per-sample.
        let loads = (map.cache_metrics().loads_issued + s.issued) as f64;
        block_on(map.close()).unwrap();
        (ns, loads)
    };

    for (mode, direct) in [("buffered(OS-hot)", false), ("O_DIRECT(pc-bypassed)", true)] {
        // Probe availability — and *close* (a dropped store releases its
        // sidecar lock asynchronously, racing the next open into AlreadyOpen).
        match Store::<Direct>::open_path(&dpath, reopen_opts(direct)) {
            Ok(s) => block_on(s.close()).unwrap(),
            Err(_) => {
                println!("[{mode}: unavailable on this filesystem; cells skipped]");
                continue;
            }
        }
        println!("[direct-contig {mode}]");
        println!(
            "{:>18} {:>6} {:>10} {:>10} {:>10} {:>10} {:>22} {:>16} {:>16} {:>8}",
            "cell",
            "links",
            "naive ms",
            "grpd ms",
            "naive k/s",
            "grpd k/s",
            "ratio min/med/max",
            "loads-n lo/md/hi",
            "loads-g lo/md/hi",
            "distinct"
        );
        for &(links, samples) in sizes {
            let mut cell = Cell::default();
            for i in 0..samples {
                let key = vkey(links, (i as u64 ^ perturb) as usize % nvals);
                let flip = (i as u64 + perturb) % 2 == 0;
                for arm_on in if flip { [false, true] } else { [true, false] } {
                    let (ns, loads) =
                        cold_sample(&dpath, &|| reopen_opts(direct), &key, links, arm_on);
                    if arm_on {
                        cell.grouped_ns.push(ns);
                        cell.grouped_loads.push(loads);
                    } else {
                        cell.naive_ns.push(ns);
                        cell.naive_loads.push(loads);
                    }
                }
            }
            print_cell("cold chain", links, &cell);
        }
        // Control: small values (2 links — never engages; the ratio is pure
        // routing overhead and must sit at ~1).
        let mut small = Cell::default();
        for i in 0..16 {
            let key = vkey(2, (i as u64 ^ perturb) as usize % nvals);
            let flip = (i as u64 + perturb) % 2 == 0;
            for arm_on in if flip { [false, true] } else { [true, false] } {
                let (ns, loads) = cold_sample(&dpath, &|| reopen_opts(direct), &key, 2, arm_on);
                if arm_on {
                    small.grouped_ns.push(ns);
                    small.grouped_loads.push(loads);
                } else {
                    small.naive_ns.push(ns);
                    small.naive_loads.push(loads);
                }
            }
        }
        print_cell("small ctrl", 2, &small);

        // Control: genuinely warm chain (primed by a demand read, so the links
        // are cache-resident; the warm-head guard must decline and both arms
        // must be load-free and equal).
        {
            let store = open_direct_retry(&dpath, &|| reopen_opts(direct));
            let map = BTreeMap::over(store);
            let key = vkey(16, 0);
            block_on(map.get(key.clone())).unwrap().unwrap(); // demand prime
            let primed_loads = map.cache_metrics().loads_issued;
            let mut warm = Cell::default();
            for i in 0..24 {
                for arm_on in if i % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    map.set_chain_prefetch(arm_on);
                    let t = Instant::now();
                    std::hint::black_box(block_on(map.get(key.clone())).unwrap().unwrap());
                    let ns = t.elapsed().as_nanos() as f64;
                    if arm_on {
                        warm.grouped_ns.push(ns);
                    } else {
                        warm.naive_ns.push(ns);
                    }
                }
            }
            assert_eq!(
                map.cache_metrics().loads_issued,
                primed_loads,
                "warm control issued physical reads"
            );
            assert_eq!(
                map.chain_prefetch_stats().engaged_chains,
                0,
                "warm-head guard failed"
            );
            print_cell("warm ctrl", 16, &warm);
            block_on(map.close()).unwrap();
        }

        // Cost row (not a gate cell — the tradeoff made visible): prefetch
        // leaves links out of the cache, so a *re-read* after a prefetched cold
        // read demand-loads the chain again, where the naive arm re-reads warm.
        {
            let mut cost = Cell::default();
            for i in 0..8 {
                for arm_on in [false, true] {
                    let store = open_direct_retry(&dpath, &|| reopen_opts(direct));
                    let map = BTreeMap::over(store);
                    map.set_chain_prefetch(arm_on);
                    let key = vkey(16, (i as u64 ^ perturb) as usize % nvals);
                    block_on(map.get(key.clone())).unwrap().unwrap(); // cold
                    let t = Instant::now();
                    std::hint::black_box(block_on(map.get(key.clone())).unwrap().unwrap());
                    let ns = t.elapsed().as_nanos() as f64;
                    // Total device reads across cold+reread: cache loads plus
                    // speculative reads — the on-arm's reread demand-loads the
                    // links its cold read kept private.
                    let loads = (map.cache_metrics().loads_issued
                        + map.chain_prefetch_stats().issued) as f64;
                    if arm_on {
                        cost.grouped_ns.push(ns);
                        cost.grouped_loads.push(loads);
                    } else {
                        cost.naive_ns.push(ns);
                        cost.naive_loads.push(loads);
                    }
                    block_on(map.close()).unwrap();
                }
            }
            print_cell("reread cost", 16, &cost);
        }
    }

    // ---- WAL fixtures: overlay layout (records, buffered-only member), then
    // the same store post-checkpoint (relocated into the data member).
    let wpath = std::path::PathBuf::from(&dir).join(format!("t22-wal-{}", std::process::id()));
    {
        let store = Store::<Wal>::create_path(&wpath, Options::default()).unwrap();
        let map = BTreeMap::over(store);
        for i in 0..nvals {
            let b = WriteBatch::new().insert(vkey(16, i), chain_value(16, i as u8));
            block_on(map.apply(b)).unwrap();
        }
        block_on(map.close()).unwrap();
    }
    let wal_cell = |label: &str, direct: bool, expect_overlay: bool| {
        let opts = || Options {
            direct_io: direct,
            verify_tree_on_open: false,
            ..Options::default()
        };
        match Store::<Wal>::open_path(&wpath, opts()) {
            Err(e) => {
                println!("[{label}: unavailable ({e}); cell skipped]");
                return;
            }
            Ok(s) => {
                assert_eq!(
                    s.metrics().overlay_pages > 0,
                    expect_overlay,
                    "fixture layout is not what this cell claims to measure"
                );
                // Close, not drop: lock release must not race the next open.
                block_on(s.close()).unwrap();
            }
        }
        let mut cell = Cell::default();
        for i in 0..12 {
            let key = vkey(16, (i as u64 ^ perturb) as usize % nvals);
            let flip = (i as u64 + perturb) % 2 == 0;
            for arm_on in if flip { [false, true] } else { [true, false] } {
                let store = open_wal_retry(&wpath, &opts);
                let map = BTreeMap::over(store);
                map.set_chain_prefetch(arm_on);
                let t = Instant::now();
                let v = block_on(map.get(key.clone())).unwrap().unwrap();
                let ns = t.elapsed().as_nanos() as f64;
                std::hint::black_box(&v);
                let s = map.chain_prefetch_stats();
                if arm_on {
                    // The same exact per-sample contract as the Direct cells.
                    assert_eq!(s.engaged_chains, 1);
                    assert_eq!(s.consumed, 15);
                    assert_eq!((s.unused, s.redundant, s.fallbacks), (0, 0, 0));
                    cell.grouped_ns.push(ns);
                    cell.grouped_loads
                        .push((map.cache_metrics().loads_issued + s.issued) as f64);
                } else {
                    assert_eq!(s.engaged_chains, 0);
                    cell.naive_ns.push(ns);
                    cell.naive_loads
                        .push(map.cache_metrics().loads_issued as f64);
                }
                block_on(map.close()).unwrap();
            }
        }
        print_cell(label, 16, &cell);
    };
    println!("[wal layouts] (overlay member is always buffered)");
    println!(
        "{:>18} {:>6} {:>10} {:>10} {:>10} {:>10} {:>22} {:>16} {:>16} {:>8}",
        "cell",
        "links",
        "naive ms",
        "grpd ms",
        "naive k/s",
        "grpd k/s",
        "ratio min/med/max",
        "loads-n lo/md/hi",
        "loads-g lo/md/hi",
        "distinct"
    );
    wal_cell("wal-overlay buf", false, true);
    {
        // Checkpoint: relocate the live set into the data member.
        let store = open_wal_retry(&wpath, &Options::default);
        block_on(store.checkpoint()).unwrap();
        block_on(store.close()).unwrap();
    }
    wal_cell("wal-ckpt buf", false, false);
    wal_cell("wal-ckpt O_DIRECT", true, false);

    // ---- Issue-lead trace on a single-value
    // Direct store: per chain position p, lead = predecessor-completion −
    // issue; a positive lead ≥ one median device service time means the read
    // was issued ≥ 1 device-latency before demand could have known its address.
    let tpath = std::path::PathBuf::from(&dir).join(format!("t22-trace-{}", std::process::id()));
    {
        let store = Store::<Direct>::create_path(&tpath, Options::default()).unwrap();
        let map = BTreeMap::over(store);
        block_on(map.apply(WriteBatch::new().insert(b"t".to_vec(), chain_value(64, 7)))).unwrap();
        block_on(map.close()).unwrap();
    }
    println!("[issue-lead trace] 64-link chain, per-position lead vs device service time");
    for (mode, direct) in [("buffered(OS-hot)", false), ("O_DIRECT(pc-bypassed)", true)] {
        let inner = match SyscallIo::open_existing_direct(&tpath, direct) {
            Ok(d) => Arc::new(d) as Arc<dyn PageIo>,
            Err(e) => {
                println!("  {mode}: unavailable ({e})");
                continue;
            }
        };
        let traced = Arc::new(TraceIo {
            inner,
            log: std::sync::Mutex::new(Vec::new()),
        });
        for arm_on in [false, true] {
            let store = Store::<Direct>::open_over(
                Arc::clone(&traced) as Arc<dyn PageIo>,
                Options {
                    verify_tree_on_open: false,
                    ..Options::default()
                },
            )
            .unwrap();
            let map = BTreeMap::over(store);
            map.set_chain_prefetch(arm_on);
            traced.take(); // discard open-time reads
            let t0 = Instant::now();
            std::hint::black_box(block_on(map.get(b"t".to_vec())).unwrap().unwrap());
            let mut log = traced.take();
            block_on(map.close()).unwrap();
            // The single-leaf tree reads: leaf (highest offset) + 64 chain
            // pages at 64 consecutive offsets below it. Chain positions are
            // ascending offsets.
            log.sort_by_key(|(off, _, _)| *off);
            let leaf = log.pop().expect("trace has reads");
            std::hint::black_box(&leaf);
            assert_eq!(log.len(), 64, "expected exactly the chain's reads");
            for w in log.windows(2) {
                assert_eq!(w[1].0 - w[0].0, 16384, "chain offsets must be contiguous");
            }
            let service: Vec<f64> = log
                .iter()
                .map(|(_, i, d)| d.duration_since(*i).as_nanos() as f64)
                .collect();
            let service_med = median(service.clone());
            // lead(p) = completion(p−1) − issue(p): positive ⇔ issued early.
            let leads: Vec<f64> = log
                .windows(2)
                .map(|w| {
                    let ready = w[0].2;
                    let issue = w[1].1;
                    if issue <= ready {
                        ready.duration_since(issue).as_nanos() as f64
                    } else {
                        -(issue.duration_since(ready).as_nanos() as f64)
                    }
                })
                .collect();
            let early = leads.iter().filter(|l| **l >= service_med).count();
            let (l_min, l_max) = leads
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), l| {
                    (lo.min(*l), hi.max(*l))
                });
            println!(
                "  {mode} prefetch={}: service med {:.0}µs; lead min/med/max {:.0}/{:.0}/{:.0}µs; \
                 {}/63 positions ≥1 service early; total {:.2}ms",
                if arm_on { "on " } else { "off" },
                service_med / 1e3,
                l_min / 1e3,
                median(leads.clone()) / 1e3,
                l_max / 1e3,
                early,
                t0.elapsed().as_nanos() as f64 / 1e6
            );
        }
    }

    for p in [&dpath, &wpath, &tpath] {
        let _ = std::fs::remove_file(p);
        // WAL segments / sidecars share the stem.
        if let Some(parent) = p.parent() {
            if let Ok(rd) = std::fs::read_dir(parent) {
                let stem = p.file_name().unwrap().to_string_lossy().to_string();
                for e in rd.flatten() {
                    if e.file_name().to_string_lossy().starts_with(&stem) {
                        let _ = std::fs::remove_file(e.path());
                    }
                }
            }
        }
    }
    println!();
}

/// One measured cell. Per-sample latencies AND physical loads are kept **per
/// arm** — with default-capacity caches the two traversals can evict and
/// re-load differently (a real route property), so equal-I/O is measured,
/// never assumed. `distinct_median` is the
/// median distinct-page count of the frozen traces, measured by an untimed
/// ample-cache pass (independent of either timed arm).
#[derive(Default)]
struct Cell {
    naive_ns: Vec<f64>,
    grouped_ns: Vec<f64>,
    naive_loads: Vec<f64>,
    grouped_loads: Vec<f64>,
    /// Per-trace distinct-page counts from the untimed ample-cache pass.
    distinct: Vec<f64>,
}

fn paired_cold_cell(
    sets: &[Vec<Vec<u8>>],
    fresh_reader: &dyn Fn() -> TreeReader,
    ample_reader: &dyn Fn() -> TreeReader,
) -> Cell {
    let mut cell = Cell::default();
    for (i, s) in sets.iter().enumerate() {
        let run_naive = |cell: &mut Cell| {
            let r = fresh_reader();
            let t = Instant::now();
            let out = block_on(r.get_many(s.clone())).unwrap();
            cell.naive_ns.push(t.elapsed().as_nanos() as f64);
            cell.naive_loads.push(r.metrics().loads_issued as f64);
            out
        };
        let run_grouped = |cell: &mut Cell| {
            let r = fresh_reader();
            let t = Instant::now();
            let out = block_on(r.get_many_grouped(s.clone())).unwrap();
            cell.grouped_ns.push(t.elapsed().as_nanos() as f64);
            cell.grouped_loads.push(r.metrics().loads_issued as f64);
            out
        };
        // Counterbalanced arm order.
        let (a, b) = if i % 2 == 0 {
            (run_naive(&mut cell), run_grouped(&mut cell))
        } else {
            let g = run_grouped(&mut cell);
            let n = run_naive(&mut cell);
            (n, g)
        };
        assert_eq!(a, b, "arms disagree on identical frozen batch");
        std::hint::black_box((a, b));
        // Untimed: this trace's true distinct-page count (ample cache → one
        // load per distinct page, no eviction re-loads).
        let v = ample_reader();
        std::hint::black_box(block_on(v.get_many(s.clone())).unwrap());
        cell.distinct.push(v.metrics().loads_issued as f64);
    }
    cell
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn print_cell(name: &str, batch: usize, cell: &Cell) {
    let n_ms = median(cell.naive_ns.clone()) / 1e6;
    let g_ms = median(cell.grouped_ns.clone()) / 1e6;
    let n_ks = batch as f64 / (n_ms / 1e3);
    let g_ks = batch as f64 / (g_ms / 1e3);
    // Paired per-round ratios (same index = same frozen set), not a ratio of
    // marginal medians.
    let ratios: Vec<f64> = cell
        .naive_ns
        .iter()
        .zip(&cell.grouped_ns)
        .map(|(n, g)| n / g)
        .collect();
    let (r_min, r_max) = ratios.iter().fold((f64::INFINITY, 0f64), |(lo, hi), r| {
        (lo.min(*r), hi.max(*r))
    });
    let r_med = median(ratios);
    let range = |v: &Vec<f64>| -> String {
        if v.is_empty() {
            return "0".into();
        }
        let (lo, hi) = v.iter().fold((f64::INFINITY, 0f64), |(lo, hi), x| {
            (lo.min(*x), hi.max(*x))
        });
        format!("{:.0}/{:.0}/{:.0}", lo, median(v.clone()), hi)
    };
    let distinct_med = if cell.distinct.is_empty() {
        0
    } else {
        median(cell.distinct.clone()) as u64
    };
    println!(
        "{name:>18} {batch:>6} {n_ms:>10.3} {g_ms:>10.3} {n_ks:>10.0} {g_ks:>10.0} \
         {:>6.2}/{:>6.2}/{:>6.2} {:>16} {:>16} {:>8}",
        r_min,
        r_med,
        r_max,
        range(&cell.naive_loads),
        range(&cell.grouped_loads),
        distinct_med
    );
}

/// Filesystem kind of `path` via statfs, with the magics that matter for the
/// cold claim (tmpfs cannot honor O_DIRECT, so no page-cache-bypass cell can
/// run there).
fn fs_kind(path: &str) -> String {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(std::path::Path::new(path).as_os_str().as_bytes()).unwrap();
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statfs(c.as_ptr(), &mut st) } != 0 {
        return "statfs failed".into();
    }
    match st.f_type as i64 {
        0x9123683E => "btrfs".into(),
        0xEF53 => "ext4".into(),
        0x58465342 => "xfs".into(),
        0x01021994 => "tmpfs (WARNING: O_DIRECT/page-cache-bypass unavailable)".into(),
        t => format!("fs magic {t:#x}"),
    }
}

/// Runs `f` over each key set, returns keys/second.
fn time_keys(
    sets: &[Vec<Vec<u8>>],
    mut f: impl FnMut(Vec<Vec<u8>>) -> Vec<Option<Vec<u8>>>,
) -> f64 {
    let total_keys: usize = sets.iter().map(|s| s.len()).sum();
    let start = Instant::now();
    for s in sets {
        std::hint::black_box(f(s.clone()));
    }
    total_keys as f64 / start.elapsed().as_secs_f64()
}

// --- writes-visible latency ------------------------------------------------

/// A Direct map with a large dirty budget so a bulk batch clears the invariant-16
/// half-budget cap and no flush pressure intrudes on the timing.
fn roomy_direct() -> BTreeMap<Direct> {
    let store = Store::<Direct>::create_with_io(
        Arc::new(FakeIo::new(0)),
        Options {
            limits: mapdb_uring::admission::AdmissionLimits {
                max_dirty_pages: 4_000_000,
                max_ops_per_batch: 100_000,
                ..Default::default()
            },
            ..Options::default()
        },
    )
    .unwrap();
    BTreeMap::over(store)
}

fn writes_visible() {
    println!("== writes-visible latency (Direct, publish round trip) ==");

    // Healthy path: one bulk batch — the CoW builder packs all leaves once.
    let map = roomy_direct();
    let bulk = 5_000u32;
    let mut b = WriteBatch::new();
    for i in 0..bulk {
        b = b.insert(format!("w{i:08}").into_bytes(), vec![0u8; VAL_LEN]);
    }
    let start = Instant::now();
    block_on(async { map.apply(b).await.unwrap() });
    let per = start.elapsed().as_nanos() as f64 / bulk as f64;
    println!(
        "bulk {bulk}-op batch:        {per:>10.0} ns/op ({:.0} ops/s)",
        1e9 / per
    );

    // Batched path: 100-op batches — realistic sustained ingest.
    let map = roomy_direct();
    let batches = 50u32;
    let start = Instant::now();
    block_on(async {
        let mut n = 0u32;
        for _ in 0..batches {
            let mut b = WriteBatch::new();
            for _ in 0..100 {
                b = b.insert(format!("w{n:08}").into_bytes(), vec![0u8; VAL_LEN]);
                n += 1;
            }
            map.apply(b).await.unwrap();
        }
    });
    let total = (batches * 100) as f64;
    let per = start.elapsed().as_nanos() as f64 / total;
    println!(
        "50 x 100-op batches:      {per:>10.0} ns/op ({:.0} ops/s)",
        1e9 / per
    );

    // Single-op **endpoint** insert latency vs tree size, ascending (append) vs
    // interior order. This measures the cost of one insert *at* a given tree size
    // (a window of inserts on a pre-built tree), NOT the cumulative average of
    // building up to it.
    //
    // History: the earlier single-op table
    // reported "superlinear O(n^2)" latency with a "live pages/key" column. Both
    // were artifacts: the latency was `FakeIo::datasync` cloning the whole growing
    // WAL image every idle-flush sync (fixed — datasync now folds only the delta),
    // and "live pages/key" read `overlay_pages` (cumulative WAL-history locator
    // cardinality), not tree density. Per-op builder cost is actually bounded —
    // O(height x fanout x PAGE_SIZE) — so endpoint latency is roughly flat, with a
    // step when the tree gains height. Interior inserts fragment the touched leaf
    // (the greedy `[full][tiny]` split), dirtying more pages per op than an
    // ascending append, so interior endpoint latency sits above ascending but is
    // still bounded, not superlinear. Random/interior *space* amplification is the
    // real defect, tracked by the `builder_ascending_dense_random_fragmented` test.
    const WINDOW: u32 = 200; // inserts timed starting at each pre-built size
                             // Each row is the MEAN ns/op over WINDOW inserts starting at the listed size
                             // (so the tree grows from n to n+WINDOW during the row, and the interior arm
                             // fragments as it goes) — not the latency of a single isolated op at exactly
                             // n. That is enough to show the cost is flat/bounded in n (refuting the old
                             // O(n^2) curve); it deliberately does not claim point-latency isolation.
                             // The absolute ascending↔interior gap is the fragmenting
                             // split's extra per-op dirty pages.
    println!("single-op mean latency over a {WINDOW}-insert window starting at each size:");
    println!(
        "{:>10} {:>16} {:>16}",
        "start size", "ascending", "interior"
    );
    for &n in &[250u32, 500, 1000, 1500, 3000] {
        // Ascending endpoint: pre-build k{0..n}, then time appends k{n..n+WINDOW}.
        let asc = roomy_direct();
        block_on(async {
            for i in 0..n {
                asc.apply(
                    WriteBatch::new().insert(format!("k{i:08}").into_bytes(), vec![0u8; VAL_LEN]),
                )
                .await
                .unwrap();
            }
        });
        let start = Instant::now();
        block_on(async {
            for i in n..n + WINDOW {
                asc.apply(
                    WriteBatch::new().insert(format!("k{i:08}").into_bytes(), vec![0u8; VAL_LEN]),
                )
                .await
                .unwrap();
            }
        });
        let asc_ns = start.elapsed().as_nanos() as f64 / WINDOW as f64;

        // Interior endpoint: pre-build the same k{0..n}, then time WINDOW inserts
        // whose keys sort *between* existing keys (`k{i}x{s}`, scattered i), so each
        // routes into a full interior leaf and triggers the fragmenting split.
        let int = roomy_direct();
        block_on(async {
            for i in 0..n {
                int.apply(
                    WriteBatch::new().insert(format!("k{i:08}").into_bytes(), vec![0u8; VAL_LEN]),
                )
                .await
                .unwrap();
            }
        });
        let start = Instant::now();
        block_on(async {
            for s in 0..WINDOW {
                let i = ((s as u64).wrapping_mul(2_654_435_761) % n as u64) as u32;
                int.apply(
                    WriteBatch::new()
                        .insert(format!("k{i:08}x{s:04}").into_bytes(), vec![0u8; VAL_LEN]),
                )
                .await
                .unwrap();
            }
        });
        let int_ns = start.elapsed().as_nanos() as f64 / WINDOW as f64;
        println!("{n:>10} {asc_ns:>16.0} {int_ns:>16.0}");
    }
    println!();
}

// --- Direct flush ----------------------------------------------------------

fn direct_flush() {
    println!("== Direct flush latency (100-op batch + flush(version)) ==");
    let map = roomy_direct();
    let rounds = 30u32;
    let start = Instant::now();
    block_on(async {
        let mut n = 0u32;
        for _ in 0..rounds {
            let mut b = WriteBatch::new();
            for _ in 0..100 {
                b = b.insert(format!("f{n:08}").into_bytes(), vec![0u8; VAL_LEN]);
                n += 1;
            }
            let out = map.apply(b).await.unwrap();
            if let ApplyOutcome::Applied { version, .. } = out {
                map.flush(version).await.unwrap();
            }
        }
    });
    let elapsed = start.elapsed();
    println!(
        "{:>10.0} flushes/s   {:>10.0} us/flush (100 ops each)\n",
        rounds as f64 / elapsed.as_secs_f64(),
        elapsed.as_micros() as f64 / rounds as f64
    );
}

// --- WAL group commit ------------------------------------------------------

fn wal_group_commit() {
    println!("== WAL group commit (concurrent apply+commit; fewer fdatasyncs than commits) ==");
    let wal_dev = Arc::new(FakeIo::new(0));
    let store = Store::<Wal>::create_with_io(
        Arc::new(FakeIo::new(0)) as Arc<dyn PageIo>,
        Arc::clone(&wal_dev) as Arc<dyn PageIo>,
        Options::default(),
    )
    .unwrap();
    let map = BTreeMap::over(store);

    // Baseline: syncs already spent on create (the WAL header create sync). Only
    // syncs *past* this baseline are attributable to the committers, so subtract
    // it before computing the commits/sync ratio.
    let base_syncs = wal_dev.sync_count();

    // Concurrent committers, each committing small (8-op) batches. Bounded total
    // so the O(n) single-batch build cost (above) doesn't dominate; the point
    // here is the sync-amortization ratio under contention.
    let threads = 8usize;
    let per = 150usize;
    let ops_per = 8u32;
    let start = Instant::now();
    let mut handles = Vec::new();
    for t in 0..threads {
        let map = map.clone();
        handles.push(std::thread::spawn(move || {
            block_on(async {
                for i in 0..per {
                    let mut b = WriteBatch::new();
                    for j in 0..ops_per {
                        b = b.insert(
                            format!("c{t:02}{i:06}{j:02}").into_bytes(),
                            vec![0u8; VAL_LEN],
                        );
                    }
                    let out = map.apply(b).await.unwrap();
                    if let ApplyOutcome::Applied { version, .. } = out {
                        map.commit(version).await.unwrap();
                    }
                }
            });
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let elapsed = start.elapsed();
    let commits = (threads * per) as u64;
    let syncs = wal_dev.sync_count() - base_syncs; // committer syncs only
                                                   // Group commit never issues *more* syncs than commits — that invariant always
                                                   // holds and is asserted. Actual coalescing (syncs < commits) is opportunistic:
                                                   // it needs commits to overlap in time, which a fully-serializing scheduler
                                                   // could deny, so it is reported, not a hard failure. In
                                                   // practice concurrent committers coalesce heavily (~2 commits/sync here).
    assert!(syncs > 0, "committers must have synced");
    assert!(
        syncs <= commits,
        "group commit must never add syncs: {syncs} syncs for {commits} commits"
    );
    if syncs == commits {
        println!("(note: no coalescing this run — committers did not overlap)");
    }
    println!(
        "{commits} commits ({threads} threads), {syncs} committer fdatasyncs \
         ({:.2} commits/sync), {:.0} commits/s\n",
        commits as f64 / syncs.max(1) as f64,
        commits as f64 / elapsed.as_secs_f64(),
    );
}

// --- cache hit vs miss -----------------------------------------------------

fn cache_hit_vs_miss() {
    // Honest labelling: the "cache-fill" pass is NOT a controlled
    // all-miss population. A ~50k-entry tree is only a few hundred pages, so over
    // `sample` reads the first pass faults each page once and then hits it — it is
    // hit-dominated, and its average is close to the warm pass. The engine exposes
    // no hit/miss counters to report the exact ratio. The real miss cost (device
    // latency) shows against slow media, which the io_boundary bench measures.
    println!("== cache: first-pass fill (miss on first touch of each page) vs warm (all hits) ==");
    let es = entries();
    let dev: Arc<dyn PageIo> = Arc::new(FakeIo::new(0));
    let keys: Vec<Vec<u8>> = es.iter().map(|(k, _)| k.clone()).collect();
    let sample = 20_000usize;
    let mut rng = Rng(0x1234_5678);
    let trace: Vec<Vec<u8>> = (0..sample)
        .map(|_| keys[rng.below(keys.len())].clone())
        .collect();

    let reader = TreeReader::with_cache(
        builder().build(Arc::clone(&dev), &es).expect("build"),
        Arc::new(PageCache::new()),
    );
    let start = Instant::now();
    block_on(async {
        for k in &trace {
            std::hint::black_box(reader.get(k.clone()).await.unwrap());
        }
    });
    let fill_ns = start.elapsed().as_nanos() as f64 / sample as f64;

    let start = Instant::now();
    block_on(async {
        for k in &trace {
            std::hint::black_box(reader.get(k.clone()).await.unwrap());
        }
    });
    let warm_ns = start.elapsed().as_nanos() as f64 / sample as f64;
    println!("first-pass fill {fill_ns:>8.0} ns/read   warm (all hits) {warm_ns:>8.0} ns/read\n");
}

// --- cold vs hot file ------------------------------------------------------

fn cold_vs_hot_file() {
    // Honest labelling: `TreeBuilder::build` writes+syncs every page
    // just before the "cold" pass, so those pages are hot in the OS page cache.
    // The first pass therefore measures cold *engine* cache over an OS-page-cache-
    // hot file (one pread syscall + decode), not cold device I/O. Dropping the OS
    // cache needs privileges; a true cold-device number is the io_boundary bench's
    // job. This axis isolates the engine cache layer over a cheap syscall.
    println!("== file: OS-cache-hot first touch (engine-cache-cold) vs engine-cache-hot ==");
    let dir = std::env::temp_dir();
    let path = dir.join(format!("mapdb-uring-bench-{}.data", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let es = entries();
    let dev: Arc<dyn PageIo> = Arc::new(SyscallIo::open(&path).expect("open temp file"));
    let _built = builder()
        .build(Arc::clone(&dev), &es)
        .expect("build on file");
    block_on(async { dev.datasync().await.unwrap() });

    let keys: Vec<Vec<u8>> = es.iter().map(|(k, _)| k.clone()).collect();
    let sample = 20_000usize;
    let mut rng = Rng(0xABCD_1234);
    let trace: Vec<Vec<u8>> = (0..sample)
        .map(|_| keys[rng.below(keys.len())].clone())
        .collect();

    // A fresh reader+cache over the same on-disk file: first pass faults pages in
    // from the device, second pass is served from the cache.
    let reader = TreeReader::with_cache(
        builder().build(Arc::clone(&dev), &es).expect("build"),
        Arc::new(PageCache::new()),
    );
    let start = Instant::now();
    block_on(async {
        for k in &trace {
            std::hint::black_box(reader.get(k.clone()).await.unwrap());
        }
    });
    let cold_ns = start.elapsed().as_nanos() as f64 / sample as f64;
    let start = Instant::now();
    block_on(async {
        for k in &trace {
            std::hint::black_box(reader.get(k.clone()).await.unwrap());
        }
    });
    let hot_ns = start.elapsed().as_nanos() as f64 / sample as f64;
    println!("engine-cold(OS-hot) {cold_ns:>8.0} ns/read   engine-hot {hot_ns:>8.0} ns/read");
    let _ = std::fs::remove_file(&path);
}
