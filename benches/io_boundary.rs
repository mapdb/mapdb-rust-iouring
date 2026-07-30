//! M1 cold-file I/O-boundary benchmark (exit-threshold evidence).
//!
//! Compares three implementations of the same `PageIo` read contract at a range
//! of queue depths on a cold file:
//!
//! 1. persistent ring — one `CompioIo` (Compio proactor, dedicated thread);
//! 2. per-operation ring — a fresh proactor built and torn down per read;
//! 3. syscall fallback — `SyscallIo` positional `pread` on a worker pool.
//!
//! The file is `posix_fadvise(DONTNEED)`'d before each measurement to force
//! reads to the device rather than the page cache. Run with:
//!
//! ```sh
//! cargo bench --features io-uring
//! ```
//!
//! This is a `harness = false` binary (no criterion dependency): it prints a
//! reads/sec table.

use mapdb_uring::io::{uring_bench, CompioIo, PageIo, SyscallIo};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const PAGE: usize = 4096;
const FILE_PAGES: u64 = 16_384; // 64 MiB
const TOTAL_READS: usize = 8_192;
const QUEUE_DEPTHS: &[usize] = &[1, 8, 32, 64, 128];
/// Timed repetitions per cell (plus one discarded warmup); report the median to
/// tame single-run scheduler/allocator noise.
const REPS: usize = 5;

/// One discarded warmup, then `REPS` timed runs; return the median reads/sec.
fn measure(mut run: impl FnMut() -> std::time::Duration) -> f64 {
    let _ = run();
    let mut samples: Vec<f64> = (0..REPS).map(|_| reads_per_sec(run())).collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    // The file MUST live on a real block device: on tmpfs (`/tmp` here) reads
    // are pure memcpy and `fadvise(DONTNEED)` is a no-op, so io_uring has
    // nothing to offload. Default to the crate directory (real disk); override
    // with MAPDB_BENCH_DIR.
    let dir = std::env::var("MAPDB_BENCH_DIR").unwrap_or_else(|_| ".".to_string());
    let tmp = tempfile::Builder::new()
        .prefix("mapdb-io-bench")
        .tempfile_in(&dir)
        .expect("temp file on real disk");
    let path = tmp.path().to_path_buf();
    println!("bench file: {}", path.display());
    seed_file(&path);

    println!(
        "cold-file random 4 KiB reads — {TOTAL_READS} reads/run, {FILE_PAGES} pages, \
         median of {REPS} (+1 warmup)\n"
    );
    print!("{:<18}", "strategy \\ qd");
    for &qd in QUEUE_DEPTHS {
        print!(" {qd:>8}");
    }
    println!();

    // persistent ring — rebuild per rep so setup cost is amortized identically.
    print!("{:<18}", "persistent-ring");
    for &qd in QUEUE_DEPTHS {
        let rps = measure(|| {
            drop_cache(&path);
            let dev = Arc::new(CompioIo::open(&path).expect("open ring"));
            let elapsed = run_pageio(Arc::clone(&dev) as Arc<dyn PageIo>, qd);
            futures_lite::future::block_on(dev.shutdown()).expect("shutdown");
            elapsed
        });
        print!(" {rps:>8.0}");
    }
    println!();

    // per-operation ring — single attempt; if it exhausts ring resources it is
    // disqualified regardless of speed.
    print!("{:<18}", "per-op-ring");
    for &qd in QUEUE_DEPTHS {
        drop_cache(&path);
        let file = Arc::new(File::open(&path).expect("open file"));
        match run_per_op(file, qd) {
            Ok(elapsed) => print!(" {:>8.0}", reads_per_sec(elapsed)),
            Err(_) => print!(" {:>8}", "ENOMEM"),
        }
    }
    println!();

    // Shipped configuration: `SyscallIo::open` clamps the pool to <= 8. This is
    // the row the deferral decision must be measured against.
    print!("{:<18}", "syscall(shipped)");
    for &qd in QUEUE_DEPTHS {
        let rps = measure(|| {
            drop_cache(&path);
            let dev = Arc::new(SyscallIo::open(&path).expect("open syscall"));
            let elapsed = run_pageio(Arc::clone(&dev) as Arc<dyn PageIo>, qd);
            futures_lite::future::block_on(dev.shutdown()).expect("shutdown");
            elapsed
        });
        print!(" {rps:>8.0}");
    }
    println!();

    // Hypothetical: a pool sized to the queue depth, to show syscall's ceiling
    // if MapDB scaled the pool with load (it does not, today).
    print!("{:<18}", "syscall(pool=qd)");
    for &qd in QUEUE_DEPTHS {
        let rps = measure(|| {
            drop_cache(&path);
            let dev = Arc::new(SyscallIo::with_threads(&path, qd).expect("open syscall"));
            let elapsed = run_pageio(Arc::clone(&dev) as Arc<dyn PageIo>, qd);
            futures_lite::future::block_on(dev.shutdown()).expect("shutdown");
            elapsed
        });
        print!(" {rps:>8.0}");
    }
    println!();

    println!("\n(values are median reads/sec; higher is better)");
}

/// Write `FILE_PAGES` pages and fdatasync so the file is fully allocated.
fn seed_file(path: &Path) {
    let dev = SyscallIo::with_threads(path, 4).expect("seed device");
    futures_lite::future::block_on(async {
        for i in 0..FILE_PAGES {
            let page = vec![(i as u8).wrapping_mul(7).wrapping_add(1); PAGE].into_boxed_slice();
            dev.write_all_at(i * PAGE as u64, page)
                .await
                .result
                .unwrap();
        }
        dev.datasync().await.unwrap();
        dev.shutdown().await.unwrap();
    });
}

/// Approximate a cold file: advise the kernel to drop this file's cached pages.
fn drop_cache(path: &Path) {
    let file = File::open(path).expect("open for fadvise");
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

/// `qd` worker threads, each issuing `TOTAL_READS / qd` random reads via the
/// `PageIo` boundary. Returns wall-clock elapsed.
fn run_pageio(dev: Arc<dyn PageIo>, qd: usize) -> std::time::Duration {
    let per = TOTAL_READS / qd;
    let start = Instant::now();
    let mut handles = Vec::new();
    for t in 0..qd {
        let dev = Arc::clone(&dev);
        handles.push(std::thread::spawn(move || {
            let mut off_gen = OffsetGen::new(t as u64);
            futures_lite::future::block_on(async {
                for _ in 0..per {
                    let off = off_gen.next_offset();
                    let buf = dev.read_exact_at(off, PAGE).await.unwrap();
                    std::hint::black_box(&buf);
                }
            });
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    start.elapsed()
}

/// `qd` worker threads, each building a fresh single-use ring per read. Returns
/// the elapsed time, or the first error (e.g. ring-resource exhaustion).
fn run_per_op(file: Arc<File>, qd: usize) -> std::io::Result<std::time::Duration> {
    let per = TOTAL_READS / qd;
    let start = Instant::now();
    let mut handles = Vec::new();
    for t in 0..qd {
        let file = Arc::clone(&file);
        handles.push(std::thread::spawn(move || -> std::io::Result<()> {
            let mut off_gen = OffsetGen::new(t as u64);
            for _ in 0..per {
                let off = off_gen.next_offset();
                uring_bench::per_op_read(&file, off, PAGE)?;
            }
            Ok(())
        }));
    }
    let mut first_err = None;
    for h in handles {
        if let Err(e) = h.join().unwrap() {
            first_err.get_or_insert(e);
        }
    }
    match first_err {
        Some(e) => Err(e),
        None => Ok(start.elapsed()),
    }
}

fn reads_per_sec(elapsed: std::time::Duration) -> f64 {
    TOTAL_READS as f64 / elapsed.as_secs_f64()
}

/// Deterministic per-thread random page offset generator (LCG).
struct OffsetGen {
    state: u64,
}

impl OffsetGen {
    fn new(seed: u64) -> Self {
        OffsetGen {
            state: 0x9E37_79B9_7F4A_7C15u64.wrapping_add(seed.wrapping_mul(0x100_0000_01B3)),
        }
    }

    fn next_offset(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state % FILE_PAGES) * PAGE as u64
    }
}
