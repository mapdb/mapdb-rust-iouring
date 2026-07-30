//! Adverse-path tests for the `PageIo` device boundary.
//!
//! These run against the *real* devices — `SyscallIo` always, `CompioIo` under
//! the `io-uring` feature — and cover owned-buffer round-trips, zero lengths,
//! short-file reads, caller-cancellation detach, post-shutdown fast-fail,
//! idempotent and concurrent close teardown, queue saturation beyond ring depth,
//! O_DIRECT alignment bounce buffers, and the headline case: concurrent random
//! reads at arbitrary offsets while the same file is sequentially appended and
//! `fdatasync`'d.
//!
//! Conditions that cannot be injected deterministically on a live regular file
//! (EINTR/EAGAIN retry, ring-fatal poison) are covered structurally instead: the
//! retry loop in `SyscallIo`, the panic-to-poison latch in `CompioIo`, and the
//! deterministic `FakeIo` fault knobs.

use crate::io::{PageIo, SyscallIo};
use futures_lite::future::block_on;
use std::sync::Arc;

const PAGE: usize = 4096;

fn page_bytes(tag: u8) -> Box<[u8]> {
    vec![tag; PAGE].into_boxed_slice()
}

/// Shared round-trip: write three pages, datasync, read each back, check size.
fn check_roundtrip(dev: &dyn PageIo) {
    block_on(async {
        for i in 0..3u8 {
            let w = dev
                .write_all_at(i as u64 * PAGE as u64, page_bytes(i + 1))
                .await;
            w.result.expect("write");
            assert_eq!(w.buffer.len(), PAGE, "buffer returned intact");
        }
        dev.datasync().await.expect("datasync");
        for i in 0..3u8 {
            let got = dev
                .read_exact_at(i as u64 * PAGE as u64, PAGE)
                .await
                .expect("read");
            assert!(got.iter().all(|&b| b == i + 1), "page {i} content");
        }
        assert_eq!(dev.size().await.expect("size"), 3 * PAGE as u64);
    });
}

/// Zero-length read and write complete inline and touch nothing.
fn check_zero_length(dev: &dyn PageIo) {
    block_on(async {
        assert!(dev
            .read_exact_at(0, 0)
            .await
            .expect("empty read")
            .is_empty());
        let w = dev.write_all_at(0, Box::default()).await;
        w.result.expect("empty write");
        assert!(w.buffer.is_empty());
    });
}

/// A read that runs past the end of the file is an error, not a partial buffer.
fn check_read_past_eof(dev: &dyn PageIo) {
    block_on(async {
        let w = dev.write_all_at(0, page_bytes(9)).await;
        w.result.expect("seed write");
        dev.datasync().await.expect("sync");
        let err = dev
            .read_exact_at(0, PAGE * 4)
            .await
            .expect_err("read past EOF must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    });
}

/// After `shutdown`, submissions fail fast and a rejected write returns its
/// buffer; `shutdown` is idempotent.
fn check_shutdown_semantics(dev: &dyn PageIo) {
    block_on(async {
        let w = dev.write_all_at(0, page_bytes(1)).await;
        w.result.expect("write before shutdown");
        dev.shutdown().await.expect("shutdown");
        dev.shutdown().await.expect("idempotent shutdown");

        let rejected = dev.write_all_at(0, page_bytes(2)).await;
        assert!(rejected.result.is_err(), "write after shutdown fails");
        assert_eq!(rejected.buffer.len(), PAGE, "buffer returned on rejection");
        assert!(
            dev.read_exact_at(0, PAGE).await.is_err(),
            "read after shutdown fails"
        );
    });
}

// ---------------------------------------------------------------------------
// SyscallIo instantiations (always available).
// ---------------------------------------------------------------------------

fn syscall_dev(threads: usize) -> (SyscallIo, tempfile::NamedTempFile) {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let dev = SyscallIo::with_threads(tmp.path(), threads).unwrap();
    (dev, tmp)
}

#[test]
fn syscall_roundtrip() {
    let (dev, _tmp) = syscall_dev(4);
    check_roundtrip(&dev);
}

#[test]
fn syscall_zero_length() {
    let (dev, _tmp) = syscall_dev(2);
    check_zero_length(&dev);
}

#[test]
fn syscall_read_past_eof() {
    let (dev, _tmp) = syscall_dev(2);
    check_read_past_eof(&dev);
}

#[test]
fn syscall_shutdown_semantics() {
    let (dev, _tmp) = syscall_dev(2);
    check_shutdown_semantics(&dev);
}

#[test]
fn syscall_concurrent_random_reads_during_append_and_sync() {
    let (dev, tmp) = syscall_dev(4);
    concurrent_reads_during_append(Arc::new(dev));
    drop(tmp);
}

// ---------------------------------------------------------------------------
// CompioIo instantiations (io-uring feature).
// ---------------------------------------------------------------------------

#[cfg(feature = "io-uring")]
mod uring {
    use super::*;
    use crate::io::CompioIo;

    fn uring_dev() -> (CompioIo, tempfile::NamedTempFile) {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let dev = CompioIo::open(tmp.path()).unwrap();
        (dev, tmp)
    }

    /// A fixture path under `target/` (real disk — O_DIRECT-capable), never `/tmp`
    /// (tmpfs). The O_DIRECT test probe-and-skips if even this fs refuses it.
    fn real_fs_path(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push("target");
        dir.push("m7a-direct-tests");
        let _ = std::fs::create_dir_all(&dir);
        dir.push(format!("uring-{}-{}-{}.data", tag, std::process::id(), n));
        let _ = std::fs::remove_file(&dir);
        dir
    }

    #[test]
    fn uring_direct_roundtrip_is_byte_correct() {
        // M7-A commit 2: an O_DIRECT ring read/write bounces through an aligned buffer
        // (DioBuf::Aligned) and is byte-identical to a normal read. Uses 16 KiB
        // (DIO_ALIGN) pages; probe-and-skips where O_DIRECT is unavailable.
        const DA: usize = 16 * 1024;
        let path = real_fs_path("direct");
        let dev = match CompioIo::open_direct(&path, true) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: O_DIRECT unavailable here ({e})");
                let _ = std::fs::remove_file(&path);
                return;
            }
        };
        block_on(async {
            for pg in 0u64..3 {
                let buf = vec![(pg as u8) + 1; DA].into_boxed_slice();
                dev.write_all_at(pg * DA as u64, buf).await.result.unwrap();
            }
            dev.datasync().await.unwrap();
            for pg in 0u64..3 {
                let got = dev.read_exact_at(pg * DA as u64, DA).await.unwrap();
                assert!(got.iter().all(|&b| b == (pg as u8) + 1), "page {pg}");
            }
            // Unaligned ops are hard errors, not silent buffered ops,
            // symmetric with the syscall backend: read offset, write offset, write
            // length. The write guard returns the caller's buffer back.
            assert_eq!(
                dev.read_exact_at(100, DA).await.unwrap_err().kind(),
                std::io::ErrorKind::InvalidInput
            );
            let bad_off = dev
                .write_all_at(100, vec![0u8; DA].into_boxed_slice())
                .await;
            assert_eq!(
                bad_off.result.unwrap_err().kind(),
                std::io::ErrorKind::InvalidInput
            );
            assert_eq!(
                bad_off.buffer.len(),
                DA,
                "the buffer is returned on rejection"
            );
            let bad_len = dev.write_all_at(0, vec![0u8; 100].into_boxed_slice()).await;
            assert_eq!(
                bad_len.result.unwrap_err().kind(),
                std::io::ErrorKind::InvalidInput
            );
            dev.shutdown().await.unwrap();
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn uring_roundtrip() {
        let (dev, _tmp) = uring_dev();
        check_roundtrip(&dev);
        block_on(dev.shutdown()).unwrap();
    }

    #[test]
    fn uring_zero_length() {
        let (dev, _tmp) = uring_dev();
        check_zero_length(&dev);
        block_on(dev.shutdown()).unwrap();
    }

    #[test]
    fn uring_read_past_eof() {
        let (dev, _tmp) = uring_dev();
        check_read_past_eof(&dev);
        block_on(dev.shutdown()).unwrap();
    }

    #[test]
    fn uring_shutdown_semantics() {
        let (dev, _tmp) = uring_dev();
        check_shutdown_semantics(&dev);
    }

    /// Dropping many read futures after they were submitted must not wedge or
    /// corrupt the device: the driver still reaps each CQE and frees the buffer
    /// on its own thread (never before the CQE), and the ring keeps serving.
    #[test]
    fn uring_dropped_read_futures_leave_device_healthy() {
        let (dev, _tmp) = uring_dev();
        block_on(async {
            let w = dev.write_all_at(0, page_bytes(7)).await;
            w.result.unwrap();
            dev.datasync().await.unwrap();
        });
        // Each call submits its request synchronously; dropping the future
        // detaches the reply without cancelling the physical load.
        for _ in 0..64 {
            let fut = dev.read_exact_at(0, PAGE);
            drop(fut);
        }
        // Device is still healthy.
        let got = block_on(dev.read_exact_at(0, PAGE)).unwrap();
        assert!(got.iter().all(|&b| b == 7));
        block_on(dev.shutdown()).unwrap();
    }

    /// Saturate well beyond ring depth from many threads at once; every read
    /// must still complete correctly (channel backlog + reserved-slot gating).
    #[test]
    fn uring_queue_saturation() {
        let (dev, _tmp) = uring_dev();
        let dev = Arc::new(dev);
        block_on(async {
            for i in 0..8u8 {
                dev.write_all_at(i as u64 * PAGE as u64, page_bytes(i + 1))
                    .await
                    .result
                    .unwrap();
            }
            dev.datasync().await.unwrap();
        });
        let mut handles = Vec::new();
        for t in 0..400usize {
            let dev = Arc::clone(&dev);
            handles.push(std::thread::spawn(move || {
                let page = (t % 8) as u8;
                let got = block_on(dev.read_exact_at(page as u64 * PAGE as u64, PAGE)).unwrap();
                assert!(got.iter().all(|&b| b == page + 1));
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        block_on(Arc::try_unwrap(dev).ok().unwrap().shutdown()).unwrap();
    }

    #[test]
    fn uring_concurrent_random_reads_during_append_and_sync() {
        let (dev, _tmp) = uring_dev();
        concurrent_reads_during_append(Arc::new(dev));
    }

    /// A fatal driver fault latches poison: `terminal()` reports it and new
    /// submissions fail as poisoned (not as a routine shutdown), with write
    /// buffers returned. Exercises the fatal path deterministically.
    #[test]
    fn uring_fault_poisons_device() {
        let (dev, _tmp) = uring_dev();
        block_on(async {
            dev.write_all_at(0, page_bytes(5)).await.result.unwrap();
            dev.datasync().await.unwrap();
        });
        dev.inject_fault();
        // Wait for the driver to latch poison.
        let mut spins = 0u32;
        while !dev.is_poisoned() {
            std::thread::yield_now();
            spins += 1;
            assert!(spins < 5_000_000, "driver never latched poison");
        }
        let err = block_on(dev.read_exact_at(0, PAGE)).unwrap_err();
        assert!(err.to_string().contains("poison"), "read err: {err}");
        let w = block_on(dev.write_all_at(0, page_bytes(6)));
        assert!(w.result.is_err(), "write after poison fails");
        assert_eq!(w.buffer.len(), PAGE, "poisoned write returns its buffer");
    }

    /// Concurrent `shutdown` callers are safe and idempotent while work is in
    /// flight (the "concurrent close" criterion).
    #[test]
    fn uring_concurrent_close() {
        let (dev, _tmp) = uring_dev();
        let dev = Arc::new(dev);
        block_on(async {
            dev.write_all_at(0, page_bytes(3)).await.result.unwrap();
        });
        let a = {
            let dev = Arc::clone(&dev);
            std::thread::spawn(move || block_on(dev.shutdown()))
        };
        let b = {
            let dev = Arc::clone(&dev);
            std::thread::spawn(move || block_on(dev.shutdown()))
        };
        a.join().unwrap().unwrap();
        b.join().unwrap().unwrap();
    }
}

/// The headline M1 exit test, shared by both devices: a writer thread appends
/// pages sequentially and `fdatasync`s after each, while several reader threads
/// issue random reads at already-durable offsets and verify content. No reader
/// ever observes a torn or wrong page.
fn concurrent_reads_during_append<D: PageIo + 'static>(dev: Arc<D>) {
    const SEEDED: u64 = 32; // pages durable before readers start
    const APPENDED: u64 = 64; // pages the writer adds concurrently
    const READERS: usize = 4;
    const READS_EACH: usize = 400;

    // Seed a durable prefix the readers can target.
    block_on(async {
        for i in 0..SEEDED {
            dev.write_all_at(i * PAGE as u64, page_bytes(tag_for(i)))
                .await
                .result
                .unwrap();
        }
        dev.datasync().await.unwrap();
    });

    let mut handles = Vec::new();
    // Writer: sequential append + datasync per page.
    {
        let dev = Arc::clone(&dev);
        handles.push(std::thread::spawn(move || {
            block_on(async {
                for i in SEEDED..SEEDED + APPENDED {
                    dev.write_all_at(i * PAGE as u64, page_bytes(tag_for(i)))
                        .await
                        .result
                        .unwrap();
                    dev.datasync().await.unwrap();
                }
            });
        }));
    }
    // Readers: random reads within the durable prefix, deterministic LCG.
    for r in 0..READERS {
        let dev = Arc::clone(&dev);
        handles.push(std::thread::spawn(move || {
            let mut state = 0x1234_5678u64.wrapping_add(r as u64 * 0x9E37_79B9);
            block_on(async {
                for _ in 0..READS_EACH {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let page = state % SEEDED;
                    let got = dev.read_exact_at(page * PAGE as u64, PAGE).await.unwrap();
                    assert!(
                        got.iter().all(|&b| b == tag_for(page)),
                        "page {page} read torn/wrong under concurrent append"
                    );
                }
            });
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    block_on(dev.shutdown()).unwrap();
}

fn tag_for(page: u64) -> u8 {
    (page as u8).wrapping_mul(3).wrapping_add(1)
}
