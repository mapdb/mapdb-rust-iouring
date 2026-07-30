//! Positional-syscall [`PageIo`]: the always-available portable fallback.
//!
//! This is M1 option 1 — `pread`/`pwrite`/`fdatasync` on a small pool of
//! dedicated blocking worker threads. It needs no async runtime and no io_uring,
//! and the core (pread/pwrite/fdatasync) compiles everywhere `libc` does — it is
//! the boundary the engine falls back to when the ring is unavailable. It is also
//! the benchmark's "syscall fallback" arm: its ceiling concurrency is the worker
//! count, which is exactly the characteristic the cold-file benchmark measures
//! against a persistent ring.
//!
//! **Linux-only fast read (M7-0).** On Linux, `read_exact_at` first tries one
//! `preadv2(RWF_NOWAIT)` on the caller thread to serve a page-cache hit inline
//! without a worker hop. `preadv2`/`RWF_NOWAIT` are Linux
//! symbols, so the whole fast path is `#[cfg(target_os = "linux")]`; on every other
//! target the module compiles unchanged and every read takes the worker-pool path.
//!
//! Ownership contract (invariant 8, identical to the ring boundary): a
//! submitted buffer is owned by the device until the returned future resolves.
//! Each job carries its owned `Box<[u8]>` onto a worker thread and hands it back
//! through the reply channel; dropping the caller future detaches the reply but
//! the worker still finishes the syscall and drops the buffer safely (a blocking
//! syscall cannot outlive the buffer the way a queued kernel SQE can). Positional
//! `pread`/`pwrite` never touch the shared file offset, so workers run
//! concurrently on one fd without interfering.

use super::aligned::{AlignedBuf, DIO_ALIGN};
use super::{BufResult, IoFuture, IoKind, PageIo};
use async_channel::{Receiver, Sender};
use futures_channel::oneshot;
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Linux caps a single `read`/`write` at `0x7ffff000`; clamp each syscall to it
/// and resubmit the remainder so oversized buffers still transfer in full.
const MAX_RW: usize = 0x7fff_f000;

/// Process-global latch for the `preadv2(RWF_NOWAIT)` read fast path (M7-0):
/// cleared the first time the *syscall itself* is unavailable (`ENOSYS`,
/// kernel < 4.6). Never re-set — an absent syscall does not reappear. Per-device
/// `EOPNOTSUPP` (the filesystem lacks `RWF_NOWAIT`) latches on the device instead,
/// since it is fs-specific, not process-wide.
#[cfg(target_os = "linux")]
static PREADV2_AVAILABLE: AtomicBool = AtomicBool::new(true);

/// A positional-syscall device backed by a fixed pool of worker threads.
pub struct SyscallIo {
    inner: Arc<Inner>,
}

struct Inner {
    file: Arc<File>,
    jobs: Sender<Job>,
    /// Fast-path rejection for submissions after `shutdown` began, so callers do
    /// not race the channel close.
    shut: AtomicBool,
    teardown: Mutex<Teardown>,
    /// This device was opened with `O_DIRECT` (M7-A). Every data-file op then
    /// bounces through a [`DIO_ALIGN`]-aligned buffer on the worker, and the
    /// `preadv2(RWF_NOWAIT)` inline read fast path is disabled: under O_DIRECT there
    /// is no page cache to hit, and an align-1 `preadv2` probe against an O_DIRECT fd
    /// would still serve a cached range inline on filesystems that keep a coherent
    /// cache (btrfs, probed), turning a "cold" benchmark arm into a warm one — the
    /// exact measurement lie `direct_io` exists to prevent.
    direct: bool,
    /// Per-device latch for the `preadv2(RWF_NOWAIT)` fast path: cleared on
    /// `EOPNOTSUPP` (this file's filesystem does not support `RWF_NOWAIT`). After
    /// the first probe trips it, no further probe runs on this device. Unused off
    /// Linux (the fast path is Linux-only).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    nowait_ok: AtomicBool,
    /// Fast-path instrumentation (M7-E harness). Over the device
    /// lifetime: full inline hits; partial inline reads (a cached prefix handed to
    /// the pool to finish); dispatches (a probe that ran but read nothing usable
    /// inline — `EAGAIN`/EOF/error); and fallbacks (a read that skipped the probe
    /// because the fast path was latched off). `hits + partials + dispatches` =
    /// probes actually run; `+ fallbacks` = reads while the fast path was a live
    /// option. The harness reads [`SyscallIo::nowait_active`] to tell a genuinely
    /// inline-serving arm from one silently latched off: a cell whose device
    /// is not `nowait_active` measured the worker path, not the fast path.
    nowait_hits: AtomicU64,
    nowait_partials: AtomicU64,
    nowait_dispatches: AtomicU64,
    nowait_fallbacks: AtomicU64,
}

enum Teardown {
    Running(Vec<JoinHandle<()>>),
    Done,
}

enum Job {
    Read {
        offset: u64,
        len: usize,
        reply: oneshot::Sender<io::Result<Box<[u8]>>>,
    },
    /// A read the `preadv2(RWF_NOWAIT)` fast path started inline: it carries the
    /// buffer already allocated (and, for a partial hit, the `done` bytes already
    /// read) so the worker finishes from `offset + done` without repeating any I/O
    /// (never discard a positive partial).
    ReadResume {
        offset: u64,
        buf: Vec<u8>,
        done: usize,
        reply: oneshot::Sender<io::Result<Box<[u8]>>>,
    },
    Write {
        offset: u64,
        buffer: Box<[u8]>,
        reply: oneshot::Sender<BufResult>,
    },
    Sync {
        reply: oneshot::Sender<io::Result<()>>,
    },
}

impl SyscallIo {
    /// Opens (creating if absent) `path` with a default worker pool sized to the
    /// machine's parallelism (capped), never truncating existing data.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let threads = thread::available_parallelism()
            .map(|n| n.get().clamp(1, 8))
            .unwrap_or(4);
        Self::with_threads(path, threads)
    }

    /// Opens an **existing** file for I/O, never creating it. Used by the
    /// store's `open_path` so a data file deleted after its existence check is
    /// reported as `NotFound`, not silently re-created as a junk zero-length
    /// store.
    pub fn open_existing(path: impl AsRef<Path>) -> io::Result<Self> {
        let threads = thread::available_parallelism()
            .map(|n| n.get().clamp(1, 8))
            .unwrap_or(4);
        Self::open_with(path, threads, false, false)
    }

    /// Opens (creating if absent) `path`, with `O_DIRECT` if `direct` (M7-A). Used
    /// by the device factory for the `Data`/`CompactTemp` roles under `direct_io`.
    pub fn open_direct(path: impl AsRef<Path>, direct: bool) -> io::Result<Self> {
        let threads = thread::available_parallelism()
            .map(|n| n.get().clamp(1, 8))
            .unwrap_or(4);
        Self::open_with(path, threads, true, direct)
    }

    /// Opens an **existing** `path`, never creating it, with `O_DIRECT` if `direct`
    /// (M7-A). The factory's open-path variant of [`open_direct`](Self::open_direct).
    pub fn open_existing_direct(path: impl AsRef<Path>, direct: bool) -> io::Result<Self> {
        let threads = thread::available_parallelism()
            .map(|n| n.get().clamp(1, 8))
            .unwrap_or(4);
        Self::open_with(path, threads, false, direct)
    }

    /// Opens with an explicit worker count, creating the file if absent. The pool
    /// size is the device's concurrency ceiling — the benchmark varies it to
    /// compare against a ring.
    pub fn with_threads(path: impl AsRef<Path>, threads: usize) -> io::Result<Self> {
        Self::open_with(path, threads, true, false)
    }

    fn open_with(
        path: impl AsRef<Path>,
        threads: usize,
        create: bool,
        direct: bool,
    ) -> io::Result<Self> {
        let threads = threads.max(1);
        #[cfg(not(target_os = "linux"))]
        if direct {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "O_DIRECT is only supported on Linux",
            ));
        }
        let mut opts = OpenOptions::new();
        opts.create(create).read(true).write(true).truncate(false);
        #[cfg(target_os = "linux")]
        if direct {
            use std::os::unix::fs::OpenOptionsExt;
            opts.custom_flags(libc::O_DIRECT);
        }
        let file = opts.open(path)?;
        // Refuse (never silently downgrade to buffered) if the filesystem does not
        // honor O_DIRECT at our alignment — a buffered fallback would make a "cold"
        // benchmark arm secretly page-cached. Validated on the
        // already-open fd against `statx(STATX_DIOALIGN)`.
        if direct {
            super::aligned::validate_dio_alignment(&file)?;
        }
        let file = Arc::new(file);
        // The worker's per-op alignment: `Some` bounces every read/write through an
        // aligned buffer for O_DIRECT; `None` is the pre-M7 buffered path.
        let align = if direct { Some(DIO_ALIGN) } else { None };
        // Unbounded: admission above the boundary is the only bound (invariant
        // 15). Workers pull jobs; a closed channel drains then stops them.
        let (jobs, receiver) = async_channel::unbounded::<Job>();
        let handles = (0..threads)
            .map(|i| {
                let file = Arc::clone(&file);
                let receiver = receiver.clone();
                thread::Builder::new()
                    .name(format!("mapdb-syscall-io-{i}"))
                    .spawn(move || run_worker(&file, &receiver, align))
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(SyscallIo {
            inner: Arc::new(Inner {
                file,
                jobs,
                shut: AtomicBool::new(false),
                teardown: Mutex::new(Teardown::Running(handles)),
                direct,
                nowait_ok: AtomicBool::new(true),
                nowait_hits: AtomicU64::new(0),
                nowait_partials: AtomicU64::new(0),
                nowait_dispatches: AtomicU64::new(0),
                nowait_fallbacks: AtomicU64::new(0),
            }),
        })
    }

    /// Fast-path counters `(hits, partials, dispatches, fallbacks)` over the device
    /// lifetime (M7-E harness). `hits + partials + dispatches` = probes run;
    /// `fallbacks` = reads that skipped the probe because it was latched off. Read
    /// alongside [`nowait_active`](Self::nowait_active) to tell an inline-serving
    /// run from a latched-off one.
    pub fn nowait_stats(&self) -> (u64, u64, u64, u64) {
        (
            self.inner.nowait_hits.load(Ordering::Relaxed),
            self.inner.nowait_partials.load(Ordering::Relaxed),
            self.inner.nowait_dispatches.load(Ordering::Relaxed),
            self.inner.nowait_fallbacks.load(Ordering::Relaxed),
        )
    }

    /// Whether the `preadv2(RWF_NOWAIT)` fast path is still a live option for this
    /// device: the syscall exists process-wide and this filesystem has not returned
    /// `EOPNOTSUPP`. The M7-E harness checks this after a run so an arm labeled
    /// "inline-NOWAIT on" whose device silently latched off (e.g. a tmpfs-homed
    /// fixture) is not misread as fast-path data.
    /// Always `false` off Linux (the fast path is Linux-only).
    pub fn nowait_active(&self) -> bool {
        self.fast_read_enabled()
    }

    /// Whether the `preadv2(RWF_NOWAIT)` fast path is live for this device: the
    /// syscall exists process-wide and this filesystem has not returned
    /// `EOPNOTSUPP`.
    #[cfg(target_os = "linux")]
    fn fast_read_enabled(&self) -> bool {
        // Never live under O_DIRECT: there is no page cache for the probe to hit,
        // and an align-1 probe on an O_DIRECT fd would read a still-cached range
        // inline on a coherent-cache filesystem, faking a warm hit in a cold arm.
        !self.inner.direct
            && PREADV2_AVAILABLE.load(Ordering::Relaxed)
            && self.inner.nowait_ok.load(Ordering::Relaxed)
    }

    /// Off Linux the fast path does not exist, so it is never live.
    #[cfg(not(target_os = "linux"))]
    fn fast_read_enabled(&self) -> bool {
        false
    }

    /// The `preadv2(RWF_NOWAIT)` fast-path attempt (Linux only). Returns `Some` if
    /// it served the read inline (a full cache hit) or dispatched a resume job to
    /// finish it; `None` to take the normal worker path (device shut, or the fast
    /// path latched off). Isolated here so `read_exact_at` stays portable.
    #[cfg(target_os = "linux")]
    fn try_fast_read(
        &self,
        offset: u64,
        len: usize,
    ) -> Option<IoFuture<'_, io::Result<Box<[u8]>>>> {
        // Under O_DIRECT the fast path does not exist — take the worker's
        // aligned-bounce path with no probe and no fallback accounting (there is no
        // fast path to fall back from).
        if self.inner.direct {
            return None;
        }
        // A read once `shutdown` began must be rejected, not served from the
        // still-open fd — fall through to `submit`, which reproduces `lost()`.
        if self.inner.shut.load(Ordering::Acquire) {
            return None;
        }
        if !self.fast_read_enabled() {
            self.inner.nowait_fallbacks.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut buf = vec![0u8; len];
        match self.probe_nowait(offset, &mut buf) {
            Probe::Hit => {
                self.inner.nowait_hits.fetch_add(1, Ordering::Relaxed);
                Some(Box::pin(std::future::ready(Ok(buf.into_boxed_slice()))))
            }
            Probe::Resume(done) => {
                if done > 0 {
                    self.inner.nowait_partials.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.inner.nowait_dispatches.fetch_add(1, Ordering::Relaxed);
                }
                // Carry the buffer + inline progress to the worker (never discard a
                // positive partial). `submit` drops the buffer if the device is
                // shut / the channel closed, resolving to `lost()`.
                let wait = self.submit(
                    move |reply| Job::ReadResume {
                        offset,
                        buf,
                        done,
                        reply,
                    },
                    || Err(lost()),
                );
                Some(Box::pin(async move {
                    wait.await.unwrap_or_else(|_| Err(lost()))
                }))
            }
        }
    }

    /// Off Linux there is no fast path.
    #[cfg(not(target_os = "linux"))]
    fn try_fast_read(
        &self,
        _offset: u64,
        _len: usize,
    ) -> Option<IoFuture<'_, io::Result<Box<[u8]>>>> {
        None
    }

    /// One `preadv2(RWF_NOWAIT)` attempt on the caller thread, into an
    /// already-allocated `buf`. Returns [`Probe::Hit`] when the whole range was
    /// served from cache inline; otherwise [`Probe::Resume`] with the count already
    /// read (0 on `EAGAIN`/EOF/other), so the caller hands `buf` + `done` to the
    /// worker pool. Latches the fast path off on an unsupported syscall (`ENOSYS`,
    /// global) or filesystem (`EOPNOTSUPP`, this device). Never busy-retries — the
    /// worker's blocking loop, not this probe, owns retryable errnos.
    #[cfg(target_os = "linux")]
    fn probe_nowait(&self, offset: u64, buf: &mut [u8]) -> Probe {
        let fd = self.inner.file.as_raw_fd();
        let iov = libc::iovec {
            iov_base: buf.as_mut_ptr() as *mut libc::c_void,
            iov_len: buf.len(),
        };
        // SAFETY: `iov` points at `buf`'s valid, initialized `buf.len()` bytes for
        // the duration of the call; `fd` is the device's live file descriptor.
        let ret = unsafe { libc::preadv2(fd, &iov, 1, offset as libc::off_t, libc::RWF_NOWAIT) };
        if ret > 0 {
            let n = ret as usize;
            if n >= buf.len() {
                Probe::Hit
            } else {
                // Positive partial: keep the cached prefix, finish the rest.
                Probe::Resume(n)
            }
        } else if ret == 0 {
            // EOF / nothing available inline: let the worker produce the
            // authoritative result (a true short file becomes `UnexpectedEof`).
            Probe::Resume(0)
        } else {
            let err = io::Error::last_os_error();
            match err.raw_os_error() {
                // The syscall itself is absent (kernel < 4.6): disable everywhere.
                Some(libc::ENOSYS) => PREADV2_AVAILABLE.store(false, Ordering::Relaxed),
                // This filesystem lacks RWF_NOWAIT: disable for this device only.
                Some(libc::EOPNOTSUPP) => self.inner.nowait_ok.store(false, Ordering::Relaxed),
                // EAGAIN/EWOULDBLOCK (not cached) and any other error (EINTR, EIO,
                // …): do not reinterpret here — dispatch and let the
                // blocking worker read authoritatively. EINVAL is deliberately NOT
                // treated as "unsupported": it can be a real argument defect (a bug
                // we want to surface deterministically), so we never latch on it.
                // Cost note: kernels 4.6–4.13 have `preadv2` but not `RWF_NOWAIT`
                // and return EINVAL for it, so every read there pays one wasted
                // probe syscall forever. Acceptable — those kernels are long EOL;
                // if such a host ever mattered, a bounded consecutive-EINVAL
                // per-device latch would cap it without swallowing real defects.
                _ => {}
            }
            Probe::Resume(0)
        }
    }

    fn submit<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> Job,
        on_lost: impl FnOnce() -> T,
    ) -> oneshot::Receiver<T> {
        let (reply, wait) = oneshot::channel();
        if self.inner.shut.load(Ordering::Acquire) {
            let _ = reply.send(on_lost());
            return wait;
        }
        // If the channel closed between the flag check and the send, the job
        // (and its reply sender) is returned in the error and dropped; the
        // dropped sender resolves the waiter to a canceled `Err`, which the
        // caller's future maps back to `lost()`. No explicit handling needed.
        let _ = self.inner.jobs.try_send(make(reply));
        wait
    }
}

/// The outcome of a `preadv2(RWF_NOWAIT)` probe on the caller thread.
#[cfg(target_os = "linux")]
enum Probe {
    /// The whole requested range was served from cache inline.
    Hit,
    /// `done` bytes were read inline (0 if nothing usable); the worker finishes
    /// the read from `offset + done` into the same buffer.
    Resume(usize),
}

/// Runs one worker. `align` is `Some(a)` for an O_DIRECT device — every read/write
/// bounces through an `a`-aligned buffer (M7-A); `None` is the buffered path.
fn run_worker(file: &File, jobs: &Receiver<Job>, align: Option<usize>) {
    while let Ok(job) = jobs.recv_blocking() {
        match job {
            Job::Read { offset, len, reply } => {
                let res = match align {
                    Some(a) => pread_direct(file, offset, len, a),
                    None => pread_resume(file, offset, vec![0u8; len], 0),
                };
                let _ = reply.send(res);
            }
            Job::ReadResume {
                offset,
                buf,
                done,
                reply,
            } => {
                // The fast path that produces `ReadResume` is disabled under
                // O_DIRECT, so a direct device never sees this job; a
                // buffered resume finishes on the buffered path.
                let _ = reply.send(pread_resume(file, offset, buf, done));
            }
            Job::Write {
                offset,
                buffer,
                reply,
            } => {
                let result = match align {
                    Some(a) => pwrite_direct(file, offset, &buffer, a),
                    None => pwrite_all(file, offset, &buffer),
                };
                let _ = reply.send(BufResult { result, buffer });
            }
            Job::Sync { reply } => {
                let _ = reply.send(fdatasync(file));
            }
        }
    }
}

/// Drives an aligned O_DIRECT positional transfer to completion (M7-A).
/// `op(from, count)` performs one syscall over the aligned sub-region
/// `[from, from+count)` — `from` and `count` are always multiples of `align` — and
/// returns the byte count (`Ok(0)` = EOF). `capacity` is the aligned buffer length;
/// `need` is how many bytes must transfer to succeed (the logical `len` for a read;
/// `capacity` for a write). `allow_eof` distinguishes a read (a short/zero return is
/// the file's end) from a write (a zero return is `WriteZero`).
///
/// **Forward progress is guaranteed** — the property the naive loop lacked. `done`
/// is invariantly `align`-aligned at the top of each iteration, so each syscall is
/// aligned. A completion is resumed only when it is *full* (`n == count`) or lands
/// on an alignment boundary (`(from+n) % align == 0`): both strictly advance `done`
/// by at least `align`, so the loop cannot spin. A **non-boundary short completion**
/// that is not EOF cannot be resumed with an aligned request, so — rather than
/// reissue a doomed suffix forever (an ENOSPC `pwrite` returns a fs-block-granular
/// partial, not an `align`-granular one) — it is a **precise I/O error** for a write,
/// and for a read it is the file's true end: tolerated if `done >= need` already (an
/// older unpadded manifest read past EOF), else a genuine short read.
fn drive_direct(
    capacity: usize,
    need: usize,
    align: usize,
    allow_eof: bool,
    mut op: impl FnMut(usize, usize) -> io::Result<usize>,
) -> io::Result<usize> {
    let mut done = 0usize;
    while done < need {
        debug_assert_eq!(done % align, 0, "a resume offset must stay alignment-valid");
        let from = done;
        let count = capacity - from;
        let n = match op(from, count) {
            Ok(n) => n,
            Err(ref e) if is_retryable(e) => continue,
            Err(e) => return Err(e),
        };
        if n == 0 {
            if allow_eof {
                break; // hard EOF
            }
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "zero-length O_DIRECT write with bytes outstanding",
            ));
        }
        let new_done = from + n;
        // Full completion, or one that lands on an alignment boundary: `done` stays
        // aligned and strictly advances → resume.
        if n == count || new_done % align == 0 {
            done = new_done;
            continue;
        }
        // Non-boundary short completion: cannot resume with an aligned request.
        done = new_done;
        if allow_eof {
            break; // the file's true end (may still satisfy `need` — checked below)
        }
        return Err(io::Error::other(
            "short O_DIRECT write cannot resume on an alignment boundary",
        ));
    }
    if done < need {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "short read before requested length",
        ));
    }
    Ok(done)
}

/// An O_DIRECT positional read of `len` bytes at `offset` (M7-A). O_DIRECT requires
/// the memory address, offset, and length to be alignment multiples, so the read
/// goes into a [`DIO_ALIGN`]-aligned bounce buffer whose length is `len` rounded up
/// to `align`, then exactly `len` bytes are copied into the natural-aligned
/// `Box<[u8]>` the trait returns. `offset` must be aligned (the data file's page
/// I/O always is); an unaligned offset is rejected as a hard error rather than left
/// to the filesystem — some filesystems (btrfs) silently buffer an unaligned
/// O_DIRECT op instead of failing, which would corrupt a cold benchmark arm.
/// A rounded read that runs past EOF is legal and returns a short
/// count; as long as at least `len` true bytes were read it succeeds (this is what
/// lets an O_DIRECT open read an older, non-page-padded checkpoint manifest whose
/// true length is unaligned).
fn pread_direct(file: &File, offset: u64, len: usize, align: usize) -> io::Result<Box<[u8]>> {
    if offset as usize % align != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unaligned offset for O_DIRECT read",
        ));
    }
    let rounded = len.next_multiple_of(align);
    let mut bounce = AlignedBuf::zeroed(rounded);
    let fd = file.as_raw_fd();
    let base = bounce.as_mut_slice().as_mut_ptr();
    // Drive the aligned read loop. `op` performs one aligned `pread` over
    // `[from, from+count)`; `drive_direct` guarantees forward progress and the
    // `n >= len` past-EOF tolerance. The raw `base` is a copy
    // (no live borrow of `bounce`), so `bounce` is free to be read after the loop.
    drive_direct(
        rounded,
        len,
        align,
        /* allow_eof */ true,
        |from, count| {
            // SAFETY: `base` is `align`-aligned and `from` is a multiple of `align`, so
            // `base + from` is aligned and in-bounds (`from < rounded`); `count` bytes
            // from there stay within the `rounded`-byte allocation.
            let ret = unsafe {
                libc::pread(
                    fd,
                    base.add(from) as *mut libc::c_void,
                    count,
                    (offset + from as u64) as libc::off_t,
                )
            };
            if ret < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(ret as usize)
            }
        },
    )?;
    Ok(bounce.as_slice()[..len].to_vec().into_boxed_slice())
}

/// An O_DIRECT positional write of `buffer` at `offset` (M7-A). The caller's
/// natural-aligned bytes are copied into a [`DIO_ALIGN`]-aligned bounce and written
/// from there. `offset` and `buffer.len()` must both be alignment multiples — the
/// data file's page writes always are, and the checkpoint manifest is padded to a
/// page before it reaches here; an unaligned op is a hard error, not
/// a silent buffered fallback.
fn pwrite_direct(file: &File, offset: u64, buffer: &[u8], align: usize) -> io::Result<()> {
    if offset as usize % align != 0 || buffer.len() % align != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unaligned offset or length for O_DIRECT write",
        ));
    }
    let bounce = AlignedBuf::from_slice(buffer);
    let fd = file.as_raw_fd();
    let base = bounce.as_slice().as_ptr();
    let total = bounce.len();
    drive_direct(
        total,
        total,
        align,
        /* allow_eof */ false,
        |from, count| {
            // SAFETY: as `pread_direct`; `base + from` is aligned and in-bounds and
            // `count` bytes stay within the `total`-byte allocation.
            let ret = unsafe {
                libc::pwrite(
                    fd,
                    base.add(from) as *const libc::c_void,
                    count,
                    (offset + from as u64) as libc::off_t,
                )
            };
            if ret < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(ret as usize)
            }
        },
    )?;
    Ok(())
}

/// Completes a read of `buf.len()` bytes at `offset` into `buf`, starting from
/// byte `done` (0 for a fresh read; the `preadv2(RWF_NOWAIT)` fast path passes the
/// count it already read inline so no I/O is repeated). Blocking; runs on a
/// worker thread.
fn pread_resume(
    file: &File,
    offset: u64,
    mut buf: Vec<u8>,
    mut done: usize,
) -> io::Result<Box<[u8]>> {
    let len = buf.len();
    let fd = file.as_raw_fd();
    while done < len {
        let count = (len - done).min(MAX_RW);
        let ret = unsafe {
            libc::pread(
                fd,
                buf[done..].as_mut_ptr() as *mut libc::c_void,
                count,
                (offset + done as u64) as libc::off_t,
            )
        };
        if ret < 0 {
            let err = io::Error::last_os_error();
            if is_retryable(&err) {
                continue;
            }
            return Err(err);
        }
        if ret == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short read before requested length",
            ));
        }
        done += ret as usize;
    }
    Ok(buf.into_boxed_slice())
}

fn pwrite_all(file: &File, offset: u64, buffer: &[u8]) -> io::Result<()> {
    let fd = file.as_raw_fd();
    let mut done = 0usize;
    while done < buffer.len() {
        let count = (buffer.len() - done).min(MAX_RW);
        let ret = unsafe {
            libc::pwrite(
                fd,
                buffer[done..].as_ptr() as *const libc::c_void,
                count,
                (offset + done as u64) as libc::off_t,
            )
        };
        if ret < 0 {
            let err = io::Error::last_os_error();
            if is_retryable(&err) {
                continue;
            }
            return Err(err);
        }
        if ret == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "zero-length write with bytes outstanding",
            ));
        }
        done += ret as usize;
    }
    Ok(())
}

fn fdatasync(file: &File) -> io::Result<()> {
    let fd = file.as_raw_fd();
    loop {
        let ret = unsafe { libc::fdatasync(fd) };
        if ret == 0 {
            return Ok(());
        }
        let err = io::Error::last_os_error();
        if err.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return Err(err);
    }
}

/// EINTR and EAGAIN are both retried (both are injected in tests): a regular-file
/// positional syscall makes forward progress on retry.
fn is_retryable(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::Interrupted
        || err.raw_os_error() == Some(libc::EAGAIN)
        || err.raw_os_error() == Some(libc::EWOULDBLOCK)
}

fn lost() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "syscall device shut down")
}

impl PageIo for SyscallIo {
    fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
        if len == 0 {
            return Box::pin(std::future::ready(Ok(Box::default())));
        }
        // Fast path (M7-0, Linux only): one `preadv2(RWF_NOWAIT)` on the
        // caller thread serves a page-cache hit inline with no worker hop; anything
        // else hands the (possibly partially filled) buffer to the pool to finish,
        // so no I/O already done inline is repeated. `try_fast_read` returns `None`
        // (device shut, fast path latched off, or non-Linux) to take the normal
        // worker path below — which reproduces the pre-M7 behaviour exactly.
        if let Some(fut) = self.try_fast_read(offset, len) {
            return fut;
        }
        let wait = self.submit(|reply| Job::Read { offset, len, reply }, || Err(lost()));
        Box::pin(async move { wait.await.unwrap_or_else(|_| Err(lost())) })
    }

    fn write_all_at(&self, offset: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult> {
        if buffer.is_empty() {
            return Box::pin(std::future::ready(BufResult {
                result: Ok(()),
                buffer,
            }));
        }
        // Reject after shutdown while still returning the caller's buffer.
        if self.inner.shut.load(Ordering::Acquire) {
            return Box::pin(std::future::ready(BufResult {
                result: Err(lost()),
                buffer,
            }));
        }
        let (reply, wait) = oneshot::channel();
        if let Err(err) = self.inner.jobs.try_send(Job::Write {
            offset,
            buffer,
            reply,
        }) {
            // Closed after the flag check: recover the buffer from the job.
            let Job::Write { buffer, .. } = err.into_inner() else {
                unreachable!("try_send returns the same job variant")
            };
            return Box::pin(std::future::ready(BufResult {
                result: Err(lost()),
                buffer,
            }));
        }
        Box::pin(async move {
            wait.await.unwrap_or_else(|_| BufResult {
                result: Err(lost()),
                buffer: Box::default(),
            })
        })
    }

    fn datasync(&self) -> IoFuture<'_, io::Result<()>> {
        let wait = self.submit(|reply| Job::Sync { reply }, || Err(lost()));
        Box::pin(async move { wait.await.unwrap_or_else(|_| Err(lost())) })
    }

    fn size(&self) -> IoFuture<'_, io::Result<u64>> {
        // fstat on the shared fd; positional writes have already extended the
        // inode by the time their futures resolved, so this is consistent for a
        // caller that awaited them.
        let result = self.inner.file.metadata().map(|m| m.len());
        Box::pin(std::future::ready(result))
    }

    fn shutdown(&self) -> IoFuture<'_, io::Result<()>> {
        // Mark shut first so new submissions fail fast, then drain+join. Closing
        // the sender lets workers finish already-queued jobs and exit.
        self.inner.shut.store(true, Ordering::Release);
        self.inner.jobs.close();
        let mut teardown = self.inner.teardown.lock();
        if let Teardown::Running(handles) = std::mem::replace(&mut *teardown, Teardown::Done) {
            // Shared with Store::close: joining on the polling executor is a
            // momentary block here — workers stop as soon as the drained channel
            // reports closed. A later milestone parks the join on a dedicated
            // owner if any device grows a long teardown.
            for handle in handles {
                let _ = handle.join();
            }
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn kind(&self) -> IoKind {
        IoKind::Syscall
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_lite::future::block_on;
    use std::path::{Path, PathBuf};

    fn tmp_path(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "mapdb-syscall-nowait-{}-{}-{}.data",
            tag,
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    /// A directory on a filesystem that *may* support `RWF_NOWAIT` — the repo's
    /// `target/` (real disk on dev hosts and CI), never `/tmp` (tmpfs on this host,
    /// where `RWF_NOWAIT` is `EOPNOTSUPP`). The fast-path
    /// tests still probe-and-skip if even this fs lacks it, so they never
    /// wrong-reason-pass in a latched-off regime.
    fn real_fs_path(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push("target");
        dir.push("m7-nowait-tests");
        let _ = std::fs::create_dir_all(&dir);
        dir.push(format!("{}-{}-{}.data", tag, std::process::id(), n));
        let _ = std::fs::remove_file(&dir);
        dir
    }

    /// Writes `bytes` to a fresh device at `path`, fsyncs, and reads once to let the
    /// device probe the filesystem's `RWF_NOWAIT` support. Returns the device and
    /// whether the fast path stayed live (i.e. the fs supports `RWF_NOWAIT`).
    fn warm_device(path: &Path, bytes: &[u8]) -> (SyscallIo, bool) {
        let dev = SyscallIo::open(path).unwrap();
        block_on(async {
            dev.write_all_at(0, bytes.to_vec().into_boxed_slice())
                .await
                .result
                .unwrap();
            dev.datasync().await.unwrap();
            // First read probes the fs; if unsupported it latches off here.
            let _ = dev.read_exact_at(0, bytes.len()).await.unwrap();
        });
        let supported = dev.nowait_active();
        (dev, supported)
    }

    /// M7-0 (host-independent): `pread_resume` finishes a read from
    /// `offset + done` into `buf[done..]` and never touches the cached prefix
    /// `buf[..done]`. This is the exact arithmetic the fast path's partial-carry
    /// depends on — pinned without any filesystem `RWF_NOWAIT` dependency, so it
    /// runs identically on tmpfs, btrfs, or a non-Linux target.
    #[test]
    fn pread_resume_preserves_cached_prefix_and_resumes_at_offset() {
        let path = tmp_path("resume");
        let dev = SyscallIo::open(&path).unwrap();
        // On disk: 4096 bytes of 0xBB.
        block_on(async {
            dev.write_all_at(0, vec![0xBBu8; 4096].into_boxed_slice())
                .await
                .result
                .unwrap();
            dev.datasync().await.unwrap();
        });
        let file = std::fs::OpenOptions::new().read(true).open(&path).unwrap();
        // Simulate the fast path having read the first `done` bytes inline: prefill
        // buf[..done] with a *distinct* marker (0xAA) that does NOT match the disk.
        let done = 1000usize;
        let mut buf = vec![0u8; 4096];
        for b in &mut buf[..done] {
            *b = 0xAA;
        }
        let out = pread_resume(&file, 0, buf, done).unwrap();
        // The cached prefix survives verbatim (proves it was neither re-read from
        // disk — where it is 0xBB — nor overwritten).
        assert!(
            out[..done].iter().all(|b| *b == 0xAA),
            "cached prefix must be preserved"
        );
        // The suffix is read from disk at offset+done (0xBB), proving the resume
        // read the right range into buf[done..], not from offset 0.
        assert!(
            out[done..].iter().all(|b| *b == 0xBB),
            "suffix must come from offset+done"
        );
        block_on(dev.shutdown()).unwrap();
        let _ = std::fs::remove_file(&path);
    }

    /// M7-0: on a filesystem that supports `RWF_NOWAIT`, a warm read is a
    /// deterministic **Hit** and a cached cross-EOF read is a deterministic
    /// **positive partial** (the probe reads the whole file, the worker then hits
    /// EOF → `UnexpectedEof`). Probe-and-skip if the fs lacks `RWF_NOWAIT`, so this
    /// never silently passes in the latched-off regime that fooled the first draft.
    #[test]
    fn fast_path_hit_and_partial_are_deterministic_on_supporting_fs() {
        let path = real_fs_path("hitpartial");
        let file_len = 16 * 1024usize;
        let (dev, supported) = warm_device(&path, &vec![9u8; file_len]);
        if !supported {
            eprintln!("skip: {} does not support RWF_NOWAIT", path.display());
            block_on(dev.shutdown()).unwrap();
            let _ = std::fs::remove_file(&path);
            return;
        }
        block_on(async {
            // A cached full-length read is a Hit.
            let (h0, _, _, _) = dev.nowait_stats();
            let r = dev.read_exact_at(0, file_len).await.unwrap();
            assert!(r.iter().all(|b| *b == 9));
            let (h1, p1, _, _) = dev.nowait_stats();
            assert_eq!(h1, h0 + 1, "a cached full read must be an inline Hit");
            // A cached read past EOF: the probe reads the whole cached file (a
            // positive partial), then the worker resumes at EOF and returns
            // UnexpectedEof — exercising Job::ReadResume with done > 0.
            let err = dev.read_exact_at(0, file_len + 16 * 1024).await;
            assert!(err.is_err(), "reading past EOF must fail");
            let (_, p2, _, _) = dev.nowait_stats();
            assert_eq!(
                p2,
                p1 + 1,
                "a cross-EOF cached read must be a positive partial"
            );
            dev.shutdown().await.unwrap();
        });
        let _ = std::fs::remove_file(&path);
    }

    /// M7-0: fast-path reads are byte-identical to the normal path across sizes and
    /// offsets — on any filesystem (a supporting fs exercises the inline path; an
    /// unsupporting one exercises the worker fallback, and correctness must hold
    /// either way). No counter assertions here (those are the deterministic tests
    /// above); this is purely a correctness sweep.
    #[test]
    fn fast_path_reads_are_byte_correct() {
        let path = real_fs_path("correct");
        let dev = SyscallIo::open(&path).unwrap();
        block_on(async {
            for pg in 0u64..4 {
                dev.write_all_at(
                    pg * 16 * 1024,
                    vec![pg as u8 + 1; 16 * 1024].into_boxed_slice(),
                )
                .await
                .result
                .unwrap();
            }
            dev.datasync().await.unwrap();
            for _ in 0..10 {
                for pg in 0u64..4 {
                    let r = dev.read_exact_at(pg * 16 * 1024, 16 * 1024).await.unwrap();
                    assert!(
                        r.iter().all(|b| *b == pg as u8 + 1),
                        "page {pg} bytes wrong"
                    );
                }
                // A sub-page read spanning two pages.
                let r = dev.read_exact_at(16 * 1024 - 8, 16).await.unwrap();
                assert_eq!(&r[..8], &[1u8; 8]);
                assert_eq!(&r[8..], &[2u8; 8]);
            }
            dev.shutdown().await.unwrap();
        });
        let _ = std::fs::remove_file(&path);
    }

    /// M7-0: a read after `shutdown` is rejected, not served inline from the still-
    /// open fd. Run on a supporting fs so the fast path is genuinely live when the
    /// guard is tested — otherwise (tmpfs) the read falls through regardless and the
    /// test would pass even with the guard deleted (the first draft's wrong-reason
    /// pass). Probe-and-skip if the fs lacks `RWF_NOWAIT`.
    #[test]
    fn read_after_shutdown_is_rejected_not_served_inline() {
        let path = real_fs_path("shut");
        let (dev, supported) = warm_device(&path, &vec![7u8; 16 * 1024]);
        if !supported {
            eprintln!(
                "skip: {} does not support RWF_NOWAIT (guard not exercised)",
                path.display()
            );
            block_on(dev.shutdown()).unwrap();
            let _ = std::fs::remove_file(&path);
            return;
        }
        block_on(async {
            // The page is cached and the fast path is live: a normal read Hits.
            assert!(dev.read_exact_at(0, 16 * 1024).await.is_ok());
            dev.shutdown().await.unwrap();
            // After shutdown the read is rejected even though the page is cached and
            // the fd is still open — the guard sends it down the reject path instead
            // of serving it inline (delete the guard and this Ok()s).
            assert!(dev.read_exact_at(0, 16 * 1024).await.is_err());
        });
        let _ = std::fs::remove_file(&path);
    }

    // ---- M7-A: O_DIRECT device path ----

    /// A plain buffered file at `path`, for the host-independent `pread_direct` /
    /// `pwrite_direct` unit tests (the alignment rejection and the bounce/round-up
    /// arithmetic hold on any fd — they do not need a filesystem that honors
    /// O_DIRECT, so these run identically on tmpfs, btrfs, or a non-Linux target).
    fn plain_file(path: &Path) -> File {
        OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(path)
            .unwrap()
    }

    #[test]
    fn direct_helpers_reject_unaligned_ops() {
        // The alignment check is a hard error returned before the syscall,
        // never a silent buffered fallback (some filesystems would otherwise buffer
        // an unaligned O_DIRECT op). Host-independent — pure arithmetic on any fd.
        let path = tmp_path("dio-reject");
        let file = plain_file(&path);
        let a = DIO_ALIGN;
        // Unaligned read offset.
        assert_eq!(
            pread_direct(&file, 100, a, a).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        // Unaligned write offset.
        let page = vec![0u8; a].into_boxed_slice();
        assert_eq!(
            pwrite_direct(&file, 100, &page, a).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        // Unaligned write length (aligned offset).
        let stub = vec![0u8; 20];
        assert_eq!(
            pwrite_direct(&file, 0, &stub, a).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_helpers_bounce_roundtrip_and_slice() {
        // The write bounce and the read round-up+slice are byte-correct: a two-page
        // write reads back identically at full-page length, and an unaligned-length
        // read returns exactly the requested prefix (the manifest-read mechanism).
        // Host-independent.
        let path = tmp_path("dio-bounce");
        let file = plain_file(&path);
        let a = DIO_ALIGN;
        let mut page0 = vec![0u8; a];
        let mut page1 = vec![0u8; a];
        for (i, b) in page0.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        for (i, b) in page1.iter_mut().enumerate() {
            *b = ((i + 7) % 251) as u8;
        }
        pwrite_direct(&file, 0, &page0, a).unwrap();
        pwrite_direct(&file, a as u64, &page1, a).unwrap();
        // Full-page reads round-trip byte-for-byte.
        assert_eq!(&pread_direct(&file, 0, a, a).unwrap()[..], &page0[..]);
        assert_eq!(
            &pread_direct(&file, a as u64, a, a).unwrap()[..],
            &page1[..]
        );
        // An unaligned-length read at a page offset rounds up internally and slices
        // back to exactly `len` bytes (here 20, the manifest-entry shape).
        assert_eq!(&pread_direct(&file, 0, 20, a).unwrap()[..], &page0[..20]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn direct_read_tolerates_rounded_read_past_eof() {
        // The old-unpadded-manifest device mechanism: a file whose
        // true length is unaligned (100 bytes) is read with `pread_direct` — the read
        // rounds the length up to a page, runs past EOF (a legal short return), and
        // still yields the 100 true bytes because at least `len` bytes were read.
        // Host-independent (a past-EOF read short-returns on any fd).
        let path = tmp_path("dio-eof");
        let file = plain_file(&path);
        let a = DIO_ALIGN;
        let tail: Vec<u8> = (0..100u32).map(|i| (i % 251) as u8).collect();
        // Write exactly 100 bytes so the file is unaligned in length.
        pwrite_all(&file, 0, &tail).unwrap();
        let got = pread_direct(&file, 0, tail.len(), a).unwrap();
        assert_eq!(&got[..], &tail[..]);
        // Asking for more true bytes than exist is a genuine short read → error.
        assert_eq!(
            pread_direct(&file, 0, a, a).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        let _ = std::fs::remove_file(&path);
    }

    // ---- The direct-transfer state machine, driven by injected completions
    // (host-independent — proves forward progress without a real syscall). ----

    const A: usize = DIO_ALIGN;

    #[test]
    fn drive_direct_write_errors_on_repeated_subalign_short() {
        // The livelock case: a write that keeps returning a
        // sub-alignment partial (the ENOSPC-at-fs-block shape) must terminate with a
        // precise error, never spin. `op` returns a 4 KiB partial each call.
        let mut calls = 0u32;
        let r = drive_direct(2 * A, 2 * A, A, false, |_from, _count| {
            calls += 1;
            assert!(
                calls < 1000,
                "drive_direct livelocked on a repeated short write"
            );
            Ok(4096)
        });
        assert_eq!(r.unwrap_err().kind(), io::ErrorKind::Other);
        assert_eq!(
            calls, 1,
            "a non-boundary short write errors on the first attempt"
        );
    }

    #[test]
    fn drive_direct_write_resumes_on_aligned_partial() {
        // An alignment-boundary partial (one page of a three-page write) is resumed
        // to completion.
        let mut calls = 0u32;
        let r = drive_direct(3 * A, 3 * A, A, false, |from, _count| {
            calls += 1;
            assert_eq!(from % A, 0, "every reissue is aligned");
            Ok(A) // one page per call
        });
        assert_eq!(r.unwrap(), 3 * A);
        assert_eq!(calls, 3);
    }

    #[test]
    fn drive_direct_write_completes_in_one_full_op() {
        let r = drive_direct(2 * A, 2 * A, A, false, |_from, count| Ok(count));
        assert_eq!(r.unwrap(), 2 * A);
    }

    #[test]
    fn drive_direct_read_tolerates_past_eof_short() {
        // A read whose logical need (20 B, the manifest shape) is satisfied by a
        // non-boundary short return (the file's true end) succeeds.
        let r = drive_direct(A, 20, A, true, |_from, _count| Ok(20));
        assert_eq!(r.unwrap(), 20);
    }

    #[test]
    fn drive_direct_read_resumes_aligned_partial_then_eofs_short() {
        // A boundary partial resumes; a subsequent short-before-need is UnexpectedEof.
        let mut calls = 0u32;
        let r = drive_direct(2 * A, 2 * A, A, true, |_from, _count| {
            calls += 1;
            if calls == 1 {
                Ok(A) // aligned partial → resume
            } else {
                Ok(0) // EOF before `need`
            }
        });
        assert_eq!(r.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(
            calls, 2,
            "the first (aligned) partial is resumed, not treated as EOF"
        );
    }

    #[test]
    fn drive_direct_read_retries_are_not_progress() {
        // A retryable error reissues the same aligned request without advancing, then
        // succeeds — no spurious progress, no error.
        let mut calls = 0u32;
        let r = drive_direct(A, A, A, true, |_from, count| {
            calls += 1;
            if calls == 1 {
                Err(io::Error::from_raw_os_error(libc::EINTR))
            } else {
                Ok(count)
            }
        });
        assert_eq!(r.unwrap(), A);
        assert_eq!(calls, 2);
    }

    /// Opens a device O_DIRECT at `path`, or `None` if this filesystem/host does not
    /// honor O_DIRECT (the open or the `statx` validation refuses) — so the O_DIRECT
    /// tests **skip** rather than fail on tmpfs/CI where it is unavailable (the same
    /// probe-and-skip discipline the M7-0 fast-path tests use).
    fn try_open_direct(path: &Path) -> Option<SyscallIo> {
        match SyscallIo::open_direct(path, true) {
            Ok(dev) => Some(dev),
            Err(e) => {
                eprintln!("skip: O_DIRECT unavailable here ({e})");
                None
            }
        }
    }

    #[test]
    fn direct_open_roundtrip_is_byte_correct() {
        let path = real_fs_path("dio-roundtrip");
        let Some(dev) = try_open_direct(&path) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        let a = DIO_ALIGN;
        block_on(async {
            for pg in 0u64..3 {
                let buf = vec![(pg as u8) + 1; a].into_boxed_slice();
                dev.write_all_at(pg * a as u64, buf).await.result.unwrap();
            }
            dev.datasync().await.unwrap();
            for pg in 0u64..3 {
                let got = dev.read_exact_at(pg * a as u64, a).await.unwrap();
                assert!(got.iter().all(|&b| b == (pg as u8) + 1), "page {pg}");
            }
            dev.shutdown().await.unwrap();
        });
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fast_path_is_disabled_under_direct() {
        // An O_DIRECT device never runs the preadv2(RWF_NOWAIT) fast path — it
        // reports inactive and no probe ever runs, so a "cold" arm cannot secretly
        // serve page-cache hits inline.
        let path = real_fs_path("dio-nofast");
        let Some(dev) = try_open_direct(&path) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        let a = DIO_ALIGN;
        block_on(async {
            dev.write_all_at(0, vec![9u8; a].into_boxed_slice())
                .await
                .result
                .unwrap();
            dev.datasync().await.unwrap();
            let _ = dev.read_exact_at(0, a).await.unwrap();
            dev.shutdown().await.unwrap();
        });
        assert!(
            !dev.nowait_active(),
            "the fast path must be inactive under O_DIRECT"
        );
        let (hits, partials, dispatches, _fallbacks) = dev.nowait_stats();
        assert_eq!(
            (hits, partials, dispatches),
            (0, 0, 0),
            "no preadv2 probe may run under O_DIRECT"
        );
        let _ = std::fs::remove_file(&path);
    }
}
