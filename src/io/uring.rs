//! io_uring [`PageIo`] built on Compio's low-level `Proactor`, the selected
//! io_uring boundary implementation.
//!
//! Compio owns the hard, kernel-adjacent state: submission/completion queues,
//! buffer lifetime across the kernel boundary, `EINTR` on submit, and ring
//! teardown. This wrapper adds only the thin, higher-level layer MapDB needs and
//! the prototype had to hand-roll: a dedicated driver thread, an owned-buffer
//! request/reply protocol, short-read/short-write resubmission, a priority lane
//! with reserved ring slots for commit-critical `datasync`, a panic/fatal-error
//! poison latch, and deterministic drain-then-join teardown. Everything the rest
//! of the engine sees is `Arc<dyn PageIo>`; Compio never leaks into other
//! modules, so the engine stays runtime-neutral (M1 requirement).
//!
//! Ownership (invariant 8): the `Proactor` holds each op — and therefore
//! its buffer — until the terminal CQE. Dropping a caller future only drops the
//! reply `Receiver`; the driver still reaps the CQE and drops the buffer on the
//! driver thread. A dropped read waiter never cancels the physical load and
//! never frees a buffer the kernel still owns. On an unrecoverable ring error we
//! leak in-flight buffers and the ring rather than free memory the kernel may
//! still touch (the prototype's hard-won lesson).

use super::aligned::{AlignedBuf, DIO_ALIGN};
use super::{BufResult, IoFuture, IoKind, PageIo};
use async_channel::{Receiver, Sender};
use compio_buf::{BufResult as CBufResult, IntoInner, IoBuf, IoBufMut, SetLen, Slice};
use compio_driver::op::{ReadAt, Sync as SyncOp, WriteAt};
use compio_driver::{Key, Proactor, ProactorBuilder, PushEntry};
use futures_channel::oneshot;
use futures_lite::future::{block_on, or};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::Waker;
use std::thread::{self, JoinHandle};

/// Submission-queue depth of the persistent ring.
const RING_ENTRIES: u32 = 256;
/// Ring slots kept in reserve for commit-critical priority ops (`datasync`), so
/// a flood of admitted reads/writes can never starve a commit. This is the
/// "priority reservation" the M1 gate requires, expressed at the boundary.
const RESERVED_PRIORITY_SLOTS: usize = 8;

type ReadOp = ReadAt<Slice<DioBuf>, DupFd>;
type WriteOp = WriteAt<Slice<DioBuf>, DupFd>;
type SyncKeyOp = SyncOp<DupFd>;

/// The owned buffer of a ring read/write op: a plain `Vec` for the buffered path
/// (the pre-M7 behaviour, byte-identical) or a page-aligned [`AlignedBuf`] for the
/// O_DIRECT path (M7-A commit 2). Delegating the compio buffer traits keeps one
/// driver code path; the `direct` flag on each `Pending` selects the aligned resume
/// discipline.
enum DioBuf {
    Plain(Vec<u8>),
    Aligned(AlignedBuf),
}

impl DioBuf {
    /// Extracts the read result: exactly `total` logical bytes as an owned `Box`.
    /// For the aligned buffer this copies the leading `total` bytes out of the
    /// (rounded-capacity) bounce; for the plain `Vec` it publishes the initialized
    /// prefix, exactly as the pre-M7 path did.
    fn into_read_result(self, total: usize) -> Box<[u8]> {
        match self {
            DioBuf::Plain(mut v) => {
                // SAFETY: the kernel initialized `[0, total)` across the resubmission
                // chain; capacity was allocated for at least `total`.
                unsafe { v.set_len(total) };
                v.into_boxed_slice()
            }
            DioBuf::Aligned(a) => a.as_slice()[..total].to_vec().into_boxed_slice(),
        }
    }

    /// Prepares a **read** resubmission whose tail begins at `from`. The plain `Vec`
    /// grew from a zero-length `with_capacity`, so its `[0, from)` prefix (read across
    /// prior completions) must be published as initialized before slicing the tail —
    /// exactly the pre-M7 discipline. The aligned buffer is fully initialized already.
    fn slice_read_from(mut self, from: usize) -> Slice<DioBuf> {
        if let DioBuf::Plain(v) = &mut self {
            // SAFETY: `[0, from)` is initialized (read across prior completions);
            // `from <= capacity`.
            unsafe { v.set_len(from) };
        }
        self.slice(from..)
    }

    /// Prepares a **write** resubmission whose tail begins at `from`. The write buffer
    /// already holds its full initialized length (a plain `Vec` came from the caller's
    /// `Box`; an aligned bounce is a full copy), so — unlike a read — its length must
    /// **not** be truncated: just slice the unwritten tail `[from, len)`.
    fn slice_write_from(self, from: usize) -> Slice<DioBuf> {
        self.slice(from..)
    }
}

impl IoBuf for DioBuf {
    fn as_init(&self) -> &[u8] {
        match self {
            DioBuf::Plain(v) => v.as_init(),
            DioBuf::Aligned(a) => a.as_init(),
        }
    }
}

impl SetLen for DioBuf {
    unsafe fn set_len(&mut self, len: usize) {
        match self {
            // SAFETY: forwarded under the caller's `set_len` contract.
            DioBuf::Plain(v) => unsafe { v.set_len(len) },
            DioBuf::Aligned(a) => unsafe { a.set_len(len) },
        }
    }
}

impl IoBufMut for DioBuf {
    fn as_uninit(&mut self) -> &mut [MaybeUninit<u8>] {
        match self {
            DioBuf::Plain(v) => v.as_uninit(),
            DioBuf::Aligned(a) => a.as_uninit(),
        }
    }
}

/// The buffer handed back to the caller when a write op reaches a terminal outcome.
/// For an O_DIRECT write the caller's `original` bytes are returned (the op wrote a
/// separate aligned bounce, which drops here); the buffered path has no `original`
/// and returns the plain buffer it wrote.
fn terminal_write_buffer(buf: DioBuf, original: Option<Box<[u8]>>) -> Box<[u8]> {
    match original {
        Some(o) => o,
        None => match buf {
            DioBuf::Plain(v) => v.into_boxed_slice(),
            DioBuf::Aligned(_) => unreachable!("a buffered write uses a plain buffer"),
        },
    }
}

/// A cheap, `'static`, clonable `AsFd` handle to the shared data file, so every
/// in-flight op can name the fd without borrowing.
#[derive(Clone)]
struct DupFd(Arc<File>);

impl AsFd for DupFd {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

/// An io_uring [`PageIo`] driven by a persistent ring on a dedicated thread.
pub struct CompioIo {
    inner: Arc<Inner>,
}

struct Inner {
    file: Arc<File>,
    /// O_DIRECT alignment for this device (M7-A commit 2): `Some(DIO_ALIGN)` bounces
    /// every read/write through a page-aligned buffer with the aligned resume
    /// discipline; `None` is the pre-M7 buffered path.
    align: Option<usize>,
    /// Priority requests (`datasync`, `Stop`): drained before normal work and
    /// allowed to use the reserved ring slots.
    prio: Sender<Req>,
    /// Normal requests (reads/writes): capped below ring capacity.
    norm: Sender<Req>,
    /// Interrupts a blocked `Proactor::poll` when a request is enqueued.
    waker: Waker,
    /// Set once teardown begins so new submissions fail fast.
    shut: AtomicBool,
    /// Set by the driver thread on a fatal ring error or panic; submissions
    /// then fail terminally as poisoned rather than as a routine shutdown.
    /// Shared with the driver (which owns no other handle to `Inner`).
    poisoned: Arc<AtomicBool>,
    teardown: Mutex<Teardown>,
}

enum Teardown {
    Running(JoinHandle<io::Result<()>>),
    Done(Result<(), (io::ErrorKind, String)>),
}

enum Req {
    Read {
        off: u64,
        len: usize,
        align: Option<usize>,
        reply: oneshot::Sender<io::Result<Box<[u8]>>>,
    },
    Write {
        off: u64,
        buffer: Box<[u8]>,
        align: Option<usize>,
        reply: oneshot::Sender<BufResult>,
    },
    Sync {
        reply: oneshot::Sender<io::Result<()>>,
    },
    Stop {
        ack: oneshot::Sender<()>,
    },
    /// Test-only: make the driver panic, exercising the poison/leak path
    /// deterministically (no real ring fault is injectable on a live fd).
    #[cfg(test)]
    Fault,
}

/// Per-op driver-thread state, keyed by the op's `user_data`.
enum Pending {
    Read {
        key: Key<ReadOp>,
        off: u64,
        total: usize,
        done: usize,
        /// `Some(align)` for an O_DIRECT read: resubmission stays alignment-valid and
        /// a rounded read past EOF is tolerated once `done >= total` (M7-A commit 2).
        align: Option<usize>,
        reply: oneshot::Sender<io::Result<Box<[u8]>>>,
    },
    Write {
        key: Key<WriteOp>,
        off: u64,
        total: usize,
        done: usize,
        align: Option<usize>,
        /// The caller's original buffer for an O_DIRECT write (the op carries a
        /// separate aligned bounce), returned verbatim on completion; `None` for the
        /// buffered path, which returns the written buffer itself.
        original: Option<Box<[u8]>>,
        reply: oneshot::Sender<BufResult>,
    },
    Sync {
        key: Key<SyncKeyOp>,
        reply: oneshot::Sender<io::Result<()>>,
    },
}

impl CompioIo {
    /// Opens (creating if absent) `path` and starts the driver thread with a
    /// persistent ring. Never truncates existing data.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_with(path, true, false)
    }

    /// Opens (creating if absent) `path`, with `O_DIRECT` if `direct` (M7-A commit
    /// 2). The device factory calls this for the `Data`/`CompactTemp` roles under
    /// `direct_io`.
    pub fn open_direct(path: impl AsRef<Path>, direct: bool) -> io::Result<Self> {
        Self::open_with(path, true, direct)
    }

    /// Opens an **existing** `path`, never creating it, with `O_DIRECT` if `direct`.
    pub fn open_existing_direct(path: impl AsRef<Path>, direct: bool) -> io::Result<Self> {
        Self::open_with(path, false, direct)
    }

    /// Opens an **existing** `path` for I/O, never creating it (M7-0). The
    /// path-backed `open_path` / `open_segment` / `create_temp` sites need this
    /// so the same guarantee holds under the `Uring` backend: a file deleted after
    /// its existence check reports `NotFound`, not a silently re-created junk
    /// store — exactly as [`SyscallIo::open_existing`] does. Without it the
    /// factory would regress that guarantee and `FsDataHost::create_temp`'s
    /// `create_new` discipline.
    pub fn open_existing(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_with(path, false, false)
    }

    fn open_with(path: impl AsRef<Path>, create: bool, direct: bool) -> io::Result<Self> {
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
        // Refuse (never silently downgrade to buffered) if the fs does not honor
        // O_DIRECT at our alignment — same measurement-integrity guard as SyscallIo.
        if direct {
            super::aligned::validate_dio_alignment(&file)?;
        }
        let align = if direct { Some(DIO_ALIGN) } else { None };
        let file = Arc::new(file);
        let (prio_tx, prio_rx) = async_channel::unbounded::<Req>();
        let (norm_tx, norm_rx) = async_channel::unbounded::<Req>();
        // The ring (and thus the waker) is built on the driver thread so we do
        // not require `Proactor: Send`; the waker is handed back before `open`
        // returns.
        let (wtx, wrx) = std::sync::mpsc::sync_channel::<io::Result<Waker>>(1);
        let thread_file = Arc::clone(&file);
        let poisoned = Arc::new(AtomicBool::new(false));
        let thread_poisoned = Arc::clone(&poisoned);
        let handle = thread::Builder::new()
            .name("mapdb-uring-driver".into())
            .spawn(move || driver_main(thread_file, prio_rx, norm_rx, wtx, thread_poisoned))?;
        let waker = match wrx.recv() {
            Ok(Ok(waker)) => waker,
            Ok(Err(err)) => {
                let _ = handle.join();
                return Err(err);
            }
            Err(_) => {
                let _ = handle.join();
                return Err(io::Error::other("uring driver died before start"));
            }
        };
        Ok(CompioIo {
            inner: Arc::new(Inner {
                file,
                align,
                prio: prio_tx,
                norm: norm_tx,
                waker,
                shut: AtomicBool::new(false),
                poisoned,
                teardown: Mutex::new(Teardown::Running(handle)),
            }),
        })
    }

    fn terminal(&self) -> Option<io::Error> {
        if self.inner.poisoned.load(Ordering::Acquire) {
            Some(io::Error::other("uring driver poisoned"))
        } else if self.inner.shut.load(Ordering::Acquire) {
            Some(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "uring device shut down",
            ))
        } else {
            None
        }
    }

    /// Test-only: force the driver to panic, so tests can assert the poison
    /// latch and the completed-op delivery of the fatal path.
    #[cfg(test)]
    pub(crate) fn inject_fault(&self) {
        let _ = self.inner.prio.try_send(Req::Fault);
        self.inner.waker.wake_by_ref();
    }

    /// Test-only: whether the driver has latched poison.
    #[cfg(test)]
    pub(crate) fn is_poisoned(&self) -> bool {
        self.inner.poisoned.load(Ordering::Acquire)
    }
}

fn lost() -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, "uring driver stopped")
}

impl PageIo for CompioIo {
    fn read_exact_at(&self, offset: u64, len: usize) -> IoFuture<'_, io::Result<Box<[u8]>>> {
        if len == 0 {
            return Box::pin(std::future::ready(Ok(Box::default())));
        }
        if let Some(err) = self.terminal() {
            return Box::pin(std::future::ready(Err(err)));
        }
        // O_DIRECT: the offset must be alignment-valid (the length is rounded up into
        // an aligned bounce). Reject rather than let the fs silently buffer an
        // unaligned op.
        if let Some(a) = self.inner.align {
            if offset as usize % a != 0 {
                return Box::pin(std::future::ready(Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "unaligned offset for O_DIRECT read",
                ))));
            }
        }
        let (reply, wait) = oneshot::channel();
        let _ = self.inner.norm.try_send(Req::Read {
            off: offset,
            len,
            align: self.inner.align,
            reply,
        });
        self.inner.waker.wake_by_ref();
        Box::pin(async move { wait.await.unwrap_or_else(|_| Err(lost())) })
    }

    fn write_all_at(&self, offset: u64, buffer: Box<[u8]>) -> IoFuture<'_, BufResult> {
        if buffer.is_empty() {
            return Box::pin(std::future::ready(BufResult {
                result: Ok(()),
                buffer,
            }));
        }
        if let Some(err) = self.terminal() {
            return Box::pin(std::future::ready(BufResult {
                result: Err(err),
                buffer,
            }));
        }
        // O_DIRECT: offset and length must both be alignment-valid; reject rather than
        // let the fs silently buffer an unaligned op.
        if let Some(a) = self.inner.align {
            if offset as usize % a != 0 || buffer.len() % a != 0 {
                return Box::pin(std::future::ready(BufResult {
                    result: Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "unaligned offset or length for O_DIRECT write",
                    )),
                    buffer,
                }));
            }
        }
        let (reply, wait) = oneshot::channel();
        if let Err(send_err) = self.inner.norm.try_send(Req::Write {
            off: offset,
            buffer,
            align: self.inner.align,
            reply,
        }) {
            let Req::Write { buffer, .. } = send_err.into_inner() else {
                unreachable!("try_send returns the same request")
            };
            return Box::pin(std::future::ready(BufResult {
                result: Err(lost()),
                buffer,
            }));
        }
        self.inner.waker.wake_by_ref();
        Box::pin(async move {
            wait.await.unwrap_or_else(|_| BufResult {
                result: Err(lost()),
                buffer: Box::default(),
            })
        })
    }

    fn datasync(&self) -> IoFuture<'_, io::Result<()>> {
        if let Some(err) = self.terminal() {
            return Box::pin(std::future::ready(Err(err)));
        }
        let (reply, wait) = oneshot::channel();
        // Priority lane: commit-critical, drained ahead of reads/writes and
        // allowed the reserved ring slots.
        let _ = self.inner.prio.try_send(Req::Sync { reply });
        self.inner.waker.wake_by_ref();
        Box::pin(async move { wait.await.unwrap_or_else(|_| Err(lost())) })
    }

    fn size(&self) -> IoFuture<'_, io::Result<u64>> {
        // fstat on the shared fd; no ring op needed and cheap.
        let result = self.inner.file.metadata().map(|m| m.len());
        Box::pin(std::future::ready(result))
    }

    fn shutdown(&self) -> IoFuture<'_, io::Result<()>> {
        self.inner.shut.store(true, Ordering::Release);
        let mut teardown = self.inner.teardown.lock();
        match &*teardown {
            Teardown::Done(result) => {
                let cloned = clone_terminal(result);
                return Box::pin(std::future::ready(cloned));
            }
            Teardown::Running(_) => {}
        }
        // Ask the driver to drain in-flight work and exit, then join. Joining on
        // the executor is the momentary block shared with Store::close.
        let (ack, ack_wait) = oneshot::channel();
        let _ = self.inner.prio.try_send(Req::Stop { ack });
        self.inner.waker.wake_by_ref();
        let _ = block_on(ack_wait);
        let Teardown::Running(handle) = std::mem::replace(&mut *teardown, Teardown::Done(Ok(())))
        else {
            unreachable!("checked Running above")
        };
        let result = match handle.join() {
            Ok(driver_result) => driver_result,
            Err(_) => Err(io::Error::other("uring driver thread panicked")),
        };
        *teardown = Teardown::Done(
            result
                .as_ref()
                .map(|_| ())
                .map_err(|e| (e.kind(), e.to_string())),
        );
        Box::pin(std::future::ready(result))
    }

    fn kind(&self) -> IoKind {
        IoKind::Uring
    }
}

fn clone_terminal(result: &Result<(), (io::ErrorKind, String)>) -> io::Result<()> {
    match result {
        Ok(()) => Ok(()),
        Err((kind, message)) => Err(io::Error::new(*kind, message.clone())),
    }
}

/// Driver thread entry point: build the ring, publish the waker, run the loop,
/// and convert any panic into a poison so `shutdown`/`join` never wedges.
fn driver_main(
    file: Arc<File>,
    prio: Receiver<Req>,
    norm: Receiver<Req>,
    waker_tx: std::sync::mpsc::SyncSender<io::Result<Waker>>,
    poisoned: Arc<AtomicBool>,
) -> io::Result<()> {
    let mut proactor = match build_proactor(&file) {
        Ok(p) => p,
        Err(err) => {
            let kind = err.kind();
            let _ = waker_tx.send(Err(err));
            return Err(io::Error::from(kind));
        }
    };
    let _ = waker_tx.send(Ok(proactor.waker()));

    let fd = DupFd(file);
    let mut driver = Driver {
        proactor: &mut proactor,
        fd,
        prio: &prio,
        norm: &norm,
        pending: HashMap::new(),
        next_id: std::cell::Cell::new(0),
        stopping: false,
    };

    let outcome = catch_unwind(AssertUnwindSafe(|| driver.run()));
    match outcome {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => {
            // Latch poison BEFORE waking waiters, so a racing submission sees a
            // poisoned device rather than a routine shutdown.
            poisoned.store(true, Ordering::Release);
            driver.abort_and_leak(&format!("uring driver poisoned: {err}"));
            std::mem::forget(proactor);
            Err(err)
        }
        Err(_) => {
            poisoned.store(true, Ordering::Release);
            driver.abort_and_leak("uring driver poisoned: driver panicked");
            std::mem::forget(proactor);
            Err(io::Error::other("uring driver panicked"))
        }
    }
}

fn build_proactor(file: &File) -> io::Result<Proactor> {
    let mut builder = ProactorBuilder::new();
    builder.capacity(RING_ENTRIES);
    let mut proactor = builder.build()?;
    use std::os::fd::AsRawFd;
    // A no-op on io_uring and the poll fallback (only IOCP attaches); called for
    // uniformity. Ignore an Unsupported result so the ring path is unaffected.
    match proactor.attach(file.as_raw_fd()) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::Unsupported => {}
        Err(err) => return Err(err),
    }
    Ok(proactor)
}

struct Driver<'a> {
    proactor: &'a mut Proactor,
    fd: DupFd,
    prio: &'a Receiver<Req>,
    norm: &'a Receiver<Req>,
    pending: HashMap<u64, Pending>,
    /// Monotonic bookkeeping id for the pending map. Unrelated to compio's
    /// internal op identity; we only need a unique key per in-flight op.
    next_id: std::cell::Cell<u64>,
    stopping: bool,
}

impl Driver<'_> {
    /// A fresh bookkeeping id for the pending map. `&self` (via `Cell`) so it
    /// composes with `self.pending.insert(self.alloc_id(), _)`.
    fn alloc_id(&self) -> u64 {
        let id = self.next_id.get();
        self.next_id.set(id.wrapping_add(1));
        id
    }

    fn run(&mut self) -> io::Result<()> {
        loop {
            self.drain_requests()?;

            if self.stopping && self.pending.is_empty() {
                return Ok(());
            }

            if !self.pending.is_empty() {
                // Block for at least one completion; a newly enqueued request
                // wakes this through the proactor waker.
                if let Err(err) = self.proactor.poll(None) {
                    // Transient: a signal on the driver thread (EINTR) or CQ
                    // backpressure that Compio maps to `Interrupted`/`TimedOut`.
                    // These are retry signals, not device faults — never poison
                    // a healthy ring because someone ran a profiler. Fall
                    // through to harvest anything already reaped, then loop.
                    if !matches!(
                        err.kind(),
                        io::ErrorKind::Interrupted | io::ErrorKind::TimedOut
                    ) {
                        return Err(err);
                    }
                }
                self.harvest()?;
            } else if !self.stopping {
                // Nothing in flight: park until a request arrives, preferring
                // the priority lane.
                match block_on(or(self.prio.recv(), self.norm.recv())) {
                    Ok(req) => self.accept(req)?,
                    Err(_) => return Ok(()), // all senders dropped
                }
            }
        }
    }

    /// Pull as many ready requests as the ring can hold without blocking.
    fn drain_requests(&mut self) -> io::Result<()> {
        // Priority first, up to full ring capacity.
        while self.pending.len() < RING_ENTRIES as usize {
            match self.prio.try_recv() {
                Ok(req) => self.accept(req)?,
                Err(_) => break,
            }
        }
        // Normal work leaves the reserved slots free for priority ops.
        let normal_cap = (RING_ENTRIES as usize).saturating_sub(RESERVED_PRIORITY_SLOTS);
        while !self.stopping && self.pending.len() < normal_cap {
            match self.norm.try_recv() {
                Ok(req) => self.accept(req)?,
                Err(_) => break,
            }
        }
        Ok(())
    }

    fn accept(&mut self, req: Req) -> io::Result<()> {
        match req {
            #[cfg(test)]
            Req::Fault => panic!("injected driver fault for tests"),
            Req::Stop { ack } => {
                self.stopping = true;
                self.prio.close();
                self.norm.close();
                self.fail_queued(&lost().to_string());
                // Acknowledge intent immediately; the loop then drains in-flight
                // ops before the thread exits and `shutdown` joins it.
                let _ = ack.send(());
                Ok(())
            }
            Req::Sync { reply } => {
                let op = SyncOp::new(self.fd.clone(), true);
                match self.proactor.push(op) {
                    PushEntry::Pending(key) => {
                        self.pending
                            .insert(self.alloc_id(), Pending::Sync { key, reply });
                    }
                    PushEntry::Ready(CBufResult(res, _op)) => {
                        let _ = reply.send(res.map(|_| ()));
                    }
                }
                Ok(())
            }
            Req::Read {
                off,
                len,
                align,
                reply,
            } => {
                // The read buffer: an aligned bounce rounded up to a page for
                // O_DIRECT, else a plain `Vec` (the pre-M7 path). `total` is the
                // logical length; the aligned bounce may be larger.
                let buf = match align {
                    Some(a) => DioBuf::Aligned(AlignedBuf::zeroed(len.next_multiple_of(a))),
                    None => DioBuf::Plain(Vec::with_capacity(len)),
                };
                let op = ReadAt::new(self.fd.clone(), off, buf.slice(0..));
                match self.proactor.push(op) {
                    PushEntry::Pending(key) => {
                        self.pending.insert(
                            self.alloc_id(),
                            Pending::Read {
                                key,
                                off,
                                total: len,
                                done: 0,
                                align,
                                reply,
                            },
                        );
                    }
                    PushEntry::Ready(CBufResult(res, op)) => {
                        self.finish_read(res, op, off, len, 0, align, reply)?;
                    }
                }
                Ok(())
            }
            Req::Write {
                off,
                buffer,
                align,
                reply,
            } => {
                let total = buffer.len();
                // For O_DIRECT the op writes a page-aligned bounce and the caller's
                // original buffer is returned verbatim on completion; the buffered
                // path writes (and returns) the buffer itself.
                let (buf, original) = match align {
                    Some(_) => (
                        DioBuf::Aligned(AlignedBuf::from_slice(&buffer)),
                        Some(buffer),
                    ),
                    None => (DioBuf::Plain(buffer.into_vec()), None),
                };
                let op = WriteAt::new(self.fd.clone(), off, buf.slice(0..));
                match self.proactor.push(op) {
                    PushEntry::Pending(key) => {
                        self.pending.insert(
                            self.alloc_id(),
                            Pending::Write {
                                key,
                                off,
                                total,
                                done: 0,
                                align,
                                original,
                                reply,
                            },
                        );
                    }
                    PushEntry::Ready(CBufResult(res, op)) => {
                        self.finish_write(res, op, off, total, 0, align, original, reply)?;
                    }
                }
                Ok(())
            }
        }
    }

    /// After a `poll`, retrieve every ready op and complete or resubmit it.
    fn harvest(&mut self) -> io::Result<()> {
        let keys: Vec<u64> = self.pending.keys().copied().collect();
        for ud in keys {
            let Some(entry) = self.pending.remove(&ud) else {
                continue;
            };
            match entry {
                Pending::Read {
                    key,
                    off,
                    total,
                    done,
                    align,
                    reply,
                } => match self.proactor.pop(key) {
                    PushEntry::Pending(key) => {
                        self.pending.insert(
                            self.alloc_id(),
                            Pending::Read {
                                key,
                                off,
                                total,
                                done,
                                align,
                                reply,
                            },
                        );
                    }
                    PushEntry::Ready(CBufResult(res, op)) => {
                        self.finish_read(res, op, off, total, done, align, reply)?;
                    }
                },
                Pending::Write {
                    key,
                    off,
                    total,
                    done,
                    align,
                    original,
                    reply,
                } => match self.proactor.pop(key) {
                    PushEntry::Pending(key) => {
                        self.pending.insert(
                            self.alloc_id(),
                            Pending::Write {
                                key,
                                off,
                                total,
                                done,
                                align,
                                original,
                                reply,
                            },
                        );
                    }
                    PushEntry::Ready(CBufResult(res, op)) => {
                        self.finish_write(res, op, off, total, done, align, original, reply)?;
                    }
                },
                Pending::Sync { key, reply } => match self.proactor.pop(key) {
                    PushEntry::Pending(key) => {
                        self.pending
                            .insert(self.alloc_id(), Pending::Sync { key, reply });
                    }
                    PushEntry::Ready(CBufResult(res, _op)) => {
                        let _ = reply.send(res.map(|_| ()));
                    }
                },
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_read(
        &mut self,
        res: io::Result<usize>,
        op: ReadOp,
        off: u64,
        total: usize,
        done: usize,
        align: Option<usize>,
        reply: oneshot::Sender<io::Result<Box<[u8]>>>,
    ) -> io::Result<()> {
        let buf = op.into_inner().into_inner();
        match res {
            Err(err) => {
                let _ = reply.send(Err(err));
            }
            Ok(0) => {
                let _ = reply.send(Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short read before requested length",
                )));
            }
            Ok(n) => {
                let new_done = done + n;
                if new_done >= total {
                    let _ = reply.send(Ok(buf.into_read_result(total)));
                } else if align.is_some_and(|a| new_done % a != 0) {
                    // O_DIRECT: a completion that is not a full alignment block means
                    // the file ended (a regular-file O_DIRECT read returns short only
                    // at EOF). Since `new_done < total`, it is a genuine short read;
                    // resubmitting at an unaligned offset would loop / EINVAL, so stop.
                    // The buffered path instead resubmits the tail and
                    // gets a follow-up `Ok(0)` → `UnexpectedEof`.
                    let _ = reply.send(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "short read before requested length",
                    )));
                } else {
                    // Resume from `new_done` (an alignment boundary under O_DIRECT, so
                    // the reissue stays aligned and re-reads nothing).
                    let op = ReadAt::new(
                        self.fd.clone(),
                        off + new_done as u64,
                        buf.slice_read_from(new_done),
                    );
                    match self.proactor.push(op) {
                        PushEntry::Pending(key) => {
                            self.pending.insert(
                                self.alloc_id(),
                                Pending::Read {
                                    key,
                                    off,
                                    total,
                                    done: new_done,
                                    align,
                                    reply,
                                },
                            );
                        }
                        PushEntry::Ready(CBufResult(res, op)) => {
                            self.finish_read(res, op, off, total, new_done, align, reply)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_write(
        &mut self,
        res: io::Result<usize>,
        op: WriteOp,
        off: u64,
        total: usize,
        done: usize,
        align: Option<usize>,
        original: Option<Box<[u8]>>,
        reply: oneshot::Sender<BufResult>,
    ) -> io::Result<()> {
        let buf = op.into_inner().into_inner();
        match res {
            Err(err) => {
                let _ = reply.send(BufResult {
                    result: Err(err),
                    buffer: terminal_write_buffer(buf, original),
                });
            }
            Ok(0) => {
                let _ = reply.send(BufResult {
                    result: Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "zero-length write with bytes outstanding",
                    )),
                    buffer: terminal_write_buffer(buf, original),
                });
            }
            Ok(n) => {
                let new_done = done + n;
                if new_done >= total {
                    let _ = reply.send(BufResult {
                        result: Ok(()),
                        buffer: terminal_write_buffer(buf, original),
                    });
                } else if align.is_some_and(|a| new_done % a != 0) {
                    // O_DIRECT short write not on an alignment boundary: it cannot be
                    // resumed with an aligned request, so surface a precise error
                    // rather than reissue a doomed suffix forever.
                    let _ = reply.send(BufResult {
                        result: Err(io::Error::other(
                            "short O_DIRECT write cannot resume on an alignment boundary",
                        )),
                        buffer: terminal_write_buffer(buf, original),
                    });
                } else {
                    let op = WriteAt::new(
                        self.fd.clone(),
                        off + new_done as u64,
                        buf.slice_write_from(new_done),
                    );
                    match self.proactor.push(op) {
                        PushEntry::Pending(key) => {
                            self.pending.insert(
                                self.alloc_id(),
                                Pending::Write {
                                    key,
                                    off,
                                    total,
                                    done: new_done,
                                    align,
                                    original,
                                    reply,
                                },
                            );
                        }
                        PushEntry::Ready(CBufResult(res, op)) => {
                            self.finish_write(
                                res, op, off, total, new_done, align, original, reply,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Fail all queued (not yet submitted) requests, returning write buffers.
    fn fail_queued(&self, message: &str) {
        for rx in [self.prio, self.norm] {
            while let Ok(req) = rx.try_recv() {
                match req {
                    Req::Read { reply, .. } => {
                        let _ = reply.send(Err(io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            message.to_owned(),
                        )));
                    }
                    Req::Write { buffer, reply, .. } => {
                        let _ = reply.send(BufResult {
                            result: Err(io::Error::new(
                                io::ErrorKind::BrokenPipe,
                                message.to_owned(),
                            )),
                            buffer,
                        });
                    }
                    Req::Sync { reply } => {
                        let _ = reply.send(Err(io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            message.to_owned(),
                        )));
                    }
                    Req::Stop { ack } => {
                        let _ = ack.send(());
                    }
                    #[cfg(test)]
                    Req::Fault => {}
                }
            }
        }
    }

    /// Fatal path. First drain ops whose terminal CQE was *already reaped* (their
    /// buffers are kernel-free) and deliver their real outcomes — a sync that
    /// durably completed reports success, a finished read hands back its bytes.
    /// Only genuinely in-flight ops are leaked (`forget`) so the kernel cannot
    /// write into freed memory; the caller then forgets the ring. This preserves
    /// the comparator's P7 discipline: never fail a provably-completed
    /// op nor over-leak its buffer.
    ///
    /// `pop` performs no ring syscall — it only checks `key.has_result()` — so it
    /// is safe even when the ring itself is unusable.
    fn abort_and_leak(&mut self, message: &str) {
        self.prio.close();
        self.norm.close();
        self.fail_queued(message);
        let entries: Vec<Pending> = self.pending.drain().map(|(_, e)| e).collect();
        for entry in entries {
            match entry {
                Pending::Read {
                    key,
                    total,
                    done,
                    reply,
                    ..
                } => match self.proactor.pop(key) {
                    PushEntry::Ready(CBufResult(res, op)) => {
                        let buf = op.into_inner().into_inner();
                        match res {
                            Ok(n) if done + n >= total => {
                                let _ = reply.send(Ok(buf.into_read_result(total)));
                            }
                            // Completed but partial, or errored: cannot finish
                            // mid-teardown. Buffer is kernel-free; just drop it.
                            _ => {
                                let _ = reply.send(Err(io::Error::other(message.to_owned())));
                            }
                        }
                    }
                    PushEntry::Pending(key) => {
                        std::mem::forget(key);
                        let _ = reply.send(Err(io::Error::other(message.to_owned())));
                    }
                },
                Pending::Write {
                    key,
                    total,
                    done,
                    original,
                    reply,
                    ..
                } => match self.proactor.pop(key) {
                    PushEntry::Ready(CBufResult(res, op)) => {
                        // The CQE landed: the buffer is kernel-free and returnable
                        // (the caller's original for O_DIRECT, else the written one).
                        let buffer = terminal_write_buffer(op.into_inner().into_inner(), original);
                        let result = match res {
                            Ok(n) if done + n >= total => Ok(()),
                            _ => Err(io::Error::other(message.to_owned())),
                        };
                        let _ = reply.send(BufResult { result, buffer });
                    }
                    PushEntry::Pending(key) => {
                        // The kernel may still write the op's buffer, so the op (and
                        // its kernel-visible buffer) is leaked with `key`. For an
                        // O_DIRECT write that leaked buffer is the aligned *bounce*;
                        // the caller's `original` is a separate, kernel-invisible Box,
                        // so return it verbatim (invariant 8 holds unconditionally). A
                        // buffered write has no `original` — its caller buffer is the
                        // leaked one, so an empty box is the best we can return.
                        std::mem::forget(key);
                        let _ = reply.send(BufResult {
                            result: Err(io::Error::other(message.to_owned())),
                            buffer: original.unwrap_or_default(),
                        });
                    }
                },
                Pending::Sync { key, reply } => match self.proactor.pop(key) {
                    PushEntry::Ready(CBufResult(res, _op)) => {
                        // A durable sync reports success even during teardown.
                        let _ = reply.send(res.map(|_| ()));
                    }
                    PushEntry::Pending(key) => {
                        std::mem::forget(key);
                        let _ = reply.send(Err(io::Error::other(message.to_owned())));
                    }
                },
            }
        }
    }
}

/// Benchmark-only helpers comparing ring topologies. Kept here so they can build
/// the ring the same way the production boundary does.
#[doc(hidden)]
pub mod bench {
    use super::*;
    use std::time::Instant;

    /// One read on a freshly built, single-use ring — the "per-operation ring"
    /// arm. Pays ring setup/teardown on every call, which is exactly the cost
    /// the persistent ring amortizes.
    pub fn per_op_read(file: &Arc<File>, off: u64, len: usize) -> io::Result<()> {
        // Smallest ring that fits one op: a per-operation ring should pay the
        // least setup it can. Larger rings only make this arm look worse (and
        // exhaust locked memory under concurrency).
        let mut builder = ProactorBuilder::new();
        builder.capacity(4);
        let mut proactor = builder.build()?;
        let fd = DupFd(Arc::clone(file));
        let vec: Vec<u8> = Vec::with_capacity(len);
        let mut key = match proactor.push(ReadAt::new(fd, off, vec.slice(0..))) {
            PushEntry::Pending(key) => key,
            PushEntry::Ready(CBufResult(res, _)) => return res.map(|_| ()),
        };
        loop {
            proactor.poll(None)?;
            match proactor.pop(key) {
                PushEntry::Ready(CBufResult(res, _)) => return res.map(|_| ()),
                PushEntry::Pending(k) => key = k,
            }
        }
    }

    /// Wall-clock a closure, returning its result and elapsed nanoseconds.
    pub fn timed<T>(f: impl FnOnce() -> T) -> (T, u128) {
        let start = Instant::now();
        let out = f();
        (out, start.elapsed().as_nanos())
    }
}

#[cfg(test)]
mod dio_buf_tests {
    use super::*;

    #[test]
    fn write_resume_slice_submits_the_unwritten_tail() {
        // Regression guard: a buffered write
        // resume must submit the tail `[from, total)`, never an empty slice —
        // `slice_write_from` must not truncate the payload the way read-resume prep
        // does. Host-independent: pure compio `Slice` math, no ring.
        let buf = DioBuf::Plain(vec![1u8, 2, 3, 4, 5, 6, 7, 8]);
        let sl = buf.slice_write_from(4);
        assert_eq!(
            sl.as_init(),
            &[5, 6, 7, 8],
            "write resume submits [from, total)"
        );

        // The aligned write path slices the full bounce tail identically.
        let ab = DioBuf::Aligned(AlignedBuf::from_slice(&[9u8; DIO_ALIGN]));
        let sl = ab.slice_write_from(DIO_ALIGN / 2);
        assert_eq!(
            sl.as_init().len(),
            DIO_ALIGN / 2,
            "aligned write tail is the suffix"
        );
    }

    #[test]
    fn read_resume_slice_offers_the_tail_capacity() {
        // A buffered read resume publishes the `[0, from)` prefix and offers the rest
        // of capacity for the next read; the aligned bounce offers its full tail.
        let mut buf = DioBuf::Plain(Vec::with_capacity(8));
        if let DioBuf::Plain(v) = &mut buf {
            v.extend_from_slice(&[0u8; 4]); // 4 bytes already "read"
        }
        let mut sl = buf.slice_read_from(4);
        assert_eq!(sl.as_uninit().len(), 4, "read resume offers [from, cap)");

        let ab = DioBuf::Aligned(AlignedBuf::zeroed(2 * DIO_ALIGN));
        let mut sl = ab.slice_read_from(DIO_ALIGN);
        assert_eq!(
            sl.as_uninit().len(),
            DIO_ALIGN,
            "aligned read resume offers the tail"
        );
    }

    // The three tests below are the Miri surface for the unsafe ownership/length
    // transitions (T3.5): host-independent,
    // no ring, run under `cargo miri test --features io-uring --lib
    // io::uring::dio_buf_tests` in CI.

    #[test]
    fn into_read_result_publishes_exactly_total() {
        // Plain: capacity may exceed the logical transfer; only the initialized
        // `[0, total)` prefix is published.
        let mut v = Vec::with_capacity(8);
        v.extend_from_slice(&[1u8, 2, 3, 4, 5]);
        let out = DioBuf::Plain(v).into_read_result(5);
        assert_eq!(&*out, &[1, 2, 3, 4, 5]);

        // Aligned: the rounded-capacity bounce yields exactly the `total` prefix
        // as an owned plain Box; the bounce (and its exact Layout) drops here.
        let ab = AlignedBuf::from_slice(&[7u8; DIO_ALIGN]);
        let out = DioBuf::Aligned(ab).into_read_result(10);
        assert_eq!(out.len(), 10);
        assert!(out.iter().all(|&b| b == 7));
    }

    #[test]
    fn terminal_write_buffer_returns_the_original_for_direct_writes() {
        // O_DIRECT arm: the caller's original bytes come back untouched and the
        // aligned bounce is dropped (Miri checks the layout-exact deallocation).
        let original: Box<[u8]> = vec![3u8; 100].into_boxed_slice();
        let bounce = DioBuf::Aligned(AlignedBuf::from_slice(&[3u8; DIO_ALIGN]));
        let back = terminal_write_buffer(bounce, Some(original));
        assert_eq!(back.len(), 100);
        assert!(back.iter().all(|&b| b == 3));

        // Buffered arm: no original — the plain buffer the ring wrote is returned.
        let back = terminal_write_buffer(DioBuf::Plain(vec![9u8, 8, 7]), None);
        assert_eq!(&*back, &[9, 8, 7]);
    }

    #[test]
    fn set_len_respects_each_variants_contract() {
        // Plain forwards to Vec::set_len — the published initialized length moves.
        let mut v = Vec::with_capacity(8);
        v.extend_from_slice(&[5u8; 8]);
        let mut buf = DioBuf::Plain(v);
        // SAFETY: `[0, 8)` was initialized above; 4 <= that.
        unsafe { buf.set_len(4) };
        assert_eq!(buf.as_init(), &[5u8; 4]);
        // SAFETY: `[0, 8)` remains initialized from the extend above.
        unsafe { buf.set_len(8) };
        assert_eq!(buf.as_init(), &[5u8; 8]);

        // Aligned is a deliberate no-op (the driver owns the logical length; the
        // allocation is always fully initialized): `as_init` stays the full len.
        let mut buf = DioBuf::Aligned(AlignedBuf::zeroed(DIO_ALIGN));
        assert_eq!(buf.as_init().len(), DIO_ALIGN);
        // SAFETY: len <= capacity, the whole allocation is zero-initialized.
        unsafe { buf.set_len(DIO_ALIGN / 2) };
        assert_eq!(
            buf.as_init().len(),
            DIO_ALIGN,
            "AlignedBuf::set_len records nothing — the driver tracks transfer length"
        );
    }
}
