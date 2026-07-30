//! The WAL **file-set owner** capability (M6 Phase B0).
//!
//! [`PageIo`](crate::io::PageIo) is deliberately a *byte device*: positional
//! read/write/sync over one already-open file. It has no create, rename, unlink,
//! or truncate — which is exactly right for the read/write path, and exactly what
//! a checkpoint cannot express.
//!
//! A checkpoint reclaims WAL space by **rotating**: it mints segment `seq+1`,
//! commits the data header naming it, and `unlink`s segment `seq`. Nothing is
//! ever reused in place, and nothing is truncated — a file *prefix* cannot be
//! truncated anyway. Unlink is safe
//! with no refcount because POSIX keeps the inode alive for every fd still open
//! on it: a pre-checkpoint snapshot holding the old generation's `Arc<dyn PageIo>`
//! keeps reading the unlinked segment, and its bytes free the instant that last
//! reference drops. Invariant 8 does the retention for free.
//!
//! That namespace authority is this trait. It is held only by stores that **own**
//! their file set — the path-backed constructors, plus the in-memory
//! [`MemWalFileSet`] the deterministic tests drive. A store over caller-supplied
//! `Arc<dyn PageIo>` devices (`create_with_io` / `open_over`) owns no namespace —
//! a raw device cannot be rotated — and so cannot checkpoint
//! ([`CheckpointError::Unsupported`](crate::error::CheckpointError)).

use crate::error::OpenError;
use crate::io::PageIo;
use std::io;
use std::sync::Arc;

/// Owns a StoreWal's WAL segment namespace: mints, opens, and unlinks segments.
///
/// Implementations are **blocking**. They run on the writer thread inside the
/// checkpoint's maintenance barrier, where no other command is in flight and no
/// caller is waiting on a durability receipt; a checkpoint is rare and already
/// pays several `fsync`s, so a blocking `open`/`fsync` of the parent directory
/// costs nothing measurable and keeps the namespace protocol linear and readable.
pub trait WalFileSet: Send + Sync + 'static {
    /// Creates WAL segment `seq` **durably**: write its identity header, sync it,
    /// and make the directory entry itself durable, so a crash right after this
    /// returns can never leave the header committed but the segment missing.
    /// Fails if the segment already exists (a live segment is never recreated).
    fn create_segment(&self, seq: u64, store_uuid: [u8; 16]) -> io::Result<Arc<dyn PageIo>>;

    /// Opens the existing WAL segment `seq`.
    fn open_segment(&self, seq: u64) -> io::Result<Arc<dyn PageIo>>;

    /// Unlinks WAL segment `seq`. Any still-open handle keeps reading it; the
    /// bytes free when the last one drops. Unlinking an already-absent segment
    /// succeeds (the operation is idempotent — a crash may have done it).
    fn unlink_segment(&self, seq: u64) -> io::Result<()>;

    /// Unlinks every segment except `keep`, returning how many were removed.
    ///
    /// This is the **crash-completion** step of rotation, run at open. A
    /// checkpoint unlinks the segment it retires, but a crash between the header
    /// commit and that unlink leaves it behind, and nothing would ever remove it:
    /// the next checkpoint retires only *its own* predecessor. The retired segment
    /// would then occupy up to `max_wal_bytes` forever, per crash.
    ///
    /// Safe by the design's own rules, at open, under the sidecar lock, before the
    /// writer starts and while no snapshot can exist: `keep` is the only segment
    /// the winning header names; a lower seq belongs to the alternate slot, which
    /// recovery **refuses to fall back to** by contract; a higher seq is an
    /// unpublished orphan from a checkpoint that never committed.
    fn retire_other_segments(&self, keep: u64) -> io::Result<u64>;
}

/// An in-memory file set over [`FakeIo`](crate::io::FakeIo) segments.
///
/// The engine's counterpart to `FakeIo` for the *namespace*: rotation, retention,
/// and reclamation become deterministically observable. It holds one strong `Arc`
/// per live segment, so [`unlink_segment`](WalFileSet::unlink_segment) dropping
/// that `Arc` models `unlink` exactly — the segment's bytes live on for as long as
/// some *other* holder (an old generation pinned by a snapshot) keeps its handle,
/// and vanish when the last one drops.
pub struct MemWalFileSet {
    segments: parking_lot::Mutex<std::collections::HashMap<u64, Arc<crate::io::FakeIo>>>,
}

impl MemWalFileSet {
    pub fn new() -> Arc<Self> {
        Arc::new(MemWalFileSet {
            segments: parking_lot::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// The concrete device for segment `seq`, if it is still linked (tests
    /// inspect its bytes / arm its sync gate).
    pub fn segment(&self, seq: u64) -> Option<Arc<crate::io::FakeIo>> {
        self.segments.lock().get(&seq).cloned()
    }

    /// The sequence numbers of the segments currently linked.
    pub fn live_segments(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.segments.lock().keys().copied().collect();
        v.sort_unstable();
        v
    }

    /// Installs `dev` as segment `seq` without writing a header — for building a
    /// store over an explicit (e.g. crash-image) segment.
    pub fn insert(&self, seq: u64, dev: Arc<crate::io::FakeIo>) {
        self.segments.lock().insert(seq, dev);
    }
}

impl WalFileSet for MemWalFileSet {
    fn create_segment(&self, seq: u64, store_uuid: [u8; 16]) -> io::Result<Arc<dyn PageIo>> {
        let mut segs = self.segments.lock();
        if segs.contains_key(&seq) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "wal segment already exists",
            ));
        }
        let dev = Arc::new(crate::io::FakeIo::new(0));
        futures_lite::future::block_on(async {
            dev.write_all_at(0, crate::wal::encode_wal_header(store_uuid, seq))
                .await
                .result?;
            dev.datasync().await
        })?;
        segs.insert(seq, Arc::clone(&dev));
        Ok(dev as Arc<dyn PageIo>)
    }

    fn open_segment(&self, seq: u64) -> io::Result<Arc<dyn PageIo>> {
        match self.segments.lock().get(&seq) {
            Some(d) => Ok(Arc::clone(d) as Arc<dyn PageIo>),
            None => Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no such wal segment",
            )),
        }
    }

    fn unlink_segment(&self, seq: u64) -> io::Result<()> {
        self.segments.lock().remove(&seq);
        Ok(())
    }

    fn retire_other_segments(&self, keep: u64) -> io::Result<u64> {
        let mut segs = self.segments.lock();
        let before = segs.len();
        segs.retain(|seq, _| *seq == keep);
        Ok((before - segs.len()) as u64)
    }
}

/// Reopens a StoreWal over an in-memory file set, running crash recovery. Like
/// `open_path`, the segment to scan is **resolved from the data header**, not
/// assumed — so a test that reopens over a crash image exercises exactly the
/// segment-resolution the real path does. The deterministic-test counterpart of
/// `open_path`.
pub fn open_mem_store(
    data: Arc<dyn PageIo>,
    file_set: Arc<MemWalFileSet>,
    options: crate::store::Options,
) -> Result<crate::store::Store<crate::backend::Wal>, OpenError> {
    let header = futures_lite::future::block_on(crate::checkpoint::read_winning_header(&*data))?;
    let wal = file_set.open_segment(header.wal_segment_seq)?;
    // Complete any rotation a crash interrupted before its unlink.
    file_set.retire_other_segments(header.wal_segment_seq)?;
    crate::store::Store::<crate::backend::Wal>::open_owned(
        data,
        wal,
        Some(file_set as Arc<dyn WalFileSet>),
        None,
        options,
        None,
    )
}

/// [`open_mem_store`] for a store that also owns its (in-memory) data
/// namespace: the data device is the host's **active** slot, and the store can
/// compact (M6 Phase D). The deterministic-test counterpart of `open_path`
/// after a compaction: the caller resolves the current data image *from the
/// host*, exactly as `open_path` resolves it from the path.
pub fn open_mem_store_hosted(
    host: Arc<crate::compact::MemDataHost>,
    file_set: Arc<MemWalFileSet>,
    options: crate::store::Options,
) -> Result<crate::store::Store<crate::backend::Wal>, OpenError> {
    let data = host.active() as Arc<dyn PageIo>;
    let header = futures_lite::future::block_on(crate::checkpoint::read_winning_header(&*data))?;
    let wal = file_set.open_segment(header.wal_segment_seq)?;
    file_set.retire_other_segments(header.wal_segment_seq)?;
    // Active-path-wins: discard any temp a "crash" left mid-compaction.
    use crate::compact::DataFileHost;
    host.remove_temp();
    crate::store::Store::<crate::backend::Wal>::open_owned(
        data,
        wal,
        Some(file_set as Arc<dyn WalFileSet>),
        Some(host),
        options,
        None,
    )
}

/// Builds a fresh StoreWal over an in-memory file set: creates segment 0 through
/// `file_set` and hands the store both the data device and the namespace, so it
/// can checkpoint. The deterministic-test counterpart of `create_path`.
pub fn create_mem_store(
    data: Arc<dyn PageIo>,
    file_set: Arc<MemWalFileSet>,
    options: crate::store::Options,
) -> Result<crate::store::Store<crate::backend::Wal>, OpenError> {
    let store_uuid = crate::direct::generate_store_uuid();
    let wal = file_set.create_segment(0, store_uuid)?;
    crate::store::Store::<crate::backend::Wal>::create_with_io_owned(
        data,
        wal,
        store_uuid,
        Some(file_set as Arc<dyn WalFileSet>),
        None,
        options,
        None,
    )
}

/// [`create_mem_store`] for a store that also owns its (in-memory) data
/// namespace, so it can compact (M6 Phase D). The data device is the host's
/// active slot. The deterministic-test counterpart of `create_path`.
pub fn create_mem_store_hosted(
    host: Arc<crate::compact::MemDataHost>,
    file_set: Arc<MemWalFileSet>,
    options: crate::store::Options,
) -> Result<crate::store::Store<crate::backend::Wal>, OpenError> {
    let store_uuid = crate::direct::generate_store_uuid();
    let wal = file_set.create_segment(0, store_uuid)?;
    crate::store::Store::<crate::backend::Wal>::create_with_io_owned(
        host.active() as Arc<dyn PageIo>,
        wal,
        store_uuid,
        Some(file_set as Arc<dyn WalFileSet>),
        Some(host),
        options,
        None,
    )
}
