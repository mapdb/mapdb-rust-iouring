//! A point-in-time metrics snapshot.
//!
//! M0 exposes the counters the state machine and admission already track. The
//! WAL-specific fields (`wal_bytes`, `overlay_pages`, `open_scan_ms`) are named
//! now — required at M4 — but only populated once StoreWal exists.

/// A snapshot of store counters. Cheap to clone.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct Metrics {
    /// Entries in the currently visible root.
    pub entry_count: u64,
    /// Highest published txid.
    pub visible_txid: u64,
    /// Highest backend-durable txid (Direct flush / WAL commit frontier).
    pub durable_txid: u64,
    /// Admission: queue slots currently held.
    pub admission_slots: u32,
    /// Admission: command bytes currently held.
    pub admission_bytes: u64,
    /// Admission: dirty-page budget currently *reserved* — the sum of queued
    /// commands' worst-case estimates, published-but-unflushed batches'
    /// reconciled charges, and any reconcile overshoot. This is the admission
    /// throttle figure, **not** the count of pinned pages resident in the cache
    /// (that is `CacheMetrics::dirty_resident`).
    pub dirty_pages: u32,
    /// StoreWal logical append frontier (WAL header + accepted records).
    pub wal_logical_bytes: u64,
    /// StoreWal reserved frontier (logical + outstanding worst-case).
    pub wal_reserved_bytes: u64,
    /// StoreWal WAL **file length**. After a torn-tail recovery this may exceed
    /// `wal_logical_bytes`, because recovery never truncates and new records
    /// overwrite stale bytes without extending the file. Never
    /// used for admission — that is `wal_reserved_bytes` vs `max_wal_bytes`.
    pub wal_physical_bytes: u64,
    /// StoreWal runtime page-location overlay entries: the count of **distinct
    /// page ids the locator currently indexes**. Page ids are globally unique and
    /// never reused, and the overlay is append-only within an open (superseded
    /// CoW ancestor pages are *not* removed when a new root supersedes them), so
    /// this is the historical count of pages ever written to the WAL this open —
    /// **not** the number of pages reachable from the current root. It grows with
    /// every committed CoW page and is an index-size / WAL-occupancy signal, not a
    /// tree-density one.
    pub overlay_pages: u64,
    /// StoreWal pages the last checkpoint relocated into the data file and that
    /// the current locator therefore resolves there. Unlike
    /// [`overlay_pages`](Self::overlay_pages) this **is** a live-set figure: a
    /// checkpoint relocates exactly the pages reachable from `R@T`, dropping
    /// superseded CoW ancestors. `0` before the first checkpoint.
    pub checkpoint_pages: u64,
    /// Checkpoints completed this open. A StoreWal compaction is counted here
    /// **too** (it is a checkpoint into a fresh dense file — M6 Phase D), as
    /// well as in [`compactions`](Self::compactions).
    pub checkpoints: u64,
    /// Data-file compactions completed this open (both backends, M6 Phase D).
    pub compactions: u64,
    /// The txid the data file holds outright: every WAL record at or below it is
    /// superseded and is never scanned again. Survives reopen (it is read from
    /// the winning data header), unlike [`checkpoints`](Self::checkpoints).
    pub last_checkpoint_txid: u64,
    /// Data-file length in bytes (M6 Phase F). File *length*, like every other
    /// `*_physical_bytes` here — not blocks the filesystem allocated, which a
    /// sparse file makes materially different.
    ///
    /// Exact for StoreWal (a checkpoint/compaction knows where it stopped writing).
    /// For **StoreDirect** it is the allocator frontier the header commits, which
    /// can exceed the bytes actually written: a batch that allocates pages and then
    /// supersedes them within itself burns those ids without writing them,
    /// and Phase F lets one large value burn up to 65 at a time. So on Direct this
    /// term can over-report by the burned tail.
    pub data_physical_bytes: u64,
    /// The store's total owned bytes (M6 Phase F): `data_physical_bytes +
    /// wal_physical_bytes + retained_segment_bytes` — the live data file, the WAL,
    /// and the superseded inodes retained generations still read.
    ///
    /// Neither bound is guaranteed, and both directions are known:
    ///
    /// - it **under**-reports bytes it cannot see — pinned by an in-flight
    ///   current-root read (rather than a retained snapshot), by an orphaned
    ///   single-flight load (deferred to Phase H), or behind
    ///   `leaked_segments` (counted, never sized);
    /// - it can **over**-report on StoreDirect, by `data_physical_bytes`' burned
    ///   tail (see above).
    ///
    /// This is the same accounting
    /// [`Options::max_data_bytes`](crate::Options::max_data_bytes) is enforced
    /// with, so the two can be reasoned about together — but the checkpoint's
    /// decision is made on *projected* terms plus maintenance headroom, so this
    /// value is not itself the quantity compared against the limit. A skipped
    /// checkpoint reports that quantity in
    /// [`SpaceSkip::projected`](crate::SpaceSkip::projected).
    pub store_physical_bytes: u64,
    /// Live pages in the data file **as of the last maintenance** (M6 Phase H).
    /// Set by every checkpoint (WAL) and compaction (both backends) from the
    /// walked live set. Against `data_physical_bytes / page_size` (the total
    /// resident pages) this gives the live-vs-total fragmentation ratio.
    /// Caveats, so it is not misread:
    ///
    /// - It is the figure **at the last maintenance**, not a live gauge: it goes
    ///   stale as writes land, and on StoreWal counts only pages the checkpoint
    ///   relocated **into the data file** — WAL-overlay pages are not in it (that
    ///   is [`overlay_pages`](Self::overlay_pages)).
    /// - `0` means **no maintenance has walked the live set this open**, which on
    ///   a fresh StoreDirect store means *unknown*, not *empty* — every page is
    ///   live there until the first compaction. Treat `0` as "no signal yet".
    /// - For StoreWal after a checkpoint it equals
    ///   [`checkpoint_pages`](Self::checkpoint_pages); the field exists so the
    ///   same ratio is available on **StoreDirect**, where `checkpoint_pages` is
    ///   always `0` (no locator) and only a compaction reveals the live count.
    pub data_live_pages: u64,
    /// Data-file bytes reclaimed by compaction so far this open (M6 Phase H): the
    /// cumulative `data_bytes_before − data_bytes_after` over compactions that
    /// shrank the file (a dense rewrite dropping superseded CoW pages). A
    /// checkpoint *appends* live pages and so never adds to this. Lifetime this
    /// open, monotonic; it does **not** survive reopen. Distinct from WAL segment
    /// reclamation (segments unlink at rotation and are tracked by
    /// [`retained_segment_bytes`](Self::retained_segment_bytes) /
    /// [`leaked_segments`](Self::leaked_segments)).
    pub reclaimed_bytes: u64,
    /// Generations published **this open** (M6 Phase H): `1` for the generation
    /// open mints, then `+1` per rebind (each checkpoint/compaction retires one
    /// generation and publishes its successor). Monotonic lifetime count; does
    /// not survive reopen. Read with
    /// [`retained_generations`](Self::retained_generations) /
    /// [`draining_generations`](Self::draining_generations) /
    /// [`expired_generations`](Self::expired_generations) to see how many of the
    /// published generations are still pinned, draining, or were force-expired.
    pub generations_created: u64,
    /// Checkpoints skipped because they would breach `max_data_bytes` (M6 Phase
    /// F). Healthy: the store keeps serving, and `max_wal_bytes` is still what
    /// refuses writes. A rising count means maintenance is wedged against the
    /// space limit and the store will hit `StoreFull` when the WAL fills.
    pub checkpoints_skipped_space: u64,
    /// Retired WAL segments a rotation could not unlink. Not a correctness
    /// problem (the winning header does not reference them) but they occupy disk
    /// until removed out of band.
    pub leaked_segments: u64,
    /// Retired generations still pinned by snapshots and **not yet revoked**:
    /// superseded by a checkpoint's rebind but kept alive for snapshots that
    /// hold them. This is the figure `Options::max_retained_generations`
    /// governs (oldest-wins revocation); `0` when no snapshot outlives a
    /// rebind (M6 Phase E).
    pub retained_generations: u64,
    /// Revoked generations whose in-flight snapshot reads have not yet drained.
    /// Already expired (outside the bound), but their segments are still
    /// physically pinned until the last lease completes — a wedged leased read
    /// keeps this non-zero, which is exactly when an operator diagnosing disk
    /// pressure needs to see it.
    pub draining_generations: u64,
    /// Bytes pinned by retained **and** draining generations together — each
    /// holds its retired WAL segment (unlinked at cutover; attributed at its
    /// **physical** file length, which after a torn-tail recovery exceeds the
    /// logical frontier), and a generation retired by a *compaction* (M6
    /// Phase D) additionally pins the superseded data-file inode, whose length
    /// is included here too. Two documented pin classes are *outside* this figure:
    /// an in-flight current-root read future (`BTreeMap::get/get_many/verify`
    /// capture the root at call and pin its generation for the future's own
    /// lifetime), and a snapshot read cancelled mid-load, whose orphaned
    /// single-flight load can hold one device clone past release (a known
    /// residue of the Phase H sweep).
    pub retained_segment_bytes: u64,
    /// Generations force-expired by the retention bound so far (lifetime count
    /// this open; counts only revocations of live generations, never natural
    /// releases). A held snapshot whose generation was expired reads
    /// `ReadError::SnapshotExpired`.
    pub expired_generations: u64,
    /// StoreWal last open scan duration (milliseconds). Coarse: a small, fast
    /// recovery rounds to `0`, indistinguishable from a fresh create — use
    /// [`open_scan_bytes`](Self::open_scan_bytes) as the unambiguous
    /// "a recovery scan ran" signal.
    pub open_scan_ms: u64,
    /// StoreWal bytes accepted by the last open recovery scan: the recovered
    /// logical frontier (WAL header + every committed record accepted), which
    /// **excludes** any rejected torn tail past the accepted prefix — so it is the
    /// bytes recovery *consumed*, not the file size (that is `wal_physical_bytes`,
    /// which may be larger after a torn overwrite). **`0` iff no scan ran** — `0`
    /// for StoreDirect and a fresh `create`, and `> 0` (at least the WAL header)
    /// after any `open`/recovery. Deterministic, so tests can assert a scan
    /// happened without depending on wall-clock resolution.
    pub open_scan_bytes: u64,
}
