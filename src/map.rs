//! `BTreeMap<B>`, `WritePermit<B>`, and the backend-specific durability
//! barriers.
//!
//! `BTreeMap<B>` is a cheap handle over a [`Store<B>`]. Reads capture the root
//! synchronously at call and hold no lock across `.await` (invariant 12); the
//! returned futures are `Send`. Writes go through admission then the single
//! writer actor.

use crate::admission::{Reservation, WriteCost};
use crate::backend::{Backend, Direct, Wal};
use crate::batch::{ApplyOutcome, WriteBatch};
use crate::error::{CloseError, CommitError, FlushError, PermitMismatch, ReadError, WriteError};
use crate::metrics::Metrics;
use crate::store::{Snapshot, Store, StoreInner, DEPTH_BOUND};
use crate::version::Version;
use std::future::Future;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// A single limited B+tree map over either backend.
pub struct BTreeMap<B: Backend> {
    store: Store<B>,
}

/// The batch size at which [`BTreeMap::get_many`] routes to the concurrent
/// grouped traversal (T2.1 decision). Chosen by the pre-declared
/// rule from the paired cold cross-page benchmark (`store_bench get_many_cold`,
/// 200k-entry / ~1100-leaf / height-3 fixture on btrfs): the smallest measured
/// batch size whose grouped win on cold, page-cache-bypassed (O_DIRECT)
/// cross-page batches (≥ 20% required; ~10× measured) comes with **no control
/// cell** — fully-warm, clustered-cold at this very size, buffered
/// engine-cold — regressing more than 10% on median paired ratios, on both the
/// decision trace and a held-out seed. Warm driver overhead shrinks as batches
/// grow (≈0.85 → 0.94–0.98 → ~1.0 from 512 → 2048 → 4096).
///
/// **Unstable policy data, not a contract**: the value is shape-dependent
/// (tree height/fanout, driver allocation profile, backend) and moves whenever
/// the benchmark and its pre-declared rule are re-run and say so. It is public only
/// so a caller can see where the route flips today.
pub const GROUPED_ROUTE_MIN_BATCH: usize = 2048;

impl<B: Backend> Clone for BTreeMap<B> {
    fn clone(&self) -> Self {
        BTreeMap {
            store: self.store.clone(),
        }
    }
}

/// The result of verifying a map.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct VerifyReport {
    /// Entries reachable from the visible root.
    pub entry_count: u64,
}

impl<B: Backend> BTreeMap<B> {
    /// Wraps a store as its (single, M0) map.
    pub fn over(store: Store<B>) -> Self {
        BTreeMap { store }
    }

    /// The underlying store handle.
    pub fn store(&self) -> &Store<B> {
        &self.store
    }

    /// Page-cache metrics (clean/dirty residency, evictions). Originally the
    /// test-only G3 exit-guard accessor (`Metrics` deliberately does not
    /// surface these); widened for the T2.2 benchmark's per-arm physical-load
    /// accounting. Not a stability contract.
    #[doc(hidden)]
    pub fn cache_metrics(&self) -> crate::cache::CacheMetrics {
        self.store.inner.core.cache.metrics()
    }

    /// Point read. The root is captured synchronously at call; the returned
    /// future is `Send` and holds no lock.
    pub fn get(
        &self,
        key: Vec<u8>,
    ) -> impl Future<Output = Result<Option<Vec<u8>>, ReadError>> + Send {
        let root = self.store.inner.core.root();
        let cache = Arc::clone(&self.store.inner.core.cache);
        let terminal = self.store.inner.core.state.check_terminal();
        async move {
            terminal?;
            crate::read::tree_get(&cache, &root.file, root.root, &key).await
        }
    }

    /// Multi-get against a single captured root, so all results are from one
    /// consistent version. Descents run **sequentially** below
    /// [`GROUPED_ROUTE_MIN_BATCH`] keys and route to the concurrent grouped
    /// traversal at or above it — the T2.1 benchmark-gated threshold policy:
    /// cold, page-cache-bypassed (O_DIRECT) cross-page batches
    /// gain ~10× from the grouped fan-out, while fully-warm small batches lose
    /// 10–17% to its driver overhead; ≥ 2048 keys is where every measured
    /// control cell stays within the pre-declared regression bound. A caller
    /// that *knows* its batch is cold and page-spread should use
    /// [`get_many_grouped`](Self::get_many_grouped) directly at any size.
    ///
    /// The two forms agree on every result. On failure both are deterministic
    /// but not identical: the sequential form reports the error met at the
    /// lowest input position it examined; the grouped form reports the error
    /// covering the lowest input position overall.
    pub fn get_many(
        &self,
        keys: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<Vec<Option<Vec<u8>>>, ReadError>> + Send {
        let root = self.store.inner.core.root();
        let cache = Arc::clone(&self.store.inner.core.cache);
        let terminal = self.store.inner.core.state.check_terminal();
        async move {
            terminal?;
            if keys.len() >= GROUPED_ROUTE_MIN_BATCH {
                crate::read::tree_get_many_grouped(&cache, &root.file, root.root, &keys).await
            } else {
                crate::read::tree_get_many_naive(&cache, &root.file, root.root, &keys).await
            }
        }
    }

    /// Grouped **concurrent** multi-get (T2.1): sorts the batch, descends each
    /// branch once, and keeps one physical read in flight per distinct cold
    /// child page — overflow chains advance concurrently across keys too.
    /// Results are in input order; duplicates are preserved.
    ///
    /// Opt-in at any batch size: on a cold, page-cache-bypassed (O_DIRECT)
    /// cross-page batch this measured ~3–12× over the sequential per-key form
    /// that [`get_many`](Self::get_many) uses below its threshold;
    /// on a fully-warm batch it is up to ~17% slower (pure driver overhead,
    /// shrinking to parity as batches grow).
    /// [`get_many`](Self::get_many) applies the batch-size threshold
    /// automatically when you don't know which case you're in.
    pub fn get_many_grouped(
        &self,
        keys: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<Vec<Option<Vec<u8>>>, ReadError>> + Send {
        let root = self.store.inner.core.root();
        let cache = Arc::clone(&self.store.inner.core.cache);
        let terminal = self.store.inner.core.state.check_terminal();
        async move {
            terminal?;
            crate::read::tree_get_many_grouped(&cache, &root.file, root.root, &keys).await
        }
    }

    /// In-order range scan through a consumer callback. Captures a
    /// [`Snapshot`] internally, so the whole scan sees one consistent root and
    /// stays valid across concurrent cutovers — up to the retention bound
    /// (a retention-expired generation yields `SnapshotExpired`, exactly like
    /// a snapshot read). See [`Snapshot::for_each_range`] for the delivery
    /// contract; use a snapshot directly to run several scans against the same
    /// version.
    pub fn for_each_range<K, R, F>(
        &self,
        range: R,
        consumer: F,
    ) -> impl Future<Output = Result<crate::read::ScanSummary, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: std::ops::RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
    {
        let terminal = self.store.inner.core.state.check_terminal();
        let scan = self.snapshot().for_each_range(range, consumer);
        async move {
            terminal?;
            scan.await
        }
    }

    /// The descending mirror of [`for_each_range`](Self::for_each_range):
    /// same snapshot capture and retention semantics, entries delivered in
    /// **descending** key order.
    pub fn for_each_range_rev<K, R, F>(
        &self,
        range: R,
        consumer: F,
    ) -> impl Future<Output = Result<crate::read::ScanSummary, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: std::ops::RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
    {
        let terminal = self.store.inner.core.state.check_terminal();
        let scan = self.snapshot().for_each_range_rev(range, consumer);
        async move {
            terminal?;
            scan.await
        }
    }

    /// Captures a consistent snapshot of the current visible state. The
    /// snapshot binds to the generation's revocable holder, not to the files
    /// themselves — the retention policy can expire it (M6 Phase E; see
    /// [`Snapshot::is_expired`]).
    pub fn snapshot(&self) -> Snapshot<B> {
        Snapshot::capture(&self.store.inner.core)
    }

    /// Applies a conditional batch. Permit-less sugar: admission occurs lazily
    /// on first poll. Equivalent to reserve-then-apply with the
    /// batch's exact cost.
    ///
    /// **Cancellation boundary:** dropping this future *before* admission
    /// completes releases every reservation and enqueues nothing. Once admission
    /// resolves, the command is enqueued synchronously (`StoreInner::enqueue`) —
    /// from that point a dropped future no longer cancels the write; only its
    /// outcome is lost, exactly like [`WritePermit::apply`]'s `Unknown` case. A
    /// caller that must distinguish "never applied" from "applied, outcome lost"
    /// should use the permit path.
    pub fn apply(
        &self,
        batch: WriteBatch,
    ) -> impl Future<Output = Result<ApplyOutcome, WriteError>> + Send {
        let inner = Arc::clone(&self.store.inner);
        async move {
            let max_ops = inner.core.budget.limits().max_ops_per_batch;
            batch.validate(max_ops).map_err(WriteError::InvalidBatch)?;
            let cost = batch.cost(DEPTH_BOUND);
            // StoreWal: reserve WAL bytes (fail-fast `StoreFull`) before the dirty
            // budget; held across the awaiting dirty reservation so a cancellation
            // releases both. `reserve_wal_bytes` is a no-op for StoreDirect.
            let wal_res = crate::store::reserve_wal_bytes(&inner.core, &cost)?;
            let mut reservation = inner.core.budget.reserve(cost, &inner.core.state).await?;
            if let Some(w) = wal_res {
                reservation.attach_wal(w);
            }
            inner.enqueue(batch, reservation).await
        }
    }

    /// Reserves admission capacity (awaiting) for a later [`WritePermit::apply`].
    pub fn reserve_write(
        &self,
        cost: WriteCost,
    ) -> impl Future<Output = Result<WritePermit<B>, WriteError>> + Send {
        self.store.reserve_write(cost)
    }

    /// Non-blocking reservation.
    pub fn try_reserve_write(&self, cost: WriteCost) -> Result<WritePermit<B>, WriteError> {
        self.store.try_reserve_write(cost)
    }

    /// Entries in the visible root.
    pub fn len(&self) -> u64 {
        self.store.inner.core.root().entry_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A metrics snapshot.
    pub fn metrics(&self) -> Metrics {
        self.store.metrics()
    }

    /// EXPERIMENT (T2.2): forwards [`Store::set_chain_prefetch`] —
    /// see its doc for the cold-only opt-in guidance and caveats.
    #[doc(hidden)]
    pub fn set_chain_prefetch(&self, enabled: bool) {
        self.store.set_chain_prefetch(enabled);
    }

    /// EXPERIMENT (T2.2): forwards [`Store::chain_prefetch_stats`].
    #[doc(hidden)]
    pub fn chain_prefetch_stats(&self) -> crate::cache::ChainPrefetchStats {
        self.store.chain_prefetch_stats()
    }

    /// Structurally verifies the visible tree: page checksums and structure,
    /// key order, subtree key bounds, child refs, overflow chains, and the
    /// generation-wide reachable-once rule — at least as strict as every
    /// production walker (see `read::tree_verify`).
    pub fn verify(&self) -> impl Future<Output = Result<VerifyReport, ReadError>> + Send {
        let root = self.store.inner.core.root();
        let cache = Arc::clone(&self.store.inner.core.cache);
        let terminal = self.store.inner.core.state.check_terminal();
        async move {
            terminal?;
            crate::read::tree_verify_with_count(&cache, &root.file, root.root, root.entry_count)
                .await
        }
    }

    /// Stops intake, drains accepted writes, and closes the store.
    pub fn close(&self) -> impl Future<Output = Result<(), CloseError>> + Send {
        let store = self.store.clone();
        async move { store.close().await }
    }
}

impl BTreeMap<Direct> {
    /// Waits for the direct page/header I/O and sync fence for `version`: pages
    /// are written and fdatasync'd, then the header commits the new root and is
    /// fdatasync'd (the real Direct flush coordinator). This does **not** make a
    /// multi-page update recoverably atomic across a crash — see the crate docs
    /// for `StoreDirect`'s crash caveat.
    pub fn flush(&self, version: Version) -> impl Future<Output = Result<(), FlushError>> + Send {
        let core = Arc::clone(&self.store.inner.core);
        async move {
            if version.incarnation() != core.incarnation {
                return Err(FlushError::StaleIncarnation);
            }
            request_flush(&core, version.txid());
            wait_durable(&core, version.txid(), FlushError::from).await
        }
    }
}

impl BTreeMap<Wal> {
    /// Waits until a complete checksummed WAL commit record for `version` is
    /// synced (the group-commit frontier). A successful result survives a crash:
    /// recovery replays every acknowledged commit.
    pub fn commit(&self, version: Version) -> impl Future<Output = Result<(), CommitError>> + Send {
        let core = Arc::clone(&self.store.inner.core);
        async move {
            if version.incarnation() != core.incarnation {
                return Err(CommitError::StaleIncarnation);
            }
            request_flush(&core, version.txid());
            wait_durable(&core, version.txid(), CommitError::from).await
        }
    }
}

/// Signals the writer to flush at least up to `txid`: raises `flush_request`
/// monotonically and wakes the writer's select. Without this an explicit
/// `flush`/`commit` on a continuously busy command channel could wait far longer
/// than necessary for the writer to next idle.
fn request_flush(core: &Arc<crate::store::Core>, txid: u64) {
    let mut cur = core.flush_request.load(Ordering::Acquire);
    while txid > cur {
        match core.flush_request.compare_exchange_weak(
            cur,
            txid,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => break,
            Err(observed) => cur = observed,
        }
    }
    core.state.event().notify(usize::MAX);
}

/// Shared flush/commit wait: resolve once `durable_txid >= txid`, or on a
/// terminal state (mapped through `map_terminal`). Checks the predicate before
/// listening and after registering (invariant 6).
///
/// **Durability is checked before the terminal state**: a receipt for
/// a txid that is already durable resolves `Ok` even if the store has since
/// closed or poisoned — `durable_txid` never regresses within an open, so the
/// barrier the caller asked for is genuinely satisfied ("`flush(V)` where
/// `V ≤ flushed_version` completes without I/O" and "clean close
/// preserves visible state"). Only when the txid is *not* durable does a
/// terminal state turn into the caller's error.
async fn wait_durable<E>(
    core: &Arc<crate::store::Core>,
    txid: u64,
    map_terminal: impl Fn(crate::error::Terminal) -> E,
) -> Result<(), E> {
    loop {
        if core.durable_txid.load(Ordering::Acquire) >= txid {
            return Ok(());
        }
        if let Err(t) = core.state.check_terminal() {
            return Err(map_terminal(t));
        }
        let listener = core.state.listen();
        if core.durable_txid.load(Ordering::Acquire) >= txid {
            return Ok(());
        }
        if let Err(t) = core.state.check_terminal() {
            return Err(map_terminal(t));
        }
        listener.await;
    }
}

/// A granted admission permit. Consumed by [`apply`](WritePermit::apply) or
/// released in full by drop. A permit reserves against one store and one open
/// incarnation; there is no partial use and no reuse across batches.
pub struct WritePermit<B: Backend> {
    inner: Arc<StoreInner<B>>,
    reservation: Reservation,
    cost: WriteCost,
}

impl<B: Backend> WritePermit<B> {
    pub(crate) fn new(
        inner: Arc<StoreInner<B>>,
        reservation: Reservation,
        cost: WriteCost,
    ) -> Self {
        WritePermit {
            inner,
            reservation,
            cost,
        }
    }

    /// Enqueues synchronously at call: on return the command owns this permit
    /// and survives dropping the future (outcome then `Unknown` to the caller).
    ///
    /// Returns `Err((batch, PermitMismatch))` **without enqueueing** if the
    /// batch's admission cost exceeds what this permit reserved. Applying after
    /// close/poison still returns `Ok`: the future is already resolved to the
    /// store's terminal `WriteError`, and the permit and batch are consumed
    /// locally.
    pub fn apply(
        self,
        batch: WriteBatch,
    ) -> Result<
        impl Future<Output = Result<ApplyOutcome, WriteError>> + Send,
        (WriteBatch, PermitMismatch),
    > {
        let cost = batch.cost(DEPTH_BOUND);
        // Compare every WriteCost dimension. `wal_worst_case_bytes` matters at
        // M4 (invariant 18: no root visible until WAL bytes reserved) and a
        // caller may build an under-declared `WriteCost`, so freeze the check
        // complete now.
        if cost.encoded_bytes > self.cost.encoded_bytes
            || cost.op_count > self.cost.op_count
            || cost.worst_case_pages > self.cost.worst_case_pages
            || cost.wal_worst_case_bytes > self.cost.wal_worst_case_bytes
        {
            return Err((batch, PermitMismatch));
        }
        // Batch validation happens inside `enqueue` (the single choke point), so
        // the permit path cannot skip the size caps.
        Ok(self.inner.enqueue(batch, self.reservation))
    }
}
