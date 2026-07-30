//! The single writer actor, plus the merged Direct flush coordinator.
//!
//! One thread runs a runtime-neutral `block_on` loop (invariant 12: no
//! `parking_lot`/cache/page guard crosses `.await`). It dequeues admitted
//! commands, evaluates conditions against the current real tree, applies the
//! batch as a copy-on-write of the published pages ([`crate::btree`]), inserts
//! the new pages into the cache as `Dirty`/pinned **before** one infallible
//! `ArcSwap` publication (invariant 2), and resolves the caller's reply.
//!
//! Durability progresses without any explicit `flush`: whenever the writer
//! would otherwise idle (command channel empty) it flushes all pending batches,
//! and it force-flushes when any [`DirectCoordinator::should_flush`] trigger
//! fires (dirty high-water, an explicit flush request via `core.flush_request`,
//! or the pending-ledger cap). A caller's `flush`/`commit` raises
//! `flush_request` and waits on `durable_txid` (see [`crate::map`]); `close`
//! drains then flushes the highest visible root.

use crate::admission::{ReconciledWalReservation, Reservation};
use crate::batch::{ApplyOutcome, BatchError, WriteBatch};
use crate::direct::{Allocator, DirectCoordinator};
use crate::error::{CheckpointError, PoisonReason, ReadError, Terminal, WriteError};
use crate::page::{Page, PageRef};
use crate::read::tree_get;
use crate::state::Lifecycle;
use crate::store::{next_descriptor, Command, Core, RootDescriptor};
use crate::wal::WalCoordinator;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

/// Max pending-drive loads the writer polls in one `drive_pending` batch before
/// returning to the loop top. Bounds command/resume/close latency to
/// `≤ DRIVE_BATCH` non-blocking polls between command checks and amortizes the
/// loop-top preamble; fixed (not scaled with in-flight-load count — that would
/// reintroduce the unbounded latency this bounds) and not configurable. A
/// `Ready`-heavy batch costs more than 64 pending checks (each does CRC/decode/map
/// transition), but that decode work is necessary work for those pages regardless;
/// the batch bounds the *count* of helper polls, the sweep-amplification metric.
const DRIVE_BATCH: usize = 64;

/// The backend durability coordinator, dispatched by the writer loop. Both run
/// inside this thread, so append/flush order and — for
/// WAL — the checksum chain are serialized trivially.
pub(crate) enum Coordinator {
    Direct(DirectCoordinator),
    Wal(WalCoordinator),
}

/// One freshly published batch handed to the coordinator for durability. `wal` is
/// `Some` (the reconciled WAL-byte reservation) iff this is a StoreWal.
pub(crate) struct Published {
    pub txid: u64,
    pub root: PageRef,
    pub entry_count: u64,
    pub pages: Vec<Arc<Page>>,
    pub page_charge: u32,
    pub wal: Option<ReconciledWalReservation>,
}

impl Coordinator {
    fn allocator(&mut self) -> &mut Allocator {
        match self {
            Coordinator::Direct(c) => c.allocator(),
            Coordinator::Wal(c) => c.allocator(),
        }
    }

    fn has_pending(&self) -> bool {
        match self {
            Coordinator::Direct(c) => c.has_pending(),
            Coordinator::Wal(c) => c.has_pending(),
        }
    }

    fn should_flush(&self, requested_txid: u64) -> bool {
        match self {
            Coordinator::Direct(c) => c.should_flush(requested_txid),
            Coordinator::Wal(c) => c.should_flush(requested_txid),
        }
    }

    fn on_published(&mut self, p: Published) {
        match self {
            Coordinator::Direct(c) => {
                debug_assert!(p.wal.is_none(), "Direct batch carries a WAL reservation");
                c.on_published(p.txid, p.root, p.entry_count, p.pages, p.page_charge);
            }
            Coordinator::Wal(c) => {
                let wal = p.wal.expect("StoreWal batch must carry a WAL reservation");
                c.on_published(p.txid, p.root, p.entry_count, p.pages, p.page_charge, wal);
            }
        }
    }

    async fn flush_all(&mut self, core: &Arc<Core>) -> Result<(), Terminal> {
        match self {
            Coordinator::Direct(c) => c.flush_all(core).await,
            Coordinator::Wal(c) => c.commit_all(core).await,
        }
    }
}

/// What woke the writer's select in the Running arm.
enum Wake {
    Command(Result<Command, async_channel::RecvError>),
    StateChanged,
    /// A page load was installed or completed (H5 idle drive). The writer re-loops
    /// and sweeps parked loads, driving an orphaned single-flight load to terminal
    /// completion so its superseded-generation device pin releases promptly.
    LoadActivity,
}

/// Runs the writer loop until closed (drain), poisoned (fail-drain), or all
/// command senders drop (drain). A panic in write processing routes into the
/// poison path and fail-drains queued commands.
pub(crate) fn run(
    core: Arc<Core>,
    commands: async_channel::Receiver<Command>,
    coordinator: Coordinator,
) {
    let cleanup_core = Arc::clone(&core);
    let cleanup_commands = commands.clone();
    let outcome =
        std::panic::catch_unwind(AssertUnwindSafe(|| run_loop(core, commands, coordinator)));
    if outcome.is_err() {
        cleanup_core
            .state
            .poison(PoisonReason::Internal("writer thread panicked"));
        fail_drain(&cleanup_core, &cleanup_commands);
    }
    // T1.2: release the path-backed sidecar lock as the writer's FINAL act —
    // clean drain, poison, and panic exits alike. Until this line the store's
    // path stays locked, so a last-handle drop cannot expose the namespace to a
    // second opener while this thread is still draining admitted commands, and
    // `close()`'s join-then-return means the path is free once close returns.
    drop(cleanup_core.namespace_lock.lock().take());
}

fn run_loop(
    core: Arc<Core>,
    commands: async_channel::Receiver<Command>,
    mut coordinator: Coordinator,
) {
    futures_lite::future::block_on(async move {
        // Commands pulled out of the channel by a maintenance pause (so a queued
        // checkpoint could be failed rather than left waiting on an operator) and
        // held for when the writer resumes. Always drained before the channel, so
        // admission order is preserved.
        let mut deferred: std::collections::VecDeque<Command> = std::collections::VecDeque::new();
        // The base header + slot B are laid down durably by the create protocol
        // (`write_base_header_blocking`) before this thread starts, so the writer
        // begins straight at the command loop.
        loop {
            match core.state.phase() {
                Lifecycle::Poisoned(_) | Lifecycle::Closed => {
                    fail_deferred(&core, &mut deferred);
                    fail_drain(&core, &commands);
                    // H5: drain parked loads on the poison/closed
                    // teardown too, not only clean close — an orphaned load may hold
                    // a superseded device Arc (and, io_uring, an in-flight kernel
                    // op), which must not outlive the store. The device's own
                    // failure paths resolve pending reads with errors, so a poisoned
                    // device cannot hang this drain.
                    core.cache.drain_parked_loads().await;
                    return;
                }
                Lifecycle::Closing => {
                    // Deferred commands were admitted before the close, so a clean
                    // close must apply them too.
                    while let Some(cmd) = deferred.pop_front() {
                        apply_command(&core, &mut coordinator, cmd).await;
                    }
                    drain_and_finish(&core, &commands, &mut coordinator).await;
                    return;
                }
                Lifecycle::Maintenance => {
                    // A checkpoint/compact *deferred* at a resume boundary (the
                    // drain below re-checks the phase per command and defers
                    // once it flips to Running) must not survive into a
                    // re-pause: it would wait parked on the operator — the
                    // exact state `Compacting` promises away.
                    // Sweep them out with the same fail-fast the channel drain
                    // applies; deferred applies keep their order.
                    for _ in 0..deferred.len() {
                        match deferred.pop_front().expect("len-bounded") {
                            Command::Checkpoint { reply } => {
                                let _ = reply.send(Err(CheckpointError::Compacting));
                            }
                            Command::Compact { reply } => {
                                let _ = reply.send(Err(CheckpointError::Compacting));
                            }
                            other => deferred.push_back(other),
                        }
                    }
                    // Flush pending so paused dirty pages release, then park.
                    // Guarded by `has_pending` so the per-batch `continue` from the
                    // bounded drive below does not repeat a real flush every
                    // batch: at a parked barrier it is a no-op.
                    if coordinator.has_pending() {
                        let _ = coordinator.flush_all(&core).await;
                    }
                    // A checkpoint enqueued *just before* the pause would otherwise
                    // sit in the channel while this thread parks — waiting on an
                    // operator, not on I/O, which is exactly what
                    // `CheckpointError::Compacting` promises cannot happen.
                    // Enqueue-side gating alone cannot close this: the phase can
                    // flip after a command is already queued.
                    //
                    // The applies pulled out alongside it are held here, in order,
                    // and re-dequeued before the channel on resume: they were
                    // admitted before anything still in the channel, so
                    // deferred-first preserves FIFO exactly.
                    while let Ok(cmd) = commands.try_recv() {
                        // The phase can flip back to Running while this drain
                        // holds a command it already popped: a checkpoint or
                        // compact enqueued *after* `end_maintenance` must not
                        // be failed `Compacting` (the
                        // channel's acquire ordering makes this per-command
                        // re-check sufficient: a command sent after the resume
                        // is only observed together with the Running phase).
                        // Defer it instead — deferred commands run first on
                        // resume, so admission order is preserved.
                        if core.state.phase() != Lifecycle::Maintenance {
                            deferred.push_back(cmd);
                            break;
                        }
                        match cmd {
                            Command::Checkpoint { reply } => {
                                let _ = reply.send(Err(CheckpointError::Compacting));
                            }
                            Command::Compact { reply } => {
                                let _ = reply.send(Err(CheckpointError::Compacting));
                            }
                            other => deferred.push_back(other),
                        }
                    }
                    if core.state.phase() != Lifecycle::Maintenance {
                        continue;
                    }
                    let listener = core.state.listen();
                    if core.state.phase() != Lifecycle::Maintenance {
                        continue;
                    }
                    // H5 idle drive: reads stay admitted during a maintenance pause,
                    // which may be arbitrarily long (operator-controlled), so an
                    // orphaned load here needs the same wall-clock bound as the
                    // Running-idle arm. Listen-before-drive, then race the pause's
                    // resume (state change) against load activity. The drive touches
                    // only page reads and no coordinator state, so it does not
                    // violate the pause contract (which is about write/rebind work).
                    // This is a *bounded* `drive_pending` (not the full-set
                    // sweep) — if it leaves work queued, re-loop (re-checking the
                    // phase first) instead of parking, so a large fanout is worked
                    // through in ≤ DRIVE_BATCH-sized interruptible batches.
                    let activity = core.cache.load_activity_listen();
                    if core.cache.drive_pending(DRIVE_BATCH) {
                        continue;
                    }
                    // Park on resume (state change), load activity, OR channel
                    // closure: without the recv
                    // arm, a store dropped while paused strands this thread —
                    // and its device fds and sidecar lock — forever, since no
                    // handle remains to call `end_maintenance`. A command
                    // received here is *deferred*, not applied (the pause
                    // contract): the loop-top sweep fail-fasts a deferred
                    // Checkpoint/Compact and holds applies for resume in FIFO
                    // order.
                    let recv = &commands;
                    let wake = futures_lite::future::or(
                        async move { Wake::Command(recv.recv().await) },
                        futures_lite::future::or(
                            async move {
                                listener.await;
                                Wake::StateChanged
                            },
                            async move {
                                activity.await;
                                Wake::LoadActivity
                            },
                        ),
                    )
                    .await;
                    match wake {
                        Wake::Command(Ok(cmd)) => {
                            deferred.push_back(cmd);
                        }
                        Wake::Command(Err(_closed)) => {
                            // Every handle is gone: no operator can ever resume
                            // this pause. Proceed to a clean close exactly like
                            // the Closing arm — deferred commands were admitted
                            // before the drop and must land (never
                            // `drain_and_finish` directly, which sees only the
                            // channel and would drop them).
                            while let Some(cmd) = deferred.pop_front() {
                                apply_command(&core, &mut coordinator, cmd).await;
                            }
                            drain_and_finish(&core, &commands, &mut coordinator).await;
                            return;
                        }
                        Wake::StateChanged | Wake::LoadActivity => {}
                    }
                    continue;
                }
                Lifecycle::Running => {}
            }

            // Commands parked by a maintenance pause run first: they were admitted
            // before anything still in the channel.
            if let Some(cmd) = deferred.pop_front() {
                apply_command(&core, &mut coordinator, cmd).await;
                if coordinator.should_flush(
                    core.flush_request
                        .load(std::sync::atomic::Ordering::Acquire),
                ) {
                    let _ = coordinator.flush_all(&core).await;
                }
                // Drain a bounded batch after servicing a command so the
                // pending-drive queue stays proportional to in-flight edges even when
                // the writer is continuously busy and never reaches the idle
                // arm. Command priority is unaffected — this runs
                // *after* the command; near-free when the queue is empty (one
                // `pop_front` → `None`). The `more` return is ignored on the busy
                // path: the loop naturally re-checks commands first next turn.
                core.cache.drive_pending(DRIVE_BATCH);
                continue;
            }

            match commands.try_recv() {
                Ok(cmd) => {
                    apply_command(&core, &mut coordinator, cmd).await;
                    if coordinator.should_flush(
                        core.flush_request
                            .load(std::sync::atomic::Ordering::Acquire),
                    ) {
                        // Force a flush to bound pinned dirty growth under
                        // sustained writes. A flush error poisons; the next loop
                        // turn observes it.
                        let _ = coordinator.flush_all(&core).await;
                    }
                    // Bound the pending-drive queue under sustained command traffic
                    // (see the deferred arm above).
                    core.cache.drive_pending(DRIVE_BATCH);
                }
                Err(async_channel::TryRecvError::Empty) => {
                    // Idle: advance durability, then block until a command, a state
                    // change, or page-load activity. Flushing here is what makes
                    // `flush(V)` (a pure wait on `durable_txid`) resolve and what
                    // advances durability with no explicit flush.
                    if coordinator.has_pending() {
                        let _ = coordinator.flush_all(&core).await;
                        continue;
                    }
                    let listener = core.state.listen();
                    if core.state.phase() != Lifecycle::Running {
                        continue;
                    }
                    // H5 idle drive: register the load-activity listener *before*
                    // driving. A load installed/completed before the listen is caught
                    // by this drain; one after it notifies the listener and wakes the
                    // race below. Without this, an orphaned load created after the
                    // last rebind, on a store that then goes idle, would pin its
                    // superseded generation's device until the next rebind or close
                    // (no wall-clock bound). The drive does no coordinator work, so it
                    // cannot disturb `flush(V)`/durability.
                    //
                    // This is a *bounded* `drive_pending` over the pending-drive
                    // queue (edge-targeted), not the old O(N) full-set sweep. If it
                    // leaves work queued, re-loop instead of parking — the loop top
                    // re-checks commands/state first, so a large fanout is drained in
                    // ≤ DRIVE_BATCH interruptible batches with command latency bounded
                    // by one batch.
                    let activity = core.cache.load_activity_listen();
                    if core.cache.drive_pending(DRIVE_BATCH) {
                        continue;
                    }
                    let recv = &commands;
                    let wake = futures_lite::future::or(
                        async move { Wake::Command(recv.recv().await) },
                        futures_lite::future::or(
                            async move {
                                listener.await;
                                Wake::StateChanged
                            },
                            async move {
                                activity.await;
                                Wake::LoadActivity
                            },
                        ),
                    )
                    .await;
                    match wake {
                        Wake::Command(Ok(cmd)) => {
                            apply_command(&core, &mut coordinator, cmd).await;
                            if coordinator.should_flush(
                                core.flush_request
                                    .load(std::sync::atomic::Ordering::Acquire),
                            ) {
                                let _ = coordinator.flush_all(&core).await;
                            }
                            // Same bounded post-apply drain as the other command arms,
                            // so a command that arrives via the idle park is symmetric
                            // with one via `try_recv`.
                            core.cache.drive_pending(DRIVE_BATCH);
                        }
                        Wake::Command(Err(_closed)) => {
                            drain_and_finish(&core, &commands, &mut coordinator).await;
                            return;
                        }
                        // Re-loop: the next idle entry re-registers the listener and
                        // re-drives, driving the load that just moved. Going through
                        // the loop top re-checks the lifecycle phase first.
                        Wake::StateChanged | Wake::LoadActivity => continue,
                    }
                }
                Err(async_channel::TryRecvError::Closed) => {
                    drain_and_finish(&core, &commands, &mut coordinator).await;
                    return;
                }
            }
        }
    });
}

/// Dispatches one command. `Apply` is the write path; `Checkpoint` is the
/// maintenance path, which runs *here* — between commands, on this thread —
/// because that is the only place where the coordinator's whole state (devices,
/// locator, allocator, chain, append offset) is owned by one stack with nothing
/// in flight.
async fn apply_command(core: &Arc<Core>, coordinator: &mut Coordinator, cmd: Command) {
    match cmd {
        Command::Apply {
            batch,
            reservation,
            reply,
        } => apply_batch(core, coordinator, batch, reservation, reply).await,
        Command::Checkpoint { reply } => {
            let _ = reply.send(run_checkpoint(core, coordinator).await);
        }
        Command::Compact { reply } => {
            let _ = reply.send(run_compact(core, coordinator).await);
        }
    }
}

/// Runs a compaction at a **drained, fully-flushed barrier** — established the
/// same way as the checkpoint's: reaching this command means every batch
/// admitted before it has been applied, and the flush below makes every one of
/// them durable, so the visible root is the committed/flushed root when the
/// protocol starts (M6 Phase D).
async fn run_compact(
    core: &Arc<Core>,
    coordinator: &mut Coordinator,
) -> Result<crate::compact::CompactReport, CheckpointError> {
    if let Err(terminal) = core.state.check_terminal() {
        return Err(terminal.into());
    }
    if let Err(terminal) = coordinator.flush_all(core).await {
        return Err(terminal.into());
    }
    let report = match coordinator {
        Coordinator::Direct(c) => c.compact(core).await,
        Coordinator::Wal(c) => c.compact(core).await,
    };
    if report.is_ok() {
        core.cache.sweep_parked_loads();
    }
    report
}

/// Runs a checkpoint at a **drained, fully-committed barrier**.
///
/// The barrier is established, not assumed: reaching this command means every
/// batch admitted before it has already been applied (the channel is FIFO), and
/// the commit below makes every one of them durable — so `committed_txid ==
/// visible` and the dirty window is empty when the protocol starts. Batches
/// admitted *after* this command are still sitting in the channel and will append
/// into the new segment.
async fn run_checkpoint(
    core: &Arc<Core>,
    coordinator: &mut Coordinator,
) -> Result<crate::wal::CheckpointReport, CheckpointError> {
    if let Err(terminal) = core.state.check_terminal() {
        return Err(terminal.into());
    }
    // Commit everything published so far; a checkpoint of a root whose records
    // are not yet durable would supersede WAL bytes that do not exist.
    if let Err(terminal) = coordinator.flush_all(core).await {
        return Err(terminal.into());
    }
    let report = match coordinator {
        Coordinator::Wal(c) => c.checkpoint(core).await,
        // StoreDirect has no WAL to reclaim; its data-file compaction is Phase D.
        Coordinator::Direct(_) => Err(CheckpointError::Unsupported),
    };
    if report.is_ok() {
        // H5: the checkpoint rebind retired a generation (its WAL segment leaves
        // the namespace). Poll parked loads so an orphaned load over that segment
        // is driven to completion and releases its pin promptly.
        core.cache.sweep_parked_loads();
    }
    report
}

/// Applies one write command: evaluate conditions and old values against the
/// current tree, build the CoW replacement, pin its pages, publish once, and
/// record the batch for durability.
async fn apply_batch(
    core: &Arc<Core>,
    coordinator: &mut Coordinator,
    batch: WriteBatch,
    reservation: Reservation,
    reply: futures_channel::oneshot::Sender<Result<ApplyOutcome, WriteError>>,
) {
    if let Err(terminal) = core.state.check_terminal() {
        let _ = reply.send(Err(terminal.into()));
        return;
    }

    let current = core.root();

    // 1. All checks observe current state before any op.
    match first_failing_check(core, &current, &batch).await {
        Ok(Some(first_failed)) => {
            // ConditionFailed: publishes nothing, consumes no txid, releases the
            // reservation on drop. It is an `Ok` outcome.
            let _ = reply.send(Ok(ApplyOutcome::ConditionFailed { first_failed }));
            return;
        }
        Ok(None) => {}
        Err(read_err) => {
            let _ = reply.send(Err(fail_from_read(core, read_err)));
            return;
        }
    }

    // 2. Capture requested old values (aligned to ops) against the pre-batch
    //    tree.
    let old_values = if batch.wants_old_values() {
        match read_old_values(core, &current, &batch, core.max_command_bytes).await {
            Ok(Ok(v)) => Some(v),
            Ok(Err(e)) => {
                let _ = reply.send(Err(WriteError::InvalidBatch(e)));
                return;
            }
            Err(read_err) => {
                let _ = reply.send(Err(fail_from_read(core, read_err)));
                return;
            }
        }
    } else {
        None
    };

    // 3. Build the CoW replacement tree privately.
    //
    // The txid must stay strictly below the cap recovery enforces, so the writer
    // can never acknowledge a version a later open would refuse — the same
    // "as strict as recovery" rule the frontier and record length follow. Reachable
    // only from an adversarially crafted header near the cap; a real store cannot
    // execute 2^48 durable writes.
    if current.version.txid() >= crate::wal::MAX_TXID.saturating_sub(1) {
        let _ = reply.send(Err(WriteError::StoreFull));
        return;
    }
    let version = current.version.next();
    let built = crate::btree::apply_batch(
        &core.cache,
        &current.file,
        current.root,
        current.entry_count,
        batch.ops(),
        coordinator.allocator(),
        version.txid(),
    )
    .await;
    let built = match built {
        Ok(b) => b,
        Err(read_err) => {
            let _ = reply.send(Err(fail_from_read(core, read_err)));
            return;
        }
    };

    // 4. Reconcile the dirty-page reservation to the actual page count before
    //    publication. Either direction: release the
    //    excess (waking blocked admission), or charge the difference — the M0
    //    page estimate is not a hard upper bound for a many-op batch over a
    //    multi-level tree, so a legal write charges the overshoot (released at
    //    the next flush) rather than poisoning.
    let mut reservation = reservation;
    let actual = built.pages.len() as u32;
    reservation.reconcile_to_actual(actual);

    // 5. StoreWal: reconcile the WAL-byte reservation to the exact encoded record
    //    length before publication. The estimate is not a
    //    hard bound (the no-merge builder), so reconcile is
    //    bidirectional: it releases excess, or charges the overshoot and, if the
    //    batch no longer fits `max_wal_bytes`, returns `StoreFull`. That is a
    //    **healthy pre-publication fail-fast** — nothing is published, no txid is
    //    consumed (the visible root is unchanged), and the dirty reservation
    //    releases on return — never a poison.
    let wal_res = if reservation.has_wal() {
        // The recorded frontier (the allocator's next id after this build) must
        // stay strictly below the cap recovery enforces, so the writer can never
        // acknowledge a record a later recovery would reject.
        // Reachable only from an adversarially crafted WAL that seeded
        // a near-`MAX_FRONTIER` frontier — a real store cannot allocate 2^48 ids.
        if coordinator.allocator().peek_next() >= crate::wal::MAX_FRONTIER {
            let _ = reply.send(Err(WriteError::StoreFull));
            return;
        }
        let wr = reservation.take_wal().expect("has_wal");
        let exact = crate::wal::record_len(&built.pages);
        // A record larger than what recovery will scan (`MAX_RECORD_BYTES`) must
        // never be acknowledged — otherwise the next open would refuse it and lose
        // an acknowledged commit. Reject pre-publication
        // with `StoreFull`, exactly as strict as recovery. Reachable only under a
        // non-default, very large `max_dirty_pages`/`max_wal_bytes` config.
        if exact > crate::wal::MAX_RECORD_BYTES as u64 {
            let _ = reply.send(Err(WriteError::StoreFull));
            return;
        }
        match wr.reconcile(exact) {
            Ok(reconciled) => Some(reconciled),
            Err(store_full) => {
                let _ = reply.send(Err(store_full));
                return;
            }
        }
    } else {
        None
    };

    // 6. Pin every new page as Dirty in the cache *before* the root swap
    //    (invariant 2: published pages reachable in cache before publication).
    for page in &built.pages {
        core.cache.insert_dirty(page.clone());
    }

    // 7. Hand the reconciled page charge to the durability frontier (slot+bytes
    //    free now, pages free at flush/commit — invariant 5), then one infallible
    //    publication.
    let page_charge = reservation.hand_pages_to_frontier();
    let next = next_descriptor(&current, built.root, built.entry_count);
    core.root.store(Arc::new(next));
    coordinator.on_published(Published {
        txid: version.txid(),
        root: built.root,
        entry_count: built.entry_count,
        pages: built.pages,
        page_charge,
        wal: wal_res,
    });
    let _ = reply.send(Ok(ApplyOutcome::Applied {
        version,
        old_values,
    }));
}

/// Returns the index of the first failing check, `None` if all pass, or a read
/// error propagated from a page load.
async fn first_failing_check(
    core: &Arc<Core>,
    current: &RootDescriptor,
    batch: &WriteBatch,
) -> Result<Option<usize>, ReadError> {
    for (i, chk) in batch.checks().iter().enumerate() {
        let actual = tree_get(&core.cache, &current.file, current.root, &chk.key).await?;
        let ok = match &chk.expected {
            Some(expected) => actual.as_deref() == Some(expected.as_slice()),
            None => actual.is_none(),
        };
        if !ok {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

/// Reads each op's pre-batch value, aligned to `ops`.
/// Reads the requested pre-batch values, **bounded by `max_command_bytes`**.
///
/// Admission charges a command's *input* encoding. The old-value response is an
/// output, and nothing charged it — which stopped mattering only because values
/// were small. M6 Phase F raised `MAX_VALUE_LEN` 128x, and the two multiply: a
/// ~2 KiB batch of 1,000 `remove(k)` ops with `returning_old_values`, against a
/// 1 MiB value at `k`, asks this loop to materialise ~1 GiB **on the writer
/// thread** before it can apply or reply — and dropping the caller's future does
/// not stop it (the command owns the work). Default limits admit that batch
/// happily.
///
/// So the same ceiling that bounds what a command may carry *in* now bounds what
/// it may carry *out*, and the check accumulates as it reads, so peak memory is
/// the limit plus one value rather than the whole answer.
///
/// **Sharing the ceiling is a choice, and it is not free**: a batch of small
/// values on a store with a tuned-down `max_command_bytes` can now be refused
/// where it was not before — a shape that has nothing to do with Phase F. It is
/// still the right knob: a store that asked for a tight byte budget is precisely
/// the one that must not be handed an unbounded reply, and giving the response its
/// own larger limit would defeat the tuning that store asked for. The defaults
/// leave a margin no inline-sized batch can cross (see `BatchError::OldValuesTooLarge`).
///
/// This cannot move to admission: the answer's size is a property of the store,
/// not of the batch. It fails healthily, pre-publication, consuming no txid.
async fn read_old_values(
    core: &Arc<Core>,
    current: &RootDescriptor,
    batch: &WriteBatch,
    max_bytes: u64,
) -> Result<Result<Vec<Option<Vec<u8>>>, BatchError>, ReadError> {
    let mut out = Vec::with_capacity(batch.ops().len());
    let mut total: u64 = 0;
    for op in batch.ops() {
        let v = tree_get(&core.cache, &current.file, current.root, op.key()).await?;
        total = total.saturating_add(v.as_ref().map_or(0, |b| b.len() as u64));
        if total > max_bytes {
            return Ok(Err(BatchError::OldValuesTooLarge));
        }
        out.push(v);
    }
    Ok(Ok(out))
}

/// Maps a read error hit on the write path to a truthful `WriteError`, poisoning
/// on structural or I/O failure (the writer cannot continue past a corrupt or
/// unreadable old-path page). A terminal read error just maps through.
fn fail_from_read(core: &Arc<Core>, err: ReadError) -> WriteError {
    match err {
        ReadError::Corrupt(what) => poison_and_terminal(core, PoisonReason::Corrupt(what)),
        ReadError::Io(e) => poison_and_terminal(core, PoisonReason::Io(e)),
        ReadError::Poisoned(r) => WriteError::Poisoned(r),
        ReadError::Closed => WriteError::Closed,
        // Writer old-path reads go through the generation's own strong `file`
        // handle, never a snapshot lease — unreachable, and loud if a refactor
        // ever misroutes it (M6 Phase E).
        ReadError::SnapshotExpired => {
            poison_and_terminal(core, PoisonReason::Internal("snapshot lease on write path"))
        }
    }
}

/// Poisons with `reason` (idempotent/sticky) and returns the store's truthful
/// terminal `WriteError`.
fn poison_and_terminal(core: &Arc<Core>, reason: PoisonReason) -> WriteError {
    core.state.poison(reason);
    core.state
        .check_terminal()
        .err()
        .map(WriteError::from)
        .unwrap_or(WriteError::Closed)
}

/// Closing: apply every already-admitted command, flush the highest visible
/// root durable (clean close preserves visible
/// state), then finalize `Closed`. A flush failure poisons; the
/// `close()` waiter then observes `Poisoned`.
async fn drain_and_finish(
    core: &Arc<Core>,
    commands: &async_channel::Receiver<Command>,
    coordinator: &mut Coordinator,
) {
    while let Ok(cmd) = commands.try_recv() {
        apply_command(core, coordinator, cmd).await;
    }
    let _ = coordinator.flush_all(core).await;
    // H5: drive every load parked in the cache to terminal completion before the
    // store closes, so no orphaned single-flight load's device clone (nor, for an
    // io_uring backend, an in-flight kernel op) outlives the store lifecycle.
    core.cache.drain_parked_loads().await;
    core.state.finish_close();
}

/// Fails every command a maintenance pause deferred, with the store's terminal
/// error, releasing its reservation — the `fail_drain` treatment for the commands
/// that are no longer in the channel.
fn fail_deferred(core: &Arc<Core>, deferred: &mut std::collections::VecDeque<Command>) {
    let terminal: WriteError = match core.state.check_terminal() {
        Err(t) => t.into(),
        Ok(()) => WriteError::Closed,
    };
    for cmd in deferred.drain(..) {
        match cmd {
            Command::Apply {
                reply, reservation, ..
            } => {
                drop(reservation);
                let _ = reply.send(Err(terminal.clone()));
            }
            Command::Checkpoint { reply } => {
                let _ = reply.send(Err(checkpoint_terminal(core)));
            }
            Command::Compact { reply } => {
                let _ = reply.send(Err(checkpoint_terminal(core)));
            }
        }
    }
}

/// The store's terminal reason as a `CheckpointError`, for failing a queued
/// checkpoint on the poisoned/closed drain.
fn checkpoint_terminal(core: &Arc<Core>) -> CheckpointError {
    core.state
        .check_terminal()
        .err()
        .map(CheckpointError::from)
        .unwrap_or(CheckpointError::Closed)
}

/// Poisoned/Closed: fail every queued command with the terminal error and
/// release its reservation.
fn fail_drain(core: &Arc<Core>, commands: &async_channel::Receiver<Command>) {
    let terminal: WriteError = match core.state.check_terminal() {
        Err(t) => t.into(),
        Ok(()) => WriteError::Closed,
    };
    while let Ok(cmd) = commands.try_recv() {
        match cmd {
            Command::Apply {
                reply, reservation, ..
            } => {
                drop(reservation);
                let _ = reply.send(Err(terminal.clone()));
            }
            Command::Checkpoint { reply } => {
                let _ = reply.send(Err(checkpoint_terminal(core)));
            }
            Command::Compact { reply } => {
                let _ = reply.send(Err(checkpoint_terminal(core)));
            }
        }
    }
    if matches!(core.state.phase(), Lifecycle::Closing) {
        core.state.finish_close();
    }
}
