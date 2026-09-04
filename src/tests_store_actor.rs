//! Acceptance for the store actor's command surface, run in-crate so the tests
//! can drive poison and maintenance transitions on the internals directly.
//!
//! Covers apply/get round-trips through the actor, root-at-call snapshot
//! capture, conditional batches publishing nothing when a check fails, old-value
//! alignment with the batch's ops, admission permits surviving caller
//! cancellation, the maintenance pause rejecting fresh admission, poison waking
//! both blocked admission and already-queued commands, flush visibility under a
//! stale incarnation, and close draining accepted writes before every later call
//! resolves terminally.

use crate::admission::AdmissionLimits;
use crate::backend::Direct;
use crate::batch::{ApplyOutcome, WriteBatch};
use crate::error::{MaintenanceError, PoisonReason, WriteError};
use crate::io::FakeIo;
use crate::map::BTreeMap;
use crate::store::{Options, Store};
use crate::version::Incarnation;
use futures_lite::future::{block_on, poll_once};
use std::sync::Arc;
use std::time::Duration;

fn tiny_store(slots: u32, pages: u32) -> BTreeMap<Direct> {
    let data = Arc::new(FakeIo::new(0));
    let options = Options {
        limits: AdmissionLimits {
            max_queue_slots: slots,
            max_command_bytes: 1 << 20,
            max_dirty_pages: pages,
            max_ops_per_batch: 100,
        },
        max_wal_bytes: 1 << 30,
        incarnation: Some(Incarnation::from_raw(0xABCD)),
        ..Options::default()
    };
    let store = Store::<Direct>::create_with_io(data, options).unwrap();
    BTreeMap::over(store)
}

fn spin_until(mut cond: impl FnMut() -> bool) {
    for _ in 0..2000 {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("condition never became true");
}

/// Basic end-to-end apply/get through the actor.
#[test]
fn apply_then_get_roundtrips() {
    let map = tiny_store(8, 64);
    let outcome =
        block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))).unwrap();
    assert!(matches!(outcome, ApplyOutcome::Applied { .. }));
    assert_eq!(
        block_on(map.get(b"k".to_vec())).unwrap(),
        Some(b"v".to_vec())
    );
    assert_eq!(map.len(), 1);
    block_on(map.close()).unwrap();
}

/// Root-at-call / commit-capture: a snapshot taken before a later write keeps
/// observing the version it captured.
#[test]
fn snapshot_captures_root_at_call() {
    let map = tiny_store(8, 64);
    block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"v1".to_vec()))).unwrap();
    let snap = map.snapshot();
    let v1 = snap.version();

    block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"v2".to_vec()))).unwrap();

    // Snapshot still sees v1's value; the live map sees the new one.
    assert_eq!(
        block_on(snap.get(b"k".to_vec())).unwrap(),
        Some(b"v1".to_vec())
    );
    assert_eq!(
        block_on(map.get(b"k".to_vec())).unwrap(),
        Some(b"v2".to_vec())
    );
    assert_eq!(v1.txid() + 1, map.snapshot().version().txid());
    block_on(map.close()).unwrap();
}

/// Conditional batch: a failed check publishes nothing and consumes no txid.
#[test]
fn condition_failed_publishes_nothing() {
    let map = tiny_store(8, 64);
    block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))).unwrap();
    let before = map.snapshot().version().txid();

    // Require k == "wrong"; it isn't, so this must fail without publishing.
    let batch = WriteBatch::new()
        .check(b"k".to_vec(), Some(b"wrong".to_vec()))
        .insert(b"k".to_vec(), b"v2".to_vec());
    let outcome = block_on(map.apply(batch)).unwrap();
    assert!(matches!(
        outcome,
        ApplyOutcome::ConditionFailed { first_failed: 0 }
    ));
    assert_eq!(map.snapshot().version().txid(), before, "no txid consumed");
    assert_eq!(
        block_on(map.get(b"k".to_vec())).unwrap(),
        Some(b"v".to_vec())
    );
    block_on(map.close()).unwrap();
}

/// Old values are returned aligned to ops when requested.
#[test]
fn old_values_aligned_to_ops() {
    let map = tiny_store(8, 64);
    block_on(map.apply(WriteBatch::new().insert(b"a".to_vec(), b"1".to_vec()))).unwrap();
    let batch = WriteBatch::new()
        .insert(b"a".to_vec(), b"2".to_vec())
        .remove(b"b".to_vec())
        .returning_old_values();
    match block_on(map.apply(batch)).unwrap() {
        ApplyOutcome::Applied { old_values, .. } => {
            assert_eq!(old_values, Some(vec![Some(b"1".to_vec()), None]));
        }
        other => panic!("expected Applied, got {other:?}"),
    }
    block_on(map.close()).unwrap();
}

/// A queued permit's command survives the caller dropping the apply future
/// (invariant 4).
#[test]
fn queued_permit_survives_caller_cancellation() {
    let map = tiny_store(8, 64);
    let cost = WriteBatch::new()
        .insert(b"k".to_vec(), b"v".to_vec())
        .cost(crate::store::DEPTH_BOUND);
    let permit = map.try_reserve_write(cost).unwrap();
    let fut = permit
        .apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))
        .unwrap_or_else(|_| panic!("cost within permit"));
    // Cancel the caller: drop the future without awaiting. The command was
    // enqueued synchronously at apply() and still owns the permit.
    drop(fut);
    // The writer still applies it.
    spin_until(|| map.len() == 1);
    assert_eq!(
        block_on(map.get(b"k".to_vec())).unwrap(),
        Some(b"v".to_vec())
    );
    // And the reservation was released after processing.
    spin_until(|| map.store().metrics().admission_slots == 0);
    block_on(map.close()).unwrap();
}

/// Maintenance: fresh admission fails fast with `Compacting`; a permit granted
/// before the pause still enqueues but is not applied until resume, and its
/// reservation stays charged meanwhile.
#[test]
fn maintenance_pauses_writer_and_fails_fresh_admission() {
    let map = tiny_store(8, 64);
    let cost = WriteBatch::new()
        .insert(b"k".to_vec(), b"v".to_vec())
        .cost(crate::store::DEPTH_BOUND);

    // Permit obtained while Running.
    let permit = map.try_reserve_write(cost).unwrap();
    block_on(map.store().begin_maintenance()).unwrap();

    // Fresh admission now fails fast.
    assert!(matches!(
        map.try_reserve_write(cost),
        Err(WriteError::Compacting)
    ));
    assert!(matches!(
        block_on(map.apply(WriteBatch::new().insert(b"x".to_vec(), b"y".to_vec()))),
        Err(WriteError::Compacting)
    ));

    // The pre-maintenance permit enqueues (writer parked, not applied yet).
    let fut = permit
        .apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))
        .unwrap_or_else(|_| panic!("cost within permit"));
    std::thread::sleep(Duration::from_millis(10));
    assert_eq!(map.len(), 0, "writer parked: not applied");
    assert_eq!(
        map.store().metrics().admission_slots,
        1,
        "queued reservation stays charged during maintenance"
    );

    // Resume: the queued command applies.
    map.store().end_maintenance();
    let outcome = block_on(fut).unwrap();
    assert!(matches!(outcome, ApplyOutcome::Applied { .. }));
    assert_eq!(map.len(), 1);
    block_on(map.close()).unwrap();
}

/// Poison wakes an awaiting admission reservation.
#[test]
fn poison_wakes_blocked_admission() {
    let map = tiny_store(1, 64); // one slot
    let cost = WriteBatch::new()
        .insert(b"k".to_vec(), b"v".to_vec())
        .cost(crate::store::DEPTH_BOUND);
    // Occupy the only slot and keep it.
    let _held = map.try_reserve_write(cost).unwrap();

    // Reserve again: blocks (Overloaded → awaits).
    let store = map.store().clone();
    let poisoner = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        store.inner.core.state.poison(PoisonReason::Corrupt("test"));
    });
    let result = block_on(map.reserve_write(cost));
    poisoner.join().unwrap();
    assert!(matches!(result, Err(WriteError::Poisoned(_))));
    // Held permit drops here.
}

/// Poison wakes a queued command waiter: park the writer, enqueue, poison, and
/// the apply future resolves with the terminal error (invariant 6).
#[test]
fn poison_wakes_queued_command() {
    let map = tiny_store(8, 64);
    let cost = WriteBatch::new()
        .insert(b"k".to_vec(), b"v".to_vec())
        .cost(crate::store::DEPTH_BOUND);
    let permit = map.try_reserve_write(cost).unwrap();
    block_on(map.store().begin_maintenance()).unwrap(); // park the writer
    let fut = permit
        .apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))
        .unwrap_or_else(|_| panic!("cost within permit"));

    // Poison from another thread; the parked writer wakes and fail-drains.
    let store = map.store().clone();
    let poisoner = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10));
        store.inner.core.state.poison(PoisonReason::Corrupt("boom"));
    });
    let result = block_on(fut);
    poisoner.join().unwrap();
    assert!(matches!(result, Err(WriteError::Poisoned(_))));
    assert_eq!(map.len(), 0, "poisoned batch never published");
}

/// Direct flush resolves for a visible version and rejects a stale incarnation.
#[test]
fn flush_visible_and_stale_incarnation() {
    let map = tiny_store(8, 64);
    let outcome =
        block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))).unwrap();
    let version = match outcome {
        ApplyOutcome::Applied { version, .. } => version,
        other => panic!("expected Applied, got {other:?}"),
    };
    block_on(map.flush(version)).unwrap();

    // A version from a different incarnation is rejected.
    let stale = crate::version::Version::new(Incarnation::from_raw(0x9999), version.txid());
    assert!(matches!(
        block_on(map.flush(stale)),
        Err(crate::error::FlushError::StaleIncarnation)
    ));
    block_on(map.close()).unwrap();
}

/// Applying on a closed store returns `Ok` with an already-resolved terminal
/// error: the `Err` arm of permit.apply is cost-mismatch only.
#[test]
fn apply_after_close_resolves_terminal() {
    let map = tiny_store(8, 64);
    block_on(map.close()).unwrap();
    let result = block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec())));
    assert!(matches!(result, Err(WriteError::Closed)));
}

/// The permit path validates the batch: the two apply paths agree that a
/// malformed batch is rejected, not silently applied.
#[test]
fn permit_apply_validates_batch() {
    let map = tiny_store(8, 64);
    // Size the permit for a normal insert; an empty-key insert fits the cost but
    // is malformed.
    let cost = WriteBatch::new()
        .insert(b"k".to_vec(), b"v".to_vec())
        .cost(crate::store::DEPTH_BOUND);
    let permit = map.try_reserve_write(cost).unwrap();
    let fut = permit
        .apply(WriteBatch::new().insert(Vec::new(), b"v".to_vec()))
        .unwrap_or_else(|_| panic!("empty-key cost fits the permit"));
    let result = block_on(fut);
    assert!(
        matches!(
            result,
            Err(WriteError::InvalidBatch(crate::batch::BatchError::EmptyKey))
        ),
        "permit path must reject malformed batch, got {result:?}"
    );
    assert_eq!(map.len(), 0, "malformed batch never applied");
    // Reservation released after the rejection.
    spin_until(|| map.store().metrics().admission_slots == 0);
    block_on(map.close()).unwrap();
}

/// A permit applied after a completed close returns `Ok` with an
/// already-resolved terminal error (the `Err` arm is cost-mismatch only).
#[test]
fn permit_apply_after_close_resolves_terminal() {
    let map = tiny_store(8, 64);
    let cost = WriteBatch::new()
        .insert(b"k".to_vec(), b"v".to_vec())
        .cost(crate::store::DEPTH_BOUND);
    let permit = map.try_reserve_write(cost).unwrap();
    block_on(map.close()).unwrap();
    // apply() still returns Ok(future); the future carries the terminal error.
    let fut = permit
        .apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))
        .unwrap_or_else(|_| panic!("cost fits the permit; Err arm is mismatch only"));
    assert!(matches!(block_on(fut), Err(WriteError::Closed)));
}

/// Close drains an accepted write before finishing (clean close preserves
/// visible state).
#[test]
fn close_drains_accepted_write() {
    let map = tiny_store(8, 64);
    let cost = WriteBatch::new()
        .insert(b"k".to_vec(), b"v".to_vec())
        .cost(crate::store::DEPTH_BOUND);
    // Park writer, enqueue, then close: the drain must apply the queued command.
    let permit = map.try_reserve_write(cost).unwrap();
    block_on(map.store().begin_maintenance()).unwrap();
    let _fut = permit
        .apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))
        .unwrap_or_else(|_| panic!("cost within permit"));
    // Resume then immediately close; close drains the accepted write.
    map.store().end_maintenance();
    block_on(map.close()).unwrap();
    assert_eq!(map.len(), 1, "accepted write applied before close finished");
}

/// `begin_maintenance` is a handshake, not an enum flip: it must not resolve
/// while the writer is inside a command it claimed before the pause. Gate the
/// writer immediately before publication, start the pause, and prove it stays
/// pending until the gated command completes — and that the command *does*
/// complete before the pause is established (it was admitted and claimed
/// first), while a command enqueued after it is held for resume. Sleep-free:
/// each negative assertion is a single `poll_once` on the pinned future while
/// the writer is known to be blocked in the gate.
#[test]
fn begin_maintenance_waits_for_a_claimed_command_to_publish() {
    let map = tiny_store(8, 64);
    let core = &map.store().inner.core;
    core.test_gates.publish.arm();

    // The writer claims this command and blocks at the publish gate.
    let in_flight = {
        let map = map.clone();
        std::thread::spawn(move || {
            block_on(map.apply(WriteBatch::new().insert(b"a".to_vec(), b"1".to_vec())))
        })
    };
    core.test_gates.publish.wait_arrived();
    assert_eq!(map.len(), 0, "gated before publication");

    // A permit taken while still Running: enqueue-allowed across the pause.
    let cost = WriteBatch::new()
        .insert(b"b".to_vec(), b"2".to_vec())
        .cost(crate::store::DEPTH_BOUND);
    let permit = map.try_reserve_write(cost).unwrap();

    let mut pause = Box::pin(map.store().begin_maintenance());
    assert!(
        block_on(poll_once(&mut pause)).is_none(),
        "begin_maintenance resolved while a claimed command was still in flight"
    );
    // Fresh admission fails fast as soon as the phase flips...
    assert!(matches!(
        map.try_reserve_write(cost),
        Err(WriteError::Compacting)
    ));
    // ...but the pause is still not established: the writer is mid-command.
    assert!(block_on(poll_once(&mut pause)).is_none());
    assert_eq!(map.len(), 0);

    // The held command lands *before* the pause completes.
    core.test_gates.publish.release();
    block_on(pause).unwrap();
    assert!(matches!(
        in_flight.join().unwrap().unwrap(),
        ApplyOutcome::Applied { .. }
    ));
    assert_eq!(
        map.len(),
        1,
        "the claimed command published before the pause was established"
    );

    // Anything enqueued now waits for resume.
    let mut later = Box::pin(
        permit
            .apply(WriteBatch::new().insert(b"b".to_vec(), b"2".to_vec()))
            .unwrap_or_else(|_| panic!("cost within permit")),
    );
    assert!(block_on(poll_once(&mut later)).is_none(), "held for resume");
    assert_eq!(map.len(), 1);
    map.store().end_maintenance();
    assert!(matches!(
        block_on(later).unwrap(),
        ApplyOutcome::Applied { .. }
    ));
    assert_eq!(map.len(), 2);
    block_on(map.close()).unwrap();
}

/// `begin_maintenance` must not resolve while a flush is running, and the
/// established pause must have made everything published so far durable.
#[test]
fn begin_maintenance_waits_for_a_running_flush() {
    let map = tiny_store(8, 64);
    let core = &map.store().inner.core;
    core.test_gates.flush.arm();

    // Publishes, then the writer idles into `flush_all` and blocks at the gate.
    let outcome =
        block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec()))).unwrap();
    let version = match outcome {
        ApplyOutcome::Applied { version, .. } => version,
        other => panic!("expected Applied, got {other:?}"),
    };
    core.test_gates.flush.wait_arrived();
    assert!(
        core.durable_txid.load(std::sync::atomic::Ordering::Acquire) < version.txid(),
        "flush gated before it advanced durability"
    );

    let mut pause = Box::pin(map.store().begin_maintenance());
    assert!(
        block_on(poll_once(&mut pause)).is_none(),
        "begin_maintenance resolved while a flush was still running"
    );
    assert_eq!(core.state.phase(), crate::state::Lifecycle::Maintenance);

    core.test_gates.flush.release();
    block_on(pause).unwrap();
    assert!(
        core.durable_txid.load(std::sync::atomic::Ordering::Acquire) >= version.txid(),
        "an established pause has flushed everything published before it"
    );
    map.store().end_maintenance();
    block_on(map.close()).unwrap();
}

/// A poison while the writer is stuck mid-command must release a pending
/// `begin_maintenance` (invariant 6: terminal transitions wake every waiter
/// class) — as a **failure**: the in-flight command may still publish, so the
/// caller must not be told the store is quiescent.
#[test]
fn begin_maintenance_is_failed_by_poison() {
    let map = tiny_store(8, 64);
    let core = &map.store().inner.core;
    core.test_gates.publish.arm();
    let in_flight = {
        let map = map.clone();
        std::thread::spawn(move || {
            block_on(map.apply(WriteBatch::new().insert(b"a".to_vec(), b"1".to_vec())))
        })
    };
    core.test_gates.publish.wait_arrived();

    let mut pause = Box::pin(map.store().begin_maintenance());
    assert!(block_on(poll_once(&mut pause)).is_none());

    core.state.poison(PoisonReason::Corrupt("mid-pause"));
    assert!(
        matches!(block_on(pause), Err(MaintenanceError::Poisoned(_))),
        "a terminal transition aborts the pause, never establishes it"
    );
    core.test_gates.publish.release();
    let _ = in_flight.join().unwrap();
}

/// Waiter-side ABA at the store level: a pause resumed by another caller — and
/// possibly re-entered by them — is `Interrupted` for its original waiter, even
/// though the phase reads `Maintenance` again; the re-entered pause belongs to
/// the second caller and is established for *them* only once the writer acks
/// it.
#[test]
fn begin_maintenance_is_interrupted_by_a_concurrent_resume_and_repause() {
    let map = tiny_store(8, 64);
    let core = &map.store().inner.core;
    core.test_gates.publish.arm();
    let in_flight = {
        let map = map.clone();
        std::thread::spawn(move || {
            block_on(map.apply(WriteBatch::new().insert(b"a".to_vec(), b"1".to_vec())))
        })
    };
    core.test_gates.publish.wait_arrived();

    let mut first = Box::pin(map.store().begin_maintenance());
    assert!(block_on(poll_once(&mut first)).is_none());
    // Another operator resumes and immediately re-pauses.
    map.store().end_maintenance();
    let mut second = Box::pin(map.store().begin_maintenance());
    assert!(block_on(poll_once(&mut second)).is_none());
    assert_eq!(core.state.phase(), crate::state::Lifecycle::Maintenance);

    assert!(
        matches!(block_on(first), Err(MaintenanceError::Interrupted)),
        "the first pause was resumed under the waiter; the re-pause is not theirs"
    );
    // A third caller while paused does not own the pause either.
    assert!(matches!(
        block_on(map.store().begin_maintenance()),
        Err(MaintenanceError::AlreadyPaused)
    ));

    core.test_gates.publish.release();
    block_on(second).unwrap();
    assert!(matches!(
        in_flight.join().unwrap().unwrap(),
        ApplyOutcome::Applied { .. }
    ));
    assert_eq!(map.len(), 1);
    map.store().end_maintenance();
    block_on(map.close()).unwrap();
}

/// A concurrent `close()` aborts a pending pause with `Closed`: closing applies
/// and flushes admitted work, so it is not a quiescent barrier.
#[test]
fn begin_maintenance_is_failed_by_close() {
    let map = tiny_store(8, 64);
    let core = &map.store().inner.core;
    core.test_gates.publish.arm();
    let in_flight = {
        let map = map.clone();
        std::thread::spawn(move || {
            block_on(map.apply(WriteBatch::new().insert(b"a".to_vec(), b"1".to_vec())))
        })
    };
    core.test_gates.publish.wait_arrived();

    let mut pause = Box::pin(map.store().begin_maintenance());
    assert!(block_on(poll_once(&mut pause)).is_none());

    let closer = {
        let map = map.clone();
        std::thread::spawn(move || block_on(map.close()))
    };
    spin_until(|| core.state.phase() == crate::state::Lifecycle::Closing);
    assert!(matches!(block_on(pause), Err(MaintenanceError::Closed)));
    core.test_gates.publish.release();
    assert!(matches!(
        in_flight.join().unwrap().unwrap(),
        ApplyOutcome::Applied { .. }
    ));
    closer.join().unwrap().unwrap();
}
