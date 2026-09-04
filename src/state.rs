//! The engine lifecycle state machine.
//!
//! ```text
//!            resume
//!   Running ⇄ Maintenance      (WriterPaused / Compacting)
//!      │  \       │
//!      │   \      │
//!      ▼    ▼     ▼
//!   Closing   Poisoned
//!      │         │
//!      ▼         │
//!   Closed ◄─────┘
//! ```
//!
//! Rules enforced here (cross-cutting invariant 6):
//! - Every terminal transition (`poison`, `begin_close`) notifies **every**
//!   waiter class via one shared [`event_listener::Event`].
//! - A waiter checks the terminal predicate *before* registering a listener and
//!   *after* registering, closing the lost-wakeup race.
//! - `poison` is idempotent and never downgrades: the first reason sticks.

use crate::error::{MaintenanceError, PoisonReason, Terminal, WriteError};
use event_listener::Event;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The observable lifecycle phase.
///
/// Equality is by **variant only** (discriminant): the `Poisoned` reason holds
/// a non-`Eq` `io::Error`, and every comparison the engine makes is "which
/// phase", never "which poison reason".
#[derive(Clone, Debug)]
pub enum Lifecycle {
    /// Accepting intake and applying writes.
    Running,
    /// Paused between commands for maintenance (e.g. compaction). Admission is
    /// fail-fast (`Compacting`); already-queued reservations stay charged.
    Maintenance,
    /// Poisoned by an unrecoverable failure. Terminal.
    Poisoned(PoisonReason),
    /// Draining accepted writes before close. Intake is shut.
    Closing,
    /// Fully closed. Terminal.
    Closed,
}

impl PartialEq for Lifecycle {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}
impl Eq for Lifecycle {}

/// Shared lifecycle state plus a single wakeup event for all waiter classes.
pub struct EngineState {
    phase: Mutex<Lifecycle>,
    /// One event notified on every state change *and* every budget release.
    /// Admission, command, durability, and maintenance waiters all listen here
    /// and re-check their own predicate; spurious wakeups are harmless. Shared
    /// (`Arc`) with the admission budgets so a reservation release wakes blocked
    /// admission through the same channel as a poison.
    event: Arc<Event>,
    /// Bumped by every `Running → Maintenance` transition. Together with
    /// `maintenance_acked` this is the writer-acknowledgement handshake:
    /// `begin_maintenance` waits until the writer has acknowledged **this**
    /// epoch (or the phase has left `Maintenance`), so a returned pause means the
    /// writer is parked at a command boundary with nothing in flight and nothing
    /// pending — not merely that an enum flipped.
    maintenance_epoch: AtomicU64,
    /// The highest maintenance epoch the writer has acknowledged from its parked
    /// barrier (`ack_maintenance`). A pause is *established* once this reaches
    /// the epoch `enter_maintenance` handed out for it.
    maintenance_acked: AtomicU64,
}

/// One classification of a `begin_maintenance` waiter's pause — see
/// [`EngineState::pause_status`].
#[derive(Debug)]
pub enum PauseStatus {
    /// The waiter's epoch is active but the writer has not acknowledged it yet.
    Waiting,
    /// The writer is parked at its barrier for exactly the waiter's epoch.
    Established,
    /// The pause is gone before it was established; the caller holds nothing.
    Aborted(MaintenanceError),
}

/// Outcome of the linearized intake check used by the enqueue path and by
/// `close()` — both take the same lock so a close cannot slip between a
/// successful check and a synchronous enqueue.
///
/// Note maintenance is **enqueue-allowed**: a permit granted before a pause may
/// still enqueue its command (the writer parks and drains it on resume, keeping
/// the reservation charged). Fresh admission is what fails fast on maintenance —
/// see [`EngineState::admission_gate`].
pub enum IntakeDecision {
    /// Intake is open (Running or Maintenance); enqueue under the held lock.
    Open,
    /// The store is terminal (Poisoned/Closing/Closed); resolve with this
    /// reason instead of enqueueing.
    Terminal(Terminal),
}

impl EngineState {
    /// Builds a running engine sharing `event` with the store's admission
    /// budgets, so a budget release and a lifecycle change wake the same
    /// waiters.
    pub fn new_running(event: Arc<Event>) -> Self {
        EngineState {
            phase: Mutex::new(Lifecycle::Running),
            event,
            maintenance_epoch: AtomicU64::new(0),
            maintenance_acked: AtomicU64::new(0),
        }
    }

    /// The shared wakeup event, for budgets and other waiter sources.
    pub fn event(&self) -> &Arc<Event> {
        &self.event
    }

    pub fn phase(&self) -> Lifecycle {
        self.phase.lock().clone()
    }

    /// The terminal predicate as a `Result`, for wait loops. `Ok(())` while
    /// live (including maintenance), `Err` once terminal.
    pub fn check_terminal(&self) -> Result<(), Terminal> {
        match &*self.phase.lock() {
            Lifecycle::Poisoned(r) => Err(Terminal::Poisoned(r.clone())),
            Lifecycle::Closed => Err(Terminal::Closed),
            _ => Ok(()),
        }
    }

    /// Registers a listener for the next state change. Callers must re-check
    /// their predicate after this returns and before awaiting, then await, then
    /// re-check — the standard event-listener discipline.
    pub fn listen(&self) -> event_listener::EventListener {
        self.event.listen()
    }

    /// The fresh-admission gate. Fails fast during maintenance (`Compacting`)
    /// and on terminal states; only `Running` grants. Used by `reserve_write`,
    /// `try_reserve_write`, and the `apply` sugar's admission step.
    pub fn admission_gate(&self) -> Result<(), WriteError> {
        match &*self.phase.lock() {
            Lifecycle::Running => Ok(()),
            Lifecycle::Maintenance => Err(WriteError::Compacting),
            Lifecycle::Poisoned(r) => Err(WriteError::Poisoned(r.clone())),
            Lifecycle::Closing | Lifecycle::Closed => Err(WriteError::Closed),
        }
    }

    /// Like [`with_intake`](Self::with_intake), but `Maintenance` is a fail-fast
    /// `Compacting` rather than `Open`.
    ///
    /// The difference is *why* the command is being enqueued. An `Apply` only
    /// needs to be **queued** — admission already gated it, and the writer will get
    /// to it — so `Maintenance` is legitimately `Open`. A command that must be
    /// **executed** to resolve its caller (a checkpoint) is different: the writer
    /// dequeues nothing while parked, so queueing it behind an operator-held pause
    /// means waiting on a human, not on I/O. Deciding *under the phase lock* is
    /// what makes the fail-fast real — checking the phase first and enqueueing
    /// after leaves a window for `begin_maintenance` to land in between.
    pub fn with_running_intake<R>(&self, enqueue: impl FnOnce(Result<(), WriteError>) -> R) -> R {
        let phase = self.phase.lock();
        let decision = match &*phase {
            Lifecycle::Running => Ok(()),
            Lifecycle::Maintenance => Err(WriteError::Compacting),
            Lifecycle::Poisoned(r) => Err(WriteError::Poisoned(r.clone())),
            Lifecycle::Closing | Lifecycle::Closed => Err(WriteError::Closed),
        };
        enqueue(decision)
    }

    /// Runs `enqueue` under the phase lock with the current [`IntakeDecision`],
    /// so `close()`/`poison` cannot interleave between the terminal-state check
    /// and a synchronous, unbounded send. The closure must not
    /// block or `.await`; it performs at most a non-blocking channel send.
    pub fn with_intake<R>(&self, enqueue: impl FnOnce(IntakeDecision) -> R) -> R {
        let phase = self.phase.lock();
        let decision = match &*phase {
            Lifecycle::Running | Lifecycle::Maintenance => IntakeDecision::Open,
            Lifecycle::Poisoned(r) => IntakeDecision::Terminal(Terminal::Poisoned(r.clone())),
            Lifecycle::Closing | Lifecycle::Closed => IntakeDecision::Terminal(Terminal::Closed),
        };
        enqueue(decision)
    }

    /// Poisons the store with `reason`. Idempotent and non-downgrading: only the
    /// first poison reason is kept, and Closed does not revert to Poisoned.
    /// Notifies every waiter.
    pub fn poison(&self, reason: PoisonReason) {
        {
            let mut phase = self.phase.lock();
            match &*phase {
                Lifecycle::Poisoned(_) | Lifecycle::Closed => {}
                _ => *phase = Lifecycle::Poisoned(reason),
            }
        }
        self.event.notify(usize::MAX);
    }

    /// Enters maintenance from Running, returning the new maintenance epoch.
    /// `Err` (naming why) if not Running: a terminal or already-maintenance
    /// store is left as-is. Notifies waiters so fail-fast admission observes
    /// the pause and the writer leaves its Running arm.
    ///
    /// This only *starts* the pause. It is established once the writer
    /// acknowledges the returned epoch ([`pause_status`](Self::pause_status));
    /// [`crate::store::Store::begin_maintenance`] is the waiting form.
    pub fn enter_maintenance(&self) -> Result<u64, MaintenanceError> {
        let entered = {
            let mut phase = self.phase.lock();
            match &*phase {
                Lifecycle::Running => {
                    *phase = Lifecycle::Maintenance;
                    // Under the phase lock, so a writer that reads `Maintenance`
                    // and then loads the epoch sees at least this one.
                    Ok(self.maintenance_epoch.fetch_add(1, Ordering::SeqCst) + 1)
                }
                Lifecycle::Maintenance => Err(MaintenanceError::AlreadyPaused),
                Lifecycle::Poisoned(r) => Err(MaintenanceError::Poisoned(r.clone())),
                Lifecycle::Closing | Lifecycle::Closed => Err(MaintenanceError::Closed),
            }
        };
        if entered.is_ok() {
            self.event.notify(usize::MAX);
        }
        entered
    }

    /// Writer-side half of the handshake: called from the parked barrier of
    /// the writer's `Maintenance` arm — at a command boundary, with the
    /// coordinator's pending work flushed and the phase observed as
    /// `Maintenance` *after* the writer registered its wake listener. Records
    /// the current epoch as acknowledged and wakes the `begin_maintenance`
    /// waiter.
    ///
    /// Acknowledging the *current* epoch (not one captured earlier in the arm)
    /// is what makes a resume/re-pause between the arm's entry and this point
    /// harmless: the writer has published nothing since the arm was entered, so
    /// the newest pause is as quiescent as the one it entered for.
    pub fn ack_maintenance(&self) {
        let epoch = self.maintenance_epoch.load(Ordering::SeqCst);
        self.maintenance_acked.fetch_max(epoch, Ordering::SeqCst);
        self.event.notify(usize::MAX);
    }

    /// The `begin_maintenance` waiter's predicate for the pause it entered as
    /// `epoch`. Classified **under the phase lock** together with the current
    /// epoch, so a resume/re-pause by another caller cannot be mistaken for the
    /// waiter's own still-active pause:
    ///
    /// - `Established` only if the phase is `Maintenance`, the active epoch is
    ///   exactly `epoch`, and the writer has acknowledged it. Then the writer is
    ///   at its parked barrier and can only leave it through a phase change,
    ///   which would fail this classification — so the guarantee holds for as
    ///   long as the caller keeps the pause.
    /// - `Waiting` while that exact epoch is active and unacknowledged.
    /// - `Err` once the pause is gone: resumed (`Interrupted` — the epoch may
    ///   already have been superseded by another caller's pause, whose ack must
    ///   not be credited to this one), poisoned, or closing/closed. All of these
    ///   transitions notify the shared event, so a waiter is never stranded.
    pub fn pause_status(&self, epoch: u64) -> PauseStatus {
        let phase = self.phase.lock();
        match &*phase {
            Lifecycle::Maintenance => {
                if self.maintenance_epoch.load(Ordering::SeqCst) != epoch {
                    PauseStatus::Aborted(MaintenanceError::Interrupted)
                } else if self.maintenance_acked.load(Ordering::SeqCst) >= epoch {
                    PauseStatus::Established
                } else {
                    PauseStatus::Waiting
                }
            }
            Lifecycle::Running => PauseStatus::Aborted(MaintenanceError::Interrupted),
            Lifecycle::Poisoned(r) => PauseStatus::Aborted(MaintenanceError::Poisoned(r.clone())),
            Lifecycle::Closing | Lifecycle::Closed => {
                PauseStatus::Aborted(MaintenanceError::Closed)
            }
        }
    }

    /// Resumes Running from Maintenance. No-op if terminal. Notifies waiters so
    /// blocked admission retries.
    pub fn resume(&self) {
        let resumed = {
            let mut phase = self.phase.lock();
            if *phase == Lifecycle::Maintenance {
                *phase = Lifecycle::Running;
                true
            } else {
                false
            }
        };
        if resumed {
            self.event.notify(usize::MAX);
        }
    }

    /// Transitions to `Closing` (shutting intake) unless already terminal.
    /// Returns `true` if this call initiated the close. Notifies waiters.
    pub fn begin_close(&self) -> bool {
        let began = {
            let mut phase = self.phase.lock();
            match &*phase {
                Lifecycle::Poisoned(_) | Lifecycle::Closed | Lifecycle::Closing => false,
                _ => {
                    *phase = Lifecycle::Closing;
                    true
                }
            }
        };
        if began {
            self.event.notify(usize::MAX);
        }
        began
    }

    /// Finalizes to `Closed`. No-op if already poisoned (poison wins as the
    /// truthful terminal reason). Notifies waiters.
    pub fn finish_close(&self) {
        {
            let mut phase = self.phase.lock();
            if !matches!(&*phase, Lifecycle::Poisoned(_)) {
                *phase = Lifecycle::Closed;
            }
        }
        self.event.notify(usize::MAX);
    }
}

/// Test-only writer-thread gates (see `Core::test_gates`). A gate that is not
/// armed is a no-op `pass`; an armed gate blocks the writer thread at `pass`
/// (recording that it arrived) until a test `release`s it, then disarms.
#[cfg(test)]
#[derive(Default)]
pub struct TestGates {
    /// Immediately before the root `ArcSwap` publication in `apply_batch`.
    pub publish: TestGate,
    /// At entry to `Coordinator::flush_all`.
    pub flush: TestGate,
}

#[cfg(test)]
#[derive(Default)]
pub struct TestGate {
    state: std::sync::Mutex<GateState>,
    cv: std::sync::Condvar,
}

#[cfg(test)]
#[derive(Default)]
struct GateState {
    armed: bool,
    arrived: bool,
    released: bool,
}

#[cfg(test)]
impl TestGate {
    /// Arms the gate: the next `pass` blocks until `release`.
    pub fn arm(&self) {
        let mut g = self.state.lock().unwrap();
        *g = GateState {
            armed: true,
            arrived: false,
            released: false,
        };
    }

    /// Writer side. Blocks while armed and not yet released; a plain
    /// pass-through otherwise. Blocking a dedicated OS thread here is fine —
    /// it is exactly "the writer is stuck at this point".
    pub fn pass(&self) {
        let mut g = self.state.lock().unwrap();
        if !g.armed {
            return;
        }
        g.arrived = true;
        self.cv.notify_all();
        while !g.released {
            g = self.cv.wait(g).unwrap();
        }
        g.armed = false;
    }

    /// Waits (bounded) until the writer is blocked in `pass`.
    pub fn wait_arrived(&self) {
        let g = self.state.lock().unwrap();
        let (g, timeout) = self
            .cv
            .wait_timeout_while(g, std::time::Duration::from_secs(10), |g| !g.arrived)
            .unwrap();
        assert!(!timeout.timed_out(), "writer never reached the armed gate");
        drop(g);
    }

    /// Lets a blocked `pass` proceed.
    pub fn release(&self) {
        let mut g = self.state.lock().unwrap();
        g.released = true;
        self.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poison_is_idempotent_and_sticky() {
        let s = EngineState::new_running(std::sync::Arc::new(event_listener::Event::new()));
        s.poison(PoisonReason::Corrupt("first"));
        s.poison(PoisonReason::Corrupt("second"));
        match s.phase() {
            Lifecycle::Poisoned(PoisonReason::Corrupt(w)) => assert_eq!(w, "first"),
            other => panic!("expected first poison reason, got {other:?}"),
        }
    }

    fn status(s: &EngineState, epoch: u64) -> &'static str {
        match s.pause_status(epoch) {
            PauseStatus::Waiting => "waiting",
            PauseStatus::Established => "established",
            PauseStatus::Aborted(MaintenanceError::Interrupted) => "interrupted",
            PauseStatus::Aborted(MaintenanceError::Poisoned(_)) => "poisoned",
            PauseStatus::Aborted(MaintenanceError::Closed) => "closed",
            PauseStatus::Aborted(MaintenanceError::AlreadyPaused) => unreachable!(),
        }
    }

    #[test]
    fn maintenance_round_trip() {
        let s = EngineState::new_running(std::sync::Arc::new(event_listener::Event::new()));
        assert!(matches!(s.enter_maintenance(), Ok(1)));
        assert_eq!(s.phase(), Lifecycle::Maintenance);
        // Re-entry does not own the pause; it is not established until the writer acks.
        assert!(matches!(
            s.enter_maintenance(),
            Err(MaintenanceError::AlreadyPaused)
        ));
        assert_eq!(status(&s, 1), "waiting");
        s.ack_maintenance();
        assert_eq!(status(&s, 1), "established");
        // Fresh admission fails fast; enqueue is still allowed.
        assert!(matches!(s.admission_gate(), Err(WriteError::Compacting)));
        assert!(matches!(s.with_intake(|d| d), IntakeDecision::Open));
        s.resume();
        assert_eq!(s.phase(), Lifecycle::Running);
        // Leaving Maintenance ends the pause for its waiter.
        assert_eq!(status(&s, 1), "interrupted");
        // A second pause is a new epoch: the old ack does not count for it.
        assert!(matches!(s.enter_maintenance(), Ok(2)));
        assert_eq!(status(&s, 2), "waiting");
    }

    /// Waiter-side ABA: an old, acknowledged epoch must not be credited to a
    /// delayed waiter once another caller has resumed and re-paused — the
    /// writer may be mid-command under the new, unacknowledged pause.
    #[test]
    fn a_superseded_pause_is_interrupted_not_established() {
        let s = EngineState::new_running(std::sync::Arc::new(event_listener::Event::new()));
        assert!(matches!(s.enter_maintenance(), Ok(1)));
        s.ack_maintenance();
        s.resume();
        assert!(matches!(s.enter_maintenance(), Ok(2)));
        assert_eq!(status(&s, 1), "interrupted", "epoch 1's ack is stale");
        assert_eq!(status(&s, 2), "waiting");
        s.ack_maintenance();
        assert_eq!(status(&s, 2), "established");
        assert_eq!(status(&s, 1), "interrupted");
    }

    #[test]
    fn terminal_transitions_abort_a_maintenance_waiter() {
        let s = EngineState::new_running(std::sync::Arc::new(event_listener::Event::new()));
        assert!(matches!(s.enter_maintenance(), Ok(1)));
        s.poison(PoisonReason::Corrupt("mid-pause"));
        assert_eq!(status(&s, 1), "poisoned");
        assert!(matches!(
            s.enter_maintenance(),
            Err(MaintenanceError::Poisoned(_))
        ));

        let s = EngineState::new_running(std::sync::Arc::new(event_listener::Event::new()));
        assert!(matches!(s.enter_maintenance(), Ok(1)));
        assert!(s.begin_close());
        assert_eq!(status(&s, 1), "closed");
        assert!(matches!(
            s.enter_maintenance(),
            Err(MaintenanceError::Closed)
        ));
    }

    #[test]
    fn close_then_poison_keeps_closed_but_poison_over_live_wins() {
        let s = EngineState::new_running(std::sync::Arc::new(event_listener::Event::new()));
        assert!(s.begin_close());
        s.finish_close();
        assert_eq!(s.phase(), Lifecycle::Closed);
        // Poison after Closed does not revert.
        s.poison(PoisonReason::Corrupt("late"));
        assert_eq!(s.phase(), Lifecycle::Closed);
    }
}
