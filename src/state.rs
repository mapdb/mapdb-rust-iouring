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

use crate::error::{PoisonReason, Terminal, WriteError};
use event_listener::Event;
use parking_lot::Mutex;
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

    /// Enters maintenance from Running. No-op if not Running (a terminal or
    /// already-maintenance store is left as-is). Notifies waiters so fail-fast
    /// admission observes the pause.
    pub fn enter_maintenance(&self) -> bool {
        let entered = {
            let mut phase = self.phase.lock();
            if *phase == Lifecycle::Running {
                *phase = Lifecycle::Maintenance;
                true
            } else {
                false
            }
        };
        if entered {
            self.event.notify(usize::MAX);
        }
        entered
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

    #[test]
    fn maintenance_round_trip() {
        let s = EngineState::new_running(std::sync::Arc::new(event_listener::Event::new()));
        assert!(s.enter_maintenance());
        assert_eq!(s.phase(), Lifecycle::Maintenance);
        // Fresh admission fails fast; enqueue is still allowed.
        assert!(matches!(s.admission_gate(), Err(WriteError::Compacting)));
        assert!(matches!(s.with_intake(|d| d), IntakeDecision::Open));
        s.resume();
        assert_eq!(s.phase(), Lifecycle::Running);
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
