//! Bounded weighted admission.
//!
//! Two accounting shapes:
//!
//! - [`Budget`] — *releasable* weighted capacity for queue slots, command
//!   bytes, and dirty pages. A [`Reservation`] holds its charge and returns it
//!   on drop or when the writer reconciles it down to the actual cost. Capacity
//!   releases only at the backend's completion frontier for the dirty-page
//!   dimension (invariant 5), modelled here as an explicit `release`.
//! - [`WalMeter`] — *monotonic* WAL-byte accounting. WAL bytes
//!   never fall below the logical append frontier, so this is not a semaphore:
//!   `reserve` compares `wal_reserved_bytes` against `max_wal_bytes` and returns
//!   `StoreFull` (never `Overloaded`) because waiting cannot help.
//!
//! Both wake blocked waiters through the store's shared [`event_listener::Event`]
//! so a release and a poison use one wakeup path (invariant 6).

use crate::error::WriteError;
use crate::state::EngineState;
use event_listener::Event;
use parking_lot::Mutex;
use std::sync::Arc;

/// Reconciliation found the actual cost above the admitted worst case — a codec
/// bug by construction. The caller poisons.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReconcileOverflow;

/// The admission cost of one write.
///
/// `worst_case_pages` and `wal_worst_case_bytes` are admission-time **estimates**
/// (not hard upper bounds — the no-merge builder makes a tight provable bound
/// impractical); the writer reconciles them to the exact
/// figures in *either* direction before publication, and an upward WAL-byte
/// reconcile that would exceed `max_wal_bytes` fails pre-publication with
/// `StoreFull`. A caller pre-flighting with these
/// figures should treat `StoreFull` as possible even when its estimate fit.
/// Encoded bytes and op count are exact at call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WriteCost {
    pub encoded_bytes: u64,
    pub op_count: u32,
    pub worst_case_pages: u32,
    pub wal_worst_case_bytes: u64,
}

/// Static admission capacities.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionLimits {
    pub max_queue_slots: u32,
    pub max_command_bytes: u64,
    pub max_dirty_pages: u32,
    pub max_ops_per_batch: u32,
}

impl Default for AdmissionLimits {
    fn default() -> Self {
        AdmissionLimits {
            max_queue_slots: 256,
            max_command_bytes: 64 * 1024 * 1024,
            max_dirty_pages: 4096,
            max_ops_per_batch: 4096,
        }
    }
}

struct Used {
    slots: u32,
    bytes: u64,
    pages: u32,
    /// Tickets of currently-awaiting `reserve` callers, ascending. Only the head
    /// (minimum ticket) may consume capacity, giving weighted **FIFO** admission
    /// so a large request at the head cannot be starved by a stream of small
    /// ones racing each release. A `BTreeSet` gives O(log n)
    /// head lookup and cancellation removal.
    waiters: std::collections::BTreeSet<u64>,
}

/// Releasable weighted admission budget (slots, bytes, dirty pages).
pub struct Budget {
    limits: AdmissionLimits,
    used: Mutex<Used>,
    notify: Arc<Event>,
    /// Monotonic ticket source for FIFO waiter ordering.
    next_ticket: std::sync::atomic::AtomicU64,
}

impl Budget {
    pub fn new(limits: AdmissionLimits, notify: Arc<Event>) -> Arc<Self> {
        Arc::new(Budget {
            limits,
            used: Mutex::new(Used {
                slots: 0,
                bytes: 0,
                pages: 0,
                waiters: std::collections::BTreeSet::new(),
            }),
            notify,
            next_ticket: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn limits(&self) -> AdmissionLimits {
        self.limits
    }

    /// Returns `Err(CostExceedsLimit)` if `cost` can never be admitted (exceeds
    /// a hard cap, or its worst-case pages exceed *half* the dirty budget — a
    /// legal batch is at most half, so reconciliation always has room ahead of
    /// new admission).
    fn check_hard_caps(&self, cost: &WriteCost) -> Result<(), WriteError> {
        if cost.op_count > self.limits.max_ops_per_batch
            || cost.encoded_bytes > self.limits.max_command_bytes
            || u64::from(cost.worst_case_pages) * 2 > u64::from(self.limits.max_dirty_pages)
        {
            return Err(WriteError::CostExceedsLimit);
        }
        Ok(())
    }

    /// Grants a reservation if `cost` fits the remaining capacity, mutating
    /// `used`. Saturating adds so extreme configured limits never wrap into a
    /// false "fits" (note 3).
    fn grant_locked(self: &Arc<Self>, used: &mut Used, cost: WriteCost) -> Option<Reservation> {
        let fits = used.slots < self.limits.max_queue_slots
            && used.bytes.saturating_add(cost.encoded_bytes) <= self.limits.max_command_bytes
            && used.pages.saturating_add(cost.worst_case_pages) <= self.limits.max_dirty_pages;
        if !fits {
            return None;
        }
        used.slots += 1;
        used.bytes += cost.encoded_bytes;
        used.pages += cost.worst_case_pages;
        Some(Reservation {
            budget: Arc::clone(self),
            bytes: cost.encoded_bytes,
            pages: cost.worst_case_pages,
            active: true,
            wal: None,
        })
    }

    /// Non-blocking reservation. `CostExceedsLimit` is permanent; `Overloaded`
    /// is transient (retry / await). **Yields to the FIFO**: if any `reserve`
    /// caller is waiting, a non-blocking grab returns `Overloaded` rather than
    /// jumping ahead of the head.
    pub fn try_reserve(self: &Arc<Self>, cost: WriteCost) -> Result<Reservation, WriteError> {
        self.check_hard_caps(&cost)?;
        let mut used = self.used.lock();
        if !used.waiters.is_empty() {
            return Err(WriteError::Overloaded);
        }
        self.grant_locked(&mut used, cost)
            .ok_or(WriteError::Overloaded)
    }

    /// Awaiting reservation with FIFO fairness. The caller takes a ticket and
    /// only consumes capacity once it is the **head** (oldest waiter), so a
    /// large request cannot be starved by later small ones. Returns
    /// `CostExceedsLimit` immediately (waiting cannot satisfy a hard cap) and the
    /// terminal error if the store poisons/closes while waiting. On any exit —
    /// grant, error, or cancellation (dropped future) — the ticket is removed and
    /// the next head woken.
    pub async fn reserve(
        self: &Arc<Self>,
        cost: WriteCost,
        state: &EngineState,
    ) -> Result<Reservation, WriteError> {
        self.check_hard_caps(&cost)?;
        let ticket = self
            .next_ticket
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.used.lock().waiters.insert(ticket);
        let _guard = WaiterGuard {
            budget: self,
            ticket,
        };
        loop {
            // Fail fast on terminal or maintenance *before* listening.
            state.admission_gate()?;
            {
                let mut used = self.used.lock();
                if used.waiters.iter().next() == Some(&ticket) {
                    if let Some(r) = self.grant_locked(&mut used, cost) {
                        return Ok(r);
                    }
                }
            }
            // Register, then re-check both predicates before awaiting (closes
            // the lost-wakeup race for capacity, head-change, and state change).
            let listener = state.listen();
            state.admission_gate()?;
            {
                let mut used = self.used.lock();
                if used.waiters.iter().next() == Some(&ticket) {
                    if let Some(r) = self.grant_locked(&mut used, cost) {
                        return Ok(r);
                    }
                }
            }
            listener.await;
        }
    }

    fn release(&self, bytes: u64, pages: u32) {
        {
            let mut used = self.used.lock();
            // T3.4: a release exceeding the charge means double-release or a
            // frontier/reservation mismatch — loud in tests, clamped in release
            // (silently inflating capacity is the failure the assert surfaces).
            debug_assert!(used.slots >= 1, "slot release underflow");
            debug_assert!(used.bytes >= bytes, "byte release underflow");
            debug_assert!(used.pages >= pages, "page release underflow");
            used.slots = used.slots.saturating_sub(1);
            used.bytes = used.bytes.saturating_sub(bytes);
            used.pages = used.pages.saturating_sub(pages);
        }
        self.notify.notify(usize::MAX);
    }

    fn release_pages_only(&self, pages: u32) {
        {
            let mut used = self.used.lock();
            debug_assert!(used.pages >= pages, "page release underflow");
            used.pages = used.pages.saturating_sub(pages);
        }
        self.notify.notify(usize::MAX);
    }

    /// Releases a queue slot and command bytes **without** touching the dirty
    /// pages, used when a published batch's page charge is handed to the flush
    /// frontier (invariant 5): slot/bytes free at publication, pages free at
    /// flush. See [`Reservation::hand_pages_to_frontier`].
    fn release_slot_bytes(&self, bytes: u64) {
        {
            let mut used = self.used.lock();
            debug_assert!(used.slots >= 1, "slot release underflow");
            debug_assert!(used.bytes >= bytes, "byte release underflow");
            used.slots = used.slots.saturating_sub(1);
            used.bytes = used.bytes.saturating_sub(bytes);
        }
        self.notify.notify(usize::MAX);
    }

    /// Releases `pages` dirty-page charges at the backend completion frontier
    /// (Direct flush / WAL commit). Public counterpart to the internal
    /// reconciliation release, so the flush coordinator can free pinned pages'
    /// budget once they are durably readable.
    pub fn release_pages(&self, pages: u32) {
        self.release_pages_only(pages);
    }

    /// Charges `extra` additional dirty pages (saturating). Used when a batch's
    /// *actual* private-page count exceeds its admission worst-case estimate —
    /// the M0 page estimate (`op_count + depth`) is not a guaranteed upper bound
    /// for a many-op batch over a multi-level tree, so the writer charges the
    /// difference rather than poisoning a legal write. This is a soft, transient
    /// overshoot of the dirty budget that the next flush releases; the
    /// admission-time half-budget cap still governs *estimated* fairness.
    fn charge_pages(&self, extra: u32) {
        let mut used = self.used.lock();
        used.pages = used.pages.saturating_add(extra);
    }

    /// Snapshot of current usage (for metrics/tests): (slots, bytes, pages).
    pub fn usage(&self) -> (u32, u64, u32) {
        let u = self.used.lock();
        (u.slots, u.bytes, u.pages)
    }

    /// Number of awaiting FIFO reservers (for tests — lets a test wait until a
    /// spawned `reserve_write` has actually parked before releasing capacity,
    /// instead of racing a timer).
    #[cfg(test)]
    pub(crate) fn waiter_count(&self) -> usize {
        self.used.lock().waiters.len()
    }
}

/// Removes a `reserve` caller's FIFO ticket on any exit (grant, error, or
/// cancellation via a dropped future) and wakes the next head so it can proceed.
struct WaiterGuard<'a> {
    budget: &'a Arc<Budget>,
    ticket: u64,
}

impl Drop for WaiterGuard<'_> {
    fn drop(&mut self) {
        self.budget.used.lock().waiters.remove(&self.ticket);
        // Wake the new head (and any try_reserve that can now proceed).
        self.budget.notify.notify(usize::MAX);
    }
}

/// A held admission charge. Releases the full charge on drop; the writer may
/// [`reconcile`](Reservation::reconcile) the dirty-page charge down to actual
/// before publication.
pub struct Reservation {
    budget: Arc<Budget>,
    bytes: u64,
    pages: u32,
    active: bool,
    /// StoreWal only: the WAL-byte reservation that rides along with the dirty
    /// budget so cancellation/drop at any admission stage releases both charges
    /// through one path (invariant 18). The writer [`take_wal`](Self::take_wal)s
    /// it at publication, reconciles it to the exact record length, and hands it
    /// to the WAL coordinator to `commit` at the durable frontier.
    wal: Option<WalReservation>,
}

impl Reservation {
    /// The dirty pages currently reserved.
    pub fn pages(&self) -> u32 {
        self.pages
    }

    /// Attaches a WAL-byte reservation (StoreWal admission). Set once, right
    /// after the dirty-page reservation is granted.
    pub fn attach_wal(&mut self, wal: WalReservation) {
        self.wal = Some(wal);
    }

    /// Whether this reservation carries a WAL-byte reservation (i.e. StoreWal).
    pub fn has_wal(&self) -> bool {
        self.wal.is_some()
    }

    /// Takes the WAL-byte reservation out for the writer to reconcile and hand to
    /// the coordinator. After this the reservation no longer releases WAL bytes on
    /// drop (the coordinator owns them).
    pub fn take_wal(&mut self) -> Option<WalReservation> {
        self.wal.take()
    }

    /// Reduces the dirty-page charge to `actual`, returning the excess to the
    /// budget immediately (so a concurrently blocked admission can proceed). An
    /// `actual` above the current reservation is rejected. Retained for its unit
    /// test; the live dirty-page path uses the bidirectional
    /// [`reconcile_to_actual`](Self::reconcile_to_actual), and WAL-byte
    /// reconciliation is the bidirectional [`WalReservation::reconcile`] (M4 made
    /// both soft — the estimate is not a hard bound).
    pub fn reconcile(&mut self, actual: u32) -> Result<(), ReconcileOverflow> {
        if actual > self.pages {
            return Err(ReconcileOverflow);
        }
        let excess = self.pages - actual;
        if excess > 0 {
            self.budget.release_pages_only(excess);
            self.pages = actual;
        }
        Ok(())
    }

    /// Reconciles the dirty-page charge to the batch's *actual* private-page
    /// count, in **either** direction: releases the excess if `actual` is below
    /// the estimate (waking blocked admission), or charges the difference if it
    /// is above. Unlike [`reconcile`](Self::reconcile) this never fails — the
    /// M0 page estimate is not a guaranteed upper bound for a many-op batch over
    /// a multi-level tree, so a legal write must not poison. After this the
    /// reservation holds exactly `actual`, which
    /// [`hand_pages_to_frontier`](Self::hand_pages_to_frontier) returns for the
    /// flush frontier to release.
    pub fn reconcile_to_actual(&mut self, actual: u32) {
        if actual <= self.pages {
            let excess = self.pages - actual;
            if excess > 0 {
                self.budget.release_pages_only(excess);
            }
        } else {
            self.budget.charge_pages(actual - self.pages);
        }
        self.pages = actual;
    }

    /// Consumes the reservation without releasing the slot/bytes back — used
    /// when the charge has been transferred to a durable frontier that will
    /// release it later. (M0 has no such frontier; retained for M3/M4.)
    #[allow(dead_code)]
    pub fn forget(mut self) {
        self.active = false;
    }

    /// Hands the (already-reconciled) dirty-page charge to the flush frontier:
    /// releases the queue slot and command bytes **now** (the command is done),
    /// keeps the `pages` charged in the budget, and disarms `Drop` so nothing is
    /// double-released. Returns the page count the frontier must later release
    /// via [`Budget::release_pages`] once the pages are durable (invariant 5).
    pub fn hand_pages_to_frontier(mut self) -> u32 {
        let pages = self.pages;
        self.budget.release_slot_bytes(self.bytes);
        self.active = false;
        pages
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.active {
            self.budget.release(self.bytes, self.pages);
        }
    }
}

// ---------------------------------------------------------------------------
// WAL-byte metering (monotonic)
// ---------------------------------------------------------------------------

struct WalUsed {
    /// Logical append frontier: WAL header + all accepted/encoded records.
    logical: u64,
    /// Outstanding worst-case reservations on top of `logical`.
    reserved: u64,
}

/// Monotonic WAL-byte accounting for StoreWal admission.
///
/// `reserved = logical + Σ outstanding worst-case`. Admission compares
/// `logical + reserved_extra + cost` against `max_wal_bytes`; over-limit is
/// `StoreFull` (pre-publication, consumes no txid), never `Overloaded`.
pub struct WalMeter {
    max_wal_bytes: u64,
    used: Mutex<WalUsed>,
    notify: Arc<Event>,
}

impl WalMeter {
    /// `header_bytes` seeds the logical frontier (the limit includes the WAL
    /// header).
    pub fn new(max_wal_bytes: u64, header_bytes: u64, notify: Arc<Event>) -> Arc<Self> {
        Arc::new(WalMeter {
            max_wal_bytes,
            used: Mutex::new(WalUsed {
                logical: header_bytes,
                reserved: 0,
            }),
            notify,
        })
    }

    /// Reserves `worst_case` WAL bytes for a batch. `StoreFull` if the reserved
    /// frontier would exceed `max_wal_bytes` (saturating arithmetic).
    pub fn reserve(self: &Arc<Self>, worst_case: u64) -> Result<WalReservation, WriteError> {
        let mut used = self.used.lock();
        let projected = used
            .logical
            .saturating_add(used.reserved)
            .saturating_add(worst_case);
        if projected > self.max_wal_bytes {
            return Err(WriteError::StoreFull);
        }
        used.reserved = used.reserved.saturating_add(worst_case);
        Ok(WalReservation {
            meter: Arc::clone(self),
            outstanding: worst_case,
        })
    }

    /// Rebases the logical frontier onto a freshly rotated WAL segment (M6 Phase
    /// B checkpoint step 6): the retired segment's bytes no longer count against
    /// `max_wal_bytes` — they are unlinked, and their space is the retained
    /// generation's problem, not admission's.
    ///
    /// **Outstanding reservations are preserved.** Batches admitted before the
    /// checkpoint but not yet applied still hold their charge in `reserved` and
    /// will append their records into the *new* segment, so dropping their charge
    /// here would let the meter under-count real bytes.
    /// The rebase can only ever make room, never take it away.
    pub fn rebase(&self, logical: u64) {
        let mut used = self.used.lock();
        used.logical = logical;
        // A blocked `StoreFull` caller is *not* woken: `reserve` fails fast rather
        // than waiting (WAL bytes are monotonic within a segment), so there is no
        // waiter to wake — a caller retries on its own and now fits.
    }

    /// Current `(logical, reserved_total)` where `reserved_total = logical +
    /// outstanding`. For metrics/tests.
    pub fn usage(&self) -> (u64, u64) {
        let u = self.used.lock();
        (u.logical, u.logical.saturating_add(u.reserved))
    }
}

/// A held WAL-byte reservation.
///
/// `outstanding` is the reservation this handle currently holds against
/// `WalMeter::reserved` — the admitted worst case until [`reconcile`] lowers it
/// to the exact encoded length. Drop releases whatever remains outstanding
/// unless [`commit`] advanced it into the logical frontier.
///
/// [`reconcile`]: WalReservation::reconcile
/// [`commit`]: WalReservation::commit
pub struct WalReservation {
    meter: Arc<WalMeter>,
    outstanding: u64,
}

impl WalReservation {
    /// Reconciles the reservation to the **exact** encoded record length in *either*
    /// direction and returns a
    /// [`ReconciledWalReservation`]. Down: releases the excess immediately
    /// (notifying any blocked admission — the M4 exit test requires this to be
    /// visible to a concurrently blocked waiter). Up: the admission estimate was
    /// below actual (the no-merge builder makes the estimate a soft figure, not a
    /// hard bound), so it **charges the difference, but only if
    /// the batch still fits `max_wal_bytes`**; otherwise it releases the whole
    /// reservation and fails with `StoreFull` — a healthy **pre-publication**
    /// fail-fast (invariant 18), never a poison.
    ///
    /// Consuming `self` and returning a distinct type makes it **impossible to
    /// commit an unreconciled reservation in any build**: committing the
    /// estimate would desync `WalMeter.logical` from the real append frontier.
    pub fn reconcile(mut self, actual: u64) -> Result<ReconciledWalReservation, WriteError> {
        {
            let mut used = self.meter.used.lock();
            if actual <= self.outstanding {
                used.reserved = used.reserved.saturating_sub(self.outstanding - actual);
            } else {
                let extra = actual - self.outstanding;
                let projected = used
                    .logical
                    .saturating_add(used.reserved)
                    .saturating_add(extra);
                if projected > self.meter.max_wal_bytes {
                    // Won't fit: release our whole outstanding charge and fail
                    // pre-publication. `self.outstanding = 0` disarms `Drop`.
                    used.reserved = used.reserved.saturating_sub(self.outstanding);
                    drop(used);
                    self.meter.notify.notify(usize::MAX);
                    self.outstanding = 0;
                    return Err(WriteError::StoreFull);
                }
                used.reserved = used.reserved.saturating_add(extra);
            }
        }
        self.meter.notify.notify(usize::MAX);
        // Transfer ownership of the (now exact) outstanding bytes to the
        // reconciled handle; disarm this handle's release.
        let reconciled = ReconciledWalReservation {
            meter: Arc::clone(&self.meter),
            outstanding: actual,
            committed: false,
        };
        self.outstanding = 0;
        Ok(reconciled)
    }
}

impl Drop for WalReservation {
    fn drop(&mut self) {
        if self.outstanding > 0 {
            {
                let mut used = self.meter.used.lock();
                used.reserved = used.reserved.saturating_sub(self.outstanding);
            }
            self.meter.notify.notify(usize::MAX);
        }
    }
}

/// A WAL-byte reservation reconciled to its exact encoded length. Only this type
/// can [`commit`](Self::commit), so an unreconciled reservation can never advance
/// the logical frontier (a codec-safety typestate). Drop before commit
/// releases the reserved bytes.
pub struct ReconciledWalReservation {
    meter: Arc<WalMeter>,
    outstanding: u64,
    committed: bool,
}

impl ReconciledWalReservation {
    /// Advances the logical frontier by the reconciled length and clears the
    /// reservation. Called once the record is durably appended.
    pub fn commit(mut self) {
        {
            let mut used = self.meter.used.lock();
            used.reserved = used.reserved.saturating_sub(self.outstanding);
            used.logical = used.logical.saturating_add(self.outstanding);
        }
        self.committed = true;
        self.meter.notify.notify(usize::MAX);
    }
}

impl Drop for ReconciledWalReservation {
    fn drop(&mut self) {
        if !self.committed {
            {
                let mut used = self.meter.used.lock();
                used.reserved = used.reserved.saturating_sub(self.outstanding);
            }
            self.meter.notify.notify(usize::MAX);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(pages: u32) -> Arc<Budget> {
        Budget::new(
            AdmissionLimits {
                max_queue_slots: 4,
                max_command_bytes: 1_000,
                max_dirty_pages: pages,
                max_ops_per_batch: 100,
            },
            Arc::new(Event::new()),
        )
    }

    fn cost(bytes: u64, pages: u32) -> WriteCost {
        WriteCost {
            encoded_bytes: bytes,
            op_count: 1,
            worst_case_pages: pages,
            wal_worst_case_bytes: 0,
        }
    }

    #[test]
    fn batch_over_half_dirty_budget_is_permanent() {
        let b = budget(10);
        // 6 pages * 2 > 10 -> CostExceedsLimit.
        assert!(matches!(
            b.try_reserve(cost(1, 6)),
            Err(WriteError::CostExceedsLimit)
        ));
    }

    #[test]
    fn reservation_releases_on_drop() {
        let b = budget(10);
        {
            let _r = b.try_reserve(cost(100, 4)).unwrap();
            assert_eq!(b.usage(), (1, 100, 4));
        }
        assert_eq!(b.usage(), (0, 0, 0));
    }

    #[test]
    fn reconcile_returns_excess() {
        let b = budget(10);
        let mut r = b.try_reserve(cost(100, 4)).unwrap();
        r.reconcile(1).unwrap();
        assert_eq!(b.usage(), (1, 100, 1));
        assert!(r.reconcile(5).is_err(), "cannot grow past reservation");
    }

    #[test]
    fn reconcile_to_actual_both_directions() {
        // Down: releases the excess.
        let b = budget(100);
        let mut r = b.try_reserve(cost(10, 8)).unwrap();
        r.reconcile_to_actual(3);
        assert_eq!(b.usage(), (1, 10, 3), "excess released");
        // Up: charges the overshoot (a legal batch that dirtied more than its
        // estimate) instead of failing.
        let mut r2 = b.try_reserve(cost(10, 2)).unwrap();
        assert_eq!(b.usage(), (2, 20, 5));
        r2.reconcile_to_actual(9);
        assert_eq!(b.usage(), (2, 20, 12), "overshoot charged, no poison");
        // The frontier hand-off returns the reconciled actual for later release.
        assert_eq!(r2.hand_pages_to_frontier(), 9);
        // Slot + bytes freed now; the 9 pages stay charged until the frontier
        // releases them.
        assert_eq!(b.usage(), (1, 10, 12));
        b.release_pages(9);
        drop(r);
        assert_eq!(b.usage(), (0, 0, 0));
    }

    #[test]
    fn reserve_is_fifo_large_head_not_starved() {
        // A large awaiting reservation at the head must be served before later
        // small non-blocking grabs, which yield to it.
        let event = Arc::new(Event::new());
        let state = Arc::new(EngineState::new_running(Arc::clone(&event)));
        let b = Budget::new(
            AdmissionLimits {
                max_queue_slots: 20,
                max_command_bytes: 1_000_000,
                max_dirty_pages: 20,
                max_ops_per_batch: 100,
            },
            Arc::clone(&event),
        );
        // Hold 16 pages (two 8-page reservations), leaving 4 free.
        let hold1 = b.try_reserve(cost(1, 8)).unwrap();
        let _hold2 = b.try_reserve(cost(1, 8)).unwrap();

        // Thread A awaits a 5-page reservation → becomes the FIFO head; 5 > 4
        // free, so it blocks.
        let (ba, sa) = (Arc::clone(&b), Arc::clone(&state));
        let a = std::thread::spawn(move || {
            futures_lite::future::block_on(ba.reserve(cost(1, 5), &sa)).map(|r| r.pages())
        });
        for _ in 0..2000 {
            if b.waiter_count() == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(
            b.waiter_count(),
            1,
            "head reservation must be enqueued and waiting"
        );

        // A small non-blocking grab yields to the head even though 1 ≤ 4 free.
        assert!(
            matches!(b.try_reserve(cost(1, 1)), Err(WriteError::Overloaded)),
            "try_reserve must yield to the waiting FIFO head, not jump ahead"
        );

        // Free capacity: the head now fits and is served.
        drop(hold1);
        let served = a.join().unwrap();
        assert_eq!(served.unwrap(), 5, "large head served once capacity freed");
    }

    #[test]
    fn slots_saturate_to_overloaded() {
        let b = budget(1000);
        let _a = b.try_reserve(cost(1, 1)).unwrap();
        let _b = b.try_reserve(cost(1, 1)).unwrap();
        let _c = b.try_reserve(cost(1, 1)).unwrap();
        let _d = b.try_reserve(cost(1, 1)).unwrap();
        assert!(matches!(
            b.try_reserve(cost(1, 1)),
            Err(WriteError::Overloaded)
        ));
    }

    #[test]
    fn wal_meter_reports_store_full_not_overloaded() {
        let m = WalMeter::new(1000, 100, Arc::new(Event::new()));
        let r1 = m.reserve(800).unwrap();
        assert!(matches!(m.reserve(200), Err(WriteError::StoreFull)));
        drop(r1);
        // After releasing the reservation, room returns.
        let _r2 = m.reserve(200).unwrap();
    }

    #[test]
    fn wal_reconcile_release_visible_to_next_reserve() {
        // logical=100. Reserve 500 (total 600). A second 500 would be 1100>1000
        // → StoreFull. Reconcile the first down to 100, releasing 400; now the
        // second 500 fits (100+100+500=700). This is the
        // "reconciliation release visible to a blocked admission" case.
        let m = WalMeter::new(1000, 100, Arc::new(Event::new()));
        let r1 = m.reserve(500).unwrap();
        assert!(matches!(m.reserve(500), Err(WriteError::StoreFull)));
        let _r1 = r1.reconcile(100).unwrap(); // keep the reconciled handle alive
        assert_eq!(
            m.usage(),
            (100, 200),
            "excess released at reconcile, not commit"
        );
        let _r2 = m.reserve(500).unwrap();
    }

    #[test]
    fn wal_commit_advances_logical_by_actual() {
        let m = WalMeter::new(10_000, 100, Arc::new(Event::new()));
        let r = m.reserve(500).unwrap();
        let r = r.reconcile(120).unwrap();
        r.commit();
        assert_eq!(m.usage(), (220, 220));
    }

    #[test]
    fn wal_reservation_drop_before_commit_releases() {
        let m = WalMeter::new(10_000, 100, Arc::new(Event::new()));
        {
            let r = m.reserve(500).unwrap();
            let _r = r.reconcile(120).unwrap();
            assert_eq!(m.usage(), (100, 220));
            // dropped without commit
        }
        assert_eq!(m.usage(), (100, 100), "reconciled-but-uncommitted release");
    }
}
