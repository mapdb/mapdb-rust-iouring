# Round 2: `begin_maintenance` handshake — fix round for your FIX verdict

Prior round: brief `/tmp/iouring-handshake-brief.md`, your review
`/tmp/iouring-handshake-codex.md` (verdict FIX, two MUST-FIX). Re-read both.
Repo: `/home/play2/mapdb/mapdb-rust-uring`, uncommitted tree on `62aaeaa`.
Current full diff: `/tmp/iouring-handshake-r2.diff` (supersedes the r1 diff).
Note: a `ci/check.sh` gate run may be executing in `target/` concurrently — do NOT run
`cargo test` for the whole suite yourself (it will contend on the build lock); run only
focused tests, e.g. `cargo test --locked --lib maintenance` and
`cargo test --locked --lib superseded`, or none.

## What changed since r1

MUST-FIX 1 (waiter-side ABA / close+poison reported as success):
- New `error::MaintenanceError { AlreadyPaused, Interrupted, Poisoned(PoisonReason), Closed }`
  (re-exported from lib). `Store::begin_maintenance` is now
  `async fn -> Result<(), MaintenanceError>`; `Ok` only for an established pause.
- `EngineState::enter_maintenance -> Result<u64, MaintenanceError>` (classifies the
  non-Running phase under the lock).
- `EngineState::pause_status(epoch) -> PauseStatus { Waiting | Established | Aborted(MaintenanceError) }`
  computed **under the phase mutex**: `Established` iff phase==Maintenance AND
  current epoch == caller's epoch AND acked >= epoch; `Waiting` iff phase==Maintenance
  AND epoch == current AND unacked; otherwise `Aborted` — `Interrupted` for Running or a
  superseded epoch, `Poisoned`/`Closed` for terminal/Closing.
- Unit tests in `state.rs`: `a_superseded_pause_is_interrupted_not_established` (enter 1,
  ack, resume, enter 2 unacked → status(1)=Interrupted, status(2)=Waiting; ack → 2
  Established, 1 still Interrupted), `terminal_transitions_abort_a_maintenance_waiter`.
- Store-level tests in `tests_store_actor.rs` (all sleep-free now, using `poll_once` on the
  pinned pause future while the writer is blocked in a test gate):
  `begin_maintenance_waits_for_a_claimed_command_to_publish`,
  `begin_maintenance_waits_for_a_running_flush`,
  `begin_maintenance_is_failed_by_poison` (expects `Err(Poisoned)`),
  `begin_maintenance_is_interrupted_by_a_concurrent_resume_and_repause` (first waiter
  → `Err(Interrupted)`, a third caller → `Err(AlreadyPaused)`, second waiter → `Ok` after
  release), `begin_maintenance_is_failed_by_close` (expects `Err(Closed)`).

MUST-FIX 2: `a_checkpoint_queued_before_maintenance_fails_fast_rather_than_waiting` now
pins the pause, asserts `poll_once` → pending while the sync gate is held, releases, then
`block_on(pause).unwrap()` before awaiting the checkpoint. Passes.

All in-repo `begin_maintenance` callers use `block_on(...).unwrap()`.

## Questions

1. Does `pause_status` close the waiter-side ABA fully? Specifically: can `Established`
   be returned while the writer is not at the barrier? (Argument: acked>=epoch was set by
   the writer at the barrier during this exact epoch; the writer leaves the barrier only
   via a phase change (resume/close/poison) or channel closure (impossible — the waiter's
   store handle keeps the channel open); any phase change makes the classification
   non-Established.) Find a counterexample or confirm.
2. Is the `MaintenanceError` variant set right, and is `AlreadyPaused` the correct
   outcome for a read-only fixture store and for a second concurrent caller?
3. Any remaining lost-wakeup, deadlock (waiter holding phase lock across await — it does
   not; `pause_status` takes and drops the lock synchronously), or ordering issue?
4. Are the tests now deterministic proofs (no sleep-dependence anywhere except
   `spin_until` for the *Closing* phase observation in the close test)?
5. Anything else before this ships?

Verdict format: SHIP / FIX (exact required changes) / REJECT. Cite file:line.
Write your answer to /tmp/iouring-handshake-codex-r2.md
