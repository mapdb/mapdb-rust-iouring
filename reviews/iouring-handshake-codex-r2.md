# Verdict: SHIP

The two round-1 MUST-FIX findings are resolved. I found no remaining safety, wakeup, deadlock, or ordering counterexample in the handshake.

## 1. Waiter-side ABA and the writer barrier

`pause_status` closes the waiter-side ABA. It takes the phase mutex and returns `Established` only for the conjunction `phase == Maintenance`, `maintenance_epoch == epoch`, and `maintenance_acked >= epoch` (`src/state.rs:267-284`). Thus an acknowledgement from epoch 1 cannot establish a delayed epoch-1 waiter during epoch 2, and leaving Maintenance for Running, Closing, Closed, or Poisoned makes the result non-success even if the old acknowledgement remains monotonic.

I found no execution in which that conjunction holds while the writer can be performing write/rebind work. The writer acknowledges only after it has reached the loop-top Maintenance arm, flushed pending coordinator work, drained commands into fail-fast/deferred outcomes, registered its listener, and rechecked Maintenance (`src/writer.rs:177-254`). After the ack it can drive read-only cache loads or re-loop, but it cannot dispatch a write without first observing Running (`src/writer.rs:255-315`). That is the relevant meaning of being at the barrier; it need not literally be asleep in `listener.await`.

The ack itself need not take the phase mutex. If resume occurs after the writer's final phase check, `pause_status` sees Running and aborts. If resume plus re-pause completes before the ack's epoch load, the writer may acknowledge the new epoch, but it has not dispatched or mutated anything in between, so that new pause is genuinely quiescent. If the re-pause occurs after the epoch load, its notification reaches the already-registered listener and the new epoch remains unacknowledged until the writer re-runs the arm (`src/state.rs:235-249`, `src/writer.rs:243-254`).

As with any concurrent API, another thread may call `end_maintenance` immediately after the final successful classification. The `Ok` linearizes before that resume; no implementation that returns an unowned `()` pause can promise otherwise.

## 2. Error variants and `AlreadyPaused`

The variant set is appropriate (`src/error.rs:275-305`): `Interrupted` distinguishes loss/supersession of a live pause, while `Poisoned` and `Closed` preserve materially different terminal causes. `AlreadyPaused` is correct for a second caller because that caller neither performed the Running-to-Maintenance transition nor owns its guarantee (`src/state.rs:214-232`). It is also coherent for the internal read-only fixture: construction has already permanently parked admission, there is no writer to acknowledge a newly requested epoch, and treating the call as re-entry avoids inventing ownership for the caller (`src/store.rs:790-808`).

Two non-blocking documentation cleanups would improve precision:

- `MaintenanceError::Interrupted` says the pause was interrupted "before the writer acknowledged it" (`src/error.rs:285-289`), but an acknowledged waiter that is delayed until after resume correctly receives `Interrupted` too, as the test at `src/state.rs:461-475` demonstrates. Say instead that the caller's epoch was no longer active when establishment was observed/returned.
- The `AlreadyPaused` sentence says it applies when the store "was not Running" (`src/store.rs:909-913`), although Poisoned/Closing/Closed produce their dedicated variants. Say "was already in Maintenance." Also, the absolute guarantee at `src/store.rs:903-905` is naturally conditional on the phase remaining Maintenance; explicit concurrent `close()` is another phase-changing operation that can drain/write admitted work without an `end_maintenance` call.

Neither wording issue changes the safe rule already stated at `src/error.rs:275-277`: every `Err` carries no quiescence guarantee.

## 3. Wakeups, locks, and ordering

There is no lost wakeup in the waiter. `begin_maintenance` performs check/listen/re-check/await and never holds the phase mutex across an await (`src/store.rs:925-942`); all resume, close, and poison transitions notify after changing phase (`src/state.rs:192-203`, `src/state.rs:287-332`). The writer uses the corresponding listen-then-recheck discipline before acknowledging (`src/writer.rs:240-254`).

There is no new deadlock: phase locking is synchronous and short, the writer never awaits while holding it, and a pending `begin_maintenance` future retains a store handle, so channel closure cannot silently remove the writer underneath that waiter. Explicit close and poison both classify and wake it. `SeqCst` atomics plus the phase-mutex classification are sufficient; `fetch_max` correctly keeps the acknowledged frontier monotonic (`src/state.rs:246-249`).

The checkpoint ordering fix is correct: the checkpoint and pause futures are each polled once while sync is deterministically gated, the gate is released, and only then is the pause driven to `Ok` before the checkpoint result is awaited (`src/tests_checkpoint_compaction.rs:1232-1256`).

## 4. Tests

The new handshake tests are deterministic proofs of the intended negative and positive orderings. Their negative assertions use one `poll_once` while the writer is known to be blocked at a gate, rather than elapsed time (`src/tests_store_actor.rs:355-409`, `src/tests_store_actor.rs:423-452`, `src/tests_store_actor.rs:463-524`). Poison and close then use explicit phase transitions and require the exact error class (`src/tests_store_actor.rs:478-484`, `src/tests_store_actor.rs:549-563`). The close test's bounded `spin_until` merely observes that the independently scheduled closer has installed Closing before checking the result; it does not use time passage as evidence that a future is pending (`src/tests_store_actor.rs:41-49`, `src/tests_store_actor.rs:552-558`).

## Verification

- `cargo test --locked --lib maintenance`: 13 passed.
- `cargo test --locked --lib superseded`: 3 passed.
- `git diff --check`: clean.

Per the brief, I did not run the whole suite.
