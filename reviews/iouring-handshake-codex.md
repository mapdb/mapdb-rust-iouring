# Verdict: FIX

The writer-side barrier is basically the right mechanism, and the Direct/WAL durability semantics support it. However, the waiter-side predicate still has an ABA hole, its terminal/abort outcome is reported as a successful pause, and the sync-to-async conversion deadlocks an existing deterministic test.

## MUST-FIX 1 — do not treat an old acknowledgement or a phase exit as successful establishment

`maintenance_established` currently returns true for either `maintenance_acked >= epoch` or `phase != Maintenance` (`src/state.rs:237-249`), and `begin_maintenance` maps every such result to `true` (`src/store.rs:920-935`). That is not the documented meaning of `true` at `src/store.rs:901-908`.

There are two concrete failures:

1. **Waiter-side resume/re-pause ABA.** Suppose epoch 1 is acknowledged and its waiter has not run yet. Another thread resumes, the writer starts a command, and another caller enters epoch 2. The epoch-1 waiter can now run, observe `maintenance_acked >= 1`, and return `true` while the epoch-2 writer is still inside that command. Loading the current epoch at the writer ack point protects the *new* epoch in some interleavings, but it does not stop a delayed old waiter from accepting an old ack during a later Maintenance phase.
2. **Close/poison is reported as a successful backup barrier.** If the writer is gated immediately before publication, a pause starts, and another thread poisons or closes the store, the waiter returns `true` immediately because the phase left Maintenance. The already-started `apply_batch` can still publish after that return: its terminal check is much earlier than the publication at `src/writer.rs:540-549,660-673`. The new poison test actually constructs this shape, returns `true`, releases the gate, and deliberately ignores the in-flight result (`src/tests_store_actor.rs:458-487`). Closing can additionally drain and publish accepted commands at `src/writer.rs:784-797`. Thus `true` does not imply “no root is published and no file is written” as promised.

Required change: replace the boolean OR predicate with a phase/epoch-aware tri-state result such as `Waiting | Established | Aborted`:

- `Established` only when the phase is Maintenance, the currently active maintenance epoch is exactly the caller's epoch, and that epoch has been acknowledged.
- `Waiting` only while that exact epoch remains active and unacknowledged.
- `Aborted` if the phase leaves Maintenance or a later maintenance epoch supersedes it.

Take the phase mutex while classifying phase plus current epoch, so resume/re-pause cannot be collapsed into one sampled Maintenance state. `begin_maintenance` should return success only for `Established`. With the existing `bool` API, return `false` for an abort. Since this API is already being changed and has never been released, my preference is a `Result` that distinguishes concurrent close/poison from a concurrent resume/supersession; at minimum, close/poison must not return `true`. Update the contradictory documentation at `src/store.rs:910-912`.

Add deterministic coverage for the waiter-side ABA: enter epoch 1, ack epoch 1, resume, enter epoch 2 without acknowledging it, and prove the epoch-1 waiter is aborted rather than established. Update the poison test to require the abort/error outcome. A corresponding close-transition assertion is worthwhile because close deliberately applies and flushes admitted work.

## MUST-FIX 2 — update the gated checkpoint test for the async handshake

`a_checkpoint_queued_before_maintenance_fails_fast_rather_than_waiting` gates the writer inside WAL `datasync`, then calls `block_on(store.begin_maintenance())` before releasing that gate (`src/tests_checkpoint_compaction.rs:1226-1243`). The new handshake correctly waits for that same in-flight sync, so the test deadlocks. In a full `cargo test --locked` run it remained stuck there for more than 60 seconds.

Required change: pin and poll `begin_maintenance` once, assert that it is pending, release the sync gate, and only then drive the pause future to completion. This retains the original ordering proof: the checkpoint is already queued, the maintenance phase is installed, and the writer sees Maintenance after the gated commit finishes.

## Answers to the review questions

1. **Handshake mechanics:** there is no lost wakeup between `enter_maintenance` and waiter registration. The check/listen/re-check loop at `src/store.rs:925-934` is correct. The writer also registers its listener and rechecks the phase before acknowledging (`src/writer.rs:240-254`). A re-pause before the ack's epoch load is covered by acknowledging the newer epoch; one after the load notifies the registered listener and causes another loop. The remaining ABA is the delayed **waiter** accepting an old ack, described above. Assuming backend I/O eventually completes, all live writer paths reach the barrier or transition to a notifying terminal phase; writer panics are caught and poison the state (`src/writer.rs:114-133`).

   The ack is not issued with work pending. Publication is followed synchronously by `on_published` before the command returns (`src/writer.rs:660-679`), so the coordinator ledger covers every published root. Direct retains `pending` through page writes, both durability barriers, the header commit, and frontier publication (`src/direct.rs:677-797`). WAL retains it through record writes, `datasync`, overlay installation, and frontier publication (`src/wal.rs:1102-1176`). Therefore `has_pending == false` at the barrier means there is no unpublished durability work in either backend. Errors poison rather than falsely emptying the ledger.

2. **Memory ordering:** `SeqCst` is more than sufficient, and `fetch_max` is the right monotonic operation for an acknowledged frontier. It prevents a delayed acknowledgement from regressing the counter. Memory ordering does not repair the logical phase/epoch classification issue; that needs the mutex-protected tri-state check above.

3. **`phase != Maintenance`:** it is correct as a *wake/abort* condition, not as “established.” Recommend `false` for concurrent resume if retaining `bool`, and a terminal error for close/poison if adopting a result type. Returning `true` is unsafe and contradicts the external-backup contract.

4. **Sync to async:** acceptable for this unreleased API, and normal in-repo call sites were mechanically updated. I found no README or `doc/` call site documenting the old synchronous `begin_maintenance` signature. The stale statement in `src/wal.rs:1230-1233` refers specifically to low-level `enter_maintenance()` and remains technically true. The gated checkpoint test above is the missed behavioral caller update.

5. **Tests:** the gates make the positive ordering assertions deterministic. The 50 ms sleeps for negative assertions are understandable but can produce false passes if the pauser thread is starved. A cleaner sleep-free form is to pin `begin_maintenance`, call the existing `poll_once`, and assert `None` while the writer gate is known to be occupied; then release the gate and `block_on` the same future. The later “held until resume” check can use the same one-poll technique on the apply future instead of sleeping (`src/tests_store_actor.rs:401-410`). The poison test should likewise first poll the pause pending and then assert an abort/error after poison.

6. **Close/drop lifecycle:** once a pause is genuinely acknowledged, channel closure is safely observed by the Maintenance receive arm, which applies deferred commands and calls `drain_and_finish` (`src/writer.rs:279-309`). An outstanding `begin_maintenance` future itself holds a store handle, so last-handle channel closure cannot occur while that future is alive. Explicit `close()` and poison both notify, so they can release a waiter without hanging. The problem is only that they currently release it as `true`; close/drop is allowed to mutate and flush afterward, so that outcome must be abort/error, not successful quiescence.

## Verification

- `cargo test --locked begin_maintenance -- --nocapture`: all 3 new focused tests passed.
- `cargo test --locked`: the suite progressed normally but hung for over 60 seconds in `tests_checkpoint_compaction::a_checkpoint_queued_before_maintenance_fails_fast_rather_than_waiting`, confirming MUST-FIX 2; I interrupted the run.
