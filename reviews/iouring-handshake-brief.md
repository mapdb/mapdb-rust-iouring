# Review brief: `begin_maintenance` writer-acknowledgement handshake (mapdb-rust-iouring)

Repo: `/home/play2/mapdb/mapdb-rust-uring` (uncommitted working tree on top of `62aaeaa`).
Full diff: `/tmp/iouring-handshake.diff`. Read the diff AND the surrounding code:
`src/state.rs` (`EngineState`, new epoch/ack fields, `TestGate`), `src/store.rs`
(`Store::begin_maintenance`, now `async`), `src/writer.rs` (Maintenance arm ack point,
`defer_if_paused` doc, two `#[cfg(test)]` gates), `src/tests_store_actor.rs` (three new
tests at the end).

## Context

Your previous review (`reviews/iouring-ci-codex.md`) had this MUST-FIX:

> `begin_maintenance()` still does not establish a completed pause. [...] This needs a
> writer acknowledgment/quiescence handshake, not another phase read. For example,
> transition intake to Maintenance, wake the writer, and make `begin_maintenance()` wait
> until the writer acknowledges a command boundary after finishing any claimed command
> and flushing prior pending work. [...] Close and poison must also release a waiter for
> that acknowledgment. A deterministic regression test should stop the writer immediately
> after its final Running check (or gate an already-started command before publication),
> call `begin_maintenance()` to completion, then release the writer and assert that no
> publication/file mutation occurs until resume. A second test should gate `flush_all`,
> call `begin_maintenance`, and prove the call does not return until that flush has
> completed and the writer has acknowledged the parked boundary.

This diff implements it.

## Design

- `EngineState` gains `maintenance_epoch: AtomicU64` (bumped under the phase lock by
  every Running→Maintenance transition; `enter_maintenance` now returns `Option<u64>`)
  and `maintenance_acked: AtomicU64` (`fetch_max` by the writer's `ack_maintenance`).
- `maintenance_established(epoch)` = `acked >= epoch || phase != Maintenance`. The second
  disjunct releases a waiter on concurrent resume / close (`Closing`) / poison, all of
  which notify the shared event.
- `Store::begin_maintenance` is now `async fn -> bool`: `enter_maintenance` (returns
  `false` at once if not Running), then the standard check/listen/re-check/await loop on
  `maintenance_established`.
- Writer: in the Maintenance arm, after the `has_pending → flush_all`, the `try_recv`
  drain-to-deferred, the `listen()`, and the post-listen phase re-check, the writer calls
  `core.state.ack_maintenance()`. It acks the *current* epoch (loaded at ack time, after
  the listener is registered), so a resume+re-pause between arm entry and the ack is
  either observed by the load (acked) or bumps after the load and notifies the
  already-registered listener (re-loop, re-ack). Nothing after the ack in that arm
  publishes or writes (the bounded `drive_pending` is read-only; a received command is
  deferred).
- `defer_if_paused` is unchanged in code; its doc now says why the sampled check is
  sufficient: a command it releases finishes before the writer reaches the barrier, so it
  lands before `begin_maintenance` returns, never after.
- Test gates: `Core::test_gates` (`#[cfg(test)]`), a `publish` gate immediately before
  the root `ArcSwap` store in `apply_batch`, and a `flush` gate at entry of
  `Coordinator::flush_all`. An unarmed gate is a no-op; an armed one blocks the writer
  thread until released. Tests: (1) claimed command gated before publication →
  `begin_maintenance` stays pending ≥50 ms, completes only after release, the command
  is published *before* the pause is established, a later permit-enqueued command is
  held until resume; (2) flush gated → pause pending until release, `durable_txid`
  covers the published version once established; (3) poison mid-command releases the
  waiter. All three fail when `maintenance_established` is mutated to `true`.
- `build_readonly` (no writer) still enters Maintenance directly; nothing waits on it.
- The ABA comment you flagged (NIT, `writer.rs` maintenance drain) was reworded.

## Questions

1. Is the handshake correct? Look hard for: (a) a lost wakeup between `enter_maintenance`
   and the waiter's listen; (b) the resume/re-pause ABA around the ack point (epoch
   loaded after `listen()`); (c) any path where the writer is live and Running→Maintenance
   happened but the writer never reaches the ack (hang); (d) any path where the ack is
   given while a command is in flight or the coordinator has pending unflushed work
   (`has_pending` false but files not yet synced?) — check Direct `flush_all` and Wal
   `commit_all` semantics for "pending" vs "durable".
2. Are the memory orderings adequate (SeqCst everywhere; the epoch bump is under the
   phase mutex)? Is `fetch_max` correct for the ack?
3. Is `maintenance_established`'s "phase != Maintenance → established" disjunct the right
   contract, or should a concurrent close/poison make `begin_maintenance` return `false`
   or an error? State a recommendation.
4. Is the sync→async signature change acceptable? All in-repo callers updated
   (`block_on(...)`). Any external doc (README/docs) still describing the old contract?
5. Are the three tests deterministic proofs (no sleep-dependence for the *positive*
   assertions)? The 50 ms sleeps only back the *negative* "has not returned" assertions;
   is that acceptable, and is there a sleep-free formulation you would prefer?
6. Anything in `close()` / `drain_and_finish` / the handle-drop path
   (`tests_handle_drop_lifecycle`) that interacts badly with a pending waiter?

Verdict format: SHIP / FIX (with exact required changes) / REJECT. Cite file:line.
Write your answer to /tmp/iouring-handshake-codex.md
