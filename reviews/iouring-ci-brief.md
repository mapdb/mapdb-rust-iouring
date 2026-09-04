# Review brief: two CI failures in mapdb-rust-iouring (Rust, single-writer-actor embedded B-tree store)

Repo checkout: `/home/play2/mapdb/mapdb-rust-uring` (read it — the full source is there).
Key files: `src/writer.rs` (the single writer actor loop), `src/state.rs` (`EngineState`
lifecycle + phase lock), `src/store.rs` (`StoreInner::enqueue`, `begin_maintenance`),
`src/direct.rs` (Direct flush coordinator, where `data_physical_bytes` moves),
`src/tests_store_actor.rs`, `src/tests_checkpoint_compaction.rs`.

## The two failures (GitHub Actions `ci`, run 30612311188)

1. job `ring-required` (`cargo test --locked --features io-uring`):
   `tests_store_actor::maintenance_pauses_writer_and_fails_fresh_admission`
   `assertion left == right failed: writer parked: not applied   left: 1  right: 0`

2. job `gate` (`cargo test --locked`):
   `tests_checkpoint_compaction::the_data_space_metric_tracks_direct_writes_and_compaction`
   `assertion failed: map.store().metrics().data_physical_bytes < grown`

Neither reproduces on an unloaded 32-core dev host; both are load-dependent.

## My root-cause analysis (please confirm or refute)

### Failure 1 — a real ordering hole in the maintenance/writer handshake

`EngineState::enter_maintenance()` flips the phase to `Maintenance` **under the phase
lock** and notifies. `StoreInner::enqueue` performs its channel `try_send` **under that
same phase lock** (`EngineState::with_intake`), and `with_intake` deliberately treats
`Maintenance` as `IntakeDecision::Open` — a permit granted before the pause is
*enqueue-allowed*, it just must not be *applied* until resume.

The writer loop (`writer::run_loop`) checks `core.state.phase()` only at the **top of the
loop**. Three dispatch sites then apply a command with no re-check:

  a. the loop-top `deferred.pop_front()` arm;
  b. the `commands.try_recv()` `Ok(cmd)` arm (window: loop-top read -> try_recv);
  c. the Running-idle park's `Wake::Command(Ok(cmd))` arm.

(c) is the one CI hits. The idle park is
`futures_lite::future::or(recv.recv(), or(state_listener, load_activity))`, and
`future::or` polls its **first** future first. So the very `notify` that announces the
pause wakes this thread, and if a command was enqueued in the meantime (which is legal —
`with_intake` says Open), the `recv` arm wins the race and the writer applies a batch
*during* the pause. That is exactly what the test observes: `map.len() == 1` where the
pause contract requires 0.

I reproduced this deterministically: with a temporary 20 ms `thread::sleep` inserted
at the head of the `recv` arm of that `or` (modelling a slow wake on a loaded 2-core
runner), plus a 30 ms sleep in the test before `begin_maintenance()` so the writer is
genuinely parked first, the test fails with the **exact** CI message
(`left: 1  right: 0`) on ~2 of 3 runs. With my fix applied and the same instrumentation
still in place: 0 failures in 20 runs.

### Failure 2 — a test bug (a flush-time metric read without flushing)

For `StoreDirect`, `core.data_physical_bytes` only advances in
`DirectCoordinator::flush_all` (`src/direct.rs`, `logical_tail` store) and in the
compaction cutover. `map.apply(..)` resolves at **publication**, not durability, so
`churn_direct(&map, 3)` (81 applies, no flush) leaves the background flush running
behind it. The test then reads `let grown = ...data_physical_bytes;` with no
synchronisation at all, so `grown` is whatever the writer's idle flushes happened to
have reached.

Measured on this host: with mild CPU load, `grown` ranged from 1,916,928 to 2,408,448
bytes while the true flushed tail was always 2,408,448. `report.data_bytes_after`
(the dense post-compaction size) is deterministically 98,304 and `empty` is 32,768
(PGSZ = 16,384). So the assertion `data_physical_bytes < grown` is false exactly when
the metric read catches only the first few flushes (`grown` <= 98,304), which on a
starved 2-core runner is entirely reachable.

I reproduced this deterministically too: temporarily capping the number of idle flushes
the writer is allowed to perform (env-gated) to 1, 2 or 3 makes the test fail with the
**exact** CI assertion `assertion failed: map.store().metrics().data_physical_bytes <
grown`; at 5 or more it passes.

I concluded this is a **test** bug, not a metric bug: the metric's contract really is
"the data file length as of the last flush/compaction", it is documented that way in
`src/direct.rs` and `src/store.rs`, and the test's own preamble asserts the seeded
empty-store value. Please push back if you think the metric itself should be defined
differently.

## The diff under review

```diff
diff --git a/src/tests_checkpoint_compaction.rs b/src/tests_checkpoint_compaction.rs
index e76f58f..42ad509 100644
--- a/src/tests_checkpoint_compaction.rs
+++ b/src/tests_checkpoint_compaction.rs
@@ -3170,8 +3170,15 @@ fn the_data_space_metric_tracks_direct_writes_and_compaction() {
     );
 
     churn_direct(&map, 3);
-    // A compaction runs at a flushed barrier, so it is also the barrier this
-    // metric reads after: force the writer to flush by compacting below.
+    // `apply` resolves at *publication*, so the churn above leaves the flush
+    // running behind it: read the metric now and it reports however far the
+    // writer's background flushes happened to get (observed anywhere from one
+    // page to the full tail). Flush the visible version first — the metric moves
+    // at a flush, so this is the only point where it means "the whole churn".
+    // Without it the pre-compaction value can be *below* the post-compaction
+    // dense size and the shrink assertion below fails for a bookkeeping reason.
+    let churned = map.snapshot().version();
+    block_on(map.flush(churned)).unwrap();
     let grown = map.store().metrics().data_physical_bytes;
     assert!(
         grown > empty,
diff --git a/src/writer.rs b/src/writer.rs
index 3ee5a3a..89ea2e7 100644
--- a/src/writer.rs
+++ b/src/writer.rs
@@ -304,7 +304,14 @@ fn run_loop(
             }
 
             // Commands parked by a maintenance pause run first: they were admitted
-            // before anything still in the channel.
+            // before anything still in the channel. Re-check the pause: it can be
+            // re-entered between the loop-top read and this pop (see
+            // `defer_if_paused` for the ordering); park instead of applying. The
+            // check is before the pop, not after — popping and re-queueing would
+            // rotate `deferred` and break FIFO.
+            if !deferred.is_empty() && core.state.phase() == Lifecycle::Maintenance {
+                continue;
+            }
             if let Some(cmd) = deferred.pop_front() {
                 apply_command(&core, &mut coordinator, cmd).await;
                 if coordinator.should_flush(
@@ -326,6 +333,9 @@ fn run_loop(
 
             match commands.try_recv() {
                 Ok(cmd) => {
+                    let Some(cmd) = defer_if_paused(&core, &mut deferred, cmd) else {
+                        continue;
+                    };
                     apply_command(&core, &mut coordinator, cmd).await;
                     if coordinator.should_flush(
                         core.flush_request
@@ -389,6 +399,9 @@ fn run_loop(
                     .await;
                     match wake {
                         Wake::Command(Ok(cmd)) => {
+                            let Some(cmd) = defer_if_paused(&core, &mut deferred, cmd) else {
+                                continue;
+                            };
                             apply_command(&core, &mut coordinator, cmd).await;
                             if coordinator.should_flush(
                                 core.flush_request
@@ -770,6 +783,36 @@ async fn drain_and_finish(
     core.state.finish_close();
 }
 
+/// Re-checks the maintenance pause immediately before a dequeued command would
+/// be dispatched. Returns `None` (the command is held for resume, at the back of
+/// `deferred`) if the pause is in force, `Some(cmd)` to apply it now.
+///
+/// **Ordering.** `StoreInner::enqueue` performs its channel send *under the
+/// phase lock* (`EngineState::with_intake`) and `EngineState::enter_maintenance`
+/// flips the phase under that same lock. So a command this thread already holds
+/// plus a phase read that observes `Maintenance` proves the pause was in force
+/// before this command could run, and no send can slip between this check and
+/// the apply that was not already ordered before the pause.
+///
+/// The loop-top phase check alone is *not* sufficient: the idle park races
+/// `commands.recv()` against the state-change listener with `future::or`, which
+/// polls the `recv` arm **first** — so the very notify that announces the pause
+/// can hand this thread a command enqueued after it. `try_recv` has the same
+/// window between the loop-top read and the receive. Deferring keeps FIFO:
+/// everything already in `deferred` was admitted earlier than a command only now
+/// arriving from the channel.
+fn defer_if_paused(
+    core: &Arc<Core>,
+    deferred: &mut std::collections::VecDeque<Command>,
+    cmd: Command,
+) -> Option<Command> {
+    if core.state.phase() == Lifecycle::Maintenance {
+        deferred.push_back(cmd);
+        return None;
+    }
+    Some(cmd)
+}
+
 /// Fails every command a maintenance pause deferred, with the store's terminal
 /// error, releasing its reservation — the `fail_drain` treatment for the commands
 /// that are no longer in the channel.
```

After the fix: `ci/check.sh` is green locally (fmt, clippy -D warnings on both workspace
members, `cargo test --locked` = 458 tests, `cargo test --locked --features io-uring`,
and the unprivileged process-crash smoke tier); `cargo-audit` is not installed on this
host, CI runs it.

## What I want from you

Write your answer to `/tmp/iouring-ci-codex.md`. Mark each finding MUST-FIX / SHOULD-FIX
/ NIT.

(i) Confirm or refute each root cause above. If you think either is wrong, say what the
    real cause is and how to show it.

(ii) Hunt for **remaining ordering holes in the maintenance/writer handshake** after this
     diff. Specifically:
     - Is `defer_if_paused` placed at every site that can dispatch a command? Are there
       sites I missed (e.g. `drain_and_finish`, the `Closing` arm, the Maintenance arm's
       own channel drain)?
     - Is the lock/ordering argument in the `defer_if_paused` doc comment actually
       correct given `EngineState::with_intake` / `enter_maintenance` / `resume`?
     - Does the pre-pop phase check on the `deferred` arm preserve FIFO, and can the
       `continue` it introduces spin (livelock) rather than park?
     - Is there an analogous hole for `Poisoned`/`Closing` flipping between the loop-top
       read and a dispatch (I deliberately left that out of scope — is that defensible,
       or is it a real bug that must land here)?
     - Does deferring a command in the idle-park arm risk **losing** it on any teardown
       path (handles dropped while paused, poison, close)?
     - Any way the fix can starve or delay a command indefinitely, or break the
       `flush_request` / `should_flush` durability progress that the arms I edited also
       drive (note I `continue` past the `should_flush` + `drive_pending` calls when a
       command is deferred).

(iii) Check the amended test now measures what its name claims
      (`the_data_space_metric_tracks_direct_writes_and_compaction`): is
      `map.snapshot().version()` + `map.flush(v)` the right barrier, can that flush hang
      or return `StaleIncarnation`, and does the test still prove the "compaction shrinks
      it" claim rather than trivially passing?

(iv) Anything else in `src/writer.rs`'s maintenance handling that looks wrong.
