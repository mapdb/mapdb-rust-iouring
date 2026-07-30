# 03 — Operations: the maintenance runbook

This document is the maintenance contract. It is normative and self-contained:
the thresholds, failure partition and recovery path stated here are the
supported ones.

## Policy: maintenance is caller-controlled

Maintenance — WAL checkpoint, and compaction on either backend — is
**caller-controlled** in v1. There is no automatic background trigger.

The reasoning is not that automation is hard: this is an embedded engine, the
host application owns the latency budget, and both operations run **on the
writer thread**, so an engine-initiated trigger would insert multi-`fsync`
pauses at moments the application cannot predict. What production readiness
requires here is not automation but a complete contract — thresholds you can
watch, the exact behaviour when you do not, and a documented recovery path.
That contract is this file.

An automatic policy remains open as a later opt-in
(`Options::auto_checkpoint: Option<f64>`, a fraction), gated on a real workload
asking for it.

## Path-backed StoreWal (`create_path` / `open_path`) — checkpoints and compacts

**Space model.** WAL bytes grow with every write until a checkpoint rotates the
segment. Admission fails fast with `WriteError::StoreFull` once a write's
worst-case bytes cannot fit `Options::max_wal_bytes` (default 256 MiB).
`StoreFull` is **not terminal**: reads keep working, and a checkpoint that
actually rotates restores write capacity.

**Admission is metered on `wal_reserved_bytes`, not on file size.** The three
WAL byte gauges are distinct and are not interchangeable:

| Metric | What it is | What to use it for |
|---|---|---|
| `wal_reserved_bytes` | logical frontier **plus** the worst-case bytes of every admitted-but-not-yet-applied batch | **the admission signal** — this is what `reserve()` compares against `max_wal_bytes` |
| `wal_logical_bytes` | header plus every accepted, encoded record | durable consumption; reopen scan cost |
| `wal_physical_bytes` | the segment file's length on disk | filesystem/space accounting and recovery cleanup |

They diverge in both directions, so watching the wrong one misleads. A
reservation can fail while `wal_physical_bytes` is well under the cap (the
outstanding worst-case charge is what filled the budget), and after a recovery
that stopped at a torn tail `wal_physical_bytes` can exceed
`wal_logical_bytes` — recovery never truncates.

**Watch `wal_reserved_bytes / max_wal_bytes`:**

| Fraction | Action |
|---|---|
| 0.5 – 0.7 | run `checkpoint()`; pick a point in this band that leaves headroom for your largest batch |
| 0.85 | alert — checkpoints are not keeping up |
| 1.0 | writers see `StoreFull` (non-terminal; reads keep working) |

**Also watch:**

- `open_scan_bytes` / `open_scan_ms` after a reopen. Reopen cost grows with
  un-checkpointed WAL, so checkpointing bounds restart time, not just space.
- `checkpoints_skipped_space` — a checkpoint declined because `max_data_bytes`
  headroom was insufficient. Raise `max_data_bytes`, or compact.

  **`Ok` from `checkpoint()` does not mean the WAL was rotated.** A skip is a
  healthy outcome, not an error: it returns `Ok(CheckpointReport)` with
  `skipped_space: Some(..)` and every other field zero — no relocation, no
  rotation, no WAL reclaimed. Code that checkpoints in order to make room must
  inspect the report:

  ```rust
  let report = store.checkpoint().await?;
  if let Some(skip) = report.skipped_space {
      // Nothing was reclaimed. Writes will still hit StoreFull.
      // limit = configured max_data_bytes, projected = what the store
      // would have owned had the checkpoint run.
      return Err(need_more_room(skip.limit, skip.projected));
  }
  ```

  Treating a bare `Ok` as "capacity restored" is the mistake this field exists
  to prevent.
- Retention pressure: long-lived snapshots pin superseded generations.
  `Options::max_retained_generations` bounds this by revoking the oldest.
  Watch `leaked_segments` (crash-orphaned segments, swept at the next open) and
  the retained-segment byte gauges on `Metrics`.

## Path-backed StoreDirect — compacts, never checkpoints

**Space model.** The data file grows with every flush (copy-on-write, no reuse
within a generation); only `compact()` reclaims. The fragmentation signal is
`data_physical_bytes` versus `data_live_pages × PAGE_SIZE` — the gap is
reclaimable garbage.

**`data_live_pages` is only valid after a maintenance operation has walked the
live set.** It is `0` until the first compaction, and on a non-empty Direct
store `0` means *unknown*, not *empty* — the documented garbage formula would
read almost 100% there and must not be used to trigger anything. Between
maintenance operations the value is stale (it describes the last walk, not the
current tree), so treat it as a slow fragmentation trend, not a live gauge.
Concretely: ignore the ratio while `data_live_pages == 0`, and re-read it only
after the compaction you triggered has returned.

Compact when the garbage fraction exceeds roughly 0.5 at a size you care about,
or on a schedule during quiesce windows.

**`max_data_bytes` does not bound a Direct store.** It is a budget the
*StoreWal checkpoint* holds itself to, and StoreDirect — which has no
checkpoint — ignores it entirely, exactly as it ignores `max_wal_bytes`
(`src/store.rs`, `Options::max_data_bytes`). A path-backed Direct store has no
configured size ceiling and never returns `StoreFull` on capacity; the data
file grows until the filesystem refuses it. If you need a ceiling on Direct,
enforce it above the engine.

Failure behaviour is the same partition as WAL compaction: a clean abort before
promote, a poison on promote failure.

## Caller-device stores (`create_with_io` / `open_over`) — cannot reclaim

Checkpoint and compaction return `CheckpointError::Unsupported`. A bare
`Arc<dyn PageIo>` has no namespace to rotate or rename, so no policy —
automatic or manual — could promise reclamation there. The contract:

- **StoreWal over a caller device:** `max_wal_bytes` is a *lifetime* write
  budget. Size it for the store's whole life, or plan a migration:
  `StoreFull` → read everything → write into a fresh path-backed store →
  switch. Reads never stop working and acknowledged data is never at risk.
- **StoreDirect over a caller device:** the device grows unboundedly under
  churn; the device owner (test harness, embedded environment) owns space.

These constructors are primarily test and embedding seams. **Production
deployments that need reclamation must use the path constructors.**

## Failure partition

| Class | Meaning | Response |
|---|---|---|
| `CheckpointError::Io` | **compaction only**, and only *before* its commit point: an I/O failure while probing or building the private temp file. Clean abort — nothing shared changed, store state unchanged | free the cause (e.g. `ENOSPC`), then retry with backoff |
| **poison** (`CheckpointError::Poisoned`, **any** I/O failure during a checkpoint, or a compaction failure at or after promote) | the shared data file was being mutated and the result is unpublishable | the store rejects writes; `close()` still releases the namespace; `open_path` replays every acknowledged commit |
| `CheckpointError::Compacting` | an operator pause is active | `end_maintenance()`, then retry |
| `CheckpointError::Unsupported` | caller-device store | see above; nothing to do |
| `CheckpointError::Internal` | an engine invariant was violated | a bug — file it. Nothing was mutated; do not retry blindly |

The asymmetry is deliberate and worth stating plainly, because it decides
whether "retry" is a safe response. Compaction builds into a *private* temp
file and only then promotes, so a failure before the promote touched nothing
shared and is recoverable in place. A checkpoint has no such staging area — it
relocates live pages into the *shared* data file — so once it starts writing,
any failure leaves the store unable to vouch for what is on disk, and the only
truthful answer is to poison. **Never retry a checkpoint in place after an I/O
failure; close and reopen so recovery runs.**

**Backoff.** On a clean-abort `Io`, retry with exponential backoff (for example
1 s doubling to a 60 s cap) *after* freeing space. Checkpointing in a tight
loop against a full disk converts a space problem into log noise. On a poison:
close, reopen (which recovers), and page the operator.

## Alert summary

| Signal | Threshold | Action |
|---|---|---|
| `wal_reserved_bytes / max_wal_bytes` | > 0.85 | checkpoint now; page if `StoreFull` is observed |
| `wal_physical_bytes` ≫ `wal_logical_bytes` | sustained | not admission pressure — un-reclaimed segment bytes; check retention and `leaked_segments` |
| `checkpoints_skipped_space` increments | any | raise `max_data_bytes`, or compact |
| Direct garbage fraction | > 0.5 (policy), **and** `data_live_pages > 0` | compact in a quiesce window |
| any poison transition | any | close, reopen (recovery), page — fatal alert |
| `CheckpointError::Internal` | any | file a bug; nothing was mutated |
| `leaked_segments` steadily growing | trend | investigate a crash loop |
| `failed_resident` / `negative_evictions` | persistent | persistent read failures — investigate the device |
| admission saturation | sustained | writers are queueing; size the budget or the workload |

## Runbook: `StoreFull` on a path-backed WAL store

1. Confirm from metrics that this is WAL pressure — `wal_reserved_bytes` close
   to `max_wal_bytes`, not `wal_physical_bytes`, which can sit either side of
   it — and not `max_data_bytes` pressure (`checkpoints_skipped_space`).

   Do not expect `wal_reserved_bytes` to *equal* the cap. Admission refuses when
   current reservations **plus the requested batch's worst case** would exceed
   it, so a large batch can be refused with the gauge materially below the cap.
   The larger your batches, the wider that gap. If the gauge is high and rising
   and writes are failing, this is WAL pressure regardless of the exact value.
2. Call `checkpoint()`.
   - `Ok` **and** `report.skipped_space.is_none()` — done: the segment rotated,
     capacity is restored and the alert clears.
   - `Ok` **and** `report.skipped_space.is_some()` — **nothing was reclaimed.**
     This is the `max_data_bytes` case reached by another route: the checkpoint
     declined because the store would not fit afterwards. Go to step 6. Do not
     retry the checkpoint; it will decline identically.
3. `checkpoint()` does not return `Io` — that variant reaches you only from
   `compact()`, whose pre-commit phase is the one that can fail cleanly. If you
   are running a compaction as part of this procedure and it returns `Io`, free
   disk space in the store's directory and retry it with backoff. A checkpoint
   that hits an I/O error poisons instead; see step 5.
4. If it returns `Compacting`, an operator pause is live: call
   `end_maintenance()` first.
5. If it returns `Poisoned`: `close()` (which releases the path even when
   poisoned), then `open_path` (recovery replays every acknowledged commit),
   then checkpoint again.
6. Skipped on space: raise `max_data_bytes` (the report's `projected` tells you
   what the checkpoint needed), or `compact()` to shrink the data file first,
   then checkpoint again. Checkpointing alone cannot resolve this.
7. Recurrence budget: if the alert re-fires within minutes, `max_wal_bytes` is
   undersized for the write rate. Raise it, or checkpoint on a timer.
