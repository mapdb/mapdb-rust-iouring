# 08 — Public API contract

This document is the API contract. It is normative and self-contained.

## Ownership: futures own their bytes; scan callbacks borrow

Two distinct rules, and conflating them is a lifetime bug:

- **Every byte buffer crossing the point-operation boundary is owned.** Keys
  and values passed to point operations and to `WriteBatch` are owned `Vec<u8>`,
  taken before the future is created; point reads and collected results return
  owned buffers. There is no zero-copy handle into a cached page that you can
  hold past a call. (This is about *bytes*. A scan future also captures your
  callback, which may itself borrow whatever you closed over — that borrow is
  yours to reason about, and the compiler will.)
- **Ordered-scan consumers receive borrowed slices.** `for_each`,
  `for_each_range` and their descending forms take
  `FnMut(&[u8], &[u8]) -> ControlFlow<()>` and hand the callback `(key, value)`
  slices borrowed from the page being visited. Those borrows are **ephemeral**:
  valid for the duration of the call and no longer. A consumer that needs a
  value beyond its own invocation must copy it (`to_vec()`).

The owned-by-default rule is deliberate. Every byte the engine reads passes
through the [`io::PageIo`] boundary into an owned buffer, and pages are
copy-on-write and evictable, so a borrowed view held across an `await` would
have to pin a page for an unbounded caller lifetime. Owned bytes make
cancellation safe (dropping a read future can never leave a dangling view),
make the futures `Send` without a lifetime parameter, and keep the eviction
policy free. The scan callbacks are the exception that proves it: the borrow
never crosses an `await` in caller code, because the engine calls *you*.

Practical consequence: batch your reads — but know the threshold.
`get_many` amortises traversal across keys only at or above
`GROUPED_ROUTE_MIN_BATCH` (2048), where it routes to the concurrent grouped
descent. Below that it runs a sequential per-key path and amortises nothing;
its benefit there is one captured root (a consistent version), not saved
traversal. A caller that knows its batch is cold and page-spread can call
`get_many_grouped` directly at any size. The two forms agree on every result;
on failure both are deterministic but may report different positions.

## Futures are `Send` and runtime-neutral

Every public future is `impl Future<Output = …> + Send`. The engine runs its
own writer thread and its own I/O boundary; it does not spawn onto, or require,
any async runtime. You may poll these futures on Tokio, smol,
`futures_lite::future::block_on`, or your own executor, and they may migrate
between worker threads mid-poll under a work-stealing scheduler.

The engine never holds an internal lock across an `await` in a way that a
foreign scheduler can observe; metadata critical sections are short and
non-awaiting.

## Cancellation

Dropping a future is always memory-safe and never corrupts state, but it does
not always cancel the operation:

- **Reads.** Dropping a read future releases its share of any in-flight load
  without abandoning or duplicating the load for other waiters.
- **`BTreeMap::apply` (permit-less).** Dropping the future *before* admission
  completes releases every reservation and enqueues nothing. Once admission
  resolves, the command is enqueued synchronously; from that point a dropped
  future no longer cancels the write — only its **outcome** is lost.
- **`WritePermit::apply`.** Enqueues synchronously at call. On return the
  command owns the permit and survives dropping the future; the outcome is then
  `Unknown` to the caller.

A caller that must distinguish "never applied" from "applied, outcome lost"
must use the permit path.

## Admission permits

```rust
pub fn reserve_write(&self, cost: WriteCost)
    -> impl Future<Output = Result<WritePermit<B>, WriteError>> + Send;
pub fn try_reserve_write(&self, cost: WriteCost)
    -> Result<WritePermit<B>, WriteError>;

impl<B: Backend> WritePermit<B> {
    pub fn apply(self, batch: WriteBatch)
        -> Result<impl Future<Output = Result<ApplyOutcome, WriteError>> + Send,
                  (WriteBatch, PermitMismatch)>;
}
```

Rules:

- A permit is consumed by `apply` or released **in full** by drop. There is no
  partial use and no reuse across batches.
- The batch's admission cost — encoded bytes, op count, worst-case private
  pages, and worst-case WAL bytes — must be within the permit's reservation.
  Every dimension is compared. A larger batch is a **caller error**, reported
  before enqueue as `Err((batch, PermitMismatch))` so the caller can re-reserve.
  It never silently re-enters admission and never blocks.
- A permit's reservation is against one store and one open incarnation.
- Applying after close or poison **normally** returns `Ok`: the returned future
  is already resolved to the store's terminal `WriteError`, the permit and batch
  are consumed locally, and nothing is enqueued. **The `Err` arm is reserved for
  cost mismatch only.** The terminal-state check and the enqueue are one
  linearized step under the same intake gate `close()` uses to stop intake, so a
  close cannot slip between a successful check and the synchronous enqueue:
  every applied permit's command is either drained by close or resolved with the
  terminal error, never lost.
- `BTreeMap::apply(&self, batch)` is the permit-less sugar, equivalent to
  reserve-then-apply with the batch's exact cost, except that its admission
  occurs lazily during the first poll rather than at call.

### Precedence: caller errors are checked before store state

The terminal-state rule above is not unconditional. Three checks happen in a
fixed order, and an earlier one wins even on a closed or poisoned store:

1. **Permit cost mismatch** (`WritePermit::apply` only) — compared before
   anything is enqueued. A batch larger than its permit yields
   `Err((batch, PermitMismatch))` whatever the store's state.
2. **Batch validation** — `validate()` runs inside the single enqueue choke
   point, *before* the intake gate is consulted. A malformed batch yields
   `WriteError::InvalidBatch(BatchError)` whatever the store's state.
3. **Terminal state** — only now does a closed or poisoned store resolve the
   future to `Closed` / `Poisoned`.

So "apply an oversized, malformed batch to a closed store" does **not** report
`Closed`; it reports the caller error. This is the intended ordering — a caller
error is the caller's to fix regardless of what the store is doing — but code
that infers store liveness from an apply result will infer it wrongly. Use the
terminal variants as the liveness signal, and only after the batch is known
valid.

## Write batch semantics

- All checks observe writer-current state **before any op** in the batch.
- Ops apply in order; the last write to a repeated key wins.
- A failed check is `Ok(ApplyOutcome::ConditionFailed)`, not an error: it
  publishes nothing, consumes no txid, and releases the whole reservation.
- An applied batch performs exactly one in-memory root swap — a concurrent
  snapshot never observes half a batch.
- Requested old values are aligned to the batch's operations; values that were
  not requested are not read.
- Encoded bytes, op count, and worst-case private pages are all bounded, but
  **by two different mechanisms returning two different errors**, and which one
  you see depends on where the batch trips:

  | Trips on | Where | Error |
  |---|---|---|
  | empty key, over-long key or value, op **or check** count over `max_ops_per_batch` | `WriteBatch::validate`, before admission | `WriteError::InvalidBatch(BatchError)` |
  | old-value response larger than `max_command_bytes` | during apply, once the store's own values are known | `WriteError::InvalidBatch(BatchError::OldValuesTooLarge)` |
  | encoded bytes over `max_command_bytes`, op count over `max_ops_per_batch`, or worst-case pages over **half** `max_dirty_pages` | admission hard-cap check | `WriteError::CostExceedsLimit` |

  Structural validation runs first, so an over-long batch is normally rejected
  as `InvalidBatch` before admission ever costs it; `CostExceedsLimit` is what
  you get from `reserve_write`/`try_reserve_write` on a `WriteCost` you built
  yourself. Neither is retryable — both are caller errors — but they are
  distinct variants and an exhaustive match must handle both.

  `OldValuesTooLarge` is the one that cannot be predicted from the batch: it
  depends on what the store holds for those keys, not on what you sent. Ask for
  fewer keys, or don't request old values. Nothing is published and no txid is
  consumed.

## Error taxonomy

There is one reservation error story: **both** `reserve_write` and
`try_reserve_write` fail with `WriteError`, because poison, close and cost caps
can fail either path. `WriteError` distinguishes the two rejection meanings so a
caller can never confuse "retry later" with "never":

| Variant | Meaning | Retryable |
|---|---|---|
| `Overloaded` | transient lack of capacity. Returned **only** by the `try_` form — the awaiting form waits instead | yes, later |
| `CostExceedsLimit` | the requested cost exceeds a hard cap that no amount of waiting can satisfy | no |
| `StoreFull` | a capacity ceiling refused the write **pre-publication** — usually `max_wal_bytes`, but also a format ceiling (see below). **Not terminal**: reads keep working, no txid is consumed | usually, after maintenance |
| `Compacting` | an operator maintenance pause is active | yes, after `end_maintenance()` |
| `InvalidBatch` | the batch violates a structural rule or a size cap (carries a `BatchError`) | no |
| `Poisoned` | terminal: an I/O failure left state unpublishable | no — close and reopen |
| `Closed` | terminal: the store was closed | no |

### `StoreFull` is not only about `max_wal_bytes`

The common cause is the WAL byte budget, and maintenance clears it. Two other
causes exist and maintenance does **not** clear them:

| Cause | When | Remedy |
|---|---|---|
| `max_wal_bytes` exhausted | ordinary operation under write load | `checkpoint()`, then retry — see [03-operations.md](03-operations.md) |
| a format ceiling reached: txid, allocator frontier, or a single record over the maximum recovery will scan | only under an adversarially crafted header, or a very large non-default `max_dirty_pages`/`max_wal_bytes` | not retryable; the batch or the configuration must change |

The format ceilings exist so the writer can never acknowledge something a later
`open_path` would refuse — the refusal is deliberately **as strict as
recovery**. All of them refuse pre-publication: nothing is written, no txid is
consumed, the reservation is released, and the store is not poisoned.

Note also what `StoreFull` is *not*. `max_data_bytes` does not produce it: a
checkpoint that would breach that limit returns `Ok` with `skipped_space` set,
and StoreDirect ignores the option entirely.

`Poisoned` and `Closed` are the shared terminal variants — with **one
exception**: `CloseError` spells the second one `AlreadyClosed`, because
closing an already-closed store is a different statement from operating on one.
Every other enum below uses `Closed`.

The full public set:

| Enum | Variants |
|---|---|
| `WriteError` | `Overloaded`, `CostExceedsLimit`, `StoreFull`, `Compacting`, `InvalidBatch`, `Poisoned`, `Closed` |
| `ReadError` | `Corrupt`, `Io`, `Poisoned`, `Closed`, `SnapshotExpired` |
| `FlushError` | `StaleIncarnation`, `Poisoned`, `Closed` |
| `CommitError` | `StaleIncarnation`, `Poisoned`, `Closed` |
| `CheckpointError` | `Unsupported`, `Compacting`, `Io`, `Poisoned`, `Closed`, `Internal` |
| `CloseError` | `Poisoned`, `AlreadyClosed` |
| `OpenError` | `Corrupt`, `UnsupportedFormat`, `UuidMismatch`, `AlreadyOpen`, `AlreadyExists`, `Io` |
| `BatchError` | `EmptyKey`, `KeyTooLong`, `ValueTooLong`, `TooManyOps`, `OldValuesTooLarge` |
| `RangeWriteError` | `Read(ReadError)`, `Write(WriteError)` |

Notes on the less obvious ones:

- `ReadError::Corrupt` is a structural or checksum failure in a page; the store
  is **not** necessarily poisoned. `ReadError::Io` is a device-level read
  failure, likewise non-terminal.
- `FlushError`/`CommitError::StaleIncarnation` means the handle belongs to a
  previous open of the store; the operation did nothing.
- `ReadError::SnapshotExpired` means the snapshot's generation was revoked
  under retention pressure (see `max_retained_generations`).
- `RangeWriteError` is returned by the scan-then-write path
  (`SortedMap::remove_range`) and simply says which half failed.
- `CheckpointError` is described in the operations doc's failure partition.

**`#[non_exhaustive]` is not universal.** Most public error enums carry it —
`OpenError`, `ReadError`, `WriteError`, `FlushError`, `CommitError`,
`CheckpointError`, `CloseError`, `Terminal`, `PoisonReason`. Two do **not**:
`BatchError` and `RangeWriteError`. You may match those exhaustively without a
wildcard arm, and adding a variant to either is a breaking change we would
treat as such. For every other enum, match with a wildcard arm.
