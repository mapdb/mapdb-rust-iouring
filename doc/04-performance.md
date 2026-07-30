# 04 — Performance: backends, cache, batching

## I/O backends

- **`IoBackend::Syscall` (default):** positional pread/pwrite/fdatasync
  on a small dedicated worker pool. This is the production default: the
  A/B against io_uring closed **unresolved** — the measurement floor on a
  non-quiet host was wider than any effect — so the incumbent stays.
- **`IoBackend::Uring`** (`io-uring` feature, Rust ≥1.85): compio-based
  io_uring ring. Fully supported, with a
  dedicated CI job (`MAPDB_REQUIRE_URING=1`) that fails rather than
  silently falling back; just not the default. O_DIRECT with
  device-side bounce buffers lives at this boundary.

  Flush-write fan-out does **not**: the per-flush in-flight cap
  (`Options::fanout_cap`, default 64) is coordinated in backend-independent
  StoreDirect code and applies whichever backend is selected. StoreWal
  ignores it — its checkpoint has a separate write path.
- **`FakeIo`** (tests): deterministic, with fault injection, read gates,
  and in-flight accounting — the substrate for most acceptance suites.

Choosing: stay on Syscall unless you have measured **your** workload.
The gap between backends was below the certification floor on every
regime that could be measured here.

If you re-measure, the rule that governs a backend switch is: pre-register
the decision criterion *before* you run, measure on a quiet host with the
production `create_path` constructor (not a proxy path that skips the
checkpoint), report a confidence interval rather than a mean, and treat a
result whose interval spans zero as "no change" — not as a win for
whichever side happens to be ahead. That is what the original comparison
failed, and why the default did not move.

## Page cache (`cache.rs`)

- **Clean tier:** `quick_cache` (the custom CLOCK evictor was deleted in
  its favour). Sharded; per-shard capacity derives from
  `available_parallelism()`. Its `ahash` is per-process-keyed — **never
  persist a hash value**.
- **Dirty pages** are pinned outside the clean tier until flushed.
- **Single-flight loads** are cancellation-independent: a dropped reader
  future does not abandon or duplicate an in-flight load.
- **Bounded negative cache:** `Failed` verdicts live in a FIFO
  ring of 1024 (not an options knob). Eviction is correctness-safe —
  a re-read revalidates. Counters: `negative_hits`, `failed_resident`,
  `negative_evictions`; a negative answer is neither hit nor miss.

## Admission

Writes pass a bounded weighted admission gate (queue slots, bytes, dirty
pages) — backpressure is at intake, not mid-flight. FIFO fairness is
weighted, not strict.

## `get_many` routing

- Batches < 2048 keys: sequential per-key path.
  Batches ≥ `GROUPED_ROUTE_MIN_BATCH` (2048): the concurrent grouped
  driver (level-ordered, shared-waker join).
- `get_many_grouped` opts into the grouped driver at any size — use it
  for batches you know are cold.
- Error identity: sequential returns the first error in input scan
  order; grouped returns the error covering the lowest input position.
- The 2048 threshold is policy data measured on one tree shape and one
  host. It is not a derived constant. Before changing it, re-measure on
  your own shape with a pre-registered criterion, exactly as for a backend
  switch above; a threshold moved on a single unrepeated run is a
  regression waiting to happen.

## Range scans (`SortedMap` / `Snapshot::for_each_range`, `…_rev`)

A scan is one future driving a branch-path-stack descent (no leaf sibling
links in the format): it pins O(height) branch pages plus one leaf, checks
bounds at the branch level to skip untouched subtrees, filters exactly at
the leaf, and reassembles overflow values before delivery. The descending
variants (`for_each_range_rev`, `collect_range_rev`) are the same
traversal mirrored — identical cost shape, no extra I/O. Costs to know:

- The whole scan holds **one snapshot lease** — an active scan pins its
  generation against retention until the future completes or is dropped.
  Stop early with `ControlFlow::Break` (or use bounded `collect_range`)
  rather than holding long scans open.
- A scan keeps a visited-page-id set (corruption guard): memory grows on the
  order of one `u64` per page visited, the same order as `verify` — budget
  several times 8 bytes per page in practice, since hash-table load factor,
  control bytes and allocator overhead all sit on top of the raw key. Bounded scans
  (`collect_range{,_rev}`, early `Break`) keep this small.
- Scan cost tracks *historical* tree shape: deletes never merge leaves, so
  a delete-heavy range can cross many underfull leaves (compaction rewrites
  pages but does not rebalance).
- Cross-leaf key order is enforced during the scan; a corrupt
  cycle/shared-subtree structure fails with `Corrupt` before any duplicate
  delivery.

## Overflow-chain prefetch (opt-in, hidden)

An id-stride prefetch for long overflow chains exists behind
`PageLocator::predict_overflow_location`. It is `#[doc(hidden)]`, off by
default, certified only for Direct + O_DIRECT + chains of ≥ 4 links, and
prefetched links deliberately bypass the page cache.

**Default-on is blocked**, and the block is a rule, not an oversight:
prediction is advisory, so on any shape where the guess is usually wrong
it converts one demand read into a demand read plus wasted device work.
Turning it on by default requires evidence that the waste counters stay
low across shapes that are *not* long Direct overflow chains — evidence
that does not exist. Enable it per-store, with the waste counters
watched.
