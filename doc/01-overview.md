# 01 — Overview

`mapdb-uring` is a Linux-first asynchronous storage engine ported from the
MapDB5 direction: two page stores and a limited B+tree map. It is
runtime-neutral (no tokio/async-std dependency in the engine; a dedicated
writer thread plus small I/O worker pools), and io_uring is an optional
backend, not a requirement.

## The three public pieces

- **`Store<Direct>` (StoreDirect)** — group-flush durability, no WAL.
  Unflushed writes are lost on crash; flushed state survives any
  single-slot header tear via the v4 double-slot alternating root header
  (see [02-durability.md](02-durability.md)).
- **`Store<Wal>` (StoreWal)** — checksummed write-ahead page-batch log
  with group commit, crash recovery to the last completely committed
  batch, checkpoint-by-rotation, and compaction.
- **`BTreeMap<B>`** — one map type over either backend: bytewise-ordered
  owned keys/values, immutable copy-on-write pages, snapshots, point ops
  (`get`, `get_many`, `get_many_grouped`), conditional atomic batches,
  one writer + many concurrent snapshot readers, snapshot-consistent
  callback range scans in both directions (`for_each_range`,
  `for_each_range_rev`).
- **`SortedMap<B>`** — the high-level sorted-map layer over `BTreeMap`,
  vocabulary-aligned with `mapdb-collections`: `insert`/`remove`, bulk
  `put_all`, `put_if_absent`/`compare_and_set`, the NavigableMap lookups
  (`first_entry`/`last_entry`/`floor_entry`/`ceiling_entry`/`lower_entry`/
  `higher_entry`), `remove_range`/`clear`, and ordered range/prefix scans
  through a **consumer callback** (`ControlFlow` to continue/stop; no
  iterator type), plus bounded `collect_range` and `count_range`. All of
  it exists descending too (`for_each_range_rev`, `collect_range_rev` —
  newest-first pagination). Scans capture a snapshot internally and hold
  one read lease for the whole traversal. Still deferred: a resumable
  cursor/`Stream` type.

## Architecture in one paragraph

All mutation flows through a single detached **writer actor thread**
per store, fed by a bounded, weighted **admission** gate (queue slots,
bytes, dirty pages). Reads go through an engine-owned **page cache**
(a sharded `quick_cache` clean tier whose warm hits are served without
taking the cache's map lock, dirty pins, cancellation-independent
single-flight loads, bounded negative cache) and never block the writer.
The current root is published as one atomic `(root, file-generation)`
Arc swap; snapshots are generation leases with revocation
(`retention.rs`). Physical I/O goes through the **`PageIo`** boundary,
with three implementations: `FakeIo` (deterministic, fault-injectable —
the test substrate), `SyscallIo` (positional pread/pwrite/fdatasync
worker pool — the production default), and `CompioIo` (io_uring via
compio, behind the `io-uring` feature).

## Module map (`src/`)

| Module | Role |
|---|---|
| `lib.rs` | crate root, public re-exports |
| `store.rs` | `Store<B>`, shared `Core` (incl. the namespace flock), `Options`, enqueue path |
| `state.rs` | lifecycle state machine (Running / Maintenance / Poisoned / Closing …) |
| `writer.rs` | single-writer actor loop; maintenance park; lock release as the thread's final act |
| `map.rs` / `btree.rs` / `read.rs` / `batch.rs` | `BTreeMap`, CoW B+tree writer, traversal + `get_many` driver + range scan, write batches |
| `sorted.rs` | `SortedMap` — the high-level sorted-map layer (callback scans, CAS) |
| `page.rs` / `backend.rs` / `version.rs` | page codec + validation; `Backend`/`PageLocator` traits; store UUID/incarnation |
| `cache.rs` / `admission.rs` | page cache; bounded weighted admission |
| `direct.rs` / `direct_fs.rs` | StoreDirect layout, v4 double-slot header, allocator, compaction; file protocols |
| `wal.rs` / `wal_fs.rs` / `checkpoint.rs` / `compact.rs` / `fileset.rs` | WAL codec + recovery, checkpoint header, compaction seam, segment namespace |
| `fs_util.rs` | shared FS durability helpers; `CreateClaim` (atomic creation) |
| `io/` | the `PageIo` boundary: `SyscallIo`, `CompioIo`, `FakeIo` + fault/gate knobs, aligned buffers |
| `retention.rs` / `metrics.rs` / `error.rs` | snapshot retention, metrics, error taxonomy |

The crash-tier substrate lives in `tools/crash-harness/`: a separate,
unpublished workspace member, not part of the library.

## Features and toolchain

- **Default features**: syscall backend only. **MSRV 1.75** (measured;
  `rust-version` in Cargo.toml).
- **`io-uring`**: enables `CompioIo` / `IoBackend::Uring`. Needs
  **Rust ≥ 1.85** (compio's locked graph contains edition2024 crates).

- `Cargo.lock` is committed; CI builds `--locked`.
- `quick_cache` is pinned `=0.6.24`, and the pin is an **audit boundary**,
  not a compatibility pin. Three facts about that version's internals were
  audited and are relied upon: the map-to-shard lock order, that `on_evict`
  touches only an atomic, and that `remove` bypasses `on_evict`. Bumping
  the version requires re-auditing those three facts first; do not relax
  the pin as routine dependency maintenance.
- Version 0.1.0. Licensed EPL-1.0 OR BSD-3-Clause (see `NOTICE.md`).

## Status

Not yet a production storage engine, but the reviewed gap list is closed:
all hardening tasks done and the release gate discharged by a real
power-cut test matrix (see [05-testing-and-ci.md](05-testing-and-ci.md)).
Run `./ci/check.sh` for the current suite result rather than trusting a
count written down here.
