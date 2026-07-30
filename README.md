# mapdb-uring

A Linux-first asynchronous storage engine: two page stores and a limited B+tree
map, ported from the MapDB5 direction. io_uring is an optional backend
(`io-uring` feature, `IoBackend::Uring`); the default durable-byte backend is a
positional-syscall worker pool.

- **`StoreDirect`** — group-flush durability, no WAL: unflushed writes are lost
  on crash. The root header is double-slot and alternating (format v4): a
  crash that tears the in-flight header slot falls back to the
  previous committed root — flushed state survives any single-slot tear,
  **assuming powersafe overwrite** of the not-written slot.
- **`StoreWal`** — checksummed write-ahead page-batch log with group commit,
  crash recovery, checkpoint-by-rotation, and compaction. Recovery yields the
  last completely committed batch and never exposes a partial one, **assuming
  powersafe overwrite** (SQLite's `SQLITE_POWERSAFE_OVERWRITE`) — supported on
  ordinary local ext4/xfs; CoW/compressed/network filesystems are outside the v1
  supported set. The contract is stated in full in
  [`doc/02-durability.md`](doc/02-durability.md).
- **`BTreeMap<B>`** — one map over either backend: bytewise-ordered owned
  keys/values, immutable copy-on-write pages, snapshots, point ops
  (`get`/`get_many`), conditional atomic batches, one writer + many concurrent
  snapshot readers, and snapshot-consistent **consumer-callback range scans**.
- **`SortedMap<B>`** — the high-level sorted-map layer over `BTreeMap`,
  vocabulary-aligned with `mapdb-collections`: `insert`/`remove`/`put_all`/
  `put_if_absent`/`compare_and_set`, NavigableMap lookups (`first_entry`,
  `floor_entry`, `ceiling_entry`, …), `remove_range`, and ordered range/prefix
  scans through a callback in **both directions** (`for_each_range` /
  `for_each_range_rev`, bounded `collect_range{,_rev}`; no iterator type).
  Still no resumable cursor/`Stream` type.

## Status

**Not yet a production storage engine, but past the prototype stage.** What
exists: real I/O backends (a syscall worker pool, plus an optional compio
io_uring ring and optional O_DIRECT), WAL recovery with a crash-image test
matrix and property-based fuzzing, checkpoints, compaction, snapshot retention,
weighted admission, and a clean/dirty page cache. The hardening work is
complete and its release gate was discharged by a real power-cut test matrix
over {StoreDirect, StoreWal} × {ext4, xfs}.

Known limits before you rely on it: maintenance is caller-triggered, there is
no resumable cursor, the crash matrix is syscall-backend only, and the
durability claim rests on a powersafe-overwrite assumption that holds on
ordinary local ext4/xfs and is not verified elsewhere. The API has never been
released, so it is not stable.

## Documentation

[`doc/`](doc/) is the complete, self-contained documentation set: overview,
durability contract, operations runbook, performance, testing and CI, the
decision index, and the public API contract. Every normative rule lives there.

## Layout

```
src/
  lib.rs        crate root, public re-exports
  version.rs    Version, Incarnation
  error.rs      Read/Write/Flush/Commit/Close error + state enums
  page.rs       Page, PageId, PageRef, page codec + structural validation
  backend.rs    Backend trait, Direct/Wal markers, FileGeneration, PageLocator
  io/           PageIo boundary: SyscallIo, CompioIo (io-uring), FakeIo + faults
  state.rs      engine lifecycle state machine (Running/Maintenance/Poisoned/…)
  admission.rs  bounded weighted admission (queue slots, bytes, dirty pages)
  batch.rs      WriteBatch, ApplyOutcome, WriteCost
  store.rs      Store<B> / StoreInner, Options, Snapshot
  map.rs        BTreeMap<B>, WritePermit
  btree.rs      copy-on-write B+tree builder (splits, overflow chains)
  read.rs       tree traversal, get/get_many, verify, live-set walker
  writer.rs     single-writer actor loop
  wal.rs        WAL record codec, recovery scan, checkpoint protocol
  checkpoint.rs double-slot checkpoint header codec
  direct.rs     Direct flush coordinator, header, allocator
  cache.rs      page cache: lock-free quick_cache clean tier, dirty pins, single-flight loads
  retention.rs  snapshot generation retention/revocation
  compact.rs    DataFileHost seam + CompactReport
  fileset.rs    WAL segment namespace trait
  metrics.rs    Metrics
ci/check.sh     the local CI gate (fmt, clippy, tests across feature sets)
tools/crash-harness/
                destructive crash-tier tooling; a separate, unpublished
                workspace member, not part of the library
```

## Build & test

```sh
./ci/check.sh   # fmt + clippy + tests on all feature sets
cargo test      # default features only
```

## Examples

```sh
cargo run --example sorted_kv   # SortedMap tour: durable WAL store, scans both
                                # directions, pagination, remove_range, reopen
cargo run --example http_kv     # hand-rolled HTTP KV over StoreDirect,
                                # fail-fast overload (503) via admission
cargo run --example axum_kv     # Tokio + Axum service over StoreWal
```

## Benchmarks

`benches/` holds the criterion benchmarks (`cargo bench`). They measure this
engine against itself across configurations; they are tuning instruments, not
competitive claims. **This project makes no published performance comparison
against any other storage engine.**

## License

Dual EPL-1.0 / EDL-1.0 (`SPDX-License-Identifier: EPL-1.0 OR BSD-3-Clause`).
See [`LICENSE-EPL-1.0.txt`](LICENSE-EPL-1.0.txt),
[`LICENSE-EDL-1.0.txt`](LICENSE-EDL-1.0.txt) and [`NOTICE.md`](NOTICE.md).
