# 06 — Decision index

The binding decisions of the engine, and where each one is *stated* today.

**The Outcome column is the authority.** Every rule a user needs is written out
in this `doc/` tree; where a decision has a home document, the "Stated in"
column names it. Nothing here defers to a document you cannot read.

| Decision | Outcome | Stated in |
|---|---|---|
| WAL durability rules | PSOW assumption; committed-batch recovery contract; `incarnation` carries no acceptance rule; `max_wal_bytes` accounting | [02-durability.md](02-durability.md) |
| Public API rules | owned bytes, `Send` futures, permit semantics, one reservation error story, error taxonomy | [08-api-contract.md](08-api-contract.md) |
| Direct root header | format v4 double-slot alternating, no migration, mutual refusal between v3 and v4 | [02-durability.md](02-durability.md) |
| drop vs close | drop ≠ close; the flock is owned by the writer thread and released as its final act | [02-durability.md](02-durability.md) |
| Fail-safe creation | atomic sidecar create-claim; destruction only via `create_path_overwrite` | [02-durability.md](02-durability.md) |
| WAL `incarnation` validation | rejected as unsound; the field stays diagnostic | [02-durability.md](02-durability.md) |
| Maintenance policy | caller-controlled in v1; thresholds, alert table and runbook; automatic trigger deferred | [03-operations.md](03-operations.md) |
| Default I/O backend | Syscall retained; io_uring available and supported but not default. The comparison closed **unresolved** — the measurement floor on a non-quiet host was wider than the effect — so the incumbent stays | [04-performance.md](04-performance.md) |
| io_uring boundary implementation | compio selected for `CompioIo` | [04-performance.md](04-performance.md) |
| O_DIRECT + bounce buffers | device-side, in the I/O boundary rather than the store | [04-performance.md](04-performance.md) |
| Flush-write fan-out | ships; in-flight cap 64; identical durable bytes to the serial path, and no header write after any chunk failure | [04-performance.md](04-performance.md) |
| Clean cache tier | adopt `quick_cache`, delete the custom CLOCK evictor. The `=0.6.24` pin is an audit boundary, not a compatibility pin | [04-performance.md](04-performance.md) |
| Queue/prefetch exit measurement | rejected — no measured win at the store path | [04-performance.md](04-performance.md) |
| Parallel recovery checksum | rejected — the crossover lies past any batch size recovery actually produces | [04-performance.md](04-performance.md) |
| Negative cache bound | FIFO ring, capacity 1024; deliberately not an option | [04-performance.md](04-performance.md) |
| `get_many` routing | grouped driver at ≥ 2048 keys; `get_many_grouped` opts in at any size; error-identity rule pinned | [04-performance.md](04-performance.md) |
| Overflow-chain prefetch | opt-in and hidden; Direct + O_DIRECT + chains ≥ 4 links only; default-on blocked | [04-performance.md](04-performance.md) |
| Heap-range aliasing | blessed uniformly for leaf and branch heap objects: immutable pages make aliasing benign, and interval-set validation was rejected as a pure page-load tax. Pinned by `page::tests::aliased_heap_ranges_are_blessed` | — |
| `loom` | feature deleted — it models none of the third-party components the engine actually depends on | [05-testing-and-ci.md](05-testing-and-ci.md) |
| Hand-back notify race | no notify on hand-back (proved unnecessary by the retained-listener argument); `poll_load_once` must notify | — |
| Release engineering | version 0.1.0; `--locked` everywhere; MSRV 1.75 for default features, ≥ 1.85 with `io-uring` | [01-overview.md](01-overview.md) |
| Crash-test model | suspend-and-image power cut, plus a write-ahead intent journal and an exhaustive oracle | [05-testing-and-ci.md](05-testing-and-ci.md) |
