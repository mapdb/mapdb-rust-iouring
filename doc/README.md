# doc/ — mapdb-uring documentation

Permanent documentation for the crate. **This directory is complete on its
own**: every rule you need in order to use, operate or evaluate the engine
is stated here, and nothing here defers to a document you cannot read.

The engineering trail that produced these rules — design plans, review
rounds and adjudications — is kept privately and is not needed to use the
crate.

| File | Contents |
|---|---|
| [01-overview.md](01-overview.md) | What the crate is, architecture, module map, features, MSRV |
| [02-durability.md](02-durability.md) | Durability contracts: PSOW assumption, Direct v4 double-slot root, WAL recovery, creation/lifecycle safety |
| [03-operations.md](03-operations.md) | Maintenance runbook: checkpoint/compaction thresholds, `StoreFull`, poison, alerts |
| [04-performance.md](04-performance.md) | Backends, cache design, `get_many` routing, opt-in prefetch |
| [05-testing-and-ci.md](05-testing-and-ci.md) | Test suites, the local gate, the external crash tier and its non-claims |
| [06-decision-index.md](06-decision-index.md) | The binding decisions and their outcomes, with the doc that states each one |
| [07-mapdb-collections-alignment.md](07-mapdb-collections-alignment.md) | API vocabulary alignment with the sibling mapdb-collections crate: adopted names, deliberate divergences |
| [08-api-contract.md](08-api-contract.md) | Public API contract: owned bytes, `Send` futures, cancellation, permits, batch semantics, error taxonomy |
