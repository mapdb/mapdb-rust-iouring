# 05 — Testing and CI

## Local gate

```sh
./ci/check.sh   # fmt + clippy + tests across feature sets (the gate)
cargo test      # default features only
```

In-crate acceptance suites are named for the invariant under test:
`tests_store_actor`, `tests_read_path`, `tests_direct_writes`,
`tests_wal_recovery`, `tests_runtime_and_load`,
`tests_checkpoint_compaction`, `tests_create_path_safety`,
`tests_direct_header_slots`, `tests_directory_fsync_model`,
`tests_crash_recovery_fuzz`, `tests_get_many_concurrency`,
`tests_handle_drop_lifecycle`, `tests_real_io_parity`,
`tests_overflow_prefetch`, `tests_sorted_map_scans`, and
`io::device_boundary_tests`. They are in-crate because they need access to
internals — fault injection, read gates, and crash-image construction.
Alongside them: proptest fuzzing with committed regressions under
`proptest-regressions/`, Miri-clean aligned-buffer modules, integration
tests in `tests/`, and criterion benchmarks in `benches/`.

## CI workflows

- **`.github/workflows/ci.yml`** (push/PR): gate incl. unprivileged
  crash-smoke; MSRV 1.75 (default features); ring-required job
  (`MAPDB_REQUIRE_URING=1` + in-suite Compio round-trip canary); Miri;
  cargo-audit (pinned `=0.22.2`); and the `crash-tier` matrix —
  {direct,wal} × {ext4,xfs}, 3 rounds each, fail-fast:false.
- **`.github/workflows/crash-campaign.yml`**: weekly cron + manual
  dispatch, ~20 rounds per cell.

These workflows were authored before the repository had a remote, so their
first hosted execution is the first push. The release gate itself was
discharged by a manual, VM-hosted run of the same scripts.

## The external crash tier

The strongest evidence in the repo: real power-cut semantics, out of
process.

- **Model:** VM-hosted **suspend-and-image**. `dmsetup suspend --noflush
  --nolockfs` is the cut; the backing file is copied while suspended
  (a pristine crash image); the copy is mounted read-write so the
  filesystem performs **ordinary journal replay** (never fsck-repair or
  `norecovery`). Everything completed below dm is treated as stable — so
  the tier tests dirty-cache loss, write/sync ordering, journal
  recovery, and the namespace protocols.
- **Oracle:** a write-ahead **intent journal** on the harness side
  (intents fdatasync'd before enqueue; ACK recorded only after the
  backend durability barrier). The checker proves
  `max_ACK ≤ visible_txid ≤ max_intent`, then byte-exact-sweeps the
  entire key universe and checks triple count equality — exhaustive, not
  sampled.
- **Record:** every cell of {direct,wal} × {ext4,xfs} green on the manual
  qualifying run. First contact with the tier found and fixed a real
  D-state cut-sequence deadlock, which is the evidence that it bites.
  Re-run it yourself with `ci/crash/crash-tier.sh` (root required) rather
  than citing a number from this file.

### Non-claims (repeat these in any status report)

The tier does **not** cover: a lying volatile device cache, FUA/flush
dishonesty, sector tearing, PSOW violation, unreachable-garbage
corruption, or the io_uring backend. The named stronger tiers that would
cover them — `dm-log-writes` replay at every FLUSH/FUA point, and an
io_uring crash cell — are not implemented.

## Toolchain oracles

- **Miri:** the `io::aligned` and `io::uring::dio_buf_tests` modules pass
  under nightly Miri; CI's miri job is the continuous oracle.
- **MSRV:** default-feature graph verified on a real 1.75 toolchain;
  `io-uring` needs ≥1.85 (edition2024 crates in compio's graph).
