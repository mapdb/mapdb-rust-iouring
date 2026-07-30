#!/usr/bin/env bash
# The baseline gate. Run before every
# push/merge; .github/workflows/ci.yml runs the same matrix once a remote exists.
# Any failure fails the gate (set -e); the io_uring job is REQUIRED on a host
# whose kernel allows io_uring — tests skip individually only when the ring is
# blocked, and the dedicated-runner policy is enforced there by
# ci.yml's `ring-required` job, not this local script.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== fmt =="
cargo fmt --check

echo "== clippy (default) =="
cargo clippy --locked --all-targets -- -D warnings

echo "== clippy (crash harness workspace member) =="
# The harness is a separate, unpublished workspace member; a root --all-targets
# pass does not reach it.
cargo clippy --locked --all-targets -p mapdb-uring-crash-harness -- -D warnings

echo "== test (default) =="
cargo test --locked

echo "== test (io-uring) =="
cargo test --locked --features io-uring

echo "== process-crash-smoke (unprivileged crash tier) =="
ci/crash/crash-smoke.sh

echo "== audit (RustSec advisories vs committed lock) =="
if command -v cargo-audit >/dev/null 2>&1; then
  cargo audit
else
  echo "cargo-audit not installed (cargo install cargo-audit --version =0.22.2 --locked)" >&2
  exit 1
fi

echo "== gate PASSED =="
