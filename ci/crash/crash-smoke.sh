#!/usr/bin/env bash
# process-crash-smoke: the UNPRIVILEGED tier of the external crash
# harness — SIGKILL at a random point after readiness, on the host filesystem,
# no device fault. This proves process-crash consistency and wires the harness
# binaries against the real syscall open/lock/recovery path; it is NOT a
# power-cut test (the OS page cache survives SIGKILL) and a green run here is
# no ext4/XFS durability evidence — that is ci/crash/crash-tier.sh's job.
# Runs in the required local gate
# (ci/check.sh) with fixed seeds; any failure is a real bug by definition:
# an acked-then-lost write or a corrupt reopen.
set -euo pipefail
cd "$(dirname "$0")/../.."

ROUNDS="${CRASH_SMOKE_ROUNDS:-3}"
WORK="${CRASH_SMOKE_DIR:-target/crash-smoke}"
# With the workload defaults (group=8, readiness after >=3 durable groups) the
# ACK frontier at readiness is txid >= 24; the checker independently enforces
# it so a cut can never pass vacuously.
MIN_ACK=24

echo "== process-crash-smoke: build + unit tests (crash harness) =="
cargo build --locked -p mapdb-uring-crash-harness --bins
cargo test --locked -p mapdb-uring-crash-harness --lib --quiet

WORKLOAD="$PWD/target/debug/crash_workload"
CHECKER="$PWD/target/debug/crash_check"
rm -rf "$WORK"
mkdir -p "$WORK"

fail() {
  echo "process-crash-smoke: FAILED: $1 (artifacts kept in $2)" >&2
  exit 1
}

for backend in direct wal; do
  for round in $(seq 1 "$ROUNDS"); do
    R="$WORK/$backend-$round"
    mkdir -p "$R"
    # Run-id must be per-EXECUTION unique, not per-round-name: fixed strings
    # would let images and journals from two executions of the same cell
    # cross-pair undetected.
    RUN_ID="smoke-$backend-$round-$$-$(date +%s%N)"
    "$WORKLOAD" --backend "$backend" --store "$R/store.data" \
      --journal "$R/journal" --run-id "$RUN_ID" \
      --seed "$round" 2>"$R/workload.err" &
    WPID=$!
    # Readiness protocol, not a fixed sleep: wait for the durable
    # sentinel with a timeout and a liveness check.
    for _ in $(seq 1 600); do
      [ -f "$R/journal.ready" ] && break
      kill -0 "$WPID" 2>/dev/null || break
      sleep 0.1
    done
    if ! [ -f "$R/journal.ready" ]; then
      kill -9 "$WPID" 2>/dev/null || true
      cat "$R/workload.err" >&2 || true
      fail "$backend round $round never became ready" "$R"
    fi
    # Vary the cut point deterministically per round, then SIGKILL.
    sleep "0.$((round * 17 % 90 + 5))"
    kill -9 "$WPID"
    set +e
    wait "$WPID"
    STATUS=$?
    set -e
    if [ "$STATUS" -ne 137 ]; then
      fail "$backend round $round exited $STATUS, expected SIGKILL (137)" "$R"
    fi
    # Strict per-round bound: a recovery deadlock
    # must fail the gate, never hang it. 124 = timeout's own verdict.
    if ! timeout -k 10 120 "$CHECKER" --backend "$backend" --store "$R/store.data" \
      --journal "$R/journal" --min-ack "$MIN_ACK" | tee "$R/verdict"; then
      fail "$backend round $round checker verdict (124 = 120s timeout)" "$R"
    fi
    rm -rf "$R"
  done
done

echo "== process-crash-smoke PASSED ($ROUNDS rounds x {direct,wal}) =="
