#!/usr/bin/env bash
# The PRIVILEGED external crash tier: power-cut
# by suspend-and-image on a real ext4/XFS filesystem over device-mapper.
#
# Model: the filesystem runs on a linear
# dm target over a loop device. The cut is
#     dmsetup suspend --noflush --nolockfs
# — `--nolockfs` keeps the suspend from syncing the filesystem, `--noflush`
# keeps queued-but-unmapped I/O from being pushed down; after a successful
# suspend the loop backing file holds exactly the writes the filesystem
# actually completed to the device, and everything still in the dirty page
# cache is lost. That backing file, copied while suspended, IS the crash
# image. The recovery copy is mounted read-write with ordinary journal replay
# (never norecovery/fsck — replay is the boot analogue; repair would mask a
# failure) and handed to crash_check.
#
# What this qualifies as: real dirty-cache loss, write/sync
# ordering, ext4/XFS journal recovery, and the store's namespace protocols.
# It does NOT model a lying volatile device cache, FUA/flush dishonesty,
# sector tearing, or PSOW violation — dm-log-writes replay at every FLUSH/FUA
# point is the named stronger future tier.
#
# Requires root. Build the binaries as an ordinary user first; this script
# only orchestrates loop/dm/mkfs/mount and takes the binaries by absolute
# path.
set -euo pipefail

usage() {
  echo "usage: crash-tier.sh --backend direct|wal --fs ext4|xfs --rounds N \\" >&2
  echo "         --workload /abs/crash_workload --checker /abs/crash_check --work <dir>" >&2
  exit 2
}

BACKEND="" FS="" ROUNDS=3 WORKLOAD="" CHECKER="" WORK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --fs) FS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --checker) CHECKER="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$BACKEND" ] && [ -n "$FS" ] && [ -n "$WORKLOAD" ] && [ -n "$CHECKER" ] && [ -n "$WORK" ] || usage
case "$BACKEND" in direct|wal) ;; *) usage ;; esac
case "$FS" in ext4|xfs) ;; *) usage ;; esac

# Preflight: fail clearly, before any state is created.
[ "$(id -u)" = 0 ] || { echo "crash-tier: must run as root" >&2; exit 1; }
for tool in dmsetup losetup "mkfs.$FS" blockdev findmnt awk stat timeout tar; do
  command -v "$tool" >/dev/null || { echo "crash-tier: missing $tool" >&2; exit 1; }
done
[ -x "$WORKLOAD" ] && [ -x "$CHECKER" ] || { echo "crash-tier: binaries not executable" >&2; exit 1; }
modprobe dm-mod 2>/dev/null || true

MIN_ACK=24  # workload defaults: group=8, ready >= 3 durable groups
mkdir -p "$WORK"
MIDMAINT_TOTAL=0

# Per-round cleanup state for the trap (preserve the exit status).
CUR_DM="" CUR_LOOP="" CUR_LOOP2="" CUR_MNT="" WPID=""
cleanup() {
  status=$?
  [ -n "$WPID" ] && kill -9 "$WPID" 2>/dev/null || true
  [ -n "$CUR_DM" ] && dmsetup resume "$CUR_DM" 2>/dev/null || true
  [ -n "$CUR_MNT" ] && umount "$CUR_MNT" 2>/dev/null || true
  [ -n "$CUR_DM" ] && dmsetup remove "$CUR_DM" 2>/dev/null || true
  [ -n "$CUR_LOOP" ] && losetup -d "$CUR_LOOP" 2>/dev/null || true
  [ -n "$CUR_LOOP2" ] && losetup -d "$CUR_LOOP2" 2>/dev/null || true
  # EVERY failure path packages the round sparse-aware and drops the raw
  # images — not just checker failures. Runs
  # after teardown so nothing under $R/mnt is still mounted.
  if [ "$status" -ne 0 ] && [ -n "${R:-}" ] && [ -d "$R" ]; then
    tar --sparse -zcf "$WORK/failure-$BACKEND-$FS-round-${round:-0}.tar.gz" -C "$R" . 2>/dev/null || true
    rm -f "$R/backing.img" "$R/pristine.img" "$R/recover.img"
    echo "crash-tier: failure artifacts: $WORK/failure-$BACKEND-$FS-round-${round:-0}.tar.gz" >&2
  fi
  exit "$status"
}
trap cleanup EXIT

for round in $(seq 1 "$ROUNDS"); do
  R="$WORK/$BACKEND-$FS-round-$round"
  rm -rf "$R"
  mkdir -p "$R/mnt"
  CUR_MNT="$R/mnt"
  IMG="$R/backing.img"
  truncate -s 1G "$IMG"
  CUR_LOOP=$(losetup --find --show "$IMG")
  SECTORS=$(blockdev --getsz "$CUR_LOOP")
  CUR_DM="crashtier-$$-$round"
  dmsetup create "$CUR_DM" --table "0 $SECTORS linear $CUR_LOOP 0"
  case "$FS" in
    ext4) mkfs.ext4 -q -F "/dev/mapper/$CUR_DM" ;;
    xfs) mkfs.xfs -q -f "/dev/mapper/$CUR_DM" ;;
  esac
  mount "/dev/mapper/$CUR_DM" "$R/mnt"

  DELAY=$(awk -v r="$RANDOM" 'BEGIN{printf "%.2f", 1 + (r % 300) / 100}')
  SEED=$((round * 1000 + RANDOM % 1000))
  # The environment record every round must carry.
  LOOPBASE=$(basename "$CUR_LOOP")
  {
    echo "kernel: $(uname -a)"
    echo "os: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"; echo "runner-image: ${ImageOS:-n/a}/${ImageVersion:-n/a}")"
    echo "git-rev: ${GIT_REV:-unknown}"
    echo "backend: $BACKEND  fs: $FS  round: $round  seed: $SEED  cut-delay: ${DELAY}s"
    echo "io-backend: syscall  direct_io: false  min-ack: $MIN_ACK"
    echo "workload-options: (binary defaults) keys=512 batch-ops=6 group=8 value-policy=vp1 max-wal-bytes=$((64 << 20))"
    echo "mkfs: $(mkfs.$FS -V 2>&1 | head -1)"
    echo "requested-mount-opts: (none — filesystem defaults)"
    echo "mount-effective: $(findmnt -no FSTYPE,OPTIONS "$R/mnt")"
    case "$FS" in
      ext4) echo "fs-features: $(tune2fs -l "/dev/mapper/$CUR_DM" 2>/dev/null | grep -i 'features\|journal' | tr '\n' '; ')" ;;
      xfs) echo "fs-features: $(xfs_info "$R/mnt" 2>/dev/null | tr '\n' '; ')" ;;
    esac
    echo "host-backing-fs: $(stat -f -c %T "$R")"
    echo "loop: $(losetup -l "$CUR_LOOP" | tail -1)"
    echo "dm-table: $(dmsetup table "$CUR_DM")"
    echo "sector-sizes: logical $(blockdev --getss "$CUR_LOOP") physical $(blockdev --getpbsz "$CUR_LOOP")"
    echo "queue-write-cache: $(cat "/sys/block/$LOOPBASE/queue/write_cache" 2>/dev/null || echo n/a)"
    echo "queue-fua: $(cat "/sys/block/$LOOPBASE/queue/fua" 2>/dev/null || echo n/a)"
  } | tee "$R/environment"

  "$WORKLOAD" --backend "$BACKEND" --store "$R/mnt/store.data" \
    --journal "$R/journal" --run-id "tier-$BACKEND-$FS-$round-${GITHUB_RUN_ID:-local}-$$-$(date +%s%N)" \
    --seed "$SEED" 2>"$R/workload.err" &
  WPID=$!
  # Readiness protocol with timeout + liveness.
  for _ in $(seq 1 1200); do
    [ -f "$R/journal.ready" ] && break
    kill -0 "$WPID" 2>/dev/null || break
    sleep 0.1
  done
  if ! [ -f "$R/journal.ready" ]; then
    cat "$R/workload.err" >&2 || true
    echo "crash-tier: FAILED: round $round never became ready" >&2
    exit 1
  fi
  sleep "$DELAY"

  # --- the cut ---
  # The suspend instant IS the cut: from here no write reaches the backing
  # file, and a workload mid-write freezes in uninterruptible sleep exactly
  # as under power loss. ORDER MATTERS (a real bug found on first contact): the
  # SIGKILL'd workload can sit in D-state on the suspended device and is
  # unreapable until its I/O completes — so `wait` must come AFTER resume,
  # never between suspend and the copy, or the round deadlocks.
  dmsetup suspend --noflush --nolockfs "$CUR_DM"
  SUSP=$(dmsetup info -c --noheadings -o suspended "$CUR_DM")
  if [ "$SUSP" != "Suspended" ]; then
    echo "crash-tier: FAILED: device not suspended after cut (state: $SUSP)" >&2
    exit 1
  fi
  kill -9 "$WPID"
  # The crash image: copied while the device is still suspended — the frozen
  # workload cannot alter it — then synced.
  cp --sparse=always "$IMG" "$R/pristine.img"
  sync "$R/pristine.img"
  cp --sparse=always "$R/pristine.img" "$R/recover.img"
  sync "$R/recover.img"

  # Resume releases the workload's queued I/O against the ORIGINAL (discarded)
  # device — harmless — and lets the SIGKILL reap it. Bounded: a process still
  # unreaped 30s after resume is a real defect, not a slow disk.
  dmsetup resume "$CUR_DM"
  for _ in $(seq 1 150); do
    kill -0 "$WPID" 2>/dev/null || break
    sleep 0.2
  done
  # The bound must be real: `wait` only after exit is observed — an
  # unconditional wait here could still block forever on a wedged process
  # and falsify the advertised 30 s.
  if kill -0 "$WPID" 2>/dev/null; then
    echo "crash-tier: FAILED: workload still unreaped 30s after resume" >&2
    exit 1
  fi
  set +e
  wait "$WPID"
  STATUS=$?
  set -e
  WPID=""
  if [ "$STATUS" -ne 137 ]; then
    echo "crash-tier: FAILED: workload exited $STATUS, expected SIGKILL (137)" >&2
    exit 1
  fi
  umount "$R/mnt"
  dmsetup remove "$CUR_DM"
  losetup -d "$CUR_LOOP"
  CUR_DM="" CUR_LOOP=""

  # Mount the recovery copy read-write: ordinary journal replay, no repair.
  CUR_LOOP2=$(losetup --find --show "$R/recover.img")
  mount "$CUR_LOOP2" "$R/mnt"
  set +e
  # Bounded: a recovery deadlock fails the round
  # (124 = timeout) instead of hanging the job for hours. Both pipe legs are
  # checked — a tee failure must not fabricate a passing round.
  timeout -k 15 300 "$CHECKER" --backend "$BACKEND" --store "$R/mnt/store.data" \
    --journal "$R/journal" --min-ack "$MIN_ACK" | tee "$R/verdict"
  # PIPESTATUS must be captured in ONE command — the first assignment is
  # itself a command and resets it.
  STATUSES=("${PIPESTATUS[@]}")
  RC=${STATUSES[0]}
  TEE_RC=${STATUSES[1]}
  set -e
  if [ "$TEE_RC" -ne 0 ]; then
    echo "crash-tier: FAILED: verdict artifact write failed (tee rc=$TEE_RC)" >&2
    exit 1
  fi
  umount "$R/mnt"
  losetup -d "$CUR_LOOP2"
  CUR_LOOP2="" CUR_MNT=""
  if [ "$RC" -ne 0 ]; then
    echo "crash-tier: FAILED: round $round checker rc=$RC (124 = 300s timeout); the exit trap packages the round sparse-aware" >&2
    exit 1
  fi
  if grep -q "maint_open_at_cut=1" "$R/verdict"; then
    MIDMAINT_TOTAL=$((MIDMAINT_TOTAL + 1))
  fi
  rm -f "$IMG" "$R/pristine.img" "$R/recover.img"
done

# Coverage honesty: count of rounds whose cut
# left an unmatched `M begin` at the witness cutoff. This is NOT proof the cut
# landed inside checkpoint/compaction (the process may have died before the
# `M done` sync) — a campaign must not claim mid-maintenance coverage from
# this counter alone, and must not claim it at all while it stays 0.
echo "crash-tier PASSED: backend=$BACKEND fs=$FS rounds=$ROUNDS maint-open-at-cut=$MIDMAINT_TOTAL"
