# 02 — Durability contracts

This document is the durability contract. It is normative *for behaviour*: if
the code disagrees with a rule stated here, that is a bug in one of them, and no
other document is needed to decide which rule was intended.

**It is not a format specification, and it is not sufficient to reimplement
against.** It states what survives a crash and what does not — the recovery
guarantee, the acceptance rules, the failure partition. It deliberately does not
specify header and record field layouts, widths, endianness, constants,
checksum algorithms and their byte coverage, padding, or version negotiation. A
reader can use this to reason about, operate and rely on the engine; a reader
cannot use it to write a second implementation that opens the same files. The
on-disk format is not stabilised (see the README) and no byte-level contract is
published, so identifiers such as `direct.rs::pick_winning_header` are pointers
into *this* implementation, not portable specification.

## The environment assumption: powersafe overwrite (PSOW)

Both stores assume **powersafe overwrite** in SQLite's sense
(`SQLITE_POWERSAFE_OVERWRITE`, <https://www.sqlite.org/psow.html>):

> A write of byte range `[a, b)` never alters any byte outside `[a, b)`, even
> if power is lost during the write or during later writeback.

Bytes *inside* the written range may be old, new, torn, or garbage after a
crash — the assumption says nothing about them. This is deliberately weaker
than atomic sector replacement, and atomicity is **not** required anywhere in
these protocols.

### Why it is load-bearing

WAL record `N+1` begins at the byte immediately after record `N`, so its first
write lands in a physical unit (device sector, filesystem block, page-cache
page) that also holds the tail of record `N` — already acknowledged and
`fdatasync`'d. Under the assumption, record `N`'s bytes lie outside every later
write's range and therefore survive verbatim; arbitrary tearing inside record
`N+1`'s range damages only the unacknowledged record, which recovery rejects at
its checksum. The same argument covers the post-recovery overwrite, whose start
offset is the accepted-prefix end. Without the assumption, a crash during the
shared-unit write could destroy record `N`'s tail, and recovery would lose an
acknowledged commit with no protocol bug.

### What the assumption rests on

Powersafe overwrite is an **end-to-end property of the whole storage stack** —
page-cache writeback granularity, filesystem block mapping and delayed
allocation, CoW/reflink/compression, device caches and firmware. It does not
follow from device sector size alone, and buffered `pwrite`/io_uring writes
request no atomic-write semantics from Linux. The contract is therefore a
**supported-configuration claim**, not a proof:

- **Supported:** ordinary local ext4 and xfs on conventional block devices,
  where SQLite also ships with PSOW enabled by default.
- **Outside the v1 supported set:** CoW filesystems (btrfs, ZFS), compressed
  filesystems, network filesystems, and exotic devices, unless you verify PSOW
  holds for your stack yourself.

Treat it like "fdatasync reaches stable storage": an environmental
prerequisite you are responsible for, not something the engine can check.

Padding every record to a conservative blast-radius boundary would remove the
assumption, at a cost of up to `blast_radius − 1` bytes per record (4095 at a
4 KiB filesystem-block radius) and a format change. That was considered and not
taken for v1. Linux `RWF_ATOMIC` untorn writes are a separate facility
requiring `O_DIRECT`, alignment, and filesystem/device support; also out of v1
scope.

## StoreDirect: v4 double-slot alternating root header

- Two self-checking header slots: page 0 = slot A, page 1 = slot B.
- Flush writes the **stale** slot, then `fdatasync`s. The committed slot is
  never in the write set of the next flush, so under PSOW a crash that tears
  the in-flight slot leaves the previous committed root intact.
- Open picks the winner by highest `header_txid`
  (`direct.rs::pick_winning_header`). On equal txids both slots must be
  byte-identical (A is the canonical winner); divergent equal-txid slots are
  `Corrupt`.
- Format byte is 4. **There is no migration**: a v4 build refuses v3 files and
  a v3 build refuses v4 files (`check_open_format`). This is deliberate — it
  closes the silent-downgrade hazard.
- **Contract:** unflushed writes are lost on crash; flushed state survives any
  single-slot tear.

## StoreWal: recovery contract

- The log is a chain of checksummed page-batch records with group commit.
  Recovery scans the longest valid prefix and yields **the last completely
  committed batch — never a partial one**, under PSOW.
- `commit(version)` is the durability barrier. `open_path` recovery replays
  **every acknowledged commit**: for a crash at any point, the recovered
  visible txid satisfies `max_ACK ≤ visible_txid ≤ max_intent` — nothing
  acknowledged is lost, and nothing unacknowledged is fabricated.
- **Record recovery** is **read-only and idempotent**. Scanning the WAL for the
  longest valid prefix reads and writes nothing; it never truncates the file,
  and new records later overwrite stale tail bytes in place. This is why
  `wal_physical_bytes` can exceed `wal_logical_bytes` after a crash.

  Distinguish this from **opening the path**, which is not read-only: before
  recovery runs, `open_path` performs namespace cleanup under the held lock —
  it removes an orphaned compaction temp file (active-path-wins) and retires
  WAL segments other than the one the winning data header names, completing a
  rotation a crash interrupted before its unlink. Both are safe (no snapshot
  can exist yet, and the winning header names the only segment anything may
  read), but a store directory does change on open. The invariant is that
  recovery never *loses* a record, not that opening leaves no trace.
- **The `incarnation` field carries no acceptance rule.** Each record stores
  the opening handle's 64-bit random nonce. Its bytes are covered by the record
  checksum, so corrupting it causes an ordinary checksum rejection. Recovery
  must **not** apply any *semantic* predicate to its value — not equality with
  the current open, not monotonicity, not ordering across records. Incarnations
  are unordered nonces, so such a check is unsound; anti-resurrection is
  carried entirely by the consecutive-txid rule plus the previous-record
  checksum chain. **A recovery implementation that rejects a record because of
  the incarnation *value*, rather than a checksum failure, is a bug.** The
  field exists so offline tooling can attribute records to opens when
  debugging. If a durably monotonic open epoch is ever needed, it will be a new
  field, not a reinterpretation of this one.

### `max_wal_bytes` and WAL byte accounting

`Options::max_wal_bytes` (default 256 MiB) bounds the log. The budget is
charged **at admission, before root publication**, so `WriteError::StoreFull`
is always a pre-publication result: the batch consumes no txid and is never
visible.

Three quantities are distinguished, and all three are observable through
`Metrics`:

| Quantity | Meaning |
|---|---|
| `wal_logical_bytes` | the logical append frontier — WAL header plus all accepted records |
| `wal_reserved_bytes` | `wal_logical_bytes` plus outstanding worst-case reservations; this is what admission compares against `max_wal_bytes` (the limit includes the WAL header), with saturating arithmetic |
| `wal_physical_bytes` | the file length. After torn-tail recovery this may exceed the logical frontier, because recovery never truncates |

The rules:

1. Admission reserves the **worst-case** encoded record size for the batch
   (record framing plus page payloads at the admitted worst-case private page
   count). A batch whose worst case exceeds the remaining budget fails
   admission with `StoreFull`.
2. After the writer builds the actual page batch, the exact encoded record
   length is computed from the frozen codec and the reservation is reconciled
   to it **before** root publication. Reconciliation is **bidirectional**: an
   actual length below the reservation releases the excess; an actual length
   above it charges the difference.
   The worst-case page count is a *pessimistic estimate*, not a hard bound —
   the no-merge copy-on-write builder leaves empty leaves that persist across
   batches, so a tall-but-sparse tree can path-copy more pages than any
   count/height figure sampled at admission predicts, and the tree can grow
   between admission and application. If the reconciled batch no longer fits
   `max_wal_bytes`, the write fails **pre-publication with `StoreFull`**. An
   overage does **not** poison the store: poisoning a legal write on a healthy
   store would be a bug. `wal_reserved_bytes` therefore never exceeds
   `max_wal_bytes`; only the *dirty-page* budget tolerates a transient
   overshoot.
3. Group commit concatenates already-sized per-txid records and shares only
   I/O and the `fdatasync`. It adds no envelope bytes, so reconciled figures
   stay exact.
4. Every non-applied outcome releases its full reservation: a
   `ConditionFailed` batch, a pre-publication failure of any kind, and a
   `WritePermit` dropped unused.
5. On reopen the logical frontier initialises to the accepted-prefix end, so
   overwritten torn-tail bytes are not double-counted. If that frontier already
   meets or exceeds the configured `max_wal_bytes` — possible when the option
   is lowered between opens — **the open succeeds**, reads and recovery are
   unaffected, and every subsequent write admission fails fast with
   `StoreFull` until the store is reopened with a larger limit.

A codec self-inconsistency (a framed length disagreeing with the recomputed
length) is caught by the recovery and trailer checks, not by reconciliation.

## Creation and lifecycle safety

- **Atomic create-claim.** Creation claims the namespace via a sidecar
  `create_new` token (`fs_util.rs::CreateClaim`); an existing store is never
  silently destroyed. Plain creation refuses with `OpenError::AlreadyExists`.
  The guarantee is precise: **every pre-existing store member is left
  byte-unchanged, and no new namespace entry survives the refusal.** It is not
  that the directory is untouched — the claim creates a `.lock` sidecar and a
  failed attempt removes it again (only if that attempt created it), so
  directory metadata such as mtime does change. Destruction happens **only**
  through `create_path_overwrite`.
- **Drop is not close.** The namespace `flock` is owned by the writer thread
  and released as its final act, so dropping the last handle can never orphan a
  locked namespace or tear down mid-write; the store drains its accepted writes
  first. `close()` is the graceful path, it releases the namespace **even when
  poisoned**, and a subsequent `open_path` recovers.
- **Clean close commits the visible root.** `close()` on `Store<Wal>` (and
  `BTreeMap<Wal>`) stops intake, drains accepted writes, then commits the
  highest visible root before releasing files. After a clean close that
  returned `Ok`, reopen recovers exactly the last visible pre-close state.
  "Visible but uncommitted state may disappear" applies to **crashes only**,
  never to a clean close. If that final commit fails with an I/O error,
  `close()` poisons and returns the error, and recovery yields the last
  previously committed prefix.
- **Caller-device stores** (`create_with_io` / `open_over`): the engine does
  not own the namespace or the file lifecycle, so checkpoint-by-rotation is
  `Unsupported` there. See [03-operations.md](03-operations.md).

## What durability does NOT claim

The validated model does not cover: a lying volatile device cache, FUA/flush
dishonesty, PSOW violation, or the io_uring backend (the crash matrix is
syscall-only). Say so in any status report that cites the crash tier. See
[05-testing-and-ci.md](05-testing-and-ci.md).

**On torn writes specifically**, two different things are true of the two
evidence tiers, and stating only one of them reads as a contradiction against
the PSOW section above:

- The **fake-I/O model** *does* exercise persistence within a written range:
  after an injected crash a write's byte range may be left old, new or garbled,
  at several blast radii and including regions shared between writes. This is
  what makes the PSOW-consistent claim ("bytes inside the range may be old,
  new, torn or garbage") a tested one rather than an assumed one.
- The **external VM crash campaign** tests real crash *ordering* against a real
  kernel and device. It does not emulate arbitrary device-sector tear modes,
  and specifically does not assume or verify a fixed 512-byte old/new sector
  granularity.

So: torn content within a write is modelled and tested in the fake tier;
arbitrary hardware sector-tear behaviour is not validated in either tier, and
is what the PSOW assumption exists to exclude.
