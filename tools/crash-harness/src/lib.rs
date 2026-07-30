//! Shared substrate of the external crash tier's two binaries: the
//! deterministic workload generator and the write-ahead intent journal.
//!
//! Unsupported test tooling. This crate is never published and its API carries
//! no stability guarantee; it exists only so `ci/crash/*.sh` has something to
//! run.
//!
//! Both `crash_workload` and `crash_check` must reproduce the exact same keys,
//! values, and operation order from the journal header alone — the journal
//! records *coordinates* (seed, sequence, op index), never value bytes, and the
//! digest on an intent line is an integrity cross-check, not the expected value.
//!
//! # Journal grammar (schema 1)
//!
//! One record per line: `<crc32c-hex8> <payload>\n`, CRC over the payload
//! bytes. Records, in the order a well-formed journal may contain them:
//!
//! ```text
//! H 1 <run-id> <backend> <seed> <keys> <batch-ops> <group> <vp1> <max-wal-bytes>
//! I <seq> <expected-txid> <batch-digest-hex8>     # write-ahead intent, synced
//! P <seq> <observed-txid>                          # post-apply diagnostic
//! F <highest-durable-txid>                         # durability ACK, synced
//! M <begin|done> <checkpoint|compact> <ordinal>    # maintenance coverage
//! R <ack-txid> <groups> <checkpoints> <compactions># readiness/coverage record
//! ```
//!
//! Ordering protocol: every `I` of a group is appended and
//! `fdatasync`'d **before** any of the group's applies is enqueued; `F` is
//! appended and synced only after the backend barrier (`flush` for Direct,
//! `commit` for WAL) returned for the group's highest version. An ACK
//! therefore proves durability, and a recovered txid always has its intent
//! on disk — the journal lives on a different filesystem than the store.
//!
//! Torn-tail rule: exactly one trailing fragment may be ignored, and only
//! when it lacks a terminating newline. A checksum-mismatched *complete*
//! line, a malformed interior record, a sequence gap or duplicate, or an
//! ACK/ready without its antecedents is journal corruption and fails the
//! round — the parser never scans past a bad record.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Journal schema version.
pub const SCHEMA: u32 = 1;
/// Readiness policy, shared by the workload (which journals `R` only at or
/// past these thresholds) and the checker (which independently REQUIRES them
/// from the prefix-validated `R` record).
pub const READY_MIN_GROUPS: u64 = 3;
pub const READY_MIN_COMPACTIONS: u64 = 1;
pub const READY_MIN_CHECKPOINTS_WAL: u64 = 1;
/// The reserved marker key carried by the first batch: its value is the
/// run-id, so a journal can never be checked against another round's image.
/// Outside the generated key universe (universe keys start `k`).
pub const RUN_ID_KEY: &[u8] = b"!crash-run-id";

/// SplitMix64 (Steele et al., documented fixed constants) — the harness PRNG.
/// Chosen over `RandomState` for cross-binary, cross-version determinism.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The per-sequence RNG: every batch is generated from stable coordinates
/// `(seed, seq)`, never from accumulated RNG state, so any prefix replays
/// identically.
pub fn rng_for(seed: u64, seq: u64) -> SplitMix64 {
    SplitMix64(seed ^ seq.wrapping_mul(0xA24B_AED4_963E_E407))
}

/// Workload configuration, fully carried by the `H` record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub run_id: String,
    pub backend: String,
    pub seed: u64,
    /// Key-universe size (`k…` keys `0..keys`).
    pub keys: u32,
    /// Operations per batch.
    pub batch_ops: u32,
    /// Batches per durable group.
    pub group: u32,
    pub max_wal_bytes: u64,
}

/// Key `i` of the universe: `k<i zero-padded>` plus a deterministic filler so
/// key lengths vary (splits both by count and by byte volume).
pub fn key_bytes(i: u32) -> Vec<u8> {
    let mut k = format!("k{i:05}").into_bytes();
    let filler = (i as usize * 7) % 24;
    k.extend(std::iter::repeat_n(b'x', filler));
    k
}

/// A value's identity is its generation coordinates; bytes are regenerated on
/// demand so neither binary ever stores values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueId {
    pub seq: u64,
    pub op_idx: u32,
    pub len: u32,
}

/// The deterministic value byte stream for a coordinate: SplitMix64 words
/// seeded from `(seed, seq, op_idx)`, truncated to `len`.
pub fn value_bytes(seed: u64, v: ValueId) -> Vec<u8> {
    let mut rng =
        SplitMix64(seed ^ v.seq.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ ((v.op_idx as u64) << 17));
    let mut out = Vec::with_capacity(v.len as usize);
    while out.len() < v.len as usize {
        out.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    out.truncate(v.len as usize);
    out
}

/// The forced size-class cycle: below / near-below / exactly at
/// the 8 KiB inline limit, just above it, and two multi-link classes. Class
/// selection cycles by absolute op coordinate so every short campaign hits
/// every class; every 8th sequence's op 0 draws a large 256 KiB–1 MiB value.
const SIZE_CLASSES: [u32; 6] = [64, 1000, 8 * 1024, 8 * 1024 + 1, 40_000, 100_000];

fn value_len(rng: &mut SplitMix64, seq: u64, op_idx: u32) -> u32 {
    if seq.is_multiple_of(8) && op_idx == 0 {
        262_144 + (rng.next_u64() % 786_432) as u32 // 256 KiB ..< 1 MiB
    } else {
        SIZE_CLASSES[((seq as usize).wrapping_mul(31) + op_idx as usize) % SIZE_CLASSES.len()]
    }
}

/// A generated operation over the key universe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenOp {
    Insert(u32, ValueId),
    Remove(u32),
}

/// The replayable model: key index → the value coordinates it holds.
pub type Model = BTreeMap<u32, ValueId>;

/// Generates batch `seq` (1-based) against the model **and applies it** to the
/// model, returning the ops in application order. Deterministic in
/// `(cfg.seed, seq, model-state)`, and the model state is itself a pure
/// function of the applied prefix — so the checker replays exactly.
///
/// Forced mix: op 0 inserts an absent key (guaranteed effective
/// write), op 1 overwrites a present key when one exists, op 2 deletes a
/// present key when one exists; remaining ops are random inserts/removes.
pub fn gen_batch(model: &mut Model, cfg: &Config, seq: u64) -> Vec<GenOp> {
    let mut rng = rng_for(cfg.seed, seq);
    let mut ops = Vec::with_capacity(cfg.batch_ops as usize);
    for op_idx in 0..cfg.batch_ops {
        let r = rng.next_u64();
        let op = match op_idx {
            0 => {
                // Insert an absent key if any; else overwrite (still a write).
                let k = pick_absent(model, cfg.keys, r).unwrap_or((r % cfg.keys as u64) as u32);
                GenOp::Insert(
                    k,
                    ValueId {
                        seq,
                        op_idx,
                        len: value_len(&mut rng, seq, op_idx),
                    },
                )
            }
            1 => match pick_present(model, r) {
                Some(k) => GenOp::Insert(
                    k,
                    ValueId {
                        seq,
                        op_idx,
                        len: value_len(&mut rng, seq, op_idx),
                    },
                ),
                None => GenOp::Insert(
                    (r % cfg.keys as u64) as u32,
                    ValueId {
                        seq,
                        op_idx,
                        len: value_len(&mut rng, seq, op_idx),
                    },
                ),
            },
            2 => match pick_present(model, r) {
                Some(k) => GenOp::Remove(k),
                // Removing an absent key is a legal no-op — still exercised.
                None => GenOp::Remove((r % cfg.keys as u64) as u32),
            },
            _ => {
                let k = (r % cfg.keys as u64) as u32;
                if r & 8 == 0 {
                    GenOp::Remove(k)
                } else {
                    GenOp::Insert(
                        k,
                        ValueId {
                            seq,
                            op_idx,
                            len: value_len(&mut rng, seq, op_idx),
                        },
                    )
                }
            }
        };
        match op {
            GenOp::Insert(k, v) => {
                model.insert(k, v);
            }
            GenOp::Remove(k) => {
                model.remove(&k);
            }
        }
        ops.push(op);
    }
    ops
}

fn pick_absent(model: &Model, keys: u32, r: u64) -> Option<u32> {
    let start = (r % keys as u64) as u32;
    (0..keys)
        .map(|d| (start + d) % keys)
        .find(|k| !model.contains_key(k))
}

fn pick_present(model: &Model, r: u64) -> Option<u32> {
    if model.is_empty() {
        return None;
    }
    let idx = (r % model.len() as u64) as usize;
    model.keys().nth(idx).copied()
}

/// The intent digest: CRC32C over a canonical encoding of the batch's ops.
/// Cross-checks that workload and checker generated the same batch for a
/// coordinate — it is not the expected value.
pub fn batch_digest(ops: &[GenOp]) -> u32 {
    let mut buf = Vec::new();
    for op in ops {
        match op {
            GenOp::Insert(k, v) => {
                buf.push(1u8);
                buf.extend_from_slice(&k.to_le_bytes());
                buf.extend_from_slice(&v.seq.to_le_bytes());
                buf.extend_from_slice(&v.op_idx.to_le_bytes());
                buf.extend_from_slice(&v.len.to_le_bytes());
            }
            GenOp::Remove(k) => {
                buf.push(2u8);
                buf.extend_from_slice(&k.to_le_bytes());
            }
        }
    }
    crc32c::crc32c(&buf)
}

// ---------------------------------------------------------------------------
// Journal records
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Record {
    Header(Config),
    Intent {
        seq: u64,
        txid: u64,
        digest: u32,
    },
    PostApply {
        seq: u64,
        txid: u64,
    },
    Ack {
        txid: u64,
    },
    Maint {
        begin: bool,
        kind: MaintKind,
        ordinal: u64,
    },
    Ready {
        ack_txid: u64,
        groups: u64,
        checkpoints: u64,
        compactions: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaintKind {
    Checkpoint,
    Compact,
}

impl MaintKind {
    fn token(self) -> &'static str {
        match self {
            MaintKind::Checkpoint => "checkpoint",
            MaintKind::Compact => "compact",
        }
    }
}

/// Encodes a record as one complete journal line (with CRC framing and the
/// terminating newline), ready for a single `write_all`.
pub fn encode_line(rec: &Record) -> Vec<u8> {
    let payload = match rec {
        Record::Header(c) => format!(
            "H {} {} {} {} {} {} {} vp1 {}",
            SCHEMA, c.run_id, c.backend, c.seed, c.keys, c.batch_ops, c.group, c.max_wal_bytes
        ),
        Record::Intent { seq, txid, digest } => format!("I {seq} {txid} {digest:08x}"),
        Record::PostApply { seq, txid } => format!("P {seq} {txid}"),
        Record::Ack { txid } => format!("F {txid}"),
        Record::Maint {
            begin,
            kind,
            ordinal,
        } => format!(
            "M {} {} {ordinal}",
            if *begin { "begin" } else { "done" },
            kind.token()
        ),
        Record::Ready {
            ack_txid,
            groups,
            checkpoints,
            compactions,
        } => format!("R {ack_txid} {groups} {checkpoints} {compactions}"),
    };
    let mut line = String::new();
    writeln!(line, "{:08x} {payload}", crc32c::crc32c(payload.as_bytes())).unwrap();
    line.into_bytes()
}

/// A parse failure is journal corruption: the reason is a stable code for the
/// checker's FAIL verdict line.
#[derive(Debug, PartialEq, Eq)]
pub struct JournalError(pub &'static str);

/// Parses a journal byte stream under the strict grammar. Returns the records
/// and whether a torn (newline-less) tail was ignored.
pub fn parse_journal(bytes: &[u8]) -> Result<(Vec<Record>, bool), JournalError> {
    let mut records = Vec::new();
    let mut rest = bytes;
    let mut torn = false;
    while !rest.is_empty() {
        match rest.iter().position(|&b| b == b'\n') {
            Some(nl) => {
                let line = &rest[..nl];
                rest = &rest[nl + 1..];
                records.push(parse_line(line)?);
            }
            None => {
                // The single permitted torn tail — ignored, never validated.
                torn = true;
                break;
            }
        }
    }
    validate_order(&records)?;
    Ok((records, torn))
}

fn parse_line(line: &[u8]) -> Result<Record, JournalError> {
    let line = std::str::from_utf8(line).map_err(|_| JournalError("journal-not-utf8"))?;
    let (crc_hex, payload) = line
        .split_once(' ')
        .ok_or(JournalError("journal-missing-crc"))?;
    let crc = u32::from_str_radix(crc_hex, 16).map_err(|_| JournalError("journal-bad-crc-hex"))?;
    if crc != crc32c::crc32c(payload.as_bytes()) {
        return Err(JournalError("journal-crc-mismatch"));
    }
    let t: Vec<&str> = payload.split(' ').collect();
    let u = |s: &str| {
        s.parse::<u64>()
            .map_err(|_| JournalError("journal-bad-int"))
    };
    match t.first().copied() {
        Some("H") => {
            if t.len() != 10 || t[8] != "vp1" {
                return Err(JournalError("journal-bad-header"));
            }
            if u(t[1])? != SCHEMA as u64 {
                return Err(JournalError("journal-schema"));
            }
            // Checked narrowing + the same minimum invariants the workload CLI
            // enforces: a checksum-valid but out-of-range header must be a
            // stable FAIL, never a silent wrap that panics replay.
            let narrow = |s: &str| -> Result<u32, JournalError> {
                u32::try_from(u(s)?).map_err(|_| JournalError("journal-bad-header"))
            };
            let (keys, batch_ops, group) = (narrow(t[5])?, narrow(t[6])?, narrow(t[7])?);
            if keys == 0 || batch_ops < 3 || group == 0 {
                return Err(JournalError("journal-bad-header"));
            }
            Ok(Record::Header(Config {
                run_id: t[2].to_string(),
                backend: t[3].to_string(),
                seed: u(t[4])?,
                keys,
                batch_ops,
                group,
                max_wal_bytes: u(t[9])?,
            }))
        }
        Some("I") if t.len() == 4 => Ok(Record::Intent {
            seq: u(t[1])?,
            txid: u(t[2])?,
            digest: u32::from_str_radix(t[3], 16).map_err(|_| JournalError("journal-bad-int"))?,
        }),
        Some("P") if t.len() == 3 => Ok(Record::PostApply {
            seq: u(t[1])?,
            txid: u(t[2])?,
        }),
        Some("F") if t.len() == 2 => Ok(Record::Ack { txid: u(t[1])? }),
        Some("M") if t.len() == 4 => Ok(Record::Maint {
            begin: match t[1] {
                "begin" => true,
                "done" => false,
                _ => return Err(JournalError("journal-bad-maint")),
            },
            kind: match t[2] {
                "checkpoint" => MaintKind::Checkpoint,
                "compact" => MaintKind::Compact,
                _ => return Err(JournalError("journal-bad-maint")),
            },
            ordinal: u(t[3])?,
        }),
        Some("R") if t.len() == 5 => Ok(Record::Ready {
            ack_txid: u(t[1])?,
            groups: u(t[2])?,
            checkpoints: u(t[3])?,
            compactions: u(t[4])?,
        }),
        _ => Err(JournalError("journal-unknown-record")),
    }
}

/// Structural rules: header first (exactly once), intents contiguous from 1
/// with `txid == seq`, ACKs monotone and never above the intent frontier,
/// ready at most once, maintenance ordinals per-kind contiguous with no
/// nested/unpaired `begin` except a single trailing one (the cut window).
///
/// The `R` record is validated against the journal prefix, not taken on
/// faith: its `ack_txid` must equal the running ACK
/// frontier, its `groups` the count of `F` records, its checkpoint/compaction
/// counts the completed `M` pairs, and it must not sit inside an open
/// maintenance interval — a fabricated coverage record is journal corruption.
fn validate_order(records: &[Record]) -> Result<(), JournalError> {
    let mut saw_header = false;
    let mut next_seq = 1u64;
    let mut max_ack = 0u64;
    let mut acks = 0u64;
    let mut done_pairs = [0u64, 0u64]; // completed [checkpoint, compact]
    let mut saw_ready = false;
    let mut open_maint: Option<(MaintKind, u64)> = None;
    let mut next_ord = [1u64, 1u64]; // [checkpoint, compact]
    for (i, rec) in records.iter().enumerate() {
        if i == 0 && !matches!(rec, Record::Header(_)) {
            return Err(JournalError("journal-no-header"));
        }
        match rec {
            Record::Header(_) => {
                if saw_header {
                    return Err(JournalError("journal-duplicate-header"));
                }
                saw_header = true;
            }
            Record::Intent { seq, txid, .. } => {
                if *seq != next_seq || *txid != *seq {
                    return Err(JournalError("journal-intent-sequence"));
                }
                next_seq += 1;
            }
            Record::PostApply { seq, txid } => {
                if *seq >= next_seq || *txid != *seq {
                    return Err(JournalError("journal-postapply-order"));
                }
            }
            Record::Ack { txid } => {
                if *txid < max_ack || *txid >= next_seq {
                    return Err(JournalError("journal-ack-order"));
                }
                max_ack = *txid;
                acks += 1;
            }
            Record::Maint {
                begin,
                kind,
                ordinal,
            } => {
                let slot = match kind {
                    MaintKind::Checkpoint => 0,
                    MaintKind::Compact => 1,
                };
                if *begin {
                    if open_maint.is_some() || *ordinal != next_ord[slot] {
                        return Err(JournalError("journal-maint-order"));
                    }
                    open_maint = Some((*kind, *ordinal));
                    next_ord[slot] += 1;
                } else {
                    if open_maint != Some((*kind, *ordinal)) {
                        return Err(JournalError("journal-maint-order"));
                    }
                    open_maint = None;
                    done_pairs[slot] += 1;
                }
            }
            Record::Ready {
                ack_txid,
                groups,
                checkpoints,
                compactions,
            } => {
                if saw_ready {
                    return Err(JournalError("journal-duplicate-ready"));
                }
                if open_maint.is_some()
                    || *ack_txid != max_ack
                    || *groups != acks
                    || *checkpoints != done_pairs[0]
                    || *compactions != done_pairs[1]
                {
                    return Err(JournalError("journal-ready-mismatch"));
                }
                saw_ready = true;
            }
        }
    }
    if !saw_header {
        return Err(JournalError("journal-no-header"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            run_id: "testrun".into(),
            backend: "direct".into(),
            seed: 42,
            keys: 64,
            batch_ops: 6,
            group: 4,
            max_wal_bytes: 64 << 20,
        }
    }

    /// The generator is a pure function of `(seed, prefix)`: two independent
    /// replays produce identical ops, digests, and model states.
    #[test]
    fn generator_replays_identically() {
        let c = cfg();
        let (mut m1, mut m2) = (Model::new(), Model::new());
        for seq in 1..=40 {
            let a = gen_batch(&mut m1, &c, seq);
            let b = gen_batch(&mut m2, &c, seq);
            assert_eq!(a, b);
            assert_eq!(batch_digest(&a), batch_digest(&b));
        }
        assert_eq!(m1, m2);
        assert!(!m1.is_empty());
    }

    /// Every forced ingredient appears in a short campaign: insert-absent,
    /// overwrite-present, delete-present, and every size class incl. a large
    /// (>256 KiB) value and the exact inline-limit edge pair.
    #[test]
    fn forced_mix_covers_ops_and_size_classes() {
        let c = cfg();
        let mut model = Model::new();
        let (mut overwrote, mut deleted_present, mut lens) = (false, false, Vec::new());
        for seq in 1..=40 {
            let before = model.clone();
            for op in gen_batch(&mut model.clone(), &c, seq) {
                match op {
                    GenOp::Insert(k, v) => {
                        if before.contains_key(&k) {
                            overwrote = true;
                        }
                        lens.push(v.len);
                    }
                    GenOp::Remove(k) => {
                        if before.contains_key(&k) {
                            deleted_present = true;
                        }
                    }
                }
            }
            gen_batch(&mut model, &c, seq);
        }
        assert!(overwrote && deleted_present);
        for want in SIZE_CLASSES {
            assert!(lens.contains(&want), "size class {want} never generated");
        }
        assert!(lens.iter().any(|&l| l >= 262_144));
    }

    #[test]
    fn journal_round_trips_and_rejects_corruption() {
        let mut bytes = Vec::new();
        let recs = vec![
            Record::Header(cfg()),
            Record::Intent {
                seq: 1,
                txid: 1,
                digest: 0xDEAD_BEEF,
            },
            Record::PostApply { seq: 1, txid: 1 },
            Record::Ack { txid: 1 },
            Record::Maint {
                begin: true,
                kind: MaintKind::Checkpoint,
                ordinal: 1,
            },
            Record::Maint {
                begin: false,
                kind: MaintKind::Checkpoint,
                ordinal: 1,
            },
            Record::Ready {
                ack_txid: 1,
                groups: 1,
                checkpoints: 1,
                compactions: 0,
            },
        ];
        for r in &recs {
            bytes.extend_from_slice(&encode_line(r));
        }
        let (parsed, torn) = parse_journal(&bytes).unwrap();
        assert_eq!(parsed, recs);
        assert!(!torn);

        // A newline-less tail is tolerated (exactly once, by construction)…
        let mut torn_bytes = bytes.clone();
        torn_bytes.extend_from_slice(b"deadbeef I 2 2 0000");
        let (parsed, torn) = parse_journal(&torn_bytes).unwrap();
        assert_eq!(parsed.len(), recs.len());
        assert!(torn);

        // …but a corrupt *complete* line fails, wherever it sits.
        let mut bad = bytes.clone();
        let flip = bad.len() / 2;
        bad[flip] ^= 0x01;
        assert!(parse_journal(&bad).is_err());

        // A sequence gap fails.
        let mut gap = bytes.clone();
        gap.extend_from_slice(&encode_line(&Record::Intent {
            seq: 3,
            txid: 3,
            digest: 0,
        }));
        assert_eq!(
            parse_journal(&gap).unwrap_err(),
            JournalError("journal-intent-sequence")
        );
    }

    /// A fabricated coverage record is corruption: `R` must agree with the
    /// journal prefix on every field.
    #[test]
    fn ready_record_is_validated_against_the_prefix() {
        let base = [
            Record::Header(cfg()),
            Record::Intent {
                seq: 1,
                txid: 1,
                digest: 0,
            },
            Record::Ack { txid: 1 },
        ];
        // Honest R: ack 1, one group, no maintenance.
        let good = Record::Ready {
            ack_txid: 1,
            groups: 1,
            checkpoints: 0,
            compactions: 0,
        };
        let mut ok = Vec::new();
        for r in base.iter().chain([&good]) {
            ok.extend_from_slice(&encode_line(r));
        }
        assert!(parse_journal(&ok).is_ok());
        // Any inflated field fails.
        for bad in [
            Record::Ready {
                ack_txid: 2,
                groups: 1,
                checkpoints: 0,
                compactions: 0,
            },
            Record::Ready {
                ack_txid: 1,
                groups: 3,
                checkpoints: 0,
                compactions: 0,
            },
            Record::Ready {
                ack_txid: 1,
                groups: 1,
                checkpoints: 1,
                compactions: 0,
            },
            Record::Ready {
                ack_txid: 1,
                groups: 1,
                checkpoints: 0,
                compactions: 1,
            },
        ] {
            let mut bytes = Vec::new();
            for r in base.iter().chain([&bad]) {
                bytes.extend_from_slice(&encode_line(r));
            }
            assert_eq!(
                parse_journal(&bytes).unwrap_err(),
                JournalError("journal-ready-mismatch")
            );
        }
        // R inside an open maintenance interval fails.
        let mut in_maint = Vec::new();
        for r in base.iter().chain([
            &Record::Maint {
                begin: true,
                kind: MaintKind::Compact,
                ordinal: 1,
            },
            &good,
        ]) {
            in_maint.extend_from_slice(&encode_line(r));
        }
        assert_eq!(
            parse_journal(&in_maint).unwrap_err(),
            JournalError("journal-ready-mismatch")
        );
    }

    /// A checksum-valid but out-of-range header is a stable FAIL, never a
    /// silent wrap.
    #[test]
    fn header_integers_are_strict() {
        for (keys, batch_ops, group) in [
            (1u64 << 32, 6, 4), // u32 overflow wraps to 0 without the check
            (0, 6, 4),          // zero universe
            (64, 2, 4),         // below the forced-mix minimum
            (64, 6, 0),         // zero group
            (u64::MAX, 6, 4),   // overflow
        ] {
            let payload = format!("H 1 r direct 42 {keys} {batch_ops} {group} vp1 67108864");
            let line = format!("{:08x} {payload}\n", crc32c::crc32c(payload.as_bytes()));
            assert_eq!(
                parse_journal(line.as_bytes()).unwrap_err(),
                JournalError("journal-bad-header"),
                "keys={keys} batch_ops={batch_ops} group={group}"
            );
        }
    }
}
