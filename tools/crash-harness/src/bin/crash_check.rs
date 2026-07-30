//! Crash-tier checker: opens the crashed (or crash-imaged) store through the
//! production `open_path`, and holds it to the exhaustive oracle:
//!
//! - `open_path` must succeed — any refusal or corruption verdict FAILs;
//! - `metrics().visible_txid` is the recovered txid `R`, and
//!   `max_ACK <= R <= highest write-ahead intent` must hold (txids compared
//!   as `u64` — reopen mints a fresh incarnation);
//! - the journal must satisfy the strict grammar (contiguous intents with
//!   `txid == seq`, digest agreement with the regenerated batches, one torn
//!   tail at most);
//! - the recovered contents must EXACTLY equal the deterministic replay
//!   through `R`: every key of the declared finite universe is `get` and
//!   compared byte-exact (present or absent), the run-id marker must match
//!   the journal's, and `VerifyReport::entry_count`, `map.len()`, and
//!   `metrics().entry_count` must all equal the replayed count — matching
//!   every expected entry plus the exact verified count rules out an
//!   unexpected extra key even outside the universe.
//!
//! Verdict: one stable stdout line
//! `CRASH_CHECK verdict=PASS|FAIL [reason=<code>] backend=… recovered_txid=…
//! ack_txid=… intents=… entries=… ready_groups=… ready_checkpoints=…
//! ready_compactions=… last_record=… maint_open_at_cut=0|1`; diagnostics on
//! stderr; exit 0 iff PASS. `maint_open_at_cut` reports an unmatched
//! `M begin` at the witness cutoff — NOT proof the cut landed inside
//! checkpoint/compaction (the process may die before the `M done` sync).

use mapdb_uring::backend::{Direct, Wal};
use mapdb_uring::map::BTreeMap;
use mapdb_uring::store::{Options, Store};
use mapdb_uring_crash_harness::{self as ch, Config, Model, Record, RUN_ID_KEY};

use futures_lite::future::block_on;
use std::path::PathBuf;

struct Fail {
    reason: &'static str,
    detail: String,
}

fn fail(reason: &'static str, detail: impl Into<String>) -> Fail {
    Fail {
        reason,
        detail: detail.into(),
    }
}

struct JournalView {
    cfg: Config,
    max_intent: u64,
    max_ack: u64,
    /// The prefix-validated `R` record: (ack_txid, groups, checkpoints,
    /// compactions). Coverage is REQUIRED from these fields, not assumed.
    ready: Option<(u64, u64, u64, u64)>,
    /// A trailing `M begin` without its `done` at the witness cutoff. This
    /// does NOT prove the cut landed inside checkpoint/compaction — the
    /// process may have died between the call returning and the `M done`
    /// sync — hence the honest name.
    maint_open_at_cut: bool,
    /// The type token of the last complete journal record (result metadata).
    last_record: &'static str,
    intents: Vec<(u64, u32)>, // (seq == txid, digest)
}

fn load_journal(path: &PathBuf) -> Result<JournalView, Fail> {
    let bytes = std::fs::read(path).map_err(|e| fail("journal-read", e.to_string()))?;
    let (records, _torn) =
        ch::parse_journal(&bytes).map_err(|e| fail(e.0, "journal grammar violation"))?;
    let mut cfg = None;
    let mut intents = Vec::new();
    let mut max_ack = 0;
    let mut ready = None;
    let mut open_maint = false;
    let mut last_record = "none";
    for rec in records {
        last_record = match &rec {
            Record::Header(_) => "H",
            Record::Intent { .. } => "I",
            Record::PostApply { .. } => "P",
            Record::Ack { .. } => "F",
            Record::Maint { .. } => "M",
            Record::Ready { .. } => "R",
        };
        match rec {
            Record::Header(c) => cfg = Some(c),
            Record::Intent { seq, digest, .. } => intents.push((seq, digest)),
            Record::PostApply { .. } => {}
            Record::Ack { txid } => max_ack = txid,
            Record::Maint { begin, .. } => open_maint = begin,
            Record::Ready {
                ack_txid,
                groups,
                checkpoints,
                compactions,
            } => ready = Some((ack_txid, groups, checkpoints, compactions)),
        }
    }
    let cfg = cfg.ok_or_else(|| fail("journal-no-header", "missing header"))?;
    Ok(JournalView {
        max_intent: intents.last().map(|(s, _)| *s).unwrap_or(0),
        max_ack,
        ready,
        maint_open_at_cut: open_maint,
        last_record,
        intents,
        cfg,
    })
}

/// Replays the generator through `upto`, verifying every intent digest along
/// the way (the workload and checker must agree batch-for-batch), and returns
/// the model at exactly `upto`.
fn replay(view: &JournalView, upto: u64) -> Result<Model, Fail> {
    let mut model = Model::new();
    let mut at_upto = Model::new();
    for &(seq, digest) in &view.intents {
        let ops = ch::gen_batch(&mut model, &view.cfg, seq);
        if ch::batch_digest(&ops) != digest {
            return Err(fail(
                "intent-digest-mismatch",
                format!("seq {seq}: regenerated batch disagrees with journal"),
            ));
        }
        if seq == upto {
            at_upto = model.clone();
        }
    }
    if upto == 0 {
        return Err(fail("recovered-txid-zero", "no committed state to check"));
    }
    Ok(at_upto)
}

enum AnyMap {
    Direct(BTreeMap<Direct>),
    Wal(BTreeMap<Wal>),
}

impl AnyMap {
    fn open(backend: &str, store: &PathBuf) -> Result<AnyMap, Fail> {
        let opts = Options::default();
        match backend {
            "direct" => Store::<Direct>::open_path(store, opts)
                .map(|s| AnyMap::Direct(BTreeMap::over(s)))
                .map_err(|e| fail("open", format!("{e:?}"))),
            "wal" => Store::<Wal>::open_path(store, opts)
                .map(|s| AnyMap::Wal(BTreeMap::over(s)))
                .map_err(|e| fail("open", format!("{e:?}"))),
            _ => Err(fail("config", "unknown backend")),
        }
    }
    fn visible_txid(&self) -> u64 {
        match self {
            AnyMap::Direct(m) => m.metrics().visible_txid,
            AnyMap::Wal(m) => m.metrics().visible_txid,
        }
    }
    fn entry_counts(&self) -> Result<(u64, u64, u64), Fail> {
        let (verify, len, metrics) = match self {
            AnyMap::Direct(m) => (
                block_on(m.verify()).map_err(|e| fail("verify", format!("{e:?}")))?,
                m.len(),
                m.metrics().entry_count,
            ),
            AnyMap::Wal(m) => (
                block_on(m.verify()).map_err(|e| fail("verify", format!("{e:?}")))?,
                m.len(),
                m.metrics().entry_count,
            ),
        };
        Ok((verify.entry_count, len, metrics))
    }
    fn get(&self, key: Vec<u8>) -> Result<Option<Vec<u8>>, Fail> {
        match self {
            AnyMap::Direct(m) => block_on(m.get(key)),
            AnyMap::Wal(m) => block_on(m.get(key)),
        }
        .map_err(|e| fail("read", format!("{e:?}")))
    }
    fn close(self) -> Result<(), Fail> {
        match self {
            AnyMap::Direct(m) => block_on(m.close()),
            AnyMap::Wal(m) => block_on(m.close()),
        }
        .map_err(|e| fail("close", format!("{e:?}")))
    }
}

fn check(
    backend: &str,
    store: &PathBuf,
    journal: &PathBuf,
    min_ack: u64,
) -> Result<(JournalView, u64, u64), Fail> {
    let view = load_journal(journal)?;
    if view.cfg.backend != backend {
        return Err(fail("config", "journal backend disagrees with --backend"));
    }
    // Coverage is enforced from the prefix-validated R record itself, per
    // backend — a journal without real maintenance coverage cannot pass,
    // whatever its other records claim.
    let Some((_r_ack, r_groups, r_ckpts, r_compacts)) = view.ready else {
        return Err(fail("not-ready", "cut before the readiness record"));
    };
    if r_groups < ch::READY_MIN_GROUPS
        || r_compacts < ch::READY_MIN_COMPACTIONS
        || (backend == "wal" && r_ckpts < ch::READY_MIN_CHECKPOINTS_WAL)
    {
        return Err(fail(
            "coverage",
            format!(
                "R records groups={r_groups} checkpoints={r_ckpts} compactions={r_compacts} \
                 below the readiness policy"
            ),
        ));
    }
    if view.max_ack < min_ack {
        return Err(fail(
            "min-ack",
            format!("ack frontier {} below required {min_ack}", view.max_ack),
        ));
    }

    let map = AnyMap::open(backend, store)?;
    let recovered = map.visible_txid();
    if recovered < view.max_ack || recovered > view.max_intent {
        map.close().ok();
        return Err(fail(
            "recovered-txid-range",
            format!(
                "recovered {recovered} outside [ack {}, intent {}]",
                view.max_ack, view.max_intent
            ),
        ));
    }
    let model = replay(&view, recovered)?;

    // Run-id marker: this store image belongs to this journal.
    match map.get(RUN_ID_KEY.to_vec())? {
        Some(v) if v == view.cfg.run_id.as_bytes() => {}
        other => {
            map.close().ok();
            return Err(fail(
                "run-id",
                format!("marker {other:?} != {:?}", view.cfg.run_id),
            ));
        }
    }

    // Exhaustive universe sweep: byte-exact presence and absence.
    for k in 0..view.cfg.keys {
        let key = ch::key_bytes(k);
        let got = map.get(key.clone())?;
        let want = model
            .get(&k)
            .map(|&vid| ch::value_bytes(view.cfg.seed, vid));
        if got != want {
            map.close().ok();
            return Err(fail(
                "state-mismatch",
                format!(
                    "key {k}: got {:?}, want {:?}",
                    got.map(|v| v.len()),
                    want.map(|v| v.len())
                ),
            ));
        }
    }

    // Exact-count triple: rules out unexpected extra keys anywhere.
    let expected = model.len() as u64 + 1; // + run-id marker
    let (vc, lc, mc) = map.entry_counts()?;
    if vc != expected || lc != expected || mc != expected {
        map.close().ok();
        return Err(fail(
            "count-mismatch",
            format!("verify {vc} / len {lc} / metrics {mc} != replayed {expected}"),
        ));
    }

    let entries = expected;
    map.close()?;
    Ok((view, recovered, entries))
}

fn usage() -> ! {
    eprintln!(
        "usage: crash_check --backend direct|wal --store <path> --journal <path> [--min-ack N]"
    );
    std::process::exit(2);
}

fn main() {
    let mut backend = None;
    let mut store = None;
    let mut journal = None;
    let mut min_ack = 1u64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--backend" => backend = Some(val()),
            "--store" => store = Some(PathBuf::from(val())),
            "--journal" => journal = Some(PathBuf::from(val())),
            "--min-ack" => min_ack = val().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    let (Some(backend), Some(store), Some(journal)) = (backend, store, journal) else {
        usage()
    };
    match check(&backend, &store, &journal, min_ack) {
        Ok((view, recovered, entries)) => {
            let (_, r_groups, r_ckpts, r_compacts) = view.ready.unwrap_or_default();
            println!(
                "CRASH_CHECK verdict=PASS backend={backend} recovered_txid={recovered} \
                 ack_txid={} intents={} entries={entries} ready_groups={r_groups} \
                 ready_checkpoints={r_ckpts} ready_compactions={r_compacts} \
                 last_record={} maint_open_at_cut={}",
                view.max_ack,
                view.max_intent,
                view.last_record,
                u8::from(view.maint_open_at_cut)
            );
        }
        Err(f) => {
            eprintln!("crash_check: {}: {}", f.reason, f.detail);
            println!(
                "CRASH_CHECK verdict=FAIL reason={} backend={backend}",
                f.reason
            );
            std::process::exit(1);
        }
    }
}
