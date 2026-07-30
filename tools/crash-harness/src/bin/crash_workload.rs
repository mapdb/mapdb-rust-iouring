//! Crash-tier workload: writes forever until SIGKILLed, under the
//! write-ahead intent journal protocol of `crash_harness` — every batch's
//! intent is durable in the journal (on a *different* filesystem) before its
//! apply is enqueued, and a durability ACK is journaled only after the
//! backend barrier returned. `crash_check` replays the journal against the
//! recovered store.
//!
//! Group commit is real: a group's
//! permits are reserved first, then `WritePermit::apply` enqueues each batch
//! synchronously before any outcome is awaited, giving the writer a genuine
//! pending window; one barrier per group ACKs the frontier. Maintenance is
//! explicit — WAL checkpoint and both-backend compaction on a fixed group
//! cadence, bracketed by journaled `M begin`/`M done` markers.
//!
//! Readiness: only after the header, the run-id batch plus two
//! later durable groups, one completed WAL checkpoint (WAL) and one completed
//! compaction does the workload journal an `R` record and create the
//! `<journal>.ready` sentinel the harness scripts wait for. A cut before
//! readiness fails the round rather than passing vacuously.

use mapdb_uring::backend::{Backend, Direct, Wal};
use mapdb_uring::batch::WriteBatch;
use mapdb_uring::error::OpenError;
use mapdb_uring::map::BTreeMap;
use mapdb_uring::store::{Options, Store};
use mapdb_uring::version::Version;
use mapdb_uring::ApplyOutcome;
use mapdb_uring_crash_harness::{self as ch, Config, GenOp, MaintKind, Model, Record, RUN_ID_KEY};

use futures_lite::future::block_on;
use std::future::Future;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The tree-depth bound `BTreeMap::apply` itself charges with (store.rs
/// `DEPTH_BOUND`); the permit's cost must be computed identically or
/// `WritePermit::apply` reports a mismatch.
const DEPTH_BOUND: u32 = 8;

/// Maintenance cadence in durable groups: checkpoint early and
/// often, compaction less frequently; both before the readiness threshold.
const CHECKPOINT_EVERY: u64 = 4; // groups 1, 5, 9, …
const COMPACT_EVERY: u64 = 12; // groups 3, 15, 27, …

trait HarnessBackend: Backend + Sized + 'static {
    const HAS_CHECKPOINT: bool;
    fn create(path: &Path, opts: Options) -> Result<Store<Self>, OpenError>;
    fn barrier(map: &BTreeMap<Self>, v: Version)
        -> impl Future<Output = Result<(), String>> + Send;
    fn checkpoint(map: &BTreeMap<Self>) -> impl Future<Output = Result<(), String>> + Send;
}

impl HarnessBackend for Direct {
    const HAS_CHECKPOINT: bool = false;
    fn create(path: &Path, opts: Options) -> Result<Store<Self>, OpenError> {
        Store::<Direct>::create_path(path, opts)
    }
    async fn barrier(map: &BTreeMap<Self>, v: Version) -> Result<(), String> {
        map.flush(v).await.map_err(|e| format!("flush: {e:?}"))
    }
    async fn checkpoint(_map: &BTreeMap<Self>) -> Result<(), String> {
        unreachable!("Direct has no checkpoint")
    }
}

impl HarnessBackend for Wal {
    const HAS_CHECKPOINT: bool = true;
    fn create(path: &Path, opts: Options) -> Result<Store<Self>, OpenError> {
        Store::<Wal>::create_path(path, opts)
    }
    async fn barrier(map: &BTreeMap<Self>, v: Version) -> Result<(), String> {
        map.commit(v).await.map_err(|e| format!("commit: {e:?}"))
    }
    async fn checkpoint(map: &BTreeMap<Self>) -> Result<(), String> {
        map.store()
            .checkpoint()
            .await
            .map(|_| ())
            .map_err(|e| format!("checkpoint: {e:?}"))
    }
}

struct Journal {
    file: std::fs::File,
}

impl Journal {
    fn create(path: &Path) -> Result<Journal, String> {
        let file = std::fs::OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("journal create {}: {e}", path.display()))?;
        Ok(Journal { file })
    }
    /// One complete line per `write_all` — the checker tolerates exactly one
    /// torn (newline-less) tail, so a mid-line crash is expected, never fatal.
    fn append(&mut self, rec: &Record) -> Result<(), String> {
        self.file
            .write_all(&ch::encode_line(rec))
            .map_err(|e| format!("journal write: {e}"))
    }
    fn sync(&mut self) -> Result<(), String> {
        self.file
            .sync_data()
            .map_err(|e| format!("journal sync: {e}"))
    }
}

fn build_batch(cfg: &Config, seq: u64, ops: &[GenOp]) -> WriteBatch {
    let mut b = WriteBatch::new();
    if seq == 1 {
        // The run-id marker rides the first batch: a journal can
        // never be checked against another round's store image.
        b = b.insert(RUN_ID_KEY.to_vec(), cfg.run_id.clone().into_bytes());
    }
    for op in ops {
        b = match *op {
            GenOp::Insert(k, v) => b.insert(ch::key_bytes(k), ch::value_bytes(cfg.seed, v)),
            GenOp::Remove(k) => b.remove(ch::key_bytes(k)),
        };
    }
    b
}

async fn run<B: HarnessBackend>(
    cfg: Config,
    store_path: PathBuf,
    journal_path: PathBuf,
) -> Result<(), String> {
    let opts = Options {
        max_wal_bytes: cfg.max_wal_bytes,
        ..Options::default()
    };
    let store = B::create(&store_path, opts).map_err(|e| format!("create: {e:?}"))?;
    let map = BTreeMap::over(store);

    let mut journal = Journal::create(&journal_path)?;
    journal.append(&Record::Header(cfg.clone()))?;
    journal.sync()?;

    let mut model = Model::new();
    let mut seq = 0u64;
    let mut groups = 0u64;
    let (mut checkpoints, mut compactions) = (0u64, 0u64);
    let mut ready = false;

    loop {
        // --- generate the group and journal its intents (write-ahead) ---
        let mut group: Vec<(u64, Vec<GenOp>)> = Vec::with_capacity(cfg.group as usize);
        for _ in 0..cfg.group {
            seq += 1;
            let ops = ch::gen_batch(&mut model, &cfg, seq);
            group.push((seq, ops));
        }
        let batches: Vec<(u64, WriteBatch)> = group
            .iter()
            .map(|(s, ops)| (*s, build_batch(&cfg, *s, ops)))
            .collect();
        // Cap the group against dirty-page admission rather than trusting the
        // CLI: a group that cannot co-reserve is a config error.
        let total_pages: u64 = batches
            .iter()
            .map(|(_, b)| b.cost(DEPTH_BOUND).worst_case_pages as u64)
            .sum();
        let limits = mapdb_uring::admission::AdmissionLimits::default();
        if total_pages > limits.max_dirty_pages as u64 {
            return Err(format!(
                "config: group worst-case pages {total_pages} exceeds admission {}",
                limits.max_dirty_pages
            ));
        }
        for ((s, ops), _) in group.iter().zip(&batches) {
            journal.append(&Record::Intent {
                seq: *s,
                txid: *s,
                digest: ch::batch_digest(ops),
            })?;
        }
        journal.sync()?;

        // --- reserve every permit, then enqueue synchronously in order ---
        let mut reserved = Vec::with_capacity(batches.len());
        for (s, batch) in batches {
            let permit = map
                .reserve_write(batch.cost(DEPTH_BOUND))
                .await
                .map_err(|e| format!("reserve seq {s}: {e:?}"))?;
            reserved.push((s, batch, permit));
        }
        let mut outcomes = Vec::with_capacity(reserved.len());
        for (s, batch, permit) in reserved {
            let fut = permit
                .apply(batch)
                .map_err(|(_, e)| format!("permit mismatch seq {s}: {e:?}"))?;
            outcomes.push((s, fut));
        }
        let mut last_version: Option<Version> = None;
        for (s, fut) in outcomes {
            match fut.await.map_err(|e| format!("apply seq {s}: {e:?}"))? {
                ApplyOutcome::Applied { version, .. } => {
                    // Sole writer on a fresh store: txids are consecutive and
                    // intent txid == seq. Any drift breaks the oracle — abort.
                    if version.txid() != s {
                        return Err(format!("txid drift: seq {s} applied as {}", version.txid()));
                    }
                    journal.append(&Record::PostApply {
                        seq: s,
                        txid: version.txid(),
                    })?;
                    last_version = Some(version);
                }
                other => return Err(format!("unexpected outcome seq {s}: {other:?}")),
            }
        }

        // --- one barrier per group, then the durability ACK ---
        let v = last_version.expect("non-empty group");
        B::barrier(&map, v).await?;
        journal.append(&Record::Ack { txid: v.txid() })?;
        journal.sync()?;
        groups += 1;

        // --- explicit maintenance, journaled around the call ---
        if B::HAS_CHECKPOINT && groups % CHECKPOINT_EVERY == 1 {
            checkpoints += 1;
            journal.append(&Record::Maint {
                begin: true,
                kind: MaintKind::Checkpoint,
                ordinal: checkpoints,
            })?;
            journal.sync()?;
            B::checkpoint(&map).await?;
            journal.append(&Record::Maint {
                begin: false,
                kind: MaintKind::Checkpoint,
                ordinal: checkpoints,
            })?;
            journal.sync()?;
        }
        if groups % COMPACT_EVERY == 3 {
            compactions += 1;
            journal.append(&Record::Maint {
                begin: true,
                kind: MaintKind::Compact,
                ordinal: compactions,
            })?;
            journal.sync()?;
            map.store()
                .compact()
                .await
                .map_err(|e| format!("compact: {e:?}"))?;
            journal.append(&Record::Maint {
                begin: false,
                kind: MaintKind::Compact,
                ordinal: compactions,
            })?;
            journal.sync()?;
        }

        // --- readiness: durable coverage, then the sentinel ---
        if !ready
            && groups >= ch::READY_MIN_GROUPS
            && compactions >= ch::READY_MIN_COMPACTIONS
            && (!B::HAS_CHECKPOINT || checkpoints >= ch::READY_MIN_CHECKPOINTS_WAL)
        {
            journal.append(&Record::Ready {
                ack_txid: v.txid(),
                groups,
                checkpoints,
                compactions,
            })?;
            journal.sync()?;
            let sentinel = sentinel_path(&journal_path);
            std::fs::write(&sentinel, b"ready")
                .and_then(|_| std::fs::File::open(&sentinel)?.sync_all())
                .map_err(|e| format!("sentinel: {e}"))?;
            eprintln!(
                "crash_workload: ready after {groups} groups (ack txid {}, {checkpoints} checkpoints, {compactions} compactions)",
                v.txid()
            );
            ready = true;
        }
    }
}

fn sentinel_path(journal: &Path) -> PathBuf {
    let mut p = journal.as_os_str().to_owned();
    p.push(".ready");
    PathBuf::from(p)
}

fn usage() -> ! {
    eprintln!(
        "usage: crash_workload --backend direct|wal --store <path> --journal <path> \
         --run-id <id> [--seed N] [--keys N] [--batch-ops N] [--group N]"
    );
    std::process::exit(2);
}

fn main() {
    let mut backend = None;
    let mut store = None;
    let mut journal = None;
    let mut run_id = None;
    let (mut seed, mut keys, mut batch_ops, mut group) = (1u64, 512u32, 6u32, 8u32);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--backend" => backend = Some(val()),
            "--store" => store = Some(PathBuf::from(val())),
            "--journal" => journal = Some(PathBuf::from(val())),
            "--run-id" => run_id = Some(val()),
            "--seed" => seed = val().parse().unwrap_or_else(|_| usage()),
            "--keys" => keys = val().parse().unwrap_or_else(|_| usage()),
            "--batch-ops" => batch_ops = val().parse().unwrap_or_else(|_| usage()),
            "--group" => group = val().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    let (Some(backend), Some(store), Some(journal), Some(run_id)) =
        (backend, store, journal, run_id)
    else {
        usage()
    };
    if batch_ops < 3 || group == 0 || keys == 0 || run_id.contains(' ') {
        usage();
    }
    let cfg = Config {
        run_id,
        backend: backend.clone(),
        seed,
        keys,
        batch_ops,
        group,
        max_wal_bytes: 64 << 20,
    };
    let result = match backend.as_str() {
        "direct" => block_on(run::<Direct>(cfg, store, journal)),
        "wal" => block_on(run::<Wal>(cfg, store, journal)),
        _ => usage(),
    };
    // The loop only returns on error; a healthy workload dies by SIGKILL.
    if let Err(e) = result {
        eprintln!("crash_workload: FATAL: {e}");
        std::process::exit(1);
    }
}
