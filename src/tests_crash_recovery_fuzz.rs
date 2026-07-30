//! Property fuzzing over batch/op sequences crossed with crash points.
//!
//! The other differentials fuzz op sequences against `std::BTreeMap` but never
//! crash, and the hand-written WAL crash tests crash over fixed sequences. This
//! module closes the cross product: a random multi-op batch sequence with random
//! commit points, then a random — possibly torn — crash image of the WAL, and
//! the two recovery invariants that must hold for every such image.
//!
//! Since an acknowledged commit is synced before its ack, and a crash image only
//! rewrites the unsynced tail: (1) every committed op must survive and resolve
//! to its committed value, and (2) the recovered tree must be structurally
//! sound — `verify` passes, `entry_count` is self-consistent, and no phantom key
//! outside the written domain appears. Uncommitted-but-applied ops may or may
//! not survive; both are legal, so the invariants bound recovery from below
//! rather than pinning equality.

use crate::backend::Wal;
use crate::batch::{ApplyOutcome, WriteBatch};
use crate::io::{FakeIo, PageIo, Persist};
use crate::map::BTreeMap;
use crate::store::{Options, Store};
use futures_lite::future::block_on;
use proptest::prelude::*;
use std::collections::BTreeMap as StdMap;
use std::sync::Arc;

/// The key domain is small so sequences collide (overwrites, remove-then-insert)
/// — the interesting cases for recovery — and probing the whole domain is cheap.
const KEY_DOMAIN: u16 = 32;

fn value_for(seed: u16) -> Vec<u8> {
    // Big enough to span/segment records and force tree structure.
    vec![(seed & 0xff) as u8; 200 + (seed as usize % 400)]
}

fn dev(a: &Arc<FakeIo>) -> Arc<dyn PageIo> {
    Arc::clone(a) as Arc<dyn PageIo>
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 160, ..ProptestConfig::default() })]

    /// A sequence of multi-op batches, each optionally committed, applied to a
    /// StoreWal; then a crash image of the WAL at a random granularity with a
    /// per-granule Old/New/Garble decision.
    ///
    /// The precise invariant (a subtlety the fuzzer itself surfaced): this store
    /// appends every `apply` to the WAL *immediately*; `commit` only forces the
    /// fdatasync. Recovery replays the longest valid **record-chain prefix**, so a
    /// batch that was applied-but-not-committed still recovers *if its record
    /// happens to persist* — and can then supersede a key an earlier committed
    /// batch set. So the guarantee is **not** "committed values survive verbatim";
    /// it is:
    ///
    ///  1. Recovery yields exactly the state after some whole-batch prefix
    ///     `batches[0..k]` — never a partial batch, a reordering, or a phantom.
    ///  2. That prefix reaches **at least** the last commit (`k ≥ committed_len`):
    ///     an acknowledged commit is synced, so it is never torn away.
    ///  3. The recovered tree is structurally sound (`verify`).
    #[test]
    fn recovery_yields_a_valid_prefix_at_least_the_committed_floor(
        // Each batch: 1–4 ops. 0–40 batches.
        batches in prop::collection::vec(
            (
                prop::collection::vec(
                    (0u16..KEY_DOMAIN, prop::option::of(any::<u16>())),
                    1..=4,
                ),
                any::<bool>(), // commit this batch's version durable?
            ),
            0..40,
        ),
        // Crash-image shape.
        radius_pick in 0usize..4,       // blast granule: 512 / 4096 / 64 / 1 pages-ish
        decide_seed in any::<u64>(),
    ) {
        let data = Arc::new(FakeIo::new(0));
        let wal = Arc::new(FakeIo::new(0));

        // `prefix_states[i]` = ground-truth state after applying `i` batches
        // (index 0 = empty). Recovery must reproduce one of these. `committed_len`
        // is the batch count as of the last durable commit — the floor recovery
        // may not drop below.
        let mut prefix_states: Vec<StdMap<u16, Vec<u8>>> = vec![StdMap::new()];
        let mut live: StdMap<u16, Vec<u8>> = StdMap::new();
        let mut committed_len: usize = 0;
        let img: Vec<u8>;
        let data_img: Vec<u8>;

        {
            let map = BTreeMap::over(
                Store::<Wal>::create_with_io(dev(&data), dev(&wal), Options::default()).unwrap(),
            );
            block_on(async {
                // NB the crash image is captured from the *live* device below, before
                // `close()` — `close()` would fdatasync the whole WAL and so make even
                // uncommitted appends durable, collapsing the committed/uncommitted
                // distinction this test turns on.
                for (ops, do_commit) in &batches {
                    // Build one batch from the ops, folding intra-batch into `live`
                    // so a duplicate key in the batch resolves to its last write
                    // (batch semantics = last op wins for a key).
                    let mut b = WriteBatch::new();
                    let mut batch_effect: StdMap<u16, Option<Vec<u8>>> = StdMap::new();
                    for (k, maybe_v) in ops {
                        let key = k.to_be_bytes().to_vec();
                        match maybe_v {
                            Some(seed) => {
                                let v = value_for(*seed);
                                b = b.insert(key, v.clone());
                                batch_effect.insert(*k, Some(v));
                            }
                            None => {
                                b = b.remove(key);
                                batch_effect.insert(*k, None);
                            }
                        }
                    }
                    let out = map.apply(b).await.unwrap();
                    if let ApplyOutcome::Applied { version, .. } = out {
                        // Fold the batch's net effect into the live truth and record
                        // this new whole-batch prefix state.
                        for (k, eff) in batch_effect {
                            match eff {
                                Some(v) => { live.insert(k, v); }
                                None => { live.remove(&k); }
                            }
                        }
                        prefix_states.push(live.clone());
                        if *do_commit {
                            map.commit(version).await.unwrap();
                            committed_len = prefix_states.len() - 1;
                        }
                    }
                }
            });

            // Capture the crash image + durable data image from the LIVE devices,
            // *before* `close()`: an acknowledged commit is already synced (so it is
            // in `synced_bytes`), while the uncommitted tail is still unsynced and so
            // subject to `crash_image`'s per-granule Old/New/Garble. `crash_image`
            // wants a pure `Fn`, so each granule's fate is a splitmix hash of
            // (granule, seed).
            let radius = [512usize, 4096, 64, 1][radius_pick];
            img = wal.crash_image(radius, |granule| {
                let mut z = (granule as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(decide_seed);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 27;
                match z % 3 {
                    0 => Persist::Old,
                    1 => Persist::New,
                    _ => Persist::Garble,
                }
            });
            data_img = data.synced_bytes();
            block_on(map.close()).unwrap();
        }

        // Recover over the durable data image + the crashed WAL image.
        let d = Arc::new(FakeIo::from_vec(data_img));
        let w = Arc::new(FakeIo::from_vec(img));
        let store = Store::<Wal>::open_over(dev(&d), dev(&w), Options::default())
            .expect("a crash confined to the unsynced WAL tail must still recover");
        let map = BTreeMap::over(store);

        block_on(async {
            // Invariant 3: structurally sound.
            let report = map.verify().await.expect("recovered tree verifies");
            let len = map.len();
            prop_assert_eq!(report.entry_count, len, "verify and len must agree");

            // Materialise the recovered state by probing the whole (small) domain —
            // this also catches any in-domain phantom, and `present == len` catches
            // an out-of-domain one (it would inflate `len` without a probe hit).
            let mut recovered: StdMap<u16, Vec<u8>> = StdMap::new();
            for k in 0..KEY_DOMAIN {
                if let Some(v) = map.get(k.to_be_bytes().to_vec()).await.unwrap() {
                    recovered.insert(k, v);
                }
            }
            prop_assert_eq!(recovered.len() as u64, len,
                "every resident entry must be an in-domain key — no phantom survived");

            // Invariants 1 & 2: the recovered state is exactly some whole-batch
            // prefix, and that prefix is at least the committed floor.
            let matched_k = (committed_len..prefix_states.len())
                .find(|&k| prefix_states[k] == recovered);
            prop_assert!(
                matched_k.is_some(),
                "recovered state matches no valid prefix in [{committed_len}, {}] — \
                 recovery produced a non-prefix or sub-committed state",
                prefix_states.len() - 1,
            );

            map.close().await.unwrap();
            Ok(())
        })?;
    }
}
