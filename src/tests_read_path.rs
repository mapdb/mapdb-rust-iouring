//! Acceptance for the read path over real encoded pages.
//!
//! The mechanism-level checks live next to their code: single-flight of N
//! waiters on one cold page in [`crate::cache`], decode rejection of bad ids,
//! checksums, lengths, key order and child refs in [`crate::page`], and
//! snapshot/`get_many` agreement in [`crate::read`]. This module adds the
//! end-to-end layer: a live `BTreeMap`/`Snapshot` dispatching point gets, range
//! reads and `get_many` onto a real multi-level tree, writes against a read-only
//! tree failing fast without damaging it, and a randomized differential of the
//! whole read surface against `std::collections::BTreeMap`.

use crate::io::FakeIo;
use crate::read::TreeBuilder;
use crate::store::Options;
use crate::{BTreeMap, Store};
use futures_lite::future::block_on;
use std::collections::BTreeMap as StdMap;
use std::sync::Arc;

fn build_store(data: &[(Vec<u8>, Vec<u8>)], builder: TreeBuilder) -> BTreeMap<crate::Direct> {
    let dev = Arc::new(FakeIo::new(0));
    let built = builder.build(dev, data).unwrap();
    let store = Store::<crate::Direct>::open_readonly_tree(built, Options::default());
    BTreeMap::over(store)
}

fn kvs(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..n)
        .map(|i| {
            (
                format!("k{i:06}").into_bytes(),
                format!("v{i}").into_bytes(),
            )
        })
        .collect()
}

#[test]
fn btreemap_dispatches_reads_onto_real_pages() {
    let data = kvs(400);
    let map = build_store(
        &data,
        TreeBuilder {
            max_fanout: 5,
            max_leaf_entries: 6,
            ..Default::default()
        },
    );
    block_on(async {
        // Point gets traverse the real tree (root is non-empty).
        for (k, v) in &data {
            assert_eq!(
                map.get(k.clone()).await.unwrap().as_deref(),
                Some(v.as_slice())
            );
        }
        assert_eq!(map.get(b"missing".to_vec()).await.unwrap(), None);

        // Multi-get through the grouped path.
        let query: Vec<Vec<u8>> = vec![
            b"k000005".to_vec(),
            b"absent".to_vec(),
            b"k000399".to_vec(),
            b"k000200".to_vec(),
        ];
        let got = map.get_many(query.clone()).await.unwrap();
        assert_eq!(
            got,
            vec![
                Some(b"v5".to_vec()),
                None,
                Some(b"v399".to_vec()),
                Some(b"v200".to_vec()),
            ]
        );

        // verify() walks the whole tree structurally.
        assert_eq!(map.verify().await.unwrap().entry_count, 400);

        // Snapshot reads the same real pages.
        let snap = map.snapshot();
        assert_eq!(
            snap.get(b"k000123".to_vec()).await.unwrap().as_deref(),
            Some(&b"v123"[..])
        );
        assert_eq!(snap.len(), 400);
    });
    assert_eq!(map.len(), 400);
    // Metrics reflect real page loads through the cache.
    assert!(map.metrics().entry_count == 400);
}

#[test]
fn empty_real_store_reads_none() {
    // An empty builder yields an empty root → live map falls back cleanly.
    let map = build_store(&[], TreeBuilder::default());
    block_on(async {
        assert_eq!(map.get(b"x".to_vec()).await.unwrap(), None);
        assert_eq!(map.verify().await.unwrap().entry_count, 0);
    });
    assert!(map.is_empty());
}

#[test]
fn writes_to_readonly_tree_fail_fast_without_wiping_it() {
    use crate::error::WriteError;
    use crate::WriteBatch;

    let data = kvs(50);
    let map = build_store(
        &data,
        TreeBuilder {
            max_fanout: 4,
            max_leaf_entries: 4,
            ..Default::default()
        },
    );
    block_on(async {
        // A write must fail fast (Compacting) — never silently publish an empty
        // root over the read-only tree.
        let out = map
            .apply(WriteBatch::new().insert(b"k000000".to_vec(), b"x".to_vec()))
            .await;
        assert!(matches!(out, Err(WriteError::Compacting)), "got {out:?}");

        // The tree is intact: still 50 entries, still readable.
        assert_eq!(map.len(), 50);
        assert_eq!(
            map.get(b"k000000".to_vec()).await.unwrap().as_deref(),
            Some(&b"v0"[..])
        );
        assert_eq!(map.verify().await.unwrap().entry_count, 50);
    });
    // Clean close of a writer-less store.
    block_on(map.close()).unwrap();
}

// Randomized differential of the full read surface vs std::BTreeMap.
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 60, ..ProptestConfig::default() })]

    #[test]
    fn read_surface_matches_std_btreemap(
        // Distinct small keys (proptest dedups by collecting into a map) and a
        // set of probe keys (some present, some absent).
        pairs in prop::collection::vec((any::<u16>(), any::<u32>()), 0..200),
        probes in prop::collection::vec(any::<u16>(), 0..80),
    ) {
        // Deduplicate/sort into the ground truth.
        let truth: StdMap<Vec<u8>, Vec<u8>> = pairs
            .into_iter()
            .map(|(k, v)| (k.to_be_bytes().to_vec(), v.to_be_bytes().to_vec()))
            .collect();
        let sorted: Vec<(Vec<u8>, Vec<u8>)> = truth.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Force a multi-level tree even for small inputs.
        let map = build_store(&sorted, TreeBuilder { max_fanout: 3, max_leaf_entries: 3, ..Default::default() });

        block_on(async {
            // len / verify.
            prop_assert_eq!(map.len(), truth.len() as u64);
            prop_assert_eq!(map.verify().await.unwrap().entry_count, truth.len() as u64);

            // Point gets for every truth key and every probe.
            for (k, v) in &truth {
                let got = map.get(k.clone()).await.unwrap();
                prop_assert_eq!(got.as_deref(), Some(v.as_slice()));
            }
            let query: Vec<Vec<u8>> = probes.iter().map(|p| p.to_be_bytes().to_vec()).collect();
            let expected: Vec<Option<Vec<u8>>> =
                query.iter().map(|k| truth.get(k).cloned()).collect();

            // Grouped multi-get matches truth...
            let grouped = map.get_many(query.clone()).await.unwrap();
            prop_assert_eq!(&grouped, &expected);
            // ...and equals per-key point gets (naïve), in order.
            let mut naive = Vec::new();
            for k in &query {
                naive.push(map.get(k.clone()).await.unwrap());
            }
            prop_assert_eq!(&naive, &expected);
            Ok(())
        })?;
    }
}

#[test]
fn map_verify_rejects_descriptor_count_mismatch() {
    for count in [0, 1, 999, u64::MAX] {
        let mut built = TreeBuilder::default()
            .build(Arc::new(FakeIo::new(0)), &kvs(5))
            .unwrap();
        built.entry_count = count;
        let store = Store::<crate::Direct>::open_readonly_tree(built, Options::default());
        let map = BTreeMap::over(store);
        assert!(matches!(
            block_on(map.verify()),
            Err(crate::error::ReadError::Corrupt(_))
        ));
    }
}
