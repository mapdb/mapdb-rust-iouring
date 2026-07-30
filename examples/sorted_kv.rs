//! A tour of [`SortedMap`] — the high-level sorted-map layer — on a real
//! path-backed `StoreWal`: durable writes, ordered scans in both directions,
//! prefix scans, NavigableMap lookups, pagination, ranged deletes, and crash
//! recovery via reopen.
//!
//! Run: `cargo run --example sorted_kv`
//!
//! Everything async in the engine is runtime-neutral; this example drives it
//! with `futures_lite::future::block_on` and no runtime at all.

use mapdb_uring::sorted::SortedMap;
use mapdb_uring::store::Options;
use mapdb_uring::{Store, StoreWal, Wal};
use std::ops::{Bound, ControlFlow};

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tour");
    futures_lite::future::block_on(run(&path));
}

async fn run(path: &std::path::Path) {
    // -- Create a WAL-backed store and wrap it in the sorted-map layer. -----
    let store: StoreWal = Store::<Wal>::create_path(path, Options::default()).unwrap();
    let map = SortedMap::over(store);

    // -- Writes. `insert`/`put_all` return a Version; `commit` makes it
    //    durable (group commit — concurrent commits coalesce). ---------------
    for user in ["ada", "bob", "eve"] {
        for n in 1..=3u32 {
            map.insert(
                format!("user:{user}:msg:{n:04}").into_bytes(),
                format!("hello #{n} from {user}").into_bytes(),
            )
            .await
            .unwrap();
        }
    }
    let v = map
        .put_all((0..100u32).map(|i| {
            (
                format!("metric:{i:06}").into_bytes(),
                i.to_le_bytes().to_vec(),
            )
        }))
        .await
        .unwrap();
    map.commit(v).await.unwrap(); // durability watermark: everything above

    // -- Point lookups and NavigableMap edges. ------------------------------
    let got = map.get(b"user:ada:msg:0002".to_vec()).await.unwrap();
    println!("point get      : {:?}", got.map(printable));
    let (k, _) = map.first_entry().await.unwrap().unwrap();
    println!("first_entry    : {}", printable(k));
    let (k, _) = map.floor_entry(b"user:bob:zzz").await.unwrap().unwrap();
    println!("floor(bob:zzz) : {}", printable(k)); // greatest key <= probe

    // -- Ordered scans: consumer callback, `ControlFlow` to stop early. -----
    let mut n = 0u32;
    map.for_each_range(b"metric:".to_vec()..b"metric:000010".to_vec(), |_k, v| {
        n += u32::from_le_bytes(v.try_into().unwrap());
        ControlFlow::Continue(())
    })
    .await
    .unwrap();
    println!("sum metric<10  : {n}");

    // Latest message of one user: descending prefix scan, stop at the first.
    let mut latest = None;
    map.for_each_prefix_rev(b"user:bob:msg:", |k, v| {
        latest = Some((printable(k.to_vec()), printable(v.to_vec())));
        ControlFlow::Break(())
    })
    .await
    .unwrap();
    println!("latest from bob: {latest:?}");

    // -- Newest-first pagination: bounded pages, resume below the last key.
    //    `metric:` .. `metric;` is the half-open prefix window (`;` = `:`+1).
    let metrics_range = |hi: Bound<Vec<u8>>| {
        let hi = match hi {
            Bound::Unbounded => Bound::Excluded(b"metric;".to_vec()),
            other => other,
        };
        (Bound::Included(b"metric:".to_vec()), hi)
    };
    let mut resume: Option<Vec<u8>> = None;
    let mut page_no = 0;
    loop {
        let hi = match &resume {
            None => Bound::Unbounded,
            Some(last) => Bound::Excluded(last.clone()),
        };
        let page = map.collect_range_rev(metrics_range(hi), 40).await.unwrap();
        if page.is_empty() {
            break;
        }
        page_no += 1;
        println!(
            "page {page_no}         : {} entries, {} .. {}",
            page.len(),
            printable(page[0].0.clone()),
            printable(page.last().unwrap().0.clone()),
        );
        resume = Some(page.last().unwrap().0.clone());
    }

    // -- Ranged delete, then close cleanly (drains + syncs). ----------------
    let removed = map
        .remove_range(b"metric:".to_vec()..b"metric:000050".to_vec())
        .await
        .unwrap();
    println!("removed        : {removed} metrics");
    map.close().await.unwrap();

    // -- Reopen: WAL recovery yields exactly the committed state. -----------
    let store = Store::<Wal>::open_path(path, Options::default()).unwrap();
    let map = SortedMap::over(store);
    let metrics = map
        .count_range(b"metric:".to_vec()..b"metric;".to_vec())
        .await
        .unwrap();
    let (k, _) = map.last_entry().await.unwrap().unwrap();
    println!(
        "after reopen   : {metrics} metrics survive, last key {}",
        printable(k)
    );
    map.close().await.unwrap();
}

fn printable(b: Vec<u8>) -> String {
    String::from_utf8_lossy(&b).into_owned()
}
