//! Recovery-image parity between the in-memory and real-file open paths.
//!
//! This does not crash a real filesystem — the crash is modelled by
//! `FakeIo::crash_image` before any real file exists, and an out-of-process
//! power-cut harness remains a separate external tier. What it proves is that
//! for identical bytes the production open path (`open_path` over `SyscallIo`,
//! sidecar lock, two header pages or segment resolution) reaches exactly the
//! same recovery verdict and state as the in-memory path, on both backends. That
//! is what keeps the fake-device crash matrices honest: they certify the parser
//! the production open actually runs.

use crate::backend::{Direct, Wal};
use crate::io::{FakeIo, PageIo, Persist};
use crate::map::BTreeMap;
use crate::store::{Options, Store};
use crate::{ApplyOutcome, WriteBatch};
use futures_lite::future::block_on;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Decision = (&'static str, fn(usize) -> Persist);

fn tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mapdb-parity-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Opens `img` via the in-memory path and returns `get(key)`'s verdict, encoded
/// as `Ok(value-option)` / `Err(kind-string)` for comparison.
fn fake_open_get(img: Vec<u8>, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
    let dev = Arc::new(FakeIo::from_vec(img));
    let store = Store::<Direct>::open_over(Arc::clone(&dev) as Arc<dyn PageIo>, Options::default())
        .map_err(|e| format!("{e:?}").split('(').next().unwrap().to_string())?;
    let map = BTreeMap::over(store);
    let got = block_on(map.get(key.to_vec())).unwrap();
    block_on(map.close()).unwrap();
    Ok(got)
}

/// Writes `img` to a real file and opens it via the PRODUCTION `open_path`
/// (SyscallIo, sidecar lock, real directory), returning the same encoding.
fn real_open_get(dir: &Path, img: &[u8], key: &[u8]) -> Result<Option<Vec<u8>>, String> {
    let path = dir.join("store.data");
    std::fs::write(&path, img).unwrap();
    let store = Store::<Direct>::open_path(&path, Options::default())
        .map_err(|e| format!("{e:?}").split('(').next().unwrap().to_string())?;
    let map = BTreeMap::over(store);
    let got = block_on(map.get(key.to_vec())).unwrap();
    block_on(map.close()).unwrap();
    std::fs::remove_file(&path).unwrap();
    let _ = std::fs::remove_file(dir.join("store.data.lock"));
    Ok(got)
}

/// Direct: every crash image the T1.1 matrix shape produces must open (or
/// refuse) IDENTICALLY through the real open path.
#[test]
fn direct_crash_images_agree_between_fake_and_real_open() {
    let dir = tmp_dir("direct");
    // Build P (committed) then N (applied, unsynced) on FakeIo.
    let dev = Arc::new(FakeIo::new(0));
    let store =
        Store::<Direct>::create_with_io(Arc::clone(&dev) as Arc<dyn PageIo>, Options::default())
            .unwrap();
    let map = BTreeMap::over(store);
    let out = block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"P".to_vec()))).unwrap();
    let ApplyOutcome::Applied { version, .. } = out else {
        panic!()
    };
    block_on(map.flush(version)).unwrap();
    block_on(map.apply(WriteBatch::new().insert(b"k".to_vec(), b"N".to_vec()))).unwrap();
    // No flush: N's pages + header are unsynced when the store closes... a clean
    // close WOULD flush, so take the images now, before closing.
    let decisions: &[Decision] = &[
        ("old", |_| Persist::Old),
        ("new", |_| Persist::New),
        ("garble", |_| Persist::Garble),
        ("mixed", |g| match g % 3 {
            0 => Persist::Old,
            1 => Persist::New,
            _ => Persist::Garble,
        }),
    ];
    let mut images: Vec<(String, Vec<u8>)> = decisions
        .iter()
        .map(|(name, d)| (name.to_string(), dev.crash_image(128, d)))
        .collect();
    block_on(map.close()).unwrap();
    // Post-close image too (the fully-committed shape).
    images.push(("closed".into(), dev.snapshot_bytes()));
    // And adversarial statics: torn slot A, both slots torn.
    let base = dev.snapshot_bytes();
    let mut torn_a = base.clone();
    for b in &mut torn_a[64..96] {
        *b ^= 0xFF;
    }
    images.push(("torn-slot-a".into(), torn_a));
    let mut torn_both = base.clone();
    for off in [64usize, crate::page::PAGE_SIZE + 64] {
        for b in &mut torn_both[off..off + 32] {
            *b ^= 0xFF;
        }
    }
    images.push(("torn-both".into(), torn_both));

    for (name, img) in images {
        let fake = fake_open_get(img.clone(), b"k");
        let real = real_open_get(&dir, &img, b"k");
        assert_eq!(
            fake, real,
            "recovery parity broke on image '{name}': fake={fake:?} real={real:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Wal: a torn-tail crash image (data + segment files) recovers to the same
/// committed prefix through `open_path` as through the in-memory recovery.
#[test]
fn wal_crash_images_agree_between_fake_and_real_open() {
    // Build over plain in-memory devices (no namespace needed — this scenario
    // never checkpoints), then crash-image the WAL device.
    let data = Arc::new(FakeIo::new(0));
    let wal = Arc::new(FakeIo::new(0));
    let store = Store::<Wal>::create_with_io(
        Arc::clone(&data) as Arc<dyn PageIo>,
        Arc::clone(&wal) as Arc<dyn PageIo>,
        Options::default(),
    )
    .unwrap();
    let map = BTreeMap::over(store);
    // Two committed batches, one uncommitted (unsynced) tail batch.
    for (k, v) in [(&b"a"[..], &b"1"[..]), (&b"b"[..], &b"2"[..])] {
        let out = block_on(map.apply(WriteBatch::new().insert(k.to_vec(), v.to_vec()))).unwrap();
        let ApplyOutcome::Applied { version, .. } = out else {
            panic!()
        };
        block_on(map.commit(version)).unwrap();
    }
    block_on(map.apply(WriteBatch::new().insert(b"c".to_vec(), b"3".to_vec()))).unwrap();

    let decisions: &[Decision] = &[
        ("old", |_| Persist::Old),
        ("new", |_| Persist::New),
        ("garble", |_| Persist::Garble),
        ("alternate", |g| {
            if g % 2 == 0 {
                Persist::New
            } else {
                Persist::Garble
            }
        }),
    ];
    let data_img = data.synced_bytes();
    let wal_imgs: Vec<(String, Vec<u8>)> = decisions
        .iter()
        .map(|(name, d)| (name.to_string(), wal.crash_image(64, d)))
        .collect();
    drop(map); // T1.2: drop-drain; devices stay alive via our Arcs

    let dir = tmp_dir("wal");
    for (name, wal_img) in wal_imgs {
        // In-memory recovery verdict.
        let fake = {
            let d = Arc::new(FakeIo::from_vec(data_img.clone()));
            let w = Arc::new(FakeIo::from_vec(wal_img.clone()));
            let r = Store::<Wal>::open_over(
                Arc::clone(&d) as Arc<dyn PageIo>,
                Arc::clone(&w) as Arc<dyn PageIo>,
                Options::default(),
            )
            .map(|s| {
                let m = BTreeMap::over(s);
                let vals = block_on(async {
                    (
                        m.get(b"a".to_vec()).await.unwrap(),
                        m.get(b"b".to_vec()).await.unwrap(),
                        m.get(b"c".to_vec()).await.unwrap(),
                    )
                });
                block_on(m.close()).unwrap();
                vals
            })
            .map_err(|e| format!("{e:?}").split('(').next().unwrap().to_string());
            fake_normalize(r)
        };
        // Real-file recovery verdict through open_path.
        let real = {
            let path = dir.join("store.data");
            std::fs::write(&path, &data_img).unwrap();
            std::fs::write(dir.join("store.data.wal.0"), &wal_img).unwrap();
            let r = Store::<Wal>::open_path(&path, Options::default())
                .map(|s| {
                    let m = BTreeMap::over(s);
                    let vals = block_on(async {
                        (
                            m.get(b"a".to_vec()).await.unwrap(),
                            m.get(b"b".to_vec()).await.unwrap(),
                            m.get(b"c".to_vec()).await.unwrap(),
                        )
                    });
                    block_on(m.close()).unwrap();
                    vals
                })
                .map_err(|e| format!("{e:?}").split('(').next().unwrap().to_string());
            for f in std::fs::read_dir(&dir).unwrap() {
                let _ = std::fs::remove_file(f.unwrap().path());
            }
            fake_normalize(r)
        };
        assert_eq!(fake, real, "WAL recovery parity broke on image '{name}'");
        // Sanity on the oracle itself: committed a/b always survive when the
        // store opens at all.
        if let Ok((a, b, _)) = &fake {
            assert_eq!(a.as_deref(), Some(&b"1"[..]), "committed key lost ({name})");
            assert_eq!(b.as_deref(), Some(&b"2"[..]), "committed key lost ({name})");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

type Vals = (Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>);
fn fake_normalize(r: Result<Vals, String>) -> Result<Vals, String> {
    r
}
