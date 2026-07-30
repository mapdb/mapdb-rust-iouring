//! Acceptance for fail-safe store creation.
//!
//! The default `create_path` must never destroy pre-existing bytes: every
//! partial file-set collision refuses with the precise structured error and
//! leaves the namespace byte-identical, asserted by hashing every file in the
//! directory before and after rather than by existence checks alone. A held lock
//! reports `AlreadyOpen`; an unlocked collision reports `AlreadyExists(member)`;
//! an unheld leftover sidecar alone is claimable; and a create that fails
//! partway rolls back only the members it made. Destructive replacement exists
//! solely in `create_path_overwrite`.

use crate::backend::{Direct, Wal};
use crate::error::OpenError;
use crate::map::BTreeMap;
use crate::store::{Options, Store};
use crate::WriteBatch;
use futures_lite::future::block_on;
use std::collections::BTreeMap as StdMap;
use std::fs;
use std::path::{Path, PathBuf};

fn tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mapdb-t15-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every file in `dir` with its full contents — the byte-identity oracle.
fn dir_bytes(dir: &Path) -> StdMap<String, Vec<u8>> {
    let mut out = StdMap::new();
    for e in fs::read_dir(dir).unwrap() {
        let e = e.unwrap();
        out.insert(
            e.file_name().to_string_lossy().into_owned(),
            fs::read(e.path()).unwrap(),
        );
    }
    out
}

/// Asserts a `create_path` collision: precise error AND byte-identical dir.
fn assert_refused<T>(
    dir: &Path,
    member: &'static str,
    create: impl FnOnce() -> Result<T, OpenError>,
) {
    let before = dir_bytes(dir);
    match create() {
        Err(OpenError::AlreadyExists(m)) => assert_eq!(m, member, "collision member"),
        Err(other) => panic!("expected AlreadyExists({member}), got {other:?}"),
        Ok(_) => panic!("expected AlreadyExists({member}), got a store"),
    }
    assert_eq!(
        before,
        dir_bytes(dir),
        "a refused create must not change one byte in the namespace"
    );
}

#[test]
fn wal_collisions_refuse_precisely_and_touch_nothing() {
    // (a) data file only, no sidecar — a squatter or an unlocked older store.
    let dir = tmp_dir("wal-data");
    let path = dir.join("store.data");
    fs::write(&path, b"precious pre-existing bytes").unwrap();
    assert_refused(&dir, "data file", || {
        Store::<Wal>::create_path(&path, Options::default())
    });
    // The refusal must also have removed the sidecar it minted while claiming.
    assert!(
        !dir.join("store.data.lock").exists(),
        "a refused create must roll back its own sidecar"
    );
    let _ = fs::remove_dir_all(&dir);

    // (b) unheld sidecar + data file: still AlreadyExists (the sidecar being
    // free does not make the data file fair game), sidecar preserved.
    let dir = tmp_dir("wal-lockdata");
    let path = dir.join("store.data");
    fs::write(&path, b"data").unwrap();
    fs::write(dir.join("store.data.lock"), b"").unwrap();
    assert_refused(&dir, "data file", || {
        Store::<Wal>::create_path(&path, Options::default())
    });
    assert!(dir.join("store.data.lock").exists());
    let _ = fs::remove_dir_all(&dir);

    // (c) a stray non-zero segment alone (data file gone — a half-deleted
    // store): the segment is somebody's bytes; refuse.
    let dir = tmp_dir("wal-seg");
    let path = dir.join("store.data");
    fs::write(dir.join("store.data.wal.3"), b"orphan segment").unwrap();
    assert_refused(&dir, "wal segment", || {
        Store::<Wal>::create_path(&path, Options::default())
    });
    let _ = fs::remove_dir_all(&dir);

    // (d) a compaction temp alone: refuse (the overwrite API cleans it).
    let dir = tmp_dir("wal-temp");
    let path = dir.join("store.data");
    fs::write(dir.join("store.data.compact.tmp"), b"temp").unwrap();
    assert_refused(&dir, "compaction temp", || {
        Store::<Wal>::create_path(&path, Options::default())
    });
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn direct_collisions_refuse_precisely_and_touch_nothing() {
    let dir = tmp_dir("direct-data");
    let path = dir.join("store.data");
    fs::write(&path, b"precious").unwrap();
    assert_refused(&dir, "data file", || {
        Store::<Direct>::create_path(&path, Options::default())
    });
    assert!(!dir.join("store.data.lock").exists());
    let _ = fs::remove_dir_all(&dir);

    let dir = tmp_dir("direct-temp");
    let path = dir.join("store.data");
    fs::write(dir.join("store.data.compact.tmp"), b"temp").unwrap();
    assert_refused(&dir, "compaction temp", || {
        Store::<Direct>::create_path(&path, Options::default())
    });
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn held_lock_is_already_open_not_already_exists() {
    let dir = tmp_dir("held");
    let path = dir.join("store.data");
    let store = Store::<Wal>::create_path(&path, Options::default()).unwrap();
    match Store::<Wal>::create_path(&path, Options::default()) {
        Err(OpenError::AlreadyOpen) => {}
        Err(other) => panic!("held lock must be AlreadyOpen, got {other:?}"),
        Ok(_) => panic!("held lock must be AlreadyOpen, got a store"),
    }
    block_on(store.close()).unwrap();
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn unheld_leftover_sidecar_alone_is_claimable() {
    // Only `<path>.lock` exists (a crashed pre-T1.5 create, or a store whose
    // members were deleted): claimable, not a collision.
    let dir = tmp_dir("leftover");
    let path = dir.join("store.data");
    fs::write(dir.join("store.data.lock"), b"").unwrap();
    let store = Store::<Direct>::create_path(&path, Options::default()).unwrap();
    block_on(store.close()).unwrap();
    let reopened = Store::<Direct>::open_path(&path, Options::default()).unwrap();
    block_on(reopened.close()).unwrap();
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn overwrite_api_replaces_a_whole_existing_store() {
    let dir = tmp_dir("overwrite");
    let path = dir.join("store.data");
    // A real store with data, plus a stray orphan segment and temp.
    {
        let store = Store::<Wal>::create_path(&path, Options::default()).unwrap();
        let map = BTreeMap::over(store);
        block_on(map.apply(WriteBatch::new().insert(b"old".to_vec(), b"gone".to_vec()))).unwrap();
        block_on(map.close()).unwrap();
    }
    fs::write(dir.join("store.data.wal.9"), b"orphan").unwrap();
    fs::write(dir.join("store.data.compact.tmp"), b"temp").unwrap();

    // The default create refuses this namespace...
    match Store::<Wal>::create_path(&path, Options::default()) {
        Err(OpenError::AlreadyExists(_)) => {}
        Err(other) => panic!("expected AlreadyExists, got {other:?}"),
        Ok(_) => panic!("expected AlreadyExists, got a store"),
    }
    // ...the explicitly destructive API replaces all of it.
    let store = Store::<Wal>::create_path_overwrite(&path, Options::default()).unwrap();
    let map = BTreeMap::over(store);
    assert_eq!(block_on(map.get(b"old".to_vec())).unwrap(), None);
    assert!(
        !dir.join("store.data.wal.9").exists(),
        "orphan segment removed"
    );
    assert!(
        !dir.join("store.data.compact.tmp").exists(),
        "stale temp removed"
    );
    block_on(map.close()).unwrap();
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn failed_create_rolls_back_only_its_own_members() {
    // Force the create tail to fail after the data inode is claimed: O_DIRECT
    // on a filesystem that refuses it (env temp dir is tmpfs on most dev
    // hosts). Probe-and-skip if this filesystem accepts O_DIRECT.
    let dir = tmp_dir("rollback");
    let path = dir.join("store.data");
    let pre_existing = dir.join("bystander.txt");
    fs::write(&pre_existing, b"innocent").unwrap();
    let options = Options {
        direct_io: true,
        ..Options::default()
    };
    match Store::<Direct>::create_path(&path, options) {
        Err(OpenError::Io(_)) | Err(OpenError::Corrupt(_)) => {
            // The failed attempt removed everything it created...
            assert!(!path.exists(), "rolled-back data inode");
            assert!(!dir.join("store.data.lock").exists(), "rolled-back sidecar");
            // ...and nothing else.
            assert_eq!(fs::read(&pre_existing).unwrap(), b"innocent");
        }
        Ok(store) => {
            eprintln!("skip: this filesystem honors O_DIRECT; rollback path not exercised");
            block_on(store.close()).unwrap();
        }
        Err(other) => panic!("unexpected refusal: {other:?}"),
    }
    let _ = fs::remove_dir_all(&dir);
}
