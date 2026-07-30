//! A directory-aware crash-consistency model for the durable namespace
//! protocols: WAL segment create and rotate, and compaction promote.
//!
//! The real protocols bury their parent-directory fsync inside the host methods
//! (`FsWalFileSet::create_segment`/`unlink_segment`, `FsDataHost::promote_temp`),
//! and an in-process unit test cannot power-cut the OS mid-syscall, so a crash
//! *between* a rename or unlink and its directory fsync is unreachable against a
//! real filesystem. This model makes that point reachable by modelling the one
//! fact the protocols defend against: fsyncing a file's data does not make its
//! directory entry durable, so a freshly created, renamed or unlinked name can
//! vanish or reappear across a crash until the directory itself is fsynced.
//!
//! It proves the protocols' ordering invariants — that each directory fsync is
//! load-bearing, that a failed one is not a durable commit, and that recovery is
//! crash-consistent and idempotent at every point.
//!
//! It deliberately does not model ext4/XFS conformance: real guarantees vary by
//! filesystem, mount mode, kernel, and same- versus cross-directory rename, and
//! validating those needs an out-of-process fault tier outside the scope of
//! these unit tests. The model mirrors the real steps by name; a protocol change
//! must update it in lockstep.

use std::collections::BTreeMap;

/// A minimal crash model of a **single directory** on a Linux-like filesystem:
/// two namespace views (volatile vs last-directory-fsynced) plus per-file content
/// durability (volatile vs last-file-fsynced). A `crash()` reverts every volatile
/// change back to its last durability point.
#[derive(Clone, Default)]
struct CrashFs {
    /// Live directory entries: name → content id.
    names: BTreeMap<String, u64>,
    /// Directory entries that survive a crash (as of the last [`Self::fsync_dir`]).
    durable_names: BTreeMap<String, u64>,
    /// Live file content by id.
    content: BTreeMap<u64, Vec<u8>>,
    /// File content that survives a crash (as of that file's last
    /// [`Self::fsync_file`]).
    durable_content: BTreeMap<u64, Vec<u8>>,
    next_id: u64,
}

impl CrashFs {
    /// Creates `name` as a fresh empty file (volatile). Panics if it exists — the
    /// real protocol uses `create_new` for exactly this reason (a live segment is
    /// never recreated).
    fn create(&mut self, name: &str) {
        assert!(
            !self.names.contains_key(name),
            "create_new would reject {name}"
        );
        let id = self.next_id;
        self.next_id += 1;
        self.names.insert(name.to_string(), id);
        self.content.insert(id, Vec::new());
    }

    /// Overwrites `name`'s content (volatile until [`Self::fsync_file`]).
    fn write(&mut self, name: &str, bytes: &[u8]) {
        let id = self.names[name];
        self.content.insert(id, bytes.to_vec());
    }

    /// Makes `name`'s *content* durable (an `fdatasync` of the file). Its
    /// directory *entry* is still volatile until [`Self::fsync_dir`].
    fn fsync_file(&mut self, name: &str) {
        let id = self.names[name];
        self.durable_content.insert(id, self.content[&id].clone());
    }

    /// Same-directory atomic rename (volatile). Models `rename(from, to)` /
    /// `promote_temp`.
    fn rename(&mut self, from: &str, to: &str) {
        let id = self.names.remove(from).expect("rename source exists");
        self.names.insert(to.to_string(), id);
    }

    /// Removes `name` (volatile). Idempotent, like the real `unlink_segment`.
    fn unlink(&mut self, name: &str) {
        self.names.remove(name);
    }

    /// Makes the whole directory's entry list durable (an `fsync` of the parent
    /// directory). Content durability is unaffected — that is per-file.
    fn fsync_dir(&mut self) {
        self.durable_names = self.names.clone();
    }

    /// Power loss: every volatile change since its last durability point is lost.
    /// The directory reverts to the last `fsync_dir`, and each file's content to
    /// its last `fsync_file`.
    fn crash(&mut self) {
        self.names = self.durable_names.clone();
        self.content = self.durable_content.clone();
    }

    /// The content a *durable* reader would see through `name` after a crash:
    /// `None` if the name is not durable, or names content that was never made
    /// durable (the "durable entry → unsynced content" bug).
    fn durable_view(&self, name: &str) -> Option<&[u8]> {
        let id = self.durable_names.get(name)?;
        self.durable_content.get(id).map(|v| v.as_slice())
    }

    fn durable_segments(&self) -> Vec<String> {
        self.durable_names
            .keys()
            .filter(|n| n.starts_with("wal."))
            .cloned()
            .collect()
    }
}

/// The correct WAL rotation protocol (mirrors `wal.rs` checkpoint + `FsWalFileSet`),
/// run against the model and crashed after step `crash_after` (`usize::MAX` = run
/// to completion). Returns the post-crash filesystem; the caller runs recovery.
///
/// Steps: (0) create the fresh segment, write+fsync its header, fsync the dir so
/// the new entry is durable; (1) write the data header naming the new segment and
/// fsync it — the commit point; (2) unlink the retired segment and fsync the dir.
fn rotate(fs: &mut CrashFs, prev: u64, next: u64, crash_after: usize) {
    let seg = |n: u64| format!("wal.{n}");
    // Step 0 — create + durably publish the new segment.
    fs.create(&seg(next));
    fs.write(&seg(next), b"header");
    fs.fsync_file(&seg(next));
    fs.fsync_dir();
    if crash_after == 0 {
        fs.crash();
        return;
    }
    // Step 1 — commit: the data header now names `next`, made durable.
    fs.write("data", &next.to_le_bytes());
    fs.fsync_file("data");
    if crash_after == 1 {
        fs.crash();
        return;
    }
    // Step 2 — reclaim: unlink the retired segment, make the removal durable.
    fs.unlink(&seg(prev));
    fs.fsync_dir();
    if crash_after == 2 {
        fs.crash();
    }
}

/// Recovery at open (mirrors `wal_fs::open_path` → `retire_other_segments`): read
/// the durably-committed segment seq from the data header, then remove every WAL
/// segment except that one and fsync the directory. Returns the seq it recovered.
fn recover(fs: &mut CrashFs) -> u64 {
    let bytes = fs
        .durable_view("data")
        .expect("a committed store has a durable data header");
    let keep = u64::from_le_bytes(bytes.try_into().expect("8-byte seq"));
    let keep_name = format!("wal.{keep}");
    let doomed: Vec<String> = fs
        .names
        .keys()
        .filter(|n| n.starts_with("wal.") && **n != keep_name)
        .cloned()
        .collect();
    for n in doomed {
        fs.unlink(&n);
    }
    fs.fsync_dir();
    keep
}

/// Seeds a committed store: data header names segment `prev`, which exists — all
/// durable.
fn seeded(prev: u64) -> CrashFs {
    let mut fs = CrashFs::default();
    fs.create(&format!("wal.{prev}"));
    fs.write(&format!("wal.{prev}"), b"header");
    fs.fsync_file(&format!("wal.{prev}"));
    fs.create("data");
    fs.write("data", &prev.to_le_bytes());
    fs.fsync_file("data");
    fs.fsync_dir();
    fs
}

/// The core invariant: at **every** crash point of a rotation, recovery leaves a
/// consistent store — the segment named by the durable data header exists, and no
/// other WAL segment lingers. This covers the crash-between-rename/unlink-and-
/// dir-fsync points a real filesystem cannot reach in-process.
#[test]
fn wal_rotation_recovers_consistently_at_every_crash_point() {
    for crash_after in [0usize, 1, 2, usize::MAX] {
        let mut fs = seeded(0);
        rotate(&mut fs, 0, 1, crash_after);
        let keep = recover(&mut fs);

        // The committed pointer is honoured: seg 0 before the commit lands, seg 1
        // once it does.
        let expected_keep = if crash_after == 0 { 0 } else { 1 };
        assert_eq!(keep, expected_keep, "crash_after={crash_after}");

        let keep_name = format!("wal.{keep}");
        assert!(
            fs.durable_names.contains_key(&keep_name),
            "crash_after={crash_after}: the segment the header names must exist durably"
        );
        assert_eq!(
            fs.durable_segments(),
            vec![keep_name],
            "crash_after={crash_after}: exactly one segment survives — no orphan lingers"
        );
        // Recovery is idempotent: a second open changes nothing.
        let mut again = fs.clone();
        let keep2 = recover(&mut again);
        assert_eq!(keep2, keep);
        assert_eq!(again.durable_segments(), fs.durable_segments());
    }
}

/// Teeth #1 — dropping the directory fsync after the retiring unlink lets the
/// removed segment **reappear** after a crash (its directory entry was never made
/// durable). This is the bug `unlink_segment`'s `fsync_parent_dir` prevents.
#[test]
fn omitting_the_unlink_dir_fsync_resurrects_the_retired_segment() {
    // Correct: unlink then fsync_dir → the removal is durable.
    let mut ok = seeded(0);
    rotate(&mut ok, 0, 1, usize::MAX);
    ok.crash();
    assert!(
        !ok.durable_names.contains_key("wal.0"),
        "correct protocol: seg 0 stays gone"
    );

    // Broken: reproduce steps 0–1 correctly, then a step 2 that unlinks but skips
    // the directory fsync, and crash.
    let mut bad = seeded(0);
    bad.create("wal.1");
    bad.write("wal.1", b"header");
    bad.fsync_file("wal.1");
    bad.fsync_dir();
    bad.write("data", &1u64.to_le_bytes());
    bad.fsync_file("data");
    bad.unlink("wal.0"); // removed in the live namespace...
                         // ...but NO fsync_dir here.
    bad.crash();
    assert!(
        bad.durable_names.contains_key("wal.0"),
        "without the dir fsync the retired segment resurrects after a crash"
    );
    // Recovery still cleans it up (idempotent) — the resurrection is a space leak
    // the next open reclaims, not corruption — but the point stands: the durable
    // outcome differed, so the fsync is load-bearing.
    let keep = recover(&mut bad);
    assert_eq!(keep, 1);
    assert_eq!(bad.durable_segments(), vec!["wal.1".to_string()]);
}

/// Teeth #2 — dropping the directory fsync after the promoting rename lets the
/// rename **vanish** after a crash: the active name reverts to the pre-rename
/// entry. This is the bug `promote_temp`'s `fsync_parent_dir` prevents. Modelled
/// as a compaction promote of a dense temp over the data file.
#[test]
fn omitting_the_promote_dir_fsync_loses_the_rename() {
    let build = |with_dir_fsync: bool| -> CrashFs {
        let mut fs = CrashFs::default();
        // A committed data file naming the old content.
        fs.create("data");
        fs.write("data", b"OLD");
        fs.fsync_file("data");
        fs.fsync_dir();
        // Compaction: build a dense temp, fsync its content, then rename over.
        fs.create("data.compact.tmp");
        fs.write("data.compact.tmp", b"DENSE");
        fs.fsync_file("data.compact.tmp");
        fs.fsync_dir(); // the temp's entry is durable
        fs.rename("data.compact.tmp", "data");
        if with_dir_fsync {
            fs.fsync_dir();
        }
        fs.crash();
        fs
    };

    // Correct: the promote is durable.
    assert_eq!(
        build(true).durable_view("data"),
        Some(&b"DENSE"[..]),
        "with the dir fsync the promote survives the crash"
    );
    // Broken: the rename is lost; `data` reverts to the pre-promote entry, whose
    // content ("OLD") is intact — a discarded compaction, not corruption, but a
    // different durable outcome, so the fsync is load-bearing.
    assert_eq!(
        build(false).durable_view("data"),
        Some(&b"OLD"[..]),
        "without the dir fsync the rename vanishes and the old file returns"
    );
}

/// Teeth #3 — a durable directory entry that names content which was never made
/// durable: the bug the *file* fsync-before-rename prevents. If a promote renames
/// a temp whose content was not fsynced first, a crash can leave the active name
/// pointing at content the crash discarded.
#[test]
fn a_durable_entry_naming_unsynced_content_is_the_file_fsync_bug() {
    let mut fs = CrashFs::default();
    fs.create("data");
    fs.write("data", b"OLD");
    fs.fsync_file("data");
    fs.fsync_dir();

    // Broken order: rename BEFORE the temp's content is fsynced.
    fs.create("data.compact.tmp");
    fs.write("data.compact.tmp", b"DENSE"); // content NOT fsynced
    fs.fsync_dir(); // temp entry durable
    fs.rename("data.compact.tmp", "data");
    fs.fsync_dir(); // the rename IS durable...
    fs.crash();

    // ...but the durable entry now names content that the crash lost.
    assert_eq!(
        fs.durable_view("data"),
        None,
        "the durable name resolves to content that was never fsynced — data loss"
    );
    // The correct protocol (fsync_file the temp before renaming) is what the real
    // create_temp/promote_temp does; this test pins *why* that order is mandatory.
}

/// The must-hold rule: a **failed** directory fsync must be surfaced as a
/// durability failure, never reported as a completed commit/rotation. The real
/// code propagates it — `create_segment`/`unlink_segment`/`retire_other_segments`
/// return `fsync_parent_dir(..)?`, `promote_temp` returns it (→ poison), and
/// `Store::<Wal>::create_path` stops the writer and returns `Err` on a failed
/// directory fsync. Modelled: a rotation whose final dir fsync fails does not make
/// the reclaim durable, so a caller that (wrongly) reported success would be
/// contradicted by the post-crash state.
#[test]
fn a_failed_promote_dir_fsync_is_not_a_durable_commit() {
    let mut fs = seeded(0);
    // Steps 0–1 succeed (segment published, header committed).
    fs.create("wal.1");
    fs.write("wal.1", b"header");
    fs.fsync_file("wal.1");
    fs.fsync_dir();
    fs.write("data", &1u64.to_le_bytes());
    fs.fsync_file("data");
    // Step 2's unlink lands in the live namespace, but its dir fsync *fails* — the
    // real code returns that Err rather than acknowledging the rotation.
    fs.unlink("wal.0");
    let dir_fsync_ok = false; // injected failure
    if dir_fsync_ok {
        fs.fsync_dir();
    }
    fs.crash();

    // Because the failure was surfaced (no success reported), the reclaim simply
    // did not become durable — the store is still consistent (header names 1,
    // which exists) and the next open reclaims seg 0. What must NOT happen is a
    // reported-durable rotation whose effect the crash then erased.
    assert!(
        fs.durable_names.contains_key("wal.0"),
        "a failed dir fsync leaves the removal non-durable — it must not be acked as committed"
    );
    let keep = recover(&mut fs);
    assert_eq!(keep, 1, "the committed pointer is intact and recoverable");
    assert_eq!(fs.durable_segments(), vec!["wal.1".to_string()]);
}
