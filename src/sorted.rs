//! `SortedMap<B>` — the high-level sorted-map abstraction over [`BTreeMap`].
//!
//! [`BTreeMap`] is the engine surface: batches, permits, versions, snapshots.
//! `SortedMap` wraps it with the API most callers actually want — single-op
//! `insert`/`remove`, bulk `put_all`, compare-and-set, the NavigableMap
//! lookups (`first`/`last`/`floor`/`ceiling`/`lower`/`higher` `_entry`),
//! `remove_range`, and **consumer-callback scans**: ordered range, prefix,
//! and whole-map traversal without an iterator type. Every scan captures a
//! snapshot internally, so it sees one consistent version regardless of
//! concurrent writers, and delivers borrowed `(key, value)` slices to a
//! `FnMut` consumer that returns [`ControlFlow`](std::ops::ControlFlow) to
//! continue or stop.
//!
//! Vocabulary is aligned with the `mapdb-collections` crate
//! (<https://github.com/mapdb/mapdb-rust>):
//! `insert`/`remove`/`len`/`is_empty`/`contains_key`,
//! bulk `put_all`, and the Java-NavigableMap `floor`/`ceiling`/`lower`/
//! `higher` family. Deliberate divergences (storage-layer semantics with no
//! in-memory counterpart): mutations return the published [`Version`] rather
//! than the displaced value; `put_if_absent`/`compare_and_set` exist (the
//! collections crate has no CAS); scans are callbacks, not iterators; and
//! everything is async.
//!
//! Durability is still explicit and backend-shaped: a mutation returns its
//! [`Version`]; `SortedMap<Direct>::flush` / `SortedMap<Wal>::commit` are the
//! barriers, and the `*_durable` conveniences fuse the two steps.

use crate::backend::{Backend, Direct, Wal};
use crate::batch::{ApplyOutcome, WriteBatch};
use crate::error::{CloseError, CommitError, FlushError, ReadError, WriteError};
use crate::map::BTreeMap;
use crate::read::{Entry, ScanSummary};
use crate::store::{Snapshot, Store};
use crate::version::Version;
use std::future::Future;
use std::ops::{Bound, ControlFlow, RangeBounds};

/// A sorted byte-key/byte-value map with callback scans. Cheap to clone; all
/// clones share one store.
pub struct SortedMap<B: Backend> {
    map: BTreeMap<B>,
}

impl<B: Backend> Clone for SortedMap<B> {
    fn clone(&self) -> Self {
        SortedMap {
            map: self.map.clone(),
        }
    }
}

/// The end bound that makes `[prefix, end)` cover exactly the keys starting
/// with `prefix`: the prefix with its last non-`0xFF` byte incremented (and
/// the `0xFF` tail dropped). `None` when every byte is `0xFF` — the prefix
/// range is then unbounded above.
fn prefix_end(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.last_mut() {
        if *last == 0xFF {
            end.pop();
        } else {
            *last += 1;
            return Some(end);
        }
    }
    None
}

/// The error of a scan-then-write operation ([`SortedMap::remove_range`]):
/// the collecting scan failed, or one of the remove batches did.
#[derive(Debug)]
pub enum RangeWriteError {
    Read(ReadError),
    Write(WriteError),
}

impl std::fmt::Display for RangeWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RangeWriteError::Read(e) => write!(f, "range scan failed: {e}"),
            RangeWriteError::Write(e) => write!(f, "range remove failed: {e}"),
        }
    }
}

impl std::error::Error for RangeWriteError {}

/// Unwraps the version of a batch that carries no checks — such a batch cannot
/// fail a precondition.
fn applied_version(outcome: ApplyOutcome) -> Version {
    match outcome {
        ApplyOutcome::Applied { version, .. } => version,
        ApplyOutcome::ConditionFailed { .. } => {
            unreachable!("a checkless batch cannot fail a precondition")
        }
    }
}

impl<B: Backend> SortedMap<B> {
    /// Wraps a store.
    pub fn over(store: Store<B>) -> Self {
        SortedMap {
            map: BTreeMap::over(store),
        }
    }

    /// Wraps an existing engine-level map handle.
    pub fn from_map(map: BTreeMap<B>) -> Self {
        SortedMap { map }
    }

    /// The engine-level map, for anything this surface doesn't expose
    /// (permits, grouped multi-get, verify).
    pub fn map(&self) -> &BTreeMap<B> {
        &self.map
    }

    /// The underlying store handle.
    pub fn store(&self) -> &Store<B> {
        self.map.store()
    }

    /// Entries in the visible version.
    pub fn len(&self) -> u64 {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Captures a consistent snapshot (see [`BTreeMap::snapshot`]).
    pub fn snapshot(&self) -> Snapshot<B> {
        self.map.snapshot()
    }

    // -- Point reads ---------------------------------------------------------

    /// Point read.
    pub fn get(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl Future<Output = Result<Option<Vec<u8>>, ReadError>> + Send {
        self.map.get(key.as_ref().to_vec())
    }

    /// Whether `key` is present.
    pub fn contains_key(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl Future<Output = Result<bool, ReadError>> + Send {
        let get = self.get(key);
        async move { Ok(get.await?.is_some()) }
    }

    /// Multi-get against one consistent version (routing per
    /// [`BTreeMap::get_many`]).
    pub fn get_many(
        &self,
        keys: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<Vec<Option<Vec<u8>>>, ReadError>> + Send {
        self.map.get_many(keys)
    }

    // -- Mutation ------------------------------------------------------------

    /// Inserts or replaces one entry; resolves to the published [`Version`].
    /// Visible immediately; durable after the backend barrier
    /// (`flush`/`commit`) covers the returned version.
    ///
    /// Named `insert` per mapdb-collections' std-aligned vocabulary; unlike
    /// the in-memory crate it returns the durability watermark, not the
    /// displaced value (use [`compare_and_set`](Self::compare_and_set) when
    /// the old value matters).
    pub fn insert(
        &self,
        key: impl Into<Vec<u8>>,
        value: impl Into<Vec<u8>>,
    ) -> impl Future<Output = Result<Version, WriteError>> + Send {
        let apply = self.map.apply(WriteBatch::new().insert(key, value));
        async move { Ok(applied_version(apply.await?)) }
    }

    /// Removes one entry (a no-op version bump if absent); resolves to the
    /// published [`Version`].
    pub fn remove(
        &self,
        key: impl Into<Vec<u8>>,
    ) -> impl Future<Output = Result<Version, WriteError>> + Send {
        let apply = self.map.apply(WriteBatch::new().remove(key));
        async move { Ok(applied_version(apply.await?)) }
    }

    /// Inserts/replaces every entry as **one atomic batch**; resolves to the
    /// single published [`Version`] covering them all. (Bulk name per
    /// mapdb-collections' `put_all`.)
    pub fn put_all<K, V>(
        &self,
        entries: impl IntoIterator<Item = (K, V)>,
    ) -> impl Future<Output = Result<Version, WriteError>> + Send
    where
        K: Into<Vec<u8>>,
        V: Into<Vec<u8>>,
    {
        let mut batch = WriteBatch::new();
        for (k, v) in entries {
            batch = batch.insert(k, v);
        }
        let apply = self.map.apply(batch);
        async move { Ok(applied_version(apply.await?)) }
    }

    /// Inserts `key` only if absent. `Ok(Some(version))` on insert,
    /// `Ok(None)` if the key already exists (nothing published).
    pub fn put_if_absent(
        &self,
        key: impl Into<Vec<u8>>,
        value: impl Into<Vec<u8>>,
    ) -> impl Future<Output = Result<Option<Version>, WriteError>> + Send {
        let key = key.into();
        let apply = self.map.apply(
            WriteBatch::new()
                .check(key.clone(), None)
                .insert(key, value),
        );
        async move {
            Ok(match apply.await? {
                ApplyOutcome::Applied { version, .. } => Some(version),
                ApplyOutcome::ConditionFailed { .. } => None,
            })
        }
    }

    /// Atomic compare-and-set: if `key`'s current value equals `expected`
    /// (`None` = absent), set it to `new` (`None` = remove). `Ok(Some(version))`
    /// on success, `Ok(None)` if the precondition failed (nothing published).
    pub fn compare_and_set(
        &self,
        key: impl Into<Vec<u8>>,
        expected: Option<Vec<u8>>,
        new: Option<Vec<u8>>,
    ) -> impl Future<Output = Result<Option<Version>, WriteError>> + Send {
        let key = key.into();
        let batch = WriteBatch::new().check(key.clone(), expected);
        let batch = match new {
            Some(v) => batch.insert(key, v),
            None => batch.remove(key),
        };
        let apply = self.map.apply(batch);
        async move {
            Ok(match apply.await? {
                ApplyOutcome::Applied { version, .. } => Some(version),
                ApplyOutcome::ConditionFailed { .. } => None,
            })
        }
    }

    // -- Ordered scans (consumer callbacks) ----------------------------------

    /// Scans every entry in ascending key order. The consumer receives
    /// borrowed `(key, value)` slices and returns
    /// [`ControlFlow::Continue`]`(())` to keep going or `Break` to stop (the
    /// summary then reports `completed: false`). The whole scan runs against
    /// one internally captured snapshot.
    pub fn for_each<F>(
        &self,
        consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        F: FnMut(&[u8], &[u8]) -> ControlFlow<()> + Send,
    {
        self.for_each_range::<&[u8], _, F>(.., consumer)
    }

    /// Scans the entries whose keys lie in `range`, in ascending key order —
    /// any `RangeBounds` over byte-slice-like keys: `b"a"..b"m"`,
    /// `start..=end`, `..`, `(Bound::Excluded(k), Bound::Unbounded)`, …
    pub fn for_each_range<K, R, F>(
        &self,
        range: R,
        consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> ControlFlow<()> + Send,
    {
        self.map.for_each_range(range, consumer)
    }

    /// Scans the entries whose keys lie in `range` in **descending** key
    /// order — the reverse-iteration analog, same callback
    /// contract as [`for_each_range`](Self::for_each_range).
    pub fn for_each_range_rev<K, R, F>(
        &self,
        range: R,
        consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> ControlFlow<()> + Send,
    {
        self.map.for_each_range_rev(range, consumer)
    }

    /// Scans every entry in **descending** key order.
    pub fn for_each_rev<F>(
        &self,
        consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        F: FnMut(&[u8], &[u8]) -> ControlFlow<()> + Send,
    {
        self.for_each_range_rev::<&[u8], _, F>(.., consumer)
    }

    /// Scans every entry whose key starts with `prefix`, in ascending order.
    pub fn for_each_prefix<F>(
        &self,
        prefix: impl AsRef<[u8]>,
        consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        F: FnMut(&[u8], &[u8]) -> ControlFlow<()> + Send,
    {
        let prefix = prefix.as_ref().to_vec();
        let end = match prefix_end(&prefix) {
            Some(e) => Bound::Excluded(e),
            None => Bound::Unbounded,
        };
        self.for_each_range((Bound::Included(prefix), end), consumer)
    }

    /// Scans every entry whose key starts with `prefix`, in **descending**
    /// order — e.g. the latest entry under a time-ordered prefix.
    pub fn for_each_prefix_rev<F>(
        &self,
        prefix: impl AsRef<[u8]>,
        consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        F: FnMut(&[u8], &[u8]) -> ControlFlow<()> + Send,
    {
        let prefix = prefix.as_ref().to_vec();
        let end = match prefix_end(&prefix) {
            Some(e) => Bound::Excluded(e),
            None => Bound::Unbounded,
        };
        self.for_each_range_rev((Bound::Included(prefix), end), consumer)
    }

    /// Collects up to `limit` entries of `range` into owned pairs, in
    /// ascending key order. A bounded convenience over
    /// [`for_each_range`](Self::for_each_range) — there is deliberately no
    /// unbounded collector.
    pub fn collect_range<K, R>(
        &self,
        range: R,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<Entry>, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: RangeBounds<K>,
    {
        let snap = self.snapshot();
        // Own the bounds eagerly: the async block then captures plain byte
        // vectors instead of the caller's range type (which need not be Send).
        let bounds = crate::read::own_bounds(&range);
        async move {
            let mut out: Vec<Entry> = Vec::new();
            snap.for_each_range(bounds, |k, v| {
                if out.len() >= limit {
                    return ControlFlow::Break(());
                }
                out.push((k.to_vec(), v.to_vec()));
                if out.len() >= limit {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .await?;
            Ok(out)
        }
    }

    /// Collects up to `limit` entries of `range` into owned pairs, in
    /// **descending** key order — "the last N of a range" (e.g. newest-first
    /// pagination: pass `(Unbounded, Excluded(last_page_smallest_key))` to
    /// fetch the next-older page).
    pub fn collect_range_rev<K, R>(
        &self,
        range: R,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<Entry>, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: RangeBounds<K>,
    {
        let snap = self.snapshot();
        let bounds = crate::read::own_bounds(&range);
        async move {
            let mut out: Vec<Entry> = Vec::new();
            snap.for_each_range_rev(bounds, |k, v| {
                if out.len() >= limit {
                    return ControlFlow::Break(());
                }
                out.push((k.to_vec(), v.to_vec()));
                if out.len() >= limit {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .await?;
            Ok(out)
        }
    }

    /// The entry with the smallest key, or `None` when empty.
    pub fn first_entry(&self) -> impl Future<Output = Result<Option<Entry>, ReadError>> + Send {
        let snap = self.snapshot();
        async move { snap.first_entry().await }
    }

    /// The entry with the largest key, or `None` when empty.
    pub fn last_entry(&self) -> impl Future<Output = Result<Option<Entry>, ReadError>> + Send {
        let snap = self.snapshot();
        async move { snap.last_entry().await }
    }

    /// The greatest entry with key `<= key` (NavigableMap `floor`).
    pub fn floor_entry(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl Future<Output = Result<Option<Entry>, ReadError>> + Send {
        let snap = self.snapshot();
        let key = key.as_ref().to_vec();
        async move { snap.floor_entry(key).await }
    }

    /// The greatest entry with key `< key` (NavigableMap `lower`).
    pub fn lower_entry(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl Future<Output = Result<Option<Entry>, ReadError>> + Send {
        let snap = self.snapshot();
        let key = key.as_ref().to_vec();
        async move { snap.lower_entry(key).await }
    }

    /// The least entry with key `>= key` (NavigableMap `ceiling`).
    pub fn ceiling_entry(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl Future<Output = Result<Option<Entry>, ReadError>> + Send {
        let snap = self.snapshot();
        let key = key.as_ref().to_vec();
        async move { snap.ceiling_entry(key).await }
    }

    /// The least entry with key `> key` (NavigableMap `higher`).
    pub fn higher_entry(
        &self,
        key: impl AsRef<[u8]>,
    ) -> impl Future<Output = Result<Option<Entry>, ReadError>> + Send {
        let snap = self.snapshot();
        let key = key.as_ref().to_vec();
        async move { snap.higher_entry(key).await }
    }

    /// Counts the entries in `range` (a scan that copies nothing).
    pub fn count_range<K, R>(&self, range: R) -> impl Future<Output = Result<u64, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: RangeBounds<K>,
    {
        let scan = self.for_each_range(range, |_k: &[u8], _v: &[u8]| ControlFlow::Continue(()));
        async move { Ok(scan.await?.visited) }
    }

    /// Removes every entry whose key lies in `range`; resolves to the number
    /// of keys submitted for removal. (Per mapdb-collections'
    /// `TreeMap::remove_range`, adapted to the storage layer.)
    ///
    /// **Not atomic**: the keys are collected from one snapshot, then removed
    /// in batches of at most [`MAX_OPS_PER_BATCH`](crate::admission::AdmissionLimits)-sized
    /// chunks — a concurrent writer can interleave between chunks, and a key
    /// it re-inserts after its chunk was applied survives. Each individual
    /// chunk is atomic. For an atomic bounded delete, build a
    /// [`WriteBatch`] of removes yourself.
    pub fn remove_range<K, R>(
        &self,
        range: R,
    ) -> impl Future<Output = Result<u64, RangeWriteError>> + Send
    where
        K: AsRef<[u8]>,
        R: RangeBounds<K>,
    {
        // Sized so a chunk's worst-case admission cost (~2 pages/op + the
        // tree-depth bound) stays comfortably inside the default
        // max_dirty_pages budget with its commit reserve.
        const CHUNK: usize = 512;
        let snap = self.snapshot();
        let bounds = crate::read::own_bounds(&range);
        let map = self.map.clone();
        async move {
            let mut keys: Vec<Vec<u8>> = Vec::new();
            snap.for_each_range(bounds, |k, _v| {
                keys.push(k.to_vec());
                ControlFlow::Continue(())
            })
            .await
            .map_err(RangeWriteError::Read)?;
            let removed = keys.len() as u64;
            for chunk in keys.chunks(CHUNK) {
                let mut batch = WriteBatch::new();
                for k in chunk {
                    batch = batch.remove(k.clone());
                }
                map.apply(batch).await.map_err(RangeWriteError::Write)?;
            }
            Ok(removed)
        }
    }

    /// Removes every entry ([`remove_range`](Self::remove_range) over `..`).
    /// Same non-atomicity caveat.
    pub fn clear(&self) -> impl Future<Output = Result<u64, RangeWriteError>> + Send {
        self.remove_range::<&[u8], _>(..)
    }

    // -- Lifecycle -----------------------------------------------------------

    /// Stops intake, drains accepted writes, and closes the store.
    pub fn close(&self) -> impl Future<Output = Result<(), CloseError>> + Send {
        self.map.close()
    }
}

impl SortedMap<Direct> {
    /// Direct durability barrier for `version` — see [`BTreeMap::flush`].
    pub fn flush(&self, version: Version) -> impl Future<Output = Result<(), FlushError>> + Send {
        self.map.flush(version)
    }
}

impl SortedMap<Wal> {
    /// WAL durability barrier for `version` — see [`BTreeMap::commit`].
    pub fn commit(&self, version: Version) -> impl Future<Output = Result<(), CommitError>> + Send {
        self.map.commit(version)
    }
}
