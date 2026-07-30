//! `mapdb-uring` — Linux-first asynchronous page stores and a limited B+tree
//! map, io_uring-backed.
//!
//! The complete contract — durability, operations, performance and the public
//! API rules — is in the `doc/` directory of the repository:
//! <https://github.com/mapdb/mapdb-rust-uring/tree/main/doc>.
//!
//! ## What this crate is today
//!
//! Two page stores over a copy-on-write B+tree with never-reused page ids:
//! [`StoreDirect`] (group-flush durability with a double-slot alternating root
//! header — a torn header write falls back to the previous committed root; no
//! WAL, so unflushed writes are lost on crash) and
//! [`StoreWal`] (WAL with group commit, crash recovery, checkpoint-by-rotation,
//! and compaction). Both run every byte through the [`io::PageIo`] boundary:
//! [`io::SyscallIo`] (default; pread/pwrite worker pool with a `preadv2`
//! NOWAIT inline fast path) or, under the `io-uring` feature, `io::CompioIo`
//! (a persistent ring on a dedicated driver thread). Checkpoint and compaction
//! exist and are tested. The map API is point ops (get/get_many/apply) plus
//! **consumer-callback ordered scans** — [`SortedMap`] layers range/prefix
//! scans, the NavigableMap lookups (`first_entry`/`floor_entry`/…),
//! `remove_range`, and compare-and-set over [`BTreeMap`]; there is
//! deliberately no iterator type.
//!
//! ## Backend markers
//!
//! ```
//! use mapdb_uring::{Store, Direct, Wal, io::FakeIo, store::Options};
//! use std::sync::Arc;
//!
//! let data = Arc::new(FakeIo::new(0));
//! let store: Store<Direct> = Store::<Direct>::create_with_io(data, Options::default()).unwrap();
//! # let _ = store;
//! ```

pub mod admission;
pub mod backend;
pub mod batch;
mod btree;
pub mod cache;
pub mod checkpoint;
pub mod compact;
pub mod direct;
mod direct_fs;
pub mod error;
pub mod fileset;
mod fs_util;
pub mod io;
pub mod map;
pub mod metrics;
pub mod page;
pub mod read;
mod retention;
pub mod sorted;
pub mod state;
pub mod store;
pub mod wal;
mod wal_fs;
mod writer;

#[cfg(test)]
mod tests_store_actor;

#[cfg(test)]
mod tests_read_path;

#[cfg(test)]
mod tests_direct_writes;

#[cfg(test)]
mod tests_wal_recovery;

#[cfg(test)]
mod tests_runtime_and_load;

#[cfg(test)]
mod tests_checkpoint_compaction;

#[cfg(test)]
mod tests_directory_fsync_model;

#[cfg(test)]
mod tests_handle_drop_lifecycle;

#[cfg(test)]
mod tests_direct_header_slots;

#[cfg(test)]
mod tests_create_path_safety;

#[cfg(test)]
mod tests_real_io_parity;

#[cfg(test)]
mod tests_crash_recovery_fuzz;

#[cfg(test)]
mod tests_get_many_concurrency;

#[cfg(test)]
mod tests_overflow_prefetch;

#[cfg(test)]
mod tests_sorted_map_scans;

pub use backend::{Backend, Direct, Wal};
pub use batch::{ApplyOutcome, BatchError, Check, Op, PageBatch, WriteBatch};
pub use cache::{CacheMetrics, CacheState, PageCache};
pub use compact::CompactReport;
pub use error::{
    CloseError, CommitError, FlushError, OpenError, PermitMismatch, PoisonReason, ReadError,
    Terminal, WriteError,
};
pub use fileset::{MemWalFileSet, WalFileSet};
#[cfg(feature = "bench-instrument")]
pub use io::{DeviceWrapper, IoRole};
pub use io::{IoBackend, IoKind};
pub use map::{BTreeMap, VerifyReport, WritePermit, GROUPED_ROUTE_MIN_BATCH};
pub use metrics::Metrics;
pub use read::{BuiltTree, Entry, ScanSummary, TreeBuilder, TreeReader, TreeSnapshot};
pub use sorted::{RangeWriteError, SortedMap};
pub use store::{Options, Snapshot, Store};
pub use version::{Incarnation, Version};
pub use wal::CheckpointReport;

pub mod version;

/// Convenience aliases making the crash guarantee visible at the type.
pub type StoreDirect = Store<Direct>;
pub type StoreWal = Store<Wal>;

// ---------------------------------------------------------------------------
// M0 exit requirement: public futures and handles are `Send`; map is `Sync`.
// These are compile-time assertions; if a future or handle regresses to
// non-`Send`, the crate fails to build.
// ---------------------------------------------------------------------------
#[allow(dead_code)]
mod send_sync_asserts {
    use super::*;
    use crate::admission::WriteCost;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_send_val<T: Send>(_: &T) {}

    #[allow(unused)]
    fn assertions() {
        // Handles.
        assert_send::<Store<Direct>>();
        assert_sync::<Store<Direct>>();
        assert_send::<Store<Wal>>();
        assert_sync::<Store<Wal>>();
        assert_send::<BTreeMap<Direct>>();
        assert_sync::<BTreeMap<Direct>>();
        assert_send::<BTreeMap<Wal>>();
        assert_sync::<BTreeMap<Wal>>();
        assert_send::<Snapshot<Direct>>();
        assert_sync::<Snapshot<Direct>>();
        assert_send::<WritePermit<Direct>>();

        // Public futures are `Send`.
        let map: BTreeMap<Direct> = unreachable_handle();
        assert_send_val(&map.get(Vec::new()));
        assert_send_val(&map.get_many(Vec::new()));
        assert_send_val(&map.apply(WriteBatch::new()));
        assert_send_val(&map.verify());
        assert_send_val(&map.close());
        assert_send_val(&map.flush(unreachable_version()));
        let cost = WriteCost {
            encoded_bytes: 0,
            op_count: 0,
            worst_case_pages: 0,
            wal_worst_case_bytes: 0,
        };
        assert_send_val(&map.reserve_write(cost));

        let wal: BTreeMap<Wal> = unreachable_handle();
        assert_send_val(&wal.commit(unreachable_version()));

        // The sorted-map layer's scan futures are `Send`: assert the whole
        // public scan surface, not just point reads.
        let consume = |_k: &[u8], _v: &[u8]| std::ops::ControlFlow::Continue(());
        assert_send_val(&map.for_each_range::<&[u8], _, _>(.., consume));
        assert_send_val(&map.for_each_range_rev::<&[u8], _, _>(.., consume));
        assert_send::<SortedMap<Direct>>();
        assert_sync::<SortedMap<Direct>>();
        let sorted: SortedMap<Direct> = SortedMap::from_map(unreachable_handle());
        assert_send_val(&sorted.for_each(consume));
        assert_send_val(&sorted.for_each_prefix(b"p", consume));
        assert_send_val(&sorted.collect_range::<&[u8], _>(.., 1));
        assert_send_val(&sorted.for_each_range_rev::<&[u8], _, _>(.., consume));
        assert_send_val(&sorted.for_each_rev(consume));
        assert_send_val(&sorted.for_each_prefix_rev(b"p", consume));
        assert_send_val(&sorted.collect_range_rev::<&[u8], _>(.., 1));
        assert_send_val(&sorted.first_entry());
        assert_send_val(&sorted.last_entry());
        assert_send_val(&sorted.insert(b"k".to_vec(), b"v".to_vec()));
        assert_send_val(&sorted.compare_and_set(b"k".to_vec(), None, None));
        assert_send_val(&sorted.floor_entry(b"k"));
        assert_send_val(&sorted.ceiling_entry(b"k"));
        assert_send_val(&sorted.lower_entry(b"k"));
        assert_send_val(&sorted.higher_entry(b"k"));
        assert_send_val(&sorted.remove_range::<&[u8], _>(..));
        assert_send_val(&sorted.clear());
    }

    fn unreachable_handle<B: Backend>() -> BTreeMap<B> {
        unreachable!("type-level only; never executed")
    }
    fn unreachable_version() -> Version {
        unreachable!("type-level only; never executed")
    }
}
