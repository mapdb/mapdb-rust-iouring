//! Revocable generation leases (M6 Phase E).
//!
//! Phase B's rotation is correctness-self-sufficient but *unbounded*: a retired
//! WAL segment's bytes free when the last old-generation `Arc` drops, and a
//! held/leaked [`Snapshot`](crate::store::Snapshot) used to hold exactly such an
//! `Arc` — an irrevocable strong reference no policy could reclaim.
//! This module makes old generations *forcibly* reclaimable while
//! keeping invariant 8 intact: **no in-flight read ever loses its device**.
//!
//! ## Shape
//!
//! Every published generation gets one [`SnapshotGen`] — a revocable holder of
//! the generation's `Arc<FileGeneration>`. A `Snapshot` no longer owns the
//! descriptor; it owns a counted [`GenHold`] on the `SnapshotGen` plus copies of
//! the descriptor's plain fields. A snapshot read acquires a short-lived
//! [`ReadLease`] (which clones the generation `Arc` under the state lock), does
//! its traversal, and releases it. The strong reference chain is therefore:
//!
//! ```text
//! Snapshot ──GenHold──▶ SnapshotGen ──state.file──▶ FileGeneration ──▶ devices
//!                        ▲     ReadLease.file ───────────▶ (same, per read)
//!                        └── Retention queue (after retirement)
//! ```
//!
//! ## Release rules (the whole protocol)
//!
//! The generation's strong reference (`state.file`) is dropped exactly once,
//! when one of these holds — always evaluated under the one state mutex:
//!
//! - **natural**: `retired && holders == 0 && leases == 0` — the generation was
//!   superseded by a rebind and the last snapshot handle (or pending read
//!   future) is gone. This preserves Phase B's "frees the instant the last
//!   snapshot drops".
//! - **forced**: `revoked && leases == 0` — expiration policy: new leases are
//!   refused (`ReadError::SnapshotExpired`), in-flight leases drain, and the
//!   *last lease out* performs the release. Revocation never blocks the writer
//!   and never yanks a device from a read that already holds its lease clone.
//!
//! The **active** generation (`retired == false`, reachable through the current
//! `RootDescriptor`) is never releasable: both rules require a flag only the
//! rebind/retire path sets.
//!
//! ## Expiration policy
//!
//! [`Retention`] keeps the retired-generation queue, oldest first. When a
//! rebind retires a generation past [`Options::max_retained_generations`]
//! (crate::store::Options), the oldest retained ones are revoked, oldest-wins.
//! The bound is a **generation count**, not a wall-clock age: it needs no clock,
//! is exactly testable, and maps 1:1 onto what actually pins space (one retired
//! WAL segment per retired generation). `None` opts out — snapshots then stay
//! readable indefinitely and bounded physical space is explicitly **not**
//! offered (a product-decision escape hatch, stated rather than
//! silently hoped for).
//!
//! ## Orphaned loads and device-anchored accounting (M6 Phase H / H5)
//!
//! A snapshot read future dropped *mid-load* releases its lease, but the
//! cancellation-independent single-flight may park the orphaned load future —
//! which owns a device `Arc` — in the cache's `Loading` slot with no waiter left
//! to drive it. The generation then reads as released while one page-load's
//! device clone lingers.
//!
//! Byte accounting is therefore **device-anchored**, not generation-anchored: a
//! departed device (a retired WAL segment, a compaction-superseded data inode) is
//! registered in [`Retention::note_superseded`] at retire, and its bytes count in
//! `retained_segment_bytes` for exactly as long as its `Weak` upgrades — i.e.
//! while *any* owner is alive: an unreleased generation, an in-flight lease, or a
//! parked orphaned load. So the metric stays honest even when a parked load
//! outlives its generation (the pre-H5 metric dropped to 0 while the inode was
//! still pinned). The load-bearing invariant: **no component outside
//! {generations, in-flight ops, parked loads} may retain a superseded device
//! `Arc`** — every such owner genuinely pins the inode, so counting by device
//! liveness is exact (a future device-handle cache would silently break it).
//!
//! Prompt *reclamation* of the parked clone is the cache's job: a non-blocking
//! poll-once sweep drives **every** parked `Loading` slot toward terminal
//! completion at each rebind (the set is small — the concurrently-in-flight loads
//! at a flushed barrier), and a blocking drain at close/teardown (see
//! `PageCache`). So between rebinds the residue is bounded by in-flight loads and
//! cleared at the next rebind or close; a purely *idle* store (no rebinds) with a
//! dropped mid-load snapshot read keeps that one visible, counted clone until it
//! next rebinds or closes — a bounded, accounted latency, not an unbounded leak.
//! A periodic idle sweep would close even that latency and is the one deferred
//! piece.

use crate::backend::FileGeneration;
use crate::error::ReadError;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// All lease/holder state for one generation, under one mutex. Critical
/// sections are a few loads and an `Arc` clone — never I/O, never `.await`
/// (invariant 12).
struct GenState {
    /// The revocable strong reference; `None` once released (terminal).
    file: Option<Arc<FileGeneration>>,
    /// Live `GenHold`s: snapshot handles + pending snapshot-read futures.
    holders: u32,
    /// `ReadLease`s currently traversing (device I/O possibly in flight).
    leases: u32,
    /// Superseded by a rebind; eligible for natural release.
    retired: bool,
    /// Force-expired; refuses new leases, releases at lease drain.
    revoked: bool,
}

/// The revocable holder of one generation's strong reference.
pub(crate) struct SnapshotGen {
    state: Mutex<GenState>,
}

impl SnapshotGen {
    /// A fresh holder for a newly published (active) generation.
    pub(crate) fn active(file: Arc<FileGeneration>) -> Arc<Self> {
        Arc::new(SnapshotGen {
            state: Mutex::new(GenState {
                file: Some(file),
                holders: 0,
                leases: 0,
                retired: false,
                revoked: false,
            }),
        })
    }

    /// This (unreleased) generation's device for `member`, for tests to register
    /// as departed at retire — exactly what a rebind caller does with
    /// `current.file.device(..)`, but reaching through the `SnapshotGen` the test
    /// holds. `None` once released.
    #[cfg(test)]
    fn device(&self, member: crate::backend::Member) -> Option<Arc<dyn crate::io::PageIo>> {
        self.state.lock().file.as_ref().map(|f| f.device(member))
    }

    /// Takes a counted hold, or `None` if the generation is released **or
    /// revoked** — the caller (a `snapshot()` racing a rebind) must reload the
    /// descriptor and retry; the fresh load observes the *new* generation, so
    /// the retry terminates. Refusing a revoked-but-*draining* generation (its
    /// `file` still `Some` while an in-flight lease finishes) matters: a
    /// capture that accepted it would return a born-dead snapshot whose every
    /// read fails `SnapshotExpired`.
    pub(crate) fn try_hold(self: &Arc<Self>) -> Option<GenHold> {
        let mut s = self.state.lock();
        if s.revoked {
            return None;
        }
        s.file.as_ref()?;
        s.holders += 1;
        Some(GenHold {
            gen: Arc::clone(self),
        })
    }

    /// Acquires a read lease: the generation `Arc` to traverse with. Refused
    /// once revoked or released. The lease clone keeps the devices alive for
    /// the whole traversal even if revocation lands one instant later — that is
    /// the invariant-8 half of the design.
    pub(crate) fn lease(self: &Arc<Self>) -> Result<ReadLease, ReadError> {
        let mut s = self.state.lock();
        if s.revoked {
            return Err(ReadError::SnapshotExpired);
        }
        let Some(file) = s.file.as_ref().map(Arc::clone) else {
            return Err(ReadError::SnapshotExpired);
        };
        s.leases += 1;
        Ok(ReadLease {
            file,
            gen: Arc::clone(self),
        })
    }

    /// Marks the generation superseded (rebind retired it). Releases immediately
    /// if nothing holds it. Byte attribution now rides the *device* lifetime
    /// (`Retention::note_superseded`, H5), not the generation, so retire no longer
    /// carries a byte figure.
    fn retire(&self) {
        let released = {
            let mut s = self.state.lock();
            s.retired = true;
            maybe_release_locked(&mut s)
        };
        drop(released); // generation drop (device closes) outside the lock
    }

    /// Force-expires: refuses new leases; the release happens now (no leases)
    /// or when the last in-flight lease drains. Never waits. Returns whether
    /// this call flagged a still-live generation (`false` if it had already
    /// released naturally — a racing last-hold drop must not be counted as a
    /// force-expiration) and the released `Arc`, which the caller must drop
    /// **outside every retention lock**: the last generation ref's teardown
    /// closes device handles and reclaims an unlinked, possibly
    /// `max_wal_bytes`-sized inode.
    #[must_use]
    fn revoke(&self) -> (bool, Option<Arc<FileGeneration>>) {
        let mut s = self.state.lock();
        let was_live = s.file.is_some() && !s.revoked;
        s.revoked = true;
        let released = maybe_release_locked(&mut s);
        (was_live, released)
    }

    /// Whether the strong reference has been dropped.
    pub(crate) fn is_released(&self) -> bool {
        self.state.lock().file.is_none()
    }

    /// Whether new leases would be refused (`SnapshotExpired`).
    pub(crate) fn is_expired(&self) -> bool {
        let s = self.state.lock();
        s.revoked || s.file.is_none()
    }

    fn unhold(&self) {
        let released = {
            let mut s = self.state.lock();
            debug_assert!(s.holders > 0, "unbalanced GenHold");
            s.holders -= 1;
            maybe_release_locked(&mut s)
        };
        drop(released);
    }

    fn unlease(&self) {
        let released = {
            let mut s = self.state.lock();
            debug_assert!(s.leases > 0, "unbalanced ReadLease");
            s.leases -= 1;
            maybe_release_locked(&mut s)
        };
        drop(released);
    }
}

/// The one release decision. Returns the taken `Arc` so the caller drops it
/// outside the state lock (a `FileGeneration` drop closes device handles;
/// nothing heavy should run under the mutex).
fn maybe_release_locked(s: &mut GenState) -> Option<Arc<FileGeneration>> {
    if s.file.is_none() || s.leases != 0 {
        return None;
    }
    let forced = s.revoked;
    let natural = s.retired && s.holders == 0;
    if forced || natural {
        // The heavy `file` teardown stays outside the lock. Byte attribution is
        // now device-anchored (`Retention::superseded`), so nothing generation-
        // held needs clearing here.
        s.file.take()
    } else {
        None
    }
}

/// A counted hold on a generation: one per live `Snapshot` handle and one per
/// pending snapshot-read future. Holds keep a **retired** generation pinned
/// (natural release waits for them); they do not stop a revocation.
pub(crate) struct GenHold {
    gen: Arc<SnapshotGen>,
}

impl GenHold {
    pub(crate) fn gen(&self) -> &Arc<SnapshotGen> {
        &self.gen
    }
}

impl Clone for GenHold {
    fn clone(&self) -> Self {
        self.gen.state.lock().holders += 1;
        GenHold {
            gen: Arc::clone(&self.gen),
        }
    }
}

impl Drop for GenHold {
    fn drop(&mut self) {
        self.gen.unhold();
    }
}

/// A short-lived read lease: the generation to traverse with, held for exactly
/// one read. Dropping it (including by cancelling the read future) releases the
/// lease; the last lease out of a revoked generation performs the release.
pub(crate) struct ReadLease {
    /// The strong clone the traversal reads through. Alive for the lease's whole
    /// lifetime regardless of any concurrent revocation.
    file: Arc<FileGeneration>,
    gen: Arc<SnapshotGen>,
}

impl ReadLease {
    pub(crate) fn file(&self) -> &Arc<FileGeneration> {
        &self.file
    }
}

impl Drop for ReadLease {
    fn drop(&mut self) {
        self.gen.unlease();
        // `self.file` drops right after, in the same drop glue: the release
        // decision above and the final Arc release are one synchronous step
        // from the caller's point of view.
    }
}

/// A point-in-time retention accounting snapshot (see [`Retention::metrics`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RetentionMetrics {
    /// Retired generations still pinned by holds, **not yet revoked** — the
    /// figure the policy bound governs.
    pub retained: u64,
    /// Revoked generations still pinned by in-flight leases (drain pending).
    /// Outside the bound (their space is already unreclaimable-by-policy) but
    /// their bytes are real until the drain.
    pub draining: u64,
    /// Bytes pinned by **both** classes — the operator-facing disk truth.
    pub bytes: u64,
    /// Lifetime force-expirations (revocations of live generations).
    pub expired: u64,
}

/// The retired-generation registry and expiration policy (one per store core).
///
/// Lock order: `queue` and `draining` are leaves taken one at a time, each
/// strictly before any `SnapshotGen` state lock (`is_released`/`revoke`);
/// no path acquires them the other way, and no generation `Arc` is dropped
/// while either is held.
pub(crate) struct Retention {
    /// Retired, unrevoked generations, oldest first. Entries leave when
    /// released (swept opportunistically) or when revoked by the bound.
    queue: Mutex<VecDeque<Arc<SnapshotGen>>>,
    /// Revoked generations whose in-flight leases have not yet drained. Kept
    /// visible so `retained_segment_bytes` never reads 0 while a wedged read
    /// pins a whole segment.
    draining: Mutex<Vec<Arc<SnapshotGen>>>,
    /// `Options::max_retained_generations`: how many retired generations may
    /// stay pinned before oldest-wins revocation. `None` = unbounded.
    bound: Option<u32>,
    /// Generations force-expired so far (metric `expired_generations`).
    expired: AtomicU64,
    /// Devices that have **left the namespace** and whose bytes are therefore
    /// pinned until the last owner drops them (H5). A retired WAL segment
    /// (unlinked at checkpoint) or a compaction-superseded data inode is
    /// registered here at retire; its bytes count in `retained_segment_bytes`
    /// for as long as the `Weak` upgrades — i.e. while an unreleased generation,
    /// an in-flight lease, **or a parked single-flight load** still holds the
    /// device. Anchoring on the *device* (not the generation) is what keeps an
    /// orphaned load's pin visible after its generation releases; a shared data
    /// inode is one entry (all read-through generations share one device `Arc`),
    /// so it counts once. Dead entries are pruned at each registration and read.
    superseded: Mutex<Vec<SupersededDevice>>,
}

/// One device that left the namespace, with the bytes it pins (H5).
struct SupersededDevice {
    device: std::sync::Weak<dyn crate::io::PageIo>,
    bytes: u64,
}

impl Retention {
    pub(crate) fn new(bound: Option<u32>) -> Self {
        Retention {
            queue: Mutex::new(VecDeque::new()),
            draining: Mutex::new(Vec::new()),
            bound,
            expired: AtomicU64::new(0),
            superseded: Mutex::new(Vec::new()),
        }
    }

    /// Registers a device that has left the namespace so its `bytes` are counted
    /// as physically pinned while the device Arc lives anywhere — an unreleased
    /// generation, an in-flight lease, or a parked single-flight load (H5).
    /// Called at every retire, once per departing device: a checkpoint registers
    /// its retired WAL segment; a compaction registers its retired segment **and**
    /// the superseded data inode. Prunes dead entries opportunistically so the
    /// registry stays bounded by live pins.
    pub(crate) fn note_superseded(&self, device: &Arc<dyn crate::io::PageIo>, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let weak = Arc::downgrade(device);
        let mut s = self.superseded.lock();
        s.retain(|e| e.device.strong_count() > 0);
        // A given supersede event registers a device once; guard defensively so a
        // double-call could never double-count the same inode. If it somehow
        // re-registered with a *different* byte figure, the dedup silently keeps
        // the first — assert loudly in debug so a future misuse is caught, not
        // quietly mis-counted.
        if let Some(existing) = s.iter().find(|e| e.device.ptr_eq(&weak)) {
            debug_assert_eq!(
                existing.bytes, bytes,
                "same device re-registered with different bytes"
            );
        } else {
            s.push(SupersededDevice {
                device: weak,
                bytes,
            });
        }
    }

    /// Retires a superseded generation: queues it, sweeps already-released
    /// entries, and enforces the bound by revoking the oldest retained
    /// generations. Byte attribution rides the device (`note_superseded`), so
    /// callers register the departing device(s) first. Runs on the writer thread
    /// at a rebind; never blocks on readers (revocation is flag-and-drain), and
    /// every generation teardown (device close / unlinked-inode reclaim) runs
    /// after all retention locks are released.
    pub(crate) fn retire(&self, gen: Arc<SnapshotGen>) {
        gen.retire();
        // Select victims under the queue lock; revoke them after releasing it.
        let victims = {
            let mut q = self.queue.lock();
            if !gen.is_released() {
                q.push_back(gen);
            }
            q.retain(|g| !g.is_released());
            let mut victims = Vec::new();
            if let Some(bound) = self.bound {
                while q.len() > bound as usize {
                    victims.push(q.pop_front().expect("len > bound >= 0"));
                }
            }
            victims
        };
        if victims.is_empty() {
            return;
        }
        // Revoke outside `queue`; collect released Arcs and drop them outside
        // `draining` too (no heavy teardown under any retention lock).
        let mut released = Vec::new();
        {
            let mut draining = self.draining.lock();
            draining.retain(|g| !g.is_released());
            for victim in victims {
                let (was_live, freed) = victim.revoke();
                if was_live {
                    self.expired.fetch_add(1, Ordering::AcqRel);
                }
                match freed {
                    Some(arc) => released.push(arc),
                    // Still pinned by an in-flight lease: keep it visible until
                    // the drain actually frees it.
                    None => {
                        if !victim.is_released() {
                            draining.push(victim);
                        }
                    }
                }
            }
        }
        drop(released);
    }

    /// A point-in-time accounting snapshot. Sweeps released entries from both
    /// registries as a side effect.
    pub(crate) fn metrics(&self) -> RetentionMetrics {
        let retained = {
            let mut q = self.queue.lock();
            q.retain(|g| !g.is_released());
            q.len() as u64
        };
        let draining = {
            let mut d = self.draining.lock();
            d.retain(|g| !g.is_released());
            d.len() as u64
        };
        // Bytes are device-anchored (H5): a departed device's bytes count while
        // its `Weak` still upgrades — an unreleased generation, an in-flight
        // lease, or a parked single-flight load. `upgrade` (not `strong_count`)
        // holds the Arc across the read so a concurrent last-drop cannot make the
        // count race. The upgraded Arcs are **collected and dropped after the
        // registry lock is released**: were the sample's
        // upgrade the last owner, dropping it inside the closure would run device
        // teardown (fd close, driver-thread wake) under the lock — the exact thing
        // the "teardown outside all retention locks" rule bans.
        let mut alive = Vec::new();
        let bytes = {
            let mut sup = self.superseded.lock();
            let mut total = 0u64;
            sup.retain(|e| match e.device.upgrade() {
                Some(arc) => {
                    total += e.bytes;
                    alive.push(arc);
                    true
                }
                None => false,
            });
            total
        };
        drop(alive); // any last-owner teardown now runs outside the registry lock
        RetentionMetrics {
            retained,
            draining,
            bytes,
            expired: self.expired.load(Ordering::Acquire),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DataFileLocator, FileGeneration};
    use crate::io::{FakeIo, PageIo};

    fn gen_over(dev: &Arc<FakeIo>) -> Arc<FileGeneration> {
        Arc::new(FileGeneration::direct(
            Arc::clone(dev) as Arc<dyn PageIo>,
            Arc::new(DataFileLocator::new()),
        ))
    }

    /// Registers a generation's (data) device as departed with `bytes`, then
    /// retires it — mirroring a rebind caller (`note_superseded` +
    /// `retire`). The device is read back from the generation, which stays its
    /// sole strong holder, so its bytes count exactly while the generation (or a
    /// lease/parked load) keeps it alive — the H5 device-anchored contract.
    fn retire_seg(retention: &Retention, g: &Arc<SnapshotGen>, bytes: u64) {
        if let Some(dev) = g.device(crate::backend::Member::Data) {
            retention.note_superseded(&dev, bytes);
        }
        retention.retire(Arc::clone(g));
    }

    /// The invariant-8 half: a lease acquired one instant before revocation
    /// keeps its device; the release happens only when that lease drains.
    #[test]
    fn revoke_drains_in_flight_leases_before_release() {
        let dev = Arc::new(FakeIo::new(0));
        let weak = Arc::downgrade(&dev);
        let g = SnapshotGen::active(gen_over(&dev));
        drop(dev);

        let hold = g.try_hold().expect("active generation holds");
        let lease = g.lease().expect("lease before revocation");

        let retention = Retention::new(Some(0));
        retire_seg(&retention, &g, 4096);
        // Bound 0 → revoked immediately. The in-flight lease still has its
        // device; nothing is released yet.
        assert!(g.is_expired(), "revoked: new leases refused");
        assert!(!g.is_released(), "in-flight lease blocks release");
        assert!(weak.upgrade().is_some(), "device alive while leased");
        assert!(
            matches!(g.lease(), Err(ReadError::SnapshotExpired)),
            "a new lease after revocation is refused"
        );
        assert!(
            g.try_hold().is_none(),
            "a revoked-but-draining generation must refuse new holds"
        );
        // Draining stays visible to accounting until the drain: the
        // segment's bytes are still pinned, so the metric must still say so.
        let m = retention.metrics();
        assert_eq!(
            m,
            RetentionMetrics {
                retained: 0,
                draining: 1,
                bytes: 4096,
                expired: 1
            },
            "a draining generation still counts its pinned bytes"
        );

        // The last lease out performs the release, synchronously.
        drop(lease);
        assert!(g.is_released());
        assert!(weak.upgrade().is_none(), "device freed at lease drain");
        assert_eq!(
            retention.metrics(),
            RetentionMetrics {
                retained: 0,
                draining: 0,
                bytes: 0,
                expired: 1
            },
            "the drained generation leaves the accounting"
        );
        drop(hold);
    }

    /// The natural path Phase B's tests rely on: retired + last hold out →
    /// released immediately, no revocation involved.
    #[test]
    fn retired_generation_releases_when_last_hold_drops() {
        let dev = Arc::new(FakeIo::new(0));
        let weak = Arc::downgrade(&dev);
        let g = SnapshotGen::active(gen_over(&dev));
        drop(dev);

        let hold_a = g.try_hold().unwrap();
        let hold_b = hold_a.clone();

        let retention = Retention::new(Some(8));
        retire_seg(&retention, &g, 1000);
        assert!(!g.is_released(), "two holds pin the retired gen");
        assert_eq!(
            retention.metrics(),
            RetentionMetrics {
                retained: 1,
                draining: 0,
                bytes: 1000,
                expired: 0
            }
        );

        drop(hold_a);
        assert!(!g.is_released(), "one hold still pins it");
        drop(hold_b);
        assert!(g.is_released(), "last hold out releases");
        assert!(weak.upgrade().is_none(), "device freed");
        assert_eq!(
            retention.metrics(),
            RetentionMetrics {
                retained: 0,
                draining: 0,
                bytes: 0,
                expired: 0
            },
            "swept after release"
        );
    }

    /// The active generation is never releasable, whatever the counts do.
    #[test]
    fn active_generation_never_releases() {
        let dev = Arc::new(FakeIo::new(0));
        let g = SnapshotGen::active(gen_over(&dev));
        let hold = g.try_hold().unwrap();
        let lease = g.lease().unwrap();
        drop(lease);
        drop(hold);
        assert!(
            !g.is_released(),
            "holders==leases==0 must not release an active generation"
        );
        assert!(g.lease().is_ok(), "still leasable");
    }

    /// Oldest-wins: the bound revokes from the front of the queue only.
    #[test]
    fn bound_revokes_oldest_first() {
        let retention = Retention::new(Some(1));
        let devs: Vec<Arc<FakeIo>> = (0..3).map(|_| Arc::new(FakeIo::new(0))).collect();
        let gens: Vec<Arc<SnapshotGen>> = devs
            .iter()
            .map(|d| SnapshotGen::active(gen_over(d)))
            .collect();
        // Pin each with a hold so natural release cannot fire.
        let holds: Vec<GenHold> = gens.iter().map(|g| g.try_hold().unwrap()).collect();
        // Drop the local device handles so device lifetime (which now anchors the
        // byte metric, H5) tracks generation lifetime: a revoked-and-released
        // generation's segment must free, dropping its bytes from the metric.
        drop(devs);

        retire_seg(&retention, &gens[0], 10);
        assert!(!gens[0].is_expired(), "within bound");
        retire_seg(&retention, &gens[1], 20);
        assert!(gens[0].is_expired(), "oldest revoked at bound");
        assert!(!gens[1].is_expired(), "newest retained");
        retire_seg(&retention, &gens[2], 30);
        assert!(gens[1].is_expired());
        assert!(!gens[2].is_expired());
        assert_eq!(
            retention.metrics(),
            RetentionMetrics {
                retained: 1,
                draining: 0,
                bytes: 30,
                expired: 2
            }
        );
        drop(holds);
    }

    /// M6 Phase D under H5's device-anchored accounting: a
    /// compaction-superseded inode's bytes are counted **once**, for as long as
    /// *any* holder of that device is alive — even when the compaction-retired
    /// generation releases born-dead and its own (exclusive) segment leaves with
    /// it. Count-once falls out for free: every read-through generation shares one
    /// data-device `Arc`, so it is one registry entry, upgrading while any holder
    /// lives.
    #[test]
    fn a_superseded_inode_counts_once_until_the_last_reader_releases() {
        use crate::backend::Member;
        let loc = || Arc::new(DataFileLocator::new()) as Arc<dyn crate::backend::PageLocator>;
        // One shared data inode; each generation owns its own WAL segment.
        let data: Arc<dyn PageIo> = Arc::new(FakeIo::new(0));
        let seg0 = Arc::new(FakeIo::new(0));
        let seg1 = Arc::new(FakeIo::new(0));
        let other_data: Arc<dyn PageIo> = Arc::new(FakeIo::new(0));
        let seg2 = Arc::new(FakeIo::new(0));
        let g0 = SnapshotGen::active(Arc::new(FileGeneration::wal(
            Arc::clone(&data),
            Arc::clone(&seg0) as Arc<dyn PageIo>,
            loc(),
        )));
        let g1 = SnapshotGen::active(Arc::new(FileGeneration::wal(
            Arc::clone(&data),
            Arc::clone(&seg1) as Arc<dyn PageIo>,
            loc(),
        )));
        let g2 = SnapshotGen::active(Arc::new(FileGeneration::wal(
            Arc::clone(&other_data),
            Arc::clone(&seg2) as Arc<dyn PageIo>,
            loc(),
        )));
        // Drop the local device handles so device lifetime tracks the generations.
        drop((data, seg0, seg1, other_data, seg2));
        let retention = Retention::new(Some(8));

        // g0: a checkpoint-retired generation, held by a snapshot, reading `data`
        // through its own segment (exclusive 100).
        let hold = g0.try_hold().unwrap();
        retention.note_superseded(&g0.device(Member::Wal).unwrap(), 100);
        retention.retire(Arc::clone(&g0));

        // g1: the compaction retiree — supersedes the shared `data` inode (5000)
        // and retires its own segment (200). No holders → releases born-dead, so
        // its exclusive 200 frees with it while the inode's 5000 rides g0.
        retention.note_superseded(&g1.device(Member::Wal).unwrap(), 200);
        retention.note_superseded(&g1.device(Member::Data).unwrap(), 5000);
        retention.retire(Arc::clone(&g1));
        assert!(g1.is_released(), "nothing held the retiree");
        let m = retention.metrics();
        assert_eq!(m.retained, 1);
        assert_eq!(
            m.bytes,
            100 + 5000,
            "g1's exclusive 200 left with it; the inode's 5000 rides g0"
        );

        // A generation over a *different* inode never picks up the pin.
        let hold2 = g2.try_hold().unwrap();
        retention.note_superseded(&g2.device(Member::Wal).unwrap(), 7);
        retention.retire(Arc::clone(&g2));
        assert_eq!(retention.metrics().bytes, 100 + 5000 + 7);

        drop(hold); // the last reader of the superseded inode releases
        assert_eq!(
            retention.metrics(),
            RetentionMetrics {
                retained: 1,
                draining: 0,
                bytes: 7,
                expired: 0
            },
            "the inode's bytes leave with its last reader"
        );
        drop(hold2);
    }

    /// H5 — the leak this device-anchored accounting closes: a device clone that
    /// outlives its generation (a parked orphaned single-flight load owns exactly
    /// such a clone) keeps the superseded inode's bytes counted until *it* drops,
    /// not merely until the generation releases. The pre-H5 generation-anchored
    /// metric dropped to 0 at generation release while the inode was still pinned.
    #[test]
    fn a_parked_device_clone_keeps_bytes_counted_past_generation_release() {
        let dev = Arc::new(FakeIo::new(0));
        let g = SnapshotGen::active(gen_over(&dev));
        // The "parked load": a device clone held independently of the generation.
        let parked: Arc<dyn PageIo> = g.device(crate::backend::Member::Data).unwrap();
        drop(dev);

        let retention = Retention::new(Some(8));
        retention.note_superseded(&g.device(crate::backend::Member::Data).unwrap(), 4096);
        retention.retire(Arc::clone(&g));
        // No holders → the generation releases immediately, but `parked` still owns
        // the inode. Honest accounting keeps counting it.
        assert!(g.is_released(), "no holder pins the generation");
        assert_eq!(
            retention.metrics().bytes,
            4096,
            "the parked device clone still pins the inode — the metric must say so"
        );

        // The orphaned load finally drains and drops its device clone: only now do
        // the bytes leave.
        drop(parked);
        assert_eq!(
            retention.metrics().bytes,
            0,
            "bytes leave exactly when the last device owner (the parked load) drops"
        );
    }

    /// `try_hold` on a released generation refuses, so `snapshot()`'s
    /// load-hold-recheck loop can rely on it.
    #[test]
    fn try_hold_refuses_released_generation() {
        let dev = Arc::new(FakeIo::new(0));
        let g = SnapshotGen::active(gen_over(&dev));
        let retention = Retention::new(Some(0));
        retire_seg(&retention, &g, 0);
        assert!(g.is_released(), "no holders, bound 0 → gone");
        assert!(g.try_hold().is_none());
    }
}
