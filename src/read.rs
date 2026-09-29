//! The read path over real pages: tree traversal, a read-only bulk-load builder
//! fixture, and a [`TreeReader`] handle.
//!
//! Traversal is a set of free async functions over `(cache, &FileGeneration,
//! root: PageRef, …)` so both the fixture handle here and the live
//! `BTreeMap`/`Snapshot` (once the writer builds real pages at M3) share one
//! implementation. Every page load goes through the cache, so single-flight,
//! validation, and eviction apply uniformly.

use crate::backend::{DataFileLocator, FileGeneration};
use crate::cache::{CacheMetrics, PageCache};
use crate::error::ReadError;
use crate::io::PageIo;
use crate::map::VerifyReport;
use crate::page::{
    body_capacity, encode_branch, encode_leaf, encode_overflow, leaf_entry_overhead,
    overflow_chunks, EntryValue, LeafValue, OverflowCursor, OverflowRef, Page, PageId, PageKind,
    PageRef, MAX_INLINE_VALUE_LEN,
};
use arc_swap::ArcSwap;
use futures_lite::future::block_on;
use std::future::Future;
use std::ops::Bound;
use std::pin::Pin;
use std::sync::Arc;

/// Bound on tree height a traversal will follow before declaring the structure
/// corrupt. A checksum-consistent but cyclic/over-deep structure is rejected
/// rather than looping forever.
pub(crate) const MAX_TREE_DEPTH: u32 = 64;

// ---------------------------------------------------------------------------
// Traversal.
// ---------------------------------------------------------------------------

/// Reassembles an overflow value by following its chain from the head (M6 Phase
/// F). Each link is decode-validated by the cache against the checksum its
/// *predecessor* carries, exactly as a tree page is validated against its parent.
///
/// Structure is validated by the shared [`OverflowCursor`] — the one place that
/// knows what a chain must look like — so a read can never disagree with what the
/// live-set walker relocated.
pub(crate) async fn read_overflow(
    cache: &PageCache,
    file: &FileGeneration,
    oref: OverflowRef,
) -> Result<Vec<u8>, ReadError> {
    if let Some(n_links) = prefetch_eligible(cache, file, &oref) {
        return read_overflow_prefetch(cache, file, oref, n_links).await;
    }
    let mut out: Vec<u8> = Vec::with_capacity(oref.total_len as usize);
    let mut cursor = OverflowCursor::new(oref);
    while let Some(r) = cursor.pending() {
        let page = cache.get_page(r, file).await?;
        cursor.accept(&page)?;
        out.extend_from_slice(page.overflow_chunk());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Chain prefetch (T2.2 experiment).
// ---------------------------------------------------------------------------

/// Speculative reads a chain keeps in flight ahead of its cursor (16 × 16 KiB =
/// 256 KiB of private buffers per chain). Pre-declared experiment constant.
pub(crate) const CHAIN_PREFETCH_WINDOW: usize = 16;

/// Smallest chain (in links, head included) that engages speculation for the
/// experiment. The *shipping* threshold, if the experiment's gate passes, is
/// derived from the passing benchmark cells instead of inherited from this
/// constant.
pub(crate) const CHAIN_PREFETCH_MIN_LINKS: usize = 4;

/// A speculative link read: private buffer → reference-independent decode.
/// Resolves to `None` on any read or decode failure — speculation reports no
/// errors of its own; a failed slot just falls back to a demand read.
type SpecFuture<'a> = Pin<Box<dyn Future<Output = Option<Arc<Page>>> + Send + 'a>>;

enum SpecSlotState<'a> {
    Pending(SpecFuture<'a>),
    Done(Option<Arc<Page>>),
}

/// One predicted chain position: the id it predicted, the location the
/// capability resolved for it (compared against authoritative `locate` before
/// consumption), and the read's state.
///
/// The slot **owns its global permit** for its whole life — through the read,
/// through the completed-but-unconsumed private buffer, and through the drain
/// set — so the permit pool bounds private *buffers*, not merely submitted
/// reads (a permit released at read completion would let unbounded chains
/// each retain a full window of completed pages while the pool recycles).
/// It releases when the slot drops:
/// at consumption, at discard, at terminal drain completion, or on caller
/// cancellation.
struct SpecSlot<'a> {
    pos: usize,
    predicted: crate::backend::Location,
    _permit: crate::cache::SpecPermit<'a>,
    state: SpecSlotState<'a>,
}

/// Engagement check: switch on, sane declared length, chain long enough, head
/// cold (a warm chain re-reading its links from the device is pure
/// redundancy), and the generation's locator declares the id-only
/// prediction capability for the first predicted link.
fn prefetch_eligible(
    cache: &PageCache,
    file: &FileGeneration,
    oref: &OverflowRef,
) -> Option<usize> {
    if !cache.chain_prefetch_enabled() {
        return None;
    }
    let cap = crate::page::overflow_payload_capacity() as u64;
    if oref.total_len == 0 || oref.total_len > crate::page::MAX_VALUE_LEN as u64 {
        return None; // the demand path owns the corrupt-length verdict
    }
    let n_links = oref.total_len.div_ceil(cap) as usize;
    if n_links < CHAIN_PREFETCH_MIN_LINKS {
        return None;
    }
    if cache.state_of(oref.head.page_id).is_some() {
        return None;
    }
    let first = oref.head.page_id.as_u64().checked_add(1)?;
    file.locator()
        .predict_overflow_location(PageId(first))
        .map(|_| n_links)
}

/// Polls every pending speculative slot and every draining slot once with the
/// caller's context. Draining slots that complete are removed (dropping the
/// slot releases its permit at **terminal completion**) and counted `unused` —
/// an issued speculative read is *always* driven to terminal completion by
/// this chain's driver (never abandoned to complete unobserved), exactly the
/// drain-all discipline `join_drain` gives demand children.
fn poll_spec(
    cache: &PageCache,
    slots: &mut std::collections::VecDeque<SpecSlot<'_>>,
    drain: &mut Vec<SpecSlot<'_>>,
    cx: &mut std::task::Context<'_>,
) {
    for slot in slots.iter_mut() {
        if let SpecSlotState::Pending(fut) = &mut slot.state {
            if let std::task::Poll::Ready(res) = fut.as_mut().poll(cx) {
                slot.state = SpecSlotState::Done(res);
            }
        }
    }
    drain.retain_mut(|slot| match &mut slot.state {
        SpecSlotState::Pending(fut) => match fut.as_mut().poll(cx) {
            std::task::Poll::Ready(_) => {
                cache
                    .spec()
                    .unused
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                false
            }
            std::task::Poll::Pending => true,
        },
        SpecSlotState::Done(_) => unreachable!("Done slots are discarded before entering drain"),
    });
}

/// Awaits `main` while continuously driving the speculative window and the
/// drain set with the same waker, so speculative device reads are submitted and
/// progressed even while the chain sits in a demand read.
async fn drive_with_spec<'a, T>(
    cache: &PageCache,
    main: impl Future<Output = T>,
    slots: &mut std::collections::VecDeque<SpecSlot<'a>>,
    drain: &mut Vec<SpecSlot<'a>>,
) -> T {
    let mut main = std::pin::pin!(main);
    std::future::poll_fn(|cx| {
        poll_spec(cache, slots, drain, cx);
        main.as_mut().poll(cx)
    })
    .await
}

/// Moves every issued speculative read into the drain set and drives the set to
/// terminal completion. Called on **every** exit from the prefetch driver —
/// success, fallback already drained, demand error, corrupt chain — so no path
/// returns while a speculative read this chain issued is still in flight.
/// Completed-but-unconsumed slots are discarded immediately (counted `unused`;
/// dropping releases their permits). (Caller cancellation drops the whole
/// driver and with it the submitted futures — the documented FakeIo /
/// real-backend divergence; permits release on drop, `unused` is best-effort.)
async fn drain_spec<'a>(
    cache: &PageCache,
    slots: &mut std::collections::VecDeque<SpecSlot<'a>>,
    drain: &mut Vec<SpecSlot<'a>>,
) {
    for slot in slots.drain(..) {
        match slot.state {
            SpecSlotState::Pending(_) => drain.push(slot),
            SpecSlotState::Done(_) => {
                cache
                    .spec()
                    .unused
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    std::future::poll_fn(|cx| {
        poll_spec(cache, slots, drain, cx);
        if drain.is_empty() {
            std::task::Poll::Ready(())
        } else {
            std::task::Poll::Pending
        }
    })
    .await;
}

/// Issues one speculative read for `pos` if a global permit is available. The
/// permit guard lives in the **slot**, not the future — it must outlive the
/// completed private buffer, not just the device read (see [`SpecSlot`]).
fn spawn_spec<'a>(
    cache: &'a PageCache,
    file: &FileGeneration,
    id: PageId,
    pos: usize,
    predicted: crate::backend::Location,
) -> Option<SpecSlot<'a>> {
    let permit = cache.spec_try_permit()?;
    cache
        .spec()
        .issued
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dev = file.device(predicted.member);
    let len = predicted.length as usize;
    let offset = predicted.offset;
    let fut: SpecFuture<'a> = Box::pin(async move {
        let bytes = dev.read_exact_at(offset, len).await.ok()?;
        let arc: Arc<[u8]> = Vec::from(bytes).into();
        let actual = crate::page::Checksum::crc32c(&arc);
        // Reference-independent decode, exactly the cache completer's shape:
        // the checksum gate passes by construction, the *predicted* id is the
        // expected id (never an id read out of untrusted bytes), and only
        // structural validity + embedded-id equality can fail.
        let self_ref = PageRef {
            page_id: id,
            expected_checksum: actual,
        };
        Page::decode_checked(arc, self_ref, actual)
            .ok()
            .map(Arc::new)
    });
    Some(SpecSlot {
        pos,
        predicted,
        _permit: permit,
        state: SpecSlotState::Pending(fut),
    })
}

/// [`read_overflow`], with the id-stride speculative window engaged (T2.2).
///
/// Positions number the links head = 0; ids are predicted as
/// `head.id + pos` (checked) — consecutive by builder/compaction construction
/// on both backends, but treated purely as a *guess*: before a speculative page
/// is consumed at a position, (a) the authoritative ref revealed by the
/// predecessor must locate (full-ref `locate`) to **exactly** the predicted
/// `Location`, (b) the decoded page's full `PageRef` (id and checksum) must
/// equal that ref, and (c) the cache must not already hold or be loading the
/// page (a resident/loading page wins; the private image is discarded as
/// `redundant`). Any failure of (a)/(b) — or a failed speculative read — falls
/// back to plain demand reads for the rest of the chain, drains every issued
/// read, and counts one `fallback`. Speculation is therefore never the source
/// of an error verdict and never a source of bytes that did not pass the same
/// full-ref validation a cache hit passes.
async fn read_overflow_prefetch(
    cache: &PageCache,
    file: &FileGeneration,
    oref: OverflowRef,
    n_links: usize,
) -> Result<Vec<u8>, ReadError> {
    use std::sync::atomic::Ordering::Relaxed;
    cache.spec().engaged_chains.fetch_add(1, Relaxed);
    let head_id = oref.head.page_id.as_u64();
    let locator = Arc::clone(file.locator());
    let mut out: Vec<u8> = Vec::with_capacity(oref.total_len as usize);
    let mut cursor = OverflowCursor::new(oref);
    let mut slots: std::collections::VecDeque<SpecSlot<'_>> = std::collections::VecDeque::new();
    let mut drain: Vec<SpecSlot<'_>> = Vec::new();
    // Next position to predict; `speculating == false` stops replenishment
    // (fallback, capability refusal, id overflow, permit denial, or a
    // redundant hit).
    let mut next_predict: usize = 1;
    let mut speculating = true;
    let mut pos: usize = 0;

    let result: Result<Vec<u8>, ReadError> = 'chain: {
        while let Some(want) = cursor.pending() {
            // Replenish the window up to `min(W, remaining predicted links)`.
            while speculating && slots.len() < CHAIN_PREFETCH_WINDOW && next_predict < n_links {
                let Some(id) = head_id.checked_add(next_predict as u64) else {
                    speculating = false;
                    break;
                };
                let Some(predicted) = locator.predict_overflow_location(PageId(id)) else {
                    speculating = false;
                    break;
                };
                let Some(slot) = spawn_spec(cache, file, PageId(id), next_predict, predicted)
                else {
                    // Permit pool empty: **stop speculating for this chain**
                    // (never wait, never retry). A later retry could issue a
                    // slot for a position the cursor has already passed on
                    // demand, and that stale front slot would block every
                    // later one.
                    // Sustained denial means ≥ pool/W chains are already
                    // speculating; this chain proceeds on demand reads.
                    speculating = false;
                    break;
                };
                slots.push_back(slot);
                next_predict += 1;
            }

            // A speculative slot serves this position only if it exists (the
            // head never speculates, and a pre-denial partial window can end
            // before the chain does).
            let spec_serves = slots.front().map(|s| s.pos == pos).unwrap_or(false);
            let page: Arc<Page> = if !spec_serves {
                match drive_with_spec(cache, cache.get_page(want, file), &mut slots, &mut drain)
                    .await
                {
                    Ok(p) => p,
                    Err(e) => break 'chain Err(e),
                }
            } else {
                // Wait for the front slot while driving the rest of the window.
                std::future::poll_fn(|cx| {
                    poll_spec(cache, &mut slots, &mut drain, cx);
                    match slots.front().map(|s| &s.state) {
                        Some(SpecSlotState::Done(_)) => std::task::Poll::Ready(()),
                        _ => std::task::Poll::Pending,
                    }
                })
                .await;
                let slot = slots.pop_front().expect("front slot just resolved");
                let SpecSlotState::Done(spec_page) = slot.state else {
                    unreachable!("front slot polled to Done above");
                };
                // Authoritative validation gauntlet: failed read/decode,
                // location mismatch, or ref mismatch → fall back;
                // resident/loading → cache wins.
                let confirmed = spec_page.filter(|p| {
                    file.locator().locate(want) == Some(slot.predicted) && p.page_ref() == want
                });
                match confirmed {
                    Some(p) if cache.state_of(want.page_id).is_none() => {
                        cache.spec().consumed.fetch_add(1, Relaxed);
                        p
                    }
                    Some(_resident_wins) => {
                        // The cache already holds (or is loading) this page:
                        // discard the private image, stop replenishing, serve
                        // from the cache. Not a fallback — remaining slots stay
                        // consumable, each behind the same gauntlet.
                        cache.spec().redundant.fetch_add(1, Relaxed);
                        cache.spec().unused.fetch_add(1, Relaxed);
                        speculating = false;
                        match drive_with_spec(
                            cache,
                            cache.get_page(want, file),
                            &mut slots,
                            &mut drain,
                        )
                        .await
                        {
                            Ok(p) => p,
                            Err(e) => break 'chain Err(e),
                        }
                    }
                    None => {
                        // Misprediction or failed speculative read: one
                        // fallback per chain, drain everything issued, finish
                        // on demand reads.
                        cache.spec().unused.fetch_add(1, Relaxed);
                        cache.spec().fallbacks.fetch_add(1, Relaxed);
                        speculating = false;
                        for s in slots.drain(..) {
                            match s.state {
                                SpecSlotState::Pending(_) => drain.push(s),
                                SpecSlotState::Done(_) => {
                                    // Discarded now; dropping releases the
                                    // permit.
                                    cache.spec().unused.fetch_add(1, Relaxed);
                                }
                            }
                        }
                        match drive_with_spec(
                            cache,
                            cache.get_page(want, file),
                            &mut slots,
                            &mut drain,
                        )
                        .await
                        {
                            Ok(p) => p,
                            Err(e) => break 'chain Err(e),
                        }
                    }
                }
            };
            if let Err(e) = cursor.accept(&page) {
                break 'chain Err(e);
            }
            out.extend_from_slice(page.overflow_chunk());
            pos += 1;
        }
        break 'chain Ok(out);
    };
    drain_spec(cache, &mut slots, &mut drain).await;
    result
}

/// Validates an overflow chain **without reassembling it**, recording every link
/// in `seen` (M6 Phase F).
///
/// Two reasons this is not `read_overflow(..).map(drop)`:
///
/// - **Memory.** A verify has no use for the bytes, and materialising them makes
///   a full-store verify hold a value at a time for nothing.
/// - **Work.** Without a *generation-wide* `seen`, a crafted file whose leaves all
///   name the same chain makes verify re-walk it once per entry — a 1 MiB file
///   was measured driving ~408 MiB of reassembly (~26,000×), which turns
///   `verify_tree_on_open` into a denial of service on a large crafted file. The
///   `seen` set bounds verify's work by the file, exactly as it does `LivePages`'.
async fn verify_overflow(
    cache: &PageCache,
    file: &FileGeneration,
    oref: OverflowRef,
    seen: &mut std::collections::HashSet<u64>,
) -> Result<(), ReadError> {
    let mut cursor = OverflowCursor::new(oref);
    while let Some(r) = cursor.pending() {
        if !seen.insert(r.page_id.as_u64()) {
            return Err(ReadError::Corrupt(
                "page reachable twice in one tree (cycle?)",
            ));
        }
        let page = cache.get_page(r, file).await?;
        cursor.accept(&page)?;
    }
    Ok(())
}

/// Point lookup: descend from `root` to the covering leaf. Empty root → `None`.
pub(crate) async fn tree_get(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
    key: &[u8],
) -> Result<Option<Vec<u8>>, ReadError> {
    if root.is_empty() {
        return Ok(None);
    }
    let mut current = root;
    for _ in 0..MAX_TREE_DEPTH {
        let page = cache.get_page(current, file).await?;
        match page.kind() {
            PageKind::Leaf => {
                // Copy the value out of the borrow before any further await: the
                // chain read needs the cache, and holding this page's borrow
                // across it would pin the leaf image for the whole chain.
                let Some(v) = page.leaf_get(key).map(|v| v.to_owned_value()) else {
                    return Ok(None);
                };
                drop(page);
                return Ok(Some(match v {
                    EntryValue::Inline(b) => b,
                    EntryValue::Overflow(o) => read_overflow(cache, file, o).await?,
                }));
            }
            PageKind::Branch => {
                current = page.branch_child_for(key);
                if current.is_empty() {
                    return Err(ReadError::Corrupt("branch descent reached empty ref"));
                }
            }
            PageKind::Overflow => {
                return Err(ReadError::Corrupt("tree descent reached an overflow page"))
            }
        }
    }
    Err(ReadError::Corrupt("tree exceeds max depth (cycle?)"))
}

/// Naïve multi-get: an independent descent per key, results in input order.
pub(crate) async fn tree_get_many_naive(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
    keys: &[Vec<u8>],
) -> Result<Vec<Option<Vec<u8>>>, ReadError> {
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        out.push(tree_get(cache, file, root, key).await?);
    }
    Ok(out)
}

/// Grouped **concurrent** multi-get (T2.1): sort keys, then descend
/// each branch node **once**, routing the whole covered key group into the
/// single child load — and resolve the child groups of a branch *concurrently*
/// (one physical read in flight per distinct child page, via [`join_drain`]).
/// Routing is identical to the naïve form, so the two always agree; results are
/// returned in input order, and duplicate keys each get their own slot.
///
/// Error contract: a failure anywhere first **drains every started sibling to
/// completion** (nothing is abandoned in-flight to the cache's writer-side
/// backstop, which a reader-only surface like [`TreeReader`] does not have),
/// then the deterministic winner is returned — the error whose [`RankedError`]
/// rank (the smallest **original input index** among the keys the failure
/// covers) is minimal, carried from the failure site up through every level.
/// The outcome is a pure function of the inputs, never of I/O completion order
/// or of the sort the traversal happens to use.
pub(crate) async fn tree_get_many_grouped(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
    keys: &[Vec<u8>],
) -> Result<Vec<Option<Vec<u8>>>, ReadError> {
    let mut results = vec![None; keys.len()];
    if root.is_empty() || keys.is_empty() {
        return Ok(results);
    }
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
    match resolve_group(cache, file, root, keys, &order, 0).await {
        Ok(found) => {
            for (i, v) in found {
                results[i] = Some(v);
            }
            Ok(results)
        }
        // The rank exists only to pick the winner; strip it at the boundary.
        Err(ranked) => Err(ranked.err),
    }
}

/// A traversal error tagged with the smallest **original key index** the
/// failure covers: a shared page (branch/leaf) failure covers every key routed
/// through it (rank = the minimum of the group's indices); an overflow-chain
/// failure covers exactly its value's key. [`join_drain`] keeps the error with
/// the minimal rank, so the winner is stable across completion orders and
/// independent of the sorted traversal order — including for scrambled inputs
/// and duplicate keys.
struct RankedError {
    rank: usize,
    err: ReadError,
}

/// The smallest original key index in a (non-empty) group.
fn min_rank(idxs: &[usize]) -> usize {
    *idxs.iter().min().expect("groups are never empty")
}

/// A boxed traversal step whose output feeds [`join_drain`].
type ChildFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, RankedError>> + Send + 'a>>;

/// The found `(original key index, value)` pairs one subtree contributes.
type FoundPairs = Vec<(usize, Vec<u8>)>;

/// Drives `futs` concurrently to completion — **all** of them, even after one
/// fails (drain-all; T2.1) — then returns every value, or the failed child
/// error with the smallest [`RankedError`] rank.
///
/// This is the runtime-neutral "small custom join set": a single `poll_fn` that
/// re-polls every still-pending child with the shared outer waker. A wake meant
/// for one child re-polls its pending siblings too; for a cache load that is a
/// few uncontended lock acquisitions (`drive_load` re-checks `done`, re-parks).
/// The re-poll cost is quadratic in the group size in the worst case; the
/// benchmark's warm large-batch cell measures exactly this driver overhead, and
/// per-child readiness flags are the known fix if it ever shows up there.
///
/// Once any error is known the remaining children are still driven, but their
/// values (and every already-retained value) are discarded immediately — a
/// failing call must not hold the found-lists/overflow buffers of its healthy
/// siblings until the final error return.
///
/// Dropping the aggregate future drops every child at once, exactly like the
/// serial form, so cancellation-safety is unchanged.
async fn join_drain<'a, T: Send>(futs: Vec<ChildFuture<'a, T>>) -> Result<Vec<T>, RankedError> {
    enum ChildState<'a, T> {
        Pending(ChildFuture<'a, T>),
        Done(Option<T>),
    }
    let mut children: Vec<ChildState<'a, T>> = futs.into_iter().map(ChildState::Pending).collect();
    let mut best_err: Option<RankedError> = None;
    std::future::poll_fn(|cx| {
        let mut all_done = true;
        let mut newly_failed = false;
        for child in children.iter_mut() {
            if let ChildState::Pending(fut) = child {
                match fut.as_mut().poll(cx) {
                    std::task::Poll::Ready(Ok(v)) => {
                        // Retain the value only while the whole join can still
                        // succeed.
                        let keep = best_err.is_none();
                        *child = ChildState::Done(keep.then_some(v));
                    }
                    std::task::Poll::Ready(Err(e)) => {
                        // (`map_or`, not `is_none_or`: MSRV 1.75.)
                        if best_err.as_ref().map_or(true, |b| e.rank < b.rank) {
                            best_err = Some(e);
                        }
                        newly_failed = true;
                        *child = ChildState::Done(None);
                    }
                    std::task::Poll::Pending => all_done = false,
                }
            }
        }
        if newly_failed {
            for child in children.iter_mut() {
                if let ChildState::Done(v) = child {
                    *v = None;
                }
            }
        }
        if all_done {
            std::task::Poll::Ready(())
        } else {
            std::task::Poll::Pending
        }
    })
    .await;
    if let Some(e) = best_err {
        return Err(e);
    }
    let mut out = Vec::with_capacity(children.len());
    for child in children {
        match child {
            ChildState::Done(Some(v)) => out.push(v),
            _ => unreachable!("join_drain resolved with a pending or dropped child"),
        }
    }
    Ok(out)
}

/// Recursively resolve a sorted index group at `page_ref`, sharing this page's
/// load across the whole group and fanning its child groups out concurrently.
/// `idxs` are indices into `keys`, sorted by key; the returned pairs are the
/// **found** keys only, as `(index into keys, value)`.
fn resolve_group<'a>(
    cache: &'a PageCache,
    file: &'a FileGeneration,
    page_ref: PageRef,
    keys: &'a [Vec<u8>],
    idxs: &'a [usize],
    depth: u32,
) -> ChildFuture<'a, FoundPairs> {
    Box::pin(async move {
        // A failure of THIS shared page covers every key routed through it.
        let rank = min_rank(idxs);
        let ranked = move |err: ReadError| RankedError { rank, err };
        if depth >= MAX_TREE_DEPTH {
            return Err(ranked(ReadError::Corrupt(
                "tree exceeds max depth (cycle?)",
            )));
        }
        let page = cache.get_page(page_ref, file).await.map_err(ranked)?;
        match page.kind() {
            PageKind::Leaf => {
                // Take owned representations, then drop the leaf before reading
                // any chain — same reason as `tree_get`.
                let found: Vec<(usize, EntryValue)> = idxs
                    .iter()
                    .filter_map(|&i| page.leaf_get(&keys[i]).map(|v| (i, v.to_owned_value())))
                    .collect();
                drop(page);
                // Inline values are ready now; overflow chains resolve
                // concurrently **across result keys** (T2.1). Within one chain
                // links stay serial — each link names its successor. Duplicate
                // keys naming the same chain share each link's load through the
                // cache's single-flight, so concurrency never doubles the I/O.
                let mut out: Vec<(usize, Vec<u8>)> = Vec::with_capacity(found.len());
                let mut chains: Vec<ChildFuture<'a, (usize, Vec<u8>)>> = Vec::new();
                for (i, v) in found {
                    match v {
                        EntryValue::Inline(b) => out.push((i, b)),
                        // A chain failure covers exactly this value's key.
                        EntryValue::Overflow(o) => chains.push(Box::pin(async move {
                            match read_overflow(cache, file, o).await {
                                Ok(v) => Ok((i, v)),
                                Err(err) => Err(RankedError { rank: i, err }),
                            }
                        })),
                    }
                }
                if !chains.is_empty() {
                    out.extend(join_drain(chains).await?);
                }
                Ok(out)
            }
            PageKind::Overflow => Err(ranked(ReadError::Corrupt(
                "tree descent reached an overflow page",
            ))),
            PageKind::Branch => {
                // `idxs` is sorted and `branch_child_for` is monotonic in key, so
                // keys for one child form a contiguous run. Build every child
                // future first, then drop the branch image — the subtree work
                // must not pin it — and drive them concurrently.
                let mut groups: Vec<ChildFuture<'a, FoundPairs>> = Vec::new();
                let mut start = 0;
                while start < idxs.len() {
                    let child = page.branch_child_for(&keys[idxs[start]]);
                    let mut end = start + 1;
                    while end < idxs.len() && page.branch_child_for(&keys[idxs[end]]) == child {
                        end += 1;
                    }
                    if child.is_empty() {
                        // A routing failure covers the keys of ITS run, not the
                        // whole group.
                        return Err(RankedError {
                            rank: min_rank(&idxs[start..end]),
                            err: ReadError::Corrupt("branch descent reached empty ref"),
                        });
                    }
                    groups.push(resolve_group(
                        cache,
                        file,
                        child,
                        keys,
                        &idxs[start..end],
                        depth + 1,
                    ));
                    start = end;
                }
                drop(page);
                Ok(join_drain(groups).await?.into_iter().flatten().collect())
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Range scan (the sorted-map layer's traversal).
// ---------------------------------------------------------------------------

/// An owned `(key, value)` pair, as scans and edge lookups return them.
pub type Entry = (Vec<u8>, Vec<u8>);

/// The result of a range scan.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ScanSummary {
    /// Entries delivered to the consumer.
    pub visited: u64,
    /// `false` iff the consumer ended the scan early with
    /// [`ControlFlow::Break`](std::ops::ControlFlow::Break).
    pub completed: bool,
}

/// Where a leaf key sits relative to the scan's bounds.
enum KeyPos {
    Before,
    In,
    After,
}

fn key_position(start: Bound<&[u8]>, end: Bound<&[u8]>, key: &[u8]) -> KeyPos {
    match start {
        Bound::Included(s) if key < s => return KeyPos::Before,
        Bound::Excluded(s) if key <= s => return KeyPos::Before,
        _ => {}
    }
    match end {
        Bound::Included(e) if key > e => KeyPos::After,
        Bound::Excluded(e) if key >= e => KeyPos::After,
        _ => KeyPos::In,
    }
}

/// The inclusive child-index window of a branch that can hold keys in
/// `[start, end]`. Child `i` holds keys `< sep[i]` and `>= sep[i-1]`
/// (see [`Page::branch_child_for`]). The window may over-include one child at
/// an excluded boundary; the leaf-level filter is exact.
fn branch_child_window(page: &Page, start: Bound<&[u8]>, end: Bound<&[u8]>) -> (usize, usize) {
    let last = page.branch_children().len() - 1;
    let child_for = |key: &[u8]| {
        page.branch_separators()
            .position(|sep| key < sep)
            .unwrap_or(last)
    };
    let lo = match start {
        Bound::Unbounded => 0,
        Bound::Included(k) | Bound::Excluded(k) => child_for(k),
    };
    let hi = match end {
        Bound::Unbounded => last,
        Bound::Included(k) | Bound::Excluded(k) => child_for(k),
    };
    (lo, hi)
}

/// Owns a caller's `RangeBounds` as byte-vector bounds, so a scan future can
/// carry them without borrowing the range expression.
pub(crate) fn own_bounds<K: AsRef<[u8]>, R: std::ops::RangeBounds<K>>(
    range: &R,
) -> (Bound<Vec<u8>>, Bound<Vec<u8>>) {
    fn own<K: AsRef<[u8]>>(b: Bound<&K>) -> Bound<Vec<u8>> {
        match b {
            Bound::Included(k) => Bound::Included(k.as_ref().to_vec()),
            Bound::Excluded(k) => Bound::Excluded(k.as_ref().to_vec()),
            Bound::Unbounded => Bound::Unbounded,
        }
    }
    (own(range.start_bound()), own(range.end_bound()))
}

pub(crate) fn slice_bound(b: &Bound<Vec<u8>>) -> Bound<&[u8]> {
    match b {
        Bound::Included(k) => Bound::Included(k.as_slice()),
        Bound::Excluded(k) => Bound::Excluded(k.as_slice()),
        Bound::Unbounded => Bound::Unbounded,
    }
}

/// A key fence inherited from ancestor separators: `None` = unbounded side.
type Fence = Option<Vec<u8>>;
/// A child about to be visited by a range scan: its ref plus inherited fences.
type PendingChild = (PageRef, Fence, Fence);
/// One frame of the descending scan's stack: the branch page, the next child
/// index (descending; `None` = exhausted), the window's low edge, and the
/// branch's inherited fences.
type RevFrame = (Arc<Page>, Option<usize>, usize, Fence, Fence);

/// The inherited key fence of child `i` of `branch`, given the branch's own
/// inherited fence: child `i` holds keys in `[sep[i-1], sep[i])` (the same
/// rule `verify_node` enforces).
fn child_fences(branch: &Page, i: usize, low: &Fence, high: &Fence) -> (Fence, Fence) {
    let n = branch.branch_children().len();
    let cl = if i == 0 {
        low.clone()
    } else {
        branch.branch_separators().nth(i - 1).map(<[u8]>::to_vec)
    };
    let ch = if i == n - 1 {
        high.clone()
    } else {
        branch.branch_separators().nth(i).map(<[u8]>::to_vec)
    };
    (cl, ch)
}

/// Checks a visited page against its inherited `[low, high)` fence. Decode
/// already guarantees strict internal ascending order (keys and separators),
/// so first/last suffice — this is exactly `verify_node`'s inherited-bound
/// rule, enforced during scans so a structurally misplaced page is `Corrupt`
/// *before* the traversal's early-termination shortcuts can silently skip
/// in-range keys on a corrupt graph.
fn check_fences(page: &Page, low: &Fence, high: &Fence) -> Result<(), ReadError> {
    let (first, last) = match page.kind() {
        PageKind::Leaf => (page.leaf_first_key(), page.leaf_last_key()),
        PageKind::Branch => (
            page.branch_separators().next(),
            page.branch_separators().last(),
        ),
        PageKind::Overflow => (None, None),
    };
    if let (Some(l), Some(f)) = (low, first) {
        if f < l.as_slice() {
            return Err(ReadError::Corrupt("key below subtree low bound"));
        }
    }
    if let (Some(h), Some(la)) = (high, last) {
        if la >= h.as_slice() {
            return Err(ReadError::Corrupt("key at/above subtree high bound"));
        }
    }
    Ok(())
}

/// In-order range scan: delivers every entry whose key lies in
/// `(start, end)`-per-`Bound` semantics to `consumer`, in ascending key order,
/// as borrowed `(key, value)` slices. An overflow value is reassembled from its
/// chain before delivery. The consumer returns
/// [`ControlFlow::Continue`](std::ops::ControlFlow::Continue) to keep going or
/// `Break` to end the scan early (`completed: false` in the summary).
///
/// The descent holds the path's branch pages (≤ [`MAX_TREE_DEPTH`] `Arc`s) for
/// the scan's duration — cache eviction cannot invalidate them (pages are
/// immutable; an `Arc` clone survives eviction), so a scan is consistent with
/// the single root it was given without any sibling links in the format.
pub(crate) async fn tree_scan_range<F>(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
    start: Bound<&[u8]>,
    end: Bound<&[u8]>,
    consumer: &mut F,
) -> Result<ScanSummary, ReadError>
where
    F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
{
    let mut summary = ScanSummary {
        visited: 0,
        completed: true,
    };
    if root.is_empty() {
        return Ok(summary);
    }
    // Explicit descent stack: (branch page, next child index, last child index
    // inside the bound window, inherited low/high fence). Branches are visited
    // left-to-right, so delivery is in ascending key order.
    let mut stack: Vec<(Arc<Page>, usize, usize, Fence, Fence)> = Vec::new();
    let mut pending: Option<PendingChild> = Some((root, None, None));
    // Corruption discipline, three layers, each per visited page and
    // O(1)-amortized:
    // - a scan-wide `seen` id set — a checksum-valid diamond/cycle of *empty*
    //   pages never trips a key-order check, and without this it costs
    //   exponential traversal work;
    // - inherited-fence checks (`check_fences`) — a misplaced page is Corrupt
    //   before the Before/After early-exit shortcuts could silently skip
    //   in-range keys that corruption moved elsewhere;
    // - the cross-leaf watermark below — each leaf's first key must exceed the
    //   previous leaf's last, so no duplicate/regressing key reaches the
    //   consumer.
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut prev_leaf_last: Option<Vec<u8>> = None;
    loop {
        let (page_ref, low, high) = match pending.take() {
            Some(t) => t,
            None => match stack.last_mut() {
                None => return Ok(summary),
                Some((branch, next, last, low, high)) => {
                    if *next > *last {
                        stack.pop();
                        continue;
                    }
                    let i = *next;
                    *next += 1;
                    let r = branch.branch_children()[i];
                    if r.is_empty() {
                        return Err(ReadError::Corrupt("branch descent reached empty ref"));
                    }
                    let (cl, ch) = child_fences(branch, i, low, high);
                    (r, cl, ch)
                }
            },
        };
        if stack.len() as u32 >= MAX_TREE_DEPTH {
            return Err(ReadError::Corrupt("tree exceeds max depth (cycle?)"));
        }
        if !seen.insert(page_ref.page_id.as_u64()) {
            return Err(ReadError::Corrupt(
                "page reachable twice in one tree (cycle?)",
            ));
        }
        let page = cache.get_page(page_ref, file).await?;
        check_fences(&page, &low, &high)?;
        match page.kind() {
            PageKind::Branch => {
                let (lo, hi) = branch_child_window(&page, start, end);
                stack.push((page, lo, hi, low, high));
            }
            PageKind::Leaf => {
                if let (Some(prev), Some(first)) = (&prev_leaf_last, page.leaf_first_key()) {
                    if first <= prev.as_slice() {
                        return Err(ReadError::Corrupt(
                            "leaf keys regress across scan (cycle or shared subtree?)",
                        ));
                    }
                }
                if let Some(last_key) = page.leaf_last_key() {
                    prev_leaf_last = Some(last_key.to_vec());
                }
                for (key, value) in page.leaf_entries() {
                    match key_position(start, end, key) {
                        KeyPos::Before => continue,
                        // Leaf entries and sibling subtrees both ascend, so the
                        // first key past the end bound ends the whole scan.
                        KeyPos::After => return Ok(summary),
                        KeyPos::In => {}
                    }
                    summary.visited += 1;
                    let flow = match value {
                        LeafValue::Inline(b) => consumer(key, b),
                        LeafValue::Overflow(o) => {
                            let assembled = read_overflow(cache, file, o).await?;
                            consumer(key, &assembled)
                        }
                    };
                    if flow.is_break() {
                        summary.completed = false;
                        return Ok(summary);
                    }
                }
            }
            PageKind::Overflow => {
                return Err(ReadError::Corrupt("tree descent reached an overflow page"))
            }
        }
    }
}

/// The descending mirror of [`tree_scan_range`]: delivers every entry whose
/// key lies in the bounds to `consumer` in **descending** key order. Same
/// contract otherwise — borrowed slices, overflow values reassembled before
/// delivery, `Break` ends the scan early with `completed: false`.
///
/// Branches are visited right-to-left and leaf entries iterated in reverse;
/// no format change is required (no sibling links exist to follow — the
/// path stack is the traversal for both directions).
pub(crate) async fn tree_scan_range_rev<F>(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
    start: Bound<&[u8]>,
    end: Bound<&[u8]>,
    consumer: &mut F,
) -> Result<ScanSummary, ReadError>
where
    F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
{
    let mut summary = ScanSummary {
        visited: 0,
        completed: true,
    };
    if root.is_empty() {
        return Ok(summary);
    }
    // Explicit descent stack: (branch page, next child index descending —
    // `None` when the window is exhausted — the window's low edge, and the
    // inherited low/high fence).
    let mut stack: Vec<RevFrame> = Vec::new();
    let mut pending: Option<PendingChild> = Some((root, None, None));
    // Same three-layer corruption discipline as the forward scan (see the
    // comment there): scan-wide `seen` id set, inherited-fence checks, and the
    // mirrored watermark — descending delivery means each leaf's *last* key
    // must fall strictly below the previous leaf's *first*.
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut prev_leaf_first: Option<Vec<u8>> = None;
    loop {
        let (page_ref, low, high) = match pending.take() {
            Some(t) => t,
            None => match stack.last_mut() {
                None => return Ok(summary),
                Some((branch, next, lo, low, high)) => {
                    let Some(i) = *next else {
                        stack.pop();
                        continue;
                    };
                    let r = branch.branch_children()[i];
                    *next = if i > *lo { Some(i - 1) } else { None };
                    if r.is_empty() {
                        return Err(ReadError::Corrupt("branch descent reached empty ref"));
                    }
                    let (cl, ch) = child_fences(branch, i, low, high);
                    (r, cl, ch)
                }
            },
        };
        if stack.len() as u32 >= MAX_TREE_DEPTH {
            return Err(ReadError::Corrupt("tree exceeds max depth (cycle?)"));
        }
        if !seen.insert(page_ref.page_id.as_u64()) {
            return Err(ReadError::Corrupt(
                "page reachable twice in one tree (cycle?)",
            ));
        }
        let page = cache.get_page(page_ref, file).await?;
        check_fences(&page, &low, &high)?;
        match page.kind() {
            PageKind::Branch => {
                let (lo, hi) = branch_child_window(&page, start, end);
                // An inverted bound pair can produce an empty window
                // (`lo > hi`); represent it as already-exhausted instead of
                // visiting child `hi` once.
                let next = if lo <= hi { Some(hi) } else { None };
                stack.push((page, next, lo, low, high));
            }
            PageKind::Leaf => {
                if let (Some(prev), Some(last_key)) = (&prev_leaf_first, page.leaf_last_key()) {
                    if last_key >= prev.as_slice() {
                        return Err(ReadError::Corrupt(
                            "leaf keys regress across scan (cycle or shared subtree?)",
                        ));
                    }
                }
                if let Some(first) = page.leaf_first_key() {
                    prev_leaf_first = Some(first.to_vec());
                }
                for (key, value) in page.leaf_entries().rev() {
                    match key_position(start, end, key) {
                        KeyPos::After => continue,
                        // Descending: sibling subtrees to come hold only
                        // smaller keys, so the first key below the start
                        // bound ends the whole scan.
                        KeyPos::Before => return Ok(summary),
                        KeyPos::In => {}
                    }
                    summary.visited += 1;
                    let flow = match value {
                        LeafValue::Inline(b) => consumer(key, b),
                        LeafValue::Overflow(o) => {
                            let assembled = read_overflow(cache, file, o).await?;
                            consumer(key, &assembled)
                        }
                    };
                    if flow.is_break() {
                        summary.completed = false;
                        return Ok(summary);
                    }
                }
            }
            PageKind::Overflow => {
                return Err(ReadError::Corrupt("tree descent reached an overflow page"))
            }
        }
    }
}

/// The first (`last == false`) or last (`last == true`) entry of the tree, by
/// key order — a single edge descent, no scan.
pub(crate) async fn tree_edge(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
    last: bool,
) -> Result<Option<Entry>, ReadError> {
    if root.is_empty() {
        return Ok(None);
    }
    let mut current = root;
    for _ in 0..MAX_TREE_DEPTH {
        let page = cache.get_page(current, file).await?;
        match page.kind() {
            PageKind::Leaf => {
                let entry = if last {
                    page.leaf_entries().last()
                } else {
                    page.leaf_entries().next()
                };
                let Some((k, v)) = entry.map(|(k, v)| (k.to_vec(), v.to_owned_value())) else {
                    return Ok(None);
                };
                drop(page);
                return Ok(Some((
                    k,
                    match v {
                        EntryValue::Inline(b) => b,
                        EntryValue::Overflow(o) => read_overflow(cache, file, o).await?,
                    },
                )));
            }
            PageKind::Branch => {
                let children = page.branch_children();
                current = if last {
                    *children.last().expect("branch has children")
                } else {
                    children[0]
                };
                if current.is_empty() {
                    return Err(ReadError::Corrupt("branch descent reached empty ref"));
                }
            }
            PageKind::Overflow => {
                return Err(ReadError::Corrupt("tree descent reached an overflow page"))
            }
        }
    }
    Err(ReadError::Corrupt("tree exceeds max depth (cycle?)"))
}

/// A cursor over every page reachable from a root, in a bounded breadth-first
/// walk. Each page is decode-validated by the cache, as on any read path.
///
/// This is the **shared reachability walker**: the checkpoint's relocation set
/// and, later, compaction's dense rebuild are exactly
/// "the pages this yields". Superseded CoW ancestors are unreachable and so are
/// correctly *not* yielded — dropping them is the whole point of a checkpoint.
///
/// It is a **cursor rather than a `Vec`** so a caller can process and drop each
/// page as it goes: returning the whole live set would hold every page image
/// (`PAGE_SIZE` each, pinned against eviction by the strong `Arc`) for the
/// duration, making peak memory O(live data). A store larger than RAM could then
/// never checkpoint — unacceptable for the feature whose entire purpose is
/// running unbounded in space. What it retains is only the frontier
/// of page *refs* and the seen-set of ids, orders of magnitude smaller than the
/// images, and no larger than the manifest the caller must build anyway.
///
/// **Overflow chains are part of the live set** (M6 Phase F): their pages are
/// reachable through overflow-ref leaf entries, and a walker that ignored them
/// would let checkpoint and compaction silently drop every large value.
/// One walker, one place to extend.
/// The greatest entry with key `<= key` (`strict == false`, floor) or
/// `< key` (`strict == true`, lower) — the NavigableMap descent.
///
/// One root-to-leaf descent: at each branch the taken child's left sibling is
/// remembered as the fallback (every key in it is `< sep <= key`, so its
/// maximum is always a valid candidate); if the leaf holds no qualifying key,
/// the answer is the maximum of the deepest such fallback subtree.
pub(crate) async fn tree_floor(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
    key: &[u8],
    strict: bool,
) -> Result<Option<Entry>, ReadError> {
    if root.is_empty() {
        return Ok(None);
    }
    let mut current = root;
    let mut fallback: Option<PageRef> = None;
    for _ in 0..MAX_TREE_DEPTH {
        let page = cache.get_page(current, file).await?;
        match page.kind() {
            PageKind::Branch => {
                let children = page.branch_children();
                let idx = page
                    .branch_separators()
                    .position(|sep| key < sep)
                    .unwrap_or(children.len() - 1);
                if idx > 0 {
                    fallback = Some(children[idx - 1]);
                }
                current = children[idx];
                if current.is_empty() {
                    return Err(ReadError::Corrupt("branch descent reached empty ref"));
                }
            }
            PageKind::Leaf => {
                let mut best: Option<(Vec<u8>, EntryValue)> = None;
                for (k, v) in page.leaf_entries() {
                    let qualifies = if strict { k < key } else { k <= key };
                    if qualifies {
                        best = Some((k.to_vec(), v.to_owned_value()));
                    } else {
                        break;
                    }
                }
                drop(page);
                return match best {
                    Some((k, v)) => Ok(Some((
                        k,
                        match v {
                            EntryValue::Inline(b) => b,
                            EntryValue::Overflow(o) => read_overflow(cache, file, o).await?,
                        },
                    ))),
                    None => match fallback {
                        Some(sub) => tree_edge(cache, file, sub, true).await,
                        None => Ok(None),
                    },
                };
            }
            PageKind::Overflow => {
                return Err(ReadError::Corrupt("tree descent reached an overflow page"))
            }
        }
    }
    Err(ReadError::Corrupt("tree exceeds max depth (cycle?)"))
}

/// The least entry with key `>= key` (`strict == false`, ceiling) or `> key`
/// (`strict == true`, higher) — a bounded scan that stops at its first entry.
pub(crate) async fn tree_ceiling(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
    key: &[u8],
    strict: bool,
) -> Result<Option<Entry>, ReadError> {
    let start = if strict {
        Bound::Excluded(key)
    } else {
        Bound::Included(key)
    };
    let mut hit: Option<Entry> = None;
    tree_scan_range(cache, file, root, start, Bound::Unbounded, &mut |k, v| {
        hit = Some((k.to_vec(), v.to_vec()));
        std::ops::ControlFlow::Break(())
    })
    .await?;
    Ok(hit)
}

pub(crate) struct LivePages {
    /// Breadth-first frontier. Each entry carries how it must be *bounded* —
    /// tree height for tree pages, remaining links for a chain.
    queue: std::collections::VecDeque<(PageRef, Step)>,
    /// Ids already yielded. Load-bearing beyond the cycle guard: a page reachable
    /// twice would be relocated twice, and the duplicate manifest ids would fail
    /// the manifest's own strictly-increasing rule at the next open.
    seen: std::collections::HashSet<u64>,
}

/// How a queued page is bounded.
///
/// Chain length is **not** tree depth, and conflating them is a real bug: a 1 MiB
/// value is a 65-link chain, which would blow a `MAX_TREE_DEPTH` (64) budget that
/// exists to bound *height*. So a chain rides its own [`OverflowCursor`], which
/// both bounds it and validates it to exactly the rule a read enforces — the
/// walker must not admit a structure a reader would reject, or a checkpoint would
/// durably relocate a value that is already dead.
enum Step {
    Tree { depth: u32 },
    Chain(crate::page::OverflowCursor),
}

impl LivePages {
    pub(crate) fn new(root: PageRef) -> Self {
        let mut queue = std::collections::VecDeque::new();
        if !root.is_empty() {
            queue.push_back((root, Step::Tree { depth: 0 }));
        }
        LivePages {
            queue,
            seen: std::collections::HashSet::new(),
        }
    }

    /// The next reachable page, or `None` when the walk is complete. A repeated
    /// page id is a cycle (ids are unique within a generation) → `Corrupt`.
    pub(crate) async fn next(
        &mut self,
        cache: &PageCache,
        file: &FileGeneration,
    ) -> Result<Option<Arc<Page>>, ReadError> {
        let Some((page_ref, step)) = self.queue.pop_front() else {
            return Ok(None);
        };
        if let Step::Tree { depth } = step {
            if depth >= MAX_TREE_DEPTH {
                return Err(ReadError::Corrupt("tree exceeds max depth (cycle?)"));
            }
        }
        if !self.seen.insert(page_ref.page_id.as_u64()) {
            return Err(ReadError::Corrupt(
                "page reachable twice in one tree (cycle?)",
            ));
        }
        let page = cache.get_page(page_ref, file).await?;
        match (page.kind(), step) {
            (PageKind::Branch, Step::Tree { depth }) => {
                for child in page.branch_children() {
                    self.queue
                        .push_back((*child, Step::Tree { depth: depth + 1 }));
                }
            }
            (PageKind::Leaf, Step::Tree { .. }) => {
                for o in page.leaf_overflow_refs() {
                    self.queue
                        .push_back((o.head, Step::Chain(OverflowCursor::new(o))));
                }
            }
            (_, Step::Chain(mut cursor)) => {
                // The cursor is the structural check, not just a counter: it
                // rejects a wrong kind, a short chain, a long chain, and a cycle.
                cursor.accept(&page)?;
                if let Some(next) = cursor.pending() {
                    self.queue.push_back((next, Step::Chain(cursor)));
                }
            }
            (PageKind::Overflow, Step::Tree { .. }) => {
                return Err(ReadError::Corrupt("tree ref resolves to an overflow page"))
            }
        }
        Ok(Some(page))
    }
}

/// Full structural walk: load every reachable page (each is decode-validated),
/// check every key against the subtree key bounds implied by ancestor
/// separators, and count leaf entries. Catches a mis-linked but individually
/// valid page that per-page decode alone would accept.
pub(crate) async fn tree_verify(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
) -> Result<VerifyReport, ReadError> {
    if root.is_empty() {
        return Ok(VerifyReport { entry_count: 0 });
    }
    let mut seen = std::collections::HashSet::new();
    let count = verify_node(cache, file, root, None, None, 0, &mut seen).await?;
    Ok(VerifyReport { entry_count: count })
}

#[allow(clippy::too_many_arguments)]
/// Verifies structure and the entry count recorded with that exact root.
pub(crate) async fn tree_verify_with_count(
    cache: &PageCache,
    file: &FileGeneration,
    root: PageRef,
    expected_count: u64,
) -> Result<VerifyReport, ReadError> {
    let report = tree_verify(cache, file, root).await?;
    if report.entry_count != expected_count {
        return Err(ReadError::Corrupt("root entry count disagrees with tree"));
    }
    Ok(report)
}

fn verify_node<'a>(
    cache: &'a PageCache,
    file: &'a FileGeneration,
    page_ref: PageRef,
    low: Option<Vec<u8>>,
    high: Option<Vec<u8>>,
    depth: u32,
    seen: &'a mut std::collections::HashSet<u64>,
) -> Pin<Box<dyn Future<Output = Result<u64, ReadError>> + Send + 'a>> {
    Box::pin(async move {
        if depth >= MAX_TREE_DEPTH {
            return Err(ReadError::Corrupt("tree exceeds max depth (cycle?)"));
        }
        // The same generation-wide reachable-twice rule `LivePages` enforces. Verify
        // must be **at least as strict as the walkers**: it is the tool an operator
        // runs to decide a store is healthy, and Direct's `verify_tree_on_open` is
        // the gate that admits writes. Without this, a shared subtree or chain
        // verifies clean and then poisons the store at its first checkpoint or
        // compaction, with no mutation in between.
        if !seen.insert(page_ref.page_id.as_u64()) {
            return Err(ReadError::Corrupt(
                "page reachable twice in one tree (cycle?)",
            ));
        }
        let page = cache.get_page(page_ref, file).await?;
        match page.kind() {
            PageKind::Leaf => {
                let mut n = 0u64;
                let mut chains: Vec<OverflowRef> = Vec::new();
                for (k, v) in page.leaf_entries() {
                    if let Some(l) = &low {
                        if k < l.as_slice() {
                            return Err(ReadError::Corrupt("leaf key below subtree low bound"));
                        }
                    }
                    if let Some(h) = &high {
                        if k >= h.as_slice() {
                            return Err(ReadError::Corrupt("leaf key at/above subtree high bound"));
                        }
                    }
                    if let LeafValue::Overflow(o) = v {
                        chains.push(o);
                    }
                    n += 1;
                }
                drop(page);
                // Verify walks *everything* reachable, so an overflow value's
                // chain is walked and length-checked too — a chain that is short,
                // long, cyclic, mis-kinded, or shared with another entry is exactly
                // the "individually valid pages, structurally wrong" defect this
                // function exists to catch.
                for o in chains {
                    verify_overflow(cache, file, o, seen).await?;
                }
                Ok(n)
            }
            PageKind::Overflow => Err(ReadError::Corrupt("tree ref resolves to an overflow page")),
            PageKind::Branch => {
                let children = page.branch_children().to_vec();
                let seps: Vec<Vec<u8>> = page.branch_separators().map(<[u8]>::to_vec).collect();
                // Separators must sit within the inherited (low, high) window.
                for sep in &seps {
                    if let Some(l) = &low {
                        if sep < l {
                            return Err(ReadError::Corrupt("separator below subtree low bound"));
                        }
                    }
                    if let Some(h) = &high {
                        if sep >= h {
                            return Err(ReadError::Corrupt(
                                "separator at/above subtree high bound",
                            ));
                        }
                    }
                }
                let mut total = 0u64;
                for i in 0..children.len() {
                    let child_low = if i == 0 {
                        low.clone()
                    } else {
                        Some(seps[i - 1].clone())
                    };
                    let child_high = if i == children.len() - 1 {
                        high.clone()
                    } else {
                        Some(seps[i].clone())
                    };
                    total += verify_node(
                        cache,
                        file,
                        children[i],
                        child_low,
                        child_high,
                        depth + 1,
                        seen,
                    )
                    .await?;
                }
                Ok(total)
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Read-only bulk-load builder fixture.
// ---------------------------------------------------------------------------

/// Builds an immutable multi-level B+tree onto a device by bottom-up bulk load.
///
/// `max_fanout`/`max_leaf_entries` are test knobs to force multi-level trees
/// with few entries; `0` means "page-capacity limited only".
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeBuilder {
    pub max_fanout: usize,
    pub max_leaf_entries: usize,
    /// First page id to assign (`0` → `1`). **Page ids are globally unique and
    /// never reused** (the cache is keyed on that invariant), so a later
    /// generation built for a `TreeReader::cutover` that shares one cache MUST
    /// start past the previous generation's ids — otherwise two different images
    /// collide on one id. Give each generation a disjoint id range.
    pub first_page_id: u64,
}

/// A built tree: its device+locator generation, root reference, and entry count.
pub struct BuiltTree {
    pub file: FileGeneration,
    pub root: PageRef,
    pub entry_count: u64,
}

impl TreeBuilder {
    /// Serializes `entries` (must be strictly ascending, unique keys) as a
    /// read-only tree on `dev`, one page per `PAGE_SIZE` slot. Returns the
    /// generation to read it back through.
    pub fn build(
        &self,
        dev: Arc<dyn PageIo>,
        entries: &[(Vec<u8>, Vec<u8>)],
    ) -> std::io::Result<BuiltTree> {
        for w in entries.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "builder entries must be strictly ascending"
            );
        }
        let locator = DataFileLocator::new();
        let mut next_id = self.first_page_id.max(1);
        // A fanout of 1 would never shrink a level → non-terminating build; a
        // real branch needs ≥ 2 children. `0` means page-capacity-limited only.
        let fanout_cap = if self.max_fanout == 0 {
            usize::MAX
        } else {
            self.max_fanout.max(2)
        };
        // Physical offsets are assigned **sequentially**, independent of page id
        // (offset = slot index × PAGE_SIZE). Tying offset to id would make a
        // disjoint high `first_page_id` demand a multi-terabyte file.
        // The locator records the id→offset map.
        let mut pages: Vec<(u64, Page)> = Vec::new();
        let slot_offset = |slot: usize| slot as u64 * crate::page::PAGE_SIZE as u64;

        if entries.is_empty() {
            return Ok(BuiltTree {
                file: FileGeneration::direct(dev, Arc::new(locator)),
                root: PageRef::EMPTY,
                entry_count: 0,
            });
        }

        // Spill oversized values to overflow chains up front (M6 Phase F), so the
        // leaf packer below sees fixed-size refs and its capacity math is
        // untouched. Chains are emitted **tail-first**: each link embeds its
        // successor's id *and* checksum.
        let mut prepared: Vec<(Vec<u8>, EntryValue)> = Vec::with_capacity(entries.len());
        for (k, v) in entries {
            let value = if v.len() > MAX_INLINE_VALUE_LEN {
                // Ids head-first, *encode* tail-first — see `btree::emit_overflow`.
                // The encode order is forced (a link embeds its successor's id and
                // checksum); the id order is not, and must not follow it, or the
                // value lands backwards wherever offset derives from id.
                let n = overflow_chunks(v).count();
                let first_id = next_id;
                next_id += n as u64;
                let mut built: Vec<(u64, Page)> = Vec::with_capacity(n);
                let mut next: Option<PageRef> = None;
                for (i, chunk) in overflow_chunks(v).enumerate().rev() {
                    let id = first_id + i as u64;
                    let page = encode_overflow(PageId(id), 0, chunk, next);
                    next = Some(page.page_ref());
                    built.push((id, page));
                }
                for (id, page) in built.into_iter().rev() {
                    let off = slot_offset(pages.len());
                    locator.record(id, off);
                    pages.push((off, page));
                }
                EntryValue::Overflow(OverflowRef {
                    head: next.expect("a spilled value has at least one chunk"),
                    total_len: v.len() as u64,
                })
            } else {
                EntryValue::Inline(v.clone())
            };
            prepared.push((k.clone(), value));
        }

        // Level 0: pack entries into leaves.
        let mut level: Vec<(Vec<u8>, PageRef)> = Vec::new();
        let mut i = 0;
        while i < prepared.len() {
            let mut chunk: Vec<(Vec<u8>, EntryValue)> = Vec::new();
            let mut used = 0usize;
            while i < prepared.len() {
                let (k, v) = &prepared[i];
                let add = leaf_entry_overhead(k.len(), v.stored_len());
                let over_count = self.max_leaf_entries > 0 && chunk.len() >= self.max_leaf_entries;
                if !chunk.is_empty() && (over_count || used + add > body_capacity()) {
                    break;
                }
                used += add;
                chunk.push((k.clone(), v.clone()));
                i += 1;
            }
            let id = next_id;
            next_id += 1;
            let first = chunk[0].0.clone();
            let page = encode_leaf(PageId(id), 0, &chunk);
            let r = page.page_ref();
            let off = slot_offset(pages.len());
            locator.record(id, off);
            pages.push((off, page));
            level.push((first, r));
        }

        // Build branch levels until a single root remains.
        while level.len() > 1 {
            let mut next_level: Vec<(Vec<u8>, PageRef)> = Vec::new();
            let mut j = 0;
            while j < level.len() {
                let mut children: Vec<PageRef> = Vec::new();
                let mut seps: Vec<Vec<u8>> = Vec::new();
                let mut sep_bytes = 0usize;
                let first_key = level[j].0.clone();
                while j < level.len() {
                    let (fk, r) = &level[j];
                    if children.is_empty() {
                        children.push(*r);
                        j += 1;
                        continue;
                    }
                    let over_fanout = children.len() >= fanout_cap;
                    let projected =
                        (children.len() + 1) * 16 + (seps.len() + 1) * 8 + sep_bytes + fk.len();
                    if over_fanout || projected > body_capacity() {
                        break;
                    }
                    sep_bytes += fk.len();
                    seps.push(fk.clone());
                    children.push(*r);
                    j += 1;
                }
                let id = next_id;
                next_id += 1;
                let page = encode_branch(PageId(id), 0, &children, &seps);
                let r = page.page_ref();
                let off = slot_offset(pages.len());
                locator.record(id, off);
                pages.push((off, page));
                next_level.push((first_key, r));
            }
            level = next_level;
        }
        let root = level[0].1;

        // Flush all pages durably.
        block_on(async {
            for (off, page) in &pages {
                let bytes = page.bytes().to_vec().into_boxed_slice();
                dev.write_all_at(*off, bytes).await.result?;
            }
            dev.datasync().await
        })?;

        Ok(BuiltTree {
            file: FileGeneration::direct(dev, Arc::new(locator)),
            root,
            entry_count: entries.len() as u64,
        })
    }
}

// ---------------------------------------------------------------------------
// TreeReader: a read-only handle over a built tree.
// ---------------------------------------------------------------------------

struct TreeState {
    root: PageRef,
    file: Arc<FileGeneration>,
    entry_count: u64,
}

/// A read-only handle over a real page tree, backed by a shared [`PageCache`].
/// Exposes the M2 read surface (`get`/`get_many`/`snapshot`/`len`/`verify`/
/// `metrics`) decoupled from the writer actor, so read tests never race writes.
pub struct TreeReader {
    cache: Arc<PageCache>,
    state: ArcSwap<TreeState>,
}

impl TreeReader {
    /// Wraps a freshly built tree with a new cache.
    pub fn new(built: BuiltTree) -> Self {
        TreeReader {
            cache: Arc::new(PageCache::new()),
            state: ArcSwap::from_pointee(TreeState {
                root: built.root,
                file: Arc::new(built.file),
                entry_count: built.entry_count,
            }),
        }
    }

    /// Wraps a built tree with a caller-provided (shared) cache.
    pub fn with_cache(built: BuiltTree, cache: Arc<PageCache>) -> Self {
        TreeReader {
            cache,
            state: ArcSwap::from_pointee(TreeState {
                root: built.root,
                file: Arc::new(built.file),
                entry_count: built.entry_count,
            }),
        }
    }

    /// Simulates a cutover: atomically swap the (root, file) generation. A
    /// snapshot taken before this keeps reading the prior generation.
    pub fn cutover(&self, built: BuiltTree) {
        self.state.store(Arc::new(TreeState {
            root: built.root,
            file: Arc::new(built.file),
            entry_count: built.entry_count,
        }));
    }

    /// Point read against the current generation.
    pub fn get(
        &self,
        key: Vec<u8>,
    ) -> impl Future<Output = Result<Option<Vec<u8>>, ReadError>> + Send {
        let cache = Arc::clone(&self.cache);
        let state = self.state.load_full();
        async move { tree_get(&cache, &state.file, state.root, &key).await }
    }

    /// Correct-order multi-get, always the **naïve** independent-descent form —
    /// this fixture handle stays route-free so benchmarks and tests can compare
    /// the two traversals directly. [`get_many_grouped`] is the concurrent
    /// shared-descent form; the production `BTreeMap::get_many` routes between
    /// them by the T2.1 batch-size threshold.
    ///
    /// [`get_many_grouped`]: Self::get_many_grouped
    pub fn get_many(
        &self,
        keys: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<Vec<Option<Vec<u8>>>, ReadError>> + Send {
        let cache = Arc::clone(&self.cache);
        let state = self.state.load_full();
        async move { tree_get_many_naive(&cache, &state.file, state.root, &keys).await }
    }

    /// Grouped shared-descent multi-get with **concurrent child loads** (T2.1;
    /// the benchmark-decided form). Sorts keys, descends each
    /// branch once, and drives sibling child groups' physical reads
    /// concurrently; routing is identical to `get_many`.
    pub fn get_many_grouped(
        &self,
        keys: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<Vec<Option<Vec<u8>>>, ReadError>> + Send {
        let cache = Arc::clone(&self.cache);
        let state = self.state.load_full();
        async move { tree_get_many_grouped(&cache, &state.file, state.root, &keys).await }
    }

    /// In-order range scan of the current generation through a consumer
    /// callback — see [`tree_scan_range`] for the delivery contract.
    pub fn for_each_range<K, R, F>(
        &self,
        range: R,
        mut consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: std::ops::RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
    {
        let cache = Arc::clone(&self.cache);
        let state = self.state.load_full();
        let (start, end) = own_bounds(&range);
        async move {
            tree_scan_range(
                &cache,
                &state.file,
                state.root,
                slice_bound(&start),
                slice_bound(&end),
                &mut consumer,
            )
            .await
        }
    }

    /// Descending range scan of the current generation — see
    /// [`tree_scan_range_rev`] for the delivery contract.
    pub fn for_each_range_rev<K, R, F>(
        &self,
        range: R,
        mut consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: std::ops::RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
    {
        let cache = Arc::clone(&self.cache);
        let state = self.state.load_full();
        let (start, end) = own_bounds(&range);
        async move {
            tree_scan_range_rev(
                &cache,
                &state.file,
                state.root,
                slice_bound(&start),
                slice_bound(&end),
                &mut consumer,
            )
            .await
        }
    }

    /// Verifies structure and the entry count of the captured generation.
    pub fn verify(&self) -> impl Future<Output = Result<VerifyReport, ReadError>> + Send {
        let cache = Arc::clone(&self.cache);
        let state = self.state.load_full();
        async move { tree_verify_with_count(&cache, &state.file, state.root, state.entry_count).await }
    }

    /// Entries in the current generation.
    pub fn len(&self) -> u64 {
        self.state.load().entry_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A consistent snapshot pinning the current (root, file) generation.
    pub fn snapshot(&self) -> TreeSnapshot {
        TreeSnapshot {
            cache: Arc::clone(&self.cache),
            state: self.state.load_full(),
        }
    }

    /// Cache counters.
    pub fn metrics(&self) -> CacheMetrics {
        self.cache.metrics()
    }
}

/// A pinned read-only view captured from a [`TreeReader`]. Survives cutovers.
pub struct TreeSnapshot {
    cache: Arc<PageCache>,
    state: Arc<TreeState>,
}

impl TreeSnapshot {
    pub fn get(
        &self,
        key: Vec<u8>,
    ) -> impl Future<Output = Result<Option<Vec<u8>>, ReadError>> + Send {
        let cache = Arc::clone(&self.cache);
        let state = Arc::clone(&self.state);
        async move { tree_get(&cache, &state.file, state.root, &key).await }
    }

    /// In-order range scan of the pinned generation — see [`tree_scan_range`].
    pub fn for_each_range<K, R, F>(
        &self,
        range: R,
        mut consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: std::ops::RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
    {
        let cache = Arc::clone(&self.cache);
        let state = Arc::clone(&self.state);
        let (start, end) = own_bounds(&range);
        async move {
            tree_scan_range(
                &cache,
                &state.file,
                state.root,
                slice_bound(&start),
                slice_bound(&end),
                &mut consumer,
            )
            .await
        }
    }

    /// Descending range scan of the pinned generation — see
    /// [`tree_scan_range_rev`].
    pub fn for_each_range_rev<K, R, F>(
        &self,
        range: R,
        mut consumer: F,
    ) -> impl Future<Output = Result<ScanSummary, ReadError>> + Send
    where
        K: AsRef<[u8]>,
        R: std::ops::RangeBounds<K>,
        F: FnMut(&[u8], &[u8]) -> std::ops::ControlFlow<()> + Send,
    {
        let cache = Arc::clone(&self.cache);
        let state = Arc::clone(&self.state);
        let (start, end) = own_bounds(&range);
        async move {
            tree_scan_range_rev(
                &cache,
                &state.file,
                state.root,
                slice_bound(&start),
                slice_bound(&end),
                &mut consumer,
            )
            .await
        }
    }

    pub fn len(&self) -> u64 {
        self.state.entry_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::FakeIo;
    use futures_lite::future::block_on;
    use std::collections::BTreeMap as StdMap;

    fn entries(n: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
        (0..n)
            .map(|i| {
                (
                    format!("key{i:06}").into_bytes(),
                    format!("value-{i}").into_bytes(),
                )
            })
            .collect()
    }

    /// A small-fanout builder that forces a multi-level tree for a modest count.
    fn multilevel() -> TreeBuilder {
        TreeBuilder {
            max_fanout: 4,
            max_leaf_entries: 8,
            ..Default::default()
        }
    }

    #[test]
    fn public_verify_rejects_descriptor_count_mismatch() {
        for count in [0, 1, 999, u64::MAX] {
            let mut built = multilevel()
                .build(Arc::new(FakeIo::new(0)), &entries(5))
                .unwrap();
            built.entry_count = count;
            let reader = TreeReader::new(built);
            assert!(matches!(
                block_on(reader.verify()),
                Err(ReadError::Corrupt(_))
            ));
        }
        let mut empty = multilevel().build(Arc::new(FakeIo::new(0)), &[]).unwrap();
        empty.entry_count = 1;
        assert!(matches!(
            block_on(TreeReader::new(empty).verify()),
            Err(ReadError::Corrupt(_))
        ));
    }

    #[test]
    fn public_verify_uses_count_from_captured_generation() {
        let built = multilevel()
            .build(Arc::new(FakeIo::new(0)), &entries(5))
            .unwrap();
        let reader = TreeReader::new(built);
        let verify = reader.verify();
        let next_builder = TreeBuilder {
            first_page_id: 1000,
            ..multilevel()
        };
        reader.cutover(
            next_builder
                .build(Arc::new(FakeIo::new(0)), &entries(9))
                .unwrap(),
        );
        assert_eq!(block_on(verify).unwrap().entry_count, 5);
        assert_eq!(block_on(reader.verify()).unwrap().entry_count, 9);
    }

    #[test]
    fn multilevel_roundtrip_and_verify() {
        let data = entries(500);
        let dev = Arc::new(FakeIo::new(0));
        let built = multilevel().build(dev, &data).unwrap();
        assert_eq!(built.entry_count, 500);
        let reader = TreeReader::new(built);

        block_on(async {
            for (k, v) in &data {
                assert_eq!(
                    reader.get(k.clone()).await.unwrap().as_deref(),
                    Some(v.as_slice())
                );
            }
            assert_eq!(reader.get(b"nope".to_vec()).await.unwrap(), None);
            assert_eq!(reader.get(b"key000000zzz".to_vec()).await.unwrap(), None);
            let report = reader.verify().await.unwrap();
            assert_eq!(report.entry_count, 500);
        });
        assert_eq!(reader.len(), 500);
        // The tree was genuinely multi-level: many pages resident after verify.
        assert!(
            reader.metrics().loads_issued > 10,
            "expected a multi-page tree"
        );
    }

    #[test]
    fn naive_and_grouped_get_many_agree() {
        let data = entries(300);
        let expected: StdMap<Vec<u8>, Vec<u8>> = data.iter().cloned().collect();
        let dev = Arc::new(FakeIo::new(0));
        let built = multilevel().build(dev, &data).unwrap();
        let reader = TreeReader::new(built);

        // Mix present and absent keys in a scrambled order.
        let mut query: Vec<Vec<u8>> = Vec::new();
        for i in (0..300).rev().step_by(3) {
            query.push(format!("key{i:06}").into_bytes());
        }
        query.push(b"absent-a".to_vec());
        query.push(b"key000001".to_vec());
        query.push(b"zzz".to_vec());

        block_on(async {
            let naive = reader.get_many(query.clone()).await.unwrap();
            let grouped = reader.get_many_grouped(query.clone()).await.unwrap();
            assert_eq!(naive, grouped, "naive and grouped multi-get must agree");
            // And both match the ground truth.
            let truth: Vec<Option<Vec<u8>>> =
                query.iter().map(|k| expected.get(k).cloned()).collect();
            assert_eq!(grouped, truth);
        });
    }

    #[test]
    fn snapshot_survives_simulated_cutover() {
        let data_a = entries(120);
        let dev_a = Arc::new(FakeIo::new(0));
        let built_a = multilevel().build(dev_a, &data_a).unwrap();
        let reader = TreeReader::new(built_a);
        let snap = reader.snapshot();
        assert_eq!(snap.len(), 120);

        // Cut over to a different generation. The cache is shared across the
        // cutover, so generation B MUST use a disjoint page-id range (page ids
        // are globally unique / never reused). With overlapping ids the two
        // generations' different images would collide in the cache and a
        // snapshot read of generation A would falsely observe generation B's
        // checksum → `Corrupt`. `first_page_id` prevents that.
        let data_b = entries(60);
        let dev_b = Arc::new(FakeIo::new(0));
        let built_b = TreeBuilder {
            max_fanout: 4,
            max_leaf_entries: 8,
            first_page_id: 1_000_000,
        }
        .build(dev_b, &data_b)
        .unwrap();
        reader.cutover(built_b);

        block_on(async {
            // Reader sees generation B.
            assert_eq!(reader.len(), 60);
            assert_eq!(reader.get(b"key000100".to_vec()).await.unwrap(), None); // absent in B (60)
            assert_eq!(
                reader.get(b"key000059".to_vec()).await.unwrap().as_deref(),
                Some(&b"value-59"[..])
            );
            // Snapshot still sees generation A. Probe EVERY generation-A key —
            // including ones whose page ids would have collided with B under the
            // old id-reuse bug — and require the correct value, never `Corrupt`.
            assert_eq!(snap.len(), 120);
            for (k, v) in &data_a {
                assert_eq!(
                    snap.get(k.clone()).await.unwrap().as_deref(),
                    Some(v.as_slice()),
                    "snapshot lost generation-A key under shared-cache cutover"
                );
            }
        });
    }

    #[test]
    fn empty_tree_reads_none() {
        let dev = Arc::new(FakeIo::new(0));
        let built = TreeBuilder::default().build(dev, &[]).unwrap();
        assert!(built.root.is_empty());
        let reader = TreeReader::new(built);
        block_on(async {
            assert_eq!(reader.get(b"any".to_vec()).await.unwrap(), None);
            assert_eq!(reader.verify().await.unwrap().entry_count, 0);
            let many = reader
                .get_many(vec![b"a".to_vec(), b"b".to_vec()])
                .await
                .unwrap();
            assert_eq!(many, vec![None, None]);
        });
        assert!(reader.is_empty());
    }

    #[test]
    fn verify_rejects_mislinked_but_valid_pages() {
        use crate::backend::{DataFileLocator, FileGeneration};
        use crate::cache::PageCache;
        use crate::page::{encode_branch, encode_leaf_inline, PageId, PAGE_SIZE};

        // Two individually-valid leaves whose key ranges violate the branch's
        // separator: left leaf holds "m".."z" (above the separator "d"), right
        // holds "a".."b" (below it). Each page decodes fine on its own; only a
        // structural walk that checks subtree key bounds catches the swap.
        let left = encode_leaf_inline(
            PageId(1),
            0,
            &[
                (b"m".to_vec(), b"1".to_vec()),
                (b"z".to_vec(), b"2".to_vec()),
            ],
        );
        let right = encode_leaf_inline(
            PageId(2),
            0,
            &[
                (b"a".to_vec(), b"3".to_vec()),
                (b"b".to_vec(), b"4".to_vec()),
            ],
        );
        let root = encode_branch(
            PageId(3),
            0,
            &[left.page_ref(), right.page_ref()],
            &[b"d".to_vec()],
        );

        let dev = Arc::new(FakeIo::new(0));
        let loc = DataFileLocator::new();
        block_on(async {
            for (i, page) in [&left, &right, &root].iter().enumerate() {
                let off = (i as u64 + 1) * PAGE_SIZE as u64;
                loc.record(page.id().as_u64(), off);
                dev.write_all_at(off, page.bytes().to_vec().into_boxed_slice())
                    .await
                    .result
                    .unwrap();
            }
            dev.datasync().await.unwrap();
            let file = FileGeneration::direct(dev.clone() as Arc<dyn PageIo>, Arc::new(loc));
            let cache = PageCache::new();
            let err = tree_verify(&cache, &file, root.page_ref())
                .await
                .unwrap_err();
            assert!(
                matches!(err, ReadError::Corrupt(_)),
                "mis-linked subtree must verify as Corrupt, got {err:?}"
            );
        });
    }

    /// A chain named by two entries is corrupt, and **every** consumer must agree.
    ///
    /// The two consumers once disagreed on exactly this: `tree_verify`
    /// accepted the shared chain while `LivePages` rejected it, so a store could
    /// pass Direct's `verify_tree_on_open` — the gate that admits writes — and then
    /// poison at its first checkpoint or compaction with no mutation in between.
    /// Verify is the tool used to decide a store is healthy; it must be at least as
    /// strict as the walkers, never more permissive.
    #[test]
    fn a_chain_shared_by_two_entries_is_corrupt_to_every_walker() {
        use crate::backend::{DataFileLocator, FileGeneration};
        use crate::cache::PageCache;
        use crate::page::{
            encode_leaf, encode_overflow, EntryValue, OverflowRef, PageId, PAGE_SIZE,
        };

        let chain = encode_overflow(PageId(1), 0, b"payload", None);
        let oref = OverflowRef {
            head: chain.page_ref(),
            total_len: 7,
        };
        let root = encode_leaf(
            PageId(2),
            0,
            &[
                (b"a".to_vec(), EntryValue::Overflow(oref)),
                (b"b".to_vec(), EntryValue::Overflow(oref)),
            ],
        );
        let dev = Arc::new(FakeIo::new(0));
        let loc = DataFileLocator::new();
        block_on(async {
            for (i, page) in [&chain, &root].iter().enumerate() {
                let off = (i as u64 + 1) * PAGE_SIZE as u64;
                loc.record(page.id().as_u64(), off);
                dev.write_all_at(off, page.bytes().to_vec().into_boxed_slice())
                    .await
                    .result
                    .unwrap();
            }
            let file = FileGeneration::direct(dev.clone() as Arc<dyn PageIo>, Arc::new(loc));
            let cache = PageCache::new();
            // The live-set walker rejects it: relocating one page under two refs
            // would duplicate it and break the manifest's unique-id rule.
            let mut walk = LivePages::new(root.page_ref());
            assert!(walk.next(&cache, &file).await.unwrap().is_some()); // the leaf
            assert!(walk.next(&cache, &file).await.unwrap().is_some()); // the chain, once
            assert!(
                matches!(walk.next(&cache, &file).await, Err(ReadError::Corrupt(_))),
                "the second reference to one chain page must be Corrupt"
            );
            // And so must verify — the point of the fix.
            assert!(
                matches!(
                    tree_verify(&cache, &file, root.page_ref()).await,
                    Err(ReadError::Corrupt(
                        "page reachable twice in one tree (cycle?)"
                    ))
                ),
                "verify must not call healthy a store the next checkpoint poisons"
            );
        });
    }

    /// Diamond-DAG corpus, case 1 (Phase-H debt): a
    /// checksum-valid branch naming the same child in two slots. Per-page
    /// decode deliberately tolerates duplicate child refs (it validates one
    /// page at a time); the generation-wide `seen` set is the only guard, and
    /// every walker must trip it — otherwise verify blesses a store whose
    /// next compaction would poison (or duplicate the subtree).
    #[test]
    fn a_branch_naming_one_child_twice_is_corrupt_to_every_walker() {
        use crate::backend::{DataFileLocator, FileGeneration};
        use crate::cache::PageCache;
        use crate::page::{encode_branch, encode_leaf_inline, PageId, PAGE_SIZE};

        let leaf = encode_leaf_inline(
            PageId(1),
            0,
            &[
                (b"a".to_vec(), b"1".to_vec()),
                (b"b".to_vec(), b"2".to_vec()),
            ],
        );
        let root = encode_branch(
            PageId(2),
            0,
            &[leaf.page_ref(), leaf.page_ref()],
            &[b"c".to_vec()],
        );

        let dev = Arc::new(FakeIo::new(0));
        let loc = DataFileLocator::new();
        block_on(async {
            for (i, page) in [&leaf, &root].iter().enumerate() {
                let off = (i as u64 + 1) * PAGE_SIZE as u64;
                loc.record(page.id().as_u64(), off);
                dev.write_all_at(off, page.bytes().to_vec().into_boxed_slice())
                    .await
                    .result
                    .unwrap();
            }
            let file = FileGeneration::direct(dev.clone() as Arc<dyn PageIo>, Arc::new(loc));
            let cache = PageCache::new();

            let mut walk = LivePages::new(root.page_ref());
            assert!(walk.next(&cache, &file).await.unwrap().is_some()); // the branch
            assert!(walk.next(&cache, &file).await.unwrap().is_some()); // the leaf, once
            assert!(
                matches!(
                    walk.next(&cache, &file).await,
                    Err(ReadError::Corrupt(
                        "page reachable twice in one tree (cycle?)"
                    ))
                ),
                "LivePages must reject the second slot naming the same child"
            );
            assert!(
                matches!(
                    tree_verify(&cache, &file, root.page_ref()).await,
                    Err(ReadError::Corrupt(
                        "page reachable twice in one tree (cycle?)"
                    ))
                ),
                "verify must be at least as strict as the walkers"
            );
        });
    }

    /// The descending scan carries the same anti-amplification discipline as
    /// the ascending one: a branch naming one leaf twice is rejected before
    /// any delivery — the duplicated leaf is *misplaced* relative to the
    /// separator, so the inherited-fence check fires on the very first visit
    /// (the seen set and watermark back it up for structures fences pass).
    #[test]
    fn a_rev_scan_rejects_a_shared_subtree() {
        use crate::backend::{DataFileLocator, FileGeneration};
        use crate::cache::PageCache;
        use crate::page::{encode_branch, encode_leaf_inline, PageId, PAGE_SIZE};

        let leaf = encode_leaf_inline(
            PageId(1),
            0,
            &[
                (b"a".to_vec(), b"1".to_vec()),
                (b"b".to_vec(), b"2".to_vec()),
            ],
        );
        let root = encode_branch(
            PageId(2),
            0,
            &[leaf.page_ref(), leaf.page_ref()],
            &[b"c".to_vec()],
        );
        let dev = Arc::new(FakeIo::new(0));
        let loc = DataFileLocator::new();
        block_on(async {
            for (i, page) in [&leaf, &root].iter().enumerate() {
                let off = (i as u64 + 1) * PAGE_SIZE as u64;
                loc.record(page.id().as_u64(), off);
                dev.write_all_at(off, page.bytes().to_vec().into_boxed_slice())
                    .await
                    .result
                    .unwrap();
            }
            let file = FileGeneration::direct(dev.clone() as Arc<dyn PageIo>, Arc::new(loc));
            let cache = PageCache::new();
            let mut delivered = 0usize;
            let out = tree_scan_range_rev(
                &cache,
                &file,
                root.page_ref(),
                Bound::Unbounded,
                Bound::Unbounded,
                &mut |_k, _v| {
                    delivered += 1;
                    std::ops::ControlFlow::Continue(())
                },
            )
            .await;
            // Reverse order visits the right slot first; its inherited low
            // fence is the separator "c", and the shared leaf's first key "a"
            // sits below it — Corrupt before a single delivery.
            assert!(
                matches!(out, Err(ReadError::Corrupt("key below subtree low bound"))),
                "got {out:?}"
            );
            assert_eq!(delivered, 0, "no entry of a misplaced page may deliver");
        });
    }

    /// A checksum-valid DAG of **empty** pages
    /// never trips a key-order watermark or a fence (no keys), so only the
    /// scan-wide seen set bounds it — without it, `2^levels` traversal work.
    /// Both directions must reject it at the first repeated id, after O(1)
    /// pages, with nothing delivered.
    #[test]
    fn an_empty_shared_dag_is_corrupt_to_both_scan_directions() {
        use crate::backend::{DataFileLocator, FileGeneration};
        use crate::cache::PageCache;
        use crate::page::{encode_branch, encode_leaf_inline, PageId, PAGE_SIZE};

        let empty_leaf = encode_leaf_inline(PageId(1), 0, &[]);
        let b1 = encode_branch(
            PageId(2),
            0,
            &[empty_leaf.page_ref(), empty_leaf.page_ref()],
            &[b"m".to_vec()],
        );
        let b2 = encode_branch(
            PageId(3),
            0,
            &[b1.page_ref(), b1.page_ref()],
            &[b"t".to_vec()],
        );
        // Fence-silent variant: one branch sharing one empty leaf. An empty
        // leaf has no keys, so fences and watermark are both vacuous — only
        // the seen set can reject this, in either direction.
        let small = encode_branch(
            PageId(4),
            0,
            &[empty_leaf.page_ref(), empty_leaf.page_ref()],
            &[b"m".to_vec()],
        );

        let dev = Arc::new(FakeIo::new(0));
        let loc = DataFileLocator::new();
        block_on(async {
            for (i, page) in [&empty_leaf, &b1, &b2, &small].iter().enumerate() {
                let off = (i as u64 + 1) * PAGE_SIZE as u64;
                loc.record(page.id().as_u64(), off);
                dev.write_all_at(off, page.bytes().to_vec().into_boxed_slice())
                    .await
                    .result
                    .unwrap();
            }
            let file = FileGeneration::direct(dev.clone() as Arc<dyn PageIo>, Arc::new(loc));
            let cache = PageCache::new();
            // The fence-silent single-level share: only the seen set fires.
            for (root, forward) in [(small.page_ref(), true), (small.page_ref(), false)] {
                let mut delivered = 0usize;
                let mut consumer = |_k: &[u8], _v: &[u8]| {
                    delivered += 1;
                    std::ops::ControlFlow::Continue(())
                };
                let out = if forward {
                    tree_scan_range(
                        &cache,
                        &file,
                        root,
                        Bound::Unbounded,
                        Bound::Unbounded,
                        &mut consumer,
                    )
                    .await
                } else {
                    tree_scan_range_rev(
                        &cache,
                        &file,
                        root,
                        Bound::Unbounded,
                        Bound::Unbounded,
                        &mut consumer,
                    )
                    .await
                };
                assert!(
                    matches!(
                        out,
                        Err(ReadError::Corrupt(
                            "page reachable twice in one tree (cycle?)"
                        ))
                    ),
                    "forward={forward}: got {out:?}"
                );
                assert_eq!(delivered, 0);
            }
            // The two-level DAG: bounded rejection in both directions. The
            // forward walk repeats b1's id (seen set); the reverse walk
            // reaches b1 under the right-hand fence first, where its
            // separator is misplaced — either guard is a correct bound.
            let out = tree_scan_range(
                &cache,
                &file,
                b2.page_ref(),
                Bound::Unbounded,
                Bound::Unbounded,
                &mut |_k, _v| std::ops::ControlFlow::Continue(()),
            )
            .await;
            assert!(
                matches!(
                    out,
                    Err(ReadError::Corrupt(
                        "page reachable twice in one tree (cycle?)"
                    ))
                ),
                "fwd dag: got {out:?}"
            );
            let out = tree_scan_range_rev(
                &cache,
                &file,
                b2.page_ref(),
                Bound::Unbounded,
                Bound::Unbounded,
                &mut |_k, _v| std::ops::ControlFlow::Continue(()),
            )
            .await;
            assert!(
                matches!(out, Err(ReadError::Corrupt(_))),
                "rev dag: got {out:?}"
            );
        });
    }

    /// The Before/After early-exit shortcuts are
    /// only sound on a fence-valid tree. A misplaced leaf (individually valid,
    /// checksum-valid, on the wrong side of its separator) must be `Corrupt`,
    /// not a silent `completed: true` scan that skipped in-range keys — in
    /// **both** directions.
    #[test]
    fn a_misplaced_leaf_is_corrupt_not_a_silently_short_scan() {
        use crate::backend::{DataFileLocator, FileGeneration};
        use crate::cache::PageCache;
        use crate::page::{encode_branch, encode_leaf_inline, PageId, PAGE_SIZE};

        // Reverse case: right leaf {a} below the separator "m". A reverse
        // scan of [h, +inf) visits it first; without fences, "a" is Before
        // and the scan would end completed:true having skipped in-range "j".
        let left = encode_leaf_inline(PageId(1), 0, &[(b"j".to_vec(), b"1".to_vec())]);
        let right = encode_leaf_inline(PageId(2), 0, &[(b"a".to_vec(), b"2".to_vec())]);
        let root = encode_branch(
            PageId(3),
            0,
            &[left.page_ref(), right.page_ref()],
            &[b"m".to_vec()],
        );
        // Forward case: left leaf {z} above the separator. A forward scan of
        // (-inf, k] visits it first; without fences, "z" is After and the
        // scan would end completed:true having skipped in-range "j" on the
        // right.
        let left_f = encode_leaf_inline(PageId(11), 0, &[(b"z".to_vec(), b"1".to_vec())]);
        let right_f = encode_leaf_inline(PageId(12), 0, &[(b"j".to_vec(), b"2".to_vec())]);
        let root_f = encode_branch(
            PageId(13),
            0,
            &[left_f.page_ref(), right_f.page_ref()],
            &[b"m".to_vec()],
        );

        let dev = Arc::new(FakeIo::new(0));
        let loc = DataFileLocator::new();
        block_on(async {
            for (i, page) in [&left, &right, &root, &left_f, &right_f, &root_f]
                .iter()
                .enumerate()
            {
                let off = (i as u64 + 1) * PAGE_SIZE as u64;
                loc.record(page.id().as_u64(), off);
                dev.write_all_at(off, page.bytes().to_vec().into_boxed_slice())
                    .await
                    .result
                    .unwrap();
            }
            let file = FileGeneration::direct(dev.clone() as Arc<dyn PageIo>, Arc::new(loc));
            let cache = PageCache::new();

            let out = tree_scan_range_rev(
                &cache,
                &file,
                root.page_ref(),
                Bound::Included(b"h".as_slice()),
                Bound::Unbounded,
                &mut |_k, _v| std::ops::ControlFlow::Continue(()),
            )
            .await;
            assert!(
                matches!(out, Err(ReadError::Corrupt("key below subtree low bound"))),
                "rev: got {out:?}"
            );

            let out = tree_scan_range(
                &cache,
                &file,
                root_f.page_ref(),
                Bound::Unbounded,
                Bound::Included(b"k".as_slice()),
                &mut |_k, _v| std::ops::ControlFlow::Continue(()),
            )
            .await;
            assert!(
                matches!(
                    out,
                    Err(ReadError::Corrupt("key at/above subtree high bound"))
                ),
                "fwd: got {out:?}"
            );
        });
    }

    /// An inverted bound pair whose endpoints route
    /// to different children yields an empty reverse child window (`lo > hi`);
    /// the reverse frame must treat it as exhausted, not visit child `hi`
    /// once. Pinned by faulting every leaf: a correct empty scan does no
    /// leaf I/O at all.
    #[test]
    fn an_inverted_reverse_range_reads_no_leaf() {
        use crate::backend::{DataFileLocator, FileGeneration};
        use crate::cache::PageCache;
        use crate::io::Faults;
        use crate::page::{encode_branch, encode_leaf_inline, PageId, PAGE_SIZE};

        let left = encode_leaf_inline(PageId(1), 0, &[(b"a".to_vec(), b"1".to_vec())]);
        let right = encode_leaf_inline(PageId(2), 0, &[(b"x".to_vec(), b"2".to_vec())]);
        let root = encode_branch(
            PageId(3),
            0,
            &[left.page_ref(), right.page_ref()],
            &[b"m".to_vec()],
        );
        let dev = Arc::new(FakeIo::new(0));
        let loc = DataFileLocator::new();
        block_on(async {
            for (i, page) in [&left, &right, &root].iter().enumerate() {
                let off = (i as u64 + 1) * PAGE_SIZE as u64;
                loc.record(page.id().as_u64(), off);
                dev.write_all_at(off, page.bytes().to_vec().into_boxed_slice())
                    .await
                    .result
                    .unwrap();
            }
            // Every leaf offset fails; only the root stays readable.
            dev.set_faults(Faults {
                fail_read_offsets: vec![PAGE_SIZE as u64, 2 * PAGE_SIZE as u64],
                ..Default::default()
            });
            let file = FileGeneration::direct(dev.clone() as Arc<dyn PageIo>, Arc::new(loc));
            let cache = PageCache::new();
            // start "x" routes to child 1, end "a" to child 0: lo=1 > hi=0.
            let summary = tree_scan_range_rev(
                &cache,
                &file,
                root.page_ref(),
                Bound::Included(b"x".as_slice()),
                Bound::Included(b"a".as_slice()),
                &mut |_k, _v| std::ops::ControlFlow::Continue(()),
            )
            .await
            .unwrap();
            assert_eq!(summary.visited, 0);
            assert!(summary.completed);
        });
    }

    /// Diamond-DAG corpus, case 2: the classic diamond — two *distinct*
    /// branches each naming the same leaf, under one root. Distinguished from
    /// case 1 because the repeat is now across siblings, not within one page's
    /// child directory: only a walk-wide (not per-parent) `seen` set catches it.
    #[test]
    fn a_diamond_of_two_branches_sharing_a_leaf_is_corrupt_to_every_walker() {
        use crate::backend::{DataFileLocator, FileGeneration};
        use crate::cache::PageCache;
        use crate::page::{encode_branch, encode_leaf_inline, PageId, PAGE_SIZE};

        let leaf_a = encode_leaf_inline(PageId(1), 0, &[(b"a".to_vec(), b"1".to_vec())]);
        let shared = encode_leaf_inline(PageId(2), 0, &[(b"m".to_vec(), b"2".to_vec())]);
        let leaf_z = encode_leaf_inline(PageId(3), 0, &[(b"z".to_vec(), b"3".to_vec())]);
        let branch_l = encode_branch(
            PageId(4),
            0,
            &[leaf_a.page_ref(), shared.page_ref()],
            &[b"g".to_vec()],
        );
        let branch_r = encode_branch(
            PageId(5),
            0,
            &[shared.page_ref(), leaf_z.page_ref()],
            &[b"t".to_vec()],
        );
        let root = encode_branch(
            PageId(6),
            0,
            &[branch_l.page_ref(), branch_r.page_ref()],
            &[b"p".to_vec()],
        );

        let dev = Arc::new(FakeIo::new(0));
        let loc = DataFileLocator::new();
        block_on(async {
            for (i, page) in [&leaf_a, &shared, &leaf_z, &branch_l, &branch_r, &root]
                .iter()
                .enumerate()
            {
                let off = (i as u64 + 1) * PAGE_SIZE as u64;
                loc.record(page.id().as_u64(), off);
                dev.write_all_at(off, page.bytes().to_vec().into_boxed_slice())
                    .await
                    .result
                    .unwrap();
            }
            let file = FileGeneration::direct(dev.clone() as Arc<dyn PageIo>, Arc::new(loc));
            let cache = PageCache::new();

            let mut walk = LivePages::new(root.page_ref());
            let mut yielded = 0usize;
            let err = loop {
                match walk.next(&cache, &file).await {
                    Ok(Some(_)) => yielded += 1,
                    Ok(None) => panic!("the diamond must not walk to completion"),
                    Err(e) => break e,
                }
            };
            assert!(
                matches!(
                    err,
                    ReadError::Corrupt("page reachable twice in one tree (cycle?)")
                ),
                "got {err:?}"
            );
            // Bounded work: every page at most once before the repeat trips.
            assert!(yielded <= 6, "walker yielded {yielded} pages");

            assert!(
                matches!(
                    tree_verify(&cache, &file, root.page_ref()).await,
                    Err(ReadError::Corrupt(
                        "page reachable twice in one tree (cycle?)"
                    ))
                ),
                "verify must reject the cross-sibling diamond"
            );
        });
    }

    #[test]
    fn single_leaf_tree() {
        // Few entries → a single leaf as root (no branch level).
        let data = entries(3);
        let dev = Arc::new(FakeIo::new(0));
        let built = TreeBuilder::default().build(dev, &data).unwrap();
        let reader = TreeReader::new(built);
        block_on(async {
            assert_eq!(
                reader.get(b"key000001".to_vec()).await.unwrap().as_deref(),
                Some(&b"value-1"[..])
            );
            assert_eq!(reader.verify().await.unwrap().entry_count, 3);
        });
    }
}
