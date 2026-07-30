//! The copy-on-write B+tree writer: leaf/branch insert, remove, and split
//! (**no merge**), applied as one immutable new tree per batch.
//!
//! A batch is applied as **one merge descent**: the ops are key-sorted
//! (stable, so same-key ops keep their batch
//! order; distinct keys are independent and checks/old-values were already
//! evaluated against the pre-state), partitioned across children at each
//! branch, and every touched path node is decoded and re-encoded **exactly
//! once** — not once per op, which made a 100-op batch re-encode (and
//! CRC-32C) the root ~100 times. Final entry sets and counts are identical to
//! per-op application; tree *shape* can differ — split boundaries, and one
//! deliberate case: a batch that transiently
//! empties the whole tree and reinserts used to reset the working root to
//! EMPTY mid-batch (discarding the no-merge skeleton as a side effect),
//! whereas the merge descent rebuilds within the retained skeleton — the same
//! sparse shape ordinary multi-batch deletes always produce, and the same
//! remedy applies (`compact`). A node that overflows is packed greedily
//! into as many new sibling nodes as it needs (a generalized split), and a
//! node that empties on remove is left in place (no merge — a deliberate
//! v1 limit). Height grows by wrapping a split root in a new branch.
//!
//! New pages are built into a private **overlay** (a local `HashMap`), never the
//! shared cache, so a failed build touches no shared state and a concurrent
//! reader on the old root can never observe a half-built tree. Fresh page ids
//! never collide with live ids (append-only allocation), so publishing the
//! reachable pages as `Dirty` then swapping the root is atomic to readers
//! (invariant 1/2). Only pages reachable from the final root are returned;
//! pages superseded *within* the batch are dropped as garbage (never charged,
//! never written).

use crate::backend::FileGeneration;
use crate::batch::Op;
use crate::cache::PageCache;
use crate::direct::Allocator;
use crate::error::ReadError;
use crate::page::{
    body_capacity, branch_entry_overhead, branch_first_overhead, encode_branch, encode_leaf,
    encode_overflow, leaf_entry_overhead, overflow_chunks, EntryValue, OverflowRef, Page, PageId,
    PageKind, PageRef, MAX_INLINE_VALUE_LEN,
};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Bound on tree height a write descent will follow before declaring the source
/// structure corrupt (mirrors the read path's guard).
// T3.4: ONE depth constant for writer and reader — drift would let the
// builder produce a tree the read path refuses.
use crate::read::MAX_TREE_DEPTH;

/// A boxed, borrowing future returning one node's multi-op `(result, count-delta)`.
type OpsFuture<'b> = Pin<Box<dyn Future<Output = Result<(NodeResult, i64), ReadError>> + 'b>>;

/// The immutable result of applying a batch: the new root, its entry count, and
/// every page reachable from it that this batch created (to pin `Dirty` and
/// flush).
pub(crate) struct BuiltBatch {
    pub root: PageRef,
    pub entry_count: u64,
    pub pages: Vec<Arc<Page>>,
}

/// A set of sibling nodes that replace one node, with the separators between
/// them. `children.len() == seps.len() + 1`.
struct NodeSet {
    children: Vec<PageRef>,
    seps: Vec<Vec<u8>>,
}

/// Outcome of applying one op at a node.
enum NodeResult {
    /// The op did not change this subtree (e.g. removing an absent key, or
    /// overwriting a key with its current value): the parent keeps its ref.
    Unchanged,
    /// The subtree was rebuilt into `NodeSet`.
    Changed(NodeSet),
}

struct Builder<'a> {
    cache: &'a PageCache,
    file: &'a FileGeneration,
    alloc: &'a mut Allocator,
    txid: u64,
    overlay: HashMap<PageId, Arc<Page>>,
}

/// Applies `ops` to the tree rooted at `base_root` (with `base_count` entries),
/// producing a new immutable tree. Reads flow through `cache` for pre-existing
/// pages and the overlay for pages built earlier in this batch.
pub(crate) async fn apply_batch(
    cache: &PageCache,
    file: &FileGeneration,
    base_root: PageRef,
    base_count: u64,
    ops: &[Op],
    alloc: &mut Allocator,
    txid: u64,
) -> Result<BuiltBatch, ReadError> {
    let mut b = Builder {
        cache,
        file,
        alloc,
        txid,
        overlay: HashMap::new(),
    };
    // Key-sort the ops (stable: same-key ops keep their batch order) so one
    // descent can partition them across children and rebuild each touched
    // path node exactly once. Each original leaf covers a key interval and
    // receives every op whose key falls in it, so the final entry sets are
    // identical to per-op application; shape can differ (module doc).
    let mut sorted: Vec<&Op> = ops.iter().collect();
    sorted.sort_by(|a, b| a.key().cmp(b.key()));
    let mut root = base_root;
    let mut count = base_count as i64;
    if root.is_empty() {
        // Empty-tree base case: apply every op to an empty entry vec (removes
        // are no-ops) and build whatever remains.
        let mut entries: Vec<(Vec<u8>, EntryValue)> = Vec::new();
        let mut append = true;
        for op in &sorted {
            let (changed, delta, ap) = b.apply_leaf(&mut entries, op).await?;
            if changed {
                append &= ap;
            }
            count += delta;
        }
        if !entries.is_empty() {
            let ns = b.chunk_leaf(&entries, append);
            root = b.build_up(ns.children, ns.seps);
        }
    } else {
        let (res, delta) = b.ops_node(root, &sorted, 0).await?;
        count += delta;
        if let NodeResult::Changed(ns) = res {
            root = b.build_up(ns.children, ns.seps);
        }
    }
    if count <= 0 {
        // The tree emptied: canonical empty representation is EMPTY, not an
        // empty-leaf root. Any pages just built are dropped as garbage.
        root = PageRef::EMPTY;
        count = 0;
    }
    let pages = b.collect_reachable(root);
    Ok(BuiltBatch {
        root,
        entry_count: count as u64,
        pages,
    })
}

impl<'a> Builder<'a> {
    /// Reads a node: the overlay (built this batch) wins over the cache.
    ///
    /// An overlay hit is validated against `r` **exactly as a cache hit is**
    /// (`PageCache::get_page` checks the image against the parent's expected
    /// checksum). Resolving by `page_id` alone would make the overlay a hole in the
    /// parent-anchored integrity rule: ids are unique and never reused, so a *legal*
    /// ref can never name a page this batch just minted — but a **forged** one can,
    /// and recovery does not traverse leaf overflow refs, so a leaf naming a
    /// not-yet-existing id survives to be read here. That ref would then resolve to
    /// an unrelated page the batch happened to allocate at that id, and
    /// `overflow_equals` would compare against it and report the value unchanged —
    /// acknowledging a write while silently keeping a ref that later reads as
    /// `Corrupt`.
    async fn get_node(&mut self, r: PageRef) -> Result<Arc<Page>, ReadError> {
        if let Some(p) = self.overlay.get(&r.page_id) {
            if p.page_ref() != r {
                return Err(ReadError::Corrupt(
                    "overlay page does not match its reference",
                ));
            }
            return Ok(p.clone());
        }
        self.cache.get_page(r, self.file).await
    }

    fn emit_leaf(&mut self, entries: &[(Vec<u8>, EntryValue)]) -> PageRef {
        let id = self.alloc.alloc_id();
        let page = encode_leaf(id, self.txid, entries);
        let r = page.page_ref();
        self.overlay.insert(id, Arc::new(page));
        r
    }

    /// Spills `value` to a fresh overflow chain, emitting its links into the
    /// overlay and returning the ref the leaf will carry (M6 Phase F).
    ///
    /// Ids are drawn **head-first**, then the pages are *encoded* tail-first. The
    /// encode order is forced (a link embeds its successor's id and checksum); the
    /// id order need not follow it, and must not: StoreDirect's page offset is a
    /// pure function of the page id, so allocating in encode order would lay every
    /// large value out backwards on disk and make a sequential read of it walk 65
    /// descending pages, defeating readahead for exactly the objects this phase
    /// exists to store. Reserving the range first costs nothing
    /// and keeps disk order equal to value order.
    ///
    /// The chain is new pages on new ids, like every other page this batch
    /// builds; if the batch is abandoned they are overlay garbage that is never
    /// charged and never written.
    fn emit_overflow(&mut self, value: &[u8]) -> OverflowRef {
        debug_assert!(!value.is_empty(), "an empty value never spills");
        let ids: Vec<PageId> = overflow_chunks(value)
            .map(|_| self.alloc.alloc_id())
            .collect();
        let mut next: Option<PageRef> = None;
        for (i, chunk) in overflow_chunks(value).enumerate().rev() {
            let page = encode_overflow(ids[i], self.txid, chunk, next);
            next = Some(page.page_ref());
            self.overlay.insert(ids[i], Arc::new(page));
        }
        OverflowRef {
            head: next.expect("a spilled value has at least one chunk"),
            total_len: value.len() as u64,
        }
    }

    /// The stored representation of an op's value: inline, or a freshly built
    /// chain if it is over the spill threshold.
    fn store_value(&mut self, value: &[u8]) -> EntryValue {
        if value.len() > MAX_INLINE_VALUE_LEN {
            EntryValue::Overflow(self.emit_overflow(value))
        } else {
            EntryValue::Inline(value.to_vec())
        }
    }

    fn emit_branch(&mut self, children: &[PageRef], seps: &[Vec<u8>]) -> PageRef {
        let id = self.alloc.alloc_id();
        let page = encode_branch(id, self.txid, children, seps);
        let r = page.page_ref();
        self.overlay.insert(id, Arc::new(page));
        r
    }

    /// Packs `entries` into as few leaves as fit one page each, emitting them.
    ///
    /// **Split policy (M6 Phase C — insertion-direction-aware).** Greedy
    /// packing produces a dense `[full][tiny]` tail. That is *optimal* for an
    /// **append** (the tiny tail keeps receiving the next appends and densifies),
    /// but pathological for a non-append **overflow** into an already-full leaf:
    /// the tail stays near-empty and every later interior insert re-splits the hot
    /// leaf, fragmenting the tree (a real observed defect). So when this rebuild is
    /// *not* an append and greedy left ≥ 2 chunks, the **last two chunks are
    /// byte-balanced** (feasible split closest to an even byte share, both sides
    /// non-empty and page-fitting) — the standard "split a full node in two"
    /// behaviour — trading one-time density for no repeated re-splitting. Append
    /// keeps greedy so ascending/bulk stays ~100% dense. `append` is decided at
    /// the op level ([`apply_leaf`]) because a merged entry slice cannot tell an
    /// append from a rightmost overwrite. Leaves only; the branch splitter keeps
    /// its existing lone-child rebalance.
    fn chunk_leaf(&mut self, entries: &[(Vec<u8>, EntryValue)], append: bool) -> NodeSet {
        let mut children = Vec::new();
        let mut seps = Vec::new();
        if entries.is_empty() {
            children.push(self.emit_leaf(&[]));
            return NodeSet { children, seps };
        }
        // Phase 1: greedy chunk boundaries [start, end) over entry indices.
        let mut bounds: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < entries.len() {
            let start = i;
            let mut used = 0usize;
            while i < entries.len() {
                let (k, v) = &entries[i];
                let add = leaf_entry_overhead(k.len(), v.stored_len());
                if i > start && used + add > body_capacity() {
                    break;
                }
                used += add;
                i += 1;
            }
            bounds.push((start, i));
        }
        // Phase 2: on a non-append overflow, byte-balance the last two chunks. The
        // greedy boundary is always feasible, so a feasible balanced cut exists.
        if !append && bounds.len() >= 2 {
            let last = bounds.len() - 1;
            let lo = bounds[last - 1].0;
            let hi = bounds[last].1;
            let cut = self.balanced_leaf_cut(entries, lo, hi);
            bounds[last - 1] = (lo, cut);
            bounds[last] = (cut, hi);
        }
        // Phase 3: emit one leaf per chunk; promote each non-first chunk's first
        // key as the separator to its left.
        for (bi, &(start, end)) in bounds.iter().enumerate() {
            if bi > 0 {
                seps.push(entries[start].0.clone());
            }
            children.push(self.emit_leaf(&entries[start..end]));
        }
        NodeSet { children, seps }
    }

    /// The cut index in `(lo, hi)` splitting `entries[lo..hi]` into two page-fitting
    /// leaves with the most even byte share. `entries[lo..hi]` spans at most two
    /// greedy chunks (≤ 2× capacity), so a feasible cut always exists (the greedy
    /// boundary itself). Prefixes are scanned once; the search minimises the
    /// absolute byte difference among cuts where both sides fit `body_capacity`.
    fn balanced_leaf_cut(&self, entries: &[(Vec<u8>, EntryValue)], lo: usize, hi: usize) -> usize {
        let cap = body_capacity();
        // Prefix byte sums so each side's size is O(1) to evaluate.
        let mut prefix = vec![0usize; hi - lo + 1];
        for j in lo..hi {
            let (k, v) = &entries[j];
            prefix[j - lo + 1] = prefix[j - lo] + leaf_entry_overhead(k.len(), v.stored_len());
        }
        let total = prefix[hi - lo];
        let mut best_cut = lo + 1; // greedy fallback replaced below if better found
        let mut best_diff = usize::MAX;
        let mut feasible_found = false;
        for cut in (lo + 1)..hi {
            let left = prefix[cut - lo];
            let right = total - left;
            if left > cap || right > cap {
                continue; // not both page-fitting
            }
            let diff = left.abs_diff(right);
            if !feasible_found || diff < best_diff {
                best_diff = diff;
                best_cut = cut;
                feasible_found = true;
            }
        }
        best_cut
    }

    /// Packs `children`/`seps` into as few branches as fit one page each,
    /// promoting the separator between chunks. `children.len() == seps.len()+1`
    /// and always ≥ 2 (callers never pass a lone child).
    ///
    /// Greedy packing can leave a lone-child final chunk (e.g. 5 children where
    /// only 4 fit → `[4][1]`). A one-child branch is a legal but degenerate node
    /// and violates the "branches keep ≥ 2 children" invariant, so the lone tail
    /// is rebalanced into the previous chunk (`[4][1] → [3][2]`). This is always
    /// possible: with `MAX_KEY_LEN ≤ page` a branch always fits ≥ 4 children, so
    /// `cap-1 ≥ 3 ≥ 2` after moving one across.
    fn chunk_branch(&mut self, children: Vec<PageRef>, seps: Vec<Vec<u8>>) -> NodeSet {
        debug_assert_eq!(children.len(), seps.len() + 1);
        // First compute chunk boundaries [start, end) over child indices.
        let mut bounds: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < children.len() {
            let start = i;
            let mut used = branch_first_overhead();
            i += 1;
            while i < children.len() {
                let add = branch_entry_overhead(seps[i - 1].len());
                if used + add > body_capacity() {
                    break;
                }
                used += add;
                i += 1;
            }
            bounds.push((start, i));
        }
        // Rebalance a lone-child final chunk into its predecessor.
        if bounds.len() >= 2 {
            let last = bounds.len() - 1;
            if bounds[last].1 - bounds[last].0 == 1 {
                bounds[last].0 -= 1;
                bounds[last - 1].1 -= 1;
            }
        }
        // Emit one branch per chunk; the separator between two chunks (the one
        // spanning `child[start-1]..child[start]`) is promoted to this level.
        let mut out_children = Vec::with_capacity(bounds.len());
        let mut out_seps = Vec::with_capacity(bounds.len().saturating_sub(1));
        for (bi, &(start, end)) in bounds.iter().enumerate() {
            let r = self.emit_branch(&children[start..end], &seps[start..end - 1]);
            if bi > 0 {
                out_seps.push(seps[start - 1].clone());
            }
            out_children.push(r);
        }
        NodeSet {
            children: out_children,
            seps: out_seps,
        }
    }

    /// Reduces a NodeSet to a single root ref, wrapping in new branch levels
    /// until one node remains. Height grows here.
    fn build_up(&mut self, mut children: Vec<PageRef>, mut seps: Vec<Vec<u8>>) -> PageRef {
        while children.len() > 1 {
            let ns = self.chunk_branch(children, seps);
            children = ns.children;
            seps = ns.seps;
        }
        children[0]
    }

    /// Recursive path-copy at one node, applying a **key-sorted run of ops**
    /// whose keys all route into this subtree. Rebuilds the node at most once.
    fn ops_node<'b>(&'b mut self, node: PageRef, ops: &'b [&'b Op], depth: u32) -> OpsFuture<'b> {
        Box::pin(async move {
            if depth >= MAX_TREE_DEPTH {
                return Err(ReadError::Corrupt("tree exceeds max depth (cycle?)"));
            }
            let page = self.get_node(node).await?;
            match page.kind() {
                PageKind::Leaf => {
                    // `to_owned_value` copies an untouched entry's *ref*, not its
                    // chain: the rebuilt leaf re-points at the same immutable
                    // overflow pages (M6 Phase F). Large values cost
                    // nothing to carry through a neighbouring key's rewrite.
                    let mut entries: Vec<(Vec<u8>, EntryValue)> = page
                        .leaf_entries()
                        .map(|(k, v)| (k.to_vec(), v.to_owned_value()))
                        .collect();
                    drop(page);
                    // One decoded vec takes the whole run; chunk once at the
                    // end. `append` only if every changing op appended (an
                    // interior touch forces the balanced split policy).
                    let mut changed_any = false;
                    let mut append_all = true;
                    let mut delta_sum = 0i64;
                    for op in ops {
                        let (changed, delta, append) = self.apply_leaf(&mut entries, op).await?;
                        if changed {
                            changed_any = true;
                            append_all &= append;
                        }
                        delta_sum += delta;
                    }
                    if !changed_any {
                        return Ok((NodeResult::Unchanged, 0));
                    }
                    Ok((
                        NodeResult::Changed(self.chunk_leaf(&entries, append_all)),
                        delta_sum,
                    ))
                }
                PageKind::Overflow => {
                    Err(ReadError::Corrupt("tree descent reached an overflow page"))
                }
                PageKind::Branch => {
                    let children = page.branch_children().to_vec();
                    let seps: Vec<Vec<u8>> = page.branch_separators().map(<[u8]>::to_vec).collect();
                    drop(page);
                    // Ops are key-sorted, so per-child runs are contiguous and
                    // their child indices non-decreasing.
                    let mut replacements: Vec<Option<NodeSet>> = Vec::new();
                    replacements.resize_with(children.len(), || None);
                    let mut changed_any = false;
                    let mut delta_sum = 0i64;
                    let mut j = 0;
                    while j < ops.len() {
                        let ci = route(&seps, ops[j].key());
                        let start = j;
                        j += 1;
                        while j < ops.len() && route(&seps, ops[j].key()) == ci {
                            j += 1;
                        }
                        let (res, delta) = self
                            .ops_node(children[ci], &ops[start..j], depth + 1)
                            .await?;
                        delta_sum += delta;
                        if let NodeResult::Changed(ns) = res {
                            replacements[ci] = Some(ns);
                            changed_any = true;
                        }
                    }
                    if !changed_any {
                        return Ok((NodeResult::Unchanged, 0));
                    }
                    // Splice every changed child's replacement set in order —
                    // the multi-child generalization of
                    //   children = C[..i] ++ cs ++ C[i+1..]
                    //   seps     = S[..i] ++ cseps ++ S[i..]
                    let mut new_children = Vec::with_capacity(children.len());
                    let mut new_seps = Vec::with_capacity(seps.len());
                    for (i, child) in children.iter().enumerate() {
                        if i > 0 {
                            new_seps.push(seps[i - 1].clone());
                        }
                        match replacements[i].take() {
                            Some(ns) => {
                                new_children.extend(ns.children);
                                new_seps.extend(ns.seps);
                            }
                            None => new_children.push(*child),
                        }
                    }
                    Ok((
                        NodeResult::Changed(self.chunk_branch(new_children, new_seps)),
                        delta_sum,
                    ))
                }
            }
        })
    }

    /// Applies one op to a sorted leaf-entry vector, spilling an oversized new
    /// value to a chain. Returns `(changed, delta, append)`, where `append` is
    /// `true` iff the op inserted a **new** key strictly greater than every
    /// pre-image key in this leaf (position == old length). That is the append
    /// signal the split policy needs ([`Builder::chunk_leaf`]): a rightmost
    /// *overwrite* grows a value but is not an append, and a removal never is.
    ///
    /// **Overflow and the unchanged short-circuit** (M6 Phase F): an
    /// overwrite of an already-spilled value must still be able to report
    /// `Unchanged`, so this compares the incoming bytes against the existing
    /// chain. Length first (a cheap disproof); only on a length match does it
    /// stream the chain.
    ///
    /// Skipping the comparison would not be a mere inefficiency: re-inserting a
    /// value identical to the stored one would become a 65-page chain rebuild plus
    /// a root-path copy, and a retry loop would drive the WAL to its limit writing
    /// nothing. The extra reads land only on the
    /// equal-length overwrite case, and the chain is usually cache-resident from
    /// the read that preceded the write.
    ///
    /// It removes the WAL **consumption**, not the **reservation**: admission gates
    /// the command on `WriteBatch::cost`'s worst case long before this runs, and
    /// cannot know a batch is a no-op without doing the batch. So against a nearly
    /// full WAL an idempotent large re-insert is still refused `StoreFull` — pinned
    /// by `an_idempotent_large_reinsert_still_needs_admission_headroom`.
    ///
    /// The head checksum cannot stand in for a content hash: it commits the page's
    /// own id and its successor ref, so two chains holding identical bytes have
    /// different head checksums.
    async fn apply_leaf(
        &mut self,
        entries: &mut Vec<(Vec<u8>, EntryValue)>,
        op: &Op,
    ) -> Result<(bool, i64, bool), ReadError> {
        match op {
            Op::Insert { key, value } => {
                match entries.binary_search_by(|(k, _)| k.as_slice().cmp(key)) {
                    Ok(i) => {
                        let unchanged = match &entries[i].1 {
                            EntryValue::Inline(cur) => cur.as_slice() == value.as_slice(),
                            EntryValue::Overflow(o) => {
                                o.total_len == value.len() as u64
                                    && self.overflow_equals(*o, value).await?
                            }
                        };
                        if unchanged {
                            return Ok((false, 0, false));
                        }
                        entries[i].1 = self.store_value(value);
                        Ok((true, 0, false))
                    }
                    Err(i) => {
                        let append = i == entries.len();
                        let stored = self.store_value(value);
                        entries.insert(i, (key.clone(), stored));
                        Ok((true, 1, append))
                    }
                }
            }
            Op::Remove { key } => match entries.binary_search_by(|(k, _)| k.as_slice().cmp(key)) {
                Ok(i) => {
                    entries.remove(i);
                    Ok((true, -1, false))
                }
                Err(_) => Ok((false, 0, false)),
            },
        }
    }

    /// Whether the chain at `oref` holds exactly `value`, streamed a link at a
    /// time (never materialising the stored value). Caller has already matched
    /// the lengths.
    ///
    /// Reads via [`get_node`](Self::get_node), not the cache directly: a chain
    /// built earlier in *this* batch lives in the private overlay and is invisible
    /// to the shared cache.
    async fn overflow_equals(
        &mut self,
        oref: OverflowRef,
        value: &[u8],
    ) -> Result<bool, ReadError> {
        debug_assert_eq!(oref.total_len, value.len() as u64);
        let mut cursor = crate::page::OverflowCursor::new(oref);
        let mut at = 0usize;
        while let Some(r) = cursor.pending() {
            let page = self.get_node(r).await?;
            cursor.accept(&page)?;
            let chunk = page.overflow_chunk();
            // The cursor proved the chain carries exactly `total_len == value.len()`
            // bytes, so this slice is always in range.
            if chunk != &value[at..at + chunk.len()] {
                return Ok(false);
            }
            at += chunk.len();
        }
        Ok(true)
    }

    /// Collects every page reachable from `root` that lives in the overlay
    /// (i.e. was built this batch). Pre-existing pages stop the walk.
    ///
    /// Overflow chains are reachable through a leaf's refs (M6 Phase F). A chain
    /// this batch built lives in the overlay and **must** be collected — those
    /// are the pages the writer pins `Dirty` and flushes, so missing them would
    /// publish a root whose values were never written. A chain an untouched entry
    /// merely re-points at is *not* in the overlay; the walk stops there exactly
    /// as it does for a pre-existing tree page, which is what makes carrying a
    /// large value through a neighbour's rewrite free.
    fn collect_reachable(&self, root: PageRef) -> Vec<Arc<Page>> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = vec![root];
        while let Some(r) = stack.pop() {
            if r.is_empty() || !seen.insert(r.page_id) {
                continue;
            }
            if let Some(p) = self.overlay.get(&r.page_id) {
                // The one overlay lookup that does *not* re-derive trust from the
                // ref (`get_node` does): every ref reaching here was minted by this
                // builder from a page it had just encoded, so a mismatch is a
                // builder bug, not hostile input — there is no untrusted ref on
                // this path to defend against. Asserted rather than checked, so a
                // future edit that walks a *read* ref through here trips in tests
                // instead of silently reintroducing the hole `get_node` closes.
                debug_assert_eq!(
                    p.page_ref(),
                    r,
                    "collect_reachable walked a ref this builder did not mint"
                );
                match p.kind() {
                    PageKind::Branch => stack.extend(p.branch_children().iter().copied()),
                    PageKind::Leaf => stack.extend(p.leaf_overflow_refs().map(|o| o.head)),
                    PageKind::Overflow => stack.extend(p.overflow_next()),
                }
                out.push(p.clone());
            }
        }
        out
    }
}

/// The child index covering `key`: first separator strictly greater than `key`,
/// else the last child. Identical routing to [`Page::branch_child_for`].
fn route(seps: &[Vec<u8>], key: &[u8]) -> usize {
    seps.iter()
        .position(|s| key < s.as_slice())
        .unwrap_or(seps.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DataFileLocator;
    use crate::io::FakeIo;
    use crate::page::{encode_overflow, Checksum};
    use futures_lite::future::block_on;

    /// The builder's overlay must re-derive trust from the ref, exactly as the
    /// cache does.
    ///
    /// Ids are never reused, so a **legal** ref can never name a page the current
    /// batch just minted. A **forged** one can: recovery does not traverse leaf
    /// overflow refs, so a leaf naming a not-yet-allocated id survives to reach the
    /// writer, and `overflow_equals` then resolves it here. Keyed on `page_id`
    /// alone, that lookup would hand back whatever unrelated page this batch
    /// happened to allocate at that id — and the comparison would run against it,
    /// report the value unchanged, and acknowledge the write while keeping a ref
    /// that reads `Corrupt` forever after.
    #[test]
    fn an_overlay_hit_is_validated_against_its_reference() {
        let dev = Arc::new(FakeIo::new(0)) as Arc<dyn crate::io::PageIo>;
        let file = FileGeneration::direct(dev, Arc::new(DataFileLocator::new()));
        let cache = PageCache::new();
        let mut alloc = crate::direct::Allocator::for_wal(crate::direct::FIRST_DATA_INDEX);
        let mut b = Builder {
            cache: &cache,
            file: &file,
            alloc: &mut alloc,
            txid: 1,
            overlay: HashMap::new(),
        };

        // A page this batch built, as `emit_overflow` would leave it.
        let page = encode_overflow(PageId(7), 1, b"minted by this batch", None);
        let honest = page.page_ref();
        b.overlay.insert(PageId(7), Arc::new(page));

        // The honest ref resolves.
        let got = block_on(b.get_node(honest)).unwrap();
        assert_eq!(got.id(), PageId(7));

        // A forged ref naming the same id with a different checksum must NOT.
        let forged = PageRef {
            page_id: PageId(7),
            expected_checksum: Checksum(honest.expected_checksum.0 ^ 1),
        };
        let err = block_on(b.get_node(forged)).unwrap_err();
        assert!(
            matches!(
                err,
                ReadError::Corrupt("overlay page does not match its reference")
            ),
            "an overlay hit must be checked against its ref like a cache hit: {err:?}"
        );
    }
}
