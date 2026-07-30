//! Page identity, references, the fixed page codec, and structural validation.
//!
//! [`Page::decode`] is the **single trust boundary** for a loaded image: it
//! validates the header *and* the full leaf/branch body against the `PageRef`
//! its parent expects, so every later index into the page is infallible: all
//! decoded offsets/counts are validated before allocation or indexing.
//!
//! Integrity is anchored entirely in the parent: a `PageRef` carries the child's
//! `expected_checksum`, verified over the whole image at decode. No checksum is
//! stored *inside* the page.

use crate::error::ReadError;
use std::ops::Range;

/// Fixed page size for v1. The format carries a page-size field, but only this
/// value is supported.
pub const PAGE_SIZE: usize = 16 * 1024;

/// Maximum inline key length (bytes). Enforced at batch admission and re-checked
/// structurally on decode.
pub const MAX_KEY_LEN: usize = 4 * 1024;

/// Maximum length of a value stored **inline** in a leaf (bytes). A value longer
/// than this spills to an overflow chain (M6 Phase F). This is exactly the M2–E
/// `MAX_VALUE_LEN`, so every value that stored inline before Phase F still does:
/// the spill threshold adds a representation, it does not take one away.
///
/// Together with [`MAX_KEY_LEN`] this is what makes "a single entry always fits
/// one page" true (`4K + 8K + slot ≤ body_capacity()`), which every splitter
/// relies on.
pub const MAX_INLINE_VALUE_LEN: usize = 8 * 1024;

/// Maximum value length (bytes), inline or overflowed. Enforced at batch
/// admission and re-checked structurally on decode (an overflow ref's
/// `total_len` is bounded by it).
///
/// **This interacts with `AdmissionLimits`.** A spilled value's chain is
/// `ceil(len / overflow_payload_capacity())` pages — up to 65 here — and those
/// pages are charged to `max_dirty_pages` and `max_wal_bytes` at admission. A
/// store configured with limits too small to admit one maximum-size value will
/// reject such a write *permanently*, not transiently. The defaults leave ample
/// room; a tuned-down configuration must budget for the largest value it intends
/// to store.
pub const MAX_VALUE_LEN: usize = 1024 * 1024;

/// A page identifier. v1 never reuses a page id.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct PageId(pub u64);

impl PageId {
    /// The reserved id used by the distinguished empty `PageRef` (a store with
    /// no data pages). Never a real data page.
    pub const NONE: PageId = PageId(0);

    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// A CRC checksum over a page image. The algorithm is selected by the store
/// header's checksum-algorithm id; crc32c is the v1 default.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct Checksum(pub u32);

impl Checksum {
    /// crc32c over the given bytes (v1 default algorithm).
    pub fn crc32c(bytes: &[u8]) -> Checksum {
        Checksum(crc32c::crc32c(bytes))
    }
}

/// A typed reference to a child/root page: its id plus the checksum the parent
/// expects it to have. Reads validate the loaded image against
/// `expected_checksum` before trusting any decoded offset.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct PageRef {
    pub page_id: PageId,
    pub expected_checksum: Checksum,
}

impl PageRef {
    /// The distinguished empty reference: an empty tree/root. Its `page_id` is
    /// [`PageId::NONE`] and it points at no image.
    pub const EMPTY: PageRef = PageRef {
        page_id: PageId::NONE,
        expected_checksum: Checksum(0),
    };

    pub const fn is_empty(self) -> bool {
        self.page_id.0 == PageId::NONE.0
    }
}

/// Structural page kinds.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum PageKind {
    /// A B+tree leaf holding ordered key/value entries.
    Leaf = 1,
    /// A B+tree branch holding child `PageRef`s and separator keys.
    Branch = 2,
    /// One link of an overflow-value chain: a payload chunk plus the reference
    /// to the next link (M6 Phase F).
    Overflow = 3,
}

impl PageKind {
    fn from_u8(v: u8) -> Result<PageKind, ReadError> {
        match v {
            1 => Ok(PageKind::Leaf),
            2 => Ok(PageKind::Branch),
            3 => Ok(PageKind::Overflow),
            _ => Err(ReadError::Corrupt("unknown page kind")),
        }
    }
}

/// A leaf entry's reference to an overflow value: the chain's head page and the
/// value's total length. The `total_len` is what bounds a chain walk exactly —
/// a reader stops when it has collected that many bytes, so a checksum-valid but
/// cyclic chain cannot loop forever.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct OverflowRef {
    pub head: PageRef,
    pub total_len: u64,
}

/// The **cross-page trust boundary for an overflow chain** (M6 Phase F).
///
/// [`Page::decode`] is the per-image boundary, but no single image can prove a
/// chain carries exactly the bytes its leaf declared: a checksum-valid leaf
/// saying `total_len = 1 MiB` can point at a checksum-valid one-byte tail and
/// every *local* check passes. A walker that merely followed `next` would count,
/// relocate, and durably commit that structure, and only a later point read would
/// discover the value is gone — maintenance would have laundered corruption into
/// the new file.
///
/// So chain structure is validated in exactly one place, and every consumer —
/// point reads, `tree_verify`, the live-set walker, Direct's compaction rebuild —
/// drives *this* cursor rather than its own loop. The rule is canonical, so a
/// chain has exactly one legal encoding for a given value:
///
/// - every link is [`PageKind::Overflow`];
/// - every non-tail link is exactly full (enforced at decode);
/// - the chain ends exactly where `total_len` runs out, and the tail's chunk is
///   exactly the remainder.
///
/// Cycles need no separate guard: `remaining` falls by a full page at every
/// interior link, so a loop must eventually reach a link that has to be the tail
/// and is not.
pub struct OverflowCursor {
    next: Option<PageRef>,
    remaining: u64,
}

impl OverflowCursor {
    pub fn new(oref: OverflowRef) -> Self {
        OverflowCursor {
            next: Some(oref.head),
            remaining: oref.total_len,
        }
    }

    /// The next link to load, or `None` once the chain is complete.
    pub fn pending(&self) -> Option<PageRef> {
        self.next
    }

    /// Validates `page` as the link [`pending`](Self::pending) named and advances.
    /// The caller may read `page.overflow_chunk()` only after this returns `Ok`.
    ///
    /// The image is checked against the ref **this cursor is pending**, not against
    /// whatever the caller thinks it loaded: a trust boundary that took "this is
    /// the page I asked for" on faith would be trusting its caller's lookup, and
    /// not every lookup verifies (the writer's private overlay resolves by id
    /// alone). Since the cursor already knows which link it wants, checking against
    /// its own state is both stronger and simpler than taking the caller's word —
    /// there is no argument a caller can pass that weakens it.
    pub fn accept(&mut self, page: &Page) -> Result<(), ReadError> {
        let Some(want) = self.next else {
            return Err(ReadError::Corrupt("overflow chain already complete"));
        };
        if page.page_ref() != want {
            return Err(ReadError::Corrupt(
                "overflow link is not the page its ref named",
            ));
        }
        if page.kind() != PageKind::Overflow {
            return Err(ReadError::Corrupt(
                "overflow chain link is not an overflow page",
            ));
        }
        let chunk = page.overflow_chunk().len() as u64;
        match page.overflow_next() {
            Some(n) => {
                // An interior link is full (decode guarantees it), so the declared
                // length must have more than this link's worth left to run.
                if self.remaining <= chunk {
                    return Err(ReadError::Corrupt(
                        "overflow chain longer than its declared length",
                    ));
                }
                self.remaining -= chunk;
                self.next = Some(n);
            }
            None => {
                if self.remaining != chunk {
                    return Err(ReadError::Corrupt(
                        "overflow chain does not carry its declared length",
                    ));
                }
                self.remaining = 0;
                self.next = None;
            }
        }
        Ok(())
    }
}

/// A leaf entry's value as stored: bytes living in this page, or a reference to
/// a chain. Borrowed from a decoded [`Page`].
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum LeafValue<'a> {
    Inline(&'a [u8]),
    Overflow(OverflowRef),
}

impl LeafValue<'_> {
    /// The value's logical length, whichever representation it uses — and
    /// **without reading the chain**: an overflowed value's length is carried in
    /// the leaf's ref, so sizing a value costs the leaf load a caller already
    /// paid, not the value's own I/O.
    pub fn len(&self) -> u64 {
        match self {
            LeafValue::Inline(b) => b.len() as u64,
            LeafValue::Overflow(o) => o.total_len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An owned copy of this value's *representation* (not its bytes: an
    /// overflow value copies the ref, sharing the chain — see [`EntryValue`]).
    pub fn to_owned_value(&self) -> EntryValue {
        match self {
            LeafValue::Inline(b) => EntryValue::Inline(b.to_vec()),
            LeafValue::Overflow(o) => EntryValue::Overflow(*o),
        }
    }
}

/// An owned leaf-entry value handed to [`encode_leaf`].
///
/// `Overflow` carries only the *reference*. That is the whole reason a CoW
/// rebuild of a leaf is cheap for large values: chain pages are immutable and
/// id-unique like any other page, so a rebuilt leaf that did not touch an entry
/// re-points at the **same** chain rather than rewriting it (M6 Phase F).
#[derive(Clone, Eq, PartialEq, Debug)]
pub enum EntryValue {
    Inline(Vec<u8>),
    Overflow(OverflowRef),
}

impl EntryValue {
    /// Bytes this value occupies in the leaf's heap: the value itself when
    /// inline, or the fixed ref record when overflowed.
    pub fn stored_len(&self) -> usize {
        match self {
            EntryValue::Inline(v) => v.len(),
            EntryValue::Overflow(_) => OVERFLOW_REF_LEN,
        }
    }
}

// Header layout constants (offsets into the page image). Frozen in M0 so the
// fake and cache agree on where the header sits; the body codec below is M2.
mod header {
    pub const MAGIC: u32 = 0x4D50_3555; // "MP5U"
    pub const OFF_MAGIC: usize = 0;
    pub const OFF_FORMAT: usize = 4; // page-size/format field; one supported value
    pub const OFF_KIND: usize = 5;
    pub const OFF_PAGE_ID: usize = 8;
    pub const OFF_CREATED_TXID: usize = 16;
    pub const OFF_ENTRY_COUNT: usize = 24;
    pub const HEADER_LEN: usize = 32;
    /// Page format for 16K pages. **v2** (M6 Phase F): adds
    /// [`PageKind::Overflow`](super::PageKind::Overflow) and the leaf
    /// overflow-ref entry. A v1 page (M2 … M6 Phase E) is refused at decode
    /// rather than misread — the byte is bumped and old files are refused the
    /// day the kind ships, and Phase B already broke the data
    /// header the same way, so this is one whole-store format, not two.
    pub const FORMAT_16K: u8 = 2;
}

use header::HEADER_LEN;

/// Fixed leaf slot: `(key_off, key_len, val_off, val_len)`, all `u32`.
const LEAF_SLOT_LEN: usize = 16;

/// Bit 31 of a leaf slot's `val_len` marks the value as an **overflow ref**
/// rather than inline bytes. Free by construction: an inline value is bounded by
/// [`MAX_INLINE_VALUE_LEN`] (8 KiB), so a legal inline length never reaches
/// 2^31 — and a hostile one that sets the bit decodes as a ref and is validated
/// as a ref (exact 20-byte record, live head id, bounded `total_len`), never as
/// a 2 GiB inline slice.
const LEAF_VAL_OVERFLOW_FLAG: u32 = 1 << 31;

/// The fixed overflow-ref record stored in a leaf's heap in place of the value:
/// `(head_page_id u64, head_checksum u32, total_len u64)`.
pub const OVERFLOW_REF_LEN: usize = 20;

/// Overflow page layout: the standard header, then the successor reference and
/// this link's chunk length, then the payload.
mod ovf {
    pub const OFF_NEXT_ID: usize = 32; // u64 (0 == NONE == chain tail)
                                       // Only the NONE => 0 direction is a rule: crc32c can legitimately be zero, and
                                       // `PageRef` reserves no checksum value, so a live successor whose image
                                       // happens to hash to 0 must stay legal.
    pub const OFF_NEXT_CKSUM: usize = 40; // u32 (0 when next == NONE)
    pub const OFF_CHUNK_LEN: usize = 44; // u32
    pub const DATA_START: usize = 48;
}

/// Payload bytes one overflow page carries.
pub const fn overflow_payload_capacity() -> usize {
    PAGE_SIZE - ovf::DATA_START
}
/// Fixed branch child entry: `(child_page_id u64, child_checksum u32, pad u32)`.
const BRANCH_CHILD_LEN: usize = 16;
/// Fixed branch separator slot: `(key_off u32, key_len u32)`.
const BRANCH_SEP_LEN: usize = 8;

/// Parsed, validated body. Ranges index into `Page::bytes`; every range was
/// bounds- and order-checked at decode, so accessors never re-validate.
#[derive(Clone, Debug)]
enum Body {
    Leaf(Vec<LeafSlot>),
    Branch {
        children: Vec<PageRef>,
        separators: Vec<Range<usize>>,
    },
    /// One overflow-chain link: `next == None` is the tail.
    Overflow {
        next: Option<PageRef>,
        chunk: Range<usize>,
    },
}

#[derive(Clone, Debug)]
struct LeafSlot {
    key: Range<usize>,
    val: SlotValue,
}

/// A leaf slot's value: a validated heap range, or a decoded overflow ref.
#[derive(Clone, Debug)]
enum SlotValue {
    Inline(Range<usize>),
    Overflow(OverflowRef),
}

/// An immutable, in-memory page image plus its parsed body.
///
/// Once published a page is never mutated (invariant 2), so this holds an
/// `Arc<[u8]>` that reads share without copying, and a parsed `Body` whose
/// ranges were validated at decode.
#[derive(Clone, Debug)]
pub struct Page {
    id: PageId,
    kind: PageKind,
    created_txid: u64,
    /// crc32c of `bytes`, computed once at encode/decode so a cache-hit
    /// validation against a parent `PageRef` is O(1), not a re-scan.
    checksum: Checksum,
    bytes: std::sync::Arc<[u8]>,
    body: Body,
}

impl Page {
    pub fn id(&self) -> PageId {
        self.id
    }
    pub fn kind(&self) -> PageKind {
        self.kind
    }
    pub fn created_txid(&self) -> u64 {
        self.created_txid
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The checksum of this image under the v1 default algorithm. O(1): computed
    /// at encode/decode and cached.
    pub fn checksum(&self) -> Checksum {
        self.checksum
    }

    /// A `PageRef` naming this page and its current checksum.
    pub fn page_ref(&self) -> PageRef {
        PageRef {
            page_id: self.id,
            expected_checksum: self.checksum(),
        }
    }

    /// Number of entries: KV pairs for a leaf, children for a branch, `1` for an
    /// overflow link (its one chunk).
    pub fn entry_count(&self) -> usize {
        match &self.body {
            Body::Leaf(slots) => slots.len(),
            Body::Branch { children, .. } => children.len(),
            Body::Overflow { .. } => 1,
        }
    }

    // -- Leaf accessors (infallible; only valid to call when kind == Leaf) ----

    /// Resolves a slot's value against this image.
    fn slot_value(&self, v: &SlotValue) -> LeafValue<'_> {
        match v {
            SlotValue::Inline(r) => LeafValue::Inline(&self.bytes[r.clone()]),
            SlotValue::Overflow(o) => LeafValue::Overflow(*o),
        }
    }

    /// Looks up `key` in a leaf by binary search. `None` if absent or if this is
    /// not a leaf. An overflowed value resolves to its ref, not its bytes: the
    /// caller reads the chain ([`crate::read::read_overflow`]).
    pub fn leaf_get(&self, key: &[u8]) -> Option<LeafValue<'_>> {
        let Body::Leaf(slots) = &self.body else {
            return None;
        };
        let idx = slots
            .binary_search_by(|slot| self.bytes[slot.key.clone()].cmp(key))
            .ok()?;
        Some(self.slot_value(&slots[idx].val))
    }

    /// Iterates a leaf's `(key, value)` pairs in ascending key order. Empty for
    /// a branch or an overflow link. Double-ended: the descending scan
    /// ([`crate::read`]'s `tree_scan_range_rev`) walks it back-to-front.
    pub fn leaf_entries(&self) -> impl DoubleEndedIterator<Item = (&[u8], LeafValue<'_>)> {
        let slots: &[LeafSlot] = match &self.body {
            Body::Leaf(slots) => slots,
            _ => &[],
        };
        slots
            .iter()
            .map(move |slot| (&self.bytes[slot.key.clone()], self.slot_value(&slot.val)))
    }

    /// Every overflow chain head this leaf references, in entry order. Empty for
    /// any other kind. This is the reachability edge Phase F adds: a walker that
    /// ignores it silently drops every overflow page.
    pub fn leaf_overflow_refs(&self) -> impl Iterator<Item = OverflowRef> + '_ {
        let slots: &[LeafSlot] = match &self.body {
            Body::Leaf(slots) => slots,
            _ => &[],
        };
        slots.iter().filter_map(|slot| match &slot.val {
            SlotValue::Overflow(o) => Some(*o),
            SlotValue::Inline(_) => None,
        })
    }

    // -- Overflow accessors (only meaningful when kind == Overflow) -----------

    /// This link's successor, or `None` at the chain tail. `None` for any other
    /// kind — callers gate on [`kind`](Self::kind).
    pub fn overflow_next(&self) -> Option<PageRef> {
        match &self.body {
            Body::Overflow { next, .. } => *next,
            _ => None,
        }
    }

    /// This link's payload chunk. Empty for any other kind.
    pub fn overflow_chunk(&self) -> &[u8] {
        match &self.body {
            Body::Overflow { chunk, .. } => &self.bytes[chunk.clone()],
            _ => &[],
        }
    }

    // -- Branch accessors (infallible; only valid when kind == Branch) --------

    /// The child subtree that would contain `key`. Keys of `child[i]` are
    /// `< sep[i]`; `child[i+1]` holds keys `>= sep[i]`. Empty branch reference
    /// if this is not a branch.
    pub fn branch_child_for(&self, key: &[u8]) -> PageRef {
        let Body::Branch {
            children,
            separators,
        } = &self.body
        else {
            return PageRef::EMPTY;
        };
        // First separator strictly greater than `key` picks that child.
        let idx = separators
            .iter()
            .position(|sep| key < &self.bytes[sep.clone()])
            .unwrap_or(children.len() - 1);
        children[idx]
    }

    /// A branch's child references, in order. Empty for a leaf.
    pub fn branch_children(&self) -> &[PageRef] {
        match &self.body {
            Body::Branch { children, .. } => children,
            _ => &[],
        }
    }

    /// A branch's separator keys, in ascending order. Empty for a leaf.
    pub fn branch_separators(&self) -> impl Iterator<Item = &[u8]> {
        let seps: &[Range<usize>] = match &self.body {
            Body::Branch { separators, .. } => separators,
            _ => &[],
        };
        seps.iter().map(move |r| &self.bytes[r.clone()])
    }

    /// The smallest key reachable in a leaf (its first entry), for verify's
    /// cross-page order check. `None` for a branch or empty leaf.
    pub fn leaf_first_key(&self) -> Option<&[u8]> {
        match &self.body {
            Body::Leaf(slots) => slots.first().map(|s| &self.bytes[s.key.clone()]),
            _ => None,
        }
    }

    /// The largest key reachable in a leaf (its last entry). `None` for a branch
    /// or empty leaf.
    pub fn leaf_last_key(&self) -> Option<&[u8]> {
        match &self.body {
            Body::Leaf(slots) => slots.last().map(|s| &self.bytes[s.key.clone()]),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Decode: the single trust boundary.
// ---------------------------------------------------------------------------

impl Page {
    /// Decodes and fully validates a page image against the id and checksum its
    /// parent `PageRef` expects. Any structural defect is [`ReadError::Corrupt`]
    /// with a static reason.
    pub fn decode(bytes: std::sync::Arc<[u8]>, expected: PageRef) -> Result<Page, ReadError> {
        if bytes.len() != PAGE_SIZE {
            return Err(ReadError::Corrupt("page image wrong length"));
        }
        let checksum = Checksum::crc32c(&bytes);
        Self::decode_checked(bytes, expected, checksum)
    }

    /// Decode with the bytes' actual crc **already computed** by the caller, so the
    /// image is scanned for CRC only once. `actual` MUST equal `crc32c(bytes)` — it is
    /// used both as the checksum gate against `expected` and as the decoded page's
    /// stored checksum. The single-flight completer uses this: it computes `actual`
    /// once and decodes against `PageRef { page_id, expected_checksum: actual }`, so the
    /// decode is **reference-independent** (the gate passes by construction and only the
    /// structural checks can fail).
    pub(crate) fn decode_checked(
        bytes: std::sync::Arc<[u8]>,
        expected: PageRef,
        actual: Checksum,
    ) -> Result<Page, ReadError> {
        if bytes.len() != PAGE_SIZE {
            return Err(ReadError::Corrupt("page image wrong length"));
        }
        let checksum = actual;
        if checksum != expected.expected_checksum {
            return Err(ReadError::Corrupt("page checksum mismatch"));
        }
        let rd_u32 = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let rd_u64 = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        if rd_u32(header::OFF_MAGIC) != header::MAGIC {
            return Err(ReadError::Corrupt("bad page magic"));
        }
        if bytes[header::OFF_FORMAT] != header::FORMAT_16K {
            return Err(ReadError::Corrupt("unsupported page format"));
        }
        let kind = PageKind::from_u8(bytes[header::OFF_KIND])?;
        let id = PageId(rd_u64(header::OFF_PAGE_ID));
        if id != expected.page_id {
            return Err(ReadError::Corrupt("page id does not match reference"));
        }
        if id == PageId::NONE {
            return Err(ReadError::Corrupt("page id is the reserved NONE id"));
        }
        let created_txid = rd_u64(header::OFF_CREATED_TXID);
        let entry_count = rd_u32(header::OFF_ENTRY_COUNT) as usize;

        let body = match kind {
            PageKind::Leaf => decode_leaf(&bytes, entry_count)?,
            PageKind::Branch => decode_branch(&bytes, entry_count)?,
            PageKind::Overflow => decode_overflow(&bytes, entry_count)?,
        };
        Ok(Page {
            id,
            kind,
            created_txid,
            checksum,
            bytes,
            body,
        })
    }
}

/// Reads `u32` at `off` (bounds guaranteed by the caller's directory check).
fn rd_u32(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}
fn rd_u64(bytes: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
}

/// Validates a byte range lies within `[dir_end, PAGE_SIZE)` and enforces a max
/// length, returning it as a `usize` range. All arithmetic is on `u64` so a
/// hostile `u32` offset/length cannot wrap.
fn checked_range(
    off: u32,
    len: u32,
    dir_end: usize,
    max_len: usize,
    what: &'static str,
) -> Result<Range<usize>, ReadError> {
    if len as usize > max_len {
        return Err(ReadError::Corrupt(what));
    }
    let start = off as u64;
    let end = start + len as u64;
    if start < dir_end as u64 || end > PAGE_SIZE as u64 {
        return Err(ReadError::Corrupt(what));
    }
    Ok(off as usize..end as usize)
}

fn decode_leaf(bytes: &[u8], n: usize) -> Result<Body, ReadError> {
    // Directory of `n` fixed slots must fit after the header.
    let dir_end = HEADER_LEN
        .checked_add(n.checked_mul(LEAF_SLOT_LEN).ok_or(corrupt_dir())?)
        .ok_or(corrupt_dir())?;
    if dir_end > PAGE_SIZE {
        return Err(ReadError::Corrupt("leaf directory overruns page"));
    }
    let mut slots: Vec<LeafSlot> = Vec::with_capacity(n);
    let mut prev_key_end: Option<Range<usize>> = None;
    for i in 0..n {
        let base = HEADER_LEN + i * LEAF_SLOT_LEN;
        let key = checked_range(
            rd_u32(bytes, base),
            rd_u32(bytes, base + 4),
            dir_end,
            MAX_KEY_LEN,
            "leaf key out of bounds",
        )?;
        let raw_val_len = rd_u32(bytes, base + 12);
        let val = if raw_val_len & LEAF_VAL_OVERFLOW_FLAG != 0 {
            // An overflow slot's heap bytes are the fixed ref record — nothing
            // else is a legal length, so a crafted slot cannot borrow a longer
            // (or shorter) span under the flag.
            let stored = raw_val_len & !LEAF_VAL_OVERFLOW_FLAG;
            if stored as usize != OVERFLOW_REF_LEN {
                return Err(ReadError::Corrupt("leaf overflow ref has wrong length"));
            }
            let r = checked_range(
                rd_u32(bytes, base + 8),
                stored,
                dir_end,
                OVERFLOW_REF_LEN,
                "leaf overflow ref out of bounds",
            )?;
            SlotValue::Overflow(decode_overflow_ref(&bytes[r])?)
        } else {
            SlotValue::Inline(checked_range(
                rd_u32(bytes, base + 8),
                raw_val_len,
                dir_end,
                MAX_INLINE_VALUE_LEN,
                "leaf value out of bounds",
            )?)
        };
        if key.is_empty() {
            return Err(ReadError::Corrupt("leaf key is empty"));
        }
        if let Some(prev) = &prev_key_end {
            if bytes[prev.clone()] >= bytes[key.clone()] {
                return Err(ReadError::Corrupt("leaf keys not strictly ascending"));
            }
        }
        prev_key_end = Some(key.clone());
        slots.push(LeafSlot { key, val });
    }
    Ok(Body::Leaf(slots))
}

/// Decodes and validates a leaf's fixed overflow-ref record.
///
/// Deliberately **not** validated: that `total_len` exceeds
/// [`MAX_INLINE_VALUE_LEN`]. The spill threshold is writer *policy*; freezing it
/// into the format would make a later threshold change reject files written
/// under the old one. Any `total_len` a chain can legally represent is accepted
/// here, and the chain read checks it exactly against the bytes it collects.
fn decode_overflow_ref(rec: &[u8]) -> Result<OverflowRef, ReadError> {
    debug_assert_eq!(rec.len(), OVERFLOW_REF_LEN);
    let head_id = PageId(rd_u64(rec, 0));
    if head_id == PageId::NONE {
        return Err(ReadError::Corrupt("leaf overflow ref head is the NONE id"));
    }
    let total_len = rd_u64(rec, 12);
    if total_len == 0 || total_len > MAX_VALUE_LEN as u64 {
        return Err(ReadError::Corrupt("leaf overflow ref length out of range"));
    }
    Ok(OverflowRef {
        head: PageRef {
            page_id: head_id,
            expected_checksum: Checksum(rd_u32(rec, 8)),
        },
        total_len,
    })
}

/// Decodes one overflow-chain link. `entry_count` must be the canonical `1`
/// (its single chunk); the successor must be absent-or-live consistently with
/// its checksum, and the chunk must lie wholly in the payload region.
fn decode_overflow(bytes: &[u8], entry_count: usize) -> Result<Body, ReadError> {
    if entry_count != 1 {
        return Err(ReadError::Corrupt("overflow page entry count is not 1"));
    }
    let next_id = PageId(rd_u64(bytes, ovf::OFF_NEXT_ID));
    let next_cksum = rd_u32(bytes, ovf::OFF_NEXT_CKSUM);
    let next = if next_id == PageId::NONE {
        // Canonical tail: a stale checksum beside a NONE successor would mean two
        // encodings of one chain, and hints at a half-rewritten page.
        if next_cksum != 0 {
            return Err(ReadError::Corrupt(
                "overflow chain tail has a next checksum",
            ));
        }
        None
    } else {
        Some(PageRef {
            page_id: next_id,
            expected_checksum: Checksum(next_cksum),
        })
    };
    let chunk_len = rd_u32(bytes, ovf::OFF_CHUNK_LEN) as usize;
    if chunk_len == 0 || chunk_len > overflow_payload_capacity() {
        return Err(ReadError::Corrupt("overflow chunk length out of range"));
    }
    // A non-tail link must be full: chunking is deterministic (fill, then link),
    // so a short interior link cannot arise from a legal write and would let a
    // crafted chain claim a `total_len` its pages do not carry.
    if next.is_some() && chunk_len != overflow_payload_capacity() {
        return Err(ReadError::Corrupt("interior overflow link is not full"));
    }
    Ok(Body::Overflow {
        next,
        chunk: ovf::DATA_START..ovf::DATA_START + chunk_len,
    })
}

fn decode_branch(bytes: &[u8], n: usize) -> Result<Body, ReadError> {
    if n == 0 {
        return Err(ReadError::Corrupt("branch has no children"));
    }
    // Note: a one-child branch (`n == 1`, zero separators) is accepted
    // here even though the writer's `chunk_branch` rebalance never *emits*
    // one. It is a legal, if degenerate, structural shape — `branch_
    // child_for` returns child 0 — so the decode trust boundary tolerates it
    // rather than rejecting a file the reader can navigate correctly.
    let children_bytes = n.checked_mul(BRANCH_CHILD_LEN).ok_or(corrupt_dir())?;
    let sep_bytes = (n - 1).checked_mul(BRANCH_SEP_LEN).ok_or(corrupt_dir())?;
    let child_dir = HEADER_LEN;
    let sep_dir = child_dir.checked_add(children_bytes).ok_or(corrupt_dir())?;
    let dir_end = sep_dir.checked_add(sep_bytes).ok_or(corrupt_dir())?;
    if dir_end > PAGE_SIZE {
        return Err(ReadError::Corrupt("branch directory overruns page"));
    }
    let mut children = Vec::with_capacity(n);
    for i in 0..n {
        let base = child_dir + i * BRANCH_CHILD_LEN;
        let page_id = PageId(rd_u64(bytes, base));
        if page_id == PageId::NONE {
            return Err(ReadError::Corrupt("branch child ref is the NONE id"));
        }
        let checksum = Checksum(rd_u32(bytes, base + 8));
        children.push(PageRef {
            page_id,
            expected_checksum: checksum,
        });
    }
    let mut separators: Vec<Range<usize>> = Vec::with_capacity(n - 1);
    let mut prev: Option<Range<usize>> = None;
    for i in 0..n - 1 {
        let base = sep_dir + i * BRANCH_SEP_LEN;
        let sep = checked_range(
            rd_u32(bytes, base),
            rd_u32(bytes, base + 4),
            dir_end,
            MAX_KEY_LEN,
            "branch separator out of bounds",
        )?;
        if sep.is_empty() {
            return Err(ReadError::Corrupt("branch separator is empty"));
        }
        if let Some(p) = &prev {
            if bytes[p.clone()] >= bytes[sep.clone()] {
                return Err(ReadError::Corrupt(
                    "branch separators not strictly ascending",
                ));
            }
        }
        prev = Some(sep.clone());
        separators.push(sep);
    }
    Ok(Body::Branch {
        children,
        separators,
    })
}

fn corrupt_dir() -> ReadError {
    ReadError::Corrupt("page directory arithmetic overflow")
}

// ---------------------------------------------------------------------------
// Encoders (M2 builder fixture + M3 writer).
// ---------------------------------------------------------------------------

fn write_header(buf: &mut [u8], id: PageId, kind: PageKind, created_txid: u64, entry_count: u32) {
    buf[header::OFF_MAGIC..header::OFF_MAGIC + 4].copy_from_slice(&header::MAGIC.to_le_bytes());
    buf[header::OFF_FORMAT] = header::FORMAT_16K;
    buf[header::OFF_KIND] = kind as u8;
    buf[header::OFF_PAGE_ID..header::OFF_PAGE_ID + 8].copy_from_slice(&id.0.to_le_bytes());
    buf[header::OFF_CREATED_TXID..header::OFF_CREATED_TXID + 8]
        .copy_from_slice(&created_txid.to_le_bytes());
    buf[header::OFF_ENTRY_COUNT..header::OFF_ENTRY_COUNT + 4]
        .copy_from_slice(&entry_count.to_le_bytes());
}

/// Bytes a leaf entry consumes: one slot plus key+value heap. `val_len` is the
/// value's **stored** length ([`EntryValue::stored_len`]) — the value itself
/// inline, or [`OVERFLOW_REF_LEN`] for an overflowed one.
pub const fn leaf_entry_overhead(key_len: usize, val_len: usize) -> usize {
    LEAF_SLOT_LEN + key_len + val_len
}

/// Usable body bytes in a page (everything after the header).
pub const fn body_capacity() -> usize {
    PAGE_SIZE - HEADER_LEN
}

/// Whether a leaf of these entries fits in one page.
pub fn leaf_fits(entries: &[(Vec<u8>, EntryValue)]) -> bool {
    let used: usize = entries
        .iter()
        .map(|(k, v)| leaf_entry_overhead(k.len(), v.stored_len()))
        .sum();
    used <= body_capacity()
}

/// Encodes the fixed 20-byte overflow-ref record into a leaf's heap.
fn write_overflow_ref(buf: &mut [u8], at: usize, o: &OverflowRef) {
    buf[at..at + 8].copy_from_slice(&o.head.page_id.as_u64().to_le_bytes());
    buf[at + 8..at + 12].copy_from_slice(&o.head.expected_checksum.0.to_le_bytes());
    buf[at + 12..at + 20].copy_from_slice(&o.total_len.to_le_bytes());
}

/// Encodes a leaf page. `entries` must be strictly ascending by key and fit the
/// page; both are the builder's responsibility (asserted).
pub fn encode_leaf(id: PageId, created_txid: u64, entries: &[(Vec<u8>, EntryValue)]) -> Page {
    assert!(id != PageId::NONE, "leaf id must not be NONE");
    assert!(leaf_fits(entries), "leaf entries overflow page");
    let n = entries.len();
    let mut buf = vec![0u8; PAGE_SIZE];
    write_header(&mut buf, id, PageKind::Leaf, created_txid, n as u32);
    let dir_end = HEADER_LEN + n * LEAF_SLOT_LEN;
    let mut cursor = dir_end;
    for (i, (key, val)) in entries.iter().enumerate() {
        assert!(!key.is_empty(), "leaf key must be non-empty");
        if i > 0 {
            assert!(entries[i - 1].0 < *key, "leaf entries must be ascending");
        }
        let key_off = cursor;
        buf[cursor..cursor + key.len()].copy_from_slice(key);
        cursor += key.len();
        let val_off = cursor;
        let raw_val_len = match val {
            EntryValue::Inline(v) => {
                assert!(
                    v.len() <= MAX_INLINE_VALUE_LEN,
                    "inline value exceeds the inline cap"
                );
                buf[cursor..cursor + v.len()].copy_from_slice(v);
                cursor += v.len();
                v.len() as u32
            }
            EntryValue::Overflow(o) => {
                assert!(
                    o.head.page_id != PageId::NONE,
                    "overflow head must not be NONE"
                );
                assert!(
                    o.total_len > 0 && o.total_len <= MAX_VALUE_LEN as u64,
                    "overflow value length out of range"
                );
                write_overflow_ref(&mut buf, cursor, o);
                cursor += OVERFLOW_REF_LEN;
                OVERFLOW_REF_LEN as u32 | LEAF_VAL_OVERFLOW_FLAG
            }
        };
        let base = HEADER_LEN + i * LEAF_SLOT_LEN;
        buf[base..base + 4].copy_from_slice(&(key_off as u32).to_le_bytes());
        buf[base + 4..base + 8].copy_from_slice(&(key.len() as u32).to_le_bytes());
        buf[base + 8..base + 12].copy_from_slice(&(val_off as u32).to_le_bytes());
        buf[base + 12..base + 16].copy_from_slice(&raw_val_len.to_le_bytes());
    }
    finish(
        buf,
        id,
        PageKind::Leaf,
        created_txid,
        Body::Leaf(build_leaf_slots(entries, dir_end)),
    )
}

fn build_leaf_slots(entries: &[(Vec<u8>, EntryValue)], dir_end: usize) -> Vec<LeafSlot> {
    let mut cursor = dir_end;
    let mut slots = Vec::with_capacity(entries.len());
    for (key, val) in entries {
        let key_off = cursor;
        cursor += key.len();
        let val_off = cursor;
        cursor += val.stored_len();
        slots.push(LeafSlot {
            key: key_off..key_off + key.len(),
            val: match val {
                EntryValue::Inline(v) => SlotValue::Inline(val_off..val_off + v.len()),
                EntryValue::Overflow(o) => SlotValue::Overflow(*o),
            },
        });
    }
    slots
}

/// Encodes one overflow-chain link carrying `chunk`, linked to `next`
/// (`None` at the tail).
///
/// Chains are built **tail-first**: a link embeds its successor's id *and*
/// checksum, so the successor must exist before this page can be encoded. Every
/// walker that renumbers a chain (Direct compaction) must respect that order.
pub fn encode_overflow(id: PageId, created_txid: u64, chunk: &[u8], next: Option<PageRef>) -> Page {
    assert!(id != PageId::NONE, "overflow id must not be NONE");
    assert!(
        !chunk.is_empty() && chunk.len() <= overflow_payload_capacity(),
        "overflow chunk out of range"
    );
    assert!(
        next.is_none() || chunk.len() == overflow_payload_capacity(),
        "an interior overflow link must be full"
    );
    let mut buf = vec![0u8; PAGE_SIZE];
    write_header(&mut buf, id, PageKind::Overflow, created_txid, 1);
    if let Some(n) = next {
        assert!(n.page_id != PageId::NONE, "overflow next must not be NONE");
        buf[ovf::OFF_NEXT_ID..ovf::OFF_NEXT_ID + 8]
            .copy_from_slice(&n.page_id.as_u64().to_le_bytes());
        buf[ovf::OFF_NEXT_CKSUM..ovf::OFF_NEXT_CKSUM + 4]
            .copy_from_slice(&n.expected_checksum.0.to_le_bytes());
    }
    buf[ovf::OFF_CHUNK_LEN..ovf::OFF_CHUNK_LEN + 4]
        .copy_from_slice(&(chunk.len() as u32).to_le_bytes());
    buf[ovf::DATA_START..ovf::DATA_START + chunk.len()].copy_from_slice(chunk);
    finish(
        buf,
        id,
        PageKind::Overflow,
        created_txid,
        Body::Overflow {
            next,
            chunk: ovf::DATA_START..ovf::DATA_START + chunk.len(),
        },
    )
}

/// Splits `value` into the chunk sizes an overflow chain uses: fill each link to
/// capacity, the tail taking the remainder. Deterministic, and the rule
/// [`decode_overflow`]'s "interior links are full" check enforces.
pub fn overflow_chunks(value: &[u8]) -> std::slice::Chunks<'_, u8> {
    value.chunks(overflow_payload_capacity())
}

/// Bytes a branch entry consumes: one child entry, plus (for non-first
/// children) a separator slot and its key.
pub const fn branch_entry_overhead(sep_len: usize) -> usize {
    BRANCH_CHILD_LEN + BRANCH_SEP_LEN + sep_len
}

/// Bytes the first child of a branch consumes (a child slot, no separator).
pub const fn branch_first_overhead() -> usize {
    BRANCH_CHILD_LEN
}

/// Whether a branch of `n_children` children with total separator-key bytes
/// `sep_total` fits one page. Mirrors [`encode_branch`]'s layout check.
pub fn branch_fits(n_children: usize, sep_total: usize) -> bool {
    if n_children == 0 {
        return false;
    }
    let dir = n_children * BRANCH_CHILD_LEN + (n_children - 1) * BRANCH_SEP_LEN;
    dir + sep_total <= body_capacity()
}

/// Encodes a branch page. `children.len() == separators.len() + 1`, separators
/// strictly ascending, and the directories must fit the page (asserted).
pub fn encode_branch(
    id: PageId,
    created_txid: u64,
    children: &[PageRef],
    separators: &[Vec<u8>],
) -> Page {
    assert!(id != PageId::NONE, "branch id must not be NONE");
    assert!(!children.is_empty(), "branch needs at least one child");
    assert_eq!(
        children.len(),
        separators.len() + 1,
        "branch needs child count == separators + 1"
    );
    let n = children.len();
    let child_dir = HEADER_LEN;
    let sep_dir = child_dir + n * BRANCH_CHILD_LEN;
    let dir_end = sep_dir + (n - 1) * BRANCH_SEP_LEN;
    let heap_used: usize = separators.iter().map(|s| s.len()).sum();
    assert!(dir_end + heap_used <= PAGE_SIZE, "branch overflows page");

    let mut buf = vec![0u8; PAGE_SIZE];
    write_header(&mut buf, id, PageKind::Branch, created_txid, n as u32);
    for (i, child) in children.iter().enumerate() {
        assert!(
            child.page_id != PageId::NONE,
            "branch child must not be NONE"
        );
        let base = child_dir + i * BRANCH_CHILD_LEN;
        buf[base..base + 8].copy_from_slice(&child.page_id.0.to_le_bytes());
        buf[base + 8..base + 12].copy_from_slice(&child.expected_checksum.0.to_le_bytes());
    }
    let mut cursor = dir_end;
    let mut sep_ranges = Vec::with_capacity(separators.len());
    for (i, sep) in separators.iter().enumerate() {
        assert!(!sep.is_empty(), "branch separator must be non-empty");
        if i > 0 {
            assert!(separators[i - 1] < *sep, "separators must be ascending");
        }
        let off = cursor;
        buf[cursor..cursor + sep.len()].copy_from_slice(sep);
        cursor += sep.len();
        let base = sep_dir + i * BRANCH_SEP_LEN;
        buf[base..base + 4].copy_from_slice(&(off as u32).to_le_bytes());
        buf[base + 4..base + 8].copy_from_slice(&(sep.len() as u32).to_le_bytes());
        sep_ranges.push(off..off + sep.len());
    }
    finish(
        buf,
        id,
        PageKind::Branch,
        created_txid,
        Body::Branch {
            children: children.to_vec(),
            separators: sep_ranges,
        },
    )
}

/// Test/fixture convenience: encode a leaf from plain inline `(key, value)`
/// pairs. Panics if any value exceeds [`MAX_INLINE_VALUE_LEN`] — spilling is the
/// writer's job ([`crate::btree`]), not this helper's.
#[cfg(test)]
pub(crate) fn encode_leaf_inline(
    id: PageId,
    created_txid: u64,
    entries: &[(Vec<u8>, Vec<u8>)],
) -> Page {
    let owned: Vec<(Vec<u8>, EntryValue)> = entries
        .iter()
        .map(|(k, v)| (k.clone(), EntryValue::Inline(v.clone())))
        .collect();
    encode_leaf(id, created_txid, &owned)
}

fn finish(buf: Vec<u8>, id: PageId, kind: PageKind, created_txid: u64, body: Body) -> Page {
    let checksum = Checksum::crc32c(&buf);
    Page {
        id,
        kind,
        created_txid,
        checksum,
        bytes: buf.into(),
        body,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(k: &[u8], v: &[u8]) -> (Vec<u8>, Vec<u8>) {
        (k.to_vec(), v.to_vec())
    }

    /// Phase-H decision:
    /// heap aliasing is **explicitly blessed**, uniformly for leaf and branch
    /// heap objects. Two slots pointing at the same heap bytes are a
    /// canonicalisation question, not a safety one — the page is immutable, so
    /// aliased ranges cannot disagree, every range is still individually
    /// bounds- and length-checked, and validating the whole interval set would
    /// tax every page load to reject images no reader misreads. This test pins
    /// the decision: a crafted leaf whose two value slots alias one heap range
    /// decodes, and both entries read the shared bytes.
    #[test]
    fn aliased_heap_ranges_are_blessed() {
        let page = encode_leaf_inline(PageId(9), 0, &[e(b"a", b"vv"), e(b"b", b"xx")]);
        let mut bytes = page.bytes().to_vec();
        // Fixed leaf slot: `(key_off, key_len, val_off, val_len)` u32 LE, slot
        // `i` at `HEADER_LEN + i·16`. Point slot 1's value range at slot 0's.
        let slot0_val_off: [u8; 4] = bytes[HEADER_LEN + 8..HEADER_LEN + 12].try_into().unwrap();
        bytes[HEADER_LEN + 16 + 8..HEADER_LEN + 16 + 12].copy_from_slice(&slot0_val_off);
        let expected = PageRef {
            page_id: PageId(9),
            expected_checksum: Checksum::crc32c(&bytes),
        };
        let decoded = Page::decode(bytes.into(), expected).expect("aliasing must decode");
        assert_eq!(decoded.leaf_get(b"a"), Some(LeafValue::Inline(&b"vv"[..])));
        assert_eq!(
            decoded.leaf_get(b"b"),
            Some(LeafValue::Inline(&b"vv"[..])),
            "the aliased slot reads the shared heap bytes"
        );
    }

    #[test]
    fn leaf_roundtrip_and_lookup() {
        let entries = vec![e(b"alpha", b"1"), e(b"beta", b"22"), e(b"gamma", b"333")];
        let page = encode_leaf_inline(PageId(42), 7, &entries);
        let decoded = Page::decode(page.bytes().to_vec().into(), page.page_ref()).unwrap();
        assert_eq!(decoded.id(), PageId(42));
        assert_eq!(decoded.kind(), PageKind::Leaf);
        assert_eq!(decoded.entry_count(), 3);
        assert_eq!(
            decoded.leaf_get(b"beta"),
            Some(LeafValue::Inline(&b"22"[..]))
        );
        assert_eq!(decoded.leaf_get(b"missing"), None);
        assert_eq!(decoded.leaf_first_key(), Some(&b"alpha"[..]));
        assert_eq!(decoded.leaf_last_key(), Some(&b"gamma"[..]));
        let collected: Vec<_> = decoded.leaf_entries().map(|(k, _)| k.to_vec()).collect();
        assert_eq!(
            collected,
            vec![b"alpha".to_vec(), b"beta".to_vec(), b"gamma".to_vec()]
        );
    }

    #[test]
    fn branch_descent() {
        let c0 = PageRef {
            page_id: PageId(10),
            expected_checksum: Checksum(1),
        };
        let c1 = PageRef {
            page_id: PageId(11),
            expected_checksum: Checksum(2),
        };
        let c2 = PageRef {
            page_id: PageId(12),
            expected_checksum: Checksum(3),
        };
        let page = encode_branch(PageId(2), 5, &[c0, c1, c2], &[b"m".to_vec(), b"t".to_vec()]);
        let d = Page::decode(page.bytes().to_vec().into(), page.page_ref()).unwrap();
        assert_eq!(d.kind(), PageKind::Branch);
        assert_eq!(d.branch_child_for(b"a").page_id, PageId(10));
        assert_eq!(d.branch_child_for(b"m").page_id, PageId(11)); // >= sep
        assert_eq!(d.branch_child_for(b"s").page_id, PageId(11));
        assert_eq!(d.branch_child_for(b"t").page_id, PageId(12));
        assert_eq!(d.branch_child_for(b"z").page_id, PageId(12));
    }

    #[test]
    fn checksum_mismatch_is_corrupt() {
        let page = encode_leaf_inline(PageId(1), 0, &[e(b"x", b"y")]);
        let mut bad = page.bytes().to_vec();
        bad[HEADER_LEN + 40] ^= 0xFF;
        let err = Page::decode(bad.into(), page.page_ref()).unwrap_err();
        assert!(matches!(err, ReadError::Corrupt(_)));
    }

    #[test]
    fn wrong_id_is_corrupt() {
        let page = encode_leaf_inline(PageId(1), 0, &[e(b"x", b"y")]);
        let wrong = PageRef {
            page_id: PageId(2),
            expected_checksum: page.checksum(),
        };
        assert!(matches!(
            Page::decode(page.bytes().to_vec().into(), wrong),
            Err(ReadError::Corrupt(_))
        ));
    }

    #[test]
    fn corrupt_key_order_is_rejected() {
        // Hand-build a leaf whose slots are out of order by swapping two keys'
        // slot entries after encoding.
        let entries = vec![e(b"a", b"1"), e(b"b", b"2")];
        let page = encode_leaf_inline(PageId(3), 0, &entries);
        let mut bytes = page.bytes().to_vec();
        // Swap the two 16-byte slots so slot 0 points at key "b" and slot 1 at "a".
        let (s0, s1) = (HEADER_LEN, HEADER_LEN + LEAF_SLOT_LEN);
        for k in 0..LEAF_SLOT_LEN {
            bytes.swap(s0 + k, s1 + k);
        }
        // Fix the checksum so only the order defect remains.
        let ck = Checksum::crc32c(&bytes);
        let r = PageRef {
            page_id: PageId(3),
            expected_checksum: ck,
        };
        let err = Page::decode(bytes.into(), r).unwrap_err();
        assert!(matches!(
            err,
            ReadError::Corrupt("leaf keys not strictly ascending")
        ));
    }

    #[test]
    fn corrupt_length_out_of_bounds_is_rejected() {
        let page = encode_leaf_inline(PageId(4), 0, &[e(b"k", b"v")]);
        let mut bytes = page.bytes().to_vec();
        // Blow up slot 0's key_len to span past the page.
        bytes[HEADER_LEN + 4..HEADER_LEN + 8].copy_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
        let ck = Checksum::crc32c(&bytes);
        let r = PageRef {
            page_id: PageId(4),
            expected_checksum: ck,
        };
        assert!(matches!(
            Page::decode(bytes.into(), r),
            Err(ReadError::Corrupt(_))
        ));
    }

    // -- M6 Phase F: overflow ------------------------------------------------

    fn ovf_ref(head: &Page, total_len: u64) -> OverflowRef {
        OverflowRef {
            head: head.page_ref(),
            total_len,
        }
    }

    #[test]
    fn overflow_page_roundtrips() {
        let cap = overflow_payload_capacity();
        let tail = encode_overflow(PageId(2), 4, &[7u8; 100], None);
        let d = Page::decode(tail.bytes().to_vec().into(), tail.page_ref()).unwrap();
        assert_eq!(d.kind(), PageKind::Overflow);
        assert_eq!(d.overflow_chunk(), &[7u8; 100][..]);
        assert_eq!(d.overflow_next(), None);

        let head = encode_overflow(PageId(1), 4, &vec![9u8; cap], Some(tail.page_ref()));
        let d = Page::decode(head.bytes().to_vec().into(), head.page_ref()).unwrap();
        assert_eq!(d.overflow_next(), Some(tail.page_ref()));
        assert_eq!(d.overflow_chunk().len(), cap);
    }

    #[test]
    fn leaf_overflow_ref_roundtrips_and_is_reachable() {
        let chain = encode_overflow(PageId(9), 0, &[1u8; 50], None);
        let o = ovf_ref(&chain, 50);
        let leaf = encode_leaf(
            PageId(3),
            0,
            &[
                (b"a".to_vec(), EntryValue::Inline(b"inline".to_vec())),
                (b"b".to_vec(), EntryValue::Overflow(o)),
            ],
        );
        let d = Page::decode(leaf.bytes().to_vec().into(), leaf.page_ref()).unwrap();
        assert_eq!(d.leaf_get(b"a"), Some(LeafValue::Inline(&b"inline"[..])));
        assert_eq!(d.leaf_get(b"b"), Some(LeafValue::Overflow(o)));
        // The reachability edge the walkers follow.
        assert_eq!(d.leaf_overflow_refs().collect::<Vec<_>>(), vec![o]);
    }

    /// The defect no *local* check can catch:
    /// a leaf declaring a 1 MiB value pointing at a checksum-valid one-byte tail.
    /// Every page here decodes perfectly; only the cross-page cursor rejects it.
    #[test]
    fn a_chain_shorter_than_its_declared_length_is_corrupt() {
        let tail = encode_overflow(PageId(9), 0, &[1u8; 1], None);
        let o = ovf_ref(&tail, MAX_VALUE_LEN as u64);
        let leaf = encode_leaf(PageId(3), 0, &[(b"k".to_vec(), EntryValue::Overflow(o))]);
        // Both images are individually valid...
        Page::decode(leaf.bytes().to_vec().into(), leaf.page_ref()).unwrap();
        let tail_page = Page::decode(tail.bytes().to_vec().into(), tail.page_ref()).unwrap();
        // ...and the cursor is what refuses the pair.
        let mut c = OverflowCursor::new(o);
        assert_eq!(c.pending(), Some(tail.page_ref()));
        assert!(matches!(c.accept(&tail_page), Err(ReadError::Corrupt(_))));
    }

    #[test]
    fn a_chain_longer_than_its_declared_length_is_corrupt() {
        let cap = overflow_payload_capacity();
        let tail = encode_overflow(PageId(9), 0, &[1u8; 10], None);
        let head = encode_overflow(PageId(8), 0, &vec![2u8; cap], Some(tail.page_ref()));
        // Declare only one link's worth, but the head has a successor.
        let mut c = OverflowCursor::new(ovf_ref(&head, cap as u64));
        let head_page = Page::decode(head.bytes().to_vec().into(), head.page_ref()).unwrap();
        assert!(matches!(c.accept(&head_page), Err(ReadError::Corrupt(_))));
    }

    /// A cycle needs no dedicated guard: `remaining` falls a full page per interior
    /// link, so a loop must reach a link that has to be the tail and is not.
    #[test]
    fn a_cyclic_chain_terminates_as_corrupt() {
        let cap = overflow_payload_capacity();
        // A link whose successor is itself (a checksum-consistent 1-cycle is not
        // constructible, so point it at a second full link and walk the budget out).
        let b = encode_overflow(PageId(8), 0, &vec![2u8; cap], None);
        let a = encode_overflow(PageId(7), 0, &vec![1u8; cap], Some(b.page_ref()));
        let a_page = Page::decode(a.bytes().to_vec().into(), a.page_ref()).unwrap();
        let b_page = Page::decode(b.bytes().to_vec().into(), b.page_ref()).unwrap();
        // Declare 2 full links, then keep feeding links: the budget runs out.
        let mut c = OverflowCursor::new(ovf_ref(&a, 2 * cap as u64));
        c.accept(&a_page).unwrap();
        // `b` is a tail carrying `cap`, and `cap` is exactly what remains: legal.
        c.accept(&b_page).unwrap();
        assert_eq!(c.pending(), None);
        // But re-feeding `a` (an interior link) once nothing remains is corrupt.
        let mut c = OverflowCursor::new(ovf_ref(&a, cap as u64));
        assert!(matches!(c.accept(&a_page), Err(ReadError::Corrupt(_))));
    }

    #[test]
    fn interior_overflow_link_must_be_full() {
        // A short interior link would let a crafted chain claim a `total_len` its
        // pages do not carry, so the format forbids it.
        let tail = encode_overflow(PageId(9), 0, &[1u8; 5], None);
        let mut bytes = encode_overflow(
            PageId(8),
            0,
            &vec![2u8; overflow_payload_capacity()],
            Some(tail.page_ref()),
        )
        .bytes()
        .to_vec();
        // Shrink the interior link's chunk and re-checksum.
        bytes[ovf::OFF_CHUNK_LEN..ovf::OFF_CHUNK_LEN + 4].copy_from_slice(&100u32.to_le_bytes());
        let ck = Checksum::crc32c(&bytes);
        let r = PageRef {
            page_id: PageId(8),
            expected_checksum: ck,
        };
        assert!(matches!(
            Page::decode(bytes.into(), r),
            Err(ReadError::Corrupt("interior overflow link is not full"))
        ));
    }

    #[test]
    fn corrupt_overflow_ref_fields_are_rejected() {
        let chain = encode_overflow(PageId(9), 0, &[1u8; 50], None);
        let leaf = encode_leaf(
            PageId(3),
            0,
            &[(b"k".to_vec(), EntryValue::Overflow(ovf_ref(&chain, 50)))],
        );
        // The ref record sits right after the key in the heap.
        let rec = HEADER_LEN + LEAF_SLOT_LEN + 1;
        // A NONE head.
        let mut bytes = leaf.bytes().to_vec();
        bytes[rec..rec + 8].copy_from_slice(&0u64.to_le_bytes());
        let ck = Checksum::crc32c(&bytes);
        assert!(matches!(
            Page::decode(
                bytes.into(),
                PageRef {
                    page_id: PageId(3),
                    expected_checksum: ck
                }
            ),
            Err(ReadError::Corrupt("leaf overflow ref head is the NONE id"))
        ));
        // A total_len past the cap.
        let mut bytes = leaf.bytes().to_vec();
        bytes[rec + 12..rec + 20].copy_from_slice(&(MAX_VALUE_LEN as u64 + 1).to_le_bytes());
        let ck = Checksum::crc32c(&bytes);
        assert!(matches!(
            Page::decode(
                bytes.into(),
                PageRef {
                    page_id: PageId(3),
                    expected_checksum: ck
                }
            ),
            Err(ReadError::Corrupt("leaf overflow ref length out of range"))
        ));
        // A zero total_len.
        let mut bytes = leaf.bytes().to_vec();
        bytes[rec + 12..rec + 20].copy_from_slice(&0u64.to_le_bytes());
        let ck = Checksum::crc32c(&bytes);
        assert!(matches!(
            Page::decode(
                bytes.into(),
                PageRef {
                    page_id: PageId(3),
                    expected_checksum: ck
                }
            ),
            Err(ReadError::Corrupt("leaf overflow ref length out of range"))
        ));
        // The overflow flag set over a non-20-byte length: decoded as a ref and
        // refused as a ref, never as a 2 GiB inline slice.
        let mut bytes = leaf.bytes().to_vec();
        let slot_val_len = HEADER_LEN + 12;
        bytes[slot_val_len..slot_val_len + 4]
            .copy_from_slice(&(LEAF_VAL_OVERFLOW_FLAG | 40).to_le_bytes());
        let ck = Checksum::crc32c(&bytes);
        assert!(matches!(
            Page::decode(
                bytes.into(),
                PageRef {
                    page_id: PageId(3),
                    expected_checksum: ck
                }
            ),
            Err(ReadError::Corrupt("leaf overflow ref has wrong length"))
        ));
    }

    /// crc32c can legitimately be zero, and `PageRef` reserves no checksum value,
    /// so only the NONE => 0 direction is a rule.
    /// The cursor must not take "this is the page I asked for" on faith: not every
    /// lookup verifies (the writer's private overlay resolves by id alone), so the
    /// trust boundary checks its own invariant.
    #[test]
    fn the_cursor_rejects_a_page_that_is_not_the_link_it_asked_for() {
        let real = encode_overflow(PageId(9), 0, &[1u8; 50], None);
        let oref = OverflowRef {
            head: real.page_ref(),
            total_len: 50,
        };

        // An imposter with the same id but a different image — what an overlay
        // lookup keyed on id alone would hand back.
        let same_id = encode_overflow(PageId(9), 0, &[2u8; 50], None);
        let mut c = OverflowCursor::new(oref);
        assert!(matches!(
            c.accept(&same_id),
            Err(ReadError::Corrupt(
                "overflow link is not the page its ref named"
            ))
        ));

        // And an entirely different link that is *itself* perfectly valid, offered
        // with its own honest ref. The cursor checks against the link **it** is
        // pending, not against anything the caller supplies, so there is no
        // argument that talks it into accepting the wrong page.
        let unrelated = encode_overflow(PageId(77), 0, &[1u8; 50], None);
        let mut c = OverflowCursor::new(oref);
        assert!(matches!(
            c.accept(&unrelated),
            Err(ReadError::Corrupt(
                "overflow link is not the page its ref named"
            ))
        ));

        // The genuine page is accepted, and the chain then refuses to continue.
        let mut c = OverflowCursor::new(oref);
        c.accept(&real).unwrap();
        assert_eq!(c.pending(), None);
        assert!(matches!(
            c.accept(&real),
            Err(ReadError::Corrupt("overflow chain already complete"))
        ));
    }

    #[test]
    fn a_live_successor_may_have_a_zero_checksum() {
        let cap = overflow_payload_capacity();
        let next = PageRef {
            page_id: PageId(9),
            expected_checksum: Checksum(0),
        };
        let head = encode_overflow(PageId(8), 0, &vec![3u8; cap], Some(next));
        let d = Page::decode(head.bytes().to_vec().into(), head.page_ref()).unwrap();
        assert_eq!(d.overflow_next(), Some(next));
    }

    #[test]
    fn a_tail_with_a_next_checksum_is_rejected() {
        let mut bytes = encode_overflow(PageId(8), 0, &[1u8; 10], None)
            .bytes()
            .to_vec();
        bytes[ovf::OFF_NEXT_CKSUM..ovf::OFF_NEXT_CKSUM + 4].copy_from_slice(&7u32.to_le_bytes());
        let ck = Checksum::crc32c(&bytes);
        assert!(matches!(
            Page::decode(
                bytes.into(),
                PageRef {
                    page_id: PageId(8),
                    expected_checksum: ck
                }
            ),
            Err(ReadError::Corrupt(
                "overflow chain tail has a next checksum"
            ))
        ));
    }

    #[test]
    fn corrupt_none_child_ref_is_rejected() {
        let c0 = PageRef {
            page_id: PageId(10),
            expected_checksum: Checksum(1),
        };
        let c1 = PageRef {
            page_id: PageId(11),
            expected_checksum: Checksum(2),
        };
        let page = encode_branch(PageId(5), 0, &[c0, c1], &[b"m".to_vec()]);
        let mut bytes = page.bytes().to_vec();
        // Zero child 0's page id -> NONE.
        bytes[HEADER_LEN..HEADER_LEN + 8].copy_from_slice(&0u64.to_le_bytes());
        let ck = Checksum::crc32c(&bytes);
        let r = PageRef {
            page_id: PageId(5),
            expected_checksum: ck,
        };
        assert!(matches!(
            Page::decode(bytes.into(), r),
            Err(ReadError::Corrupt("branch child ref is the NONE id"))
        ));
    }
}
