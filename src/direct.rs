//! StoreDirect physical layout: header codec, the formula page locator, the
//! append-only allocator, and the flush coordinator.
//!
//! Layout:
//!
//! ```text
//! page 0        direct header (single active slot in v1)
//! page 1        reserved/zeroed superblock slot B (future alternating header)
//! page 2..N     data pages, allocated append-only
//! ```
//!
//! Key decision: in an append-only CoW file the page **index**
//! (`offset / PAGE_SIZE`) is itself globally unique and never reused, so Direct
//! uses the index *as* the [`PageId`]. The locator is then the pure formula
//! `offset = id × PAGE_SIZE` — no id→offset table is persisted, which is exactly
//! what makes reopen need nothing but the header (root + logical tail). The
//! `PageLocator` abstraction is unbroken: the read path still resolves through
//! `locate()`; only *this* locator's body is a formula (the M4 WAL overlay and a
//! future compaction generation carry real tables).

use crate::backend::{FileGeneration, Location, Member, PageLocator};
use crate::error::{OpenError, PoisonReason, Terminal};
use crate::io::PageIo;
use crate::page::{Checksum, Page, PageId, PageRef, PAGE_SIZE};
use crate::store::Core;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Reserved pages before the first data page: the header and slot B.
pub const FIRST_DATA_INDEX: u64 = 2;

/// Flush once this many batches accumulate unflushed, regardless of their page
/// count. Bounds the pending ledger under a stream of zero-page batches that
/// never trip the page-based high-water mark.
const PENDING_FLUSH_CAP: usize = 256;

/// Max page writes a flush keeps concurrently in flight (M7-B fan-out).
/// Bounds transient memory — up to `CAP` original 16 KiB page copies plus, under
/// O_DIRECT, `CAP` aligned bounce buffers (~32 KiB/op ⇒ ~2 MiB at 64) — and the
/// submission-channel backlog; it is *not* a ring-slot bound (the uring driver
/// already caps ring admission below its 256-entry ring, and both submission
/// channels are unbounded). Kept below the driver's normal-request cap so the
/// reserved priority slots (`datasync`) stay free. A tuning start for M7-E, which
/// sweeps 8/16/32/64/128 on both backends; deliberately a fixed constant, not
/// derived from ring depth / core count.
const FLUSH_FANOUT_CAP: usize = 64;

// -- Header codec -----------------------------------------------------------

mod hdr {
    pub const MAGIC: u32 = 0x4D50_3544; // "MP5D" (Direct)
    /// v3 (M6 Phase F). v2 added `id_delta` (Phase D); v3's *layout* is
    /// identical, and it exists purely to gate the **page** format: Phase F adds
    /// `PageKind::Overflow` and the leaf overflow-ref, so a pre-F binary must not
    /// open a post-F store and vice versa.
    ///
    /// The page format byte alone cannot do that job. Nothing at open decodes
    /// every page, so a pre-F store would open clean under a post-F binary and
    /// only *poison* later, when a read hit a v1 page — a format break has to be
    /// refused at the door, not discovered by the writer. The header is the
    /// store-level marker, so the header is what moves. Older versions are
    /// refused with a clear error, the same ship-day policy as v2.
    ///
    /// v4 (T1.1): the **double-slot** format. Layout of one header is
    /// unchanged; what changes is the protocol — both header pages hold live
    /// alternating slots (winner by `header_txid` at open) instead of one active
    /// page 0 + a zeroed reserve. The bump makes the change mutually refused:
    /// a v3 binary reading page 0 of a v4 store sees format 4 and refuses
    /// ("newer"), so it can never open a stale page-0 state and allocate from a
    /// stale tail after a v4 binary committed a newer page-1 state (the T1.1
    /// downgrade hazard).
    pub const FORMAT_V4: u8 = 4;
    pub const ALGO_CRC32C: u8 = 1;

    pub const OFF_MAGIC: usize = 0; // u32
    pub const OFF_FORMAT: usize = 4; // u8
    pub const OFF_ALGO: usize = 5; // u8
    pub const OFF_PAGE_SIZE: usize = 8; // u32
    pub const OFF_STORE_UUID: usize = 16; // 16 bytes
    pub const OFF_ROOT_ID: usize = 32; // u64
    pub const OFF_ROOT_CKSUM: usize = 40; // u32
    pub const OFF_LOGICAL_TAIL: usize = 48; // u64 (next free byte offset)
    pub const OFF_ENTRY_COUNT: usize = 56; // u64
    pub const OFF_HEADER_TXID: usize = 64; // u64
    pub const OFF_ID_DELTA: usize = 72; // u64 (page id − delta = page index)
    pub const OFF_HEADER_CKSUM: usize = 80; // u32 crc32c over [0, 80)
}

/// `id_delta` must stay strictly below this bound — the same headroom rule as
/// the WAL record frontier (`crate::wal::MAX_FRONTIER`), so `(id − delta)`
/// arithmetic driven by a crafted header can never go wild. A real store cannot
/// approach it (2^48 pages ever allocated).
pub const MAX_ID_DELTA: u64 = 1 << 48;

/// The decoded direct header.
#[derive(Clone, Copy, Debug)]
pub struct DirectHeader {
    pub store_uuid: [u8; 16],
    pub root: PageRef,
    /// Next free byte offset (allocation frontier). **Physical**: the file's
    /// dense tail, regardless of `id_delta`.
    pub logical_tail: u64,
    pub entry_count: u64,
    pub header_txid: u64,
    /// Page-id renumbering base (M6 Phase D): a page id maps to file index
    /// `id − id_delta`, so `offset = (id − id_delta) × PAGE_SIZE`. `0` until the
    /// first compaction; each compaction renumbers the live set onto **fresh**
    /// ids (ids are never reused — the shared page cache is keyed by id) and
    /// raises the delta so the file stays dense.
    pub id_delta: u64,
}

impl DirectHeader {
    /// Encodes the header into a full zero-padded page-0 image.
    pub fn encode(&self) -> Box<[u8]> {
        let mut buf = vec![0u8; PAGE_SIZE];
        buf[hdr::OFF_MAGIC..hdr::OFF_MAGIC + 4].copy_from_slice(&hdr::MAGIC.to_le_bytes());
        buf[hdr::OFF_FORMAT] = hdr::FORMAT_V4;
        buf[hdr::OFF_ALGO] = hdr::ALGO_CRC32C;
        buf[hdr::OFF_PAGE_SIZE..hdr::OFF_PAGE_SIZE + 4]
            .copy_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
        buf[hdr::OFF_STORE_UUID..hdr::OFF_STORE_UUID + 16].copy_from_slice(&self.store_uuid);
        buf[hdr::OFF_ROOT_ID..hdr::OFF_ROOT_ID + 8]
            .copy_from_slice(&self.root.page_id.as_u64().to_le_bytes());
        buf[hdr::OFF_ROOT_CKSUM..hdr::OFF_ROOT_CKSUM + 4]
            .copy_from_slice(&self.root.expected_checksum.0.to_le_bytes());
        buf[hdr::OFF_LOGICAL_TAIL..hdr::OFF_LOGICAL_TAIL + 8]
            .copy_from_slice(&self.logical_tail.to_le_bytes());
        buf[hdr::OFF_ENTRY_COUNT..hdr::OFF_ENTRY_COUNT + 8]
            .copy_from_slice(&self.entry_count.to_le_bytes());
        buf[hdr::OFF_HEADER_TXID..hdr::OFF_HEADER_TXID + 8]
            .copy_from_slice(&self.header_txid.to_le_bytes());
        buf[hdr::OFF_ID_DELTA..hdr::OFF_ID_DELTA + 8].copy_from_slice(&self.id_delta.to_le_bytes());
        let ck = crc32c::crc32c(&buf[..hdr::OFF_HEADER_CKSUM]);
        buf[hdr::OFF_HEADER_CKSUM..hdr::OFF_HEADER_CKSUM + 4].copy_from_slice(&ck.to_le_bytes());
        buf.into_boxed_slice()
    }

    /// Decodes and validates a page-0 image. Any defect is `Corrupt` — never
    /// heuristic repair.
    pub fn decode(bytes: &[u8]) -> Result<DirectHeader, OpenError> {
        if bytes.len() < PAGE_SIZE {
            return Err(OpenError::Corrupt("direct header image too short"));
        }
        let rd_u32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let rd_u64 = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        if rd_u32(hdr::OFF_MAGIC) != hdr::MAGIC {
            return Err(OpenError::Corrupt("bad direct header magic"));
        }
        // Checksum BEFORE the format byte (T1.1): under the two-slot protocol a
        // torn slot is routine, and a garbled format byte behind an intact magic
        // must read as slot damage (`Corrupt` → the other slot is consulted),
        // not as a cross-version store. The crc offset is layout-stable across
        // v2/v3/v4, so a genuine older header still verifies and reaches the
        // format refusal below with its clear older/newer verdict.
        let stored_ck = rd_u32(hdr::OFF_HEADER_CKSUM);
        if crc32c::crc32c(&bytes[..hdr::OFF_HEADER_CKSUM]) != stored_ck {
            return Err(OpenError::Corrupt("direct header checksum mismatch"));
        }
        // A v1/v2 header (older) or a future release's header (newer) reaches here
        // — refuse clearly, at open, before any mutation, rather than poisoning on
        // the first page a later read cannot decode (M6 Phase H forward policy).
        OpenError::check_open_format("direct header", bytes[hdr::OFF_FORMAT], hdr::FORMAT_V4)?;
        if bytes[hdr::OFF_ALGO] != hdr::ALGO_CRC32C {
            return Err(OpenError::Corrupt("unsupported checksum algorithm"));
        }
        if rd_u32(hdr::OFF_PAGE_SIZE) != PAGE_SIZE as u32 {
            return Err(OpenError::Corrupt("unsupported page size"));
        }
        let mut store_uuid = [0u8; 16];
        store_uuid.copy_from_slice(&bytes[hdr::OFF_STORE_UUID..hdr::OFF_STORE_UUID + 16]);
        let root_id = rd_u64(hdr::OFF_ROOT_ID);
        let root = PageRef {
            page_id: PageId(root_id),
            expected_checksum: Checksum(rd_u32(hdr::OFF_ROOT_CKSUM)),
        };
        let logical_tail = rd_u64(hdr::OFF_LOGICAL_TAIL);
        if logical_tail < FIRST_DATA_INDEX * PAGE_SIZE as u64
            || logical_tail % PAGE_SIZE as u64 != 0
        {
            return Err(OpenError::Corrupt("direct header logical tail misaligned"));
        }
        // Require a page of headroom so the allocator can advance at least once
        // without saturating `next_index × PAGE_SIZE` to a misaligned `u64::MAX`.
        // An exhausted frontier is a `Corrupt` header, per the M3 contract,
        // rather than a runtime edge.
        if logical_tail > u64::MAX - PAGE_SIZE as u64 {
            return Err(OpenError::Corrupt(
                "direct header logical tail at frontier limit",
            ));
        }
        // The id delta is bounded like the WAL frontier: as strict as anything
        // recovery will later compute from it, so `(id − delta)`/`(delta + index)`
        // arithmetic can never be driven wild by a crafted header.
        let id_delta = rd_u64(hdr::OFF_ID_DELTA);
        if id_delta >= MAX_ID_DELTA {
            return Err(OpenError::Corrupt("direct header id delta out of range"));
        }
        let entry_count = rd_u64(hdr::OFF_ENTRY_COUNT);
        // Field-*relationship* validation: a checksum-valid but
        // internally contradictory header must not open, because the count and
        // root emptiness both drive mutation semantics — a non-empty root with
        // count 0 would read as empty and the first no-growth write would discard
        // the whole tree.
        if root.is_empty() {
            // Canonical empty: NONE id, zero checksum, zero entries.
            if root.expected_checksum != Checksum(0) {
                return Err(OpenError::Corrupt("empty root with non-zero checksum"));
            }
            if entry_count != 0 {
                return Err(OpenError::Corrupt("empty root with non-zero entry count"));
            }
        } else {
            // Non-empty root must reference a real data page within the allocated
            // frontier and carry a positive entry count. Ids live in the shifted
            // space `[id_delta + FIRST_DATA_INDEX, id_delta + tail/PAGE)`.
            if root_id < id_delta + FIRST_DATA_INDEX {
                return Err(OpenError::Corrupt(
                    "direct header root points into reserved pages",
                ));
            }
            // Index comparison (no `id × PAGE_SIZE`, which could overflow):
            // the root's file index must lie within the allocated frontier.
            if root_id - id_delta >= logical_tail / PAGE_SIZE as u64 {
                return Err(OpenError::Corrupt("direct header root beyond logical tail"));
            }
            if entry_count == 0 {
                return Err(OpenError::Corrupt("non-empty root with zero entry count"));
            }
        }
        Ok(DirectHeader {
            store_uuid,
            root,
            logical_tail,
            entry_count,
            header_txid: rd_u64(hdr::OFF_HEADER_TXID),
            id_delta,
        })
    }
}

/// One of the two alternating header slots of a v4 Direct store (T1.1):
/// slot A is page 0, slot B page 1. A flush writes the *stale* slot
/// and flips it; the committed slot is never in a flush's write set, so no
/// single crash can tear it (powersafe overwrite).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectSlot {
    A,
    B,
}

impl DirectSlot {
    /// The slot's byte offset in the data file.
    pub fn offset(self) -> u64 {
        match self {
            DirectSlot::A => 0,
            DirectSlot::B => PAGE_SIZE as u64,
        }
    }

    pub fn other(self) -> DirectSlot {
        match self {
            DirectSlot::A => DirectSlot::B,
            DirectSlot::B => DirectSlot::A,
        }
    }
}

/// Picks the winning header of a v4 store from its two slot images and names
/// the **stale** slot the next flush must overwrite.
///
/// A slot that failed to read or decode is *invalid* — torn by a crash or
/// pre-creation garbage — which is not an error by itself: the other slot,
/// whose bytes the failed flush never touched, is authoritative. Both invalid
/// → `Corrupt`. Both valid: higher `header_txid` wins (it strictly increases
/// per flush and never regresses across opens); a **tie** is legal only
/// as the byte-identical creation/compaction state, where slot A is the
/// canonical winner — a divergent tie is `Corrupt`, exactly like the WAL
/// checkpoint header's equal-seq rule, compared over the full page — and two
/// valid slots naming different stores (`store_uuid`) are likewise `Corrupt`
/// rather than guessed between.
///
/// A **format** refusal (v3 store under this binary, or a future v5) must
/// surface as `UnsupportedFormat`, not be misread as "invalid slot": if either
/// slot fails specifically with `UnsupportedFormat`, that error propagates.
pub(crate) fn pick_winning_header(
    slot_a: Result<&[u8], &std::io::Error>,
    slot_b: Result<&[u8], &std::io::Error>,
) -> Result<(DirectHeader, DirectSlot), OpenError> {
    let decode = |img: Result<&[u8], &std::io::Error>| -> Result<DirectHeader, OpenError> {
        match img {
            Ok(bytes) => DirectHeader::decode(bytes),
            Err(e) => Err(OpenError::Io(std::sync::Arc::new(std::io::Error::new(
                e.kind(),
                e.to_string(),
            )))),
        }
    };
    let a = decode(slot_a);
    let b = decode(slot_b);
    // Format refusals are store-level verdicts, not slot damage.
    for r in [&a, &b] {
        if let Err(OpenError::UnsupportedFormat {
            component,
            found,
            supported,
            newer,
        }) = r
        {
            return Err(OpenError::UnsupportedFormat {
                component,
                found: *found,
                supported: *supported,
                newer: *newer,
            });
        }
    }
    match (a, b) {
        (Ok(ha), Ok(hb)) => {
            if ha.store_uuid != hb.store_uuid {
                return Err(OpenError::Corrupt("header slots name different stores"));
            }
            match ha.header_txid.cmp(&hb.header_txid) {
                std::cmp::Ordering::Greater => Ok((ha, DirectSlot::B)),
                std::cmp::Ordering::Less => Ok((hb, DirectSlot::A)),
                std::cmp::Ordering::Equal => {
                    // Only the identical creation/compaction state may tie.
                    if slot_a.ok().map(|s| &s[..PAGE_SIZE]) == slot_b.ok().map(|s| &s[..PAGE_SIZE])
                    {
                        Ok((ha, DirectSlot::B))
                    } else {
                        Err(OpenError::Corrupt(
                            "header slots tie on txid but diverge in content",
                        ))
                    }
                }
            }
        }
        (Ok(ha), Err(_)) => Ok((ha, DirectSlot::B)),
        (Err(_), Ok(hb)) => Ok((hb, DirectSlot::A)),
        (Err(ea), Err(_)) => match ea {
            // Prefer slot A's reason; a real device error is reported as such,
            // a decode failure of both slots is the store-level Corrupt.
            OpenError::Io(e) => Err(OpenError::Io(e)),
            _ => Err(OpenError::Corrupt("no valid header slot")),
        },
    }
}

/// Draws a fresh random 16-byte store UUID from the OS entropy source
/// (T3.4; SipHash-keyed fallback if `/dev/urandom` is unreadable — uniqueness
/// is the requirement, matching [`crate::version::Incarnation`]).
pub fn generate_store_uuid() -> [u8; 16] {
    if let Some(bytes) = crate::version::os_random_bytes::<16>() {
        return bytes;
    }
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut out = [0u8; 16];
    let mut h = RandomState::new().build_hasher();
    h.write_u64(seq);
    h.write_u64(0xA5A5_5A5A_C3C3_3C3C);
    out[..8].copy_from_slice(&h.finish().to_le_bytes());
    let mut h2 = RandomState::new().build_hasher();
    h2.write_u64(seq);
    h2.write_u64(0x1234_5678_9ABC_DEF0);
    out[8..].copy_from_slice(&h2.finish().to_le_bytes());
    out
}

// -- Formula locator --------------------------------------------------------

/// The Direct page locator: `offset = (page_id − base) × PAGE_SIZE`, **bounded
/// by the allocation frontier**. `base` is the header's `id_delta` — `0` until
/// the first compaction, which renumbers the live set onto fresh ids so the
/// file stays dense while ids stay globally unique (M6 Phase D). File indices
/// are ≥ [`FIRST_DATA_INDEX`] and must lie within the allocated tail; reserved
/// indices (0/1), `NONE`, any id below the base, and any id past the tail
/// resolve to `None` (→ `Corrupt` "page not in generation"). The tail bound is
/// load-bearing: without it a corrupt branch ref to a wild id would compute a
/// gigantic offset and drive a read past EOF / a huge buffer allocation.
pub struct DirectLocator {
    tail: Arc<AtomicU64>,
    base: u64,
}

impl DirectLocator {
    /// The id→offset arithmetic shared by [`PageLocator::locate`] and the
    /// prefetch capability: the formula uses only the id, so both entry points
    /// resolve identically by construction.
    fn locate_id(&self, page_id: PageId) -> Option<Location> {
        // Subtract-then-compare *indices* before multiplying so a wild id can
        // never overflow or wrap `index × PAGE_SIZE` (panic in debug, alias a
        // low page in release). `max_index = tail / PAGE_SIZE`
        // is the count of allocated file pages; `index < max_index` guarantees
        // `index × PAGE_SIZE < tail`, which fits `u64`.
        let index = page_id.as_u64().checked_sub(self.base)?;
        let max_index = self.tail.load(Ordering::Acquire) / PAGE_SIZE as u64;
        if index < FIRST_DATA_INDEX || index >= max_index {
            return None;
        }
        Some(Location {
            member: Member::Data,
            offset: index * PAGE_SIZE as u64,
            length: PAGE_SIZE as u32,
        })
    }
}

impl PageLocator for DirectLocator {
    fn locate(&self, page_ref: PageRef) -> Option<Location> {
        self.locate_id(page_ref.page_id)
    }

    /// Direct declares the chain-prefetch capability (T2.2): the data file is
    /// append-only within a generation and the locator is pure id arithmetic,
    /// so an id-only lookup is exactly `locate` minus the ref.
    fn predict_overflow_location(&self, page_id: PageId) -> Option<Location> {
        self.locate_id(page_id)
    }
}

// -- Allocator --------------------------------------------------------------

/// The append-only page allocator: hands out monotonically increasing page ids
/// (== `base` + page index), never reused. Shares its `tail` frontier with the
/// generation's [`DirectLocator`] (so `locate` is bounded) — the two are created
/// together by [`direct_generation`]. `base` is the generation's `id_delta`
/// (M6 Phase D); ids stay globally monotonic across compactions while file
/// offsets subtract it. Bound to one [`FileGeneration`] (the generation-binding
/// seam).
#[derive(Clone, Debug)]
pub struct Allocator {
    next_index: u64,
    base: u64,
    tail: Arc<AtomicU64>,
}

impl Allocator {
    /// Allocates the next page id and extends the shared frontier. Saturating
    /// arithmetic so an (absurd, 2^50-page) frontier cannot overflow `u64`;
    /// at that scale the store is long past every other bound.
    pub fn alloc_id(&mut self) -> PageId {
        let id = self.next_index;
        // Saturating so an (unreachable) exhausted id space never panics in debug
        // or wraps in release (defense-in-depth; WAL recovery already caps the
        // recovered frontier well below `u64::MAX`).
        self.next_index = self.next_index.saturating_add(1);
        self.tail.store(
            (self.next_index - self.base).saturating_mul(PAGE_SIZE as u64),
            Ordering::Release,
        );
        PageId(id)
    }

    /// The current allocation frontier in bytes (== next free **physical**
    /// offset — the file's dense tail, past the id delta).
    pub fn logical_tail(&self) -> u64 {
        (self.next_index - self.base).saturating_mul(PAGE_SIZE as u64)
    }

    /// The generation's id delta: file offset = `(id − base) × PAGE_SIZE`.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// The next page id that will be allocated (the id-space frontier). StoreWal
    /// records this in each batch's record so recovery can seed the reopened
    /// allocator past every id ever allocated — including ids burned by an
    /// intra-batch superseded page — so no id is ever reused.
    pub fn peek_next(&self) -> u64 {
        self.next_index
    }

    /// A page-id allocator for StoreWal, where the id is a pure monotonic counter
    /// (the WAL overlay maps id → offset, so the `tail` frontier is unused by the
    /// locator; it is retained only for struct symmetry). Ids start at
    /// [`FIRST_DATA_INDEX`] and never reuse.
    pub fn for_wal(next_index: u64) -> Self {
        let next_index = next_index.max(FIRST_DATA_INDEX);
        Allocator {
            next_index,
            base: 0,
            tail: Arc::new(AtomicU64::new(next_index.saturating_mul(PAGE_SIZE as u64))),
        }
    }
}

/// Writes the **identical** base header to both slots (pages 0 and 1) and
/// `fdatasync`s, **blocking** until durable. The durable-create protocol calls
/// this *before* acknowledging a create, so `durable_txid = 0` is truthful and a
/// crash right after create still reopens. Identical slots are the one legal
/// txid tie; slot A is the canonical winner and the first flush targets B.
pub(crate) fn write_base_header_blocking(
    dev: &Arc<dyn PageIo>,
    store_uuid: [u8; 16],
) -> std::io::Result<()> {
    let header = DirectHeader {
        store_uuid,
        root: PageRef::EMPTY,
        logical_tail: FIRST_DATA_INDEX * PAGE_SIZE as u64,
        entry_count: 0,
        header_txid: 0,
        id_delta: 0,
    };
    futures_lite::future::block_on(async {
        dev.write_all_at(DirectSlot::A.offset(), header.encode())
            .await
            .result?;
        dev.write_all_at(DirectSlot::B.offset(), header.encode())
            .await
            .result?;
        dev.datasync().await
    })
}

/// Builds a fresh Direct `FileGeneration` (device + bounded formula locator) and
/// its paired [`Allocator`], seeded to start allocating id `next_id` under id
/// delta `base` (`FIRST_DATA_INDEX`/`0` for a fresh store, `id_delta +
/// logical_tail / PAGE_SIZE` / `id_delta` on reopen, the renumbering frontier
/// after a compaction). The two share the tail frontier.
pub fn direct_generation(
    dev: Arc<dyn PageIo>,
    next_id: u64,
    base: u64,
) -> (FileGeneration, Allocator) {
    let next_index = next_id.max(base + FIRST_DATA_INDEX);
    let tail = Arc::new(AtomicU64::new((next_index - base) * PAGE_SIZE as u64));
    let locator = DirectLocator {
        tail: Arc::clone(&tail),
        base,
    };
    let file = FileGeneration::direct(dev, Arc::new(locator));
    let alloc = Allocator {
        next_index,
        base,
        tail,
    };
    (file, alloc)
}

// -- Flush coordinator ------------------------------------------------------

/// One published-but-not-yet-durable batch, retained until a flush persists it.
struct Pending {
    txid: u64,
    root: PageRef,
    entry_count: u64,
    pages: Vec<Arc<Page>>,
    /// The reconciled dirty-page budget charge this batch still holds
    /// (released at the flush frontier — invariant 5).
    page_charge: u32,
}

/// The StoreDirect flush coordinator. For v1 it runs **inside the writer
/// thread**; a separate coordinator thread is deferred to M5's benchmark.
/// Serialized single-thread ownership makes flush monotonic (invariant 14)
/// trivially.
pub(crate) struct DirectCoordinator {
    dev: Arc<dyn PageIo>,
    store_uuid: [u8; 16],
    alloc: Allocator,
    flushed_txid: u64,
    pending: VecDeque<Pending>,
    dirty_pages: usize,
    high_water_pages: usize,
    /// Max concurrent in-flight page writes per flush (M7-B). Sourced from
    /// [`Options::fanout_cap`](crate::store::Options::fanout_cap) via the `new`
    /// parameter; `None` there uses [`FLUSH_FANOUT_CAP`]. A test lowers it (e.g.
    /// `Some(1)` ≡ the pre-M7-B serial loop, for differential equivalence).
    fanout_cap: usize,
    /// The data-path namespace owner (M6 Phase D): present only for stores that
    /// own their data path, absent for caller-supplied devices (no compaction).
    data_host: Option<Arc<dyn crate::compact::DataFileHost>>,
    /// The header slot the **next** flush overwrites (T1.1): the
    /// loser at open (`pick_winning_header`), slot B after a create or
    /// compaction (both slots identical, A canonical winner). Flipped only
    /// after the header write *and* its fdatasync succeeded — a failed flush
    /// poisons, so the slot the crash may have torn stays the designated stale
    /// slot and the committed winner is never targeted.
    stale_slot: DirectSlot,
}

impl DirectCoordinator {
    /// Builds a coordinator over the data device with a fresh or reopened
    /// allocator. `high_water_pages` bounds pinned dirty pages between flushes.
    /// The base header is written by the durable-create protocol
    /// ([`write_base_header_blocking`]) *before* the store starts, not here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dev: Arc<dyn PageIo>,
        store_uuid: [u8; 16],
        alloc: Allocator,
        flushed_txid: u64,
        high_water_pages: usize,
        fanout_cap: Option<usize>,
        data_host: Option<Arc<dyn crate::compact::DataFileHost>>,
        stale_slot: DirectSlot,
    ) -> Self {
        DirectCoordinator {
            dev,
            store_uuid,
            alloc,
            flushed_txid,
            pending: VecDeque::new(),
            dirty_pages: 0,
            high_water_pages: high_water_pages.max(1),
            fanout_cap: fanout_cap.map_or(FLUSH_FANOUT_CAP, |c| c.max(1)),
            data_host,
            stale_slot,
        }
    }

    /// The allocator, for the CoW builder to draw fresh page ids.
    pub(crate) fn allocator(&mut self) -> &mut Allocator {
        &mut self.alloc
    }

    /// Records a freshly published batch for later durability. `pages` are
    /// already pinned `Dirty` in the cache; `page_charge` is the reconciled
    /// dirty-page budget they hold.
    pub(crate) fn on_published(
        &mut self,
        txid: u64,
        root: PageRef,
        entry_count: u64,
        pages: Vec<Arc<Page>>,
        page_charge: u32,
    ) {
        self.dirty_pages += pages.len();
        self.pending.push_back(Pending {
            txid,
            root,
            entry_count,
            pages,
            page_charge,
        });
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Whether the writer should flush now, given the highest explicitly
    /// requested durable txid. Fires on any of: pinned dirty pages crossing the
    /// high-water mark (bounds memory under sustained *page-producing* writes);
    /// an explicit `flush`/`commit` requesting a not-yet-durable txid (a receipt
    /// wait resolves promptly even on a continuously busy channel); or the
    /// pending ledger reaching its cap (bounds buildup of *zero-page* batches —
    /// no-op overwrites, absent removes, check-only batches — which would
    /// otherwise never trip the page-based high-water mark).
    pub(crate) fn should_flush(&self, requested_txid: u64) -> bool {
        self.has_pending()
            && (self.dirty_pages >= self.high_water_pages
                || requested_txid > self.flushed_txid
                || self.pending.len() >= PENDING_FLUSH_CAP)
    }

    /// Flushes **all** pending batches durable (the window `(flushed, highest
    /// published]`), then advances the frontier. Monotonic: a request for an
    /// already-flushed frontier is a no-op, so a racing/regressing flush never
    /// lowers the header (invariant 14). On any I/O/sync error it poisons the
    /// store and returns the terminal error.
    ///
    /// The page writes fan out in bounded chunks of [`Self::fanout_cap`]
    /// (M7-B): eager submission (see [`PageIo::write_all_at`]) puts a whole
    /// chunk in flight at once, then every write's `BufResult` is awaited and its
    /// result observed **before** the datasync barrier. That observation — not
    /// buffer ownership, which the device handles — is the load-bearing invariant:
    /// no page whose write silently failed may slip past the barrier into a
    /// committed header (under O_DIRECT there is no page-cache writeback error for
    /// datasync to catch). A mid-chunk error therefore still drains its peers; the
    /// lowest input-order error wins and poisons; later chunks are never submitted.
    pub(crate) async fn flush_all(&mut self, core: &Arc<Core>) -> Result<(), Terminal> {
        if self.pending.is_empty() {
            return Ok(());
        }
        // 1. Write every page of every pending batch (superseded-in-window pages
        //    included — they occupy distinct appended offsets). Offsets subtract
        //    the generation's id delta (M6 Phase D); every pending page was
        //    allocated by `self.alloc`, so its id is ≥ the base by construction.
        let base = self.alloc.base();
        let offsets: Vec<(u64, &Arc<Page>)> = self
            .pending
            .iter()
            .flat_map(|batch| batch.pages.iter())
            .map(|page| ((page.id().as_u64() - base) * PAGE_SIZE as u64, page))
            .collect();
        // Distinct offsets is the precondition that makes the
        // concurrent writes below equivalent to the old serial loop. It holds by
        // construction (monotonic never-reuse allocator); assert it so a future
        // allocator bug fails a test instead of racing two writes to one offset.
        debug_assert_eq!(
            offsets
                .iter()
                .map(|(off, _)| *off)
                .collect::<std::collections::HashSet<u64>>()
                .len(),
            offsets.len(),
            "flush fan-out requires distinct page offsets across the window",
        );
        // Bounded fan-out: submit up to `fanout_cap` writes (all now in flight via
        // eager submission), then await every one — draining the whole chunk and
        // observing each result before the next chunk or the barrier. Join, never
        // try_join: a mid-chunk error still awaits its peers. Buffers are built
        // per-chunk (not up front) so transient memory stays bounded to the cap.
        for chunk in offsets.chunks(self.fanout_cap.max(1)) {
            let writes: Vec<_> = chunk
                .iter()
                .map(|(offset, page)| {
                    let buf = page.bytes().to_vec().into_boxed_slice();
                    self.dev.write_all_at(*offset, buf)
                })
                .collect();
            let mut first_err = None;
            for write in writes {
                let outcome = write.await;
                if first_err.is_none() {
                    first_err = outcome.result.err();
                }
            }
            if let Some(e) = first_err {
                // Poison before the barrier: `data_physical_bytes` and the header
                // both stay at the committed frontier, so durability is unchanged.
                // Note vs. the old serial loop: fan-out may have landed *later*,
                // higher-offset writes of this chunk before an earlier one failed,
                // so the file can be physically longer than the committed tail.
                // Those pages are orphans past the header's `logical_tail` — never
                // referenced, and overwritten by the next open's allocations (which
                // seed from the committed tail). The metric intentionally tracks the
                // committed frontier, not the raw file size (see `metrics.rs`).
                return Err(self.poison_io(core, e));
            }
        }
        // 2. fdatasync the pages before the header commits to them.
        if let Err(e) = self.dev.datasync().await {
            return Err(self.poison_io(core, e));
        }
        // 3. Overwrite the single header with the highest pending root.
        let last = self.pending.back().expect("pending non-empty");
        // The tail the header is about to commit *is* the file's length, so this
        // is where the space metric moves for Direct — a flush is the only other
        // transition besides a compaction that grows the data file (M6 Phase F).
        let logical_tail = self.alloc.logical_tail();
        // Metric caveat: this advances *before* the header write below. If the
        // header write or its fdatasync fails, the store poisons with
        // `data_physical_bytes` reporting the uncommitted tail — the
        // committed-frontier claim above holds only for page-write failures.
        // Harmless (a poisoned store serves no further flushes) but the metric
        // can over-report by one flush on that path.
        core.data_physical_bytes
            .store(logical_tail, Ordering::Release);
        let header = DirectHeader {
            store_uuid: self.store_uuid,
            root: last.root,
            logical_tail,
            entry_count: last.entry_count,
            header_txid: last.txid,
            id_delta: base,
        };
        let target_txid = last.txid;
        // 3b. The header goes to the STALE slot (T1.1): the committed
        // winner's page is not in this flush's write set, so no crash here can
        // tear it — recovery falls back to it (previous-or-complete-new oracle).
        if let Err(e) = self
            .dev
            .write_all_at(self.stale_slot.offset(), header.encode())
            .await
            .result
        {
            return Err(self.poison_io(core, e));
        }
        // 4. fdatasync the header.
        if let Err(e) = self.dev.datasync().await {
            return Err(self.poison_io(core, e));
        }
        // Committed: the just-written slot is the winner; the other is stale.
        self.stale_slot = self.stale_slot.other();
        // 5. Advance the frontier: unpin every flushed page (now durably
        //    readable → evictable, invariant 5) and release its budget charge.
        self.flushed_txid = target_txid;
        for batch in self.pending.drain(..) {
            for page in &batch.pages {
                core.cache.release_dirty(page.id());
            }
            if batch.page_charge > 0 {
                core.budget.release_pages(batch.page_charge);
            }
        }
        self.dirty_pages = 0;
        core.durable_txid.store(target_txid, Ordering::Release);
        // Wake flush/commit waiters (they listen on the shared state event).
        core.state.event().notify(usize::MAX);
        Ok(())
    }

    fn poison_io(&self, core: &Arc<Core>, err: std::io::Error) -> Terminal {
        self.poison_with(core, PoisonReason::Io(Arc::new(err)))
    }

    fn poison_with(&self, core: &Arc<Core>, reason: PoisonReason) -> Terminal {
        core.state.poison(reason);
        match core.state.check_terminal() {
            Err(t) => t,
            Ok(()) => Terminal::Poisoned(PoisonReason::Internal("poison did not stick")),
        }
    }

    /// A read failure on the compaction's rebuild walk: a corrupt or unreadable
    /// page in the *live* set means the writer could not honour a later read of
    /// that page either, so it poisons — exactly like the checkpoint's
    /// relocation walk (`WalCoordinator::poison_read`).
    fn poison_read(
        &self,
        core: &Arc<Core>,
        err: crate::error::ReadError,
    ) -> crate::error::CheckpointError {
        use crate::error::{CheckpointError, ReadError};
        match err {
            ReadError::Corrupt(what) => self.poison_with(core, PoisonReason::Corrupt(what)).into(),
            ReadError::Io(e) => self.poison_with(core, PoisonReason::Io(e)).into(),
            ReadError::Poisoned(r) => CheckpointError::Poisoned(r),
            ReadError::Closed => CheckpointError::Closed,
            // The rebuild reads through the generation's own strong `file`
            // handle, never a snapshot lease — unreachable, and loud if a
            // refactor ever misroutes it (M6 Phase E).
            ReadError::SnapshotExpired => self
                .poison_with(
                    core,
                    PoisonReason::Internal("snapshot lease on compaction path"),
                )
                .into(),
        }
    }

    /// Runs one Direct compaction (M6 Phase D), **on the writer
    /// thread, at a flushed barrier**: rewrite the live page set of the visible
    /// root densely into a fresh temp file under **fresh** ids (ids are never
    /// reused — the shared page cache is keyed by id) with a raised `id_delta`,
    /// commit the temp's own v2 header, promote it over the data path
    /// (rename + directory fsync — the durable cutover), and rebind. The old
    /// inode leaves the namespace but stays readable by every retained
    /// generation until Phase E retention releases or revokes it.
    ///
    /// Failure policy (a deviation from the checkpoint's poison-everything):
    /// steps that touch only the private temp file abort
    /// **cleanly** (`CheckpointError::Io`, temp removed, store unpoisoned) —
    /// nothing shared changed. Live-set read failures poison (the committed
    /// tree is unreadable). A promotion failure poisons: the rename/dir-fsync
    /// pair is the commit point, and once it is entered and fails, which file a
    /// crash would resurface is unknowable (invariant 17).
    pub(crate) async fn compact(
        &mut self,
        core: &Arc<Core>,
    ) -> Result<crate::compact::CompactReport, crate::error::CheckpointError> {
        use crate::error::CheckpointError;

        let Some(host) = self.data_host.clone() else {
            return Err(CheckpointError::Unsupported);
        };
        // Hard barrier check (T3.4; was a release-elided debug_assert): an
        // off-barrier compaction would rewrite from a root that does not match
        // the flushed frontier and publish it as committed. Refuse cleanly
        // before anything is created or mutated.
        if !self.pending.is_empty() {
            return Err(CheckpointError::Internal(
                "compaction invoked with unflushed pending batches",
            ));
        }
        let current = core.root();
        if current.version.txid() != self.flushed_txid {
            return Err(CheckpointError::Internal(
                "compaction barrier is not fully flushed",
            ));
        }

        // A read-only metadata probe before anything is created or mutated: a
        // failure here is a clean abort, not a poison — the committed tree was
        // not shown unreadable and no commit is in doubt.
        let data_bytes_before = match self.dev.size().await {
            Ok(s) => s,
            Err(e) => return Err(CheckpointError::Io(Arc::new(e))),
        };

        // Fresh ids continue the monotonic space; the new delta maps the first
        // onto FIRST_DATA_INDEX. Checked as strictly as `DirectHeader::decode`,
        // so the writer can never lay down a header recovery refuses.
        let first_new_id = self.alloc.peek_next();
        let delta = first_new_id - FIRST_DATA_INDEX;
        if delta >= MAX_ID_DELTA {
            return Err(self
                .poison_with(core, PoisonReason::Internal("direct id space exhausted"))
                .into());
        }

        let temp = match host.create_temp() {
            Ok(d) => d,
            Err(e) => return Err(CheckpointError::Io(Arc::new(e))),
        };

        // Post-order dense rebuild (children first: a parent's child refs embed
        // the child's new id *and* new checksum). Reads go through the normal
        // cache/validation path; one full page image is held at a time.
        let mut next_id = first_new_id;
        let mut written = 0u64;
        let new_root = if current.root.is_empty() {
            PageRef::EMPTY
        } else {
            let mut seen = std::collections::HashSet::new();
            match rebuild_subtree(
                &core.cache,
                &current.file,
                &temp,
                current.root,
                &mut next_id,
                delta,
                &mut written,
                &mut seen,
                0,
            )
            .await
            {
                Ok(r) => r,
                Err(RebuildFail::Read(e)) => {
                    host.remove_temp();
                    return Err(self.poison_read(core, e));
                }
                Err(RebuildFail::Temp(e)) => {
                    host.remove_temp();
                    return Err(CheckpointError::Io(Arc::new(e)));
                }
            }
        };
        let logical_tail = (FIRST_DATA_INDEX + written) * PAGE_SIZE as u64;

        // The temp's own v2 header + zeroed slot B, then make its content
        // durable. Content is unchanged, so the txid is the barrier's — a
        // compaction is not a write and consumes no txid.
        let header = DirectHeader {
            store_uuid: self.store_uuid,
            root: new_root,
            logical_tail,
            entry_count: current.entry_count,
            header_txid: self.flushed_txid,
            id_delta: delta,
        };
        let commit = async {
            // Both slots get the identical header (the legal txid tie, slot A
            // canonical), so the promoted file opens under the v4
            // two-slot protocol and the next flush targets B.
            temp.write_all_at(DirectSlot::A.offset(), header.encode())
                .await
                .result?;
            temp.write_all_at(DirectSlot::B.offset(), header.encode())
                .await
                .result?;
            temp.datasync().await
        };
        if let Err(e) = commit.await {
            host.remove_temp();
            return Err(CheckpointError::Io(Arc::new(e)));
        }

        // The durable cutover. rename + parent-dir fsync; on failure the commit
        // is in doubt → poison (invariant 17).
        if let Err(e) = host.promote_temp() {
            return Err(self.poison_io(core, e).into());
        }
        // ---- committed: the dense file is the data file ----

        // Infallible rebind: new generation (device + delta locator + allocator
        // continuing at the renumbering frontier), republished under the *same*
        // version, then retire the superseded generation into Phase E retention
        // (after the swap, so a racing `snapshot()` capture retry observes the
        // new descriptor — same ordering as the checkpoint's rebind).
        let (file, alloc) = direct_generation(Arc::clone(&temp), next_id, delta);
        let file = Arc::new(file);
        self.dev = temp;
        self.alloc = alloc;
        // The promoted file's slots are identical (tie, A wins) → next flush
        // targets B, exactly as a reopen of this file would conclude.
        self.stale_slot = DirectSlot::B;
        core.root.store(Arc::new(crate::store::RootDescriptor {
            version: current.version,
            root: new_root,
            file: Arc::clone(&file),
            entry_count: current.entry_count,
            retain: crate::retention::SnapshotGen::active(file),
        }));
        // Device-anchored accounting (H5): the superseded data inode leaves the
        // namespace here. Direct has no WAL segment; each generation owns its
        // inode exclusively (generations are minted only at compactions), so the
        // one device registers once and counts while any generation, lease, or
        // parked load holds it. Uniform with the WAL compaction shape.
        let superseded = current.file.device(Member::Data);
        core.retention
            .note_superseded(&superseded, data_bytes_before);
        core.retention.retire(Arc::clone(&current.retain));
        core.data_physical_bytes
            .store(logical_tail, Ordering::Release);
        core.data_live_pages.store(written, Ordering::Release);
        core.reclaimed_bytes.fetch_add(
            data_bytes_before.saturating_sub(logical_tail),
            Ordering::AcqRel,
        );
        core.generations_created.fetch_add(1, Ordering::AcqRel);
        core.compactions.fetch_add(1, Ordering::AcqRel);

        Ok(crate::compact::CompactReport {
            live_pages: written,
            data_bytes_before,
            data_bytes_after: logical_tail,
        })
    }
}

/// Why a rebuild step failed: a live-set read (→ poison) vs a private temp-file
/// write (→ healthy abort).
enum RebuildFail {
    Read(crate::error::ReadError),
    Temp(std::io::Error),
}

/// Rewrites one overflow chain into `temp` under fresh ids, returning the ref the
/// rebuilt leaf must carry (M6 Phase F).
///
/// **Iterative, in two passes**, not recursive like the tree walk: a chain is
/// long (a 1 MiB value is 65 links) where a tree is *deep* (≤ 64), so recursing
/// per link would collide with the depth guard and stack a boxed future per
/// 16 KiB of value. Pass one walks forward collecting the payload — bounded by
/// the ref's own `total_len`, so a cyclic or over-long chain fails here rather
/// than being rebuilt. Pass two emits **tail-first**, because each link embeds
/// its successor's new id *and* checksum.
///
/// Peak memory is one value (≤ `MAX_VALUE_LEN`), the same bound a read of that
/// value already pays.
#[allow(clippy::too_many_arguments)]
async fn rebuild_chain(
    cache: &crate::cache::PageCache,
    file: &FileGeneration,
    temp: &Arc<dyn PageIo>,
    oref: crate::page::OverflowRef,
    next_id: &mut u64,
    delta: u64,
    written: &mut u64,
    seen: &mut std::collections::HashSet<u64>,
) -> Result<crate::page::OverflowRef, RebuildFail> {
    use crate::error::ReadError;
    let total = oref.total_len as usize;
    let mut payload: Vec<u8> = Vec::with_capacity(total);
    let mut created = 0u64;
    // Pass one: read the chain forward, driving the **shared** cursor. Rewriting
    // the bytes is exactly the job that must not have its own opinion of what a
    // chain looks like — a hand-rolled loop here could accept a structure the
    // reader rejects and rebuild it into the compacted file.
    let mut cursor = crate::page::OverflowCursor::new(oref);
    while let Some(r) = cursor.pending() {
        if !seen.insert(r.page_id.as_u64()) {
            return Err(RebuildFail::Read(ReadError::Corrupt(
                "page reachable twice in one tree (cycle?)",
            )));
        }
        let page = cache.get_page(r, file).await.map_err(RebuildFail::Read)?;
        cursor.accept(&page).map_err(RebuildFail::Read)?;
        payload.extend_from_slice(page.overflow_chunk());
        created = page.created_txid();
    }
    debug_assert_eq!(
        payload.len(),
        total,
        "the cursor guarantees the exact length"
    );

    // Pass two: emit tail-first, because a link embeds its successor's new id and
    // checksum. Ids are drawn **head-first** and only then filled in reverse, so a
    // chain lands ascending on disk: Direct's offset is a pure function of the id,
    // and ids assigned in encode order would lay every large value out backwards,
    // defeating readahead for exactly the objects this phase exists to store
    // (M6 Phase F). Reserving the range first costs nothing and decouples id
    // order from encode order.
    let n = crate::page::overflow_chunks(&payload).count();
    let first_id = *next_id;
    *next_id += n as u64;
    // Walking in reverse, `new_next` holds the ref of the link just encoded — so
    // after the last iteration (i == 0) it *is* the head.
    let mut new_next: Option<PageRef> = None;
    for (i, chunk) in crate::page::overflow_chunks(&payload).enumerate().rev() {
        let id = PageId(first_id + i as u64);
        let page = crate::page::encode_overflow(id, created, chunk, new_next);
        new_next = Some(page.page_ref());
        let offset = (id.as_u64() - delta) * PAGE_SIZE as u64;
        temp.write_all_at(offset, page.bytes().to_vec().into_boxed_slice())
            .await
            .result
            .map_err(RebuildFail::Temp)?;
        *written += 1;
    }
    Ok(crate::page::OverflowRef {
        head: new_next.expect("a non-empty value has at least one chunk"),
        total_len: oref.total_len,
    })
}

/// Rewrites the subtree at `page_ref` into `temp`, post-order, under fresh ids
/// drawn from `next_id` and placed at `(id − delta) × PAGE_SIZE`. Returns the
/// subtree's new root ref. Owned key/ref vectors are extracted and the source
/// page dropped before recursing, so the held set is one page image plus
/// O(height) directories.
#[allow(clippy::too_many_arguments)]
fn rebuild_subtree<'a>(
    cache: &'a crate::cache::PageCache,
    file: &'a FileGeneration,
    temp: &'a Arc<dyn PageIo>,
    page_ref: PageRef,
    next_id: &'a mut u64,
    delta: u64,
    written: &'a mut u64,
    seen: &'a mut std::collections::HashSet<u64>,
    depth: u32,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<PageRef, RebuildFail>> + Send + 'a>>
{
    use crate::page::PageKind;
    Box::pin(async move {
        if depth >= crate::read::MAX_TREE_DEPTH {
            return Err(RebuildFail::Read(crate::error::ReadError::Corrupt(
                "tree exceeds max depth (cycle?)",
            )));
        }
        // The same reachable-twice guard `LivePages` calls load-bearing: ids
        // are unique within a generation, so a repeat is a cycle/diamond in a
        // checksum-valid but corrupt structure. Without it a crafted DAG makes
        // one compaction write ~2^depth pages of duplicated subtrees — bounded
        // work on corrupt input beats a "successful" exponential rebuild.
        if !seen.insert(page_ref.page_id.as_u64()) {
            return Err(RebuildFail::Read(crate::error::ReadError::Corrupt(
                "page reachable twice in one tree (cycle?)",
            )));
        }
        let page = cache
            .get_page(page_ref, file)
            .await
            .map_err(RebuildFail::Read)?;
        let new_page = match page.kind() {
            PageKind::Leaf => {
                let mut entries: Vec<(Vec<u8>, crate::page::EntryValue)> = page
                    .leaf_entries()
                    .map(|(k, v)| (k.to_vec(), v.to_owned_value()))
                    .collect();
                let created = page.created_txid();
                drop(page);
                // Overflow chains renumber with everything else (M6 Phase F):
                // the chain's pages are live pages of this generation, and the
                // rebuilt leaf must carry the ref they land on.
                for (_, v) in entries.iter_mut() {
                    if let crate::page::EntryValue::Overflow(o) = v {
                        let moved =
                            rebuild_chain(cache, file, temp, *o, next_id, delta, written, seen)
                                .await?;
                        *v = crate::page::EntryValue::Overflow(moved);
                    }
                }
                let id = PageId(*next_id);
                *next_id += 1;
                crate::page::encode_leaf(id, created, &entries)
            }
            PageKind::Overflow => {
                // Only ever reached through a leaf's ref, via `rebuild_chain`.
                return Err(RebuildFail::Read(crate::error::ReadError::Corrupt(
                    "tree ref resolves to an overflow page",
                )));
            }
            PageKind::Branch => {
                let children = page.branch_children().to_vec();
                let separators: Vec<Vec<u8>> =
                    page.branch_separators().map(<[u8]>::to_vec).collect();
                let created = page.created_txid();
                drop(page);
                let mut new_children = Vec::with_capacity(children.len());
                for child in children {
                    new_children.push(
                        rebuild_subtree(
                            cache,
                            file,
                            temp,
                            child,
                            next_id,
                            delta,
                            written,
                            seen,
                            depth + 1,
                        )
                        .await?,
                    );
                }
                let id = PageId(*next_id);
                *next_id += 1;
                crate::page::encode_branch(id, created, &new_children, &separators)
            }
        };
        let offset = (new_page.id().as_u64() - delta) * PAGE_SIZE as u64;
        temp.write_all_at(offset, new_page.bytes().to_vec().into_boxed_slice())
            .await
            .result
            .map_err(RebuildFail::Temp)?;
        *written += 1;
        Ok(new_page.page_ref())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rebuild's reachable-twice guard gets its own crafted-DAG probe.
    /// A checksum-valid branch naming one leaf twice must abort the rebuild as
    /// `Corrupt` after **bounded** work — the guard exists precisely so a
    /// crafted diamond cannot make one compaction write ~2^depth duplicated
    /// subtrees.
    #[test]
    fn rebuild_rejects_a_diamond_after_bounded_work() {
        use crate::backend::DataFileLocator;
        use crate::io::FakeIo;
        use crate::page::{encode_branch, encode_leaf_inline};
        use futures_lite::future::block_on;

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
            let file =
                FileGeneration::direct(dev.clone() as Arc<dyn crate::io::PageIo>, Arc::new(loc));
            let cache = crate::cache::PageCache::new();
            let temp: Arc<dyn crate::io::PageIo> = Arc::new(FakeIo::new(0));

            let mut next_id = 100u64;
            let mut written = 0u64;
            let mut seen = std::collections::HashSet::new();
            let out = rebuild_subtree(
                &cache,
                &file,
                &temp,
                root.page_ref(),
                &mut next_id,
                100 - FIRST_DATA_INDEX,
                &mut written,
                &mut seen,
                0,
            )
            .await;
            assert!(
                matches!(
                    out,
                    Err(RebuildFail::Read(crate::error::ReadError::Corrupt(
                        "page reachable twice in one tree (cycle?)"
                    )))
                ),
                "the second slot naming the same leaf must abort the rebuild"
            );
            // Bounded: only the first copy of the leaf was rewritten; the
            // branch (post-order) never landed.
            assert!(written <= 1, "rebuild wrote {written} pages before abort");
        });
    }

    #[test]
    fn header_roundtrips() {
        let h = DirectHeader {
            store_uuid: [7u8; 16],
            root: PageRef {
                page_id: PageId(5),
                expected_checksum: Checksum(0xDEAD_BEEF),
            },
            logical_tail: 6 * PAGE_SIZE as u64,
            entry_count: 99,
            header_txid: 42,
            id_delta: 0,
        };
        let bytes = h.encode();
        let d = DirectHeader::decode(&bytes).unwrap();
        assert_eq!(d.store_uuid, [7u8; 16]);
        assert_eq!(d.root.page_id, PageId(5));
        assert_eq!(d.root.expected_checksum, Checksum(0xDEAD_BEEF));
        assert_eq!(d.logical_tail, 6 * PAGE_SIZE as u64);
        assert_eq!(d.entry_count, 99);
        assert_eq!(d.header_txid, 42);
    }

    /// M6 Phase D: the v2 delta round-trips; ids resolve in the shifted space
    /// and the root/delta relationship is validated as strictly as v1's bounds.
    #[test]
    fn v2_header_delta_roundtrips_and_validates() {
        let base = DirectHeader {
            store_uuid: [7u8; 16],
            root: PageRef {
                page_id: PageId(13),
                expected_checksum: Checksum(1),
            },
            logical_tail: 6 * PAGE_SIZE as u64,
            entry_count: 9,
            header_txid: 4,
            id_delta: 10,
        };
        let d = DirectHeader::decode(&base.encode()).unwrap();
        assert_eq!(d.id_delta, 10);
        assert_eq!(d.root.page_id, PageId(13));

        // Root below the shifted first data id (10 + 2) → reserved-region defect.
        let bad = DirectHeader {
            root: PageRef {
                page_id: PageId(11),
                expected_checksum: Checksum(1),
            },
            ..base
        };
        assert!(DirectHeader::decode(&bad.encode()).is_err());
        // Root index at/past the tail (10 + 6) → beyond-tail defect.
        let bad = DirectHeader {
            root: PageRef {
                page_id: PageId(16),
                expected_checksum: Checksum(1),
            },
            ..base
        };
        assert!(DirectHeader::decode(&bad.encode()).is_err());
        // Delta at the exhaustion bound → refused (as strict as the writer).
        let bad = DirectHeader {
            id_delta: MAX_ID_DELTA,
            ..base
        };
        assert!(DirectHeader::decode(&bad.encode()).is_err());
    }

    /// Every superseded header format is refused with a clear error, not silently
    /// misread. v1 lacked `id_delta` (pre-Phase D); v2 predates Phase F's page
    /// format, and opening such a store would mean reading v1 pages the decoder no
    /// longer accepts — a refusal at open, not a poison on the first read.
    #[test]
    fn superseded_format_bytes_are_refused() {
        let h = DirectHeader {
            store_uuid: [0u8; 16],
            root: PageRef::EMPTY,
            logical_tail: FIRST_DATA_INDEX * PAGE_SIZE as u64,
            entry_count: 0,
            header_txid: 0,
            id_delta: 0,
        };
        // Older media (v1/v2) and a future release's header (sentinel 255, so a
        // later legitimate format bump can never turn this case "current") are
        // both refused with the structured forward-policy error — never `Corrupt`,
        // which is reserved for a damaged image. The older/newer distinction is
        // reported so an operator knows whether to downgrade the store or the
        // binary.
        for (byte, expect_newer) in [(1u8, false), (2u8, false), (255u8, true)] {
            let mut img = h.encode().to_vec();
            img[hdr::OFF_FORMAT] = byte;
            // Recompute the checksum so only the format check can fire.
            let ck = crc32c::crc32c(&img[..hdr::OFF_HEADER_CKSUM]);
            img[hdr::OFF_HEADER_CKSUM..hdr::OFF_HEADER_CKSUM + 4]
                .copy_from_slice(&ck.to_le_bytes());
            assert!(
                matches!(
                    DirectHeader::decode(&img),
                    Err(OpenError::UnsupportedFormat {
                        component: "direct header",
                        found,
                        supported: hdr::FORMAT_V4,
                        newer,
                    }) if found == byte && newer == expect_newer
                ),
                "format byte {byte} must be refused at open as UnsupportedFormat"
            );
        }
        // A damaged magic is still `Corrupt`, not an unsupported-version refusal —
        // the two failure classes stay distinct.
        let mut bad_magic = h.encode().to_vec();
        bad_magic[hdr::OFF_MAGIC] ^= 0xFF;
        assert!(matches!(
            DirectHeader::decode(&bad_magic),
            Err(OpenError::Corrupt("bad direct header magic"))
        ));
        // And the current one still round-trips.
        assert!(DirectHeader::decode(&h.encode()).is_ok());
    }

    /// The delta locator: ids below the base or past the shifted tail never
    /// resolve; in-range ids resolve to `(id − base) × PAGE_SIZE`.
    #[test]
    fn delta_locator_is_bounded_shifted_formula() {
        let dev: Arc<dyn PageIo> = Arc::new(crate::io::FakeIo::new(0));
        let (file, mut alloc) = direct_generation(dev, 50, 48);
        let loc = file.locator();
        let refer = |id: u64| PageRef {
            page_id: PageId(id),
            expected_checksum: Checksum(0),
        };
        // Below the base and in the shifted reserved region: never resolve.
        assert!(
            loc.locate(refer(2)).is_none(),
            "pre-compaction id space is gone"
        );
        assert!(loc.locate(refer(48)).is_none(), "shifted header index");
        assert!(loc.locate(refer(49)).is_none(), "shifted slot-B index");
        // Past the (empty) frontier until allocated.
        assert!(loc.locate(refer(50)).is_none());
        assert_eq!(alloc.alloc_id(), PageId(50));
        let l = loc.locate(refer(50)).unwrap();
        assert_eq!(
            l.offset,
            2 * PAGE_SIZE as u64,
            "first data id lands at the first data page"
        );
        assert_eq!(alloc.logical_tail(), 3 * PAGE_SIZE as u64);
    }

    #[test]
    fn corrupt_header_rejected() {
        let h = DirectHeader {
            store_uuid: [0u8; 16],
            root: PageRef::EMPTY,
            logical_tail: FIRST_DATA_INDEX * PAGE_SIZE as u64,
            entry_count: 0,
            header_txid: 0,
            id_delta: 0,
        };
        let mut bytes = h.encode().to_vec();
        bytes[hdr::OFF_ENTRY_COUNT] ^= 0xFF; // flip a covered byte
        assert!(matches!(
            DirectHeader::decode(&bytes),
            Err(OpenError::Corrupt(_))
        ));
    }

    #[test]
    fn locator_is_bounded_index_formula() {
        let dev: Arc<dyn PageIo> = Arc::new(crate::io::FakeIo::new(0));
        let (file, mut alloc) = direct_generation(dev, FIRST_DATA_INDEX, 0);
        let loc = file.locator();
        // Reserved ids never resolve.
        assert!(loc
            .locate(PageRef {
                page_id: PageId(0),
                expected_checksum: Checksum(0)
            })
            .is_none());
        assert!(loc
            .locate(PageRef {
                page_id: PageId(1),
                expected_checksum: Checksum(0)
            })
            .is_none());
        // Id 2 is past the (empty) frontier until allocated.
        assert!(loc
            .locate(PageRef {
                page_id: PageId(2),
                expected_checksum: Checksum(0)
            })
            .is_none());
        assert_eq!(alloc.alloc_id(), PageId(2));
        let l = loc
            .locate(PageRef {
                page_id: PageId(2),
                expected_checksum: Checksum(0),
            })
            .unwrap();
        assert_eq!(l.offset, 2 * PAGE_SIZE as u64);
        assert_eq!(l.length, PAGE_SIZE as u32);
        // A wild out-of-range ref resolves to None (→ Corrupt), never a giant
        // offset.
        assert!(loc
            .locate(PageRef {
                page_id: PageId(1_000_000),
                expected_checksum: Checksum(0)
            })
            .is_none());
    }

    #[test]
    fn allocator_is_monotonic_from_two() {
        let dev: Arc<dyn PageIo> = Arc::new(crate::io::FakeIo::new(0));
        let (_file, mut a) = direct_generation(dev, FIRST_DATA_INDEX, 0);
        assert_eq!(a.alloc_id(), PageId(2));
        assert_eq!(a.alloc_id(), PageId(3));
        assert_eq!(a.logical_tail(), 4 * PAGE_SIZE as u64);

        let dev2: Arc<dyn PageIo> = Arc::new(crate::io::FakeIo::new(0));
        let (_f2, mut b) = direct_generation(dev2, 10 * PAGE_SIZE as u64 / PAGE_SIZE as u64, 0);
        assert_eq!(b.alloc_id(), PageId(10));
    }

    /// M6 Phase H forward-format policy (integration). A store whose on-disk data
    /// header carries an unsupported format version is **refused at open** — with
    /// the structured [`OpenError::UnsupportedFormat`], not `Corrupt` — and is
    /// **not** created-over, recovered, truncated, or poisoned: the refusal lands
    /// before any mutation, so the on-disk image is byte-for-byte unchanged. This
    /// is the guarantee the classifier's unit tests cannot give (they stop at
    /// `decode`); it proves the whole open path has no create/recover fallback for
    /// a valid-magic wrong-version header. Covers the empty store and a written
    /// one, older media (v2) and a future release (sentinel 255).
    #[test]
    fn a_version_bumped_direct_store_is_refused_at_open_not_recovered() {
        use crate::backend::Direct;
        use crate::batch::{ApplyOutcome, WriteBatch};
        use crate::io::FakeIo;
        use crate::store::{Options, Store};
        use futures_lite::future::block_on;

        // Build the durable image of a real store, `written` true = a non-empty
        // tree, false = a freshly-created empty one.
        let image_of = |written: bool| -> Vec<u8> {
            let fake = Arc::new(FakeIo::new(0));
            let store = Store::<Direct>::create_with_io(
                Arc::clone(&fake) as Arc<dyn PageIo>,
                Options::default(),
            )
            .unwrap();
            if written {
                let map = crate::map::BTreeMap::over(store);
                block_on(async {
                    let b = WriteBatch::new().insert(b"k".to_vec(), b"v".to_vec());
                    assert!(matches!(
                        map.apply(b).await.unwrap(),
                        ApplyOutcome::Applied { .. }
                    ));
                    map.close().await.unwrap();
                });
            } else {
                block_on(store.close()).unwrap();
            }
            fake.snapshot_bytes()
        };

        for written in [false, true] {
            for (byte, expect_newer) in [(2u8, false), (255u8, true)] {
                let mut img = image_of(written);
                // The Direct header is a single slot at offset 0; bump its format
                // byte and re-seal the header checksum so *only* the version check
                // can fire (not a checksum defect, which would be `Corrupt`).
                img[hdr::OFF_FORMAT] = byte;
                let ck = crc32c::crc32c(&img[..hdr::OFF_HEADER_CKSUM]);
                img[hdr::OFF_HEADER_CKSUM..hdr::OFF_HEADER_CKSUM + 4]
                    .copy_from_slice(&ck.to_le_bytes());

                let before = img.clone();
                let fake = Arc::new(FakeIo::from_vec(img));
                let opened = Store::<Direct>::open_over(
                    Arc::clone(&fake) as Arc<dyn PageIo>,
                    Options::default(),
                );
                assert!(
                    matches!(
                        opened,
                        Err(OpenError::UnsupportedFormat {
                            component: "direct header",
                            found,
                            supported: hdr::FORMAT_V4,
                            newer,
                        }) if found == byte && newer == expect_newer
                    ),
                    "written={written} byte={byte}: open must refuse with UnsupportedFormat, got {:?}",
                    opened.err()
                );
                assert_eq!(
                    fake.snapshot_bytes(),
                    before,
                    "written={written} byte={byte}: a refused open must not mutate the store"
                );
            }
        }
    }
}
