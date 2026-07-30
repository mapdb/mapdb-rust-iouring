//! StoreWal physical layout: the WAL segment identity header, the chained
//! self-framing record codec, the runtime page locator, crash recovery, and the
//! sequential append + group-commit coordinator.
//!
//! Layout (revised by M6 Phase B):
//!
//! ```text
//! data file    checkpoint header (pages 0/1, alternating slots — checkpoint.rs)
//!              checkpointed page images + manifest blob, from page 2
//! wal segment  segment header (page 0): …, store_uuid, segment_seq
//!              record*      append-only, from offset PAGE_SIZE
//! ```
//!
//! The data header is the **sole** durable carrier of reset state: it names the
//! WAL segment holding records past the checkpoint txid `T`, where in it to start
//! scanning, and the chain seed to validate from. The WAL segment header is pure
//! **identity** (uuid + seq) and is never rewritten, so a checkpoint has exactly
//! one commit point and no torn re-stamp is possible.
//! Both headers carry the same random `store_uuid`; open fails with
//! `UuidMismatch` if they disagree. Records are chained by
//! `crc32c_append(prev_checksum, …)` and self-framing (a length-prefixed page
//! payload plus a commit trailer), so recovery accepts the longest valid
//! committed prefix and stops at the first torn/corrupt record without scanning
//! arbitrary bytes.

use crate::admission::{ReconciledWalReservation, WalMeter};
use crate::backend::{FileGeneration, Location, Member, PageLocator};
use crate::direct::{Allocator, FIRST_DATA_INDEX};
use crate::error::{OpenError, PoisonReason, Terminal};
use crate::io::PageIo;
use crate::page::{Checksum, Page, PageId, PageRef, PAGE_SIZE};
use crate::store::Core;
use event_listener::Event;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Flush once this many batches accumulate uncommitted, regardless of page count
/// (bounds the pending ledger under zero-page batches — mirrors Direct's cap).
const PENDING_FLUSH_CAP: usize = 256;

// -- Header codecs ----------------------------------------------------------

mod hdr {
    pub const WAL_MAGIC: u32 = 0x4D50_5357; // "MP5W" (wal segment header)
    /// M6 Phase B: the segment header gained `segment_seq`, so the chain seed
    /// (its checksum) differs per segment.
    pub const FORMAT_V2: u8 = 2;
    pub const ALGO_CRC32C: u8 = 1;

    pub const OFF_MAGIC: usize = 0; // u32
    pub const OFF_FORMAT: usize = 4; // u8
    pub const OFF_ALGO: usize = 5; // u8
    pub const OFF_PAGE_SIZE: usize = 8; // u32
    pub const OFF_STORE_UUID: usize = 16; // 16 bytes
    pub const OFF_SEGMENT_SEQ: usize = 32; // u64
    pub const OFF_HEADER_CKSUM: usize = 40; // u32 crc32c over [0, 40)
}

/// Encodes a WAL **segment identity header**: `store_uuid` pairs the segment to
/// its data file, `segment_seq` names it. Both are covered by the header
/// checksum, which is the segment's record chain seed — so a record chained under
/// segment `n`'s seed can never validate inside segment `n+1`. This image is
/// written once, when the segment is created, and is **never** rewritten: a
/// checkpoint has exactly one commit point (the data header), so there is no
/// torn re-stamp.
pub fn encode_wal_header(store_uuid: [u8; 16], segment_seq: u64) -> Box<[u8]> {
    let mut buf = vec![0u8; PAGE_SIZE];
    buf[hdr::OFF_MAGIC..hdr::OFF_MAGIC + 4].copy_from_slice(&hdr::WAL_MAGIC.to_le_bytes());
    buf[hdr::OFF_FORMAT] = hdr::FORMAT_V2;
    buf[hdr::OFF_ALGO] = hdr::ALGO_CRC32C;
    buf[hdr::OFF_PAGE_SIZE..hdr::OFF_PAGE_SIZE + 4]
        .copy_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
    buf[hdr::OFF_STORE_UUID..hdr::OFF_STORE_UUID + 16].copy_from_slice(&store_uuid);
    buf[hdr::OFF_SEGMENT_SEQ..hdr::OFF_SEGMENT_SEQ + 8].copy_from_slice(&segment_seq.to_le_bytes());
    let ck = crc32c::crc32c(&buf[..hdr::OFF_HEADER_CKSUM]);
    buf[hdr::OFF_HEADER_CKSUM..hdr::OFF_HEADER_CKSUM + 4].copy_from_slice(&ck.to_le_bytes());
    buf.into_boxed_slice()
}

/// Decodes a WAL segment header, returning `(store_uuid, segment_seq,
/// header_checksum)`. The checksum doubles as this segment's record chain seed.
/// Any defect → `Corrupt`.
pub fn decode_wal_header(bytes: &[u8]) -> Result<([u8; 16], u64, u32), OpenError> {
    if bytes.len() < PAGE_SIZE {
        return Err(OpenError::Corrupt("wal segment header image too short"));
    }
    let rd_u32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    if rd_u32(hdr::OFF_MAGIC) != hdr::WAL_MAGIC {
        return Err(OpenError::Corrupt("bad wal segment header magic"));
    }
    // Older or future-release segment header: refuse at open, before any mutation
    // (M6 Phase H forward policy).
    OpenError::check_open_format("WAL segment header", bytes[hdr::OFF_FORMAT], hdr::FORMAT_V2)?;
    if bytes[hdr::OFF_ALGO] != hdr::ALGO_CRC32C {
        return Err(OpenError::Corrupt("unsupported wal checksum algorithm"));
    }
    if rd_u32(hdr::OFF_PAGE_SIZE) != PAGE_SIZE as u32 {
        return Err(OpenError::Corrupt("unsupported wal page size"));
    }
    let stored = rd_u32(hdr::OFF_HEADER_CKSUM);
    if crc32c::crc32c(&bytes[..hdr::OFF_HEADER_CKSUM]) != stored {
        return Err(OpenError::Corrupt("wal segment header checksum mismatch"));
    }
    let mut uuid = [0u8; 16];
    uuid.copy_from_slice(&bytes[hdr::OFF_STORE_UUID..hdr::OFF_STORE_UUID + 16]);
    let seq = u64::from_le_bytes(
        bytes[hdr::OFF_SEGMENT_SEQ..hdr::OFF_SEGMENT_SEQ + 8]
            .try_into()
            .unwrap(),
    );
    Ok((uuid, seq, stored))
}

/// The record chain seed for WAL segment `segment_seq` of this store: its
/// identity header's checksum. The first record appended into a segment must
/// carry this as its `prev_checksum`, binding the chain to that exact segment.
pub fn seed_checksum(store_uuid: [u8; 16], segment_seq: u64) -> u32 {
    let buf = encode_wal_header(store_uuid, segment_seq);
    u32::from_le_bytes(
        buf[hdr::OFF_HEADER_CKSUM..hdr::OFF_HEADER_CKSUM + 4]
            .try_into()
            .unwrap(),
    )
}

/// Writes a fresh store's durable **data** header and `fdatasync`s it,
/// **blocking** until durable: the seq-0 checkpoint header into slot A, with a
/// zeroed slot B (an invalid slot the first checkpoint overwrites). It names WAL
/// segment 0, so a never-checkpointed store is exactly the `checkpoint_seq == 0`
/// case of the general recovery path.
pub fn write_data_header_blocking(
    data: &Arc<dyn PageIo>,
    store_uuid: [u8; 16],
) -> std::io::Result<()> {
    let header = crate::checkpoint::CheckpointHeader::fresh(
        store_uuid,
        WAL_HEADER_LEN,
        seed_checksum(store_uuid, 0),
    );
    futures_lite::future::block_on(async {
        data.write_all_at(0, header.encode()).await.result?;
        data.write_all_at(PAGE_SIZE as u64, vec![0u8; PAGE_SIZE].into_boxed_slice())
            .await
            .result?;
        data.datasync().await
    })
}

/// Writes both of a fresh store's durable headers (the data header plus segment
/// 0's identity header) and `fdatasync`s both. For the **unowned** create path,
/// where the caller supplied the segment-0 device directly; a file-set-owning
/// store's segment 0 is minted (header and all) by
/// [`WalFileSet::create_segment`](crate::fileset::WalFileSet::create_segment), so
/// that path writes only the data header. Called by the durable-create protocol
/// before a create is acknowledged.
pub fn write_headers_blocking(
    data: &Arc<dyn PageIo>,
    wal: &Arc<dyn PageIo>,
    store_uuid: [u8; 16],
) -> std::io::Result<()> {
    write_data_header_blocking(data, store_uuid)?;
    futures_lite::future::block_on(async {
        wal.write_all_at(0, encode_wal_header(store_uuid, 0))
            .await
            .result?;
        wal.datasync().await
    })
}

// -- Record codec -----------------------------------------------------------

mod rec {
    pub const MAGIC: u32 = 0x4D50_5352; // "MP5R"
    pub const FORMAT_V1: u8 = 1;

    // Fixed head: magic u32 | format u8 | rsvd[3] | incarnation u64 | txid u64
    //             | prev_cksum u32 | page_count u32 | payload_len u64
    pub const HEAD_LEN: usize = 40;
    // Per-page section prefix: page_id u64 | page_len u32 | page_cksum u32.
    pub const PAGE_OVERHEAD: usize = 16;
    // Tail: root_id u64 | root_cksum u32 | logical_tail u64 | entry_count u64
    //       | rec_cksum u32.
    pub const TAIL_LEN: usize = 32;
    // Trailer: tr_txid u64 | tr_rec_len u64 | tr_cksum u32.
    pub const TRAILER_LEN: usize = 20;

    pub const OFF_MAGIC: usize = 0;
    pub const OFF_FORMAT: usize = 4;
    pub const OFF_INCARNATION: usize = 8;
    pub const OFF_TXID: usize = 16;
    pub const OFF_PREV_CKSUM: usize = 24;
    pub const OFF_PAGE_COUNT: usize = 28;
    pub const OFF_PAYLOAD_LEN: usize = 32;
    // Fixed overhead of a record regardless of payload size.
    pub const FIXED: usize = HEAD_LEN + TAIL_LEN + TRAILER_LEN; // 92
}

/// The fixed per-record overhead (head + tail + trailer), for the admission
/// worst-case WAL-byte estimate (`batch::cost`).
pub const RECORD_FIXED_OVERHEAD: u64 = rec::FIXED as u64;
/// The per-page section overhead inside a record payload.
pub const PAGE_SECTION_OVERHEAD: u64 = rec::PAGE_OVERHEAD as u64;

/// The exact on-disk length of the record encoding `pages` (page count and
/// lengths fully determine it — no serialization needed). Used to reconcile the
/// WAL-byte reservation to its exact figure before publication.
pub fn record_len(pages: &[Arc<Page>]) -> u64 {
    let payload: u64 = pages
        .iter()
        .map(|p| rec::PAGE_OVERHEAD as u64 + p.bytes().len() as u64)
        .sum();
    rec::FIXED as u64 + payload
}

/// Where a page's *bytes* landed inside an encoded record buffer, so the overlay
/// can map `page_id → (wal offset, len)` after the buffer is written at `base`.
struct PageLayout {
    page_id: PageId,
    /// Offset of the page bytes within the record buffer.
    buf_offset: usize,
    len: u32,
}

/// One encoded record ready to append: its bytes, the per-page byte layout (for
/// the overlay), and its record checksum (the new chain head after acceptance).
struct EncodedRecord {
    buf: Box<[u8]>,
    layout: Vec<PageLayout>,
    rec_cksum: u32,
}

/// Encodes one batch into a chained record. `prev_cksum` seeds the chain;
/// `frontier` is the allocator's next page id after this batch built (the
/// id-space frontier, recorded so recovery seeds the reopened allocator past
/// every allocated id).
///
/// `incarnation` is **diagnostic only** and deliberately not validated by
/// recovery: it is a fresh random per-open nonce, and a valid log's records
/// legitimately span many prior opens, so any equality/ordering rule would
/// reject acknowledged commits. Its bytes are covered by the record CRC like
/// every other field. A recovery implementation that rejects a record because
/// of the `incarnation` *value*, rather than a checksum failure, is a bug. Do
/// not add semantics here without a record-format bump.
fn encode_record(
    prev_cksum: u32,
    incarnation: u64,
    txid: u64,
    root: PageRef,
    frontier: u64,
    entry_count: u64,
    pages: &[Arc<Page>],
) -> EncodedRecord {
    let payload_len: usize = pages
        .iter()
        .map(|p| rec::PAGE_OVERHEAD + p.bytes().len())
        .sum();
    let total = rec::FIXED + payload_len;
    let mut buf = vec![0u8; total];

    buf[rec::OFF_MAGIC..rec::OFF_MAGIC + 4].copy_from_slice(&rec::MAGIC.to_le_bytes());
    buf[rec::OFF_FORMAT] = rec::FORMAT_V1;
    buf[rec::OFF_INCARNATION..rec::OFF_INCARNATION + 8].copy_from_slice(&incarnation.to_le_bytes());
    buf[rec::OFF_TXID..rec::OFF_TXID + 8].copy_from_slice(&txid.to_le_bytes());
    buf[rec::OFF_PREV_CKSUM..rec::OFF_PREV_CKSUM + 4].copy_from_slice(&prev_cksum.to_le_bytes());
    buf[rec::OFF_PAGE_COUNT..rec::OFF_PAGE_COUNT + 4]
        .copy_from_slice(&(pages.len() as u32).to_le_bytes());
    buf[rec::OFF_PAYLOAD_LEN..rec::OFF_PAYLOAD_LEN + 8]
        .copy_from_slice(&(payload_len as u64).to_le_bytes());

    let mut layout = Vec::with_capacity(pages.len());
    let mut cur = rec::HEAD_LEN;
    for p in pages {
        let bytes = p.bytes();
        buf[cur..cur + 8].copy_from_slice(&p.id().as_u64().to_le_bytes());
        buf[cur + 8..cur + 12].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        buf[cur + 12..cur + 16].copy_from_slice(&p.checksum().0.to_le_bytes());
        let boff = cur + rec::PAGE_OVERHEAD;
        buf[boff..boff + bytes.len()].copy_from_slice(bytes);
        layout.push(PageLayout {
            page_id: p.id(),
            buf_offset: boff,
            len: bytes.len() as u32,
        });
        cur = boff + bytes.len();
    }
    debug_assert_eq!(cur, rec::HEAD_LEN + payload_len);

    // Tail (up to, but excluding, the record checksum).
    let tail = cur;
    buf[tail..tail + 8].copy_from_slice(&root.page_id.as_u64().to_le_bytes());
    buf[tail + 8..tail + 12].copy_from_slice(&root.expected_checksum.0.to_le_bytes());
    buf[tail + 12..tail + 20].copy_from_slice(&frontier.to_le_bytes());
    buf[tail + 20..tail + 28].copy_from_slice(&entry_count.to_le_bytes());
    let cksum_pos = tail + 28;

    // Record checksum chains from prev_cksum over everything before the field.
    let rec_cksum = crc32c::crc32c_append(prev_cksum, &buf[..cksum_pos]);
    buf[cksum_pos..cksum_pos + 4].copy_from_slice(&rec_cksum.to_le_bytes());

    // Commit trailer.
    let tr = cksum_pos + 4;
    buf[tr..tr + 8].copy_from_slice(&txid.to_le_bytes());
    buf[tr + 8..tr + 16].copy_from_slice(&(total as u64).to_le_bytes());
    buf[tr + 16..tr + 20].copy_from_slice(&rec_cksum.to_le_bytes());
    debug_assert_eq!(tr + rec::TRAILER_LEN, total);

    EncodedRecord {
        buf: buf.into_boxed_slice(),
        layout,
        rec_cksum,
    }
}

/// Offset of a page image's own `page_id` field (see `page::header`), used to
/// cross-check a WAL section's declared id against the id inside its bytes.
const PAGE_IMG_ID_OFF: usize = 8;

/// A single record may not exceed this many bytes; a larger declared length is
/// treated as a torn/corrupt tail (bounds the per-record read so a corrupt
/// `st_size` cannot drive an unbounded allocation). Far above any realistic
/// batch (default `max_dirty_pages·(PAGE_SIZE+overhead) ≈ 67 MiB`). The
/// **writer** enforces the same cap before appending (`record_len` check in the
/// writer), so it is never possible to acknowledge a record that a later recovery
/// would refuse to scan.
pub const MAX_RECORD_BYTES: usize = 512 * 1024 * 1024;

/// Page ids (and therefore the allocator frontier) may not reach this value, so
/// the reopened allocator always has headroom and `alloc_id` never wraps.
/// `2^48` pages is 4 EiB of data — unreachable in v1. The **writer**
/// refuses to publish a batch whose recorded frontier would reach this cap, so it
/// is never possible to acknowledge a record a later recovery (which rejects
/// `frontier >= MAX_FRONTIER`) would refuse.
pub const MAX_FRONTIER: u64 = 1 << 48;

/// Transaction ids may not reach this value, so `txid + 1` can never wrap —
/// neither in recovery's "expect `prev + 1`" nor in [`Version::next`] on the write
/// path. Unreachable by any real store (2^48 writes at a million per
/// second is nine years, and every one of them is a durable barrier), so this
/// costs nothing; it exists because a **forged or damaged header can claim any
/// txid**, and the durable trust boundary is where that must be rejected — as
/// `Corrupt`, not as a writer-thread panic on the next write, nor as a silent wrap
/// that breaks the monotonic-txid invariant in release. Enforced identically at
/// both boundaries recovery reads (the checkpoint header and the record codec) and
/// at the writer, so no acknowledged txid can ever be one a later open refuses.
pub const MAX_TXID: u64 = 1 << 48;

/// One accepted record's recovered contents.
struct AcceptedRecord {
    txid: u64,
    rec_cksum: u32,
    rec_len: usize,
    root: PageRef,
    entry_count: u64,
    /// The allocator id-space frontier as of this batch (next id > every id ever
    /// allocated up to here). Recovery seeds the reopened allocator from the last
    /// accepted record's value.
    frontier: u64,
    /// `(page_id, offset-of-bytes-in-wal-file, len, checksum)` for each page image.
    pages: Vec<(PageId, u64, u32, u32)>,
}

/// Fully validates the record whose exact bytes are `rec` (`rec.len()` must equal
/// the record's framed total), located at file offset `file_pos`, chained from
/// `prev_cksum`, expecting txid `expect_txid`, page ids in
/// `[prev_frontier, frontier)`, and `frontier ≤ max_frontier`. Returns `None` for
/// **any** framing, arithmetic, canonical, id, checksum, txid, chain, or trailer
/// failure. It never mutates shared state — the caller merges only on `Some`, so a
/// rejected record cannot leak a page into the accepted overlay.
fn try_decode_record(
    rec: &[u8],
    file_pos: u64,
    prev_cksum: u32,
    expect_txid: u64,
    prev_frontier: u64,
    max_frontier: u64,
) -> Option<AcceptedRecord> {
    if rec.len() < rec::HEAD_LEN {
        return None;
    }
    let rd_u32 = |o: usize| u32::from_le_bytes(rec[o..o + 4].try_into().unwrap());
    let rd_u64 = |o: usize| u64::from_le_bytes(rec[o..o + 8].try_into().unwrap());
    if rd_u32(rec::OFF_MAGIC) != rec::MAGIC || rec[rec::OFF_FORMAT] != rec::FORMAT_V1 {
        return None;
    }
    let txid = rd_u64(rec::OFF_TXID);
    let prev_field = rd_u32(rec::OFF_PREV_CKSUM);
    let page_count = rd_u32(rec::OFF_PAGE_COUNT) as usize;
    let payload_field = rd_u64(rec::OFF_PAYLOAD_LEN);

    // Framing must describe exactly `page_count` canonical PAGE_SIZE sections and
    // exactly fill `rec` (checked arithmetic; no lossy conversions).
    let payload_len: usize = payload_field.try_into().ok()?;
    let per_section = rec::PAGE_OVERHEAD.checked_add(PAGE_SIZE)?;
    let expect_payload = page_count.checked_mul(per_section)?;
    if payload_len != expect_payload {
        return None;
    }
    let total = rec::FIXED.checked_add(payload_len)?;
    if total != rec.len() {
        return None;
    }
    // Cheap head checks (chain + consecutive txid) before page-sized work. The
    // trailer alone is never proof of commit; anti-resurrection lives here.
    // `>= MAX_TXID` so the *successor* of an accepted record can never wrap.
    if txid != expect_txid || txid >= MAX_TXID || prev_field != prev_cksum {
        return None;
    }

    let tail = rec::HEAD_LEN + payload_len;
    let root = PageRef {
        page_id: PageId(u64::from_le_bytes(rec[tail..tail + 8].try_into().unwrap())),
        expected_checksum: Checksum(u32::from_le_bytes(
            rec[tail + 8..tail + 12].try_into().unwrap(),
        )),
    };
    let frontier = u64::from_le_bytes(rec[tail + 12..tail + 20].try_into().unwrap());
    let entry_count = u64::from_le_bytes(rec[tail + 20..tail + 28].try_into().unwrap());

    // Frontier: ≥ first data id, monotonically non-decreasing, and strictly below
    // the cap so the reopened allocator always has headroom and cannot wrap
    // (`>=` so a record at the cap — whose successor would exceed it — is
    // rejected).
    if frontier < FIRST_DATA_INDEX || frontier < prev_frontier || frontier >= max_frontier {
        return None;
    }
    // Canonical root/entry-count relationship (mirrors the Direct header rule).
    if root.is_empty() {
        if root.expected_checksum != Checksum(0) || entry_count != 0 {
            return None;
        }
    } else if root.page_id.as_u64() < FIRST_DATA_INDEX
        || root.page_id.as_u64() >= frontier
        || entry_count == 0
    {
        return None;
    }

    // Walk the payload: each section's canonical length, its id (non-reserved, in
    // [prev_frontier, frontier) so it can never alias an earlier record's id, and
    // unique in this record), the id encoded *inside* the page image, and its
    // checksum.
    let mut pages = Vec::with_capacity(page_count);
    let mut seen_ids = std::collections::HashSet::with_capacity(page_count);
    let mut cur = rec::HEAD_LEN;
    for _ in 0..page_count {
        let page_id = u64::from_le_bytes(rec[cur..cur + 8].try_into().unwrap());
        let page_len = u32::from_le_bytes(rec[cur + 8..cur + 12].try_into().unwrap()) as usize;
        let page_ck = u32::from_le_bytes(rec[cur + 12..cur + 16].try_into().unwrap());
        if page_len != PAGE_SIZE {
            return None; // v1: every page image is exactly PAGE_SIZE
        }
        if page_id < FIRST_DATA_INDEX
            || page_id < prev_frontier
            || page_id >= frontier
            || !seen_ids.insert(page_id)
        {
            return None; // reserved / out-of-range / cross-record reuse / duplicate
        }
        let boff = cur + rec::PAGE_OVERHEAD;
        let bend = boff + page_len; // canonical, already bounded by expect_payload
                                    // The id declared by the section must match the id inside the page image,
                                    // so a mislabeled section cannot install a wrong-id overlay entry.
        if u64::from_le_bytes(
            rec[boff + PAGE_IMG_ID_OFF..boff + PAGE_IMG_ID_OFF + 8]
                .try_into()
                .unwrap(),
        ) != page_id
        {
            return None;
        }
        if crc32c::crc32c(&rec[boff..bend]) != page_ck {
            return None; // page image corrupt
        }
        pages.push((
            PageId(page_id),
            file_pos + boff as u64,
            page_len as u32,
            page_ck,
        ));
        cur = bend;
    }
    debug_assert_eq!(cur, tail);

    // Record checksum (chained) then the commit trailer (must agree with body).
    let cksum_pos = tail + 28;
    let stored_rec_cksum = u32::from_le_bytes(rec[cksum_pos..cksum_pos + 4].try_into().unwrap());
    if crc32c::crc32c_append(prev_cksum, &rec[..cksum_pos]) != stored_rec_cksum {
        return None;
    }
    let tr = cksum_pos + 4;
    let tr_txid = u64::from_le_bytes(rec[tr..tr + 8].try_into().unwrap());
    let tr_len = u64::from_le_bytes(rec[tr + 8..tr + 16].try_into().unwrap());
    let tr_cksum = u32::from_le_bytes(rec[tr + 16..tr + 20].try_into().unwrap());
    if tr_txid != txid || tr_len != total as u64 || tr_cksum != stored_rec_cksum {
        return None;
    }

    Some(AcceptedRecord {
        txid,
        rec_cksum: stored_rec_cksum,
        rec_len: total,
        root,
        entry_count,
        frontier,
        pages,
    })
}

// -- Overlay locator --------------------------------------------------------

/// The StoreWal page locator: a **composite** of the checkpoint base and the
/// runtime WAL overlay.
///
/// - `base` — `page_id → data-file offset` for every page a checkpoint relocated
///   into the data file, seeded once at open from the winning header's manifest
///   and immutable thereafter (a checkpoint builds a *new* locator and the writer
///   rebinds to it, rather than mutating a live one).
/// - `entries` — `page_id → (WAL offset, len)` for pages committed into the
///   current WAL segment. Recovery seeds it from the accepted record prefix; each
///   commit extends it *before* the dirty pages unpin (invariant 5).
///
/// Page ids are globally unique and never reused, so a given id lives in exactly
/// one of the two; the overlay is consulted first regardless. A cache miss
/// resolves through `locate` and reads the image from the named member.
pub struct WalLocator {
    base: std::collections::HashMap<u64, u64>,
    entries: parking_lot::RwLock<std::collections::HashMap<u64, Location>>,
}

impl WalLocator {
    fn new() -> Self {
        WalLocator::with_base(std::collections::HashMap::new())
    }

    /// A locator over `base` (`page_id → data-file offset`, from a checkpoint
    /// manifest) with an empty WAL overlay.
    pub(crate) fn with_base(base: std::collections::HashMap<u64, u64>) -> Self {
        WalLocator {
            base,
            entries: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    fn record(&self, page_id: u64, offset: u64, len: u32) {
        self.entries.write().insert(
            page_id,
            Location {
                member: Member::Wal,
                offset,
                length: len,
            },
        );
    }

    /// Number of **WAL overlay** entries — pages committed into the current
    /// segment since the last checkpoint cutover (metric `overlay_pages`).
    /// Checkpointed base pages are not counted.
    pub fn len(&self) -> u64 {
        self.entries.read().len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Number of checkpointed pages resolvable in the data file (metric
    /// `checkpoint_pages`).
    pub fn base_len(&self) -> u64 {
        self.base.len() as u64
    }
}

impl WalLocator {
    /// The id-only resolution both [`PageLocator::locate`] and the prefetch
    /// capability share: overlay entry first (member=Wal, record-interior
    /// offset), then the checkpointed base (member=Data). Neither map consults
    /// the ref's checksum, so the two entry points resolve identically.
    fn locate_id(&self, id: u64) -> Option<Location> {
        if let Some(loc) = self.entries.read().get(&id) {
            return Some(*loc);
        }
        self.base.get(&id).map(|offset| Location {
            member: Member::Data,
            offset: *offset,
            length: PAGE_SIZE as u32,
        })
    }
}

impl PageLocator for WalLocator {
    fn locate(&self, page_ref: PageRef) -> Option<Location> {
        self.locate_id(page_ref.page_id.as_u64())
    }

    /// WAL declares the chain-prefetch capability **for the experiment only**
    /// (T2.2): chain ids stay consecutive across records and checkpoints
    /// (relocation moves offsets, never ids), and both maps resolve by bare id.
    /// Shipping prefetch default-on for WAL additionally requires a named WAL
    /// workload to pass the predictor gate on its own — a StoreDirect win does
    /// not transfer.
    fn predict_overflow_location(&self, page_id: PageId) -> Option<Location> {
        self.locate_id(page_id.as_u64())
    }
}

// -- Generation construction ------------------------------------------------

/// Builds a StoreWal `FileGeneration` (data + WAL devices, overlay locator) and
/// its paired page-id [`Allocator`], seeded to start allocating at page id
/// `next_index`. Returns the shared [`WalLocator`] so recovery/commit can extend
/// it. The allocator's `tail` frontier is unused by the overlay locator.
pub fn wal_generation(
    data: Arc<dyn PageIo>,
    wal: Arc<dyn PageIo>,
    next_index: u64,
) -> (FileGeneration, Allocator, Arc<WalLocator>) {
    let next_index = next_index.max(FIRST_DATA_INDEX);
    let locator = Arc::new(WalLocator::new());
    let file = FileGeneration::wal(data, wal, locator.clone());
    let alloc = Allocator::for_wal(next_index);
    (file, alloc, locator)
}

// -- Recovery ---------------------------------------------------------------

/// The result of resolving the winning data header and scanning its named WAL
/// segment for the longest valid committed prefix.
pub struct Recovered {
    pub store_uuid: [u8; 16],
    pub root: PageRef,
    pub entry_count: u64,
    /// Last accepted txid — the checkpoint txid `T` if no post-checkpoint record
    /// was accepted (0 for a never-checkpointed store with an empty segment).
    pub last_txid: u64,
    /// Chain head after the accepted prefix (the segment's chain seed if none).
    pub chain_head: u32,
    /// Byte offset immediately after the accepted prefix (where the next append
    /// overwrites; never after the torn tail, never truncating).
    pub append_offset: u64,
    /// The rebuilt locator: the manifest base plus the accepted overlay.
    pub locator: Arc<WalLocator>,
    /// The id-space frontier — `max(checkpoint frontier, last accepted record's
    /// frontier)` — to seed the allocator so no id is ever reused.
    pub next_page_id: u64,
    /// WAL segment size (seeds `wal_physical_bytes`).
    pub scanned_bytes: u64,
    /// The winning header's checkpoint sequence number (0 = never checkpointed).
    pub checkpoint_seq: u64,
    /// The checkpoint txid `T`: every WAL record at or below it is superseded by
    /// the data file and is not scanned.
    pub checkpoint_txid: u64,
    /// The WAL segment the winning header names as holding records `> T`.
    pub wal_segment_seq: u64,
}

/// Resolves the winning data header, validates that `wal` is exactly the segment
/// it names, and scans that segment from the header's scan origin for the longest
/// valid committed prefix of records `> checkpoint_txid`. Read-only and
/// idempotent (frozen contract).
///
/// All reset state comes from the **one** winning header: there is no cross-file
/// sequence comparison and no WAL-header mutation. The header's manifest seeds
/// the locator base, the root-resolve checksum set, and the allocator frontier —
/// which is what stops a post-checkpoint zero-page (no-op) commit from starving
/// the root-resolve check and stops page-id reuse across a checkpoint. A
/// manifest that fails validation is `Corrupt` (fail open); recovery never
/// falls back to the older slot, whose superseded WAL segment may already be
/// unlinked.
pub fn recover(data: &Arc<dyn PageIo>, wal: &Arc<dyn PageIo>) -> Result<Recovered, OpenError> {
    futures_lite::future::block_on(async {
        // 1. The winning data header: highest fully-valid `checkpoint_seq` across
        //    the two slots (a torn just-written slot loses to its predecessor).
        let h = crate::checkpoint::read_winning_header(&**data).await?;

        // 2. Segment identity: the WAL handed to us must be *the* segment the
        //    header names, chained from the seed it recorded. A mismatch means the
        //    caller paired the wrong files (or a stale segment survived a
        //    cutover) — never scan it as if it were the right one.
        let wal_size = wal.size().await?;
        if wal_size < WAL_HEADER_LEN {
            return Err(OpenError::Corrupt("wal segment shorter than its header"));
        }
        let header_bytes = wal.read_exact_at(0, PAGE_SIZE).await?;
        let (wal_uuid, segment_seq, seed) = decode_wal_header(&header_bytes)?;
        if h.store_uuid != wal_uuid {
            return Err(OpenError::UuidMismatch);
        }
        if segment_seq != h.wal_segment_seq {
            return Err(OpenError::Corrupt(
                "wal segment seq is not the one the data header names",
            ));
        }
        if seed != h.wal_chain_seed {
            return Err(OpenError::Corrupt(
                "wal segment chain seed disagrees with the data header",
            ));
        }
        // The origin must name a real position in *this* segment. Every header the
        // engine writes uses exactly `WAL_HEADER_LEN`, so anything else is
        // corruption; an unbounded origin would otherwise open "successfully" and
        // seed the append frontier and meter past the end of the file, failing
        // every subsequent write with `StoreFull`.
        if h.wal_scan_origin < WAL_HEADER_LEN || h.wal_scan_origin > wal_size {
            return Err(OpenError::Corrupt("wal scan origin outside the segment"));
        }

        // 3. The manifest: seeds the locator base, the root-resolve checksums, and
        //    (with the header's `F`) the allocator.
        let manifest = crate::checkpoint::read_manifest(&**data, &h).await?;
        let mut base = std::collections::HashMap::with_capacity(manifest.len());
        let mut checksums: std::collections::HashMap<u64, u32> =
            std::collections::HashMap::with_capacity(manifest.len());
        for e in &manifest {
            base.insert(e.page_id.as_u64(), e.offset);
            checksums.insert(e.page_id.as_u64(), e.checksum.0);
        }
        // The checkpoint relocated the whole live set of `R@T`, so `R@T` itself
        // must resolve in the manifest by id *and* checksum. (The page image is
        // validated lazily, at the decode trust boundary, like every other page.)
        if !h.root.is_empty()
            && checksums.get(&h.root.page_id.as_u64()) != Some(&h.root.expected_checksum.0)
        {
            return Err(OpenError::Corrupt(
                "checkpoint root does not resolve in the manifest",
            ));
        }
        let locator = Arc::new(WalLocator::with_base(base));

        // 4. Scan records from the header's scan origin, reading each record's
        //    bytes incrementally (bounded per record — never a
        //    single allocation driven by a corrupt `st_size`). A read that hits
        //    EOF is a torn tail (stop); any other device error propagates.
        let mut file_pos = h.wal_scan_origin;
        let mut prev_cksum = h.wal_chain_seed;
        let mut prev_txid = h.checkpoint_txid;
        let mut root = h.root;
        let mut entry_count = h.entry_count;
        let mut frontier = h.allocator_frontier;
        loop {
            // Read the fixed head to learn the framed length.
            let head = match wal.read_exact_at(file_pos, rec::HEAD_LEN).await {
                Ok(h) => h,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(OpenError::from(e)),
            };
            let page_count = u32::from_le_bytes(
                head[rec::OFF_PAGE_COUNT..rec::OFF_PAGE_COUNT + 4]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let payload_field = u64::from_le_bytes(
                head[rec::OFF_PAYLOAD_LEN..rec::OFF_PAYLOAD_LEN + 8]
                    .try_into()
                    .unwrap(),
            );
            // Bound the per-record read before allocating/reading it.
            let Some(total) = usize::try_from(payload_field)
                .ok()
                .and_then(|p| p.checked_add(rec::FIXED))
            else {
                break;
            };
            if total > MAX_RECORD_BYTES
                || page_count > total
                || (total as u64) > wal_size.saturating_sub(file_pos)
            {
                break; // torn/corrupt/oversized → stop
            }
            let rec_bytes = match wal.read_exact_at(file_pos, total).await {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(OpenError::from(e)),
            };
            let Some(r) = try_decode_record(
                &rec_bytes,
                file_pos,
                prev_cksum,
                prev_txid + 1,
                frontier,
                MAX_FRONTIER,
            ) else {
                break;
            };
            // Per-record root-resolves predicate *before* merging: a
            // non-empty root must resolve — by id **and** expected checksum — in
            // this record's pages or an already-accepted one. A record that fails
            // this stops the scan and leaves the prior accepted prefix intact,
            // rather than making the whole open `Corrupt`.
            if !r.root.is_empty() {
                let id = r.root.page_id.as_u64();
                let ck = r
                    .pages
                    .iter()
                    .find(|p| p.0.as_u64() == id)
                    .map(|p| p.3)
                    .or_else(|| checksums.get(&id).copied());
                if ck != Some(r.root.expected_checksum.0) {
                    break;
                }
            }
            // Merge the fully-validated record atomically (nothing above mutated
            // shared state).
            for (pid, off, len, ck) in &r.pages {
                locator.record(pid.as_u64(), *off, *len);
                checksums.insert(pid.as_u64(), *ck);
            }
            prev_cksum = r.rec_cksum;
            prev_txid = r.txid;
            root = r.root;
            entry_count = r.entry_count;
            frontier = r.frontier;
            file_pos += r.rec_len as u64;
        }

        Ok(Recovered {
            store_uuid: h.store_uuid,
            root,
            entry_count,
            last_txid: prev_txid,
            chain_head: prev_cksum,
            append_offset: file_pos,
            locator,
            next_page_id: frontier.max(FIRST_DATA_INDEX),
            scanned_bytes: wal_size,
            checkpoint_seq: h.checkpoint_seq,
            checkpoint_txid: h.checkpoint_txid,
            wal_segment_seq: h.wal_segment_seq,
        })
    })
}

/// The WAL header length (records start here; seeds the WAL meter's logical
/// frontier).
pub const WAL_HEADER_LEN: u64 = PAGE_SIZE as u64;

// -- Coordinator ------------------------------------------------------------

/// One published-but-not-yet-committed batch, retained until a group commit
/// appends and syncs its record.
pub(crate) struct PendingWal {
    txid: u64,
    root: PageRef,
    entry_count: u64,
    /// Allocator id-space frontier as of this batch (encoded in its record so
    /// recovery seeds the reopened allocator past every id).
    frontier: u64,
    pages: Vec<Arc<Page>>,
    page_charge: u32,
    /// The reconciled WAL-byte reservation, committed at the durable frontier.
    wal_reservation: ReconciledWalReservation,
}

/// Everything a [`WalCoordinator`] needs beyond the append state: the data
/// device it checkpoints into, the identity it stamps, and the namespace it
/// rotates (absent for a store over caller-supplied devices, which cannot
/// checkpoint).
pub(crate) struct WalBinding {
    pub data: Arc<dyn PageIo>,
    pub store_uuid: [u8; 16],
    pub file_set: Option<Arc<dyn crate::fileset::WalFileSet>>,
    /// The data-path namespace owner (M6 Phase D): present only for stores that
    /// own their data path; a compaction rewrites the data file through it.
    pub data_host: Option<Arc<dyn crate::compact::DataFileHost>>,
    /// The segment currently being appended to.
    pub segment_seq: u64,
    /// The winning header's checkpoint sequence (0 = never checkpointed).
    pub checkpoint_seq: u64,
    /// Next data-file append offset: the file's current length, page-aligned up.
    /// The data file only ever grows within an open, so this is trivially past
    /// **every** extent either header slot or any retained generation references
    /// — including orphan extents a crashed checkpoint left behind.
    /// Reclaiming those orphans is Phase D's dense rebuild, not B's job.
    pub data_frontier: u64,
}

/// Why a live-set relocation failed: a read of the live tree (→ the caller
/// poisons, the committed state is unreadable) vs a write to the relocation
/// target (the checkpoint's shared data file → poison; a compaction's private
/// temp file → healthy abort, M6 Phase D).
enum RelocFail {
    Read(crate::error::ReadError),
    Target(std::io::Error),
}

/// A relocated live set: the sorted manifest, its blob's placement, and the
/// end offset of everything written.
struct RelocatedSet {
    manifest: Vec<crate::checkpoint::ManifestEntry>,
    manifest_offset: u64,
    manifest_len: u64,
    manifest_cksum: u32,
    end: u64,
}

/// Relocates the live page set of `root` into `target` starting at `start`,
/// then appends the manifest blob describing where each page landed. Streamed:
/// each page is written and dropped before the next is loaded, so peak memory
/// is one page image plus the manifest — not the whole live set, which would
/// stop a store larger than RAM from ever checkpointing. Shared by
/// the checkpoint (target = the data file, past every referenced extent) and
/// the compaction (target = a fresh temp file, from the first data page).
/// Page ids are preserved: the locator is a table, and preserved ids keep every
/// shared-cache entry valid for the old and new generations alike.
async fn relocate_live_set(
    cache: &crate::cache::PageCache,
    file: &crate::backend::FileGeneration,
    root: PageRef,
    target: &Arc<dyn PageIo>,
    start: u64,
) -> Result<RelocatedSet, RelocFail> {
    use crate::checkpoint::{encode_manifest, ManifestEntry};
    let mut walk = crate::read::LivePages::new(root);
    let mut manifest: Vec<ManifestEntry> = Vec::new();
    let mut off = start;
    loop {
        let page = match walk.next(cache, file).await {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(RelocFail::Read(e)),
        };
        let buf = page.bytes().to_vec().into_boxed_slice();
        manifest.push(ManifestEntry {
            page_id: page.id(),
            offset: off,
            checksum: page.checksum(),
        });
        drop(page); // release the image before the write's await
        target
            .write_all_at(off, buf)
            .await
            .result
            .map_err(RelocFail::Target)?;
        off += PAGE_SIZE as u64;
    }
    // The manifest codec requires strictly increasing ids (recovery validates
    // it); the walk order is breadth-first, so sort here.
    manifest.sort_unstable_by_key(|e| e.page_id.as_u64());
    let (blob, manifest_cksum) = encode_manifest(&manifest);
    let manifest_offset = off;
    let manifest_len = blob.len() as u64;
    if manifest_len > 0 {
        // Pad the *physical* manifest write up to a full page (M7-A).
        // O_DIRECT requires a page-multiple length, but the manifest's true length is
        // `entries * MANIFEST_ENTRY_LEN` — rarely a page multiple. The header records
        // the *true* `manifest_len` (the codec and checksum cover only the real
        // bytes); the zero pad is dead tail an O_DIRECT open reads and then slices
        // off. Padding unconditionally (not only under `direct_io`) keeps one write
        // path and makes every file O_DIRECT-readable regardless of the mode it was
        // written in — `direct_io` is orthogonal, so a buffered-written file may be
        // reopened O_DIRECT. It is free in address space: the frontier already rounds
        // `off` up to a page (here via `end` and at both callers), so the pad fills
        // bytes that would otherwise be skipped (it does extend `i_size` by up to one
        // page of real write traffic).
        let padded_len = (manifest_len as usize).next_multiple_of(PAGE_SIZE);
        let mut padded = blob;
        padded.resize(padded_len, 0);
        target
            .write_all_at(manifest_offset, padded.into_boxed_slice())
            .await
            .result
            .map_err(RelocFail::Target)?;
        off += padded_len as u64;
    }
    Ok(RelocatedSet {
        manifest,
        manifest_offset,
        manifest_len,
        manifest_cksum,
        end: off,
    })
}

/// Why a checkpoint declined to run. A **healthy** outcome, not an error: the
/// store keeps serving reads and writes (M6 Phase F).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpaceSkip {
    /// The configured `max_data_bytes`.
    pub limit: u64,
    /// What the store would have owned had this checkpoint run, including the
    /// maintenance headroom a later compaction needs.
    pub projected: u64,
}

/// What one checkpoint did (returned to the caller, and the source of the
/// checkpoint metrics).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointReport {
    /// `Some` if this checkpoint **skipped** on `max_data_bytes` and did nothing:
    /// no relocation, no rotation, no WAL reclaimed, and every other field zero.
    ///
    /// A caller must not read a successful return as "the WAL was reset" — an
    /// operator who checkpoints to make room and then retries a write would
    /// otherwise be surprised by `StoreFull`.
    pub skipped_space: Option<SpaceSkip>,
    /// The new checkpoint sequence number.
    pub checkpoint_seq: u64,
    /// The txid `T` the data file now holds outright: every WAL record `<= T` is
    /// superseded and will never be scanned again.
    pub checkpoint_txid: u64,
    /// Live pages relocated into the data file.
    pub relocated_pages: u64,
    /// WAL bytes the rotation dropped: the retired segment's append frontier.
    /// The bytes free once the last pre-checkpoint snapshot holding that
    /// (unlinked) segment drops — see [`crate::fileset`].
    pub retired_wal_bytes: u64,
}

/// The StoreWal append coordinator. Runs inside the writer thread (like Direct's
/// coordinator), so append order and the chain are serialized trivially — and so is the
/// checkpoint, which is why it needs no separate quiescence protocol.
pub(crate) struct WalCoordinator {
    wal: Arc<dyn PageIo>,
    incarnation: u64,
    alloc: Allocator,
    locator: Arc<WalLocator>,
    /// Next WAL append offset — kept identical to `WalMeter.logical`.
    append_offset: u64,
    /// Accepted chain head (record checksum of the last committed record, or the
    /// current segment's chain seed).
    prev_checksum: u32,
    committed_txid: u64,
    pending: VecDeque<PendingWal>,
    dirty_pages: usize,
    high_water_pages: usize,
    bind: WalBinding,
}

impl WalCoordinator {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        wal: Arc<dyn PageIo>,
        incarnation: u64,
        alloc: Allocator,
        locator: Arc<WalLocator>,
        append_offset: u64,
        chain_head: u32,
        committed_txid: u64,
        high_water_pages: usize,
        bind: WalBinding,
    ) -> Self {
        WalCoordinator {
            wal,
            incarnation,
            alloc,
            locator,
            append_offset,
            prev_checksum: chain_head,
            committed_txid,
            pending: VecDeque::new(),
            dirty_pages: 0,
            high_water_pages: high_water_pages.max(1),
            bind,
        }
    }

    pub(crate) fn allocator(&mut self) -> &mut Allocator {
        &mut self.alloc
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub(crate) fn on_published(
        &mut self,
        txid: u64,
        root: PageRef,
        entry_count: u64,
        pages: Vec<Arc<Page>>,
        page_charge: u32,
        wal_reservation: ReconciledWalReservation,
    ) {
        // Capture the allocator frontier right after this batch built (it drew its
        // ids through `self.alloc`), so the record records the id-space frontier
        // *for this batch* — including ids burned by superseded pages.
        let frontier = self.alloc.peek_next();
        self.dirty_pages += pages.len();
        self.pending.push_back(PendingWal {
            txid,
            root,
            entry_count,
            frontier,
            pages,
            page_charge,
            wal_reservation,
        });
    }

    pub(crate) fn should_flush(&self, requested_txid: u64) -> bool {
        self.has_pending()
            && (self.dirty_pages >= self.high_water_pages
                || requested_txid > self.committed_txid
                || self.pending.len() >= PENDING_FLUSH_CAP)
    }

    /// Group-commits every pending batch: encode chained records, append them,
    /// one `fdatasync`, then install the overlay, advance the frontier/chain,
    /// commit each WAL reservation, release dirty pins/budget, and wake waiters.
    /// Any WAL write/sync error poisons and the committed frontier never advances
    /// past it (invariants 9/17).
    pub(crate) async fn commit_all(&mut self, core: &Arc<Core>) -> Result<(), Terminal> {
        if self.pending.is_empty() {
            return Ok(());
        }
        // 1. Encode each record at its assigned offset, chaining the checksum
        //    (all chain math complete before any write). Keep each record's page
        //    layout for the post-sync overlay install; the buffers are handed to
        //    the device by value (no clone).
        let mut writes: Vec<(u64, Box<[u8]>)> = Vec::with_capacity(self.pending.len());
        let mut installs: Vec<(u64, Vec<PageLayout>)> = Vec::with_capacity(self.pending.len());
        let mut offset = self.append_offset;
        let mut chain = self.prev_checksum;
        for batch in &self.pending {
            let er = encode_record(
                chain,
                self.incarnation,
                batch.txid,
                batch.root,
                batch.frontier,
                batch.entry_count,
                &batch.pages,
            );
            chain = er.rec_cksum;
            let this_off = offset;
            offset += er.buf.len() as u64;
            installs.push((this_off, er.layout));
            writes.push((this_off, er.buf));
        }
        // 2. Submit every write; only after all complete, one fdatasync. Advance
        //    the physical-length high-water mark after each successful write (and,
        //    on error, refresh it from the real file size, since a partial write
        //    may have extended the file). Never used for admission.
        for (off, buf) in writes {
            let end = off + buf.len() as u64;
            if let Err(e) = self.wal.write_all_at(off, buf).await.result {
                if let Ok(sz) = self.wal.size().await {
                    let prev = core.wal_physical_bytes.load(Ordering::Acquire);
                    core.wal_physical_bytes
                        .store(prev.max(sz), Ordering::Release);
                }
                return Err(self.poison_io(core, e));
            }
            let prev = core.wal_physical_bytes.load(Ordering::Acquire);
            core.wal_physical_bytes
                .store(prev.max(end), Ordering::Release);
        }
        if let Err(e) = self.wal.datasync().await {
            return Err(self.poison_io(core, e));
        }
        // 3. Durable. Install overlay entries (before unpinning — invariant 5).
        for (base, layout) in &installs {
            for pl in layout {
                self.locator
                    .record(pl.page_id.as_u64(), base + pl.buf_offset as u64, pl.len);
            }
        }
        // 4. Advance frontier/chain; commit reservations; release pins/budget.
        self.append_offset = offset;
        self.prev_checksum = chain;
        let mut last_txid = self.committed_txid;
        for batch in self.pending.drain(..) {
            for page in &batch.pages {
                core.cache.release_dirty(page.id());
            }
            if batch.page_charge > 0 {
                core.budget.release_pages(batch.page_charge);
            }
            batch.wal_reservation.commit();
            last_txid = batch.txid;
        }
        self.committed_txid = last_txid;
        self.dirty_pages = 0;
        core.durable_txid.store(last_txid, Ordering::Release);
        core.state.event().notify(usize::MAX);
        Ok(())
    }

    fn poison_io(&self, core: &Arc<Core>, err: std::io::Error) -> Terminal {
        core.state.poison(PoisonReason::Io(Arc::new(err)));
        match core.state.check_terminal() {
            Err(t) => t,
            Ok(()) => Terminal::Poisoned(PoisonReason::Internal("poison did not stick")),
        }
    }

    fn poison_with(&self, core: &Arc<Core>, reason: PoisonReason) -> Terminal {
        core.state.poison(reason);
        match core.state.check_terminal() {
            Err(t) => t,
            Ok(()) => Terminal::Poisoned(PoisonReason::Internal("poison did not stick")),
        }
    }

    /// A read failure on the checkpoint's relocation walk: a corrupt or
    /// unreadable page in the *live* set means the writer cannot continue (it
    /// could not honour a later read of that page either), so it poisons —
    /// exactly like the same failure on the write path (`writer::fail_from_read`).
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
            // The relocation walk reads through the generation's own strong
            // `file` handle, never a snapshot lease — unreachable, and loud if
            // a refactor ever misroutes it (M6 Phase E).
            ReadError::SnapshotExpired => self
                .poison_with(
                    core,
                    PoisonReason::Internal("snapshot lease on checkpoint path"),
                )
                .into(),
        }
    }

    /// Runs one checkpoint (M6 Phase B), **on the writer thread, at a
    /// drained and fully-committed barrier**: relocate the live page set of
    /// `R@T` into the data file, rotate to a fresh WAL segment, and commit both
    /// facts with a single data-header `fdatasync`.
    ///
    /// Quiescence is *established, not asserted*: the writer reaches this only
    /// between commands, and the caller (`writer::apply_command`) has already
    /// committed everything queued ahead of it, so `committed_txid == visible ==
    /// T` and no dirty window is open. That is why this needs no maintenance
    /// phase flip — `enter_maintenance()` only changes an enum and would leave the
    /// writer free to be inside `commit_all`.
    ///
    /// **The atomic commit point is step 5's header `datasync`.** Everything
    /// before it is orphan bytes the winning (old) header does not reference;
    /// everything after it is infallible in-memory rebinding. That ordering is
    /// what makes every crash boundary land on either the pre- or the
    /// post-checkpoint state and never a mix.
    ///
    /// An I/O failure anywhere here **poisons** (invariant 17). Before the commit
    /// point that is conservative but honest; at or after it, it is *required* —
    /// a store that cannot tell whether its new header landed cannot tell which
    /// segment a later recovery would scan, so it must not keep appending.
    /// The `max_data_bytes` pre-check (M6 Phase F). `Ok(Some(skip))`
    /// means this checkpoint must decline; `Ok(None)` means proceed.
    ///
    /// **What it costs.** A full counting walk of the live set, paid only when the
    /// option is set. Deciding *before* touching the shared data file is the whole
    /// point: relocating until the budget runs out and then aborting would strand
    /// orphan extents that consume the very budget being defended, and — because
    /// the data frontier is the file's length — each retry would append again and
    /// grow the file without bound. The walk validates as it counts (it is the
    /// shared `LivePages`), so a corrupt live set fails here exactly as it would
    /// have during relocation.
    ///
    /// **What it projects.** The peak set of things simultaneously owned if this
    /// checkpoint runs to completion, each counted once:
    ///
    /// - the data file grown from its append frontier (its *length*, so no
    ///   alignment padding is missed) by the live set and the manifest;
    /// - the current WAL segment, which a snapshot may pin past the rotation,
    ///   **plus** the fresh segment's header — they coexist;
    /// - bytes already retained (retired segments, superseded data inodes);
    /// - **maintenance headroom**: room for a compaction's full second copy of the
    ///   live set. A checkpoint allowed to spend the last of the budget would
    ///   leave the store unable to compact its way back out, so the budget for
    ///   getting out is reserved before going in. It is sized on *this* live set,
    ///   so it is an estimate, not a guarantee: pages added after this checkpoint
    ///   enlarge the copy a later compaction needs.
    async fn space_check(
        &mut self,
        core: &Arc<Core>,
        current: &crate::store::RootDescriptor,
    ) -> Result<Option<SpaceSkip>, crate::error::CheckpointError> {
        let Some(limit) = core.max_data_bytes else {
            return Ok(None);
        };
        let mut live_pages = 0u64;
        let mut walk = crate::read::LivePages::new(current.root);
        loop {
            match walk.next(&core.cache, &current.file).await {
                Ok(Some(_)) => live_pages += 1,
                Ok(None) => break,
                Err(e) => return Err(self.poison_read(core, e)),
            }
        }
        let live_bytes = live_pages.saturating_mul(PAGE_SIZE as u64);
        let manifest_bytes =
            live_pages.saturating_mul(crate::checkpoint::MANIFEST_ENTRY_LEN as u64);
        // The manifest write is padded up to a full page (M7-A),
        // so the temp the compaction escape route needs reserves the *padded* size;
        // the unrounded `manifest_bytes` under-reserves by up to a page.
        let manifest_bytes_padded = manifest_bytes.next_multiple_of(PAGE_SIZE as u64);
        // Headroom for the compaction this checkpoint must leave possible: its temp
        // is the same live set plus a manifest and both header slots, **and** a
        // compaction is itself a checkpoint into that temp, so it rotates and mints
        // another segment header (reserving only the temp would leave the escape
        // route short by exactly one header).
        //
        // A store that cannot compact at all needs none of it: with no data host
        // `compact()` is permanently `Unsupported`, so reserving a second copy of
        // the live set would make such a store skip checkpoints at roughly half the
        // limit, forever, to protect an operation it can never run.
        let headroom = if self.bind.data_host.is_some() {
            live_bytes
                .saturating_add(manifest_bytes_padded)
                .saturating_add(2 * PAGE_SIZE as u64)
                .saturating_add(WAL_HEADER_LEN)
        } else {
            0
        };
        // Round as the frontier will: the next append starts at a page boundary, so
        // projecting the unrounded end would under-count by up to a page.
        let data_after = self
            .bind
            .data_frontier
            .saturating_add(live_bytes)
            .saturating_add(manifest_bytes)
            .next_multiple_of(PAGE_SIZE as u64);
        // The same accounting the `store_physical_bytes` metric reports, with the
        // terms this checkpoint would leave behind: the grown data file, and the
        // current WAL *plus* the fresh segment's header — the rotation does not
        // free the old segment while a snapshot may still pin it, so both exist.
        let projected = crate::store::owned_bytes(
            data_after,
            core.wal_physical_bytes
                .load(Ordering::Acquire)
                .saturating_add(WAL_HEADER_LEN),
            core.retention.metrics().bytes,
        )
        .saturating_add(headroom);
        if projected > limit {
            Ok(Some(SpaceSkip { limit, projected }))
        } else {
            Ok(None)
        }
    }

    pub(crate) async fn checkpoint(
        &mut self,
        core: &Arc<Core>,
    ) -> Result<CheckpointReport, crate::error::CheckpointError> {
        use crate::checkpoint::CheckpointHeader;
        use crate::error::CheckpointError;

        let Some(file_set) = self.bind.file_set.clone() else {
            return Err(CheckpointError::Unsupported);
        };
        // Hard barrier check (T3.4; was a release-elided debug_assert): a
        // checkpoint off its committed barrier would relocate a live set that
        // does not match the frontier it persists. Refuse before any mutation.
        if !self.pending.is_empty() {
            return Err(CheckpointError::Internal(
                "checkpoint invoked with uncommitted pending batches",
            ));
        }

        // 1-2. The barrier state: the visible root R@T, its count, and the
        //      allocator frontier F. `F` is *persisted*, not merely used to gate
        //      id minting, so ids can never be reused across a checkpoint even
        //      though the records that burned them are about to be dropped.
        let current = core.root();
        let t = self.committed_txid;
        if current.version.txid() != t {
            return Err(CheckpointError::Internal(
                "checkpoint barrier is not fully committed",
            ));
        }
        let frontier = self.alloc.peek_next();

        // 2b. The `max_data_bytes` guard (M6 Phase F). Before touching the shared
        //     data file, project what this checkpoint would leave the store owning
        //     and decline if it would not fit. Read-only: a skip changes nothing.
        if let Some(skip) = self.space_check(core, &current).await? {
            core.checkpoints_skipped_space
                .fetch_add(1, Ordering::AcqRel);
            return Ok(CheckpointReport {
                skipped_space: Some(skip),
                checkpoint_seq: 0,
                checkpoint_txid: 0,
                relocated_pages: 0,
                retired_wal_bytes: 0,
            });
        }

        // 3. Relocate the live set into the data file, past every referenced
        //    extent, and append the manifest describing where each landed. Any
        //    failure poisons: relocation mutates the *shared* data file, so a
        //    half-written extent leaves the store unable to vouch for what a
        //    recovery would read (a compaction, whose target is a private temp
        //    file, aborts cleanly instead — M6 Phase D).
        let reloc = match relocate_live_set(
            &core.cache,
            &current.file,
            current.root,
            &self.bind.data,
            self.bind.data_frontier,
        )
        .await
        {
            Ok(r) => r,
            Err(RelocFail::Read(e)) => return Err(self.poison_read(core, e)),
            Err(RelocFail::Target(e)) => return Err(self.poison_io(core, e).into()),
        };
        let RelocatedSet {
            manifest,
            manifest_offset,
            manifest_len,
            manifest_cksum,
            end: off,
        } = reloc;
        let relocated = manifest.len() as u64;
        if let Err(e) = self.bind.data.datasync().await {
            return Err(self.poison_io(core, e).into());
        }

        // 4. Mint the next segment — durably, with its own identity header — so
        //    that once the header commits, the segment it names certainly exists.
        //
        //    A *previous* checkpoint attempt may have crashed after minting this
        //    same segment but before committing the header naming it, leaving it
        //    on disk as an orphan. Drop it first: no valid header slot can
        //    reference it (had its header committed, this store would have
        //    reopened at `seq+1` and would now be minting `seq+2`), and the
        //    sidecar lock means no other process holds the file set. Without this,
        //    `create_segment`'s deliberate `create_new` refuses, and — since a
        //    checkpoint I/O error poisons — a single crashed checkpoint would
        //    permanently brick the store's ability to ever checkpoint again,
        //    instead of the orphan simply being re-created by the next attempt.
        // Checked, and bounded by what recovery will accept: a header carrying a
        // seq at the exhaustion bound is `Corrupt` at open, so the writer must
        // never *write* one (the same "as strict as recovery" rule the record
        // frontier follows). Unreachable in any real store — 2^64 checkpoints —
        // but it must fail as a poison, never as a writer-thread panic.
        let next_seq = match self.bind.segment_seq.checked_add(1) {
            Some(s) if s < crate::checkpoint::MAX_SEQ => s,
            _ => {
                return Err(self
                    .poison_with(
                        core,
                        PoisonReason::Internal("wal segment sequence exhausted"),
                    )
                    .into())
            }
        };
        if let Err(e) = file_set.unlink_segment(next_seq) {
            return Err(self.poison_io(core, e).into());
        }
        let new_wal = match file_set.create_segment(next_seq, self.bind.store_uuid) {
            Ok(d) => d,
            Err(e) => return Err(self.poison_io(core, e).into()),
        };
        let new_seed = seed_checksum(self.bind.store_uuid, next_seq);

        // 5. Commit: write the header to the *stale* slot and sync it. A torn
        //    write here fails its own checksum and the prior slot — whose segment
        //    is still linked — wins.
        let checkpoint_seq = match self.bind.checkpoint_seq.checked_add(1) {
            Some(s) if s < crate::checkpoint::MAX_SEQ => s,
            _ => {
                return Err(self
                    .poison_with(
                        core,
                        PoisonReason::Internal("checkpoint sequence exhausted"),
                    )
                    .into())
            }
        };
        let header = CheckpointHeader {
            store_uuid: self.bind.store_uuid,
            checkpoint_seq,
            checkpoint_txid: t,
            root: current.root,
            entry_count: current.entry_count,
            allocator_frontier: frontier,
            wal_segment_seq: next_seq,
            wal_scan_origin: WAL_HEADER_LEN,
            wal_chain_seed: new_seed,
            manifest_offset,
            manifest_len,
            manifest_cksum,
        };
        let slot_offset = (checkpoint_seq % 2) * PAGE_SIZE as u64;
        if let Err(e) = self
            .bind
            .data
            .write_all_at(slot_offset, header.encode())
            .await
            .result
        {
            return Err(self.poison_io(core, e).into());
        }
        if let Err(e) = self.bind.data.datasync().await {
            return Err(self.poison_io(core, e).into());
        }
        // ---- committed: the store is now post-checkpoint on disk ----

        // 6. Rebind. Everything below is infallible: the new segment exists and is
        //    durable, so no failure can strand the store appending into a segment
        //    the winning header does not name.
        let base = manifest
            .iter()
            .map(|e| (e.page_id.as_u64(), e.offset))
            .collect();
        let locator = Arc::new(WalLocator::with_base(base));
        let file = Arc::new(crate::backend::FileGeneration::wal(
            Arc::clone(&self.bind.data),
            Arc::clone(&new_wal),
            Arc::clone(&locator) as Arc<dyn PageLocator>,
        ));
        let retired_wal_bytes = self.append_offset;
        let retired_seq = self.bind.segment_seq;

        self.wal = new_wal;
        self.locator = Arc::clone(&locator);
        self.append_offset = WAL_HEADER_LEN;
        self.prev_checksum = new_seed;
        self.bind.segment_seq = next_seq;
        self.bind.checkpoint_seq = checkpoint_seq;
        // The frontier is where the *next* append starts (page-aligned); `off` is
        // where this one ended, i.e. the file's actual length. The metric is
        // documented as a file length, so it gets `off`; enforcement keeps
        // projecting from the frontier, which is the conservative direction for a
        // space guard.
        self.bind.data_frontier = off.next_multiple_of(PAGE_SIZE as u64);
        core.data_physical_bytes.store(off, Ordering::Release);

        core.wal_locator.store(Some(locator));
        // Republish the *same* (version, root, count) over the new generation: no
        // data changed, so the version must not advance — but every new reader
        // must resolve through the new locator/segment. A reader that already
        // captured the old descriptor keeps it, and with it the old segment's
        // handle: that is exactly the retention that makes the unlink below safe
        // (invariant 1/8).
        core.root.store(Arc::new(crate::store::RootDescriptor {
            version: current.version,
            root: current.root,
            file: Arc::clone(&file),
            entry_count: current.entry_count,
            retain: crate::retention::SnapshotGen::active(file),
        }));
        // Retire the superseded generation into the retention registry (M6
        // Phase E): snapshots bound to it keep reading until they drop — or
        // until the `max_retained_generations` bound revokes oldest-first. Must
        // run *after* the descriptor swap above: a concurrent `snapshot()`
        // whose `try_hold` loses the race against this retirement's release
        // retries its load and must observe the *new* descriptor, or it would
        // spin on the old one. The bytes attributed are the retired segment's
        // **physical** length (still the pre-rotation figure at this point) —
        // what the pinned inode actually occupies, which after a torn-tail
        // recovery exceeds the logical frontier `retired_wal_bytes` reports.
        // Byte attribution is device-anchored (H5): register the retired WAL
        // segment device (its **physical** length — the torn-tail-honest figure)
        // so its bytes are counted while any generation, lease, or **parked
        // single-flight load** still holds it, then retire the generation for the
        // count/bound.
        core.retention.note_superseded(
            &current.file.device(crate::backend::Member::Wal),
            core.wal_physical_bytes.load(Ordering::Acquire),
        );
        core.retention.retire(Arc::clone(&current.retain));
        // Rebase the WAL meter onto the empty segment while **preserving
        // outstanding reservations**: batches admitted
        // before the checkpoint but not yet applied still hold their charge and
        // will append into the new segment.
        if let Some(meter) = &core.wal {
            meter.rebase(WAL_HEADER_LEN);
        }
        core.wal_physical_bytes
            .store(WAL_HEADER_LEN, Ordering::Release);
        core.checkpoints.fetch_add(1, Ordering::AcqRel);
        core.last_checkpoint_txid.store(t, Ordering::Release);
        // Live-set (H): a checkpoint relocates exactly the pages reachable from
        // `R@T` into the data file, so the manifest length is the data file's
        // live-page count. It *appends*, so it reclaims nothing. The rebind above
        // published a fresh generation.
        core.data_live_pages
            .store(manifest.len() as u64, Ordering::Release);
        core.generations_created.fetch_add(1, Ordering::AcqRel);

        // Unlink the retired segment. Its bytes free when the last handle on it
        // drops — ours just did, so unless a pre-checkpoint snapshot pins the old
        // generation, that is now. A failure here leaks a file, nothing more: the
        // winning header does not reference it, so it is already invisible to
        // recovery. Never poison for it.
        let unlinked = file_set.unlink_segment(retired_seq).is_ok();
        if !unlinked {
            core.leaked_segments.fetch_add(1, Ordering::AcqRel);
        }

        Ok(CheckpointReport {
            skipped_space: None,
            checkpoint_seq,
            checkpoint_txid: t,
            relocated_pages: relocated,
            retired_wal_bytes,
        })
    }

    /// Runs one WAL compaction (M6 Phase D): a checkpoint whose
    /// relocation target is a **fresh dense temp file** rather than the tail of
    /// the ever-growing data file, folding away every orphan extent prior
    /// checkpoints stranded. The temp gets the live set from the first data
    /// page, the manifest, and the standard checkpoint header in **both** slots
    /// (equal seq + identical payload is the valid freshly-written state
    /// `read_winning_header` accepts); the durable cutover is the temp's
    /// promotion (rename + directory fsync), after which recovery reads the new
    /// file through the *unchanged* header protocol. The WAL rotates exactly as
    /// in a checkpoint. Runs on the writer thread at a drained, fully-committed
    /// barrier, like the checkpoint whose steps it reuses.
    ///
    /// Failure policy (a deliberate deviation from the checkpoint's):
    /// everything before promotion touches only the private temp file and the
    /// next (unreferenced) segment, so it aborts **cleanly**
    /// (`CheckpointError::Io`, temp removed, store unpoisoned). Live-set read
    /// failures poison. A promotion failure poisons: the commit is in doubt
    /// (invariant 17). Post-commit unlink failures leak-count, never poison.
    pub(crate) async fn compact(
        &mut self,
        core: &Arc<Core>,
    ) -> Result<crate::compact::CompactReport, crate::error::CheckpointError> {
        use crate::checkpoint::CheckpointHeader;
        use crate::error::CheckpointError;

        let (Some(file_set), Some(host)) =
            (self.bind.file_set.clone(), self.bind.data_host.clone())
        else {
            return Err(CheckpointError::Unsupported);
        };
        // Hard barrier check (T3.4; was a release-elided debug_assert).
        if !self.pending.is_empty() {
            return Err(CheckpointError::Internal(
                "compaction invoked with uncommitted pending batches",
            ));
        }
        let current = core.root();
        let t = self.committed_txid;
        if current.version.txid() != t {
            return Err(CheckpointError::Internal(
                "compaction barrier is not fully committed",
            ));
        }
        let frontier = self.alloc.peek_next();

        // What the superseded generation will pin: the old data inode plus the
        // old (about-to-rotate) segment inode — both leave the namespace at the
        // cutover and free when retention releases or revokes the generation.
        // A read-only metadata probe before anything is created or mutated: a
        // failure here is a clean abort, not a poison.
        let data_bytes_before = match self.bind.data.size().await {
            Ok(s) => s,
            Err(e) => return Err(CheckpointError::Io(Arc::new(e))),
        };
        let old_wal_physical = core.wal_physical_bytes.load(Ordering::Acquire);

        let temp = match host.create_temp() {
            Ok(d) => d,
            Err(e) => return Err(CheckpointError::Io(Arc::new(e))),
        };

        // Dense relocation into the temp: live pages from the first data page,
        // manifest after, ids preserved (see `relocate_live_set`).
        let reloc = match relocate_live_set(
            &core.cache,
            &current.file,
            current.root,
            &temp,
            crate::direct::FIRST_DATA_INDEX * PAGE_SIZE as u64,
        )
        .await
        {
            Ok(r) => r,
            Err(RelocFail::Read(e)) => {
                host.remove_temp();
                return Err(self.poison_read(core, e));
            }
            Err(RelocFail::Target(e)) => {
                host.remove_temp();
                return Err(CheckpointError::Io(Arc::new(e)));
            }
        };
        if let Err(e) = temp.datasync().await {
            host.remove_temp();
            return Err(CheckpointError::Io(Arc::new(e)));
        }

        // Rotate: mint the next segment durably, with the same orphan pre-unlink
        // as the checkpoint (a crashed prior attempt may have left this seq
        // behind). Sequence exhaustion is checked as strictly as recovery.
        let next_seq = match self.bind.segment_seq.checked_add(1) {
            Some(s) if s < crate::checkpoint::MAX_SEQ => s,
            _ => {
                host.remove_temp();
                return Err(self
                    .poison_with(
                        core,
                        PoisonReason::Internal("wal segment sequence exhausted"),
                    )
                    .into());
            }
        };
        if let Err(e) = file_set.unlink_segment(next_seq) {
            host.remove_temp();
            return Err(CheckpointError::Io(Arc::new(e)));
        }
        let new_wal = match file_set.create_segment(next_seq, self.bind.store_uuid) {
            Ok(d) => d,
            Err(e) => {
                host.remove_temp();
                return Err(CheckpointError::Io(Arc::new(e)));
            }
        };
        let new_seed = seed_checksum(self.bind.store_uuid, next_seq);
        let checkpoint_seq = match self.bind.checkpoint_seq.checked_add(1) {
            Some(s) if s < crate::checkpoint::MAX_SEQ => s,
            _ => {
                host.remove_temp();
                return Err(self
                    .poison_with(
                        core,
                        PoisonReason::Internal("checkpoint sequence exhausted"),
                    )
                    .into());
            }
        };

        // The temp's own header, both slots, synced. On failure the abort also
        // best-effort unlinks the minted segment (a leftover is re-absorbed by
        // the next attempt's pre-unlink and by `retire_other_segments` at open).
        let header = CheckpointHeader {
            store_uuid: self.bind.store_uuid,
            checkpoint_seq,
            checkpoint_txid: t,
            root: current.root,
            entry_count: current.entry_count,
            allocator_frontier: frontier,
            wal_segment_seq: next_seq,
            wal_scan_origin: WAL_HEADER_LEN,
            wal_chain_seed: new_seed,
            manifest_offset: reloc.manifest_offset,
            manifest_len: reloc.manifest_len,
            manifest_cksum: reloc.manifest_cksum,
        };
        let commit = async {
            temp.write_all_at(0, header.encode()).await.result?;
            temp.write_all_at(PAGE_SIZE as u64, header.encode())
                .await
                .result?;
            temp.datasync().await
        };
        if let Err(e) = commit.await {
            host.remove_temp();
            let _ = file_set.unlink_segment(next_seq);
            return Err(CheckpointError::Io(Arc::new(e)));
        }

        // The durable cutover. rename + parent-dir fsync; on failure the commit
        // is in doubt → poison (invariant 17).
        if let Err(e) = host.promote_temp() {
            return Err(self.poison_io(core, e).into());
        }
        // ---- committed: the dense file is the data file ----

        // Rebind — infallible, mirroring the checkpoint's step 6 plus the data
        // device itself. Republish the same (version, root, count) over the new
        // generation, then retire the superseded one (after the swap, so a
        // racing `snapshot()` capture retry observes the new descriptor).
        let base = reloc
            .manifest
            .iter()
            .map(|e| (e.page_id.as_u64(), e.offset))
            .collect();
        let locator = Arc::new(WalLocator::with_base(base));
        let file = Arc::new(crate::backend::FileGeneration::wal(
            Arc::clone(&temp),
            Arc::clone(&new_wal),
            Arc::clone(&locator) as Arc<dyn PageLocator>,
        ));
        let retired_seq = self.bind.segment_seq;

        self.wal = new_wal;
        self.locator = Arc::clone(&locator);
        self.append_offset = WAL_HEADER_LEN;
        self.prev_checksum = new_seed;
        self.bind.data = temp;
        self.bind.segment_seq = next_seq;
        self.bind.checkpoint_seq = checkpoint_seq;
        self.bind.data_frontier = reloc.end.next_multiple_of(PAGE_SIZE as u64);
        // `reloc.end` is the promoted file's real length — and exactly what
        // `CompactReport::data_bytes_after` reports, so the metric and the report
        // can no longer disagree.
        core.data_physical_bytes.store(reloc.end, Ordering::Release);

        core.wal_locator.store(Some(locator));
        core.root.store(Arc::new(crate::store::RootDescriptor {
            version: current.version,
            root: current.root,
            file: Arc::clone(&file),
            entry_count: current.entry_count,
            retain: crate::retention::SnapshotGen::active(file),
        }));
        // Two devices leave the namespace here (H5, device-anchored accounting):
        // the retired WAL segment (this generation's exclusively) and the
        // superseded data inode (*shared* with every older retained generation
        // still reading through it — all share one device `Arc`, so it registers
        // once and counts once until the last of them, or a parked load, drops
        // it). Register both, then retire.
        let superseded = current.file.device(crate::backend::Member::Data);
        core.retention.note_superseded(
            &current.file.device(crate::backend::Member::Wal),
            old_wal_physical,
        );
        core.retention
            .note_superseded(&superseded, data_bytes_before);
        core.retention.retire(Arc::clone(&current.retain));
        if let Some(meter) = &core.wal {
            meter.rebase(WAL_HEADER_LEN);
        }
        core.wal_physical_bytes
            .store(WAL_HEADER_LEN, Ordering::Release);
        // A compaction *is* a checkpoint (it bumps `checkpoint_seq` on disk and
        // supersedes every record ≤ T), so it advances the checkpoint metrics
        // too, plus its own counter.
        core.checkpoints.fetch_add(1, Ordering::AcqRel);
        core.last_checkpoint_txid.store(t, Ordering::Release);
        core.compactions.fetch_add(1, Ordering::AcqRel);
        // Live-set + reclaim (H): a compaction rewrites the reachable set dense
        // into a fresh file, so `reloc.manifest.len()` is the new live-page count
        // and `data_bytes_before − reloc.end` is the space the dense rewrite
        // reclaimed. The rebind above published a fresh generation.
        core.data_live_pages
            .store(reloc.manifest.len() as u64, Ordering::Release);
        core.reclaimed_bytes.fetch_add(
            data_bytes_before.saturating_sub(reloc.end),
            Ordering::AcqRel,
        );
        core.generations_created.fetch_add(1, Ordering::AcqRel);

        // Unlink the retired segment — after the promotion's directory fsync,
        // so a crash can never leave the old header naming an unlinked segment.
        // A failure leaks a file, nothing more (the winning header does not
        // reference it): count it, never poison.
        if file_set.unlink_segment(retired_seq).is_err() {
            core.leaked_segments.fetch_add(1, Ordering::AcqRel);
        }

        Ok(crate::compact::CompactReport {
            live_pages: reloc.manifest.len() as u64,
            data_bytes_before,
            data_bytes_after: reloc.end,
        })
    }
}

/// Seeds the shared WAL meter for a store whose logical frontier is `logical`
/// (the WAL header plus every accepted record).
pub fn wal_meter(max_wal_bytes: u64, logical: u64, notify: Arc<Event>) -> Arc<WalMeter> {
    WalMeter::new(max_wal_bytes, logical, notify)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M6 Phase H forward-format policy — the WAL segment header, the third
    /// open-time version gate. An older (v1) or future (sentinel 255) format byte
    /// is refused with the structured [`OpenError::UnsupportedFormat`], never
    /// `Corrupt`; a damaged magic stays `Corrupt`; the current version decodes.
    #[test]
    fn wal_segment_format_version_is_classified_at_open() {
        let uuid = [9u8; 16];
        let good = encode_wal_header(uuid, 3);
        assert!(decode_wal_header(&good).is_ok(), "current version decodes");

        for (byte, expect_newer) in [(1u8, false), (255u8, true)] {
            let mut img = good.to_vec();
            img[hdr::OFF_FORMAT] = byte;
            // Re-seal so only the version check can fire.
            let ck = crc32c::crc32c(&img[..hdr::OFF_HEADER_CKSUM]);
            img[hdr::OFF_HEADER_CKSUM..hdr::OFF_HEADER_CKSUM + 4]
                .copy_from_slice(&ck.to_le_bytes());
            assert!(
                matches!(
                    decode_wal_header(&img),
                    Err(OpenError::UnsupportedFormat {
                        component: "WAL segment header",
                        found,
                        supported: hdr::FORMAT_V2,
                        newer,
                    }) if found == byte && newer == expect_newer
                ),
                "format byte {byte} must refuse as UnsupportedFormat"
            );
        }

        // A damaged magic is a different failure class — still `Corrupt`.
        let mut bad_magic = good.to_vec();
        bad_magic[hdr::OFF_MAGIC] ^= 0xFF;
        assert!(matches!(
            decode_wal_header(&bad_magic),
            Err(OpenError::Corrupt("bad wal segment header magic"))
        ));
    }
}
