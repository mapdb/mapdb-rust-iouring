//! StoreWal checkpoint format (M6 Phase B).
//!
//! A checkpoint relocates the live page set from the WAL into the data file and
//! commits that fact by writing a **double-slot data header** (pages 0 and 1 —
//! alternating slots, so a torn write never bricks a recoverable store). The
//! header is the *sole* durable carrier of post-checkpoint reset state: the
//! checkpoint txid `T` (which authoritatively supersedes all WAL records ≤ T), the
//! base root `R@T`, the allocator frontier, and — crucially — **which WAL segment
//! holds records `> T`, where its scan starts, and the chain seed to validate
//! them**. A fresh store writes a
//! `checkpoint_seq == 0` header (empty root, empty manifest, segment 0, scan origin
//! at the WAL header end): recovery of a never-checkpointed store is then just the
//! seq-0 case of the general path.
//!
//! The **manifest** is an append-only blob of `(page_id, data-file offset,
//! checksum)` triples written into the data file before the header commit; the
//! header carries its `(offset, len, checksum)`. Recovery seeds the data-file
//! locator, the record root-resolve checksum set, AND the allocator from it, which
//! is what stops a post-checkpoint zero-page (no-op) commit from starving the scan
//! and stops page-id reuse.

use crate::direct::FIRST_DATA_INDEX;
use crate::error::OpenError;
use crate::io::PageIo;
use crate::page::{Checksum, PageId, PageRef, PAGE_SIZE};

/// The WAL data-file format version (v1 was the bare M4 base header; v2 was
/// Phase B's checkpoint header). **v3** (M6 Phase F): the layout is unchanged
/// from v2 — the bump exists to gate the **page** format, which Phase F breaks by
/// adding `PageKind::Overflow` and the leaf overflow-ref.
///
/// The page format byte cannot gate this by itself: no open path decodes every
/// page, so a pre-F store would open clean under a post-F binary and poison later
/// on the first unreadable page, instead of being refused at the door. The data
/// header is the store-level marker, so it carries the break. An older store is
/// refused with a clear error the day this ships, not silently misread.
pub const FORMAT_V3: u8 = 3;

/// Checkpoint and WAL-segment sequence numbers may not reach this value, so
/// `seq + 1` can never wrap and the writer never panics on a forged header
/// (seq exhaustion is defined: no wrap). A real store would need 2^64−1
/// checkpoints to approach it; the bound exists because a *crafted* header can
/// claim any seq, and this file is the trust boundary — a corrupt value must fail
/// here as `Corrupt`, not as an overflow panic on the writer thread later.
pub const MAX_SEQ: u64 = u64::MAX - 1;

/// Field offsets within a checkpoint header slot image (one `PAGE_SIZE` page).
mod off {
    pub const MAGIC: usize = 0; // u32 "MP5C"
    pub const FORMAT: usize = 4; // u8
    pub const ALGO: usize = 5; // u8 (crc32c == 1)
    pub const PAGE_SIZE: usize = 8; // u32
    pub const STORE_UUID: usize = 16; // 16 bytes
    pub const CKPT_SEQ: usize = 32; // u64
    pub const CKPT_TXID: usize = 40; // u64 (T)
    pub const ROOT_ID: usize = 48; // u64
    pub const ROOT_CKSUM: usize = 56; // u32
    pub const ENTRY_COUNT: usize = 64; // u64
    pub const ALLOC_FRONTIER: usize = 72; // u64 (F: next unused page id)
    pub const WAL_SEG_SEQ: usize = 80; // u64 (segment holding records > T)
    pub const WAL_SCAN_ORIGIN: usize = 88; // u64 (byte offset scan starts at)
    pub const WAL_CHAIN_SEED: usize = 96; // u32 (chain seed for records > T)
    pub const MANIFEST_OFFSET: usize = 104; // u64 (in data file)
    pub const MANIFEST_LEN: usize = 112; // u64 (bytes)
    pub const MANIFEST_CKSUM: usize = 120; // u32 (crc32c over the manifest blob)
    pub const HEADER_CKSUM: usize = 124; // u32 (crc32c over [0, 124))
    pub const MAGIC_VALUE: u32 = 0x4D50_3543; // "MP5C"
    pub const ALGO_CRC32C: u8 = 1;
}

/// One `(page_id, data-file offset, checksum)` manifest entry. 20 bytes on disk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManifestEntry {
    pub page_id: PageId,
    pub offset: u64,
    pub checksum: Checksum,
}

pub(crate) const MANIFEST_ENTRY_LEN: usize = 20;

/// The decoded checkpoint data header (one slot).
#[derive(Clone, Copy, Debug)]
pub struct CheckpointHeader {
    pub store_uuid: [u8; 16],
    pub checkpoint_seq: u64,
    pub checkpoint_txid: u64,
    pub root: PageRef,
    pub entry_count: u64,
    pub allocator_frontier: u64,
    pub wal_segment_seq: u64,
    pub wal_scan_origin: u64,
    pub wal_chain_seed: u32,
    pub manifest_offset: u64,
    pub manifest_len: u64,
    pub manifest_cksum: u32,
}

impl CheckpointHeader {
    /// The seq-0 header of a freshly created store: no checkpoint has run, so the
    /// base root is empty, the manifest is empty, and the WAL to scan is segment 0
    /// starting right after its identity header, chained from `wal_seed`.
    pub fn fresh(store_uuid: [u8; 16], wal_header_len: u64, wal_seed: u32) -> Self {
        CheckpointHeader {
            store_uuid,
            checkpoint_seq: 0,
            checkpoint_txid: 0,
            root: PageRef::EMPTY,
            entry_count: 0,
            allocator_frontier: FIRST_DATA_INDEX,
            wal_segment_seq: 0,
            wal_scan_origin: wal_header_len,
            wal_chain_seed: wal_seed,
            manifest_offset: 0,
            manifest_len: 0,
            manifest_cksum: 0,
        }
    }

    /// Encodes this header into a full zero-padded `PAGE_SIZE` slot image.
    pub fn encode(&self) -> Box<[u8]> {
        let mut b = vec![0u8; PAGE_SIZE];
        let put_u32 =
            |b: &mut [u8], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        let put_u64 =
            |b: &mut [u8], o: usize, v: u64| b[o..o + 8].copy_from_slice(&v.to_le_bytes());
        put_u32(&mut b, off::MAGIC, off::MAGIC_VALUE);
        b[off::FORMAT] = FORMAT_V3;
        b[off::ALGO] = off::ALGO_CRC32C;
        put_u32(&mut b, off::PAGE_SIZE, PAGE_SIZE as u32);
        b[off::STORE_UUID..off::STORE_UUID + 16].copy_from_slice(&self.store_uuid);
        put_u64(&mut b, off::CKPT_SEQ, self.checkpoint_seq);
        put_u64(&mut b, off::CKPT_TXID, self.checkpoint_txid);
        put_u64(&mut b, off::ROOT_ID, self.root.page_id.as_u64());
        put_u32(&mut b, off::ROOT_CKSUM, self.root.expected_checksum.0);
        put_u64(&mut b, off::ENTRY_COUNT, self.entry_count);
        put_u64(&mut b, off::ALLOC_FRONTIER, self.allocator_frontier);
        put_u64(&mut b, off::WAL_SEG_SEQ, self.wal_segment_seq);
        put_u64(&mut b, off::WAL_SCAN_ORIGIN, self.wal_scan_origin);
        put_u32(&mut b, off::WAL_CHAIN_SEED, self.wal_chain_seed);
        put_u64(&mut b, off::MANIFEST_OFFSET, self.manifest_offset);
        put_u64(&mut b, off::MANIFEST_LEN, self.manifest_len);
        put_u32(&mut b, off::MANIFEST_CKSUM, self.manifest_cksum);
        let ck = crc32c::crc32c(&b[..off::HEADER_CKSUM]);
        put_u32(&mut b, off::HEADER_CKSUM, ck);
        b.into_boxed_slice()
    }

    /// Decodes and validates one slot image. Any structural, checksum, or
    /// field-relationship defect is `Corrupt` (never heuristic repair) — a torn
    /// slot decodes as `Corrupt` so the *other* slot can win. Cross-field rules
    /// mirror `DirectHeader::decode`: empty root ⇔ (zero checksum, zero count); a
    /// non-empty root's id lies in `[FIRST_DATA_INDEX, frontier)` with a positive
    /// count; the frontier is bounded so the reopened allocator cannot wrap.
    pub fn decode(bytes: &[u8]) -> Result<CheckpointHeader, OpenError> {
        if bytes.len() < PAGE_SIZE {
            return Err(OpenError::Corrupt("checkpoint header image too short"));
        }
        let rd_u32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let rd_u64 = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        if rd_u32(off::MAGIC) != off::MAGIC_VALUE {
            return Err(OpenError::Corrupt("bad checkpoint header magic"));
        }
        // A v1 base / v2 checkpoint header (older) or a future release's header
        // (newer) reaches here — refuse clearly, at open, before any mutation
        // (M6 Phase H forward policy).
        OpenError::check_open_format("data header", bytes[off::FORMAT], FORMAT_V3)?;
        if bytes[off::ALGO] != off::ALGO_CRC32C {
            return Err(OpenError::Corrupt(
                "unsupported checkpoint checksum algorithm",
            ));
        }
        if rd_u32(off::PAGE_SIZE) != PAGE_SIZE as u32 {
            return Err(OpenError::Corrupt("unsupported checkpoint page size"));
        }
        let stored_ck = rd_u32(off::HEADER_CKSUM);
        if crc32c::crc32c(&bytes[..off::HEADER_CKSUM]) != stored_ck {
            return Err(OpenError::Corrupt("checkpoint header checksum mismatch"));
        }
        let mut store_uuid = [0u8; 16];
        store_uuid.copy_from_slice(&bytes[off::STORE_UUID..off::STORE_UUID + 16]);
        let root = PageRef {
            page_id: PageId(rd_u64(off::ROOT_ID)),
            expected_checksum: Checksum(rd_u32(off::ROOT_CKSUM)),
        };
        let entry_count = rd_u64(off::ENTRY_COUNT);
        let frontier = rd_u64(off::ALLOC_FRONTIER);
        // Frontier bounds: at least the first data id, strictly below the record
        // codec's cap so the reopened allocator has headroom (mirrors wal.rs).
        if !(FIRST_DATA_INDEX..crate::wal::MAX_FRONTIER).contains(&frontier) {
            return Err(OpenError::Corrupt("checkpoint frontier out of range"));
        }
        // Root / entry-count relationship (same rule as the Direct + WAL headers).
        if root.is_empty() {
            if root.expected_checksum != Checksum(0) || entry_count != 0 {
                return Err(OpenError::Corrupt(
                    "empty checkpoint root with non-zero cksum/count",
                ));
            }
        } else if root.page_id.as_u64() < FIRST_DATA_INDEX
            || root.page_id.as_u64() >= frontier
            || entry_count == 0
        {
            return Err(OpenError::Corrupt(
                "checkpoint root out of range or zero count",
            ));
        }
        let manifest_len = rd_u64(off::MANIFEST_LEN);
        if manifest_len as usize % MANIFEST_ENTRY_LEN != 0 {
            return Err(OpenError::Corrupt(
                "checkpoint manifest length not a whole number of entries",
            ));
        }
        // Sequence bounds, so the writer's `seq + 1` can never wrap or panic.
        let checkpoint_seq = rd_u64(off::CKPT_SEQ);
        let wal_segment_seq = rd_u64(off::WAL_SEG_SEQ);
        if checkpoint_seq >= MAX_SEQ || wal_segment_seq >= MAX_SEQ {
            return Err(OpenError::Corrupt(
                "checkpoint/segment sequence at the exhaustion bound",
            ));
        }
        // The same rule for the txid this header would seed the *visible version*
        // with: accepted unchecked, `Version::next`'s `txid + 1` panics the writer
        // on the very next write (or wraps in release, breaking monotonicity).
        let checkpoint_txid = rd_u64(off::CKPT_TXID);
        if checkpoint_txid >= crate::wal::MAX_TXID {
            return Err(OpenError::Corrupt(
                "checkpoint txid at the exhaustion bound",
            ));
        }
        Ok(CheckpointHeader {
            store_uuid,
            checkpoint_seq,
            checkpoint_txid,
            root,
            entry_count,
            allocator_frontier: frontier,
            wal_segment_seq,
            wal_scan_origin: rd_u64(off::WAL_SCAN_ORIGIN),
            wal_chain_seed: rd_u32(off::WAL_CHAIN_SEED),
            manifest_offset: rd_u64(off::MANIFEST_OFFSET),
            manifest_len,
            manifest_cksum: rd_u32(off::MANIFEST_CKSUM),
        })
    }
}

/// Encodes a manifest blob (the `(id, offset, cksum)` triples) and returns
/// `(bytes, checksum)`. Entries must already be sorted by id; the encoder does not
/// reorder (the caller relocates pages in a deterministic walk order and can sort).
pub fn encode_manifest(entries: &[ManifestEntry]) -> (Vec<u8>, u32) {
    let mut buf = vec![0u8; entries.len() * MANIFEST_ENTRY_LEN];
    for (i, e) in entries.iter().enumerate() {
        let o = i * MANIFEST_ENTRY_LEN;
        buf[o..o + 8].copy_from_slice(&e.page_id.as_u64().to_le_bytes());
        buf[o + 8..o + 16].copy_from_slice(&e.offset.to_le_bytes());
        buf[o + 16..o + 20].copy_from_slice(&e.checksum.0.to_le_bytes());
    }
    let ck = crc32c::crc32c(&buf);
    (buf, ck)
}

/// Decodes and fully validates a manifest blob against the header's
/// `(len, checksum, frontier)`: exact checksum, sorted **strictly increasing**
/// (unique) ids, every id in `[FIRST_DATA_INDEX, frontier)`, and every offset a
/// non-header, page-aligned data-file location. Any defect → `Corrupt` (fail open —
/// never fall back to an older header slot whose superseded WAL bytes may be gone).
pub fn decode_manifest(
    blob: &[u8],
    expected_cksum: u32,
    frontier: u64,
) -> Result<Vec<ManifestEntry>, OpenError> {
    if blob.len() % MANIFEST_ENTRY_LEN != 0 {
        return Err(OpenError::Corrupt(
            "manifest blob not a whole number of entries",
        ));
    }
    if crc32c::crc32c(blob) != expected_cksum {
        return Err(OpenError::Corrupt("manifest checksum mismatch"));
    }
    let count = blob.len() / MANIFEST_ENTRY_LEN;
    let mut out = Vec::with_capacity(count);
    let mut prev_id: Option<u64> = None;
    let first_data_off = FIRST_DATA_INDEX * PAGE_SIZE as u64;
    for i in 0..count {
        let o = i * MANIFEST_ENTRY_LEN;
        let id = u64::from_le_bytes(blob[o..o + 8].try_into().unwrap());
        let offset = u64::from_le_bytes(blob[o + 8..o + 16].try_into().unwrap());
        let cksum = u32::from_le_bytes(blob[o + 16..o + 20].try_into().unwrap());
        if id < FIRST_DATA_INDEX || id >= frontier {
            return Err(OpenError::Corrupt("manifest id out of range"));
        }
        if let Some(p) = prev_id {
            if id <= p {
                return Err(OpenError::Corrupt("manifest ids not strictly increasing"));
            }
        }
        prev_id = Some(id);
        if offset < first_data_off || offset % PAGE_SIZE as u64 != 0 {
            return Err(OpenError::Corrupt(
                "manifest offset in reserved region or misaligned",
            ));
        }
        out.push(ManifestEntry {
            page_id: PageId(id),
            offset,
            checksum: Checksum(cksum),
        });
    }
    Ok(out)
}

/// Reads and fully validates the manifest blob the header points at. An empty
/// manifest (a never-checkpointed store) reads as an empty vector without
/// touching the device.
///
/// The blob's extent must lie wholly inside the data file, past the header
/// slots — checked *before* the read, so a corrupt `manifest_len` drives neither
/// a giant allocation nor a read past EOF. Its entry count is additionally
/// bounded by the id space the frontier permits (ids are strictly increasing in
/// `[FIRST_DATA_INDEX, frontier)`), which no valid manifest can exceed.
pub async fn read_manifest(
    data: &dyn PageIo,
    header: &CheckpointHeader,
) -> Result<Vec<ManifestEntry>, OpenError> {
    if header.manifest_len == 0 {
        return Ok(Vec::new());
    }
    let entries = header.manifest_len / MANIFEST_ENTRY_LEN as u64;
    if entries > header.allocator_frontier - FIRST_DATA_INDEX {
        return Err(OpenError::Corrupt(
            "checkpoint manifest holds more entries than the id space",
        ));
    }
    let first_data_off = FIRST_DATA_INDEX * PAGE_SIZE as u64;
    let end = header
        .manifest_offset
        .checked_add(header.manifest_len)
        .ok_or(OpenError::Corrupt("checkpoint manifest extent overflows"))?;
    let size = data.size().await?;
    if header.manifest_offset < first_data_off || end > size {
        return Err(OpenError::Corrupt(
            "checkpoint manifest extent outside the data file",
        ));
    }
    // The manifest offset is page-aligned for every writer (it is an append frontier
    // that only ever advances by `PAGE_SIZE`). Enforce it as a trust-boundary rule so
    // a corrupt or crafted header is classified `Corrupt` here, not left to fail as a
    // device-level `EINVAL`/alignment error under an O_DIRECT open (M7-A).
    if header.manifest_offset % PAGE_SIZE as u64 != 0 {
        return Err(OpenError::Corrupt(
            "checkpoint manifest offset is not page-aligned",
        ));
    }
    let len: usize = header
        .manifest_len
        .try_into()
        .map_err(|_| OpenError::Corrupt("checkpoint manifest too large for this platform"))?;
    let blob = data.read_exact_at(header.manifest_offset, len).await?;
    let entries = decode_manifest(&blob, header.manifest_cksum, header.allocator_frontier)?;
    // Every page image the manifest points at must also lie wholly in the file, so
    // a locator hit can never drive a read past EOF. `checked_add`: a page-aligned
    // offset near the top of the address space would otherwise wrap to a small
    // number and *pass* this check (panicking in debug, silently admitting a wild
    // locator entry in release) — the extent check must not be defeatable by the
    // very value it exists to bound.
    for e in &entries {
        // Each page image sits at a page-aligned offset (relocation advances by
        // `PAGE_SIZE`); enforce it so an O_DIRECT read of a corrupt entry is a
        // `Corrupt` classification, not a device alignment error.
        if e.offset % PAGE_SIZE as u64 != 0 {
            return Err(OpenError::Corrupt(
                "checkpoint manifest page offset is not page-aligned",
            ));
        }
        let end = e
            .offset
            .checked_add(PAGE_SIZE as u64)
            .ok_or(OpenError::Corrupt(
                "checkpoint manifest page extent overflows",
            ))?;
        if end > size {
            return Err(OpenError::Corrupt(
                "checkpoint manifest page extent outside the data file",
            ));
        }
    }
    Ok(entries)
}

/// Reads both header slots (pages 0 and 1) and returns the **winning** header: the
/// higher `checkpoint_seq` among slots that fully decode. If both decode with the
/// *same* seq but differ, that is corruption, not a coin-flip. If
/// neither decodes, `Corrupt`. A torn just-written slot simply fails to decode and
/// the prior slot wins — the crash-safe commit property.
pub async fn read_winning_header(data: &dyn PageIo) -> Result<CheckpointHeader, OpenError> {
    let slot0 = data.read_exact_at(0, PAGE_SIZE).await?;
    let slot1 = data.read_exact_at(PAGE_SIZE as u64, PAGE_SIZE).await?;
    let h0 = CheckpointHeader::decode(&slot0).ok();
    let h1 = CheckpointHeader::decode(&slot1).ok();
    match (h0, h1) {
        (None, None) => Err(OpenError::Corrupt("no valid checkpoint header slot")),
        (Some(h), None) | (None, Some(h)) => Ok(h),
        (Some(a), Some(b)) => {
            if a.checkpoint_seq == b.checkpoint_seq {
                // Equal seq with identical payload is harmless (both slots freshly
                // written); differing payload is corruption. The comparison spans
                // the encoded prefix THROUGH its checksum: every semantic byte is
                // inside `[0, HEADER_CKSUM+4)` today and the remainder of the page
                // is codec-guaranteed zero padding. If a field is ever appended,
                // it must land before HEADER_CKSUM (moving the checksum) or this
                // comparison silently stops covering it — compare full pages then
                // (the Direct v4 picker already compares full pages for exactly
                // this reason).
                if slot0[..off::HEADER_CKSUM + 4] == slot1[..off::HEADER_CKSUM + 4] {
                    Ok(a)
                } else {
                    Err(OpenError::Corrupt(
                        "checkpoint header slots disagree at equal seq",
                    ))
                }
            } else if a.checkpoint_seq > b.checkpoint_seq {
                Ok(a)
            } else {
                Ok(b)
            }
        }
    }
}

// Both `Result::ok` above intentionally discard the decode error; the winner is
// chosen by seq, and "no valid slot" is reported once with its own message.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::FakeIo;
    use futures_lite::future::block_on;

    fn uuid() -> [u8; 16] {
        [7u8; 16]
    }

    #[test]
    fn fresh_header_roundtrips() {
        let h = CheckpointHeader::fresh(uuid(), PAGE_SIZE as u64, 0xDEAD_BEEF);
        let img = h.encode();
        let d = CheckpointHeader::decode(&img).unwrap();
        assert_eq!(d.checkpoint_seq, 0);
        assert_eq!(d.checkpoint_txid, 0);
        assert!(d.root.is_empty());
        assert_eq!(d.allocator_frontier, FIRST_DATA_INDEX);
        assert_eq!(d.wal_scan_origin, PAGE_SIZE as u64);
        assert_eq!(d.wal_chain_seed, 0xDEAD_BEEF);
        assert_eq!(d.manifest_len, 0);
    }

    #[test]
    fn populated_header_roundtrips() {
        let h = CheckpointHeader {
            store_uuid: uuid(),
            checkpoint_seq: 5,
            checkpoint_txid: 42,
            root: PageRef {
                page_id: PageId(9),
                expected_checksum: Checksum(0x1234),
            },
            entry_count: 1000,
            allocator_frontier: 50,
            wal_segment_seq: 5,
            wal_scan_origin: PAGE_SIZE as u64,
            wal_chain_seed: 0xABCD,
            manifest_offset: 2 * PAGE_SIZE as u64,
            manifest_len: 3 * MANIFEST_ENTRY_LEN as u64,
            manifest_cksum: 0x9999,
        };
        let d = CheckpointHeader::decode(&h.encode()).unwrap();
        assert_eq!(d.checkpoint_txid, 42);
        assert_eq!(d.root.page_id.as_u64(), 9);
        assert_eq!(d.allocator_frontier, 50);
        assert_eq!(d.wal_segment_seq, 5);
    }

    #[test]
    fn decode_rejects_defects() {
        let base = CheckpointHeader {
            store_uuid: uuid(),
            checkpoint_seq: 1,
            checkpoint_txid: 1,
            root: PageRef {
                page_id: PageId(9),
                expected_checksum: Checksum(1),
            },
            entry_count: 1,
            allocator_frontier: 50,
            wal_segment_seq: 1,
            wal_scan_origin: PAGE_SIZE as u64,
            wal_chain_seed: 0,
            manifest_offset: 2 * PAGE_SIZE as u64,
            manifest_len: 0,
            manifest_cksum: 0,
        };
        // Corrupt a checksum byte.
        let mut img = base.encode();
        img[off::CKPT_TXID] ^= 0xFF;
        assert!(CheckpointHeader::decode(&img).is_err());
        // Wrong format byte (a v1 base header masquerading, and a future 255):
        // refused with the structured forward-policy error, not `Corrupt`.
        for (byte, expect_newer) in [(1u8, false), (255u8, true)] {
            let mut img = base.encode();
            img[off::FORMAT] = byte;
            // recompute the header checksum so only the format check fires
            let ck = crc32c::crc32c(&img[..off::HEADER_CKSUM]);
            img[off::HEADER_CKSUM..off::HEADER_CKSUM + 4].copy_from_slice(&ck.to_le_bytes());
            assert!(matches!(
                CheckpointHeader::decode(&img),
                Err(OpenError::UnsupportedFormat {
                    component: "data header",
                    found,
                    supported: FORMAT_V3,
                    newer,
                }) if found == byte && newer == expect_newer
            ));
        }
        // Non-empty root beyond the frontier.
        let bad = CheckpointHeader {
            root: PageRef {
                page_id: PageId(50),
                expected_checksum: Checksum(1),
            },
            ..base
        };
        assert!(CheckpointHeader::decode(&bad.encode()).is_err());
        // Empty root but non-zero count.
        let bad = CheckpointHeader {
            root: PageRef::EMPTY,
            entry_count: 5,
            ..base
        };
        assert!(CheckpointHeader::decode(&bad.encode()).is_err());
    }

    #[test]
    fn manifest_roundtrips_and_validates() {
        let entries = vec![
            ManifestEntry {
                page_id: PageId(2),
                offset: 2 * PAGE_SIZE as u64,
                checksum: Checksum(11),
            },
            ManifestEntry {
                page_id: PageId(5),
                offset: 3 * PAGE_SIZE as u64,
                checksum: Checksum(22),
            },
            ManifestEntry {
                page_id: PageId(9),
                offset: 4 * PAGE_SIZE as u64,
                checksum: Checksum(33),
            },
        ];
        let (blob, ck) = encode_manifest(&entries);
        let got = decode_manifest(&blob, ck, 50).unwrap();
        assert_eq!(got, entries);
        // Bad checksum.
        assert!(decode_manifest(&blob, ck ^ 1, 50).is_err());
        // Id beyond frontier.
        assert!(decode_manifest(&blob, ck, 6).is_err());
    }

    #[test]
    fn manifest_rejects_unsorted_or_duplicate_ids() {
        let entries = vec![
            ManifestEntry {
                page_id: PageId(5),
                offset: 2 * PAGE_SIZE as u64,
                checksum: Checksum(1),
            },
            ManifestEntry {
                page_id: PageId(5),
                offset: 3 * PAGE_SIZE as u64,
                checksum: Checksum(2),
            },
        ];
        let (blob, ck) = encode_manifest(&entries);
        assert!(decode_manifest(&blob, ck, 50).is_err());
    }

    #[test]
    fn winning_header_picks_higher_seq_and_survives_torn_slot() {
        block_on(async {
            let dev = FakeIo::new(2 * PAGE_SIZE);
            let older = CheckpointHeader::fresh(uuid(), PAGE_SIZE as u64, 1);
            let mut newer = older;
            newer.checkpoint_seq = 7;
            newer.wal_chain_seed = 2;
            // slot0 = older, slot1 = newer → newer wins.
            dev.write_all_at(0, older.encode()).await.result.unwrap();
            dev.write_all_at(PAGE_SIZE as u64, newer.encode())
                .await
                .result
                .unwrap();
            let w = read_winning_header(&dev).await.unwrap();
            assert_eq!(w.checkpoint_seq, 7);

            // Tear slot1 (the newer one): slot0 (older, still valid) must win —
            // this is the crash-safe commit-point property.
            dev.write_all_at(PAGE_SIZE as u64, vec![0xFFu8; PAGE_SIZE].into_boxed_slice())
                .await
                .result
                .unwrap();
            let w = read_winning_header(&dev).await.unwrap();
            assert_eq!(w.checkpoint_seq, 0, "torn newer slot → older slot wins");
        });
    }

    #[test]
    fn read_manifest_rejects_non_page_aligned_offsets_as_corrupt() {
        // A crafted header whose manifest — or a manifest entry —
        // sits at a non-page-aligned offset must be classified `Corrupt` at the trust
        // boundary, not left to fail as a device-level O_DIRECT alignment error.
        block_on(async {
            // (a) unaligned manifest offset. One valid entry, placed at an aligned
            // offset in the file, but the header *claims* an unaligned manifest offset.
            let entries = vec![ManifestEntry {
                page_id: PageId(2),
                offset: 2 * PAGE_SIZE as u64,
                checksum: Checksum(7),
            }];
            let (blob, ck) = encode_manifest(&entries);
            let dev = FakeIo::new(6 * PAGE_SIZE);
            dev.write_all_at(2 * PAGE_SIZE as u64, blob.clone().into_boxed_slice())
                .await
                .result
                .unwrap();
            let mut h = CheckpointHeader::fresh(uuid(), PAGE_SIZE as u64, 1);
            h.allocator_frontier = 50;
            h.manifest_offset = 2 * PAGE_SIZE as u64 + 1; // unaligned
            h.manifest_len = blob.len() as u64;
            h.manifest_cksum = ck;
            match read_manifest(&dev, &h).await {
                Err(OpenError::Corrupt(_)) => {}
                other => panic!("unaligned manifest offset must be Corrupt, got {other:?}"),
            }

            // (b) unaligned entry offset. Aligned manifest offset + valid checksum,
            // but the entry inside points at a non-page-aligned page offset.
            let bad_entries = vec![ManifestEntry {
                page_id: PageId(2),
                offset: 2 * PAGE_SIZE as u64 + 1, // unaligned
                checksum: Checksum(7),
            }];
            let (bad_blob, bad_ck) = encode_manifest(&bad_entries);
            let dev2 = FakeIo::new(6 * PAGE_SIZE);
            dev2.write_all_at(2 * PAGE_SIZE as u64, bad_blob.clone().into_boxed_slice())
                .await
                .result
                .unwrap();
            let mut h2 = CheckpointHeader::fresh(uuid(), PAGE_SIZE as u64, 1);
            h2.allocator_frontier = 50;
            h2.manifest_offset = 2 * PAGE_SIZE as u64; // aligned
            h2.manifest_len = bad_blob.len() as u64;
            h2.manifest_cksum = bad_ck;
            match read_manifest(&dev2, &h2).await {
                Err(OpenError::Corrupt(_)) => {}
                other => panic!("unaligned entry offset must be Corrupt, got {other:?}"),
            }
        });
    }
}
