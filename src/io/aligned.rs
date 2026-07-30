//! Aligned owned buffers and O_DIRECT alignment validation (M7-A).
//!
//! O_DIRECT requires every transfer's memory address, file offset, and length to
//! be a multiple of the filesystem's direct-I/O alignment. The [`PageIo`] trait
//! (io/mod.rs) hands the device a natural-aligned `Box<[u8]>`, so a device opened
//! `direct` cannot transfer that buffer straight through the kernel. It instead
//! bounces through an [`AlignedBuf`]: a page-aligned owned allocation the device
//! reads into / writes from, copying to or from the caller's `Box<[u8]>` at the
//! boundary. The trait is unchanged; the copy is the M7-A cost, measured in M7-E
//! before any trait-level aligned-buffer redesign.
//!
//! **Why a distinct type and not an over-aligned `Box<[u8]>`.** A `Box<[u8]>` is
//! freed by the global allocator with `Layout::for_value` = alignment 1; handing
//! out an allocation made at alignment `PAGE_SIZE` and letting `Box` free it as
//! alignment 1 is a layout-mismatch — undefined behaviour. [`AlignedBuf`] owns its
//! exact [`Layout`] and frees with it.
//!
//! [`PageIo`]: super::PageIo

use std::alloc::{self, Layout};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::ptr::NonNull;

use crate::page::PAGE_SIZE;

/// The direct-I/O alignment the store uses for every O_DIRECT data-file op. Equal
/// to [`PAGE_SIZE`] (16 KiB): data-file offsets and lengths are page multiples,
/// so a `PAGE_SIZE`-aligned buffer at a page offset with a
/// page-multiple length satisfies any real filesystem's requirement (validated
/// against `statx` at open — [`validate_dio_alignment`]).
pub(crate) const DIO_ALIGN: usize = PAGE_SIZE;

/// A page-aligned, zero-initialized owned heap buffer for O_DIRECT transfers.
///
/// The whole capacity is initialized (via `alloc_zeroed`), so `as_slice` /
/// `as_mut_slice` never expose uninitialized bytes. The backing
/// allocation never moves or reallocates for the lifetime of the value; only the
/// small owner struct may move.
pub(crate) struct AlignedBuf {
    /// Non-null for `len > 0`; a dangling pointer for `len == 0` (allocating a
    /// zero-size layout is UB, so a zero-length buffer holds no allocation). The
    /// dangling pointer is only `u8`-aligned, not `DIO_ALIGN`-aligned — sound because
    /// it is never dereferenced or deallocated (`Drop` and both slice accessors guard
    /// on `len != 0` / `len == 0`).
    ptr: NonNull<u8>,
    len: usize,
    /// The exact layout `ptr` was allocated with — freed with the same layout.
    /// For `len == 0` this is a zero-size layout and `Drop` skips deallocation.
    layout: Layout,
}

// SAFETY: `AlignedBuf` uniquely owns its allocation — no aliasing, no interior
// mutability shared across threads. The raw `ptr` is only ever followed through
// `&self`/`&mut self`, so moving the owner to another thread (a syscall worker job,
// or the uring driver thread) transfers exclusive access soundly. It is `Send` but
// deliberately not `Sync` and not `Clone` (a shared or duplicated aligned buffer
// would break the single-owner invariant the ownership contract, invariant 8,
// depends on).
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    /// A zeroed buffer of exactly `len` bytes, aligned to [`DIO_ALIGN`]. `len` need
    /// not be a multiple of the alignment — callers that need a page-multiple
    /// length round up before calling (a rounded read target) or pass an already
    /// page-multiple length (a write bounce).
    pub(crate) fn zeroed(len: usize) -> Self {
        Self::zeroed_aligned(len, DIO_ALIGN)
    }

    fn zeroed_aligned(len: usize, align: usize) -> Self {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        if len == 0 {
            // No allocation: a zero-size layout cannot be passed to `alloc`.
            return AlignedBuf {
                ptr: NonNull::dangling(),
                len: 0,
                layout: Layout::from_size_align(0, align).expect("zero-size layout"),
            };
        }
        let layout = Layout::from_size_align(len, align).expect("aligned dio layout");
        // SAFETY: `layout` has nonzero size (guarded above).
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        AlignedBuf { ptr, len, layout }
    }

    /// A page-aligned copy of `src` (same length). Used to bounce a caller's
    /// `Box<[u8]>` into an aligned buffer for an O_DIRECT write.
    pub(crate) fn from_slice(src: &[u8]) -> Self {
        let mut buf = Self::zeroed(src.len());
        buf.as_mut_slice().copy_from_slice(src);
        buf
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` covers `len` initialized (zeroed, then possibly overwritten)
        // bytes for the lifetime of `&self`; unique ownership forbids aliasing.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: as `as_slice`, with `&mut self` guaranteeing exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: allocated exactly once via `alloc_zeroed` with `self.layout`
            // and never reallocated; freed with the same layout, exactly once.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) }
        }
    }
}

// Compio buffer traits (io_uring backend only, M7-A commit 2), so an `AlignedBuf`
// can be the owned buffer of a `ReadAt`/`WriteAt` op on the ring. The whole
// allocation is always initialized (`alloc_zeroed`), so the "initialized length" is
// invariantly the full `len`: `as_init` exposes all of it, and `set_len` is a no-op
// (the driver tracks the logical transfer length itself, exactly as it does for the
// `Vec` path — it never relies on the buffer's own len). Viewing the initialized
// (zeroed) bytes as `MaybeUninit` in `as_uninit` is always sound.
//
// Soundness note: because `as_uninit` is a *safe* method that hands out
// `&mut [MaybeUninit<u8>]` while `as_init` reports the full `len` as initialized,
// safe code *could* in principle write `MaybeUninit::uninit()` and then observe it —
// so this abstraction relies on the internal invariant that its only caller is the
// compio ring op, which writes real read bytes and never de-initializes. That is
// the same shape as compio's own `IoBufMut for Vec<u8>` (whose `as_uninit` likewise
// exposes the initialized prefix), and `AlignedBuf` is `pub(crate)` with exactly one
// user (`DioBuf` in the uring driver), so the invariant is upheld by construction.
#[cfg(feature = "io-uring")]
mod compio_impls {
    use super::AlignedBuf;
    use compio_buf::{IoBuf, IoBufMut, SetLen};
    use std::mem::MaybeUninit;

    impl IoBuf for AlignedBuf {
        fn as_init(&self) -> &[u8] {
            self.as_slice()
        }
    }

    impl SetLen for AlignedBuf {
        unsafe fn set_len(&mut self, len: usize) {
            // The buffer is fully initialized already; the logical length lives in
            // the driver, so recording it here is unnecessary. Guard the contract.
            debug_assert!(len <= self.len, "set_len beyond capacity");
        }
    }

    // `as_uninit` covers exactly the `len`-byte allocation; every byte is initialized
    // (zeroed), so presenting it as `MaybeUninit` never exposes uninitialized memory,
    // and writing through it keeps it initialized.
    impl IoBufMut for AlignedBuf {
        fn as_uninit(&mut self) -> &mut [MaybeUninit<u8>] {
            let s = self.as_mut_slice();
            // SAFETY: `MaybeUninit<u8>` has the same layout as `u8`; the region is
            // valid for `s.len()` bytes and uniquely borrowed via `&mut self`.
            unsafe {
                std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut MaybeUninit<u8>, s.len())
            }
        }
    }
}

/// Validates that a filesystem honors O_DIRECT at [`DIO_ALIGN`] for `file`'s inode,
/// via `statx(STATX_DIOALIGN)` on the already-open fd (M7-A). The
/// policy is calibrated to the *measurement-integrity* purpose of `direct_io`: a
/// silent buffered fallback would produce a "cold" benchmark arm that is secretly
/// page-cached, so this **never** downgrades to buffered — it returns `Ok` to
/// proceed with O_DIRECT, or an error that refuses the open. The fd stays O_DIRECT.
///
/// Decision table:
/// - `statx` fails, or the kernel does not fill `STATX_DIOALIGN` (btrfs on every
///   kernel to date; any kernel < 6.1) → **indeterminate → proceed.** The
///   filesystem cannot tell us, but the store's first aligned header op at
///   open/create is itself the functional probe (a filesystem that truly cannot do
///   O_DIRECT fails at `open(O_DIRECT)` — e.g. tmpfs — or at that aligned op), and
///   M7-E's `mincore` cross-check is the coldness guarantor. Refusing here would
///   self-brick O_DIRECT on btrfs, the project's own benchmark filesystem.
/// - `STATX_DIOALIGN` present with a zero alignment → the filesystem gives positive
///   evidence it does not support direct I/O → **refuse.**
/// - present with a nonzero alignment that does **not** divide [`DIO_ALIGN`] (so a
///   page-aligned address/offset/length would not satisfy it) → **refuse.**
/// - present and dividing [`DIO_ALIGN`] → honored → **proceed.**
#[cfg(target_os = "linux")]
pub(crate) fn validate_dio_alignment(file: &File) -> io::Result<()> {
    // SAFETY: `statx` fully initializes the fields it reports (those in `stx_mask`);
    // we read only masked fields. A zeroed struct is a valid starting point.
    let mut stx: libc::statx = unsafe { std::mem::zeroed() };
    // AT_EMPTY_PATH + an empty path targets the fd itself, avoiding a path-lookup
    // race on the inode we already hold open.
    let ret = unsafe {
        libc::statx(
            file.as_raw_fd(),
            b"\0".as_ptr().cast(),
            libc::AT_EMPTY_PATH,
            libc::STATX_DIOALIGN,
            &mut stx,
        )
    };
    if ret != 0 {
        // statx unavailable (kernel < 4.11) → indeterminate → proceed.
        return Ok(());
    }
    let mask_present = stx.stx_mask & libc::STATX_DIOALIGN != 0;
    dio_alignment_verdict(
        mask_present,
        stx.stx_dio_mem_align,
        stx.stx_dio_offset_align,
    )
}

/// The pure decision behind [`validate_dio_alignment`], factored out
/// so every branch is host-independently testable (the dev host — btrfs — only ever
/// exercises the mask-absent branch). `mask_present` is whether `statx` filled
/// `STATX_DIOALIGN`; `mem_align`/`off_align` are its reported memory and
/// offset/length alignments. Returns `Ok` to proceed with O_DIRECT, or an
/// `Unsupported` error to refuse the open — never a downgrade to buffered.
#[cfg(target_os = "linux")]
fn dio_alignment_verdict(mask_present: bool, mem_align: u32, off_align: u32) -> io::Result<()> {
    if !mask_present {
        // The filesystem did not fill DIOALIGN (btrfs; kernel < 6.1) → indeterminate
        // → proceed; the store's first aligned header op is the functional probe and
        // M7-E's `mincore` is the coldness guard (refusing here self-bricks btrfs).
        return Ok(());
    }
    if mem_align == 0 || off_align == 0 {
        // Positive evidence the filesystem does no direct I/O → refuse.
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "filesystem reports direct I/O unsupported (STATX_DIOALIGN alignment is 0)",
        ));
    }
    // A DIO_ALIGN-aligned address/offset/length satisfies a required alignment `a`
    // only when `a` divides DIO_ALIGN. Checking `a <= DIO_ALIGN` is insufficient for
    // a non-divisor alignment.
    let want = DIO_ALIGN as u32;
    if want % mem_align != 0 || want % off_align != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "filesystem direct I/O alignment (mem={mem_align}, offset={off_align}) \
                 is not satisfied by {want}-byte alignment"
            ),
        ));
    }
    Ok(())
}

/// Off Linux there is no O_DIRECT/`statx`; the caller never opens `direct` there
/// (the open flag is Linux-gated), so this is unreachable but keeps the module
/// portable.
#[cfg(not(target_os = "linux"))]
pub(crate) fn validate_dio_alignment(_file: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn dio_alignment_verdict_decision_table() {
        use std::io::ErrorKind;
        // Mask absent (btrfs / kernel < 6.1): indeterminate → proceed regardless of
        // the (unfilled, ignored) alignment fields.
        assert!(dio_alignment_verdict(false, 0, 0).is_ok());
        assert!(dio_alignment_verdict(false, 999, 999).is_ok());
        // Positive evidence of no direct I/O (a reported alignment of 0) → refuse.
        assert_eq!(
            dio_alignment_verdict(true, 0, 512).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            dio_alignment_verdict(true, 512, 0).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        // Divisor alignments (a DIO_ALIGN-aligned buffer satisfies them) → proceed.
        assert!(dio_alignment_verdict(true, 512, 512).is_ok());
        assert!(dio_alignment_verdict(true, 4096, 4096).is_ok());
        assert!(dio_alignment_verdict(true, DIO_ALIGN as u32, DIO_ALIGN as u32).is_ok());
        // Non-divisor alignment (a page-aligned buffer would NOT satisfy it) → refuse,
        // in either field. 24576 = 1.5 * 16 KiB; 32768 = 2 * 16 KiB (oversized).
        assert_eq!(
            dio_alignment_verdict(true, 24576, 512).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            dio_alignment_verdict(true, 512, 32768).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
    }

    #[test]
    fn zeroed_is_aligned_and_zero() {
        for len in [1usize, 100, PAGE_SIZE, 3 * PAGE_SIZE, PAGE_SIZE + 7] {
            let buf = AlignedBuf::zeroed(len);
            assert_eq!(buf.len(), len);
            assert_eq!(buf.as_slice().as_ptr() as usize % DIO_ALIGN, 0, "len={len}");
            assert!(buf.as_slice().iter().all(|&b| b == 0), "len={len}");
        }
    }

    #[test]
    fn from_slice_copies_and_aligns() {
        let src: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 251) as u8).collect();
        let buf = AlignedBuf::from_slice(&src);
        assert_eq!(buf.len(), src.len());
        assert_eq!(buf.as_slice().as_ptr() as usize % DIO_ALIGN, 0);
        assert_eq!(buf.as_slice(), &src[..]);
    }

    #[test]
    fn zero_length_holds_no_allocation() {
        let mut buf = AlignedBuf::zeroed(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.as_slice().is_empty());
        assert!(buf.as_mut_slice().is_empty());
        // Drop must not deallocate (no allocation was made).
    }

    #[test]
    fn mutation_is_visible() {
        let mut buf = AlignedBuf::zeroed(PAGE_SIZE);
        buf.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(&buf.as_slice()[..4], &[1, 2, 3, 4]);
        assert!(buf.as_slice()[4..].iter().all(|&b| b == 0));
    }
}
