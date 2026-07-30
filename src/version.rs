//! Open incarnations and versions.
//!
//! An [`Incarnation`] is a random 64-bit nonce generated on every successful
//! open. Uniqueness — not ordering — is required: it discriminates a stale
//! receipt (flush/commit/apply outcome) from a later open. A [`Version`] pairs
//! an incarnation with a monotonic `txid`. Versions compare **only** within one
//! incarnation (cross-cutting invariant 11); comparing across incarnations is a
//! programming error and yields `None`.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

/// A random 64-bit per-open nonce. Not ordered.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Incarnation(u64);

impl Incarnation {
    /// Wraps an explicit nonce. Tests use this for deterministic incarnations.
    pub const fn from_raw(nonce: u64) -> Self {
        Incarnation(nonce)
    }

    /// The raw nonce.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Draws a fresh per-open nonce.
    ///
    /// Randomness comes from the OS (`/dev/urandom` — T3.4: the old
    /// `RandomState`-SipHash derivation was thinner than it looked for a value
    /// guarding stale-receipt discrimination). Deterministic injection for
    /// tests is unchanged (`Options::incarnation`). If the OS source is
    /// unavailable (not a real Linux failure mode), falls back to the previous
    /// SipHash-keyed derivation rather than aborting an open — v1 requires
    /// uniqueness, not a CSPRNG.
    pub fn generate() -> Self {
        if let Some(bytes) = crate::version::os_random_bytes::<8>() {
            return Incarnation(u64::from_le_bytes(bytes));
        }
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(seq);
        hasher.write_u64(0x9E37_79B9_7F4A_7C15);
        Incarnation(hasher.finish())
    }
}

impl std::fmt::Debug for Incarnation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Incarnation({:#018x})", self.0)
    }
}

/// A published state version: `(incarnation, txid)`.
///
/// `txid` is monotonic within an incarnation and, for WAL, monotonic across
/// recovery. The empty base state is `txid == 0`.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Version {
    incarnation: Incarnation,
    txid: u64,
}

impl Version {
    pub const fn new(incarnation: Incarnation, txid: u64) -> Self {
        Version { incarnation, txid }
    }

    /// The base (empty store) version for an open.
    pub const fn base(incarnation: Incarnation) -> Self {
        Version {
            incarnation,
            txid: 0,
        }
    }

    pub const fn incarnation(self) -> Incarnation {
        self.incarnation
    }

    pub const fn txid(self) -> u64 {
        self.txid
    }

    /// The next version in the same incarnation.
    pub const fn next(self) -> Version {
        Version {
            incarnation: self.incarnation,
            txid: self.txid + 1,
        }
    }

    /// Orders two versions **only** when they share an incarnation. Returns
    /// `None` across incarnations, so a stale receipt can never compare as
    /// "already durable" against a later open.
    pub fn same_incarnation_cmp(self, other: Version) -> Option<std::cmp::Ordering> {
        if self.incarnation == other.incarnation {
            Some(self.txid.cmp(&other.txid))
        } else {
            None
        }
    }
}

impl std::fmt::Debug for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Version({:#018x}#{})", self.incarnation.0, self.txid)
    }
}

/// `N` bytes from the OS entropy source, or `None` if it cannot be read
/// (T3.4). Linux-first: `/dev/urandom` never blocks after boot-time seeding and
/// exists on every target this crate supports; no new dependency.
pub(crate) fn os_random_bytes<const N: usize>() -> Option<[u8; N]> {
    use std::io::Read;
    let mut buf = [0u8; N];
    let mut f = std::fs::File::open("/dev/urandom").ok()?;
    f.read_exact(&mut buf).ok()?;
    Some(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_is_unique_within_process() {
        let a = Incarnation::generate();
        let b = Incarnation::generate();
        assert_ne!(a, b, "two opens must not collide");
    }

    #[test]
    fn versions_compare_only_within_incarnation() {
        let inc = Incarnation::from_raw(7);
        let v0 = Version::base(inc);
        let v1 = v0.next();
        assert_eq!(v0.same_incarnation_cmp(v1), Some(std::cmp::Ordering::Less));

        let other = Version::new(Incarnation::from_raw(8), 100);
        assert_eq!(v1.same_incarnation_cmp(other), None);
    }
}
