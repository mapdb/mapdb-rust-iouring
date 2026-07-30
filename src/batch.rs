//! Write batches, their conditions, results, and the physical page batch.
//!
//! `WriteBatch` rules are fixed before code:
//! - all checks observe writer current state **before any op**;
//! - ops apply in order; last write to a repeated key wins;
//! - a failed check is `Ok(ConditionFailed)` — publishes nothing, consumes no
//!   txid, releases all reservation;
//! - an applied batch has exactly one in-memory root swap;
//! - requested old values are aligned to operations; otherwise not read;
//! - encoded bytes, op count, and worst-case private pages have hard caps.

use crate::admission::WriteCost;
use crate::page::{Page, PageRef, MAX_INLINE_VALUE_LEN, MAX_KEY_LEN, MAX_VALUE_LEN};
use crate::version::Version;

/// One mutation.
#[derive(Clone, Debug)]
pub enum Op {
    /// Insert or overwrite `key` with `value`.
    Insert { key: Vec<u8>, value: Vec<u8> },
    /// Remove `key` if present.
    Remove { key: Vec<u8> },
}

impl Op {
    pub fn key(&self) -> &[u8] {
        match self {
            Op::Insert { key, .. } | Op::Remove { key } => key,
        }
    }

    fn encoded_len(&self) -> u64 {
        match self {
            Op::Insert { key, value } => 2 + key.len() as u64 + value.len() as u64,
            Op::Remove { key } => 1 + key.len() as u64,
        }
    }

    fn validate(&self) -> Result<(), BatchError> {
        let key = self.key();
        if key.is_empty() {
            return Err(BatchError::EmptyKey);
        }
        if key.len() > MAX_KEY_LEN {
            return Err(BatchError::KeyTooLong);
        }
        if let Op::Insert { value, .. } = self {
            if value.len() > MAX_VALUE_LEN {
                return Err(BatchError::ValueTooLong);
            }
        }
        Ok(())
    }
}

/// A condition evaluated against the writer's current state before any op. The
/// whole batch is atomic on all checks passing.
#[derive(Clone, Debug)]
pub struct Check {
    pub key: Vec<u8>,
    /// The value `key` must currently hold. `None` means the key must be absent.
    pub expected: Option<Vec<u8>>,
}

impl Check {
    /// Encoded size of this check for admission accounting: framing + key +
    /// expected-value bytes. Charged so a check-heavy batch cannot smuggle
    /// unbounded bytes past `max_command_bytes`.
    fn encoded_len(&self) -> u64 {
        2 + self.key.len() as u64 + self.expected.as_ref().map_or(0, |v| v.len() as u64)
    }

    /// Validates the check's key/value sizes, same contract as an op's key/value.
    fn validate(&self) -> Result<(), BatchError> {
        if self.key.is_empty() {
            return Err(BatchError::EmptyKey);
        }
        if self.key.len() > MAX_KEY_LEN {
            return Err(BatchError::KeyTooLong);
        }
        if let Some(v) = &self.expected {
            if v.len() > MAX_VALUE_LEN {
                return Err(BatchError::ValueTooLong);
            }
        }
        Ok(())
    }
}

/// A conditional, atomic batch of mutations.
#[derive(Clone, Debug, Default)]
pub struct WriteBatch {
    checks: Vec<Check>,
    ops: Vec<Op>,
    return_old_values: bool,
}

/// Why a batch is malformed (a caller error caught before admission).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchError {
    EmptyKey,
    KeyTooLong,
    ValueTooLong,
    TooManyOps,
    /// `returning_old_values` was requested and the values the store actually
    /// holds for this batch's keys total more than
    /// [`AdmissionLimits::max_command_bytes`](crate::AdmissionLimits::max_command_bytes)
    /// (M6 Phase F).
    ///
    /// Unlike its siblings this cannot be caught at admission: the size of the
    /// answer depends on the store's contents, not on the batch. It is still a
    /// caller error — ask for fewer keys, or don't ask for the old values — and
    /// it is healthy: nothing is published and no txid is consumed.
    ///
    /// **The ceiling is shared with the command-input limit**, so tuning
    /// `max_command_bytes` down also caps the old-value *response*. That is
    /// deliberate — a store that asked for a tight byte budget is exactly the one
    /// that should not be handed an unbounded reply — but it does mean a batch
    /// shaped like a pre-M6-Phase-F one (small values, small `max_command_bytes`,
    /// many keys) can now hit this where it did not before. **Under default limits
    /// that is unreachable for inline-sized values**: `max_ops_per_batch` (4096) ×
    /// `MAX_INLINE_VALUE_LEN` (8 KiB) = 32 MiB, against a 64 MiB default ceiling —
    /// a margin pinned by `default_limits_admit_any_inline_old_value_response`.
    OldValuesTooLarge,
}

impl WriteBatch {
    pub fn new() -> Self {
        WriteBatch::default()
    }

    pub fn insert(mut self, key: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>) -> Self {
        self.ops.push(Op::Insert {
            key: key.into(),
            value: value.into(),
        });
        self
    }

    pub fn remove(mut self, key: impl Into<Vec<u8>>) -> Self {
        self.ops.push(Op::Remove { key: key.into() });
        self
    }

    /// Adds a precondition. `expected == None` requires `key` absent.
    pub fn check(mut self, key: impl Into<Vec<u8>>, expected: Option<Vec<u8>>) -> Self {
        self.checks.push(Check {
            key: key.into(),
            expected,
        });
        self
    }

    /// Requests the pre-batch value of each op's key, aligned to `ops`.
    pub fn returning_old_values(mut self) -> Self {
        self.return_old_values = true;
        self
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }
    pub fn checks(&self) -> &[Check] {
        &self.checks
    }
    pub fn wants_old_values(&self) -> bool {
        self.return_old_values
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Validates key/value sizes and op/check counts. Called before admission.
    /// Checks are validated on the **same** key/value-size contract as ops and
    /// bounded by the same cap, so a batch cannot smuggle unbounded bytes through
    /// its conditions.
    pub fn validate(&self, max_ops: u32) -> Result<(), BatchError> {
        if self.ops.len() as u64 > u64::from(max_ops)
            || self.checks.len() as u64 > u64::from(max_ops)
        {
            return Err(BatchError::TooManyOps);
        }
        for op in &self.ops {
            op.validate()?;
        }
        for chk in &self.checks {
            chk.validate()?;
        }
        Ok(())
    }

    /// The admission cost. `worst_case_pages` and `wal_worst_case_bytes` are
    /// admission-time **estimates** the writer reconciles to the exact figures
    /// before publication (in *either* direction — see below).
    ///
    /// `worst_case_pages = 2·op_count + tree_depth_bound` is a heuristic, **not** a
    /// provable upper bound: the no-merge builder accumulates empty leaves across
    /// batches, so a tall-but-sparse tree can path-copy more pages than any
    /// count/height-derived bound sampled at admission predicts, and the tree
    /// can also grow between admission and application.
    /// Both reconciliations therefore tolerate an *upward* adjustment: the
    /// dirty-page charge is soft (`reconcile_to_actual` charges the overshoot,
    /// invariant 16 stays soft as in M3), and the WAL-byte reconcile charges the
    /// difference **and fails the batch pre-publication with `StoreFull` if it no
    /// longer fits `max_wal_bytes`** (a healthy fail-fast, never poison). An estimate above actual just reserves
    /// slack that reconciliation releases before publication.
    pub fn cost(&self, tree_depth_bound: u32) -> WriteCost {
        // Charge both ops and checks: the queued command owns every check's key
        // and expected value until it is applied, so those bytes count against
        // `max_command_bytes`. Saturating so hostile lengths cannot wrap.
        let ops_bytes: u64 = self
            .ops
            .iter()
            .map(Op::encoded_len)
            .fold(0, u64::saturating_add);
        let checks_bytes: u64 = self
            .checks
            .iter()
            .map(Check::encoded_len)
            .fold(0, u64::saturating_add);
        let encoded_bytes: u64 = ops_bytes.saturating_add(checks_bytes).saturating_add(8);
        let op_count = self.ops.len() as u32;
        // Heuristic estimate (not a hard bound — see doc above): ≤2 pages per op
        // (leaf copy + split) plus a nominal path depth. Both reconciliations
        // adjust to actual before publication.
        //
        // Overflow chains (M6 Phase F) are added on top, and unlike the rest of
        // this figure that term *is* exact: a spilled value's chain is
        // `ceil(len / payload cap)` pages, decided by length alone. Without it a
        // single 1 MiB value would under-count by ~65 pages and the strict WAL
        // reconcile would `StoreFull` a perfectly legal batch.
        let overflow_pages: u32 = self
            .ops
            .iter()
            .map(|op| match op {
                Op::Insert { value, .. } if value.len() > MAX_INLINE_VALUE_LEN => value
                    .len()
                    .div_ceil(crate::page::overflow_payload_capacity())
                    as u32,
                _ => 0,
            })
            .fold(0, u32::saturating_add);
        let worst_case_pages = op_count
            .saturating_mul(2)
            .saturating_add(tree_depth_bound)
            .saturating_add(overflow_pages)
            .max(1);
        // WAL record worst case (M4): the record carries full
        // `PAGE_SIZE` page *images*, so this must upper-bound the exact encoded
        // length — fixed record overhead plus every worst-case page's image and
        // section framing. `worst_case_pages` is a *heuristic* (see its doc
        // above — the no-merge builder can exceed it), which is why the writer's
        // WAL reconcile handles an overshoot with `StoreFull` rather than
        // trusting this estimate (`writer.rs`, bidirectional reconciliation).
        // The estimate exists to admit the common case cheaply, not to prove a
        // bound. (The M3 placeholder omitted the image bytes and would have
        // poisoned every write.)
        let wal_worst_case_bytes = crate::wal::RECORD_FIXED_OVERHEAD
            + u64::from(worst_case_pages)
                * (crate::page::PAGE_SIZE as u64 + crate::wal::PAGE_SECTION_OVERHEAD);
        WriteCost {
            encoded_bytes,
            op_count,
            worst_case_pages,
            wal_worst_case_bytes,
        }
    }

    /// The batch's admission cost using the engine's own default tree-depth term —
    /// the exact cost `BTreeMap::apply` reserves internally. A caller reserving a
    /// permit for this batch (`try_reserve_write`/`reserve_write`) should size it
    /// with this rather than duplicating the engine's private depth constant: it
    /// tracks the engine if that constant ever changes, so `WritePermit::apply`
    /// cannot start rejecting the batch with `PermitMismatch`.
    pub fn default_cost(&self) -> WriteCost {
        self.cost(crate::store::DEPTH_BOUND)
    }
}

/// The result of applying a batch.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ApplyOutcome {
    /// The batch applied; `version` is its published version. `old_values` is
    /// `Some` iff the batch requested them, aligned to `ops`.
    Applied {
        version: Version,
        old_values: Option<Vec<Option<Vec<u8>>>>,
    },
    /// A precondition failed. Nothing was published and no txid was consumed.
    /// `first_failed` indexes the first failing check.
    ConditionFailed { first_failed: usize },
}

/// The immutable copy-on-write page batch handed from the writer to the backend
/// completion path. Its pages are already inserted into the cache as
/// dirty/pinned before the root swap (invariant 2).
///
/// M0 does not build real pages; this type is defined
/// so the writer→backend hand-off signature is frozen. M3 fills `pages`.
#[derive(Clone, Debug)]
pub struct PageBatch {
    pub version: Version,
    pub pages: Vec<Page>,
    pub root: PageRef,
    pub logical_tail: u64,
    pub entry_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_and_cost() {
        let b = WriteBatch::new()
            .check(b"a".to_vec(), None)
            .insert(b"a".to_vec(), b"1".to_vec())
            .remove(b"b".to_vec())
            .returning_old_values();
        assert_eq!(b.ops().len(), 2);
        assert_eq!(b.checks().len(), 1);
        assert!(b.wants_old_values());
        let cost = b.cost(0);
        assert_eq!(cost.op_count, 2);
        assert!(cost.worst_case_pages >= 2);
    }

    #[test]
    fn validate_rejects_bad_ops() {
        let empty_key = WriteBatch::new().insert(Vec::new(), b"x".to_vec());
        assert_eq!(empty_key.validate(100), Err(BatchError::EmptyKey));

        let big = WriteBatch::new().insert(b"k".to_vec(), vec![0u8; MAX_VALUE_LEN + 1]);
        assert_eq!(big.validate(100), Err(BatchError::ValueTooLong));

        let many = {
            let mut b = WriteBatch::new();
            for i in 0..5u16 {
                b = b.insert(i.to_le_bytes().to_vec(), b"v".to_vec());
            }
            b
        };
        assert_eq!(many.validate(3), Err(BatchError::TooManyOps));
    }
}
