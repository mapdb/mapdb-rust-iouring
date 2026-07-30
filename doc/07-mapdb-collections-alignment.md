# 07 — API alignment with mapdb-collections

Record of the 2026-07-19 vocabulary alignment between this crate's
`SortedMap` and the sibling **`mapdb-collections`** crate
(<https://github.com/mapdb/mapdb-rust>), a Rust port of Eclipse Collections. Important
framing: that crate is **purely in-memory** (no store, no durability, no
transactions, no serializers), so there is no storage-engine interface to
plug into; the transferable value is its deliberately std-aligned naming
(it ran a documented de-Java-fication pass, `BREAKING-V2.md` there).

## Adopted (renames / additions)

| Concept | mapdb-uring now | Was | mapdb-collections precedent |
|---|---|---|---|
| single write | `insert` | `put` | `TreeMap::insert` |
| bulk write | `put_all` | `put_many` | `TreeMap::put_all` (the one surviving `put_*`) |
| delete | `remove` (unchanged) | — | `remove`, never `delete` |
| count | `len` / `is_empty` (unchanged) | — | `len`, never `size` |
| smallest/largest | `first_entry` / `last_entry` | `first` / `last` | `first_entry`/`last_entry` |
| `<= k` / `< k` | `floor_entry` / `lower_entry` (new) | — | NavigableMap family |
| `>= k` / `> k` | `ceiling_entry` / `higher_entry` (new) | — | NavigableMap family |
| bulk delete | `remove_range(range)` + `clear()` (new) | — | `TreeMap::remove_range` |

Implementation notes: `floor`/`lower` are a single root-to-leaf descent
carrying the nearest left-sibling subtree as fallback (`read::tree_floor`);
`ceiling`/`higher` are a bounded scan stopping at its first entry.
`remove_range` is scan-then-remove in chunks of 512 ops (each chunk atomic,
the whole range **not** — a single batch's worst-case page cost must fit
the admission budget; document nearby).

## Deliberate divergences (storage semantics, kept)

- Mutations return the published **`Version`** (durability watermark), not
  the displaced value — `commit`/`flush` consume it. The in-memory crate
  returns `Option<V>`; mirroring that would erase the durability contract.
- **`put_if_absent` / `compare_and_set`** exist here; mapdb-collections has
  no CAS at all (its conditional-upsert idiom is the std `entry()` API,
  which `TreeMap` there doesn't even have). An `entry`-style API could be
  layered later if demanded.
- **Scans are consumer callbacks with `ControlFlow` early-exit**, not
  iterators. Their `range()` returns a lazy `Iterator`; the async analog is
  a `Stream` cursor, which remains deferred — mapdb-collections'
  `for_each` is the closest cousin (theirs lacks early-exit; ours needs it
  because entries cost I/O).
- **Prefix scans** (`for_each_prefix`) have no precedent there; they're a
  storage-layer addition.
- Everything is **async** with `Send` futures.

## Not adopted (and why)

- `Compare<K>` comparator type parameter — byte keys are lexicographic by
  contract; adding an ordering parameter would complicate the persisted
  format for no current need.
- `rank`/`select` order statistics — branches carry no subtree entry
  counts; sublinear rank would be a format change.
- `range()`-as-iterator — deferred with the cursor/`Stream` work. The
  hazards it has to solve first are snapshot-lease lifetime across a
  suspended cursor, and what a cursor should do when its generation is
  revoked under retention pressure. Reverse iteration itself is **not**
  deferred: descending scans
  shipped as `for_each_range_rev` / `collect_range_rev` (2026-07-19).
