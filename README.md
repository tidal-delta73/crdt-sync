# crdt-sync

CRDT-based collaborative sync engine.

Pure-Python, no runtime dependencies.

## Usage

```bash
python3 -m crdt_sync version
python3 -m crdt_sync help
```

## GCounter

A grow-only counter CRDT. Each instance is a replica identified by a
non-empty `replica_id`; replicas exchange snapshots and merge by taking the
component-wise maximum.

```python
from crdt_sync import GCounter

a = GCounter("A")
a.increment()      # add 1
a.increment(5)     # or a positive integer amount
a.value()          # 6 — sum of all known replica components

snapshot = a.snapshot()          # JSON-serializable dict
restored = GCounter.from_snapshot(snapshot)

b = GCounter("B")
b.increment(2)
a.merge(b)          # in place, returns a; b is left unchanged
```

Merges are idempotent, commutative, and associative: duplicates or reordered
delivery converge to the same value.

## ORSet

An observed-remove set CRDT for a removable set of strings. Each instance is
a replica identified by a non-empty `replica_id`. Every `add` mints a unique
tag; `remove` tombstones only the additions currently visible on that
replica, so a concurrent same-name add on a peer survives, while an observed
add does not.

```python
from crdt_sync import ORSet

a = ORSet("A")
a.add("apple")
a.add("banana")
a.contains("apple")   # True
a.elements()          # {"apple", "banana"} — an independent copy
a.remove("apple")     # True — withdraws the observed add
a.remove("apple")     # False — nothing visible, state unchanged
a.add("apple")        # a fresh add is never swallowed by the old tombstone

snapshot = a.snapshot()          # JSON-serializable dict
restored = ORSet.from_snapshot(snapshot)

b = ORSet("B")
b.add("cherry")
a.merge(b)          # in place, returns a; b is left unchanged
```

Merges are idempotent, commutative, and associative: stale, duplicated or
reordered snapshots converge to the same elements and can never resurrect a
removed addition.

### Tombstone compaction

The tag map and tombstone set normally grow without bound. `compact()`
retires the fully-deleted history into a bounded causal summary. For each
tag origin it finds, starting at sequence 1, the longest prefix whose adds
are all present and all tombstoned; it moves those add records and per-tag
tombstones out of the snapshot and records the origin's retired-up-to
sequence. It stops at the first gap, still-visible tag, or unseen history,
never crosses a live add, and returns the number of add records removed.
`elements()`, `contains()` and future `add()` results are unaffected; a
second immediate call returns `0`.

```python
a.add("old")
a.remove("old")
a.compact()          # 1 — one add/tombstone pair retired
a.compact()          # 0 — there is no new fully-deleted prefix
```

Retired bounds merge by per-origin maximum. Any add or tombstone at or below
a known bound arriving in a late or duplicated snapshot is treated as
already-consumed history and ignored, so compaction never allows
resurrection; records above the bound merge exactly as before, including
ownership-conflict detection and counter advancement. A never-compacted
replica exchanging snapshots with a compacted one converges to the same
elements and canonical state (apart from its own `replica_id`) and can keep
adding and deleting.

Snapshots keep the four-field shape until any compaction has happened; once
a summary exists they carry an additional `compacted` mapping of origin ids
to positive retired bounds, with keys in sorted order. `from_snapshot`
accepts both shapes; malformed summaries (non-dict, empty origin, bool/non-
positive bound, extra fields) or explicit tags/tombstones at or below the
corresponding bound raise `ValueError`.

## LWWRegister

A last-writer-wins register CRDT for offline-writable JSON values. Each
instance is a replica identified by a non-empty `replica_id`. A fresh
register holds no value (`has_value()` is `False` and `value()` raises
`LookupError`). Every `assign` advances a local logical clock and timestamps
the write with `(logical_count, replica_id)`; `merge` keeps the entry with
the greater timestamp (count first, then the writing replica id's Unicode
order), and also remembers the greatest count observed from any peer so the
next local write is always later.

```python
from crdt_sync import LWWRegister

a = LWWRegister("A")
a.has_value()        # False
a.assign({"v": [1, True, None]})   # null/bool/str/finite numbers, lists, dicts
a.has_value()        # True
a.value()            # {"v": [1, True, None]} — an independent deep copy

snapshot = a.snapshot()          # JSON-serializable dict
restored = LWWRegister.from_snapshot(snapshot)

b = LWWRegister("B")
b.assign("offline write")
a.merge(b)          # in place, returns a; b is left unchanged
```

Merges are idempotent, commutative, and associative: duplicated, reordered or
bidirectionally exchanged snapshots converge to the same value and logical
clock. If the very same timestamp ever carries two structurally different
values (which means two live replicas share a `replica_id`), `merge` raises
`ValueError` and leaves both registers untouched.

## RGA

A replicated growable array CRDT for an editable, deletable sequence of
strings. Each instance is a replica identified by a non-empty `replica_id`; a
fresh instance is empty. Every `insert(index, value)` mints a unique node id
from a local Lamport-style counter and the replica id; the node hangs off the
visible element immediately to its left. Concurrent inserts sharing a
predecessor order by id in descending order — greater sequence first, then
the replica id's Unicode order — and each node's own branch follows
immediately, so the same set of operations always produces one sequence.

`delete(index)` is observed-remove: it tombstones only the element currently
visible at that position and returns its string. The node keeps its identity
as a tombstone, so redelivery never errors or resurrects it; an insert never
observed by the remover is unaffected, and a later insert at the same slot is
never swallowed by the old tombstone.

```python
from crdt_sync import RGA

a = RGA("A")
a.insert(0, "he")      # insert at a visible zero-based position
a.insert(1, "llo")     # append at the current length
a.values()             # ["he", "llo"] — an independent list
print(a.delete(1))     # "llo" — returns the removed string
a.values()             # ["he"]

snapshot = a.snapshot()          # JSON-serializable dict
restored = RGA.from_snapshot(snapshot)

b = RGA("B")
b.merge(a)             # in place on b, returns b; a is left unchanged
```

Merges are idempotent, commutative, and associative: stale, duplicated,
reordered or off-line-batched snapshots converge to the same `values()`, and
converged replicas share identical insertion records, predecessor links and
tombstones. JSON round-tripping a snapshot and continuing to insert never
reuses an old id. If the very same node id ever carries a different string or
predecessor (which means two live replicas share a `replica_id`), `merge`
raises `ValueError` and leaves both states untouched.

## VectorClock

A reusable causal context for ordering offline operations and deciding what
already happened at reconnect. Each clock is a replica identified by a
non-empty `replica_id` and keeps one non-negative integer component per
known replica; a fresh clock knows no components. `tick(amount=1)` advances
only the local replica's component; `merge` takes the component-wise maximum;
`compare` answers the happens-before relation (`before`, `after`, `equal`,
`concurrent`), treating components it has never seen as zero.

```python
from crdt_sync import VectorClock

a = VectorClock("A")
b = VectorClock("B")
a.tick()
b.tick()
a.compare(b)        # "concurrent" — independent offline ticks

a.merge(b)          # in place, returns a; b is left unchanged
a.tick()
a.compare(b)        # "after" — merge, then local progress
b.compare(a)        # "before"
a.components()      # {"A": 2, "B": 1} — an independent copy

snapshot = a.snapshot()             # {"replica_id": "A", "clock": {...}}
restored = VectorClock.from_snapshot(snapshot)
restored.tick(3)                    # continues from A's restored component
```

Snapshots are JSON-serializable dicts containing exactly `replica_id` and
`clock`; the `clock` mapping always lists its components sorted by replica
id and round-trips through JSON unchanged. `merge` is idempotent, commutative
and associative, so duplicate, reordered or intermediate-relayed state
exchanges converge to identical components, and converged clocks compare as
`equal`.

