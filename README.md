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

