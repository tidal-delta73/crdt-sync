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

An observed-remove set of strings. Each `add` mints a unique tag; `remove`
only tombstones the tags the replica has observed at call time, so a
concurrent add on another replica survives until it has been observed.

```python
from crdt_sync import ORSet

a = ORSet("A")
a.add("apple")
a.contains("apple")   # True
a.elements()          # {"apple"} — an independent copy
a.remove("apple")     # True — removes the observed additions
a.remove("apple")     # False — nothing visible, state unchanged

# An element removed in the past can be added again; the new add is never
# swallowed by the historical removal.
a.add("apple")

snapshot = a.snapshot()           # JSON-serializable dict
restored = ORSet.from_snapshot(snapshot)

b = ORSet("B")
a.merge(b)            # in place, returns a; b is left unchanged
```

If replica A removes an element without observing B's concurrent add of the
same element, that add remains visible once states are exchanged. If A first
merges (observing B's add) and then removes, the element disappears
everywhere. Merges union both add tags and tombstones, so stale, duplicated
or reordered snapshots never resurrect a removed addition.
