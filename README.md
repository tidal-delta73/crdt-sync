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
