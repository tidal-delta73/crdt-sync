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

### Compaction

The add and tombstone history only grows, so long-lived replicas accumulate
records for elements long since deleted. `compact()` reclaims that space
without changing any visible element or add/remove semantics. For each tag
origin it finds the longest prefix `1..n` whose every add is present locally
*and* already covered by a tombstone, moves those adds and per-tag
tombstones out of the state, and records `n` as that origin's retired bound
in a bounded `compacted` summary. It stops at the first gap, a still-visible
tag, or a tombstone whose add was never observed, and returns the number of
add records removed; a repeated call returns `0`.

```python
a.add("apple"); a.add("banana")
a.remove("apple")
a.compact()        # 1 — retires apple's fully-dead tag
a.compact()        # 0 — nothing new to retire
a.elements()       # {"banana"} — unchanged
```

On merge the per-origin retired bounds join by maximum; add and tombstone
records at or below a bound are treated as already-consumed history, so a
late or duplicated pre-compaction snapshot can never resurrect a retired
element. Records above the bound merge as before, and ownership conflicts and
the local tag counter keep working. Compacted and uncompacted replicas,
exchanged in any order or direction, converge to the same elements and to the
same canonical causal state apart from the per-replica `replica_id`/`counter`,
and can keep adding and removing afterwards.

A snapshot uses the original four fields until compaction occurs; once a
summary exists it additionally carries `compacted`, a mapping of origin
replica id to its positive retired bound, with keys sorted. `from_snapshot`
accepts both the old and the new shape and preserves the summary across a
JSON round trip; explicit tags at or below a bound, like any other malformed
summary, are rejected.

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

## RGASession

A session bundling one RGA sequence replica and its vector clock under a
single `replica_id`, so the editable sequence and its causal context travel
as one offline-editing and reconnect unit. `insert`/`delete`/`values` behave
exactly as on a bare `RGA`; every successful edit additionally ticks the
session clock once, while a failed edit changes neither the sequence nor the
clock.

```python
from crdt_sync import RGASession

a = RGASession("A")
a.insert(0, "he")
a.insert(1, "llo")
a.values()            # ["he", "llo"]

snapshot = a.snapshot()           # one JSON-serializable exchange package:
                                  # {"replica_id", "rga", "clock"}, all "A"
restored = RGASession.from_snapshot(snapshot)

b = RGASession("B")
relation = b.merge(a)  # "after" — the peer is ahead; b absorbs both the
                       # sequence and the causal context, keeps its own id
```

`merge` first decides the peer's causal relation to the receiver
(`before`/`after`/`equal`/`concurrent`) from the two clocks, then merges
sequence and clock together and returns that relation. The merge is atomic:
a conflicting shared node id raises `ValueError` with the receiver's
sequence *and* clock untouched, and a non-session argument raises
`TypeError`. Snapshots are deeply independent and round-trip through JSON;
`from_snapshot` raises `ValueError` on missing or extra fields, invalid
nested snapshots, or disagreeing `replica_id` values, and a restored session
keeps its Lamport counter and clock components so continued editing never
reuses a node id or rolls the clock back. Exchanging session snapshots in
any order, with duplicates, converges every replica to the same `values()`,
shared RGA records and clock components.

### Causal compaction

Every successful `delete` is recorded under the clock dot it mints, so a
session snapshot carries mergeable deletion *causality*, not just a
tombstone set. `compact(stable_clock)` reclaims deleted RGA nodes against a
stability frontier the caller supplies — a `VectorClock` built from the
acknowledgements of every participant currently syncing, i.e. the clock
every one of them has confirmed observing:

```python
a.insert(0, "he"); a.insert(1, "llo"); a.delete(1)
stable = ...        # VectorClock all participants have acknowledged
a.compact(stable)   # count of node value records physically removed
a.compact(stable)   # 0 — the same frontier retires nothing new
```

A node is reclaimed only when one of its deletion events is covered by the
frontier, it is not a legacy tombstone without causal information, and no
node surviving as a full record still names it as predecessor. Eligible
leaves are peeled repeatedly, so a whole stable deleted branch comes away in
one call, while compaction never crosses a live node, an unstabilized
delete, or a still-referenced node. The call never advances the session
clock and never changes `values()`.

A reclaimed node's value record disappears (that count is the return
value), but a permanent *retired summary* remains: the node id with its
predecessor edge as a tombstoned skeleton (so a branch that still hangs off
the node keeps weaving in place), the retired deletion dots, and a
per-origin retired clock prefix. Consequently an old pre-compaction
snapshot, a duplicate packet or an out-of-order delivery cannot resurrect
a retired node, and a compacted replica merging an uncompacted one — in
either direction — converges on the same visible sequence, clock and
shared snapshot content (each replica keeping only its own `replica_id`).
The bare `RGA` has no compaction entry point and is otherwise unchanged.

`compact` raises `TypeError` when its argument is not a `VectorClock`, and
`ValueError` — leaving the session completely unchanged — when the frontier
is concurrent with or ahead of the session clock. A same node id carrying
conflicting content still raises `ValueError` on merge, with the sequence,
clock and compaction metadata kept atomic.

The session snapshot keeps the original three fields until a node is
retired; afterwards it additionally carries a deterministic `compaction`
summary. Deletion causality rides on the tombstones: a tagged tombstone is
`[id, [[origin, counter], ...]]` while a tombstone restored from a legacy
plain-id snapshot stays `[sequence, origin]`, carries no deletion time and
is treated as permanently ineligible for compaction. Both the legacy
three-field session shape and the legacy four-field RGA snapshot are still
accepted with their original merge, continued-editing and exception
semantics.

