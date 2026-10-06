"""Merge algebra (idempotence / commutativity / associativity) and isolation.

These checks pin the documented contract: ``merge`` returns ``self`` and does
not touch its argument, while ``snapshot``/``from_snapshot`` hand out fully
independent, JSON-serializable copies.
"""

from __future__ import annotations

import copy
import itertools
import json

import pytest

from crdt_sync import GCounter


def build(replica_id: str, amounts: list[int]) -> GCounter:
    counter = GCounter(replica_id)
    for amount in amounts:
        counter.increment(amount)
    return counter


# Reachability set: several distinct, partially-overlapping states.
def states():
    ab = build("A", [2, 4])
    ab.merge(build("B", [1, 2]))  # {A: 6, B: 3}
    ac = build("A", [2, 4])
    ac.merge(build("C", [7]))     # {A: 6, C: 7}
    return {
        "A": build("A", [2, 4]),       # {A: 6}
        "B": build("B", [3]),          # {B: 3}
        "C": build("C", [1, 1, 5]),    # {C: 7}
        "AB": ab,
        "AC": ac,
    }


def all_components(snapshots) -> dict[str, int]:
    merged: dict[str, int] = {}
    for snap in snapshots:
        for key, count in snap["counts"].items():
            merged[key] = max(merged.get(key, 0), count)
    return merged


def test_merge_returns_receiver_self():
    receiver = build("A", [1])
    donor = build("B", [1])
    assert receiver.merge(donor) is receiver
    # Chained merges still hand back the same object.
    assert receiver.merge(build("C", [1])) is receiver


def test_merge_does_not_modify_argument():
    receiver = build("A", [9])
    donor = build("B", [3])
    donor_snapshot = copy.deepcopy(donor.snapshot())

    receiver.merge(donor)
    assert donor.snapshot() == donor_snapshot
    assert donor.value() == 3

    # Merging a "smaller" state into a superset also leaves the donor intact.
    donor.merge(receiver)
    assert donor.value() == 3 + 9
    assert receiver.value() == 12  # receiver was not altered by donor's merge


def test_commutativity_pairwise():
    ss = states()
    for left_key, right_key in itertools.combinations(ss, 2):
        left = GCounter.from_snapshot(ss[left_key].snapshot())
        right = GCounter.from_snapshot(ss[right_key].snapshot())

        lr = GCounter("x").merge(left).merge(right)
        rl = GCounter("x").merge(right).merge(left)
        assert lr.snapshot() == rl.snapshot()


def test_commutativity_all_orders_same_snapshot():
    originals = list(states().values())
    expected = all_components(s.snapshot() for s in originals)

    for order in itertools.permutations(range(len(originals))):
        receiver = GCounter("x")
        for i in order:
            receiver.merge(originals[i])
        assert receiver.snapshot()["counts"] == expected


def test_associativity_groupings():
    originals = list(states().values())
    expected = all_components(s.snapshot() for s in originals)

    # Left grouping: ((s0 ⋈ s1) ⋈ s2) ⋈ ...
    left = GCounter("g")
    for s in originals:
        left.merge(s)

    # Right grouping: s0 ⋈ (s1 ⋈ (s2 ⋈ s3 ⋈ s4))
    inner = GCounter("inner")
    for s in reversed(originals[1:]):
        inner.merge(s)
    right = GCounter("g").merge(originals[0]).merge(inner)

    # Explicit balanced grouping for the three "leaf" states.
    leaves = [build("A", [2, 4]), build("B", [3]), build("C", [1, 1, 5])]
    left_pair = GCounter("p").merge(leaves[0]).merge(leaves[1])
    balanced = GCounter("g").merge(left_pair).merge(leaves[2])
    right_pair = GCounter("p").merge(leaves[1]).merge(leaves[2])
    balanced2 = GCounter("g").merge(leaves[0]).merge(right_pair)

    for grouped in (left, right, balanced, balanced2):
        assert grouped.snapshot()["counts"] == expected
        assert grouped.value() == sum(expected.values())


def test_idempotent_merge_with_self():
    counter = build("A", [2, 4])
    before = copy.deepcopy(counter.snapshot())
    assert counter.merge(counter) is counter
    assert counter.snapshot() == before
    counter.merge(counter).merge(counter)
    assert counter.snapshot() == before


def test_idempotent_merge_with_equivalent_copy():
    counter = build("A", [2, 4])
    counter.merge(build("C", [7]))
    equivalent = GCounter.from_snapshot(counter.snapshot())
    equivalent_2 = GCounter.from_snapshot(json.loads(json.dumps(counter.snapshot())))
    before = copy.deepcopy(counter.snapshot())

    counter.merge(equivalent)
    assert counter.snapshot() == before
    counter.merge(equivalent_2)
    assert counter.snapshot() == before


def test_snapshot_is_independent_and_json_serializable():
    counter = build("A", [2])
    counter.merge(build("B", [3]))

    first = counter.snapshot()
    second = counter.snapshot()
    assert first == second
    assert first is not second
    assert first["counts"] is not second["counts"]

    # Mutating both the returned dict and its nested counts must not leak in.
    first["replica_id"] = "tampered"
    first["counts"]["A"] = 999
    first["counts"]["ZZZ"] = 1
    assert counter.replica_id == "A"
    assert counter.snapshot() == second
    assert counter.value() == 5

    json.loads(json.dumps(second))


def test_from_snapshot_preserves_id_and_components():
    source = build("A", [2, 4])
    source.merge(build("B", [3]))
    wire = source.snapshot()
    wire_copy = copy.deepcopy(wire)

    restored = GCounter.from_snapshot(wire)
    assert restored.replica_id == "A"
    assert restored.snapshot()["counts"] == {"A": 6, "B": 3}
    assert restored.value() == 9

    # Subsequent increments only touch the restored replica's own component.
    restored.increment(5)
    assert restored.snapshot()["counts"] == {"A": 11, "B": 3}
    assert restored.value() == 14

    # Neither the snapshot dict handed in nor the originating counter changed.
    assert wire == wire_copy
    assert source.snapshot()["counts"] == {"A": 6, "B": 3}
    assert source.value() == 9


def test_restored_zero_component_replica_only_touches_own_component():
    # A snapshot may legitimately contain a zero component for a known replica.
    restored = GCounter.from_snapshot(
        {"replica_id": "Z", "counts": {"A": 4, "Z": 0}}
    )
    restored.increment(2)
    assert restored.snapshot()["counts"] == {"A": 4, "Z": 2}


def test_zero_value_replica_is_merge_neutral():
    idle = GCounter("idle")
    assert idle.value() == 0
    assert idle.snapshot()["counts"] == {}

    active = build("A", [5])
    before = copy.deepcopy(active.snapshot())
    active.merge(idle)
    assert active.snapshot() == before
    idle.merge(active)
    assert idle.value() == 5
    assert active.value() == 5


def test_legal_zero_components_survive_merge():
    host = GCounter("H")
    guest = GCounter.from_snapshot(
        {"replica_id": "G", "counts": {"G": 0, "H": 3}}
    )
    host.increment(1)
    host.merge(guest)

    # Merge takes a component-wise max; a zero component is value-neutral, so
    # it need not be materialized on a receiver that had never seen "G".
    merged = host.snapshot()["counts"]
    assert merged.get("G", 0) == 0
    assert merged["H"] == 3
    assert host.value() == 3

    # A zero component explicitly present on the receiver is preserved.
    keeper = GCounter.from_snapshot(
        {"replica_id": "H", "counts": {"G": 0, "H": 1}}
    )
    keeper.merge(guest)
    assert keeper.snapshot()["counts"] == {"G": 0, "H": 3}


@pytest.mark.parametrize("replica_count", [1, 2, 3, 5])
def test_scales_across_replica_counts(replica_count):
    receivers = [GCounter(f"r{i}") for i in range(replica_count)]
    donors = []
    total = 0
    for i in range(replica_count):
        donor = GCounter(f"d{i}")
        amount = i * 3 + 1  # strictly positive, varied
        donor.increment(amount)
        total += amount
        donors.append(donor)

    for order, receiver in enumerate(receivers):
        for i in [(j + order) % replica_count for j in range(replica_count)]:
            receiver.merge(donors[i])
        assert receiver.value() == total

    canonical = receivers[0].snapshot()
    for receiver in receivers[1:]:
        assert receiver.snapshot()["counts"] == canonical["counts"]
