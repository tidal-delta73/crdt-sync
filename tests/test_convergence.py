"""Convergence under duplicate, reordered, batched and interleaved delivery.

All state is produced through the public API; expectations are computed from
snapshots with plain dict operations so nothing depends on private fields,
dict iteration order or random seeds.
"""

from __future__ import annotations

import copy
import itertools
import json

import pytest

from crdt_sync import GCounter

from conftest import merge_states, replica_history

# Three replicas, distinct non-empty ids, several rounds of positive
# increments of differing sizes. Final components: A=7, B=6, C=15.
AMOUNTS = {
    "A": [1, 4, 2],
    "B": [3, 3],
    "C": [2, 5, 7, 1],
}


@pytest.fixture(scope="module")
def histories():
    return {
        rid: replica_history(rid, amounts)[1]
        for rid, amounts in AMOUNTS.items()
    }


@pytest.fixture(scope="module")
def finals(histories):
    return [histories[rid][-1] for rid in ("A", "B", "C")]


@pytest.fixture(scope="module")
def expected(histories, finals):
    """Component-wise maximum of the final states, computed independently."""
    expected_counts: dict[str, int] = {}
    for state in finals:
        for key, count in sorted(state["counts"].items()):
            expected_counts[key] = max(expected_counts.get(key, 0), count)
    return expected_counts


def unique_permutations(seq, limit=None):
    """Distinct permutations of an index sequence (handles duplicated items)."""
    seen: set[tuple] = set()
    out: list[tuple] = []
    for perm in itertools.permutations(seq):
        if perm not in seen:
            seen.add(perm)
            out.append(perm)
            if limit is not None and len(out) >= limit:
                break
    return out


def test_finals_are_independent_snapshots(finals):
    # Delivered states must be standalone data the sender could not mutate.
    for state in finals:
        assert set(state) == {"replica_id", "counts"}
        json.dumps(state)  # JSON-serializable
    assert finals[0] is not finals[1]


def test_every_permutation_converges(finals, expected):
    for order in itertools.permutations(range(len(finals))):
        receiver = GCounter(f"rcv-{order}")
        merge_states(receiver, [finals[i] for i in order])
        assert receiver.value() == sum(expected.values())
        assert receiver.snapshot()["counts"] == expected


def test_duplicated_delivery_converges(finals, expected):
    # Every final state is delivered twice, in a variety of permutations.
    for order in unique_permutations([0, 1, 2, 0, 1, 2], limit=10):
        receiver = GCounter(f"dup-{order}")
        merge_states(receiver, [finals[i] for i in order])
        assert receiver.value() == sum(expected.values())
        assert receiver.snapshot()["counts"] == expected


def test_intermediate_states_interleaved_converge(histories, expected):
    entries = [
        copy.deepcopy(state)
        for rid in ("A", "B", "C")
        for state in histories[rid]
    ]

    orders = {
        "forward": entries,
        "reverse": list(reversed(entries)),
        "round_robin": [
            state
            for step in itertools.zip_longest(
                *(histories[rid] for rid in ("A", "B", "C"))
            )
            for state in step
            if state is not None
        ],
    }
    for order in unique_permutations(range(len(entries)), limit=6):
        orders[f"perm-{order}"] = [entries[i] for i in order]

    for label, states in orders.items():
        receiver = GCounter(f"interleave-{label}")
        merge_states(receiver, states)
        assert receiver.value() == sum(expected.values())
        assert receiver.snapshot()["counts"] == expected


def test_zero_value_replica_delivery_does_not_disturb(finals, expected):
    idle = GCounter("idle")  # never incremented
    idle_state = idle.snapshot()
    assert idle_state["counts"] == {}

    receiver = GCounter("rcv-zero")
    merge_states(receiver, [idle_state, finals[0], idle_state])
    merge_states(receiver, [finals[1], idle_state, finals[2], idle_state])
    assert receiver.snapshot()["counts"] == expected
    assert receiver.value() == sum(expected.values())


def test_distinct_receivers_agree_across_paths(histories, finals, expected):
    """Receivers that got the complete set of states agree, whatever path."""
    paths = {}

    r1 = GCounter("r1")
    merge_states(r1, finals)
    paths["plain"] = r1

    r2 = GCounter("r2")
    duplicated = [finals[i] for i in (0, 0, 1, 2, 1, 2, 0)]
    merge_states(r2, duplicated)
    paths["duplicated"] = r2

    staggered = [
        state
        for step in itertools.zip_longest(
            *(histories[rid] for rid in ("A", "B", "C"))
        )
        for state in step
        if state is not None
    ]
    r3 = GCounter("r3")
    merge_states(r3, staggered)
    paths["staggered"] = r3

    values = {receiver.value() for receiver in paths.values()}
    assert values == {sum(expected.values())}
    counts = {
        tuple(sorted(receiver.snapshot()["counts"].items()))
        for receiver in paths.values()
    }
    assert counts == {tuple(sorted(expected.items()))}


def test_snapshot_roundtrip_mid_delivery(finals, expected):
    receiver = GCounter("rt")
    merge_states(receiver, [finals[0]])

    # Snapshot -> JSON -> restore on another process-like hop, then continue.
    wire = json.loads(json.dumps(receiver.snapshot()))
    resumed = GCounter.from_snapshot(wire)
    assert resumed.replica_id == "rt"
    merge_states(resumed, [finals[2], finals[1]])  # deliberately out of order

    assert resumed.value() == sum(expected.values())
    assert resumed.snapshot()["counts"] == expected


def test_redelivering_old_or_same_state_is_a_noop(histories, finals, expected):
    receiver = GCounter("settled")
    merge_states(receiver, finals)
    before = copy.deepcopy(receiver.snapshot())

    old_states = [
        state
        for rid in ("A", "B", "C")
        for state in histories[rid]  # includes every stale intermediate state
    ]
    redeliveries = {
        "same-finals": finals + finals,
        "oldest-first": list(reversed(old_states)),
        "oldest-last": old_states,
        "scrambled": [old_states[i] for i in (0, 8, 1, 7, 2, 6, 3, 5, 4)],
    }
    for label, states in redeliveries.items():
        merge_states(receiver, states)
        assert receiver.value() == sum(expected.values()), label
        assert receiver.snapshot() == before, label


@pytest.mark.parametrize(
    "amounts",
    [
        [1],
        [1, 1, 1],
        [2, 8, 3, 100],
        [10**6, 10**9],
    ],
)
def test_single_and_multi_replica_convergence(amounts):
    """Different replica counts and positive increments all converge."""
    counter, history = replica_history("solo", amounts)
    total = sum(amounts)
    assert counter.value() == total

    # Replaying every intermediate snapshot any number of times is stable.
    for state in history:
        replay = GCounter("observer")
        merge_states(replay, [state, state])
        assert replay.value() == state["counts"]["solo"]

    fresh = GCounter("observer")
    merge_states(fresh, history + history[::-1])
    assert fresh.value() == total
