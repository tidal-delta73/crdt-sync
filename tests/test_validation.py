"""Input validation pinned to the current exception contract.

* ``replica_id``: non-string -> ``TypeError``, empty string -> ``ValueError``.
* ``increment``: bool / non-integer -> ``TypeError``, <=0 -> ``ValueError``.
* ``merge``: non-``GCounter`` -> ``TypeError``.
* ``from_snapshot``: non-dict -> ``TypeError``; malformed contents ->
  ``ValueError``.
"""

from __future__ import annotations

import copy

import pytest

from crdt_sync import GCounter


# --------------------------------------------------------------------------- #
# replica_id
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_id",
    [None, 1, 1.0, b"r", [], {}, True, False, object(), ("a",)],
)
def test_constructor_rejects_non_string_replica_id(bad_id):
    with pytest.raises(TypeError):
        GCounter(bad_id)


def test_constructor_rejects_empty_replica_id():
    with pytest.raises(ValueError):
        GCounter("")


def test_whitespace_replica_id_is_allowed():
    counter = GCounter(" ")  # non-empty, therefore legal
    assert counter.replica_id == " "
    counter.increment()
    assert counter.value() == 1


# --------------------------------------------------------------------------- #
# increment
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_amount", [True, False, 1.0, 2.5, "1", None, [1], 1j])
def test_increment_rejects_non_integer(bad_amount):
    counter = GCounter("A")
    with pytest.raises(TypeError):
        counter.increment(bad_amount)
    assert counter.value() == 0  # rejected call changed nothing


@pytest.mark.parametrize("bad_amount", [0, -1, -100, -10**9])
def test_increment_rejects_non_positive(bad_amount):
    counter = GCounter("A")
    counter.increment(3)
    with pytest.raises(ValueError):
        counter.increment(bad_amount)
    assert counter.value() == 3  # rejected call changed nothing


def test_increment_defaults_to_one_and_accumulates():
    counter = GCounter("A")
    counter.increment()
    counter.increment()
    assert counter.value() == 2
    counter.increment(10**6)
    assert counter.value() == 1_000_002


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("other", [None, 1, "GCounter", [], {}, object(), 1.0])
def test_merge_rejects_non_counter(other):
    counter = GCounter("A")
    counter.increment(1)
    with pytest.raises(TypeError):
        counter.merge(other)
    assert counter.value() == 1  # rejected call changed nothing


def test_merge_rejects_raw_snapshot_dict():
    # A snapshot merely describes a counter; it is not itself a GCounter.
    donor = GCounter("B")
    donor.increment(2)
    receiver = GCounter("A")
    with pytest.raises(TypeError):
        receiver.merge(donor.snapshot())
    assert receiver.value() == 0


def test_merge_accepts_gcounter_subclass():
    class TaggedCounter(GCounter):
        pass

    receiver = GCounter("A")
    receiver.merge(TaggedCounter("T"))
    assert receiver.value() == 0


# --------------------------------------------------------------------------- #
# from_snapshot
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_snapshot",
    [None, [], "x", 42, 4.0, True, False, {1, 2}, ("A", {})],
)
def test_from_snapshot_rejects_non_dict(bad_snapshot):
    with pytest.raises(TypeError):
        GCounter.from_snapshot(bad_snapshot)


def _valid_snapshot():
    return {"replica_id": "A", "counts": {"A": 1, "B": 2}}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("counts"),
        lambda s: s.pop("replica_id"),
        lambda s: s.update(extra=1),
        lambda s: (s.pop("replica_id"), s.update(id="A")),
        lambda s: s.update(nested={"x": 1}),
    ],
)
def test_from_snapshot_requires_exact_key_set(mutate):
    snapshot = _valid_snapshot()
    mutate(snapshot)
    with pytest.raises(ValueError):
        GCounter.from_snapshot(snapshot)


@pytest.mark.parametrize("bad_id", ["", None, 1, 1.0, True, False, [], {}])
def test_from_snapshot_rejects_invalid_replica_id(bad_id):
    snapshot = _valid_snapshot()
    snapshot["replica_id"] = bad_id
    with pytest.raises(ValueError):
        GCounter.from_snapshot(snapshot)


@pytest.mark.parametrize("bad_counts", [None, [], 1, "x", True, 1.0, {1, 2}])
def test_from_snapshot_rejects_non_dict_counts(bad_counts):
    snapshot = _valid_snapshot()
    snapshot["counts"] = bad_counts
    with pytest.raises(ValueError):
        GCounter.from_snapshot(snapshot)


@pytest.mark.parametrize("bad_key", ["", 1, 1.0, None, True, ("A",)])
def test_from_snapshot_rejects_invalid_component_keys(bad_key):
    snapshot = _valid_snapshot()
    snapshot["counts"] = {"A": 1, bad_key: 3}
    with pytest.raises(ValueError):
        GCounter.from_snapshot(snapshot)


@pytest.mark.parametrize("bad_value", [True, False, -1, -100, 1.0, 2.5, "1", None, []])
def test_from_snapshot_rejects_invalid_component_values(bad_value):
    snapshot = _valid_snapshot()
    snapshot["counts"] = {"A": bad_value}
    with pytest.raises(ValueError):
        GCounter.from_snapshot(snapshot)


def test_from_snapshot_accepts_zero_components_and_empty_counts():
    empty = GCounter.from_snapshot({"replica_id": "A", "counts": {}})
    assert empty.replica_id == "A"
    assert empty.value() == 0

    zeroed = GCounter.from_snapshot(
        {"replica_id": "A", "counts": {"A": 0, "B": 0}}
    )
    assert zeroed.value() == 0
    zeroed.increment(1)
    assert zeroed.value() == 1


def test_from_snapshot_round_trips_live_snapshots():
    source = GCounter("A")
    for amount in (1, 4, 2):
        source.increment(amount)
    other = GCounter("B")
    other.increment(7)
    source.merge(other)

    wire = source.snapshot()
    restored = GCounter.from_snapshot(copy.deepcopy(wire))
    assert restored.replica_id == source.replica_id
    assert restored.snapshot()["counts"] == wire["counts"]
    assert restored.value() == source.value()


def test_from_snapshot_does_not_alias_input_dict():
    wire = {"replica_id": "A", "counts": {"A": 1}}
    restored = GCounter.from_snapshot(wire)
    restored.increment(5)
    assert wire == {"replica_id": "A", "counts": {"A": 1}}
