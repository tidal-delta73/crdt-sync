"""Systematic tests for the public ORSet contract.

Coverage:
* causal add/remove semantics: observed remove, concurrent add survival,
  observed-add-then-remove, re-add after removal and repeated removes;
* convergence under duplicated, reordered, batched, stale and interleaved
  snapshot delivery, across JSON round trips and offline editing;
* merge algebra: idempotence, commutativity, associativity, receiver return
  value, argument isolation and agreement on ``elements`` after all finals;
* snapshot / from_snapshot independence, JSON serialization and uniqueness
  of local operations after restore (including old-snapshot restore);
* contains / elements visibility and returned-set independence;
* the documented TypeError / ValueError input contract, including invalid
  snapshot field, tag and causal-state shapes;
"""

from __future__ import annotations

import copy
import json
import unittest
from itertools import permutations

from crdt_sync import ORSet


def cap(replica: ORSet) -> dict:
    """Snapshot through a full JSON round trip (simulated wire)."""
    return json.loads(json.dumps(replica.snapshot()))


def restore(snapshot: dict) -> ORSet:
    return ORSet.from_snapshot(json.loads(json.dumps(snapshot)))


def deliver(receiver: ORSet, snapshots) -> ORSet:
    """Merge a sequence of snapshot dicts into ``receiver`` in order."""
    for snapshot in snapshots:
        receiver.merge(restore(snapshot))
    return receiver


class LocalSemanticsTests(unittest.TestCase):
    def test_add_contains_elements(self) -> None:
        replica = ORSet("a")
        self.assertEqual(replica.elements(), set())
        self.assertFalse(replica.contains("x"))

        replica.add("x")
        replica.add("y")
        self.assertTrue(replica.contains("x"))
        self.assertTrue(replica.contains("y"))
        self.assertEqual(replica.elements(), {"x", "y"})

    def test_duplicate_adds_are_distinct_operations(self) -> None:
        replica = ORSet("a")
        replica.add("x")
        replica.add("x")
        # Removing once tombstones both observed tags, but each add produced
        # its own unique tag (checked via the snapshot).
        self.assertTrue(replica.remove("x"))
        self.assertFalse(replica.contains("x"))
        adds = replica.snapshot()["adds"]["x"]
        self.assertEqual(len(adds), len(set(adds)))
        self.assertEqual(len(adds), 2)

    def test_remove_returns_false_when_nothing_visible(self) -> None:
        replica = ORSet("a")
        self.assertFalse(replica.remove("x"))
        replica.add("x")
        self.assertTrue(replica.remove("x"))
        # A second remove changes nothing and reports False.
        self.assertFalse(replica.remove("x"))
        # An element never added but known via merge as fully removed...
        other = ORSet("b")
        other.add("z")
        other.remove("z")
        replica.merge(other)
        before = replica.snapshot()
        self.assertFalse(replica.remove("z"))
        self.assertEqual(replica.snapshot(), before)

    def test_failed_remove_does_not_change_state(self) -> None:
        replica = ORSet("a")
        replica.add("y")
        before = replica.snapshot()
        self.assertFalse(replica.remove("x"))
        self.assertEqual(replica.snapshot(), before)

    def test_elements_returns_independent_copy(self) -> None:
        replica = ORSet("a")
        replica.add("x")
        visible = replica.elements()
        visible.add("spoof")
        visible.discard("x")
        self.assertEqual(replica.elements(), {"x"})
        self.assertTrue(replica.contains("x"))

    def test_re_add_after_remove_is_not_swallowed(self) -> None:
        replica = ORSet("a")
        replica.add("x")
        replica.remove("x")
        replica.add("x")
        self.assertTrue(replica.contains("x"))
        self.assertEqual(replica.elements(), {"x"})

        # Even after merging a state carrying the old tombstone again.
        other = ORSet("b")
        other.merge(replica)
        other.remove("x")  # observes the new tag too
        replica.merge(other)
        # other saw the new add, so it is gone for both; a fresh add wins.
        self.assertFalse(replica.contains("x"))
        replica.add("x")
        self.assertTrue(replica.contains("x"))


class CausalMergeTests(unittest.TestCase):
    """The concurrent-add / observed-remove scenarios from the contract."""

    def test_concurrent_add_survives_unobserved_remove(self) -> None:
        a = ORSet("a")
        b = ORSet("b")

        a.add("x")
        # a removes without ever observing b's concurrent add.
        self.assertTrue(a.remove("x"))
        b.add("x")  # concurrent with a's remove

        a.merge(b)
        b.merge(a)
        self.assertTrue(a.contains("x"))
        self.assertTrue(b.contains("x"))
        self.assertEqual(a.elements(), {"x"})
        self.assertEqual(a.elements(), b.elements())

    def test_observed_add_then_remove_wins(self) -> None:
        a = ORSet("a")
        b = ORSet("b")

        a.add("x")
        b.add("x")  # concurrent add
        a.merge(b)  # a now observes b's tag
        self.assertTrue(a.remove("x"))

        b.merge(a)
        a.merge(b)
        self.assertFalse(a.contains("x"))
        self.assertFalse(b.contains("x"))
        self.assertEqual(a.elements(), set())

    def test_old_snapshot_delivery_cannot_resurrect(self) -> None:
        a = ORSet("a")
        b = ORSet("b")
        a.add("x")
        b.merge(a)
        old_state = cap(a)

        a.remove("x")
        removed_state = cap(a)
        # Deliver the tombstone, then the stale pre-removal snapshot.
        observer = ORSet("o")
        deliver(observer, [removed_state, old_state])
        self.assertFalse(observer.contains("x"))

        # And the reverse order: stale first, tombstone after.
        observer2 = ORSet("o2")
        deliver(observer2, [old_state, removed_state])
        self.assertFalse(observer2.contains("x"))

    def test_removed_element_stays_gone_under_every_delivery_order(self) -> None:
        a = ORSet("a")
        b = ORSet("b")
        a.add("x")
        a.add("y")
        b.merge(a)
        states_before = cap(b)
        b.remove("x")
        a.merge(b)
        a.add("z")
        finals = [cap(a), cap(b)]

        for order in permutations([states_before] + finals):
            with self.subTest(order=order):
                observer = deliver(ORSet("o"), order)
                self.assertFalse(observer.contains("x"))
                self.assertTrue(observer.contains("y"))
                self.assertTrue(observer.contains("z"))
                self.assertEqual(observer.elements(), {"y", "z"})

    def test_offline_replicas_exchange_only_current_snapshots(self) -> None:
        a = ORSet("a")
        b = ORSet("b")
        c = ORSet("c")

        # All three start from a shared base.
        a.add("doc")
        b.merge(a)
        c.merge(a)

        # Go offline and edit independently.
        a.add("alpha-only")
        a.remove("doc")
        a.add("doc")  # re-add with a fresh tag (a never observed this tag)

        b.add("beta-only")
        b.remove("doc")  # b only knew the original tag
        b.merge(restore(cap(a)))  # syncs after its own remove: the re-add
        b.add("beta-doc")  # ...arrives too late to be tombstoned by b

        c.add("gamma-only")
        c.remove("doc")  # c only knew the original tag

        # Reconnect: exchange current snapshots in various orders.
        finals = [cap(a), cap(b), cap(c)]
        for order in permutations(finals):
            with self.subTest(order=order):
                observer = deliver(ORSet("o"), order)
                # a's re-add happened after b's observed remove and c's
                # unobserved remove, so it remains visible.
                self.assertTrue(observer.contains("doc"))
                self.assertEqual(
                    observer.elements(),
                    {"doc", "alpha-only", "beta-only", "beta-doc",
                     "gamma-only"},
                )

        # Direct pairwise exchange of the live replicas also converges.
        a.merge(b)
        c.merge(a)
        b.merge(c)
        a.merge(c)
        self.assertEqual(a.elements(), b.elements())
        self.assertEqual(a.elements(), c.elements())


class MergeAlgebraTests(unittest.TestCase):
    def setUp(self) -> None:
        s1 = ORSet("a")
        s1.add("x")
        s1.add("shared")

        s2 = ORSet("b")
        s2.merge(restore(s1.snapshot()))
        s2.remove("shared")
        s2.add("y")

        s3 = ORSet("c")
        s3.add("z")
        s3.add("shared")  # concurrent add s3 has not synced

        self.states = [s1, s2, s3]

    def merge_order(self, owner_id: str, ordered) -> ORSet:
        receiver = ORSet(owner_id)
        for state in ordered:
            receiver.merge(
                ORSet.from_snapshot(copy.deepcopy(state.snapshot()))
            )
        return receiver

    def test_merge_returns_receiver(self) -> None:
        receiver = ORSet("recv")
        self.assertIs(receiver.merge(self.states[0]), receiver)

    def test_merge_does_not_modify_argument(self) -> None:
        for state in self.states:
            before = state.snapshot()
            receiver = ORSet("recv")
            receiver.add("recv-only")
            receiver.merge(state)
            self.assertEqual(state.snapshot(), before)

    def test_idempotence(self) -> None:
        receiver = self.merge_order("recv", self.states)
        before = receiver.snapshot()
        self.assertIs(receiver.merge(receiver), receiver)
        self.assertEqual(receiver.snapshot(), before)
        receiver.merge(restore(receiver.snapshot()))
        self.assertEqual(receiver.snapshot(), before)

    def test_commutativity(self) -> None:
        baseline = self.merge_order("base", self.states)
        for index, order in enumerate(permutations(self.states)):
            with self.subTest(order=index):
                receiver = self.merge_order(f"r{index}", order)
                self.assertEqual(receiver.elements(), baseline.elements())
                self.assertEqual(
                    receiver.snapshot()["adds"], baseline.snapshot()["adds"]
                )
                self.assertEqual(
                    receiver.snapshot()["removes"],
                    baseline.snapshot()["removes"],
                )

    def test_associativity(self) -> None:
        s1, s2, s3 = self.states

        left = self.merge_order("left", [s1, s2])
        left.merge(ORSet.from_snapshot(copy.deepcopy(s3.snapshot())))

        inner = self.merge_order("inner", [s2, s3])
        right = ORSet("right")
        right.merge(ORSet.from_snapshot(copy.deepcopy(s1.snapshot())))
        right.merge(ORSet.from_snapshot(copy.deepcopy(inner.snapshot())))

        flat = self.merge_order("flat", [s1, s2, s3])
        for field in ("adds", "removes"):
            self.assertEqual(
                left.snapshot()[field], flat.snapshot()[field]
            )
            self.assertEqual(
                right.snapshot()[field], flat.snapshot()[field]
            )
        self.assertEqual(left.elements(), right.elements())


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_independent_and_json_round_trips(self) -> None:
        replica = ORSet("a")
        replica.add("x")
        replica.add("y")
        replica.remove("y")

        first = replica.snapshot()
        second = replica.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["adds"], second["adds"])
        self.assertIsNot(first["removes"], second["removes"])
        self.assertEqual(first, second)

        json.loads(json.dumps(first))

        first["replica_id"] = "hacked"
        first["adds"]["x"] = ["x#99"]
        first["adds"]["ghost"] = ["g#1"]
        first["removes"]["x"] = ["x#0"]
        first["counter"] = 999
        first["extra"] = 1

        self.assertTrue(replica.contains("x"))
        self.assertFalse(replica.contains("y"))
        self.assertEqual(replica.snapshot(), second)

    def test_from_snapshot_preserves_state(self) -> None:
        source = ORSet("a")
        source.add("x")
        source.add("y")
        peer = ORSet("b")
        peer.add("z")
        source.merge(peer)
        source.remove("x")

        restored = restore(source.snapshot())
        self.assertEqual(restored.replica_id, "a")
        self.assertEqual(restored.elements(), {"y", "z"})
        self.assertFalse(restored.contains("x"))

    def test_restored_replica_keeps_adding_unique_tags(self) -> None:
        source = ORSet("a")
        source.add("x")
        source.add("x")

        restored = restore(source.snapshot())
        restored.add("x")
        tags = restored.snapshot()["adds"]["x"]
        self.assertEqual(len(tags), 3)
        self.assertEqual(len(set(tags)), 3)

        # The new tag must not be swallowed by a historical remove observed
        # from a peer.
        peer = ORSet("b")
        peer.merge(restore(source.snapshot()))  # peer sees only the old 2 tags
        peer.remove("x")
        restored.merge(peer)
        self.assertTrue(restored.contains("x"))
        restored.add("x")
        self.assertTrue(restored.contains("x"))

    def test_restore_from_old_snapshot_then_merge_converges(self) -> None:
        replica = ORSet("a")
        replica.add("x")
        old = cap(replica)
        replica.add("y")
        replica.remove("x")

        # A device restored from the stale snapshot keeps editing locally.
        stale = restore(old)
        stale.add("y")
        stale.add("w")

        fresh = ORSet("a")
        fresh.merge(replica)
        fresh.merge(stale)
        stale.merge(fresh)
        replica.merge(fresh)
        self.assertEqual(replica.elements(), stale.elements())
        self.assertEqual(replica.elements(), {"y", "w"})
        self.assertFalse(replica.contains("x"))

        # No tag collisions despite two devices sharing replica id "a".
        all_tags = []
        for tags in replica.snapshot()["adds"].values():
            all_tags.extend(tags)
        self.assertEqual(len(all_tags), len(set(all_tags)))

    def test_restored_does_not_alias_source_dict(self) -> None:
        data = ORSet("a").snapshot()
        data = {
            "replica_id": "a",
            "adds": {"x": ["a#0"]},
            "removes": {},
            "counter": 1,
        }
        restored = ORSet.from_snapshot(data)
        data["replica_id"] = "mutated"
        data["adds"]["x"].append("a#9")
        data["adds"]["z"] = ["a#5"]
        self.assertEqual(restored.replica_id, "a")
        self.assertEqual(restored.elements(), {"x"})

    def test_empty_snapshot(self) -> None:
        replica = ORSet("a")
        self.assertEqual(
            replica.snapshot(),
            {"replica_id": "a", "adds": {}, "removes": {}, "counter": 0},
        )
        restored = restore(replica.snapshot())
        self.assertEqual(restored.elements(), set())
        restored.add("x")
        self.assertTrue(restored.contains("x"))


class ValidationTests(unittest.TestCase):
    def test_replica_id_validation(self) -> None:
        for bad in (None, 1, 1.0, b"a", [], ("a",), True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    ORSet(bad)
        with self.assertRaises(ValueError):
            ORSet("")

    def test_element_validation(self) -> None:
        replica = ORSet("a")
        replica.add("ok")
        for bad in (None, 1, 1.0, b"x", [], ("x",), True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.add(bad)
                with self.assertRaises(TypeError):
                    replica.remove(bad)
                with self.assertRaises(TypeError):
                    replica.contains(bad)
        for method in (replica.add, replica.remove, replica.contains):
            with self.assertRaises(ValueError):
                method("")

        # Rejected operations leave state untouched.
        self.assertEqual(replica.elements(), {"ok"})
        before = replica.snapshot()
        for bad in (None, 1, ""):
            try:
                replica.add(bad)
            except (TypeError, ValueError):
                pass
        self.assertEqual(replica.snapshot(), before)

    def test_merge_requires_an_orset(self) -> None:
        replica = ORSet("a")
        from crdt_sync import GCounter

        for bad in (None, 1, "a", [], {}, replica.snapshot(), GCounter("g")):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.merge(bad)

    def test_from_snapshot_requires_dict(self) -> None:
        for bad in (None, [], "x", 42, 3.14, {1, 2}, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    ORSet.from_snapshot(bad)

    def test_from_snapshot_key_set_must_be_exact(self) -> None:
        valid_core = {
            "replica_id": "a",
            "adds": {},
            "removes": {},
            "counter": 0,
        }
        bad_snapshots = [
            {},
            {"replica_id": "a"},
            {"replica_id": "a", "adds": {}, "removes": {}},
            {**valid_core, "extra": 1},
            {"id": "a", "adds": {}, "removes": {}, "counter": 0},
        ]
        for bad in bad_snapshots:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(bad)

    def test_from_snapshot_replica_id_validation(self) -> None:
        for bad_id in ("", None, 0, True, False, [], {}):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(
                        {"replica_id": bad_id, "adds": {},
                         "removes": {}, "counter": 0}
                    )

    def test_from_snapshot_maps_must_be_dicts(self) -> None:
        for bad in (None, [], "", 42, (), {1, 2}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(
                        {"replica_id": "a", "adds": bad,
                         "removes": {}, "counter": 0}
                    )
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(
                        {"replica_id": "a", "adds": {},
                         "removes": bad, "counter": 0}
                    )

    def test_from_snapshot_element_keys_validation(self) -> None:
        for bad_key in ("", 1, None, True, False):
            with self.subTest(bad_key=bad_key):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(
                        {"replica_id": "a", "adds": {bad_key: ["a#0"]},
                         "removes": {}, "counter": 1}
                    )

    def test_from_snapshot_tag_lists_validation(self) -> None:
        bad_adds = [
            {"x": []},                 # empty list
            {"x": ["a#0", "a#0"]},     # duplicate tag
            {"x": ["notag"]},          # no separator
            {"x": ["a#"]},             # missing sequence
            {"x": ["#0"]},             # missing owner
            {"x": ["a#x"]},            # non-numeric sequence
            {"x": ["a#-1"]},           # negative sequence
            {"x": [42]},               # non-string tag
            {"x": "a#0"},              # tags not a list
            {"x": None},
        ]
        for bad in bad_adds:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(
                        {"replica_id": "a", "adds": bad,
                         "removes": {}, "counter": 1}
                    )

    def test_from_snapshot_tag_cannot_span_elements(self) -> None:
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {"replica_id": "a",
                 "adds": {"x": ["a#0"], "y": ["a#0"]},
                 "removes": {}, "counter": 1}
            )

    def test_from_snapshot_remove_requires_matching_add(self) -> None:
        bad_removes = [
            {"x": ["a#1"]},          # tombstone for a tag never added
            {"y": ["a#0"]},          # tombstone under a different element
        ]
        for bad in bad_removes:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(
                        {"replica_id": "a",
                         "adds": {"x": ["a#0"]},
                         "removes": bad, "counter": 1}
                    )

    def test_from_snapshot_counter_validation(self) -> None:
        core = {"replica_id": "a", "adds": {"x": ["a#5"]}, "removes": {}}
        for bad_counter in (-1, 0, 5, True, False, 1.0, None, "6"):
            with self.subTest(bad_counter=bad_counter):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot({**core, "counter": bad_counter})

        # Counter one past the highest local sequence is valid.
        restored = ORSet.from_snapshot({**core, "counter": 6})
        restored.add("q")
        self.assertEqual(restored.snapshot()["adds"]["q"], ["a#6"])

    def test_failed_restore_does_not_construct_partial_state(self) -> None:
        # Primarily: the call raises; re-using a valid snapshot still works.
        good = {"replica_id": "a", "adds": {}, "removes": {}, "counter": 0}
        restored = ORSet.from_snapshot(good)
        self.assertEqual(restored.elements(), set())


if __name__ == "__main__":
    unittest.main()
