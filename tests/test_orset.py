"""Systematic tests for the public ORSet contract.

Coverage:
* observed-remove causality: a remove only tombstones observed adds; a
  concurrent same-name add survives, an observed add does not, and a re-add
  after removal is never swallowed by the historical tombstone;
* convergence under duplicated, reordered, batched, interleaved and stale
  snapshot delivery, including snapshot round trips across a JSON boundary;
* merge algebra: idempotence, commutativity, associativity, receiver return
  value and argument isolation;
* snapshot / from_snapshot independence, JSON serialization and uniqueness
  of local adds after restore (including a restored backup catching up);
* the documented TypeError / ValueError input contract and failure
  atomicity;
* ORSet being importable from the package top level.

Only the public API is used; no private attributes are relied upon.
"""

from __future__ import annotations

import copy
import json
import unittest
from itertools import permutations

from crdt_sync import GCounter, ORSet


def wire_restore(snapshot: dict) -> ORSet:
    """Restore a snapshot after a full JSON round trip (simulated wire)."""
    return ORSet.from_snapshot(json.loads(json.dumps(snapshot)))


def cap(replica: ORSet) -> dict:
    """A detached JSON-shape copy of a replica's current state."""
    return json.loads(json.dumps(replica.snapshot()))


def deliver(receiver: ORSet, snapshots) -> ORSet:
    """Merge a sequence of snapshot dicts into ``receiver`` in order."""
    for snapshot in snapshots:
        receiver.merge(wire_restore(snapshot))
    return receiver


def state(snapshot: dict) -> tuple:
    """The replica-independent causal state (adds + removes) of a snapshot."""
    return (
        json.dumps(snapshot["adds"], sort_keys=True),
        json.dumps(snapshot["removes"], sort_keys=True),
    )


def build_scenario():
    """Return ``(history, finals, expected_elements)``.

    Three replicas edit offline and exchange only state snapshots:
    * alpha adds apple/date and removes its own apple;
    * gamma concurrently adds apple without ever seeing alpha's apple, so
      gamma's apple must survive alpha's remove;
    * beta removes banana and re-adds it; beta and alpha both add date.
    """
    alpha = ORSet("alpha")
    beta = ORSet("beta")
    gamma = ORSet("gamma")

    alpha.add("apple")
    beta.add("banana")
    gamma.add("cherry")
    history = [cap(alpha), cap(beta), cap(gamma)]

    # alpha continues from a restored snapshot, then removes its apple and
    # adds date.
    alpha2 = wire_restore(alpha.snapshot())
    alpha2.add("date")
    alpha2.remove("apple")

    # gamma stays offline and concurrently adds its own apple.
    gamma.add("apple")
    history.extend([cap(alpha2), cap(gamma)])

    # beta continues from a restored snapshot: remove/re-add banana, add date.
    beta2 = wire_restore(beta.snapshot())
    beta2.remove("banana")
    beta2.add("banana")
    beta2.add("date")
    history.append(cap(beta2))

    finals = [cap(alpha2), cap(beta2), cap(gamma)]
    expected = {"apple", "banana", "cherry", "date"}
    return history, finals, expected


class ObservedRemoveSemanticsTests(unittest.TestCase):
    """The causal remove semantics called out in the contract."""

    def test_remove_without_visible_add_returns_false_and_changes_nothing(
        self,
    ) -> None:
        replica = ORSet("b")
        before = replica.snapshot()
        self.assertFalse(replica.remove("x"))
        self.assertEqual(replica.snapshot(), before)

        # An add observed only on a disconnected peer does not count.
        alpha = ORSet("alpha")
        alpha.add("x")
        beta = ORSet("beta")
        before = beta.snapshot()
        self.assertFalse(beta.remove("x"))
        self.assertEqual(beta.snapshot(), before)
        self.assertFalse(beta.contains("x"))

    def test_unobserved_concurrent_add_survives_remove_after_exchange(
        self,
    ) -> None:
        # alpha adds x; beta removes x without having observed that add.
        alpha = ORSet("alpha")
        alpha.add("x")
        beta = ORSet("beta")
        self.assertFalse(beta.remove("x"))

        # Exchange complete states in both directions.
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))

        self.assertEqual(alpha.elements(), {"x"})
        self.assertEqual(beta.elements(), {"x"})
        self.assertTrue(beta.contains("x"))

    def test_observed_add_is_removed_after_exchange(self) -> None:
        # Same setup, but the remover observes the add first.
        alpha = ORSet("alpha")
        alpha.add("x")
        beta = ORSet("beta")
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertTrue(beta.contains("x"))
        self.assertTrue(beta.remove("x"))

        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))

        self.assertEqual(alpha.elements(), set())
        self.assertEqual(beta.elements(), set())
        self.assertFalse(alpha.contains("x"))

    def test_remove_then_readd_is_not_swallowed_by_historical_tombstone(
        self,
    ) -> None:
        alpha = ORSet("alpha")
        alpha.add("x")
        alpha.remove("x")
        self.assertFalse(alpha.contains("x"))
        alpha.add("x")
        self.assertTrue(alpha.contains("x"))

        # A peer that only knew the first add tombstones just that tag; the
        # fresh add on alpha stays visible after the full exchange.
        beta = ORSet("beta")
        first_add_only = ORSet("alpha")
        first_add_only.add("x")
        beta.merge(first_add_only)
        self.assertTrue(beta.remove("x"))

        beta.merge(wire_restore(alpha.snapshot()))
        alpha.merge(wire_restore(beta.snapshot()))
        self.assertEqual(alpha.elements(), {"x"})
        self.assertEqual(beta.elements(), {"x"})

        # Even if the stale first-add snapshot is redelivered afterwards.
        beta.merge(first_add_only)
        self.assertEqual(beta.elements(), {"x"})

    def test_repeated_add_remove_cycles_keep_distinct_tags(self) -> None:
        replica = ORSet("r")
        for visible in (True, False, True, False, True):
            if visible:
                replica.add("e")
                self.assertTrue(replica.contains("e"))
                self.assertTrue(replica.remove("e"))
            else:
                self.assertFalse(replica.remove("e"))
        replica.add("e")
        self.assertEqual(replica.elements(), {"e"})

    def test_remove_is_scoped_to_the_named_element(self) -> None:
        replica = ORSet("r")
        replica.add("a")
        replica.add("b")
        self.assertTrue(replica.remove("a"))
        self.assertEqual(replica.elements(), {"b"})
        self.assertFalse(replica.contains("a"))
        self.assertTrue(replica.contains("b"))


class BasicViewTests(unittest.TestCase):
    def test_contains_and_elements_lifecycle(self) -> None:
        replica = ORSet("r")
        self.assertEqual(replica.elements(), set())
        self.assertFalse(replica.contains("missing"))

        replica.add("a")
        replica.add("a")  # a second observed add of the same element
        replica.add("b")
        self.assertTrue(replica.contains("a"))
        self.assertEqual(replica.elements(), {"a", "b"})

        # One remove tombstones both observed adds of "a".
        self.assertTrue(replica.remove("a"))
        self.assertEqual(replica.elements(), {"b"})

    def test_elements_returns_independent_set(self) -> None:
        replica = ORSet("r")
        replica.add("a")
        replica.add("b")
        view = replica.elements()
        view.clear()
        view.add("ghost")
        self.assertEqual(replica.elements(), {"a", "b"})

        second = replica.elements()
        self.assertIsNot(second, view)
        self.assertEqual(second, {"a", "b"})


class ConvergenceTests(unittest.TestCase):
    """Duplicate, out-of-order, batched, interleaved and stale delivery."""

    def setUp(self) -> None:
        self.history, self.finals, self.expected = build_scenario()

    def assertConverged(self, replica: ORSet) -> None:
        self.assertEqual(replica.elements(), self.expected)
        for element in self.expected:
            self.assertTrue(replica.contains(element))
        # The converged causal state survives a JSON wire round trip.
        again = wire_restore(replica.snapshot())
        self.assertEqual(again.elements(), self.expected)

    def test_all_permutations_of_final_states_converge(self) -> None:
        for index, order in enumerate(permutations(self.finals)):
            with self.subTest(order=index):
                receiver = deliver(ORSet(f"observer-{index}"), order)
                self.assertConverged(receiver)

    def test_duplicated_delivery_converges(self) -> None:
        fa, fb, fg = self.finals
        paths = [
            [fa, fa, fb, fg, fb, fg, fa],
            [fg, fb, fa, fa, fa],
            list(self.finals) * 3,
        ]
        for order in permutations(self.finals):
            paths.append(list(order) + [self.finals[0], self.finals[2]])
        for index, path in enumerate(paths):
            with self.subTest(path=index):
                receiver = deliver(ORSet("observer"), path)
                self.assertConverged(receiver)

    def test_interleaved_and_stale_states_converge(self) -> None:
        h = self.history
        paths = [
            h,  # oldest to newest (stale states first)
            list(reversed(h)),  # newest to oldest (stale states last)
            [h[0], h[5], h[1], h[4], h[2], h[3]],  # fixed interleave
            self.finals + h,  # complete state first, stale states after
            h + self.finals,  # stale states, then repeated finals
        ]
        for index, path in enumerate(paths):
            with self.subTest(path=index):
                receiver = deliver(ORSet("observer"), path)
                self.assertConverged(receiver)

    def test_stale_snapshot_cannot_resurrect_removed_add(self) -> None:
        # Converge, then keep replaying every old snapshot (including ones
        # that predate the removes) — nothing comes back.
        receiver = deliver(ORSet("observer"), self.finals)
        before = receiver.snapshot()
        deliver(receiver, self.history * 2)
        self.assertEqual(state(receiver.snapshot()), state(before))
        self.assertConverged(receiver)

    def test_batched_delivery_between_replicas_converges(self) -> None:
        fa, fb, fg = self.finals

        x = deliver(ORSet("x"), [fa])
        y = deliver(ORSet("y"), [fb, fg])
        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertConverged(x)
        self.assertConverged(y)
        self.assertEqual(state(x.snapshot()), state(y.snapshot()))

        p = deliver(ORSet("p"), [fa, fb])
        q = deliver(ORSet("q"), [fb, fg])
        r = deliver(ORSet("r"), [fg, fa])
        p.merge(wire_restore(q.snapshot()))
        r.merge(wire_restore(p.snapshot()))
        q.merge(wire_restore(r.snapshot()))
        for replica in (p, q, r):
            self.assertConverged(replica)
        self.assertEqual(state(p.snapshot()), state(q.snapshot()))
        self.assertEqual(state(q.snapshot()), state(r.snapshot()))

    def test_redelivery_of_old_or_same_state_changes_nothing(self) -> None:
        receiver = deliver(ORSet("observer"), self.finals)
        before = receiver.snapshot()
        deliver(
            receiver,
            [
                self.history[0],
                self.history[1],
                self.finals[0],
                self.finals[0],
                self.finals[2],
            ],
        )
        self.assertEqual(state(receiver.snapshot()), state(before))
        self.assertEqual(receiver.elements(), self.expected)

    def test_all_converged_receivers_agree_regardless_of_path(self) -> None:
        paths = [
            self.finals,
            list(reversed(self.finals)) + [self.finals[1]],
            self.history,
            list(reversed(self.history)),
        ]
        receivers = [
            deliver(ORSet(f"node-{index}"), path)
            for index, path in enumerate(paths)
        ]
        snapshots = [replica.snapshot() for replica in receivers]
        for snapshot in snapshots[1:]:
            self.assertEqual(state(snapshot), state(snapshots[0]))
        # replica_id is per receiver and must survive merges untouched.
        self.assertEqual(
            [s["replica_id"] for s in snapshots],
            ["node-0", "node-1", "node-2", "node-3"],
        )

    def test_receiver_sharing_a_known_replica_id_converges(self) -> None:
        receiver = ORSet("alpha")
        deliver(receiver, reversed(self.history))
        self.assertConverged(receiver)
        self.assertEqual(receiver.replica_id, "alpha")

    def test_offline_replicas_exchanging_only_finals_agree(self) -> None:
        # Two replicas edit the same elements while fully offline.
        alpha = ORSet("alpha")
        beta = ORSet("beta")
        alpha.add("shared")
        beta.add("shared")
        alpha.add("only-alpha")
        beta.remove("shared")  # removes beta's own tag only
        beta.add("only-beta")
        alpha.add("shared")  # alpha keeps its own two tags

        alpha_snapshot = cap(alpha)
        beta_snapshot = cap(beta)

        first = ORSet("alpha")
        second = ORSet("beta")
        first.merge(wire_restore(beta_snapshot))
        first.merge(wire_restore(alpha_snapshot))
        second.merge(wire_restore(alpha_snapshot))
        second.merge(wire_restore(beta_snapshot))

        # beta's tombstone covers only beta's tag; alpha's tags stay live.
        self.assertEqual(first.elements(), second.elements())
        self.assertEqual(
            first.elements(), {"shared", "only-alpha", "only-beta"}
        )
        self.assertEqual(state(first.snapshot()), state(second.snapshot()))


class MergeAlgebraTests(unittest.TestCase):
    """Idempotence, commutativity, associativity and object semantics."""

    def setUp(self) -> None:
        s1 = ORSet("a")
        s1.add("x")
        s1.add("y")

        s2 = ORSet("b")
        s2.merge(wire_restore(s1.snapshot()))
        s2.add("z")
        s2.remove("x")  # tombstones a's x tag after observing it

        s3 = ORSet("c")
        s3.add("y")
        s3.remove("y")
        s3.add("w")

        self.states = [s1, s2, s3]

    def merge_order(self, owner_id: str, ordered) -> ORSet:
        receiver = ORSet(owner_id)
        for replica_state in ordered:
            receiver.merge(
                ORSet.from_snapshot(copy.deepcopy(replica_state.snapshot()))
            )
        return receiver

    def test_merge_returns_receiver(self) -> None:
        receiver = ORSet("recv")
        self.assertIs(receiver.merge(self.states[0]), receiver)

    def test_merge_does_not_modify_argument(self) -> None:
        for replica_state in self.states:
            before = replica_state.snapshot()
            receiver = ORSet("recv")
            receiver.merge(replica_state)
            self.assertEqual(replica_state.snapshot(), before)

    def test_idempotent_merge_with_self(self) -> None:
        receiver = self.merge_order("recv", self.states)
        before = receiver.snapshot()
        result = receiver.merge(receiver)
        self.assertIs(result, receiver)
        self.assertEqual(state(receiver.snapshot()), state(before))

    def test_idempotent_merge_with_equivalent_copy(self) -> None:
        receiver = self.merge_order("recv", self.states)
        before = receiver.snapshot()
        duplicate = wire_restore(receiver.snapshot())
        receiver.merge(duplicate)
        receiver.merge(wire_restore(receiver.snapshot()))
        self.assertEqual(state(receiver.snapshot()), state(before))
        self.assertEqual(state(duplicate.snapshot()), state(before))

    def test_commutative_orderings(self) -> None:
        baseline = self.merge_order("baseline", self.states)
        for index, order in enumerate(permutations(self.states)):
            with self.subTest(order=index):
                receiver = self.merge_order(f"recv-{index}", order)
                self.assertEqual(
                    state(receiver.snapshot()), state(baseline.snapshot())
                )

    def test_associative_groupings(self) -> None:
        s1, s2, s3 = self.states

        left = self.merge_order("left", [s1, s2])
        left.merge(ORSet.from_snapshot(copy.deepcopy(s3.snapshot())))

        right_inner = self.merge_order("inner", [s2, s3])
        right = ORSet("right")
        right.merge(ORSet.from_snapshot(copy.deepcopy(s1.snapshot())))
        right.merge(
            ORSet.from_snapshot(copy.deepcopy(right_inner.snapshot()))
        )

        flat = self.merge_order("flat", [s1, s2, s3])
        self.assertEqual(state(left.snapshot()), state(flat.snapshot()))
        self.assertEqual(state(right.snapshot()), state(flat.snapshot()))
        self.assertEqual(left.elements(), right.elements())


class SnapshotIsolationTests(unittest.TestCase):
    def test_snapshots_are_independent_and_json_serializable(self) -> None:
        replica = ORSet("alpha")
        replica.add("a")
        replica.add("b")
        replica.remove("a")

        first = replica.snapshot()
        second = replica.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["adds"], second["adds"])
        self.assertIsNot(first["removes"], second["removes"])
        # Standard JSON round trip must preserve the snapshot exactly.
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertEqual(first, second)

        # Mutating a returned snapshot (top level and nested) cannot leak back.
        first["replica_id"] = "hacked"
        first["counter"] = 999
        first["adds"].clear()
        first["removes"].append(["ghost", 1])
        first["unexpected"] = True
        self.assertEqual(replica.elements(), {"b"})
        self.assertEqual(replica.snapshot(), second)

    def test_from_snapshot_preserves_full_state(self) -> None:
        source = ORSet("alpha")
        source.add("a")
        source.add("b")
        other = ORSet("beta")
        other.add("c")
        source.merge(other)
        source.remove("a")

        data = json.loads(json.dumps(source.snapshot()))
        restored = ORSet.from_snapshot(data)

        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.elements(), {"b", "c"})
        self.assertTrue(restored.contains("b"))
        self.assertFalse(restored.contains("a"))

    def test_restored_replica_keeps_emit_unique_tags(self) -> None:
        source = ORSet("alpha")
        source.add("a")
        source.add("b")
        source.remove("a")
        restored = wire_restore(source.snapshot())

        # Continued local adds must not collide with historical tags.
        restored.add("a")  # re-add after the historical remove
        self.assertEqual(restored.elements(), {"a", "b"})

        # Merging the pre-restore history and the re-add converges both ways;
        # the fresh add survives the historical tombstone.
        other = ORSet("beta")
        other.merge(source)
        other.merge(wire_restore(restored.snapshot()))
        restored.merge(wire_restore(other.snapshot()))
        self.assertEqual(other.elements(), {"a", "b"})
        self.assertEqual(restored.elements(), {"a", "b"})

    def test_restored_backup_catches_up_counter_via_merge(self) -> None:
        live = ORSet("alpha")
        live.add("a")
        backup = wire_restore(live.snapshot())  # counter frozen at 1

        live.add("b")
        live.add("c")
        # The stale backup absorbs the newer state of its own replica...
        backup.merge(wire_restore(live.snapshot()))
        # ...and its next add must use a tag beyond every observed own tag.
        backup.add("d")

        everyone = ORSet("observer")
        everyone.merge(live)
        everyone.merge(backup)
        self.assertEqual(everyone.elements(), {"a", "b", "c", "d"})
        # Tag sequences for alpha form 1..4 with no duplicates.
        origins = [
            tag
            for tags in everyone.snapshot()["adds"].values()
            for tag in tags
            if tag[0] == "alpha"
        ]
        self.assertEqual(sorted(seq for _, seq in origins), [1, 2, 3, 4])

    def test_restored_replica_does_not_alias_source_dict(self) -> None:
        data = json.loads(
            json.dumps(
                ORSet.from_snapshot(
                    {
                        "replica_id": "alpha",
                        "counter": 1,
                        "adds": {"a": [["alpha", 1]], "b": [["beta", 1]]},
                        "removes": [],
                    }
                ).snapshot()
            )
        )
        restored = ORSet.from_snapshot(data)
        data["replica_id"] = "mutated"
        data["counter"] = 500
        data["adds"]["a"].append(["alpha", 2])
        data["adds"]["ghost"] = [["x", 1]]
        data["removes"].append(["alpha", 1])

        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.elements(), {"a", "b"})

    def test_empty_replica_snapshot_round_trips(self) -> None:
        replica = ORSet("zero")
        self.assertEqual(
            replica.snapshot(),
            {"replica_id": "zero", "counter": 0, "adds": {}, "removes": []},
        )
        restored = wire_restore(replica.snapshot())
        self.assertEqual(restored.elements(), set())
        restored.add("later")
        self.assertEqual(restored.elements(), {"later"})


class ValidationTests(unittest.TestCase):
    """The current TypeError / ValueError contract must hold."""

    BAD_STRINGS = (None, 1, 1.0, b"a", [], ("a",), True, False, 0)

    def assert_unchanged(self, replica: ORSet, elements) -> None:
        self.assertEqual(replica.elements(), elements)
        # Snapshot stays JSON serializable after the rejected call.
        json.dumps(replica.snapshot())

    def test_replica_id_validation(self) -> None:
        for bad in self.BAD_STRINGS:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    ORSet(bad)
        with self.assertRaises(ValueError):
            ORSet("")

    def test_add_validation(self) -> None:
        replica = ORSet("alpha")
        replica.add("ok")
        for bad in self.BAD_STRINGS:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.add(bad)
        with self.assertRaises(ValueError):
            replica.add("")
        self.assert_unchanged(replica, {"ok"})

    def test_remove_validation(self) -> None:
        replica = ORSet("alpha")
        replica.add("ok")
        for bad in self.BAD_STRINGS:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.remove(bad)
        with self.assertRaises(ValueError):
            replica.remove("")
        self.assert_unchanged(replica, {"ok"})
        self.assertTrue(replica.remove("ok"))
        self.assert_unchanged(replica, set())

    def test_contains_validation(self) -> None:
        replica = ORSet("alpha")
        for bad in self.BAD_STRINGS:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.contains(bad)
        with self.assertRaises(ValueError):
            replica.contains("")

    def test_merge_requires_an_orset(self) -> None:
        replica = ORSet("alpha")
        for bad in (
            None,
            1,
            "alpha",
            [],
            {},
            replica.snapshot(),
            GCounter("alpha"),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.merge(bad)
        self.assertEqual(replica.elements(), set())

    def test_failed_merge_leaves_state_untouched(self) -> None:
        replica = ORSet("alpha")
        replica.add("a")
        before = replica.snapshot()
        with self.assertRaises(TypeError):
            replica.merge(GCounter("alpha"))
        self.assertEqual(replica.snapshot(), before)

    def test_from_snapshot_requires_dict(self) -> None:
        for bad in (None, [], "x", 42, 3.14, {1, 2}, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    ORSet.from_snapshot(bad)

    def test_from_snapshot_key_set_must_be_exact(self) -> None:
        good = {
            "replica_id": "a",
            "counter": 0,
            "adds": {},
            "removes": [],
        }
        bad_snapshots = [
            {},
            {"replica_id": "a"},
            {"replica_id": "a", "counter": 0, "adds": {}},
            {**good, "extra": 1},
            {"id": "a", "counter": 0, "adds": {}, "removes": []},
            {"replica_id": "a", "counter": 0, "state": {}, "removes": []},
        ]
        for bad in bad_snapshots:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(bad)

    def test_from_snapshot_field_types(self) -> None:
        base = {
            "replica_id": "a",
            "counter": 0,
            "adds": {},
            "removes": [],
        }
        for bad_id in ("", None, 0, True, False, [], {}):
            with self.subTest(bad_id=bad_id):
                bad = {**base, "replica_id": bad_id}
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(bad)
        for bad_counter in (-1, 1.0, "0", None, True, False, []):
            with self.subTest(bad_counter=bad_counter):
                bad = {**base, "counter": bad_counter}
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(bad)
        for bad_adds in (None, [], "", 42, (), {1, 2}):
            with self.subTest(bad_adds=bad_adds):
                bad = {**base, "adds": bad_adds}
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(bad)
        for bad_removes in (None, {}, "", 42, (), {1, 2}):
            with self.subTest(bad_removes=bad_removes):
                bad = {**base, "removes": bad_removes}
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot(bad)

    def test_from_snapshot_adds_key_and_tag_format(self) -> None:
        base = {"replica_id": "a", "counter": 1, "removes": []}
        bad_adds = [
            {"": [["a", 1]]},
            {1: [["a", 1]]},
            {None: [["a", 1]]},
            {"x": []},
            {"x": [["a", 1]]},  # counter/tags mismatch handled separately
            {"x": [["a", 1], ["a", 1, 2]]},
            {"x": [["a", 1, 2]]},
            {"x": ["a", 1]},
            {"x": [["a"]]},
            {"x": [["", 1]]},
            {"x": [[1, 1]]},
            {"x": [["a", 0]]},
            {"x": [["a", -1]]},
            {"x": [["a", "1"]]},
            {"x": [["a", 1.0]]},
            {"x": [["a", True]]},
            {"x": [["a", None]]},
            {"x": [["a", 1], ["a", 2]], "y": [["a", 1]]},
        ]
        for adds in bad_adds:
            with self.subTest(adds=adds):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot({**base, "counter": 0, "adds": adds})

    def test_from_snapshot_removes_tag_format(self) -> None:
        base = {
            "replica_id": "a",
            "counter": 1,
            "adds": {"x": [["a", 1]]},
        }
        bad_removes = [
            [["a"]],
            [["a", 1, 2]],
            [["", 1]],
            [[1, 1]],
            [["a", 0]],
            [["a", -2]],
            [["a", "1"]],
            [["a", False]],
            ["a", 1],
            [[]],
        ]
        for removes in bad_removes:
            with self.subTest(removes=removes):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot({**base, "removes": removes})

    def test_from_snapshot_causal_validity(self) -> None:
        # A tombstone without its add is not a reachable state.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 1,
                    "adds": {"x": [["a", 1]]},
                    "removes": [["b", 9]],
                }
            )
        # Counter ahead of the observed own history.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 3,
                    "adds": {"x": [["a", 1]]},
                    "removes": [],
                }
            )
        # Counter behind the observed own history.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 0,
                    "adds": {"x": [["a", 1]]},
                    "removes": [],
                }
            )
        # Gap in own tag sequence.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 2,
                    "adds": {"x": [["a", 1], ["a", 3]]},
                    "removes": [],
                }
            )
        # Foreign tags must not be counted against the local counter.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 1,
                    "adds": {"x": [["b", 1]]},
                    "removes": [],
                }
            )

    def test_from_snapshot_accepts_valid_shapes(self) -> None:
        valid = [
            {"replica_id": "a", "counter": 0, "adds": {}, "removes": []},
            {
                "replica_id": "a",
                "counter": 1,
                "adds": {"x": [["a", 1]]},
                "removes": [["a", 1]],
            },
            {
                "replica_id": "a",
                "counter": 1,
                "adds": {"x": [["a", 1], ["b", 7]]},
                "removes": [["a", 1]],
            },
        ]
        for snapshot in valid:
            with self.subTest(snapshot=snapshot):
                restored = ORSet.from_snapshot(copy.deepcopy(snapshot))
                self.assertEqual(restored.replica_id, snapshot["replica_id"])
                self.assertEqual(restored.snapshot(), snapshot)

    def test_failed_from_snapshot_constructs_nothing_usable(self) -> None:
        # Invalid input must raise rather than return a half-built replica.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 0,
                    "adds": {"x": [["a", 1]]},
                    "removes": [["a", 1]],
                }
            )


class PackageSurfaceTests(unittest.TestCase):
    def test_orset_is_top_level_importable_alongside_gcounter(self) -> None:
        import crdt_sync

        self.assertIn("ORSet", crdt_sync.__all__)
        self.assertIn("GCounter", crdt_sync.__all__)
        self.assertIs(crdt_sync.ORSet, ORSet)
        self.assertIs(crdt_sync.GCounter, GCounter)

    def test_repr_does_not_dump_internal_tags(self) -> None:
        replica = ORSet("r")
        replica.add("a")
        text = repr(replica)
        self.assertIn("ORSet", text)
        self.assertIn("r", text)
        self.assertIn("a", text)


if __name__ == "__main__":
    unittest.main()
