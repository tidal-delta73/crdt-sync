"""Systematic tests for the public RGA contract.

Coverage:
* RGA ordering: inserts by visible position (including append and front
  inserts), the ``(counter, replica_id)`` sibling tie-break with each
  element's successors following its own branch, and uniqueness of the order
  regardless of dict iteration order;
* observed-remove deletion: tombstones keep identity, replayed/duplicated
  deletes neither raise nor resurrect, an unobserved concurrent insert
  survives, observed inserts are removed, and a fresh insert after a delete
  is never swallowed by the old tombstone;
* convergence under duplicated, reordered, batched, interleaved and stale
  snapshot delivery (including multi-round offline edits and bidirectional
  reconnects), matching both ``values()`` and the full causal state, across a
  JSON wire boundary;
* merge algebra: idempotence (self and equivalent copies), commutativity,
  associativity, receiver return value and argument isolation;
* snapshot / from_snapshot independence, JSON serialization and uniqueness of
  local ids after restore (including a restored backup catching up via merge);
* the documented TypeError / ValueError / IndexError input contract, failure
  atomicity and merge conflict atomicity;
* RGA being importable from the package top level.

Only the public API is used; no private attributes are relied upon.
"""

from __future__ import annotations

import copy
import json
import random
import unittest
from itertools import permutations

from crdt_sync import GCounter, ORSet, RGA


def wire_restore(snapshot: dict) -> RGA:
    """Restore a snapshot after a full JSON round trip (simulated wire)."""
    return RGA.from_snapshot(json.loads(json.dumps(snapshot)))


def cap(replica: RGA) -> dict:
    """A detached JSON-shape copy of a replica's current state."""
    return json.loads(json.dumps(replica.snapshot()))


def deliver(receiver: RGA, snapshots) -> RGA:
    """Merge a sequence of snapshot dicts into ``receiver`` in order."""
    for snapshot in snapshots:
        receiver.merge(wire_restore(snapshot))
    return receiver


def causal_state(snapshot: dict) -> tuple:
    """The replica-independent causal state of a snapshot."""
    return (
        json.dumps(snapshot["nodes"], sort_keys=True),
        json.dumps(snapshot["tombstones"], sort_keys=True),
    )


def assert_order_independent(testcase: unittest.TestCase, replica: RGA) -> None:
    """The canonical order must not depend on node-map insertion order."""
    reference = replica.values()
    rebuilt = wire_restore(replica.snapshot())
    # Re-feeding through a dict whose keys arrive in different orders cannot
    # change the tree, but assert the values and state agree all the same.
    testcase.assertEqual(rebuilt.values(), reference)
    testcase.assertEqual(
        causal_state(rebuilt.snapshot()), causal_state(replica.snapshot())
    )


class BasicEditingTests(unittest.TestCase):
    def test_new_instance_is_empty(self) -> None:
        replica = RGA("r")
        self.assertEqual(replica.replica_id, "r")
        self.assertEqual(replica.values(), [])

    def test_insert_by_visible_position_and_append(self) -> None:
        replica = RGA("r")
        replica.insert(0, "a")
        replica.insert(1, "b")   # append
        replica.insert(1, "m")
        replica.insert(0, "z")   # front
        self.assertEqual(replica.values(), ["z", "a", "m", "b"])

    def test_insert_empty_string_is_allowed(self) -> None:
        replica = RGA("r")
        replica.insert(0, "")
        replica.insert(0, "x")
        self.assertEqual(replica.values(), ["x", ""])

    def test_values_returns_an_independent_list(self) -> None:
        replica = RGA("r")
        replica.insert(0, "a")
        replica.insert(1, "b")
        view = replica.values()
        view.append("ghost")
        view[0] = "mutated"
        view.clear()
        self.assertEqual(replica.values(), ["a", "b"])
        second = replica.values()
        self.assertIsNot(second, view)
        self.assertEqual(second, ["a", "b"])

    def test_delete_returns_the_removed_value(self) -> None:
        replica = RGA("r")
        replica.insert(0, "a")
        replica.insert(1, "b")
        replica.insert(2, "c")
        self.assertEqual(replica.delete(1), "b")
        self.assertEqual(replica.values(), ["a", "c"])
        self.assertEqual(replica.delete(0), "a")
        self.assertEqual(replica.values(), ["c"])
        self.assertEqual(replica.delete(0), "c")
        self.assertEqual(replica.values(), [])

    def test_delete_retains_identity_so_positions_shift_once(self) -> None:
        replica = RGA("r")
        for letter in "abcd":
            replica.insert(len(replica.values()), letter)
        replica.delete(1)  # b gone: a c d
        replica.delete(1)  # now removes c, not a phantom b
        self.assertEqual(replica.values(), ["a", "d"])


class OrderingSemanticsTests(unittest.TestCase):
    """The RGA sibling tie-break and subtree placement."""

    def test_equal_counter_siblings_ordered_by_replica_id_desc(self) -> None:
        # Two fresh replicas independently insert at the front: both nodes
        # are root children with counter 1; "B" > "A" in Unicode order.
        alpha = RGA("A")
        beta = RGA("B")
        alpha.insert(0, "a1")
        beta.insert(0, "b1")
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(alpha.values(), ["b1", "a1"])
        self.assertEqual(beta.values(), ["b1", "a1"])

    def test_counter_takes_precedence_over_replica_id(self) -> None:
        # A node with counter 2 sorts before a counter-1 sibling even when
        # the latter's replica id is greater. Alpha spends its first counter
        # on a node that is then deleted, so its surviving node is (A,2);
        # beta's competing node is (B,1); both hang after the shared root.
        shared = RGA("S")
        shared.insert(0, "root")          # (S,1)
        alpha = RGA("A")
        alpha.merge(wire_restore(shared.snapshot()))
        beta = RGA("B")
        beta.merge(wire_restore(shared.snapshot()))
        alpha.insert(1, "old")            # (A,1) after the root
        alpha.delete(1)
        beta.merge(wire_restore(alpha.snapshot()))
        alpha.insert(1, "a1")             # (A,2) after the root
        beta.insert(1, "b1")              # (B,1) after the root
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(alpha.values(), ["root", "a1", "b1"])
        self.assertEqual(beta.values(), alpha.values())

    def test_each_elements_successors_follow_its_branch(self) -> None:
        # Both replicas independently insert at the front (root children).
        # Each then appends a successor to its own head node. On merge, a
        # head must be immediately followed by its own successor subtree;
        # neither successor may float away to its global id rank.
        alpha = RGA("A")
        beta = RGA("B")
        alpha.insert(0, "a-head")          # (A,1), root child
        beta.insert(0, "b-head")           # (B,1), root child
        alpha.insert(1, "a-tail")          # successor of a-head
        beta.insert(1, "b-child")          # successor of b-head
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))
        # B>A puts b-head first; each successor stays glued to its branch.
        # (A naive global sort-by-id would instead put both counter-2
        # successors before the heads, so this pins the subtree semantics.)
        self.assertEqual(
            alpha.values(), ["b-head", "b-child", "a-head", "a-tail"]
        )
        self.assertEqual(beta.values(), alpha.values())
        assert_order_independent(self, alpha)

    def test_order_is_unique_for_the_same_operation_set(self) -> None:
        # Build the same operation set on receivers through every permutation
        # of the contributing states and assert one identical sequence.
        states = self._build_concurrent_states()
        orders = set()
        for permutation in permutations(states):
            receiver = deliver(RGA("observer"), permutation)
            orders.add(tuple(receiver.values()))
        self.assertEqual(len(orders), 1)

    def _build_concurrent_states(self) -> list[dict]:
        alpha = RGA("A")
        beta = RGA("B")
        gamma = RGA("C")
        alpha.insert(0, "x")
        sync = wire_restore(alpha.snapshot())
        beta.merge(sync)
        gamma.merge(sync)
        # Concurrent inserts at every visible position on three replicas.
        alpha.insert(0, "a0")
        alpha.insert(2, "a2")
        beta.insert(1, "b1")
        gamma.insert(0, "c0")
        gamma.insert(len(gamma.values()), "cend")
        return [cap(alpha), cap(beta), cap(gamma)]


class ObservedRemoveSemanticsTests(unittest.TestCase):
    def test_unobserved_concurrent_insert_survives_delete(self) -> None:
        # alpha inserts p0; beta observes it; beta then inserts q1 while
        # alpha (offline) deletes the p0 it knows. q1 must survive.
        alpha = RGA("A")
        beta = RGA("B")
        alpha.insert(0, "p0")
        beta.merge(wire_restore(alpha.snapshot()))
        beta.insert(1, "q1")
        alpha.delete(0)
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(alpha.values(), ["q1"])
        self.assertEqual(beta.values(), ["q1"])

    def test_observed_insert_is_removed_after_exchange(self) -> None:
        alpha = RGA("A")
        beta = RGA("B")
        alpha.insert(0, "p0")
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(beta.delete(0), "p0")  # beta observed p0
        alpha.merge(wire_restore(beta.snapshot()))
        self.assertEqual(alpha.values(), [])
        self.assertEqual(beta.values(), [])

    def test_duplicate_delete_delivery_does_not_raise_or_resurrect(self) -> None:
        alpha = RGA("A")
        alpha.insert(0, "a")
        alpha.insert(1, "b")
        alpha.delete(0)
        deleted = wire_restore(alpha.snapshot())  # carries the tombstone

        beta = RGA("B")
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(beta.values(), ["b"])
        # Re-deliver the tombstone-bearing state many times: still one "b".
        for _ in range(5):
            beta.merge(wire_restore(cap(deleted)))
        self.assertEqual(beta.values(), ["b"])

    def test_stale_predating_snapshot_cannot_resurrect(self) -> None:
        alpha = RGA("A")
        alpha.insert(0, "a")
        before_delete = wire_restore(alpha.snapshot())
        alpha.delete(0)
        # Replay the pre-delete state after the deletion converged.
        alpha.merge(before_delete)
        self.assertEqual(alpha.values(), [])

    def test_fresh_insert_after_delete_is_not_swallowed(self) -> None:
        alpha = RGA("A")
        alpha.insert(0, "a")
        alpha.delete(0)
        alpha.insert(0, "c")
        self.assertEqual(alpha.values(), ["c"])

        # A peer that only knew the original node then merges the fresh state:
        # the historical tombstone refers to (A,1) only; (A,2) stays visible.
        beta = RGA("B")
        first_only = RGA("A")
        first_only.insert(0, "a")
        beta.merge(first_only)
        beta.merge(wire_restore(alpha.snapshot()))
        alpha.merge(wire_restore(beta.snapshot()))
        self.assertEqual(beta.values(), ["c"])
        self.assertEqual(alpha.values(), ["c"])

    def test_concurrent_inserts_on_both_sides_of_a_deleted_node(self) -> None:
        alpha = RGA("A")
        beta = RGA("B")
        alpha.insert(0, "m")
        beta.merge(wire_restore(alpha.snapshot()))
        beta.delete(0)                       # removes observed m
        alpha.insert(1, "after")            # successor of m, unseen by beta
        alpha.insert(0, "before")           # root child, unseen by beta
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))
        # m is tombstoned; its successor subtree ("after") survives, as does
        # the independent root child ("before").
        self.assertEqual(set(alpha.values()), {"before", "after"})
        self.assertEqual(alpha.values(), beta.values())


class ConvergenceTests(unittest.TestCase):
    """Duplicate, out-of-order, batched, interleaved and stale delivery."""

    def setUp(self) -> None:
        self.history, self.finals, self.expected = build_scenario()

    def assertConverged(self, replica: RGA) -> None:
        self.assertEqual(replica.values(), self.expected)
        again = wire_restore(replica.snapshot())
        self.assertEqual(again.values(), self.expected)

    def test_all_permutations_of_final_states_converge(self) -> None:
        for index, order in enumerate(permutations(self.finals)):
            with self.subTest(order=index):
                receiver = deliver(RGA(f"observer-{index}"), order)
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
                receiver = deliver(RGA("observer"), path)
                self.assertConverged(receiver)

    def test_interleaved_and_stale_states_converge(self) -> None:
        h = self.history
        paths = [
            h,
            list(reversed(h)),
            [h[0], h[-1], h[1], h[-2], h[2], h[3]],
            self.finals + h,
            h + self.finals,
        ]
        for index, path in enumerate(paths):
            with self.subTest(path=index):
                receiver = deliver(RGA("observer"), path)
                self.assertConverged(receiver)

    def test_stale_snapshots_cannot_change_converged_state(self) -> None:
        receiver = deliver(RGA("observer"), self.finals)
        before = receiver.snapshot()
        deliver(receiver, self.history * 2)
        self.assertEqual(
            causal_state(receiver.snapshot()), causal_state(before)
        )
        self.assertConverged(receiver)

    def test_multi_round_bidirectional_reconnects_agree(self) -> None:
        fa, fb, fg = self.finals
        first = deliver(RGA("first"), [fa])
        second = deliver(RGA("second"), [fb, fg])
        for _ in range(3):
            first.merge(wire_restore(second.snapshot()))
            second.merge(wire_restore(first.snapshot()))
        self.assertEqual(first.values(), second.values())
        self.assertEqual(
            causal_state(first.snapshot()), causal_state(second.snapshot())
        )
        self.assertConverged(first)
        self.assertConverged(second)

    def test_all_converged_receivers_agree_on_full_causal_state(self) -> None:
        paths = [
            self.finals,
            list(reversed(self.finals)) + [self.finals[1]],
            self.history,
            list(reversed(self.history)),
        ]
        receivers = [
            deliver(RGA(f"node-{index}"), path)
            for index, path in enumerate(paths)
        ]
        reference = causal_state(receivers[0].snapshot())
        for receiver in receivers[1:]:
            self.assertEqual(causal_state(receiver.snapshot()), reference)
        self.assertEqual(
            [r.replica_id for r in receivers],
            ["node-0", "node-1", "node-2", "node-3"],
        )

    def test_random_offline_edits_converge_in_values_and_state(self) -> None:
        replicas = [RGA(name) for name in ("alpha", "beta", "gamma")]
        rng = random.Random(20261007)

        def sync_pair(left: RGA, right: RGA) -> None:
            left.merge(wire_restore(right.snapshot()))
            right.merge(wire_restore(left.snapshot()))

        for step in range(120):
            replica = replicas[rng.randrange(len(replicas))]
            current = replica.values()
            if current and rng.random() < 0.4:
                replica.delete(rng.randrange(len(current)))
            else:
                replica.insert(
                    rng.randrange(len(current) + 1), rng.choice("abcdefg")
                )
            if step % 7 == 6:
                sync_pair(*rng.sample(replicas, 2))

        # Full reconnect, in both directions, several times.
        for _ in range(2):
            for i, left in enumerate(replicas):
                for right in replicas[i + 1:]:
                    sync_pair(left, right)

        values = {tuple(replica.values()) for replica in replicas}
        states = {
            causal_state(replica.snapshot()) for replica in replicas
        }
        self.assertEqual(len(values), 1)
        self.assertEqual(len(states), 1)


def build_scenario() -> tuple[list[dict], list[dict], list[str]]:
    """Return ``(history, finals, expected_values)``.

    Three replicas edit offline, continue from restored snapshots and exchange
    only state snapshots. Concurrent inserts at the same predecessor and
    tombstones of observed nodes both occur.
    """
    alpha = RGA("alpha")
    beta = RGA("beta")
    gamma = RGA("gamma")

    alpha.insert(0, "a")
    history = [cap(alpha)]
    beta.merge(wire_restore(alpha.snapshot()))
    gamma.merge(wire_restore(alpha.snapshot()))

    # alpha continues from a restored snapshot: insert at front, delete "a".
    alpha2 = wire_restore(alpha.snapshot())
    alpha2.insert(0, "A")
    alpha2.delete(alpha2.values().index("a"))
    history.append(cap(alpha2))

    # gamma: concurrent front insert and an append after "a".
    gamma.insert(0, "g")
    gamma.insert(len(gamma.values()), "G")
    history.append(cap(gamma))

    # beta continues from a restore: an observed delete of the shared "a",
    # then a fresh append at the same trailing position.
    beta2 = wire_restore(beta.snapshot())
    beta2.delete(beta2.values().index("a"))
    beta2.insert(len(beta2.values()), "b")
    history.append(cap(beta2))

    finals = [cap(alpha2), cap(beta2), cap(gamma)]

    # Compute the expected order independently, by merging the finals into a
    # fresh observer once; the convergence tests then assert every delivery
    # path agrees with this. The value set (order-independent) is fixed here.
    observer = deliver(RGA("observer"), finals)
    expected = observer.values()
    self_values = set(expected)
    # "a" was observed-deleted by both alpha and beta; everyone converges on
    # its tombstone. Every other inserted value survives.
    assert "a" not in self_values
    assert self_values == {"A", "g", "G", "b"}, self_values
    return history, finals, expected


class MergeAlgebraTests(unittest.TestCase):
    """Idempotence, commutativity, associativity and object semantics."""

    def setUp(self) -> None:
        s1 = RGA("a")
        s1.insert(0, "x")
        s1.insert(1, "y")

        s2 = RGA("b")
        s2.merge(wire_restore(s1.snapshot()))
        s2.insert(0, "z")
        s2.delete(s2.values().index("x"))

        s3 = RGA("c")
        s3.insert(0, "w")
        s3.insert(0, "v")
        s3.delete(0)

        self.states = [s1, s2, s3]

    def merge_order(self, owner_id: str, ordered) -> RGA:
        receiver = RGA(owner_id)
        for replica_state in ordered:
            receiver.merge(
                RGA.from_snapshot(copy.deepcopy(replica_state.snapshot()))
            )
        return receiver

    def test_merge_returns_receiver(self) -> None:
        receiver = RGA("recv")
        self.assertIs(receiver.merge(self.states[0]), receiver)

    def test_merge_does_not_modify_argument(self) -> None:
        for replica_state in self.states:
            before = replica_state.snapshot()
            receiver = RGA("recv")
            receiver.merge(replica_state)
            self.assertEqual(replica_state.snapshot(), before)

    def test_idempotent_merge_with_self(self) -> None:
        receiver = self.merge_order("recv", self.states)
        before = receiver.snapshot()
        result = receiver.merge(receiver)
        self.assertIs(result, receiver)
        self.assertEqual(
            causal_state(receiver.snapshot()), causal_state(before)
        )

    def test_idempotent_merge_with_equivalent_copy(self) -> None:
        receiver = self.merge_order("recv", self.states)
        before = receiver.snapshot()
        duplicate = wire_restore(receiver.snapshot())
        receiver.merge(duplicate)
        receiver.merge(wire_restore(receiver.snapshot()))
        self.assertEqual(
            causal_state(receiver.snapshot()), causal_state(before)
        )
        self.assertEqual(
            causal_state(duplicate.snapshot()), causal_state(before)
        )

    def test_commutative_orderings(self) -> None:
        baseline = self.merge_order("baseline", self.states)
        for index, order in enumerate(permutations(self.states)):
            with self.subTest(order=index):
                receiver = self.merge_order(f"recv-{index}", order)
                self.assertEqual(
                    causal_state(receiver.snapshot()),
                    causal_state(baseline.snapshot()),
                )
                self.assertEqual(receiver.values(), baseline.values())

    def test_associative_groupings(self) -> None:
        s1, s2, s3 = self.states
        left = self.merge_order("left", [s1, s2])
        left.merge(RGA.from_snapshot(copy.deepcopy(s3.snapshot())))

        right_inner = self.merge_order("inner", [s2, s3])
        right = RGA("right")
        right.merge(RGA.from_snapshot(copy.deepcopy(s1.snapshot())))
        right.merge(
            RGA.from_snapshot(copy.deepcopy(right_inner.snapshot()))
        )

        flat = self.merge_order("flat", [s1, s2, s3])
        self.assertEqual(
            causal_state(left.snapshot()), causal_state(flat.snapshot())
        )
        self.assertEqual(
            causal_state(right.snapshot()), causal_state(flat.snapshot())
        )
        self.assertEqual(left.values(), right.values())


class SnapshotIsolationTests(unittest.TestCase):
    def test_snapshots_are_independent_and_json_serializable(self) -> None:
        replica = RGA("alpha")
        replica.insert(0, "a")
        replica.insert(0, "b")
        replica.delete(0)

        first = replica.snapshot()
        second = replica.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["nodes"], second["nodes"])
        self.assertIsNot(first["tombstones"], second["tombstones"])
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertEqual(first, second)

        first["replica_id"] = "hacked"
        first["counter"] = 999
        first["nodes"].append(
            {"id": ["ghost", 1], "value": "g", "predecessor": None}
        )
        first["tombstones"].append(["alpha", 1])
        first["unexpected"] = True
        self.assertEqual(replica.values(), ["a"])
        self.assertEqual(replica.snapshot(), second)

    def test_from_snapshot_preserves_full_state(self) -> None:
        source = RGA("alpha")
        source.insert(0, "a")
        source.insert(1, "b")
        other = RGA("beta")
        other.insert(0, "c")
        source.merge(other)
        source.delete(0)

        data = json.loads(json.dumps(source.snapshot()))
        restored = RGA.from_snapshot(data)
        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.values(), source.values())
        self.assertEqual(restored.snapshot(), source.snapshot())

    def test_restored_replica_keeps_emitting_unique_ids(self) -> None:
        source = RGA("alpha")
        source.insert(0, "a")
        source.insert(1, "b")
        source.delete(0)
        restored = wire_restore(source.snapshot())
        restored.insert(0, "a")  # must be (alpha,3), not (alpha,1)
        self.assertEqual(restored.values(), ["a", "b"])

        other = RGA("beta")
        other.merge(source)
        other.merge(wire_restore(restored.snapshot()))
        restored.merge(wire_restore(other.snapshot()))
        self.assertEqual(other.values(), ["a", "b"])
        self.assertEqual(restored.values(), ["a", "b"])

    def test_restored_backup_catches_up_counter_via_merge(self) -> None:
        live = RGA("alpha")
        live.insert(0, "a")
        backup = wire_restore(live.snapshot())  # counter frozen at 1
        live.insert(1, "b")
        live.insert(2, "c")
        backup.merge(wire_restore(live.snapshot()))
        backup.insert(3, "d")  # counter must have advanced to 3 -> (alpha,4)

        everyone = RGA("observer")
        everyone.merge(live)
        everyone.merge(backup)
        self.assertEqual(everyone.values(), ["a", "b", "c", "d"])
        origins = [
            node["id"]
            for node in everyone.snapshot()["nodes"]
            if node["id"][0] == "alpha"
        ]
        self.assertEqual(sorted(seq for _, seq in origins), [1, 2, 3, 4])

    def test_empty_replica_snapshot_round_trips(self) -> None:
        replica = RGA("zero")
        self.assertEqual(
            replica.snapshot(),
            {
                "replica_id": "zero",
                "counter": 0,
                "nodes": [],
                "tombstones": [],
            },
        )
        restored = wire_restore(replica.snapshot())
        self.assertEqual(restored.values(), [])
        restored.insert(0, "later")
        self.assertEqual(restored.values(), ["later"])


class ValidationTests(unittest.TestCase):
    """The current TypeError / ValueError / IndexError contract."""

    BAD_STRINGS = (None, 1, 1.0, b"a", [], ("a",), True, False, 0)
    BAD_INDICES = (1.0, "0", None, [0], (0,), True, False)

    def assert_unchanged(self, replica: RGA, values) -> None:
        self.assertEqual(replica.values(), values)
        json.dumps(replica.snapshot())

    def test_replica_id_validation(self) -> None:
        for bad in self.BAD_STRINGS:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    RGA(bad)
        with self.assertRaises(ValueError):
            RGA("")

    def test_insert_type_validation(self) -> None:
        replica = RGA("alpha")
        replica.insert(0, "ok")
        for bad in self.BAD_INDICES:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.insert(bad, "v")
        for bad in self.BAD_STRINGS:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.insert(0, bad)
        self.assert_unchanged(replica, ["ok"])

    def test_insert_range_validation(self) -> None:
        replica = RGA("alpha")
        # Empty sequence: only position 0 is valid.
        for bad in (-1, 1, 2):
            with self.subTest(bad=bad):
                with self.assertRaises(IndexError):
                    replica.insert(bad, "v")
        replica.insert(0, "ok")
        # Length 1: positions 0 and 1 valid, others not.
        for bad in (-1, 2, 3):
            with self.subTest(bad=bad):
                with self.assertRaises(IndexError):
                    replica.insert(bad, "v")
        self.assert_unchanged(replica, ["ok"])

    def test_failed_insert_leaves_state_untouched(self) -> None:
        replica = RGA("alpha")
        replica.insert(0, "a")
        replica.insert(1, "b")
        before = replica.snapshot()
        with self.assertRaises(IndexError):
            replica.insert(5, "x")
        with self.assertRaises(TypeError):
            replica.insert(0, 4)
        with self.assertRaises(TypeError):
            replica.insert(True, "x")
        self.assertEqual(replica.snapshot(), before)
        self.assertEqual(replica.values(), ["a", "b"])

    def test_delete_type_validation(self) -> None:
        replica = RGA("alpha")
        replica.insert(0, "ok")
        for bad in self.BAD_INDICES:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.delete(bad)
        self.assert_unchanged(replica, ["ok"])

    def test_delete_range_validation(self) -> None:
        empty = RGA("alpha")
        for bad in (-1, 0, 1):
            with self.subTest(bad=bad):
                with self.assertRaises(IndexError):
                    empty.delete(bad)
        replica = RGA("alpha")
        replica.insert(0, "ok")
        for bad in (-1, 1, 2):
            with self.subTest(bad=bad):
                with self.assertRaises(IndexError):
                    replica.delete(bad)
        self.assert_unchanged(replica, ["ok"])

    def test_merge_requires_an_rga(self) -> None:
        replica = RGA("alpha")
        for bad in (
            None,
            1,
            "alpha",
            [],
            {},
            replica.snapshot(),
            GCounter("alpha"),
            ORSet("alpha"),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.merge(bad)
        self.assertEqual(replica.values(), [])

    def test_conflicting_value_merge_raises_and_leaves_both_untouched(self):
        left = RGA("shared")
        left.insert(0, "same")  # (shared,1)
        right = RGA("shared")
        right.insert(0, "different")  # colliding id (shared,1)
        before_left = left.snapshot()
        before_right = right.snapshot()
        with self.assertRaises(ValueError):
            left.merge(right)
        self.assertEqual(left.snapshot(), before_left)
        self.assertEqual(right.snapshot(), before_right)
        self.assertEqual(left.values(), ["same"])
        self.assertEqual(right.values(), ["different"])

    def test_conflicting_predecessor_merge_raises_and_leaves_both_untouched(
        self,
    ) -> None:
        left = RGA.from_snapshot(
            {
                "replica_id": "shared",
                "counter": 2,
                "nodes": [
                    {"id": ["shared", 1], "value": "z", "predecessor": None},
                    {
                        "id": ["shared", 2],
                        "value": "w",
                        "predecessor": ["shared", 1],
                    },
                ],
                "tombstones": [],
            }
        )
        right = RGA.from_snapshot(
            {
                "replica_id": "shared",
                "counter": 2,
                "nodes": [
                    {
                        "id": ["shared", 1],
                        "value": "z",
                        "predecessor": ["shared", 2],
                    },
                    {"id": ["shared", 2], "value": "w", "predecessor": None},
                ],
                "tombstones": [],
            }
        )
        before_left = left.snapshot()
        with self.assertRaises(ValueError):
            left.merge(right)
        self.assertEqual(left.snapshot(), before_left)
        self.assertEqual(right.values(), ["w", "z"])

    def test_from_snapshot_requires_dict(self) -> None:
        for bad in (None, [], "x", 42, 3.14, {1, 2}, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    RGA.from_snapshot(bad)

    def test_from_snapshot_key_set_must_be_exact(self) -> None:
        good = {
            "replica_id": "a",
            "counter": 0,
            "nodes": [],
            "tombstones": [],
        }
        bad_snapshots = [
            {},
            {"replica_id": "a"},
            {"replica_id": "a", "counter": 0, "nodes": []},
            {**good, "extra": 1},
            {"id": "a", "counter": 0, "nodes": [], "tombstones": []},
            {
                "replica_id": "a",
                "counter": 0,
                "records": [],
                "tombstones": [],
            },
        ]
        for bad in bad_snapshots:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RGA.from_snapshot(bad)

    def test_from_snapshot_field_types(self) -> None:
        base = {
            "replica_id": "a",
            "counter": 0,
            "nodes": [],
            "tombstones": [],
        }
        for bad_id in ("", None, 0, True, False, [], {}):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(ValueError):
                    RGA.from_snapshot({**base, "replica_id": bad_id})
        for bad_counter in (-1, 1.0, "0", None, True, False, []):
            with self.subTest(bad_counter=bad_counter):
                with self.assertRaises(ValueError):
                    RGA.from_snapshot({**base, "counter": bad_counter})
        for bad_nodes in (None, {}, "", 42, (), {1, 2}):
            with self.subTest(bad_nodes=bad_nodes):
                with self.assertRaises(ValueError):
                    RGA.from_snapshot({**base, "nodes": bad_nodes})
        for bad_tombstones in (None, {}, "", 42, (), {1, 2}):
            with self.subTest(bad_tombstones=bad_tombstones):
                with self.assertRaises(ValueError):
                    RGA.from_snapshot({**base, "tombstones": bad_tombstones})

    def test_from_snapshot_node_format(self) -> None:
        base = {"replica_id": "a", "counter": 1, "tombstones": []}
        bad_nodes = [
            [None],
            [[]],
            ["not-a-dict"],
            [{"id": ["a", 1], "value": "x"}],  # missing predecessor
            [{"id": ["a", 1], "value": "x", "predecessor": None,
              "extra": 1}],
            [{"id": ["a", 1], "value": 3, "predecessor": None}],
            [{"id": ["a", 1], "value": None, "predecessor": None}],
            [{"id": ["a"], "value": "x", "predecessor": None}],
            [{"id": ["a", 0], "value": "x", "predecessor": None}],
            [{"id": ["a", -1], "value": "x", "predecessor": None}],
            [{"id": ["a", "1"], "value": "x", "predecessor": None}],
            [{"id": ["a", True], "value": "x", "predecessor": None}],
            [{"id": ["", 1], "value": "x", "predecessor": None}],
            [{"id": [1, 1], "value": "x", "predecessor": None}],
            [
                {"id": ["a", 1], "value": "x", "predecessor": None},
                {"id": ["a", 1], "value": "y", "predecessor": None},
            ],
            [{"id": ["a", 1], "value": "x", "predecessor": ["b"]}],
            [{"id": ["a", 1], "value": "x", "predecessor": ["b", 0]}],
        ]
        for nodes in bad_nodes:
            with self.subTest(nodes=nodes):
                with self.assertRaises(ValueError):
                    RGA.from_snapshot({**base, "nodes": nodes})

    def test_from_snapshot_tombstone_format(self) -> None:
        base = {
            "replica_id": "a",
            "counter": 1,
            "nodes": [
                {"id": ["a", 1], "value": "x", "predecessor": None}
            ],
        }
        bad_tombstones = [
            [["a"]],
            [["a", 1, 2]],
            [["", 1]],
            [[1, 1]],
            [["a", 0]],
            [["a", -2]],
            [["a", "1"]],
            [["a", False]],
            [["a", 1], ["a", 1]],  # duplicate tombstone
            [[]],
        ]
        for tombstones in bad_tombstones:
            with self.subTest(tombstones=tombstones):
                with self.assertRaises(ValueError):
                    RGA.from_snapshot({**base, "tombstones": tombstones})

    def test_from_snapshot_causal_validity(self) -> None:
        def expect_error(snapshot: dict) -> None:
            with self.assertRaises(ValueError):
                RGA.from_snapshot(copy.deepcopy(snapshot))

        node = lambda nid, pred: {  # noqa: E731
            "id": ["a", nid],
            "value": f"v{nid}",
            "predecessor": None if pred is None else ["a", pred],
        }
        # Counter ahead of observed own history.
        expect_error(
            {
                "replica_id": "a",
                "counter": 3,
                "nodes": [node(1, None)],
                "tombstones": [],
            }
        )
        # Counter behind the observed own history.
        expect_error(
            {
                "replica_id": "a",
                "counter": 0,
                "nodes": [node(1, None)],
                "tombstones": [],
            }
        )
        # Gap in own counter sequence.
        expect_error(
            {
                "replica_id": "a",
                "counter": 2,
                "nodes": [node(1, None), node(3, 1)],
                "tombstones": [],
            }
        )
        # Dangling predecessor (foreign origin, absent from nodes).
        expect_error(
            {
                "replica_id": "a",
                "counter": 1,
                "nodes": [node(1, 9)],
                "tombstones": [],
            }
        )
        # Predecessor cycle among otherwise well-formed nodes.
        expect_error(
            {
                "replica_id": "a",
                "counter": 2,
                "nodes": [node(1, 2), node(2, 1)],
                "tombstones": [],
            }
        )
        # Self-loop.
        expect_error(
            {
                "replica_id": "a",
                "counter": 1,
                "nodes": [node(1, 1)],
                "tombstones": [],
            }
        )
        # Tombstone naming an unobserved node.
        expect_error(
            {
                "replica_id": "a",
                "counter": 1,
                "nodes": [node(1, None)],
                "tombstones": [["b", 9]],
            }
        )
        # Foreign-origin own history must not count against the local counter.
        expect_error(
            {
                "replica_id": "a",
                "counter": 1,
                "nodes": [
                    {"id": ["b", 1], "value": "x", "predecessor": None}
                ],
                "tombstones": [],
            }
        )

    def test_from_snapshot_accepts_valid_shapes(self) -> None:
        valid = [
            {"replica_id": "a", "counter": 0, "nodes": [], "tombstones": []},
            {
                "replica_id": "a",
                "counter": 1,
                "nodes": [
                    {"id": ["a", 1], "value": "x", "predecessor": None}
                ],
                "tombstones": [["a", 1]],
            },
            {
                "replica_id": "a",
                "counter": 1,
                "nodes": [
                    {"id": ["a", 1], "value": "x", "predecessor": None},
                    {"id": ["b", 7], "value": "y", "predecessor": ["a", 1]},
                ],
                "tombstones": [["a", 1]],
            },
        ]
        for snapshot in valid:
            with self.subTest(snapshot=snapshot):
                restored = RGA.from_snapshot(copy.deepcopy(snapshot))
                self.assertEqual(restored.replica_id, snapshot["replica_id"])
                self.assertEqual(restored.snapshot(), snapshot)


class PackageSurfaceTests(unittest.TestCase):
    def test_rga_is_top_level_importable_alongside_the_others(self) -> None:
        import crdt_sync

        self.assertIn("RGA", crdt_sync.__all__)
        self.assertIn("GCounter", crdt_sync.__all__)
        self.assertIn("ORSet", crdt_sync.__all__)
        self.assertIn("LWWRegister", crdt_sync.__all__)
        self.assertIs(crdt_sync.RGA, RGA)

    def test_repr_includes_type_replica_and_values(self) -> None:
        replica = RGA("r")
        replica.insert(0, "a")
        text = repr(replica)
        self.assertIn("RGA", text)
        self.assertIn("r", text)
        self.assertIn("a", text)


if __name__ == "__main__":
    unittest.main()
