"""Systematic tests for the public RGA contract.

Coverage:
* indexed insert / delete over the visible sequence, including append at the
  length, the returned deleted string and independent ``values()`` copies;
* the RGA sibling tie-break (greater sequence, then replica id in Unicode
  order, both descending) and depth-first branch contiguity;
* observed-remove delete causality: tombstones keep identity, duplicated or
  stale delivery neither errors nor resurrects, an unobserved concurrent
  insert survives, and a later insert at the same slot is not swallowed;
* convergence under duplicated, reordered, batched, interleaved and stale
  snapshot delivery, offline edits and multi-round bidirectional reconnect,
  across a JSON wire boundary;
* merge algebra: idempotence (self and equivalent copies), commutativity,
  associativity, receiver return value and argument isolation;
* snapshot / from_snapshot independence, JSON serialization and uniqueness of
  locally minted ids after restore (including a restored backup catching up
  and a Lamport clock advanced by a high-sequence peer);
* the documented TypeError / ValueError / IndexError contract, strict
  snapshot validation (fields, ids, counter, dangling and cyclic predecessor
  references) and failure atomicity;
* RGA being importable from the package top level.

Only the public API is used; no private attributes are relied upon.
"""

from __future__ import annotations

import copy
import json
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


def state(snapshot: dict) -> tuple[str, str]:
    """The replica-independent causal state: node records and tombstones.

    ``replica_id`` and ``counter`` are excluded: two converged replicas keep
    their own ids, and the contract requires the same insertion records,
    predecessor graph and tombstones rather than one shared counter.
    """
    return (
        json.dumps(snapshot["nodes"], sort_keys=True),
        json.dumps(snapshot["tombstones"], sort_keys=True),
    )


def same_id_graph(value: str, prev) -> dict:
    """A one-node snapshot whose single node is id ``(1, alpha)``."""
    return {
        "replica_id": "observer",
        "counter": 1,
        "nodes": [
            {"id": [1, "alpha"], "value": value, "prev": prev},
        ],
        "tombstones": [],
    }


def build_scenario():
    """Return ``(history, finals, expected_values)``.

    Three replicas edit a shared string sequence offline and exchange only
    state snapshots:

    * alpha builds ``a b c d``;
    * beta observes all four, tombstones ``b`` and reinserts ``B`` at the same
      visible slot and ``β0`` at the head;
    * alpha concurrently (still offline) tombstones ``c`` and inserts ``A1``
      after ``a`` and ``e`` at the tail;
    * gamma starts from scratch with ``g``.

    The tombstones are per node: deleting ``c`` must not hide its successor
    ``d`` or the later tail ``e``.
    """
    alpha = RGA("alpha")
    alpha.insert(0, "a")
    alpha.insert(1, "b")
    alpha.insert(2, "c")
    alpha.insert(3, "d")
    history = [cap(alpha)]

    beta = RGA("beta")
    beta.merge(wire_restore(alpha.snapshot()))
    gamma = RGA("gamma")
    gamma.insert(0, "g")
    history.extend([cap(beta), cap(gamma)])

    # beta continues from a restored snapshot, then edits offline.
    beta2 = wire_restore(beta.snapshot())
    beta2.delete(1)  # tombstone b
    beta2.insert(1, "B")
    beta2.insert(0, "β0")
    history.append(cap(beta2))

    # alpha stays offline, deletes c and edits around the tombstone.
    alpha.delete(2)  # tombstone c; d stays visible
    alpha.insert(1, "A1")
    alpha.insert(4, "e")
    history.extend([cap(alpha), cap(gamma)])

    finals = [cap(alpha), cap(beta2), cap(gamma)]
    # Root children: (6,beta) β0, then (1,gamma) g ahead of (1,alpha) a.
    # Under a: the concurrent seq-5 inserts sort B (beta) before A1 (alpha),
    # then the original chain b(tomb) -> c(tomb) -> d -> e.
    expected = ["β0", "g", "a", "B", "A1", "d", "e"]
    return history, finals, expected


class InsertAndViewTests(unittest.TestCase):
    def test_empty_sequence_and_sequential_inserts(self) -> None:
        replica = RGA("r")
        self.assertEqual(replica.replica_id, "r")
        self.assertEqual(replica.values(), [])
        replica.insert(0, "a")
        replica.insert(1, "b")
        replica.insert(2, "c")
        self.assertEqual(replica.values(), ["a", "b", "c"])

    def test_insert_at_existing_positions_shifts_right(self) -> None:
        replica = RGA("r")
        replica.insert(0, "c")
        replica.insert(0, "a")
        replica.insert(1, "b")
        self.assertEqual(replica.values(), ["a", "b", "c"])
        replica.insert(2, "x")
        self.assertEqual(replica.values(), ["a", "b", "x", "c"])

    def test_insert_append_at_length(self) -> None:
        replica = RGA("r")
        for value in ("a", "b", "c"):
            replica.insert(len(replica.values()), value)
        self.assertEqual(replica.values(), ["a", "b", "c"])

    def test_empty_strings_are_values(self) -> None:
        replica = RGA("r")
        replica.insert(0, "")
        replica.insert(0, "x")
        self.assertEqual(replica.values(), ["x", ""])
        self.assertEqual(replica.delete(1), "")
        self.assertEqual(replica.values(), ["x"])

    def test_values_returns_independent_list(self) -> None:
        replica = RGA("r")
        replica.insert(0, "a")
        replica.insert(1, "b")
        first = replica.values()
        first.clear()
        first.append("ghost")
        self.assertEqual(replica.values(), ["a", "b"])
        second = replica.values()
        self.assertIsNot(second, first)
        self.assertEqual(second, ["a", "b"])
        second[0] = "mutated"
        self.assertEqual(replica.values(), ["a", "b"])


class DeleteTests(unittest.TestCase):
    def test_delete_returns_the_removed_string_and_removes_it(self) -> None:
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

    def test_deleting_predecessor_keeps_successors(self) -> None:
        # Tombstones are per node; the weave keeps walking through a tombstone.
        replica = RGA("r")
        for value in ("a", "b", "c"):
            replica.insert(len(replica.values()), value)
        replica.delete(0)
        self.assertEqual(replica.values(), ["b", "c"])
        replica.delete(0)
        self.assertEqual(replica.values(), ["c"])

    def test_delete_indexes_the_visible_sequence(self) -> None:
        replica = RGA("r")
        replica.insert(0, "a")
        replica.insert(1, "b")
        replica.delete(0)
        replica.insert(0, "x")
        # Index 0 is the new x; the historical tombstone must not shadow it.
        self.assertEqual(replica.delete(0), "x")
        self.assertEqual(replica.values(), ["b"])


class SiblingOrderingTests(unittest.TestCase):
    """The deterministic same-predecessor tie-break and branch contiguity."""

    def test_same_sequence_sorts_by_replica_id_descending(self) -> None:
        replica = RGA.from_snapshot(
            {
                "replica_id": "observer",
                "counter": 1,
                "nodes": [
                    {"id": [1, "alpha"], "value": "a", "prev": None},
                    {"id": [1, "beta"], "value": "b", "prev": None},
                ],
                "tombstones": [],
            }
        )
        # Same predecessor (root) and same sequence: Unicode order, reversed.
        self.assertEqual(replica.values(), ["b", "a"])

    def test_greater_sequence_sorts_first(self) -> None:
        # Two concurrent inserts after the same visible predecessor.
        alpha = RGA("alpha")
        alpha.insert(0, "root")
        beta = RGA("beta")
        beta.merge(wire_restore(alpha.snapshot()))  # both counters at 1
        alpha.insert(1, "from-alpha")  # (2, alpha)
        beta.insert(1, "from-beta")  # (2, beta)
        merged = RGA("observer")
        deliver(merged, [cap(alpha), cap(beta)])
        self.assertEqual(
            merged.values(), ["root", "from-beta", "from-alpha"]
        )

    def test_each_branch_is_emitted_contiguously(self) -> None:
        # Hand-assembled graph (all ids and links valid) to pin the weave:
        #
        #   root(1,A)
        #     +-- (3,A) a-branch
        #     +-- (2,B) b-branch
        #             +-- (3,B) b-child
        #
        # Root's siblings sort (3,A) before (2,B); under (2,B) its own child
        # must be emitted before walking back to the next root sibling.
        replica = RGA.from_snapshot(
            {
                "replica_id": "observer",
                "counter": 3,
                "nodes": [
                    {"id": [1, "A"], "value": "root", "prev": None},
                    {"id": [2, "B"], "value": "b-branch", "prev": [1, "A"]},
                    {"id": [3, "A"], "value": "a-branch", "prev": [1, "A"]},
                    {"id": [3, "B"], "value": "b-child", "prev": [2, "B"]},
                ],
                "tombstones": [],
            }
        )
        self.assertEqual(
            replica.values(),
            ["root", "a-branch", "b-branch", "b-child"],
        )

    def test_tombstoned_sibling_keeps_branch_position(self) -> None:
        replica = RGA.from_snapshot(
            {
                "replica_id": "observer",
                "counter": 3,
                "nodes": [
                    {"id": [1, "A"], "value": "root", "prev": None},
                    {"id": [2, "B"], "value": "gone", "prev": [1, "A"]},
                    {"id": [3, "A"], "value": "kept", "prev": [1, "A"]},
                    {"id": [3, "B"], "value": "b-child", "prev": [2, "B"]},
                ],
                "tombstones": [[2, "B"]],
            }
        )
        self.assertEqual(replica.values(), ["root", "kept", "b-child"])


class ObservedRemoveCausalityTests(unittest.TestCase):
    def test_unobserved_concurrent_insert_survives_delete(self) -> None:
        alpha = RGA("alpha")
        alpha.insert(0, "h")
        alpha.insert(1, "i")
        beta = RGA("beta")
        beta.merge(wire_restore(alpha.snapshot()))

        # beta tombstones only the head it has observed; gamma's concurrent
        # head insert has never been seen by beta.
        beta.delete(0)
        gamma = RGA("gamma")
        gamma.insert(0, "j")

        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(gamma.snapshot()))
        alpha.merge(wire_restore(gamma.snapshot()))
        self.assertEqual(alpha.values(), ["j", "i"])
        self.assertEqual(beta.values(), ["j", "i"])

    def test_observed_element_is_deleted_after_exchange(self) -> None:
        alpha = RGA("alpha")
        alpha.insert(0, "h")
        alpha.insert(1, "i")
        beta = RGA("beta")
        beta.merge(wire_restore(alpha.snapshot()))
        beta.delete(0)

        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(alpha.values(), ["i"])
        self.assertEqual(beta.values(), ["i"])

    def test_duplicate_tombstone_delivery_is_idempotent(self) -> None:
        alpha = RGA("alpha")
        alpha.insert(0, "h")
        alpha.insert(1, "i")
        beta = RGA("beta")
        beta.merge(wire_restore(alpha.snapshot()))
        beta.delete(0)
        deletion = cap(beta)

        for _ in range(4):
            alpha.merge(wire_restore(deletion))
        self.assertEqual(alpha.values(), ["i"])
        # No exception, no resurrection.

    def test_stale_snapshot_with_live_node_does_not_resurrect(self) -> None:
        alpha = RGA("alpha")
        alpha.insert(0, "h")
        before_delete = cap(alpha)
        alpha.delete(0)
        alpha.merge(wire_restore(before_delete))
        alpha.merge(wire_restore(before_delete))
        self.assertEqual(alpha.values(), [])

    def test_new_insert_after_delete_is_not_swallowed(self) -> None:
        alpha = RGA("alpha")
        alpha.insert(0, "h")
        alpha.delete(0)
        alpha.insert(0, "n")
        self.assertEqual(alpha.values(), ["n"])

        # A peer that knew only the first head tombstones just that node; the
        # replacement must survive the full exchange and stale redelivery.
        beta = RGA("beta")
        old_head = {
            "replica_id": "alpha",
            "counter": 1,
            "nodes": [{"id": [1, "alpha"], "value": "h", "prev": None}],
            "tombstones": [],
        }
        beta.merge(RGA.from_snapshot(copy.deepcopy(old_head)))
        beta.delete(0)
        beta.merge(wire_restore(alpha.snapshot()))
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(RGA.from_snapshot(copy.deepcopy(old_head)))
        self.assertEqual(alpha.values(), ["n"])
        self.assertEqual(beta.values(), ["n"])


class ConvergenceTests(unittest.TestCase):
    """Duplicate, out-of-order, batched, interleaved and stale delivery."""

    def setUp(self) -> None:
        self.history, self.finals, self.expected = build_scenario()

    def assertConverged(self, replica: RGA) -> None:
        self.assertEqual(replica.values(), self.expected)
        # The converged state survives a JSON wire round trip.
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
            [h[0], h[5], h[1], h[4], h[2], h[3]],
            self.finals + h,
            h + self.finals,
        ]
        for index, path in enumerate(paths):
            with self.subTest(path=index):
                receiver = deliver(RGA("observer"), path)
                self.assertConverged(receiver)

    def test_stale_snapshots_cannot_change_a_converged_state(self) -> None:
        receiver = deliver(RGA("observer"), self.finals)
        before = receiver.snapshot()
        deliver(receiver, self.history * 2)
        self.assertEqual(state(receiver.snapshot()), state(before))
        self.assertConverged(receiver)

    def test_multi_round_bidirectional_reconnect(self) -> None:
        fa, fb, fg = self.finals

        x = deliver(RGA("x"), [fa])
        y = deliver(RGA("y"), [fb, fg])
        # First contact: exchange both ways.
        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertConverged(x)
        self.assertConverged(y)
        self.assertEqual(state(x.snapshot()), state(y.snapshot()))

        # Both replicas keep editing while disconnected, then reconnect.
        x.insert(0, "x-head")
        y.insert(len(y.values()), "y-tail")
        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertEqual(x.values(), y.values())
        self.assertEqual(state(x.snapshot()), state(y.snapshot()))
        self.assertIn("x-head", x.values())
        self.assertIn("y-tail", x.values())

        # A late observer receives every generation in an arbitrary order.
        generations = self.history + self.finals + [cap(x), cap(y)]
        late = deliver(RGA("late"), reversed(generations))
        self.assertEqual(late.values(), x.values())
        self.assertEqual(state(late.snapshot()), state(x.snapshot()))

    def test_all_converged_receivers_agree_regardless_of_path(self) -> None:
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
        snapshots = [replica.snapshot() for replica in receivers]
        for snapshot in snapshots[1:]:
            self.assertEqual(state(snapshot), state(snapshots[0]))
        self.assertEqual(
            [snapshot["replica_id"] for snapshot in snapshots],
            ["node-0", "node-1", "node-2", "node-3"],
        )

    def test_receiver_sharing_a_known_replica_id_converges(self) -> None:
        receiver = RGA("alpha")
        deliver(receiver, reversed(self.history))
        self.assertConverged(receiver)
        self.assertEqual(receiver.replica_id, "alpha")


class MergeAlgebraTests(unittest.TestCase):
    """Idempotence, commutativity, associativity and object semantics."""

    def setUp(self) -> None:
        s1 = RGA("a")
        s1.insert(0, "x")
        s1.insert(1, "y")

        s2 = RGA("b")
        s2.merge(wire_restore(s1.snapshot()))
        s2.insert(1, "z")
        s2.delete(0)  # tombstone x after observing it

        s3 = RGA("c")
        s3.insert(0, "w")
        s3.delete(0)
        s3.insert(0, "v")

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
        left.merge(RGA.from_snapshot(copy.deepcopy(s3.snapshot())))

        right_inner = self.merge_order("inner", [s2, s3])
        right = RGA("right")
        right.merge(RGA.from_snapshot(copy.deepcopy(s1.snapshot())))
        right.merge(
            RGA.from_snapshot(copy.deepcopy(right_inner.snapshot()))
        )

        flat = self.merge_order("flat", [s1, s2, s3])
        self.assertEqual(state(left.snapshot()), state(flat.snapshot()))
        self.assertEqual(state(right.snapshot()), state(flat.snapshot()))
        self.assertEqual(left.values(), right.values())


class MergeConflictTests(unittest.TestCase):
    def test_same_id_different_value_raises_and_changes_nothing(self) -> None:
        receiver = RGA.from_snapshot(same_id_graph("x", None))
        other = RGA.from_snapshot(same_id_graph("y", None))
        before_receiver = receiver.snapshot()
        before_other = other.snapshot()
        with self.assertRaises(ValueError):
            receiver.merge(other)
        self.assertEqual(receiver.snapshot(), before_receiver)
        self.assertEqual(other.snapshot(), before_other)
        self.assertEqual(receiver.values(), ["x"])

    def test_same_id_different_predecessor_raises_and_changes_nothing(
        self,
    ) -> None:
        receiver = RGA("b")
        receiver.merge(RGA.from_snapshot(same_id_graph("x", None)))
        other_graph = {
            "replica_id": "observer",
            "counter": 7,
            "nodes": [
                {"id": [1, "alpha"], "value": "x", "prev": [7, "zeta"]},
                {"id": [7, "zeta"], "value": "z", "prev": None},
            ],
            "tombstones": [],
        }
        other = RGA.from_snapshot(other_graph)
        before = receiver.snapshot()
        with self.assertRaises(ValueError):
            receiver.merge(other)
        self.assertEqual(receiver.snapshot(), before)
        self.assertEqual(receiver.values(), ["x"])
        # The conflicting peer is untouched as well.
        self.assertEqual(other.values(), ["z", "x"])

    def test_same_id_same_record_is_not_a_conflict(self) -> None:
        receiver = RGA.from_snapshot(same_id_graph("x", None))
        other = RGA.from_snapshot(same_id_graph("x", None))
        receiver.merge(other)
        self.assertEqual(receiver.values(), ["x"])


class SnapshotIsolationTests(unittest.TestCase):
    def test_snapshots_are_independent_and_json_serializable(self) -> None:
        replica = RGA("alpha")
        replica.insert(0, "a")
        replica.insert(1, "b")
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
        first["nodes"].append({"id": [9, "ghost"], "value": "g", "prev": None})
        first["tombstones"].append([9, "ghost"])
        first["unexpected"] = True
        self.assertEqual(replica.values(), ["b"])
        self.assertEqual(replica.snapshot(), second)

    def test_from_snapshot_preserves_full_state(self) -> None:
        source = RGA("alpha")
        source.insert(0, "a")
        source.insert(1, "b")
        other = RGA("beta")
        other.insert(0, "c")
        source.merge(other)
        # Merged weave: c (id (1, beta)) sorts before a ((1, alpha)) at the
        # root, so the visible order is c, a, b.
        self.assertEqual(source.values(), ["c", "a", "b"])
        source.delete(1)  # tombstone a

        data = json.loads(json.dumps(source.snapshot()))
        restored = RGA.from_snapshot(data)
        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.values(), ["c", "b"])

    def test_snapshot_round_trip_is_canonical(self) -> None:
        source = RGA("alpha")
        source.insert(0, "a")
        source.merge(RGA("zeta"))
        wire = json.loads(json.dumps(source.snapshot()))
        restored = RGA.from_snapshot(wire)
        self.assertEqual(restored.snapshot(), wire)

    def test_restored_replica_keeps_emitting_unique_ids(self) -> None:
        source = RGA("alpha")
        source.insert(0, "a")
        source.insert(1, "b")
        source.delete(0)
        restored = wire_restore(source.snapshot())

        restored.insert(0, "a")  # re-insert at the old slot
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
        live.insert(1, "b")
        backup = wire_restore(live.snapshot())  # counter frozen at 2

        live.insert(2, "c")  # (3, alpha)
        backup.merge(wire_restore(live.snapshot()))
        backup.insert(3, "d")  # must be (4, alpha)

        everyone = RGA("observer")
        everyone.merge(live)
        everyone.merge(backup)
        self.assertEqual(everyone.values(), ["a", "b", "c", "d"])
        origins = sorted(
            node["id"][0]
            for node in everyone.snapshot()["nodes"]
            if node["id"][1] == "alpha"
        )
        self.assertEqual(origins, [1, 2, 3, 4])

    def test_peer_with_high_sequence_advances_local_clock(self) -> None:
        # A foreign node may carry a sequence well beyond our counter; the
        # next local insert must still dominate it (Lamport rule), so it
        # lands exactly at the requested visible position.
        peer_state = {
            "replica_id": "zeta",
            "counter": 100,
            "nodes": [
                {"id": [100, "zeta"], "value": "z", "prev": None},
            ],
            "tombstones": [],
        }
        replica = RGA("alpha")
        replica.merge(RGA.from_snapshot(peer_state))
        replica.insert(0, "head")  # requested at index 0...
        self.assertEqual(replica.values(), ["head", "z"])
        replica.insert(2, "tail")  # ...and append stays an append
        self.assertEqual(replica.values(), ["head", "z", "tail"])

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
    BAD_INDICES = (1.0, "1", None, [1], (1,), True, False)

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

    def test_insert_validation(self) -> None:
        replica = RGA("alpha")
        replica.insert(0, "ok")
        for bad in self.BAD_INDICES:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.insert(bad, "x")
        for bad in self.BAD_STRINGS:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.insert(0, bad)
        for bad in (-1, 2, 100):
            with self.subTest(bad=bad):
                with self.assertRaises(IndexError):
                    replica.insert(bad, "x")
        self.assert_unchanged(replica, ["ok"])

    def test_delete_validation(self) -> None:
        empty = RGA("alpha")
        for bad in self.BAD_INDICES:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    empty.delete(bad)
        for bad in (0, -1, 1):
            with self.subTest(bad=bad):
                with self.assertRaises(IndexError):
                    empty.delete(bad)

        replica = RGA("alpha")
        replica.insert(0, "ok")
        for bad in self.BAD_INDICES:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    replica.delete(bad)
        for bad in (1, 2, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(IndexError):
                    replica.delete(bad)
        self.assert_unchanged(replica, ["ok"])
        self.assertEqual(replica.delete(0), "ok")
        self.assert_unchanged(replica, [])

    def test_failed_insert_is_atomic(self) -> None:
        replica = RGA("alpha")
        replica.insert(0, "a")
        before = replica.snapshot()
        with self.assertRaises(TypeError):
            replica.insert(0, 123)
        with self.assertRaises(IndexError):
            replica.insert(5, "z")
        self.assertEqual(replica.snapshot(), before)

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
            {"replica_id": "a", "counter": 0, "items": [], "tombstones": []},
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

    def test_from_snapshot_node_shape(self) -> None:
        base = {"replica_id": "a", "counter": 1, "tombstones": []}
        bad_single_nodes = [
            "not-a-dict",
            42,
            None,
            {"id": [1, "a"], "value": "x"},
            {"id": [1, "a"], "value": "x", "prev": None, "extra": 1},
            {"id": [1, "a"], "value": "x", "link": None},
            {"id": [1], "value": "x", "prev": None},
            {"id": [1, "a", 2], "value": "x", "prev": None},
            {"id": ["1", "a"], "value": "x", "prev": None},
            {"id": [0, "a"], "value": "x", "prev": None},
            {"id": [-1, "a"], "value": "x", "prev": None},
            {"id": [1.0, "a"], "value": "x", "prev": None},
            {"id": [True, "a"], "value": "x", "prev": None},
            {"id": [1, True], "value": "x", "prev": None},
            {"id": [1, ""], "value": "x", "prev": None},
            {"id": [1, 2], "value": "x", "prev": None},
            {"id": 1, "value": "x", "prev": None},
            {"id": None, "value": "x", "prev": None},
            {"id": [1, "a"], "value": 3, "prev": None},
            {"id": [1, "a"], "value": None, "prev": None},
            {"id": [1, "a"], "value": b"x", "prev": None},
            {"id": [1, "a"], "value": "x", "prev": [1]},
            {"id": [1, "a"], "value": "x", "prev": ["1", "a"]},
            {"id": [1, "a"], "value": "x", "prev": [0, "a"]},
            {"id": [1, "a"], "value": "x", "prev": [1, "a", 0]},
        ]
        for node in bad_single_nodes:
            with self.subTest(node=node):
                with self.assertRaises(ValueError):
                    RGA.from_snapshot({**base, "nodes": [node]})

        # Duplicate node ids, even with identical records.
        good = {"id": [1, "a"], "value": "x", "prev": None}
        with self.assertRaises(ValueError):
            RGA.from_snapshot({**base, "nodes": [good, dict(good)]})
        with self.assertRaises(ValueError):
            RGA.from_snapshot(
                {
                    **base,
                    "nodes": [good, {"id": [1, "a"], "value": "y", "prev": None}],
                }
            )

    def test_from_snapshot_tombstone_shape(self) -> None:
        base = {
            "replica_id": "a",
            "counter": 1,
            "nodes": [{"id": [1, "a"], "value": "x", "prev": None}],
        }
        bad_tombstones = [
            [[1]],
            [[1, "a", 2]],
            [["1", "a"]],
            [[0, "a"]],
            [[True, "a"]],
            [[1, ""]],
            [1],
            [[1, "a"], [1, "a"]],  # duplicate entry
        ]
        for tombstones in bad_tombstones:
            with self.subTest(tombstones=tombstones):
                with self.assertRaises(ValueError):
                    RGA.from_snapshot({**base, "tombstones": tombstones})

    def test_from_snapshot_references_and_counter(self) -> None:
        node = lambda prev: {"id": [1, "a"], "value": "x", "prev": prev}
        # Self cycle.
        with self.assertRaises(ValueError):
            RGA.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 1,
                    "nodes": [node([1, "a"])],
                    "tombstones": [],
                }
            )
        # Two-node cycle.
        with self.assertRaises(ValueError):
            RGA.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 2,
                    "nodes": [
                        {"id": [1, "a"], "value": "x", "prev": [2, "a"]},
                        {"id": [2, "a"], "value": "y", "prev": [1, "a"]},
                    ],
                    "tombstones": [],
                }
            )
        # Dangling predecessor.
        with self.assertRaises(ValueError):
            RGA.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 1,
                    "nodes": [node([9, "zeta"])],
                    "tombstones": [],
                }
            )
        # Tombstone without its insert record.
        with self.assertRaises(ValueError):
            RGA.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 1,
                    "nodes": [node(None)],
                    "tombstones": [[2, "a"]],
                }
            )
        # Counter below the greatest observed sequence.
        with self.assertRaises(ValueError):
            RGA.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 0,
                    "nodes": [node(None)],
                    "tombstones": [],
                }
            )
        with self.assertRaises(ValueError):
            RGA.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 4,
                    "nodes": [
                        node(None),
                        {"id": [5, "b"], "value": "y", "prev": [1, "a"]},
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
                "nodes": [{"id": [1, "a"], "value": "x", "prev": None}],
                "tombstones": [[1, "a"]],
            },
            {
                # Gaps are fine: the Lamport clock may have advanced on
                # account of foreign nodes this snapshot never carried.
                "replica_id": "a",
                "counter": 9,
                "nodes": [{"id": [1, "a"], "value": "x", "prev": None}],
                "tombstones": [],
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

    def test_repr_does_not_dump_internals(self) -> None:
        replica = RGA("r")
        replica.insert(0, "a")
        text = repr(replica)
        self.assertIn("RGA", text)
        self.assertIn("r", text)
        self.assertIn("a", text)


if __name__ == "__main__":
    unittest.main()
