"""Systematic tests for the public LWWRegister contract.

Coverage:
* unassigned lifecycle (has_value / value -> LookupError) and value-copy
  isolation for both assigned input and returned result, including null;
* assign validation: null, booleans, strings, finite numbers, nested
  lists / string-keyed dicts accepted; everything else raises TypeError
  without touching state (clock included);
* last-writer-wins merge semantics: counter first, Unicode replica-id
  ordering second, clock adoption regardless of winner, receiver return
  value, argument isolation and full failure atomicity on an equal
  timestamp / different value conflict (ValueError);
* idempotence, commutativity, associativity under duplicated, reordered,
  batched, bidirectional and interleaved (stale) snapshot delivery;
* local assigns after an observed merge always postdate observed writes;
* snapshot / from_snapshot: exact JSON round trip, exact key set, field
  validation, timestamp-not-above-clock, non-JSON value rejection,
  and no aliasing of input containers;
* LWWRegister being importable from the package top level;
* unchanged ``version`` / ``help`` CLI behavior.

Only the public API is used; no private attributes are relied upon.
"""

from __future__ import annotations

import copy
import io
import json
import math
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import permutations

from crdt_sync import GCounter, LWWRegister, ORSet, __version__
from crdt_sync.__main__ import main


def wire_restore(snapshot: dict) -> LWWRegister:
    """Restore a snapshot after a full JSON round trip (simulated wire)."""
    return LWWRegister.from_snapshot(json.loads(json.dumps(snapshot)))


def cap(register: LWWRegister) -> dict:
    """A detached JSON-shape copy of a register's current state."""
    return json.loads(json.dumps(register.snapshot()))


def deliver(receiver: LWWRegister, snapshots) -> LWWRegister:
    """Merge a sequence of snapshot dicts into ``receiver`` in order."""
    for snapshot in snapshots:
        receiver.merge(wire_restore(snapshot))
    return receiver


def entry_state(snapshot: dict):
    """The replica-independent content (entry) of a snapshot."""
    return copy.deepcopy(snapshot["entry"])


def build_scenario():
    """Return ``(history, finals, expected_entry)``.

    Three replicas write offline, exchange snapshots, and keep writing on
    top of merged state; some replicas continue from restored snapshots.
    The eventual winner must be the write with the largest local counter
    (then the writer id tie-break).
    """
    alpha = LWWRegister("alpha")
    beta = LWWRegister("beta")
    gamma = LWWRegister("gamma")

    alpha.assign({"v": 1, "note": "first"})
    beta.assign("beta-1")
    gamma.assign([1, 2, 3])
    history = [cap(alpha), cap(beta), cap(gamma)]

    # alpha absorbs beta's write (observes clock 1), then writes locally.
    alpha.merge(wire_restore(beta.snapshot()))
    alpha.assign("alpha-2")  # local counter 2 -> beats beta's 1
    gamma.assign(None)  # gamma's own counter 2
    history.extend([cap(alpha), cap(gamma)])

    # beta continues from a fresh instance, catches up through snapshots and
    # keeps writing locally; its clock adopts the observed maximum (2)...
    beta2 = LWWRegister("beta")
    beta2.merge(wire_restore(beta.snapshot()))
    beta2.merge(wire_restore(alpha.snapshot()))
    beta2.assign({"v": 3})  # beta clock 3 (observed 2 + 1)
    beta2.assign("final")   # beta clock 4 -> global winner
    history.append(cap(beta2))

    finals = [cap(alpha), cap(beta2), cap(gamma)]
    expected_entry = {"timestamp": [4, "beta"], "value": "final"}
    return history, finals, expected_entry


class UnassignedLifecycleTests(unittest.TestCase):
    def test_fresh_register_has_no_value(self) -> None:
        register = LWWRegister("r")
        self.assertFalse(register.has_value())
        with self.assertRaises(LookupError):
            register.value()

    def test_snapshot_of_unassigned_register(self) -> None:
        register = LWWRegister("r")
        self.assertEqual(
            register.snapshot(),
            {"replica_id": "r", "clock": 0, "entry": None},
        )
        # JSON round trips cleanly and restores back to "no value".
        restored = wire_restore(register.snapshot())
        self.assertFalse(restored.has_value())
        with self.assertRaises(LookupError):
            restored.value()
        self.assertEqual(restored.snapshot(), register.snapshot())

    def test_replica_id_validation(self) -> None:
        for bad in (None, 1, 1.0, b"a", [], ("a",), True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    LWWRegister(bad)
        with self.assertRaises(ValueError):
            LWWRegister("")

    def test_replica_id_property_survives_merge(self) -> None:
        a = LWWRegister("alpha")
        b = LWWRegister("beta")
        b.assign("x")
        a.merge(b)
        self.assertEqual(a.replica_id, "alpha")
        self.assertEqual(b.replica_id, "beta")


class AssignAndIsolationTests(unittest.TestCase):
    ACCEPTED = [
        None,
        True,
        False,
        "",
        "héllo 世界 🎉",
        0,
        -0,
        1,
        -123,
        10**40,
        0.0,
        1.5,
        -2.25e10,
        [],
        {},
        [None, True, False, 1, 1.5, "s", [], {}],
        {"a": 1, "b": {"c": [True, None, {"d": "e"}]}},
        {"": [0, 0.0, False, None]},
        [["x"], {"y": ["z", [1, 2]]}],
    ]

    REJECTED = [
        float("inf"),
        float("-inf"),
        float("nan"),
        1 + 2j,
        b"bytes",
        (1, 2),
        {1: "x"},
        {None: 1},
        {True: 1},
        {("a",): 1},
        {1, 2},
        object(),
        [1, float("inf")],
        {"a": float("nan")},
        {"a": (1,)},
        [b"x"],
    ]

    def test_accepted_value_shapes_round_trip(self) -> None:
        for value in self.ACCEPTED:
            with self.subTest(value=value):
                register = LWWRegister("r")
                register.assign(copy.deepcopy(value))
                self.assertTrue(register.has_value())
                encoded = json.loads(json.dumps(value))
                self.assertEqual(
                    json.loads(json.dumps(register.value())), encoded
                )
                restored = wire_restore(register.snapshot())
                self.assertEqual(restored.value(), register.value())

    def test_null_is_a_real_value_not_unassigned(self) -> None:
        register = LWWRegister("r")
        register.assign(None)
        self.assertTrue(register.has_value())
        self.assertIsNone(register.value())
        entry = register.snapshot()["entry"]
        self.assertEqual(entry, {"timestamp": [1, "r"], "value": None})

    def test_bool_is_distinct_from_integer(self) -> None:
        register = LWWRegister("r")
        register.assign(True)
        self.assertIs(register.value(), True)
        self.assertEqual(register.snapshot()["entry"]["value"], True)
        register.assign(1)
        self.assertEqual(register.value(), 1)
        self.assertIsNot(register.value(), True)

    def test_rejected_inputs_raise_typeerror(self) -> None:
        for bad in self.REJECTED:
            with self.subTest(bad=bad):
                register = LWWRegister("r")
                register.assign("before")
                register.assign(None)  # populated, clock 2
                with self.assertRaises(TypeError):
                    register.assign(copy.deepcopy(bad))
                # State (including clock) is untouched by the failed assign.
                self.assertTrue(register.has_value())
                self.assertIsNone(register.value())
                self.assertEqual(
                    register.snapshot(),
                    {
                        "replica_id": "r",
                        "clock": 2,
                        "entry": {"timestamp": [2, "r"], "value": None},
                    },
                )

    def test_rejected_input_on_fresh_register_keeps_it_unassigned(self) -> None:
        register = LWWRegister("r")
        with self.assertRaises(TypeError):
            register.assign({"ok": 1, float("nan"): 0})
        with self.assertRaises(TypeError):
            register.assign({"ok": float("inf")})
        with self.assertRaises(TypeError):
            register.assign({1: 2})
        self.assertFalse(register.has_value())
        with self.assertRaises(LookupError):
            register.value()
        self.assertEqual(
            register.snapshot(),
            {"replica_id": "r", "clock": 0, "entry": None},
        )

    def test_mutating_assigned_input_does_not_reach_register(self) -> None:
        register = LWWRegister("r")
        payload = {"a": [1, 2], "b": {"c": True}}
        register.assign(payload)
        payload["a"].append(999)
        payload["b"]["c"] = False
        payload["new"] = "x"
        self.assertEqual(
            register.value(), {"a": [1, 2], "b": {"c": True}}
        )

        listed = [1, [2, 3], {"x": "y"}]
        register.assign(listed)
        listed[1].append(4)
        listed[2]["z"] = 0
        self.assertEqual(register.value(), [1, [2, 3], {"x": "y"}])

    def test_mutating_returned_value_does_not_reach_register(self) -> None:
        register = LWWRegister("r")
        register.assign({"a": [1, 2], "b": {"c": True}})
        first = register.value()
        second = register.value()
        self.assertIsNot(first, second)
        first["a"].append(999)
        first["b"]["new"] = False
        self.assertEqual(
            register.value(), {"a": [1, 2], "b": {"c": True}}
        )
        self.assertEqual(second, {"a": [1, 2], "b": {"c": True}})

        register.assign([[1], {"x": "y"}])
        out = register.value()
        out[0].append(2)
        out[1]["x"] = "mutated"
        self.assertEqual(register.value(), [[1], {"x": "y"}])

    def test_assign_advances_clock_one_per_local_write(self) -> None:
        register = LWWRegister("r")
        register.assign("a")
        self.assertEqual(
            register.snapshot()["entry"]["timestamp"], [1, "r"]
        )
        register.assign(None)
        self.assertEqual(
            register.snapshot()["entry"]["timestamp"], [2, "r"]
        )
        register.assign([1])
        self.assertEqual(
            register.snapshot()["entry"]["timestamp"], [3, "r"]
        )
        self.assertEqual(register.snapshot()["clock"], 3)


class MergeSemanticsTests(unittest.TestCase):
    def test_merge_requires_an_lwwregister(self) -> None:
        register = LWWRegister("alpha")
        register.assign("x")
        for bad in (
            None,
            1,
            "alpha",
            [],
            {},
            register.snapshot(),
            GCounter("alpha"),
            ORSet("alpha"),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    register.merge(bad)
        self.assertEqual(register.value(), "x")

    def test_merge_returns_receiver(self) -> None:
        a = LWWRegister("a")
        b = LWWRegister("b")
        b.assign("x")
        self.assertIs(a.merge(b), a)
        self.assertIs(a.merge(a), a)

    def test_merge_does_not_modify_argument(self) -> None:
        a = LWWRegister("a")
        a.assign("old")
        b = LWWRegister("b")
        b.assign({"k": [1]})
        before = b.snapshot()
        a.merge(b)
        self.assertEqual(b.snapshot(), before)
        self.assertEqual(b.value(), {"k": [1]})

        # The argument keeps its own replica id and logical clock.
        b.assign("after")
        self.assertEqual(b.snapshot()["clock"], 2)
        self.assertEqual(
            b.snapshot()["entry"]["timestamp"], [2, "b"]
        )

    def test_counter_primary_ordering(self) -> None:
        high = LWWRegister("a")  # smaller id, but higher counter
        high.assign("h1")
        high.assign("h2")
        low = LWWRegister("z")  # larger id, but lower counter
        low.assign("l1")

        high.merge(low)
        self.assertEqual(high.value(), "h2")
        low.merge(wire_restore(high.snapshot()))
        self.assertEqual(low.value(), "h2")

    def test_replica_id_tie_break_is_unicode_dictionary_order(self) -> None:
        # Equal counters: the larger Unicode code-point-ordered id wins.
        a = LWWRegister("a")
        z = LWWRegister("z")
        a.assign("from-a")
        z.assign("from-z")
        a.merge(z)
        self.assertEqual(a.value(), "from-z")
        z.merge(wire_restore(a.snapshot()))
        self.assertEqual(z.value(), "from-z")

        receiver = LWWRegister("r")
        receiver.merge(a)
        receiver.merge(z)
        self.assertEqual(receiver.value(), "from-z")

        # Non-ASCII ids order by Unicode code points ("é" > "a", "界" > "a").
        en = LWWRegister("a")
        fr = LWWRegister("é")
        en.assign("en")
        fr.assign("fr")
        en.merge(fr)
        self.assertEqual(en.value(), "fr")
        fr.merge(wire_restore(en.snapshot()))
        self.assertEqual(fr.value(), "fr")

    def test_receiver_adopts_max_clock_even_when_other_loses(self) -> None:
        a = LWWRegister("a")
        a.assign("a1")
        a.assign("a2")  # clock 2, entry (2, a)
        b = LWWRegister("b")
        b.assign("b1")  # clock 1, entry (1, b)

        b.merge(a)
        self.assertEqual(b.value(), "a2")
        self.assertEqual(b.snapshot()["clock"], 2)
        # b's next local write must postdate the observed write.
        b.assign("b2")
        self.assertEqual(
            b.snapshot()["entry"]["timestamp"], [3, "b"]
        )
        self.assertEqual(b.value(), "b2")

        # And when the receiver wins, its clock still tracks the loser's.
        x = LWWRegister("x")
        x.assign("x1")
        x.assign("x2")  # clock 2
        y = LWWRegister("y")
        y.assign("y1")  # clock 1
        before = x.snapshot()
        x.merge(y)  # x keeps its own entry
        self.assertEqual(entry_state(x.snapshot()), entry_state(before))
        self.assertEqual(x.snapshot()["clock"], 2)
        x.assign("x3")
        self.assertEqual(
            x.snapshot()["entry"]["timestamp"], [3, "x"]
        )

    def test_local_write_after_observed_merge_always_wins(self) -> None:
        alpha = LWWRegister("alpha")
        beta = LWWRegister("beta")
        for _ in range(5):
            beta.assign("beta-data")
        alpha.merge(wire_restore(beta.snapshot()))  # observes clock 5
        alpha.assign({"fresh": True})  # local counter 6

        # Every write alpha observed carried a counter <= 5, so redelivering
        # stale beta states (even duplicated/out of order) cannot overturn
        # alpha's write at (6, alpha).
        stale_beta = cap(beta)
        for path in ([stale_beta], [stale_beta, stale_beta]):
            with self.subTest(path=path):
                node = wire_restore(cap(alpha))
                deliver(node, path)
                self.assertEqual(node.value(), {"fresh": True})

        # A genuinely concurrent offline write at the same counter is
        # resolved by the documented replica-id tie-break, identically no
        # matter the delivery order — never by delivery order itself.
        beta.assign("beta-late")  # (6, beta), written without seeing alpha
        snapshot_a = cap(alpha)
        snapshot_b = cap(beta)
        node1 = deliver(LWWRegister("n1"), [snapshot_b, snapshot_a])
        node2 = deliver(LWWRegister("n2"), [snapshot_a, snapshot_b])
        self.assertEqual(
            entry_state(node1.snapshot()), entry_state(node2.snapshot())
        )
        self.assertEqual(node1.value(), "beta-late")  # "beta" > "alpha"

        # Once alpha observes that concurrent write, its clock adopts 6 and
        # its next local write at (7, alpha) deterministically dominates.
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(alpha.value(), beta.value())
        self.assertEqual(alpha.value(), "beta-late")
        alpha.assign({"after-observation": 1})
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(alpha.value(), {"after-observation": 1})
        self.assertEqual(beta.value(), {"after-observation": 1})

    def test_merging_unassigned_registers_changes_nothing(self) -> None:
        empty_a = LWWRegister("a")
        empty_b = LWWRegister("b")
        before = empty_a.snapshot()
        empty_a.merge(empty_b)
        self.assertEqual(empty_a.snapshot(), before)
        self.assertFalse(empty_a.has_value())

        populated = LWWRegister("c")
        populated.assign("v")
        empty_a.merge(populated)
        self.assertEqual(empty_a.value(), "v")
        self.assertEqual(empty_a.snapshot()["clock"], 1)

        # A populated register absorbing an empty one just adopts any larger
        # observed clock (none here); entry untouched.
        snap_before = populated.snapshot()
        populated.merge(LWWRegister("d"))
        self.assertEqual(populated.snapshot(), snap_before)

    def test_equal_timestamp_different_value_conflict(self) -> None:
        # Same replica id on two replicas, same counter, different payloads.
        left = LWWRegister("r")
        right = LWWRegister("r")
        left.assign("left")
        right.assign({"different": True})
        self.assertEqual(
            left.snapshot()["entry"]["timestamp"],
            right.snapshot()["entry"]["timestamp"],
        )

        left_before = left.snapshot()
        right_before = right.snapshot()
        with self.assertRaises(ValueError):
            left.merge(right)
        with self.assertRaises(ValueError):
            right.merge(left)
        # Both states untouched.
        self.assertEqual(left.snapshot(), left_before)
        self.assertEqual(right.snapshot(), right_before)
        self.assertEqual(left.value(), "left")
        self.assertEqual(right.value(), {"different": True})

        # Conflict detection survives the JSON boundary on both sides: two
        # restored states carrying the same timestamp but different values
        # cannot be merged in either direction, and keep their own values.
        left_copy = wire_restore(left.snapshot())
        right_copy = wire_restore(right.snapshot())
        with self.assertRaises(ValueError):
            left_copy.merge(right_copy)
        with self.assertRaises(ValueError):
            right_copy.merge(left_copy)
        self.assertEqual(left_copy.value(), "left")
        self.assertEqual(right_copy.value(), {"different": True})
        # And it is structural, not reference-based: equal payload merges.
        twin_a = LWWRegister("r")
        twin_b = LWWRegister("r")
        twin_a.assign({"k": [1, 2]})
        twin_b.assign({"k": [1, 2]})
        snap = twin_a.snapshot()
        twin_a.merge(twin_b)
        self.assertEqual(twin_a.value(), {"k": [1, 2]})
        self.assertEqual(twin_a.snapshot(), snap)

    def test_structural_equality_distinguishes_types_at_equal_timestamp(
        self,
    ) -> None:
        pairs = [
            (True, 1),
            (False, 0),
            (1, 1.0),
            (None, False),
            (None, 0),
            (0, 0.0),
            ("1", 1),
            ([1], {"0": 1}),
        ]
        for left_value, right_value in pairs:
            with self.subTest(pair=(left_value, right_value)):
                a = LWWRegister("r")
                b = LWWRegister("r")
                a.assign(left_value)
                b.assign(right_value)
                with self.assertRaises(ValueError):
                    a.merge(b)


class ConvergenceTests(unittest.TestCase):
    """Duplicate, out-of-order, batched, interleaved and stale delivery."""

    def setUp(self) -> None:
        self.history, self.finals, self.expected_entry = build_scenario()

    def assertConverged(self, register: LWWRegister) -> None:
        self.assertEqual(
            entry_state(register.snapshot()), self.expected_entry
        )
        # The converged content survives a JSON wire round trip.
        again = wire_restore(register.snapshot())
        self.assertEqual(
            entry_state(again.snapshot()), self.expected_entry
        )
        self.assertEqual(register.value(), "final")

    def test_all_permutations_of_final_states_converge(self) -> None:
        for index, order in enumerate(permutations(self.finals)):
            with self.subTest(order=index):
                receiver = deliver(
                    LWWRegister(f"observer-{index}"), order
                )
                self.assertConverged(receiver)

    def test_duplicated_delivery_converges(self) -> None:
        fa, fb, fg = self.finals
        paths = [
            [fa, fa, fb, fg, fb, fg, fa],
            [fg, fb, fa, fa, fa],
            list(self.finals) * 3,
        ]
        for order in permutations(self.finals):
            paths.append(
                list(order) + [self.finals[0], self.finals[2]]
            )
        for index, path in enumerate(paths):
            with self.subTest(path=index):
                receiver = deliver(LWWRegister("observer"), path)
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
                receiver = deliver(LWWRegister("observer"), path)
                self.assertConverged(receiver)

    def test_batched_and_bidirectional_delivery_converges(self) -> None:
        fa, fb, fg = self.finals

        x = deliver(LWWRegister("x"), [fa])
        y = deliver(LWWRegister("y"), [fb, fg])
        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertConverged(x)
        self.assertConverged(y)
        self.assertEqual(
            entry_state(x.snapshot()), entry_state(y.snapshot())
        )

        p = deliver(LWWRegister("p"), [fa, fb])
        q = deliver(LWWRegister("q"), [fb, fg])
        r = deliver(LWWRegister("r"), [fg, fa])
        p.merge(wire_restore(q.snapshot()))
        r.merge(wire_restore(p.snapshot()))
        q.merge(wire_restore(r.snapshot()))
        for register in (p, q, r):
            self.assertConverged(register)

    def test_redelivery_of_old_state_changes_nothing(self) -> None:
        receiver = deliver(LWWRegister("observer"), self.finals)
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
        self.assertEqual(receiver.snapshot(), before)
        self.assertConverged(receiver)

    def test_all_converged_receivers_agree_regardless_of_path(self) -> None:
        paths = [
            self.finals,
            list(reversed(self.finals)) + [self.finals[1]],
            self.history,
            list(reversed(self.history)),
        ]
        receivers = [
            deliver(LWWRegister(f"node-{index}"), path)
            for index, path in enumerate(paths)
        ]
        entries = [entry_state(r.snapshot()) for r in receivers]
        for entry in entries[1:]:
            self.assertEqual(entry, entries[0])
        self.assertEqual(
            [r.snapshot()["replica_id"] for r in receivers],
            ["node-0", "node-1", "node-2", "node-3"],
        )


class MergeAlgebraTests(unittest.TestCase):
    """Idempotence, commutativity, associativity and object semantics."""

    def setUp(self) -> None:
        s1 = LWWRegister("a")
        s1.assign(["x"])

        s2 = LWWRegister("b")
        s2.merge(wire_restore(s1.snapshot()))
        s2.assign("z")  # clock 2

        s3 = LWWRegister("c")
        s3.merge(wire_restore(s2.snapshot()))
        s3.assign(None)  # clock 3

        self.states = [s1, s2, s3]

    def merge_order(self, owner_id: str, ordered) -> LWWRegister:
        receiver = LWWRegister(owner_id)
        for state in ordered:
            receiver.merge(
                LWWRegister.from_snapshot(copy.deepcopy(state.snapshot()))
            )
        return receiver

    def test_idempotent_merge_with_self(self) -> None:
        receiver = self.merge_order("recv", self.states)
        before = receiver.snapshot()
        result = receiver.merge(receiver)
        self.assertIs(result, receiver)
        self.assertEqual(receiver.snapshot(), before)

    def test_idempotent_merge_with_equivalent_copy(self) -> None:
        receiver = self.merge_order("recv", self.states)
        before = receiver.snapshot()
        duplicate = wire_restore(receiver.snapshot())
        receiver.merge(duplicate)
        receiver.merge(wire_restore(receiver.snapshot()))
        self.assertEqual(receiver.snapshot(), before)
        self.assertEqual(
            entry_state(duplicate.snapshot()), entry_state(before)
        )

    def test_commutative_orderings(self) -> None:
        baseline = self.merge_order("baseline", self.states)
        for index, order in enumerate(permutations(self.states)):
            with self.subTest(order=index):
                receiver = self.merge_order(f"recv-{index}", order)
                self.assertEqual(
                    entry_state(receiver.snapshot()),
                    entry_state(baseline.snapshot()),
                )

    def test_associative_groupings(self) -> None:
        s1, s2, s3 = self.states

        left = self.merge_order("left", [s1, s2])
        left.merge(LWWRegister.from_snapshot(copy.deepcopy(s3.snapshot())))

        right_inner = self.merge_order("inner", [s2, s3])
        right = LWWRegister("right")
        right.merge(
            LWWRegister.from_snapshot(copy.deepcopy(s1.snapshot()))
        )
        right.merge(
            LWWRegister.from_snapshot(
                copy.deepcopy(right_inner.snapshot())
            )
        )

        flat = self.merge_order("flat", [s1, s2, s3])
        self.assertEqual(
            entry_state(left.snapshot()), entry_state(flat.snapshot())
        )
        self.assertEqual(
            entry_state(right.snapshot()), entry_state(flat.snapshot())
        )
        self.assertIsNone(left.value())
        self.assertEqual(left.value(), right.value())

    def test_merge_copies_value_no_shared_containers(self) -> None:
        a = LWWRegister("a")
        a.assign({"nested": [1, [2, 3]]})
        b = LWWRegister("b")
        b.merge(a)
        a.assign("overwritten")  # mutates a's own state afterwards
        self.assertEqual(b.value(), {"nested": [1, [2, 3]]})
        # Mutating the received value cannot affect the source of truth.
        view = b.value()
        view["nested"].append(4)
        again = LWWRegister("c")
        again.merge(b)
        self.assertEqual(again.value(), {"nested": [1, [2, 3]]})


class SnapshotTests(unittest.TestCase):
    def test_snapshots_are_independent_and_json_serializable(self) -> None:
        register = LWWRegister("alpha")
        register.assign({"a": [1, True, None], "b": "x"})
        first = register.snapshot()
        second = register.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["entry"], second["entry"])
        self.assertEqual(json.loads(json.dumps(first)), first)
        self.assertEqual(first, second)

        first["replica_id"] = "hacked"
        first["clock"] = 999
        first["entry"]["value"]["a"].append(999)
        first["unexpected"] = True
        self.assertEqual(register.snapshot(), second)
        self.assertEqual(
            register.value(), {"a": [1, True, None], "b": "x"}
        )

    def test_from_snapshot_preserves_full_state(self) -> None:
        source = LWWRegister("alpha")
        source.assign("v1")
        other = LWWRegister("beta")
        other.merge(wire_restore(source.snapshot()))
        other.assign({"v2": [True]})
        source.merge(wire_restore(other.snapshot()))

        data = json.loads(json.dumps(source.snapshot()))
        restored = LWWRegister.from_snapshot(data)
        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.value(), {"v2": [True]})
        self.assertEqual(
            restored.snapshot()["entry"],
            {"timestamp": [2, "beta"], "value": {"v2": [True]}},
        )

        # Continued local writes resume past the observed clock.
        restored.assign("v3")
        self.assertEqual(
            restored.snapshot()["entry"]["timestamp"], [3, "alpha"]
        )
        # And restored replicas keep merging and converging.
        peer = wire_restore(other.snapshot())
        peer.merge(wire_restore(restored.snapshot()))
        restored.merge(wire_restore(peer.snapshot()))
        self.assertEqual(peer.value(), restored.value())
        self.assertEqual(restored.value(), "v3")

    def test_restored_register_does_not_alias_source_dict(self) -> None:
        data = {
            "replica_id": "alpha",
            "clock": 2,
            "entry": {
                "timestamp": [2, "beta"],
                "value": {"k": [1, 2]},
            },
        }
        restored = LWWRegister.from_snapshot(data)
        data["replica_id"] = "mutated"
        data["clock"] = 500
        data["entry"]["timestamp"][0] = 9
        data["entry"]["value"]["k"].append(3)
        data["entry"]["value"]["new"] = False

        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.snapshot()["clock"], 2)
        self.assertEqual(
            restored.snapshot()["entry"],
            {
                "timestamp": [2, "beta"],
                "value": {"k": [1, 2]},
            },
        )

    def test_unassigned_snapshot_restores_with_value_null_entry(self) -> None:
        register = LWWRegister("zero")
        restored = wire_restore(register.snapshot())
        self.assertFalse(restored.has_value())
        with self.assertRaises(LookupError):
            restored.value()
        restored.assign("later")
        self.assertEqual(restored.value(), "later")
        self.assertEqual(
            restored.snapshot()["entry"]["timestamp"], [1, "zero"]
        )

    def test_json_round_trip_preserves_type_distinctions(self) -> None:
        for value in (
            True,
            False,
            None,
            1,
            1.0,
            0,
            0.0,
            "1",
            [1, 1.0, True, "x", None],
            {"a": False, "b": 0, "c": 0.0},
        ):
            with self.subTest(value=value):
                register = LWWRegister("r")
                register.assign(copy.deepcopy(value))
                restored = wire_restore(register.snapshot())
                self.assertEqual(
                    json.dumps(restored.value(), sort_keys=True),
                    json.dumps(value, sort_keys=True),
                )


class SnapshotValidationTests(unittest.TestCase):
    def assert_invalid(self, snapshot) -> None:
        with self.assertRaises(ValueError):
            LWWRegister.from_snapshot(copy.deepcopy(snapshot))

    def test_from_snapshot_requires_dict(self) -> None:
        for bad in (None, [], "x", 42, 3.14, {1, 2}, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    LWWRegister.from_snapshot(bad)

    def test_key_set_must_be_exact(self) -> None:
        good = {"replica_id": "a", "clock": 0, "entry": None}
        bad_snapshots = [
            {},
            {"replica_id": "a"},
            {"clock": 0},
            {"entry": None},
            {"replica_id": "a", "clock": 0},
            {"replica_id": "a", "entry": None},
            {**good, "extra": 1},
            {"id": "a", "clock": 0, "entry": None},
            {"replica_id": "a", "counter": 0, "entry": None},
        ]
        for bad in bad_snapshots:
            with self.subTest(bad=bad):
                self.assert_invalid(bad)

    def test_replica_id_validation(self) -> None:
        for bad_id in ("", None, 0, True, False, [], {}):
            with self.subTest(bad_id=bad_id):
                self.assert_invalid(
                    {"replica_id": bad_id, "clock": 0, "entry": None}
                )

    def test_clock_validation(self) -> None:
        for bad_clock in (-1, 1.0, "0", None, True, False, []):
            with self.subTest(bad_clock=bad_clock):
                self.assert_invalid(
                    {
                        "replica_id": "a",
                        "clock": bad_clock,
                        "entry": None,
                    }
                )

    def test_entry_shape_validation(self) -> None:
        for bad_entry in (
            [],
            "x",
            1,
            True,
            {},
            {"timestamp": [1, "a"]},
            {"value": 1},
            {"timestamp": [1, "a"], "value": 1, "extra": 2},
            {"stamp": [1, "a"], "value": 1},
        ):
            with self.subTest(bad_entry=bad_entry):
                self.assert_invalid(
                    {
                        "replica_id": "a",
                        "clock": 1,
                        "entry": bad_entry,
                    }
                )

    def test_timestamp_validation(self) -> None:
        def with_timestamp(timestamp) -> dict:
            return {
                "replica_id": "a",
                "clock": 2,
                "entry": {"timestamp": timestamp, "value": 1},
            }

        for bad_timestamp in (
            [1],
            [1, "a", 2],
            [],
            "x",
            None,
            1,
            {},
            [0, "a"],
            [-1, "a"],
            [1.0, "a"],
            [True, "a"],
            ["1", "a"],
            [1, ""],
            [1, None],
            [1, 2],
            [None, "a"],
        ):
            with self.subTest(bad_timestamp=bad_timestamp):
                self.assert_invalid(with_timestamp(bad_timestamp))

    def test_timestamp_count_must_not_exceed_clock(self) -> None:
        base = {
            "replica_id": "a",
            "entry": {"timestamp": [3, "b"], "value": "x"},
        }
        self.assert_invalid({**base, "clock": 2})
        self.assert_invalid({**base, "clock": 0})
        # Exactly equal is fine, as is strictly below.
        for clock in (3, 4, 10):
            restored = LWWRegister.from_snapshot({**base, "clock": clock})
            self.assertEqual(restored.snapshot()["clock"], clock)

    def test_value_must_be_json_shaped(self) -> None:
        def with_value(value) -> dict:
            return {
                "replica_id": "a",
                "clock": 1,
                "entry": {"timestamp": [1, "a"], "value": value},
            }

        for bad_value in (
            float("inf"),
            float("-inf"),
            float("nan"),
            1j,
            b"bytes",
            (1,),
            {1: 2},
            {None: 1},
            [float("inf")],
            {"a": b"x"},
            {"a": {True: 1}},
        ):
            with self.subTest(bad_value=bad_value):
                self.assert_invalid(with_value(bad_value))

    def test_accepts_valid_shapes(self) -> None:
        valid = [
            {"replica_id": "a", "clock": 0, "entry": None},
            {
                "replica_id": "a",
                "clock": 1,
                "entry": {"timestamp": [1, "a"], "value": None},
            },
            {
                "replica_id": "a",
                "clock": 3,
                "entry": {
                    "timestamp": [2, "b"],
                    "value": {"k": [1, True, None, "s"]},
                },
            },
        ]
        for snapshot in valid:
            with self.subTest(snapshot=snapshot):
                restored = LWWRegister.from_snapshot(
                    copy.deepcopy(snapshot)
                )
                self.assertEqual(
                    restored.replica_id, snapshot["replica_id"]
                )
                self.assertEqual(restored.snapshot(), snapshot)

    def test_failed_from_snapshot_leaves_no_partial_state(self) -> None:
        with self.assertRaises(ValueError):
            LWWRegister.from_snapshot(
                {
                    "replica_id": "a",
                    "clock": 1,
                    "entry": {
                        "timestamp": [2, "a"],
                        "value": "x",
                    },
                }
            )


class PackageSurfaceTests(unittest.TestCase):
    def test_lwwregister_is_top_level_exported_alongside_types(
        self,
    ) -> None:
        import crdt_sync

        self.assertIn("LWWRegister", crdt_sync.__all__)
        self.assertIn("GCounter", crdt_sync.__all__)
        self.assertIn("ORSet", crdt_sync.__all__)
        self.assertIs(crdt_sync.LWWRegister, LWWRegister)
        self.assertIs(crdt_sync.GCounter, GCounter)
        self.assertIs(crdt_sync.ORSet, ORSet)

    def test_repr_does_not_dump_internals(self) -> None:
        empty = LWWRegister("r")
        self.assertIn("LWWRegister", repr(empty))
        self.assertIn("r", repr(empty))

        empty.assign("a")
        text = repr(empty)
        self.assertIn("LWWRegister", text)
        self.assertIn("a", text)


class ExistingTypesUnaffectedTests(unittest.TestCase):
    """The baseline types keep their snapshot/merge surface exactly."""

    def test_gcounter_still_works(self) -> None:
        a = GCounter("a")
        b = GCounter("b")
        a.increment(2)
        b.increment(3)
        a.merge(b)
        self.assertEqual(a.value(), 5)
        self.assertEqual(
            a.snapshot(),
            {"replica_id": "a", "counts": {"a": 2, "b": 3}},
        )

    def test_orset_still_works(self) -> None:
        a = ORSet("a")
        a.add("x")
        a.add("y")
        a.remove("x")
        self.assertEqual(a.elements(), {"y"})
        restored = ORSet.from_snapshot(
            json.loads(json.dumps(a.snapshot()))
        )
        self.assertEqual(restored.elements(), {"y"})

    def test_cross_type_merges_are_rejected(self) -> None:
        lww = LWWRegister("r")
        lww.assign(1)
        with self.assertRaises(TypeError):
            lww.merge(GCounter("r"))
        with self.assertRaises(TypeError):
            lww.merge(ORSet("r"))
        with self.assertRaises(TypeError):
            GCounter("r").merge(lww)
        with self.assertRaises(TypeError):
            ORSet("r").merge(lww)


class CliTests(unittest.TestCase):
    """The version/help CLI contract must remain unchanged."""

    def test_version_command(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["version"])
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue().strip(), __version__)

    def test_help_command(self) -> None:
        for argv in (["help"], ["-h"], ["--help"], []):
            with self.subTest(argv=argv):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(argv)
                self.assertEqual(rc, 0)
                output = buf.getvalue()
                self.assertIn("version", output)
                self.assertIn("help", output)

    def test_unknown_command_exits_nonzero(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = main(["nope"])
        self.assertEqual(rc, 2)
        self.assertIn("unknown command", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
