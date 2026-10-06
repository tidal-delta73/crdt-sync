"""Systematic tests for the public LWWRegister contract.

Coverage:
* the unassigned lifecycle: ``has_value`` / ``value`` / ``LookupError``;
* assignable value kinds, rejection of bad input and deep-copy isolation of
  both the input and the returned value;
* LWW merge semantics: count priority, replica-id Unicode tie-break, clock
  advancement even when the remote write loses, and same-timestamp/different-
  value conflicts failing atomically;
* convergence under duplicated, reordered, batched, interleaved and stale
  snapshot delivery, including snapshot round trips across a JSON boundary;
* merge algebra: idempotence, commutativity, associativity, receiver return
  value and argument isolation;
* snapshot / from_snapshot independence, JSON serialization and the full
  validation contract;
* LWWRegister being importable from the package top level alongside GCounter
  and ORSet.

Only the public API is used; no private attributes are relied upon.
"""

from __future__ import annotations

import copy
import json
import math
import unittest
from itertools import permutations

from crdt_sync import GCounter, LWWRegister, ORSet


def wire_restore(snapshot: dict) -> LWWRegister:
    """Restore a snapshot after a full JSON round trip (simulated wire)."""
    return LWWRegister.from_snapshot(json.loads(json.dumps(snapshot)))


def cap(replica: LWWRegister) -> dict:
    """A detached JSON-shape copy of a replica's current state."""
    return json.loads(json.dumps(replica.snapshot()))


def deliver(receiver: LWWRegister, snapshots) -> LWWRegister:
    """Merge a sequence of snapshot dicts into ``receiver`` in order."""
    for snapshot in snapshots:
        receiver.merge(wire_restore(snapshot))
    return receiver


def build_scenario():
    """Return ``(history, finals, expected_value, expected_origin)``.

    Three replicas edit offline and exchange only snapshots; the winner chain
    exercises a higher count, a lower id tie-break and an observed-but-losing
    write.
    """
    alpha = LWWRegister("alpha")
    beta = LWWRegister("beta")
    gamma = LWWRegister("gamma")

    alpha.assign("a1")  # (1, alpha)
    beta.assign("b1")  # (1, beta) — same count, "alpha" < "beta"
    gamma.assign({"v": "g1", "n": [1, 2]})  # (1, gamma)
    history = [cap(alpha), cap(beta), cap(gamma)]

    # alpha continues from a restored snapshot and writes twice.
    alpha2 = wire_restore(alpha.snapshot())
    alpha2.assign(["a2", None, True])  # (2, alpha) — highest count so far
    history.append(cap(alpha2))

    # gamma observes alpha's (2, alpha), then writes locally: its clock jumps
    # to 3 regardless of whether the observed write won.
    gamma2 = wire_restore(gamma.snapshot())
    gamma2.merge(wire_restore(alpha2.snapshot()))
    gamma2.assign("g3")  # (3, gamma) — the final winner
    history.append(cap(gamma2))

    # beta observes the (3, gamma) write but then only redelivers old state.
    beta2 = wire_restore(beta.snapshot())
    beta2.merge(wire_restore(gamma2.snapshot()))
    history.append(cap(beta2))

    finals = [cap(alpha2), cap(beta2), cap(gamma2)]
    return history, finals, "g3", [3, "gamma"]


class UnassignedLifecycleTests(unittest.TestCase):
    def test_fresh_register_has_no_value(self) -> None:
        register = LWWRegister("r")
        self.assertEqual(register.replica_id, "r")
        self.assertFalse(register.has_value())
        with self.assertRaises(LookupError):
            register.value()

    def test_assign_then_value(self) -> None:
        register = LWWRegister("r")
        register.assign(None)
        self.assertTrue(register.has_value())
        self.assertIsNone(register.value())

        register.assign(False)
        self.assertIs(register.value(), False)

    def test_empty_snapshot_shape(self) -> None:
        register = LWWRegister("r")
        self.assertEqual(
            register.snapshot(),
            {"replica_id": "r", "clock": 0, "entry": None},
        )


class AssignValueTests(unittest.TestCase):
    ACCEPTED = (
        None,
        True,
        False,
        "",
        "héllo 世界 🎉",
        0,
        -17,
        10**40,
        1.5,
        -0.0,
        [],
        {},
        [1, "two", None, True, [3.0, {"x": False}]],
        {"a": 1, "b": [None, {"c": "d"}], "e": {}},
        # Non-cyclic sharing survives a JSON round trip.
    )

    REJECTED = (
        float("nan"),
        float("inf"),
        float("-inf"),
        math.nan,
        1j,
        b"bytes",
        bytearray(b"x"),
        (1, 2),
        {1, 2},
        frozenset({1}),
        object(),
        ...,
    )

    def test_accepted_value_kinds_round_trip(self) -> None:
        for value in self.ACCEPTED:
            with self.subTest(value=value):
                register = LWWRegister("r")
                register.assign(copy.deepcopy(value))
                self.assertTrue(register.has_value())
                # Deep equality after a JSON round trip is the storage shape.
                stored = json.loads(json.dumps(register.value()))
                self.assertEqual(stored, json.loads(json.dumps(value)))

    def test_non_cyclic_container_sharing_is_accepted(self) -> None:
        shared = {"k": 1}
        register = LWWRegister("r")
        register.assign([shared, shared, {"s": shared}])
        self.assertEqual(
            register.value(), [{"k": 1}, {"k": 1}, {"s": {"k": 1}}]
        )

    def test_rejected_value_kinds_raise_typeerror(self) -> None:
        for value in self.REJECTED:
            with self.subTest(value=value):
                register = LWWRegister("r")
                register.assign("prior")
                with self.assertRaises(TypeError):
                    register.assign(value)
                # A rejected assign leaves the prior state untouched.
                self.assertTrue(register.has_value())
                self.assertEqual(register.value(), "prior")

    def test_rejected_nested_values(self) -> None:
        bad_nested = [
            [1, float("nan")],
            {"x": float("inf")},
            [{"y": [1j]}],
            {1: "numeric key"},  # type: ignore[dict-item]
            {None: 1},  # type: ignore[dict-item]
            {True: 1},  # type: ignore[dict-item]
            [b"bytes"],
            [(1,)],
        ]
        for value in bad_nested:
            with self.subTest(value=value):
                register = LWWRegister("r")
                with self.assertRaises(TypeError):
                    register.assign(value)
                self.assertFalse(register.has_value())
                with self.assertRaises(LookupError):
                    register.value()

    def test_self_referential_input_is_rejected(self) -> None:
        cyclic_list: list = []
        cyclic_list.append(cyclic_list)
        register = LWWRegister("r")
        with self.assertRaises(TypeError):
            register.assign(cyclic_list)
        self.assertFalse(register.has_value())

        cyclic_dict: dict = {}
        cyclic_dict["self"] = cyclic_dict
        with self.assertRaises(TypeError):
            register.assign(cyclic_dict)
        self.assertFalse(register.has_value())

    def test_assign_deep_copies_the_input(self) -> None:
        register = LWWRegister("r")
        payload = {"a": [1, 2, {"b": 3}]}
        register.assign(payload)
        payload["a"].append(999)
        payload["a"][2]["b"] = "mutated"
        payload["added"] = True
        self.assertEqual(register.value(), {"a": [1, 2, {"b": 3}]})

        listed = [[1], [2]]
        register.assign(listed)
        listed[0].append(7)
        self.assertEqual(register.value(), [[1], [2]])

    def test_value_returns_an_independent_copy(self) -> None:
        register = LWWRegister("r")
        register.assign({"a": [1, 2], "b": {"c": 3}})
        first = register.value()
        second = register.value()
        self.assertIsNot(first, second)
        self.assertEqual(first, second)

        first["a"].append(999)
        first["b"]["c"] = "hacked"
        first["ghost"] = None
        self.assertEqual(register.value(), {"a": [1, 2], "b": {"c": 3}})
        self.assertEqual(second, {"a": [1, 2], "b": {"c": 3}})

    def test_reassign_replaces_value_and_advances_clock(self) -> None:
        register = LWWRegister("r")
        register.assign(1)
        register.assign(2)
        register.assign(3)
        self.assertEqual(register.value(), 3)
        self.assertEqual(register.snapshot()["clock"], 3)
        self.assertEqual(
            register.snapshot()["entry"]["timestamp"], [3, "r"]
        )


class MergeSemanticsTests(unittest.TestCase):
    def test_merge_requires_an_lwwregister(self) -> None:
        register = LWWRegister("alpha")
        register.assign("v")
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
        self.assertEqual(register.value(), "v")

    def test_merge_returns_receiver_and_keeps_argument(self) -> None:
        receiver = LWWRegister("r")
        other = LWWRegister("o")
        other.assign("x")
        before = other.snapshot()
        self.assertIs(receiver.merge(other), receiver)
        self.assertEqual(other.snapshot(), before)

    def test_higher_count_wins(self) -> None:
        a = LWWRegister("a")
        b = LWWRegister("b")
        a.assign("a1")
        a.assign("a2")  # (2, a)
        b.assign("b1")  # (1, b)
        a.merge(b)
        b.merge(wire_restore(a.snapshot()))
        self.assertEqual(a.value(), "a2")
        self.assertEqual(b.value(), "a2")

    def test_same_count_resolves_by_replica_id_unicode_order(self) -> None:
        low = LWWRegister("aaa")
        high = LWWRegister("zzz")
        low.assign("low")
        high.assign("high")

        first = LWWRegister("observer")
        first.merge(wire_restore(high.snapshot()))
        first.merge(wire_restore(low.snapshot()))

        second = LWWRegister("observer2")
        second.merge(wire_restore(low.snapshot()))
        second.merge(wire_restore(high.snapshot()))

        # Tuple comparison: (1, "aaa") < (1, "zzz"), so high wins either way.
        self.assertEqual(first.value(), "high")
        self.assertEqual(second.value(), "high")

    def test_unicode_tie_break_uses_code_point_order(self) -> None:
        # "A" (U+0041) sorts before "é" (U+00E9); bytes/UTF-8 order agrees
        # here too, but construct a case where code point order is decisive:
        # "é" (U+00E9, utf-8 0xC3 0xA9) vs "€" (U+20AC, utf-8 0xE2 0x82 0xAC)
        # — code point order says é < € regardless of encoding.
        accent = LWWRegister("é")
        euro = LWWRegister("€")
        accent.assign("accent")
        euro.assign("euro")
        merged = LWWRegister("m")
        merged.merge(wire_restore(accent.snapshot()))
        merged.merge(wire_restore(euro.snapshot()))
        self.assertEqual(merged.value(), "euro")

    def test_receiver_remembers_observed_count_when_remote_loses(self) -> None:
        # a's write (2, a) beats b's (1, b); b observes it without writing.
        a = LWWRegister("a")
        a.assign("a0")
        a.assign("a1")
        b = LWWRegister("b")
        b.assign("b0")
        b.merge(wire_restore(a.snapshot()))
        self.assertEqual(b.value(), "a1")

        # b's next local write must be later than the observed count of 2.
        b.assign("b2")
        self.assertEqual(b.snapshot()["entry"]["timestamp"], [3, "b"])

        # Even a losing remote write advances the remembered clock:
        # c writes (3, c), d writes (5, d); d observes c's losing entry and
        # must jump past 3 on its next assign.
        c = LWWRegister("c")
        c.assign("c0")
        c.assign("c1")
        c.assign("c2")
        d = LWWRegister("d")
        for _ in range(5):
            d.assign("d")
        before = d.snapshot()
        d.merge(wire_restore(c.snapshot()))  # (3, c) loses to (5, d)
        self.assertEqual(d.snapshot(), before)
        d.assign("d6")
        self.assertEqual(d.snapshot()["entry"]["timestamp"], [6, "d"])

    def test_unassigned_receiver_absorbs_remote_entry(self) -> None:
        receiver = LWWRegister("r")
        other = LWWRegister("o")
        other.assign({"nested": [1, True, None]})
        receiver.merge(other)
        self.assertTrue(receiver.has_value())
        self.assertEqual(receiver.value(), {"nested": [1, True, None]})
        self.assertEqual(receiver.snapshot()["clock"], 1)

    def test_merging_unassigned_other_changes_nothing(self) -> None:
        receiver = LWWRegister("r")
        receiver.assign("v")
        before = receiver.snapshot()
        receiver.merge(LWWRegister("fresh"))
        self.assertEqual(receiver.snapshot(), before)

    def test_both_unassigned_merge_is_noop(self) -> None:
        a = LWWRegister("a")
        b = LWWRegister("b")
        before = a.snapshot()
        self.assertIs(a.merge(b), a)
        self.assertEqual(a.snapshot(), before)
        self.assertFalse(a.has_value())

    def test_same_timestamp_same_value_is_fine(self) -> None:
        # Same replica id (e.g. a backup) delivering an identical entry.
        a = LWWRegister("same")
        a.assign({"x": 1})
        duplicate = wire_restore(a.snapshot())
        receiver = LWWRegister("recv")
        receiver.assign({"x": 1})
        receiver.merge(duplicate)
        self.assertEqual(receiver.value(), {"x": 1})

    def test_same_timestamp_different_value_conflicts(self) -> None:
        left_value = {"x": 1}
        right_value = {"x": 2}
        left_snap = {
            "replica_id": "a",
            "clock": 1,
            "entry": {"timestamp": [1, "w"], "value": left_value},
        }
        right_snap = {
            "replica_id": "b",
            "clock": 1,
            "entry": {"timestamp": [1, "w"], "value": right_value},
        }
        left = wire_restore(left_snap)
        right = wire_restore(right_snap)
        left_before = left.snapshot()
        right_before = right.snapshot()

        with self.assertRaises(ValueError):
            left.merge(right)
        with self.assertRaises(ValueError):
            right.merge(left)

        # Neither side changes.
        self.assertEqual(left.snapshot(), left_before)
        self.assertEqual(right.snapshot(), right_before)
        self.assertEqual(left.value(), {"x": 1})
        self.assertEqual(right.value(), {"x": 2})

    def test_conflict_requires_structural_difference(self) -> None:
        # Numeric int/float equality at the same timestamp is not a conflict.
        int_snap = {
            "replica_id": "a",
            "clock": 1,
            "entry": {"timestamp": [1, "w"], "value": 1},
        }
        float_snap = {
            "replica_id": "b",
            "clock": 1,
            "entry": {"timestamp": [1, "w"], "value": 1.0},
        }
        a = wire_restore(int_snap)
        b = wire_restore(float_snap)
        a.merge(b)
        self.assertEqual(a.value(), 1)

        # True vs 1 at the same timestamp IS a structural conflict.
        bool_snap = {
            "replica_id": "c",
            "clock": 1,
            "entry": {"timestamp": [1, "w"], "value": True},
        }
        c = wire_restore(bool_snap)
        d = wire_restore(int_snap)
        with self.assertRaises(ValueError):
            c.merge(d)

    def test_remote_entry_value_is_deep_copied(self) -> None:
        receiver = LWWRegister("r")
        other = LWWRegister("o")
        other.assign({"a": [1, 2]})
        receiver.merge(other)
        other.assign("overwritten")
        self.assertEqual(receiver.value(), {"a": [1, 2]})
        self.assertEqual(other.value(), "overwritten")


class ConvergenceTests(unittest.TestCase):
    """Duplicate, out-of-order, batched, interleaved and stale delivery."""

    def setUp(self) -> None:
        self.history, self.finals, self.expected_value, self.expected_ts = (
            build_scenario()
        )

    def assertConverged(self, register: LWWRegister) -> None:
        self.assertTrue(register.has_value())
        self.assertEqual(register.value(), self.expected_value)
        entry = register.snapshot()["entry"]
        self.assertEqual(entry["timestamp"], self.expected_ts)
        # The converged state survives a JSON wire round trip.
        again = wire_restore(register.snapshot())
        self.assertEqual(again.value(), self.expected_value)

    def test_all_permutations_of_final_states_converge(self) -> None:
        for index, order in enumerate(permutations(self.finals)):
            with self.subTest(order=index):
                receiver = deliver(LWWRegister(f"observer-{index}"), order)
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

    def test_batched_delivery_between_replicas_converges(self) -> None:
        fa, fb, fg = self.finals

        x = deliver(LWWRegister("x"), [fa])
        y = deliver(LWWRegister("y"), [fb, fg])
        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertConverged(x)
        self.assertConverged(y)
        # replica_id is per receiver, so compare the replica-independent
        # state (clock + winning entry) only.
        self.assertEqual(x.snapshot()["clock"], y.snapshot()["clock"])
        self.assertEqual(x.snapshot()["entry"], y.snapshot()["entry"])

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
                self.history[2],
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
        snapshots = [register.snapshot() for register in receivers]
        for snapshot in snapshots[1:]:
            self.assertEqual(snapshot["entry"], snapshots[0]["entry"])
            self.assertEqual(snapshot["clock"], snapshots[0]["clock"])
        # replica_id is per receiver and must survive merges untouched.
        self.assertEqual(
            [s["replica_id"] for s in snapshots],
            ["node-0", "node-1", "node-2", "node-3"],
        )

    def test_restored_replica_continues_and_converges(self) -> None:
        history, finals, _, _ = build_scenario()
        restored = wire_restore(history[3])
        # Continue writing locally past every observed count.
        restored.assign("post-restore")
        restored.assign(self.expected_value)  # equal value, fresh timestamp
        post = wire_restore(restored.snapshot())
        for final in finals:
            post.merge(wire_restore(final))
        self.assertIn(post.value(), {self.expected_value, "post-restore"})


class MergeAlgebraTests(unittest.TestCase):
    """Idempotence, commutativity, associativity and object semantics."""

    def setUp(self) -> None:
        s1 = LWWRegister("a")
        s1.assign("s1-1")
        s1.assign("s1-2")

        s2 = LWWRegister("b")
        s2.merge(wire_restore(s1.snapshot()))
        s2.assign("s2-3")  # (3, b)

        s3 = LWWRegister("c")
        s3.assign("s3-1")

        self.states = [s1, s2, s3]

    def merge_order(self, owner_id: str, ordered) -> LWWRegister:
        receiver = LWWRegister(owner_id)
        for state in ordered:
            receiver.merge(
                LWWRegister.from_snapshot(copy.deepcopy(state.snapshot()))
            )
        return receiver

    def state_tuple(self, register: LWWRegister) -> tuple:
        snapshot = register.snapshot()
        return (snapshot["clock"], json.dumps(snapshot["entry"], sort_keys=True))

    def test_merge_returns_receiver(self) -> None:
        receiver = LWWRegister("recv")
        self.assertIs(receiver.merge(self.states[0]), receiver)

    def test_merge_does_not_modify_argument(self) -> None:
        for state in self.states:
            before = state.snapshot()
            receiver = LWWRegister("recv")
            receiver.merge(state)
            self.assertEqual(state.snapshot(), before)

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
        self.assertEqual(duplicate.snapshot(), before)

    def test_commutative_orderings(self) -> None:
        baseline = self.merge_order("baseline", self.states)
        for index, order in enumerate(permutations(self.states)):
            with self.subTest(order=index):
                receiver = self.merge_order(f"recv-{index}", order)
                self.assertEqual(
                    self.state_tuple(receiver), self.state_tuple(baseline)
                )

    def test_associative_groupings(self) -> None:
        s1, s2, s3 = self.states

        left = self.merge_order("left", [s1, s2])
        left.merge(LWWRegister.from_snapshot(copy.deepcopy(s3.snapshot())))

        right_inner = self.merge_order("inner", [s2, s3])
        right = LWWRegister("right")
        right.merge(LWWRegister.from_snapshot(copy.deepcopy(s1.snapshot())))
        right.merge(
            LWWRegister.from_snapshot(copy.deepcopy(right_inner.snapshot()))
        )

        flat = self.merge_order("flat", [s1, s2, s3])
        self.assertEqual(self.state_tuple(left), self.state_tuple(flat))
        self.assertEqual(self.state_tuple(right), self.state_tuple(flat))
        self.assertEqual(left.value(), right.value())


class SnapshotIsolationTests(unittest.TestCase):
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
        first["entry"]["timestamp"] = [999, "ghost"]
        first["entry"]["value"]["a"].append("leak")
        first["unexpected"] = True
        self.assertEqual(register.value(), {"a": [1, True, None], "b": "x"})
        self.assertEqual(register.snapshot(), second)

    def test_unassigned_snapshot_round_trips(self) -> None:
        register = LWWRegister("alpha")
        self.assertEqual(
            json.loads(json.dumps(register.snapshot())),
            {"replica_id": "alpha", "clock": 0, "entry": None},
        )
        restored = wire_restore(register.snapshot())
        self.assertFalse(restored.has_value())
        with self.assertRaises(LookupError):
            restored.value()
        restored.assign("later")
        self.assertEqual(restored.value(), "later")

    def test_from_snapshot_preserves_state(self) -> None:
        source = LWWRegister("alpha")
        source.assign("a1")  # clock 1, entry (1, alpha)
        other = LWWRegister("beta")
        other.assign("b1")
        other.assign("b2")  # clock 2, entry (2, beta)
        source.merge(other)  # clock 2, entry (2, beta)
        source.assign("a3")  # clock 3, entry (3, alpha)

        data = json.loads(json.dumps(source.snapshot()))
        restored = LWWRegister.from_snapshot(data)
        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.value(), "a3")
        self.assertEqual(restored.snapshot()["clock"], 3)
        self.assertEqual(
            restored.snapshot()["entry"]["timestamp"], [3, "alpha"]
        )

        # Continued assigns move past the observed logical count.
        restored.assign("a4")
        self.assertEqual(restored.snapshot()["clock"], 4)
        self.assertEqual(
            restored.snapshot()["entry"]["timestamp"], [4, "alpha"]
        )

        # A fresh peer sharing this replica id starts at count 1; it first
        # catches up via merge, then writes beyond every observed count, and
        # both sides converge.
        live = LWWRegister("alpha")
        live.assign("live1")  # (1, alpha)
        live.merge(wire_restore(restored.snapshot()))
        self.assertEqual(live.value(), "a4")
        live.assign("live5")  # (5, alpha)
        restored.merge(wire_restore(live.snapshot()))
        self.assertEqual(restored.value(), "live5")
        self.assertEqual(
            restored.snapshot()["entry"]["timestamp"], [5, "alpha"]
        )

    def test_restored_register_does_not_alias_source_dict(self) -> None:
        data = {
            "replica_id": "alpha",
            "clock": 3,
            "entry": {"timestamp": [3, "beta"], "value": {"v": [1, 2]}},
        }
        restored = LWWRegister.from_snapshot(data)

        data["replica_id"] = "mutated"
        data["clock"] = 500
        data["entry"]["timestamp"] = [9, "ghost"]
        data["entry"]["value"]["v"].append(3)

        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.value(), {"v": [1, 2]})
        self.assertEqual(
            restored.snapshot()["entry"]["timestamp"], [3, "beta"]
        )
        self.assertEqual(restored.snapshot()["clock"], 3)


class SnapshotValidationTests(unittest.TestCase):
    """The current TypeError / ValueError contract must hold."""

    BASE_ASSIGNED = {
        "replica_id": "a",
        "clock": 1,
        "entry": {"timestamp": [1, "a"], "value": "v"},
    }

    def test_replica_id_validation(self) -> None:
        for bad in (None, 1, 1.0, b"a", [], ("a",), True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    LWWRegister(bad)
        with self.assertRaises(ValueError):
            LWWRegister("")

    def test_from_snapshot_requires_dict(self) -> None:
        for bad in (None, [], "x", 42, 3.14, {1, 2}, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    LWWRegister.from_snapshot(bad)

    def test_from_snapshot_key_set_must_be_exact(self) -> None:
        good_unassigned = {"replica_id": "a", "clock": 0, "entry": None}
        bad_snapshots = [
            {},
            {"replica_id": "a"},
            {"replica_id": "a", "clock": 0},
            {**good_unassigned, "extra": 1},
            {"id": "a", "clock": 0, "entry": None},
            {"replica_id": "a", "counter": 0, "entry": None},
            {"replica_id": "a", "clock": 0},
        ]
        for bad in bad_snapshots:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    LWWRegister.from_snapshot(bad)

    def test_from_snapshot_replica_id_validation(self) -> None:
        for bad_id in ("", None, 0, True, False, [], {}):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(ValueError):
                    LWWRegister.from_snapshot(
                        {"replica_id": bad_id, "clock": 0, "entry": None}
                    )

    def test_from_snapshot_clock_validation(self) -> None:
        for bad_clock in (-1, 1.0, "0", None, True, False, []):
            with self.subTest(bad_clock=bad_clock):
                with self.assertRaises(ValueError):
                    LWWRegister.from_snapshot(
                        {"replica_id": "a", "clock": bad_clock, "entry": None}
                    )

    def test_from_snapshot_entry_shape(self) -> None:
        base = {"replica_id": "a", "clock": 1}
        bad_entries = [
            1,
            "nullish",
            [],
            True,
            {},
            {"timestamp": [1, "a"]},
            {"value": "v"},
            {"timestamp": [1, "a"], "value": "v", "extra": 1},
        ]
        for entry in bad_entries:
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    LWWRegister.from_snapshot({**base, "entry": entry})

    def test_from_snapshot_timestamp_format(self) -> None:
        def with_ts(timestamp: object) -> dict:
            return {
                "replica_id": "a",
                "clock": 3,
                "entry": {"timestamp": timestamp, "value": "v"},
            }

        bad_timestamps = [
            None,
            [],
            [1],
            [1, "a", "x"],
            [0, "a"],
            [-1, "a"],
            [1.0, "a"],
            [True, "a"],
            ["1", "a"],
            [1, ""],
            [1, None],
            [1, 2],
            [1, "a", 3],
            (1, "a"),  # must be a JSON list, not a tuple
        ]
        for timestamp in bad_timestamps:
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    LWWRegister.from_snapshot(with_ts(timestamp))

    def test_from_snapshot_timestamp_must_not_exceed_clock(self) -> None:
        with self.assertRaises(ValueError):
            LWWRegister.from_snapshot(
                {
                    "replica_id": "a",
                    "clock": 2,
                    "entry": {"timestamp": [3, "a"], "value": "v"},
                }
            )
        # Equal is fine.
        restored = LWWRegister.from_snapshot(
            {
                "replica_id": "a",
                "clock": 3,
                "entry": {"timestamp": [3, "gamma"], "value": "v"},
            }
        )
        self.assertEqual(restored.value(), "v")
        self.assertEqual(restored.snapshot()["clock"], 3)

    def test_from_snapshot_value_must_be_json(self) -> None:
        base = {"replica_id": "a", "clock": 1}
        # Values that Python containers could carry but json / the assign API
        # reject are constructed as parsed-but-mutated structures where
        # possible; tuples and sets arrive directly.
        bad_values: tuple = (
            float("nan"),
            float("inf"),
            (1, 2),
            {1, 2},
            b"x",
            1j,
        )
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    LWWRegister.from_snapshot(
                        {
                            **base,
                            "entry": {
                                "timestamp": [1, "a"],
                                "value": value,
                            },
                        }
                    )
        # Numeric keys inside a value dict also break the JSON contract.
        with self.assertRaises(ValueError):
            LWWRegister.from_snapshot(
                {
                    **base,
                    "entry": {
                        "timestamp": [1, "a"],
                        "value": {1: "x"},
                    },
                }
            )

    def test_from_snapshot_accepts_valid_shapes(self) -> None:
        valid = [
            {"replica_id": "a", "clock": 0, "entry": None},
            {
                "replica_id": "a",
                "clock": 1,
                "entry": {"timestamp": [1, "a"], "value": None},
            },
            {
                "replica_id": "a",
                "clock": 4,
                "entry": {
                    "timestamp": [2, "beta"],
                    "value": [1, "two", False, None, {"k": "v"}],
                },
            },
        ]
        for snapshot in valid:
            with self.subTest(snapshot=snapshot):
                restored = LWWRegister.from_snapshot(copy.deepcopy(snapshot))
                self.assertEqual(restored.replica_id, snapshot["replica_id"])
                self.assertEqual(
                    json.loads(json.dumps(restored.snapshot())), snapshot
                )


class PackageSurfaceTests(unittest.TestCase):
    def test_lwwregister_is_top_level_importable(self) -> None:
        import crdt_sync

        self.assertIn("LWWRegister", crdt_sync.__all__)
        self.assertIn("GCounter", crdt_sync.__all__)
        self.assertIn("ORSet", crdt_sync.__all__)
        self.assertIs(crdt_sync.LWWRegister, LWWRegister)

    def test_repr_stays_compact(self) -> None:
        register = LWWRegister("r")
        self.assertIn("LWWRegister", repr(register))
        self.assertIn("r", repr(register))
        register.assign("v")
        text = repr(register)
        self.assertIn("LWWRegister", text)
        self.assertIn("r", text)
        self.assertIn("v", text)


if __name__ == "__main__":
    unittest.main()
