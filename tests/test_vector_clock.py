"""Systematic tests for the public VectorClock contract.

Coverage:
* causal ordering via ``compare``: before / after / equal / concurrent, with
  missing components treated as zero, including the headline offline-then-
  reconnect scenarios (concurrent ticks; merge-then-tick is strictly after);
* convergence under duplicated, reordered, batched, interleaved and relayed
  delivery, including snapshot round trips across a JSON boundary;
* merge algebra: idempotence, commutativity, associativity, receiver return
  value, stable ``replica_id`` and argument isolation;
* snapshot / from_snapshot independence and JSON serialization, with local
  ticks continuing from the restored component;
* the documented TypeError / ValueError input contract, including bools not
  counting as ints and failed calls leaving every object untouched.

Only the public API is used (``VectorClock``, its methods, ``replica_id``);
no private attributes, dict iteration order, or randomness is relied upon.
"""

from __future__ import annotations

import copy
import json
import unittest
from itertools import permutations

from crdt_sync import VectorClock


def wire_restore(snapshot: dict) -> VectorClock:
    """Restore a snapshot after a full JSON round trip (simulated wire)."""
    return VectorClock.from_snapshot(json.loads(json.dumps(snapshot)))


def deliver(receiver: VectorClock, snapshots) -> VectorClock:
    """Merge a sequence of snapshot dicts into ``receiver`` in order."""
    for snapshot in snapshots:
        receiver.merge(wire_restore(snapshot))
    return receiver


def cap(clock: VectorClock) -> dict:
    """A JSON-normalized, detached snapshot suitable for later delivery."""
    return json.loads(json.dumps(clock.snapshot()))


class CompareSemanticsTests(unittest.TestCase):
    """The happens-before relation, missing components treated as zero."""

    def test_fresh_clocks_are_equal_even_with_different_ids(self) -> None:
        self.assertEqual(VectorClock("a").compare(VectorClock("b")), "equal")
        self.assertEqual(VectorClock("b").compare(VectorClock("a")), "equal")

    def test_independent_ticks_are_concurrent(self) -> None:
        a = VectorClock("a")
        b = VectorClock("b")
        a.tick()
        b.tick()
        self.assertEqual(a.compare(b), "concurrent")
        self.assertEqual(b.compare(a), "concurrent")

        # Multiple rounds of purely local ticks stay concurrent.
        a.tick(2)
        b.tick(4)
        self.assertEqual(a.compare(b), "concurrent")
        self.assertEqual(b.compare(a), "concurrent")

    def test_merge_then_tick_is_strictly_after(self) -> None:
        a = VectorClock("a")
        b = VectorClock("b")
        a.tick(2)
        b.tick(3)
        a.merge(wire_restore(b.snapshot()))
        a.tick()  # local progress beyond everything observed
        self.assertEqual(a.compare(b), "after")
        self.assertEqual(b.compare(a), "before")

    def test_plain_happens_before_chain(self) -> None:
        a = VectorClock("a")
        a.tick()
        snapshot = cap(a)
        b = wire_restore(snapshot)  # b has observed a's state
        self.assertEqual(b.compare(a), "equal")
        a.tick()
        self.assertEqual(b.compare(a), "before")
        self.assertEqual(a.compare(b), "after")

    def test_equal_only_when_all_components_agree(self) -> None:
        a = VectorClock("a")
        b = VectorClock("b")
        a.tick(2)
        b.merge(wire_restore(a.snapshot()))
        b.tick(3)
        a.merge(wire_restore(b.snapshot()))
        # Converged knowledge; components identical despite different owners.
        self.assertEqual(a.components(), b.components())
        self.assertEqual(a.compare(b), "equal")
        self.assertEqual(b.compare(a), "equal")

    def test_missing_components_count_as_zero(self) -> None:
        a = VectorClock("a")
        b = VectorClock("b")
        a.tick()
        # From b's empty point of view a is strictly ahead; a sees b at zero.
        self.assertEqual(b.compare(a), "before")
        self.assertEqual(a.compare(b), "after")

    def test_zero_components_do_not_create_ordering(self) -> None:
        # An explicitly learned zero component behaves like a missing one.
        a = VectorClock.from_snapshot(
            {"replica_id": "a", "clock": {"a": 0, "b": 0}}
        )
        b = VectorClock.from_snapshot({"replica_id": "b", "clock": {"b": 0}})
        self.assertEqual(a.compare(b), "equal")
        a.tick()
        self.assertEqual(a.compare(b), "after")
        self.assertEqual(b.compare(a), "before")

    def test_three_way_concurrency_and_convergence(self) -> None:
        a, b, c = VectorClock("a"), VectorClock("b"), VectorClock("c")
        a.tick()
        b.tick()
        c.tick()
        for left, right in ((a, b), (b, c), (a, c)):
            self.assertEqual(left.compare(right), "concurrent")

        a.merge(wire_restore(b.snapshot()))
        a.merge(wire_restore(c.snapshot()))
        # a now dominates b and c but is concurrent with a fresh d.
        self.assertEqual(a.compare(b), "after")
        self.assertEqual(a.compare(c), "after")
        d = VectorClock("d")
        d.tick()
        self.assertEqual(a.compare(d), "concurrent")


class ConvergenceTests(unittest.TestCase):
    """Duplicate, out-of-order, batched, interleaved and relayed delivery."""

    def setUp(self) -> None:
        alpha = VectorClock("alpha")
        beta = VectorClock("beta")
        gamma = VectorClock("gamma")

        alpha.tick(3)
        beta.tick(1)
        gamma.tick(7)

        history = [cap(alpha), cap(beta), cap(gamma)]

        # Continue alpha from a restored snapshot, proving a round-tripped
        # replica can keep making local progress.
        alpha2 = wire_restore(alpha.snapshot())
        alpha2.tick(2)
        gamma.tick(4)
        history.extend([cap(alpha2), cap(gamma)])

        beta2 = wire_restore(beta.snapshot())
        beta2.tick(10)
        history.append(cap(beta2))

        self.history = history
        self.finals = [cap(alpha2), cap(beta2), cap(gamma)]
        self.expected = {"alpha": 5, "beta": 11, "gamma": 11}

    def assertConverged(self, clock: VectorClock) -> None:
        self.assertEqual(clock.components(), self.expected)
        self.assertEqual(
            json.loads(json.dumps(clock.snapshot()))["clock"], self.expected
        )

    def test_all_permutations_of_final_states_converge(self) -> None:
        for index, order in enumerate(permutations(self.finals)):
            with self.subTest(order=index):
                receiver = deliver(VectorClock(f"observer-{index}"), order)
                self.assertConverged(receiver)

    def test_duplicated_delivery_converges(self) -> None:
        fa, fb, fg = self.finals
        paths = [
            [fa, fa, fb, fg, fb, fg, fa],
            [fg, fb, fa, fa, fa],
            list(self.finals) * 3,
        ]
        for index, path in enumerate(paths):
            with self.subTest(path=index):
                receiver = deliver(VectorClock("observer"), path)
                self.assertConverged(receiver)

    def test_interleaved_intermediate_states_converge(self) -> None:
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
                receiver = deliver(VectorClock("observer"), path)
                self.assertConverged(receiver)

    def test_redelivery_of_old_or_same_state_changes_nothing(self) -> None:
        receiver = deliver(VectorClock("observer"), self.finals)
        self.assertConverged(receiver)
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

    def test_relayed_through_intermediate_replica_converges(self) -> None:
        fa, fb, fg = self.finals

        # gamma's state reaches alpha only via a chain of intermediates that
        # add their own knowledge along the way.
        relay = VectorClock("relay-1")
        deliver(relay, [fg])
        relay.tick(2)
        relay2 = VectorClock("relay-2")
        relay2.merge(wire_restore(relay.snapshot()))
        relay2.merge(wire_restore(fb))
        relay2.tick(5)

        alpha = wire_restore(fa)
        alpha.merge(wire_restore(relay2.snapshot()))

        # A fresh observer merging every original final plus the relay packet
        # must end with the same remote components (plus relay bookkeeping).
        observer = deliver(VectorClock("obs"), [fa, fb, fg])
        for replica_id, component in observer.components().items():
            self.assertEqual(alpha.components()[replica_id], component)
        self.assertEqual(alpha.components()["relay-1"], 2)
        self.assertEqual(alpha.components()["relay-2"], 5)

        # Feeding the relay packet back changes nothing on a converged clock:
        # it only adopts the relay bookkeeping, then a redelivery is a no-op.
        observer.merge(wire_restore(relay2.snapshot()))
        settled = observer.snapshot()
        observer.merge(wire_restore(relay2.snapshot()))
        self.assertEqual(observer.snapshot(), settled)
        self.assertEqual(observer.components()["relay-1"], 2)
        self.assertEqual(observer.components()["relay-2"], 5)
        for replica_id, component in self.expected.items():
            self.assertEqual(observer.components()[replica_id], component)

    def test_batched_delivery_between_replicas_converges(self) -> None:
        fa, fb, fg = self.finals
        x = deliver(VectorClock("x"), [fa])
        y = deliver(VectorClock("y"), [fb, fg])
        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertConverged(x)
        self.assertConverged(y)
        self.assertEqual(x.components(), y.components())

    def test_all_converged_receivers_agree_on_remote_components(self) -> None:
        paths = [
            self.finals,
            list(reversed(self.finals)) + [self.finals[1]],
            self.history,
            list(reversed(self.history)),
        ]
        receivers = [
            deliver(VectorClock(f"node-{index}"), path)
            for index, path in enumerate(paths)
        ]
        for receiver in receivers:
            self.assertEqual(
                {k: receiver.components()[k] for k in self.expected},
                self.expected,
            )
        # replica_id is per receiver and must not be rewritten by merges.
        self.assertEqual(
            [r.replica_id for r in receivers],
            ["node-0", "node-1", "node-2", "node-3"],
        )


class MergeAlgebraTests(unittest.TestCase):
    """Idempotence, commutativity, associativity and object semantics."""

    def setUp(self) -> None:
        s1 = VectorClock("a")
        s1.tick(2)

        # s2 also knows about a, so states partially overlap.
        s2 = VectorClock("b")
        s2.merge(wire_restore(s1.snapshot()))
        s2.tick(3)

        s3 = VectorClock("c")
        s3.tick(4)

        self.states = [s1, s2, s3]

    def merge_order(self, owner_id: str, ordered) -> VectorClock:
        receiver = VectorClock(owner_id)
        for state in ordered:
            receiver.merge(
                VectorClock.from_snapshot(copy.deepcopy(state.snapshot()))
            )
        return receiver

    def test_merge_returns_receiver(self) -> None:
        receiver = VectorClock("recv")
        self.assertIs(receiver.merge(self.states[0]), receiver)

    def test_merge_keeps_replica_id(self) -> None:
        receiver = self.merge_order("recv", self.states)
        receiver.merge(self.states[0])
        self.assertEqual(receiver.replica_id, "recv")

    def test_merge_does_not_modify_argument(self) -> None:
        for state in self.states:
            before = state.snapshot()
            VectorClock("recv").merge(state)
            self.assertEqual(state.snapshot(), before)

    def test_idempotent_merge_with_self_and_copy(self) -> None:
        receiver = self.merge_order("recv", self.states)
        before = receiver.snapshot()
        self.assertIs(receiver.merge(receiver), receiver)
        self.assertEqual(receiver.snapshot(), before)
        receiver.merge(wire_restore(receiver.snapshot()))
        receiver.merge(wire_restore(receiver.snapshot()))
        self.assertEqual(receiver.snapshot(), before)

    def test_commutative_orderings(self) -> None:
        baseline = self.merge_order("baseline", self.states)
        for index, order in enumerate(permutations(self.states)):
            with self.subTest(order=index):
                receiver = self.merge_order(f"recv-{index}", order)
                self.assertEqual(receiver.components(), baseline.components())

    def test_associative_groupings(self) -> None:
        s1, s2, s3 = self.states

        left = self.merge_order("left", [s1, s2])
        left.merge(VectorClock.from_snapshot(copy.deepcopy(s3.snapshot())))

        right_inner = self.merge_order("inner", [s2, s3])
        right = VectorClock("right")
        right.merge(VectorClock.from_snapshot(copy.deepcopy(s1.snapshot())))
        right.merge(
            VectorClock.from_snapshot(copy.deepcopy(right_inner.snapshot()))
        )

        flat = self.merge_order("flat", [s1, s2, s3])
        self.assertEqual(left.components(), flat.components())
        self.assertEqual(right.components(), flat.components())

    def test_converged_clocks_compare_equal(self) -> None:
        left = self.merge_order("left", self.states)
        right = self.merge_order("right", list(reversed(self.states)))
        self.assertEqual(left.compare(right), "equal")
        self.assertEqual(right.compare(left), "equal")


class SnapshotIsolationTests(unittest.TestCase):
    def test_fresh_clock_has_empty_components(self) -> None:
        clock = VectorClock("alpha")
        self.assertEqual(clock.components(), {})
        self.assertEqual(
            clock.snapshot(), {"replica_id": "alpha", "clock": {}}
        )

    def test_components_returned_dict_is_independent(self) -> None:
        clock = VectorClock("alpha")
        clock.tick(3)
        view = clock.components()
        view["alpha"] = 999
        view["ghost"] = 1
        self.assertEqual(clock.components(), {"alpha": 3})

        first = clock.components()
        second = clock.components()
        self.assertIsNot(first, second)
        self.assertEqual(first, second)

    def test_snapshots_are_independent_and_json_serializable(self) -> None:
        clock = VectorClock("alpha")
        clock.tick(3)
        first = clock.snapshot()
        second = clock.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["clock"], second["clock"])
        self.assertEqual(
            json.loads(json.dumps(first)),
            {"replica_id": "alpha", "clock": {"alpha": 3}},
        )

        first["replica_id"] = "hacked"
        first["clock"]["alpha"] = 999
        first["clock"]["ghost"] = 1
        first["unexpected"] = True
        self.assertEqual(
            clock.snapshot(),
            {"replica_id": "alpha", "clock": {"alpha": 3}},
        )
        self.assertEqual(second, {"replica_id": "alpha", "clock": {"alpha": 3}})

    def test_snapshot_clock_is_stable_and_contains_all_components(self) -> None:
        clock = VectorClock("alpha")
        peer_b = VectorClock("beta")
        peer_c = VectorClock("gamma")
        peer_b.tick(2)
        peer_c.tick(5)
        clock.merge(peer_b)
        clock.merge(peer_c)
        clock.tick(4)
        expected = {"alpha": 4, "beta": 2, "gamma": 5}
        self.assertEqual(clock.snapshot()["clock"], expected)
        # Repeated snapshots are byte-identical regardless of insertion path.
        self.assertEqual(
            list(clock.snapshot()["clock"]), ["alpha", "beta", "gamma"]
        )
        self.assertEqual(clock.components(), expected)

    def test_from_snapshot_preserves_id_and_components(self) -> None:
        source = VectorClock("alpha")
        source.tick(5)
        source.merge(VectorClock("beta"))
        beta = VectorClock("beta")
        beta.tick(2)
        source.merge(beta)

        data = json.loads(json.dumps(source.snapshot()))
        restored = VectorClock.from_snapshot(data)
        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(
            restored.snapshot()["clock"], {"alpha": 5, "beta": 2}
        )

        # Local ticks continue from the restored local component.
        restored.tick(4)
        self.assertEqual(
            restored.snapshot()["clock"], {"alpha": 9, "beta": 2}
        )

    def test_restored_clock_does_not_alias_source_dict(self) -> None:
        data = {"replica_id": "alpha", "clock": {"alpha": 5, "beta": 2}}
        restored = VectorClock.from_snapshot(data)
        data["replica_id"] = "mutated"
        data["clock"]["alpha"] = 500
        data["clock"]["gamma"] = 9
        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(
            restored.snapshot()["clock"], {"alpha": 5, "beta": 2}
        )

    def test_restore_tick_and_compare_round_trip(self) -> None:
        a = VectorClock("a")
        b = VectorClock("b")
        a.tick(3)
        b.tick(1)
        b.merge(wire_restore(a.snapshot()))

        restored = wire_restore(b.snapshot())
        self.assertEqual(restored.compare(b), "equal")
        restored.tick()
        self.assertEqual(restored.compare(b), "after")
        self.assertEqual(b.compare(restored), "before")
        # a has not learned about b yet, so the restored clock is still
        # concurrent with a's view (a knows nothing of b; restored knows a:3).
        self.assertEqual(restored.compare(a), "after")

        # Restored copies delivered back to the origin are idempotent.
        before = a.snapshot()
        a.merge(wire_restore(restored.snapshot()))
        a.merge(wire_restore(restored.snapshot()))
        self.assertNotEqual(a.snapshot(), before)
        self.assertEqual(a.compare(restored), "equal")

    def test_zero_components_are_legal(self) -> None:
        data = {"replica_id": "p", "clock": {"p": 0, "q": 0}}
        restored = VectorClock.from_snapshot(json.loads(json.dumps(data)))
        self.assertEqual(restored.components(), {"p": 0, "q": 0})
        restored.tick(3)
        self.assertEqual(restored.components(), {"p": 3, "q": 0})
        peer = VectorClock("q")
        peer.tick(8)
        restored.merge(peer)
        self.assertEqual(restored.components(), {"p": 3, "q": 8})

    def test_large_positive_integers(self) -> None:
        big = 10**30
        clock = VectorClock("alpha")
        clock.tick(big)
        clock.tick(big)
        restored = wire_restore(clock.snapshot())
        self.assertEqual(restored.components()["alpha"], 2 * big)
        restored.tick()
        self.assertEqual(restored.components()["alpha"], 2 * big + 1)


class ValidationTests(unittest.TestCase):
    """The TypeError / ValueError contract; failed calls change no state."""

    def test_replica_id_validation(self) -> None:
        for bad in (None, 1, 1.0, b"a", [], ("a",), True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    VectorClock(bad)
        with self.assertRaises(ValueError):
            VectorClock("")

    def test_tick_validation(self) -> None:
        clock = VectorClock("alpha")
        clock.tick(1)
        for bad in (True, False, 1.0, "1", None, [1], 1j):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    clock.tick(bad)
        for bad in (0, -1, -1000):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    clock.tick(bad)
        # A rejected tick leaves the clock untouched.
        self.assertEqual(clock.components(), {"alpha": 1})

    def test_merge_requires_a_vector_clock(self) -> None:
        clock = VectorClock("alpha")
        clock.tick()
        for bad in (None, 1, "alpha", [], {}, clock.snapshot()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    clock.merge(bad)
        self.assertEqual(clock.components(), {"alpha": 1})

    def test_compare_requires_a_vector_clock(self) -> None:
        clock = VectorClock("alpha")
        clock.tick()
        for bad in (None, 1, "alpha", [], {}, clock.snapshot()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    clock.compare(bad)
        self.assertEqual(clock.components(), {"alpha": 1})

    def test_failed_merge_leaves_both_objects_untouched(self) -> None:
        clock = VectorClock("alpha")
        clock.tick(2)
        with self.assertRaises(TypeError):
            clock.merge(object())
        self.assertEqual(clock.components(), {"alpha": 2})

        # A rejected from_snapshot must not partially populate anything.
        good_data = {"replica_id": "a", "clock": {"a": 1, "b": 2}}
        restored = VectorClock.from_snapshot(copy.deepcopy(good_data))
        for bad_snapshot in (
            {"replica_id": "a", "clock": {"a": 1, "b": -1}},
            {"replica_id": "a", "clock": {1: 1}},
            {"replica_id": "", "clock": {}},
            {"replica_id": "a", "clock": {}, "extra": 1},
        ):
            with self.subTest(bad_snapshot=bad_snapshot):
                with self.assertRaises(ValueError):
                    VectorClock.from_snapshot(copy.deepcopy(bad_snapshot))
        self.assertEqual(restored.snapshot(), good_data)

    def test_from_snapshot_requires_dict(self) -> None:
        for bad in (None, [], "x", 42, 3.14, {1, 2}, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    VectorClock.from_snapshot(bad)

    def test_from_snapshot_key_set_must_be_exact(self) -> None:
        for bad in (
            {},
            {"replica_id": "a"},
            {"clock": {}},
            {"replica_id": "a", "clock": {}, "extra": 1},
            {"id": "a", "clock": {}},
            {"replica_id": "a", "components": {}},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    VectorClock.from_snapshot(bad)

    def test_from_snapshot_replica_id_validation(self) -> None:
        for bad_id in ("", None, 0, True, False, [], {}):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(ValueError):
                    VectorClock.from_snapshot(
                        {"replica_id": bad_id, "clock": {}}
                    )

    def test_from_snapshot_clock_must_be_dict(self) -> None:
        for bad_clock in (None, [], "", 42, (), {1, 2}):
            with self.subTest(bad_clock=bad_clock):
                with self.assertRaises(ValueError):
                    VectorClock.from_snapshot(
                        {"replica_id": "a", "clock": bad_clock}
                    )

    def test_from_snapshot_component_keys_validation(self) -> None:
        for bad_clock in ({"": 1}, {1: 1}, {None: 1}, {True: 1}, {"": 0}):
            with self.subTest(bad_clock=bad_clock):
                with self.assertRaises(ValueError):
                    VectorClock.from_snapshot(
                        {"replica_id": "a", "clock": bad_clock}
                    )

    def test_from_snapshot_component_values_validation(self) -> None:
        for bad_value in (True, False, -1, -(10**9), 1.0, "1", None, [1]):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(ValueError):
                    VectorClock.from_snapshot(
                        {"replica_id": "a", "clock": {"a": bad_value}}
                    )

    def test_from_snapshot_accepts_valid_shapes(self) -> None:
        valid = [
            {"replica_id": "a", "clock": {}},
            {"replica_id": "a", "clock": {"a": 0}},
            {"replica_id": "a", "clock": {"b": 0, "a": 1}},
            {"replica_id": "x", "clock": {"x": 10**18}},
        ]
        for snapshot in valid:
            with self.subTest(snapshot=snapshot):
                restored = VectorClock.from_snapshot(copy.deepcopy(snapshot))
                self.assertEqual(restored.replica_id, snapshot["replica_id"])
                self.assertEqual(restored.snapshot()["clock"], snapshot["clock"])


if __name__ == "__main__":
    unittest.main()
