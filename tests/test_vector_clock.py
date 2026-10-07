"""Systematic tests for the public VectorClock contract.

Coverage:
* causal comparison (before / after / equal / concurrent), including the
  divergent-then-merge-then-tick scenario used by reconnect sync;
* convergence under duplicated, reordered, relayed and batched delivery,
  including snapshot round trips across a JSON boundary;
* merge algebra: idempotence, commutativity, associativity, receiver return
  value and argument isolation;
* snapshot / from_snapshot independence, stable key ordering and JSON
  serialization, with local ticks continuing from restored components;
* the documented TypeError / ValueError input contract, including bools not
  counting as integers and failed calls leaving state untouched;
* package-level export.

Only the public API is used (``VectorClock``, its methods and
``replica_id``); no private attributes or dict iteration order is relied
upon beyond the explicitly stable snapshot ordering.
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


def cap(clock: VectorClock) -> dict:
    """Snapshot a clock through a JSON boundary."""
    return json.loads(json.dumps(clock.snapshot()))


def deliver(receiver: VectorClock, snapshots) -> VectorClock:
    """Merge a sequence of snapshot dicts into ``receiver`` in order."""
    for snapshot in snapshots:
        receiver.merge(wire_restore(snapshot))
    return receiver


class BasicSemanticsTests(unittest.TestCase):
    def test_fresh_clock_has_empty_components(self) -> None:
        clock = VectorClock("alpha")
        self.assertEqual(clock.components(), {})
        self.assertEqual(clock.replica_id, "alpha")
        self.assertEqual(
            clock.snapshot(), {"replica_id": "alpha", "clock": {}}
        )

    def test_tick_only_advances_local_component(self) -> None:
        clock = VectorClock("alpha")
        clock.tick()
        self.assertEqual(clock.components(), {"alpha": 1})
        clock.tick()
        clock.tick(3)
        self.assertEqual(clock.components(), {"alpha": 5})

        # Components learned from peers are never advanced by a local tick.
        peer = VectorClock("beta")
        peer.tick(4)
        clock.merge(peer)
        clock.tick(2)
        self.assertEqual(clock.components(), {"alpha": 7, "beta": 4})

    def test_components_returned_dict_is_independent(self) -> None:
        clock = VectorClock("alpha")
        clock.tick(3)
        view = clock.components()
        view["alpha"] = 999
        view["ghost"] = 1
        self.assertEqual(clock.components(), {"alpha": 3})

        # Repeated calls hand out fresh mappings.
        again = clock.components()
        self.assertIsNot(view, again)
        again.clear()
        self.assertEqual(clock.components(), {"alpha": 3})

    def test_two_independently_ticking_replicas_are_concurrent(self) -> None:
        alpha = VectorClock("alpha")
        beta = VectorClock("beta")
        alpha.tick()
        beta.tick()
        self.assertEqual(alpha.compare(beta), "concurrent")
        self.assertEqual(beta.compare(alpha), "concurrent")

        # Further independent ticks stay concurrent.
        alpha.tick(3)
        beta.tick(2)
        self.assertEqual(alpha.compare(beta), "concurrent")
        self.assertEqual(beta.compare(alpha), "concurrent")

    def test_merge_then_tick_is_after(self) -> None:
        alpha = VectorClock("alpha")
        beta = VectorClock("beta")
        alpha.tick(2)
        beta.tick(1)

        # alpha absorbs beta's history, then makes new local progress.
        self.assertIs(alpha.merge(beta), alpha)
        self.assertEqual(alpha.compare(beta), "after")
        self.assertEqual(beta.compare(alpha), "before")

        alpha.tick()
        self.assertEqual(alpha.components(), {"alpha": 3, "beta": 1})
        self.assertEqual(alpha.compare(beta), "after")
        self.assertEqual(beta.compare(alpha), "before")

    def test_equal_clocks(self) -> None:
        alpha = VectorClock("alpha")
        beta = VectorClock("beta")
        self.assertEqual(alpha.compare(beta), "equal")

        alpha.tick(2)
        beta.tick(2)
        # Still concurrent while each only knows its own component...
        self.assertEqual(alpha.compare(beta), "concurrent")
        # ...but after exchanging state they are equal.
        alpha.merge(beta)
        beta.merge(alpha)
        self.assertEqual(alpha.compare(beta), "equal")
        self.assertEqual(beta.compare(alpha), "equal")

    def test_compare_treats_missing_components_as_zero(self) -> None:
        alpha = VectorClock("alpha")
        beta = VectorClock("beta")
        alpha.tick()
        # beta has never ticked and knows nothing of alpha.
        self.assertEqual(alpha.compare(beta), "after")
        self.assertEqual(beta.compare(alpha), "before")

    def test_merge_keeps_replica_id_and_leaves_other_untouched(self) -> None:
        alpha = VectorClock("alpha")
        beta = VectorClock("beta")
        alpha.tick(2)
        beta.tick(3)
        beta_before = beta.snapshot()

        result = alpha.merge(beta)
        self.assertIs(result, alpha)
        self.assertEqual(alpha.replica_id, "alpha")
        self.assertEqual(beta.replica_id, "beta")
        self.assertEqual(beta.snapshot(), beta_before)
        self.assertEqual(beta.components(), {"beta": 3})

    def test_merge_tolerates_self_and_equivalent_copy(self) -> None:
        clock = VectorClock("alpha")
        clock.tick(2)
        before = clock.snapshot()
        self.assertIs(clock.merge(clock), clock)
        self.assertEqual(clock.snapshot(), before)

        clock.merge(wire_restore(clock.snapshot()))
        self.assertEqual(clock.snapshot(), before)


class ConvergenceTests(unittest.TestCase):
    """Duplicate, out-of-order, relayed and batched delivery."""

    def setUp(self) -> None:
        alpha = VectorClock("alpha")
        beta = VectorClock("beta")
        gamma = VectorClock("gamma")

        alpha.tick(2)
        beta.tick(1)
        gamma.tick(5)

        self.history = [cap(alpha), cap(beta), cap(gamma)]

        # beta relays alpha's state onward before ticking again.
        beta.merge(wire_restore(alpha.snapshot()))
        beta.tick(3)
        alpha2 = wire_restore(alpha.snapshot())
        alpha2.tick(4)
        gamma.tick(2)
        self.history.extend([cap(beta), cap(alpha2), cap(gamma)])

        # gamma relays both beta and alpha state, then makes local progress.
        gamma.merge(wire_restore(beta.snapshot()))
        gamma.merge(wire_restore(alpha2.snapshot()))
        gamma.tick(1)
        self.history.append(cap(gamma))

        self.finals = [cap(alpha2), cap(beta), cap(gamma)]
        self.expected = {"alpha": 6, "beta": 4, "gamma": 8}

    def assertConverged(self, clock: VectorClock) -> None:
        self.assertEqual(clock.components(), self.expected)
        self.assertEqual(clock.snapshot()["clock"], self.expected)
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
        for index, order in enumerate(permutations(self.finals)):
            paths.append(list(order) + [fa, fa, fg])
        for index, path in enumerate(paths):
            with self.subTest(path=index):
                receiver = deliver(VectorClock("observer"), path)
                self.assertConverged(receiver)

    def test_interleaved_and_stale_history_converges(self) -> None:
        h = self.history
        paths = [
            h,
            list(reversed(h)),
            [h[0], h[6], h[5], h[1], h[4], h[2], h[3]],
            self.finals + h,
            h + self.finals,
        ]
        for index, path in enumerate(paths):
            with self.subTest(path=index):
                receiver = deliver(VectorClock("observer"), path)
                self.assertConverged(receiver)

    def test_state_relayed_through_intermediate_replica(self) -> None:
        # gamma's snapshot already carries alpha/beta components relayed
        # through beta; merging only gamma must teach the observer all of it.
        observer = VectorClock("observer")
        observer.merge(wire_restore(self.finals[2]))
        self.assertConverged(observer)

        # A receiver that only saw the early relay (beta carrying alpha)
        # converges once the remaining finals arrive, in any order.
        late = deliver(VectorClock("late"), [self.history[3]])
        self.assertEqual(late.components(), {"alpha": 2, "beta": 4})
        for index, order in enumerate(permutations(self.finals)):
            with self.subTest(order=index):
                receiver = deliver(
                    VectorClock(f"node-{index}"), [self.history[3], *order]
                )
                self.assertConverged(receiver)
        self.assertEqual(late.components(), {"alpha": 2, "beta": 4})

    def test_batched_delivery_between_replicas_converges(self) -> None:
        fa, fb, fg = self.finals
        x = deliver(VectorClock("x"), [fa])
        y = deliver(VectorClock("y"), [fb, fg])
        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertConverged(x)
        self.assertConverged(y)

        p = deliver(VectorClock("p"), [fa, fb])
        q = deliver(VectorClock("q"), [fb, fg])
        r = deliver(VectorClock("r"), [fg, fa])
        p.merge(wire_restore(q.snapshot()))
        r.merge(wire_restore(p.snapshot()))
        q.merge(wire_restore(r.snapshot()))
        for clock in (p, q, r):
            self.assertConverged(clock)

    def test_converged_clocks_agree_on_comparisons(self) -> None:
        receivers = []
        paths = [
            self.finals,
            list(reversed(self.finals)) + [self.finals[1]],
            self.history,
            list(reversed(self.history)),
        ]
        for index, path in enumerate(paths):
            receivers.append(deliver(VectorClock(f"node-{index}"), path))

        for clock in receivers:
            self.assertConverged(clock)
        first, *rest = receivers
        for clock in rest:
            self.assertEqual(first.compare(clock), "equal")
            self.assertEqual(clock.compare(first), "equal")
        self.assertEqual(
            [c.replica_id for c in receivers],
            ["node-0", "node-1", "node-2", "node-3"],
        )

    def test_redelivery_of_old_state_changes_nothing(self) -> None:
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

    def test_merge_algebra_groupings(self) -> None:
        s1 = VectorClock("a")
        s1.tick(2)
        s2 = VectorClock("b")
        s2.merge(wire_restore(s1.snapshot()))
        s2.tick(3)
        s3 = VectorClock("c")
        s3.tick(4)

        def merged(owner_id: str, ordered) -> VectorClock:
            receiver = VectorClock(owner_id)
            for state in ordered:
                receiver.merge(
                    VectorClock.from_snapshot(copy.deepcopy(state.snapshot()))
                )
            return receiver

        baseline = merged("baseline", [s1, s2, s3])
        for index, order in enumerate(permutations([s1, s2, s3])):
            with self.subTest(order=index):
                receiver = merged(f"recv-{index}", order)
                self.assertEqual(
                    receiver.snapshot()["clock"], baseline.snapshot()["clock"]
                )

        left = merged("left", [s1, s2])
        left.merge(VectorClock.from_snapshot(copy.deepcopy(s3.snapshot())))
        inner = merged("inner", [s2, s3])
        right = VectorClock("right")
        right.merge(VectorClock.from_snapshot(copy.deepcopy(s1.snapshot())))
        right.merge(
            VectorClock.from_snapshot(copy.deepcopy(inner.snapshot()))
        )
        self.assertEqual(
            left.snapshot()["clock"], baseline.snapshot()["clock"]
        )
        self.assertEqual(
            right.snapshot()["clock"], baseline.snapshot()["clock"]
        )
        self.assertEqual(left.compare(right), "equal")


class SnapshotTests(unittest.TestCase):
    def test_snapshots_are_independent_and_json_serializable(self) -> None:
        clock = VectorClock("alpha")
        clock.tick(3)
        peer = VectorClock("beta")
        peer.tick(2)
        clock.merge(peer)

        first = clock.snapshot()
        second = clock.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["clock"], second["clock"])
        self.assertEqual(first, second)

        self.assertEqual(
            json.loads(json.dumps(first)),
            {"replica_id": "alpha", "clock": {"alpha": 3, "beta": 2}},
        )

        first["replica_id"] = "hacked"
        first["clock"]["alpha"] = 999
        first["clock"]["ghost"] = 1
        first["unexpected"] = True
        self.assertEqual(clock.components(), {"alpha": 3, "beta": 2})
        self.assertEqual(
            clock.snapshot(),
            {"replica_id": "alpha", "clock": {"alpha": 3, "beta": 2}},
        )
        self.assertEqual(
            second,
            {"replica_id": "alpha", "clock": {"alpha": 3, "beta": 2}},
        )

    def test_snapshot_keys_are_stably_ordered(self) -> None:
        clock = VectorClock("zeta")
        for replica, amount in [("zeta", 1), ("alpha", 2), ("mid", 3)]:
            peer = VectorClock(replica)
            peer.tick(amount)
            clock.merge(peer)
        self.assertEqual(
            list(clock.snapshot()["clock"]), ["alpha", "mid", "zeta"]
        )
        # JSON preserves insertion order of the snapshot dict.
        self.assertEqual(
            list(json.loads(json.dumps(clock.snapshot()))["clock"]),
            ["alpha", "mid", "zeta"],
        )

    def test_from_snapshot_preserves_state_and_continues_ticking(self) -> None:
        source = VectorClock("alpha")
        source.tick(5)
        beta = VectorClock("beta")
        beta.tick(2)
        source.merge(beta)

        restored = wire_restore(source.snapshot())
        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(
            restored.snapshot()["clock"], {"alpha": 5, "beta": 2}
        )

        restored.tick(4)
        self.assertEqual(
            restored.snapshot()["clock"], {"alpha": 9, "beta": 2}
        )
        # The source is untouched by the restore or later ticks.
        self.assertEqual(source.components(), {"alpha": 5, "beta": 2})

        # Restored progress is causally after every predecessor and agrees
        # after a re-merge.
        self.assertEqual(restored.compare(source), "after")
        source.merge(restored)
        self.assertEqual(source.compare(restored), "equal")

    def test_restored_clock_does_not_alias_input_dict(self) -> None:
        data = {"replica_id": "alpha", "clock": {"alpha": 5, "beta": 2}}
        restored = VectorClock.from_snapshot(data)

        data["replica_id"] = "mutated"
        data["clock"]["alpha"] = 500
        data["clock"]["gamma"] = 9

        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(
            restored.snapshot()["clock"], {"alpha": 5, "beta": 2}
        )

    def test_zero_components_are_legal(self) -> None:
        data = {"replica_id": "p", "clock": {"p": 0, "q": 0}}
        restored = VectorClock.from_snapshot(json.loads(json.dumps(data)))
        self.assertEqual(restored.components(), {"p": 0, "q": 0})
        restored.tick(3)
        self.assertEqual(restored.snapshot()["clock"], {"p": 3, "q": 0})

        peer = VectorClock("q")
        peer.tick(8)
        restored.merge(peer)
        self.assertEqual(restored.snapshot()["clock"], {"p": 3, "q": 8})

    def test_large_components_round_trip(self) -> None:
        big = 10**30
        clock = VectorClock("alpha")
        clock.tick(big)
        clock.tick(big)
        restored = wire_restore(clock.snapshot())
        self.assertEqual(restored.components()["alpha"], 2 * big)

    def test_snapshot_after_reconnect_scenario(self) -> None:
        # Offline alpha and beta diverge; after exchange both must serialize
        # identical clock payloads even though their replica ids differ.
        alpha = VectorClock("alpha")
        beta = VectorClock("beta")
        alpha.tick(3)
        beta.tick(2)
        alpha.merge(beta)
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(
            alpha.snapshot()["clock"], beta.snapshot()["clock"]
        )
        self.assertEqual(alpha.compare(beta), "equal")


class ValidationTests(unittest.TestCase):
    """The TypeError / ValueError contract, with failed calls atomic."""

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

    def test_from_snapshot_requires_dict(self) -> None:
        for bad in (None, [], "x", 42, 3.14, {1, 2}, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    VectorClock.from_snapshot(bad)

    def test_from_snapshot_key_set_must_be_exact(self) -> None:
        bad_snapshots = [
            {},
            {"replica_id": "a"},
            {"clock": {}},
            {"replica_id": "a", "clock": {}, "extra": 1},
            {"id": "a", "clock": {}},
            {"replica_id": "a", "components": {}},
        ]
        for bad in bad_snapshots:
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
                self.assertEqual(
                    restored.snapshot()["clock"],
                    {k: v for k, v in sorted(snapshot["clock"].items())},
                )

    def test_failed_merge_leaves_both_clocks_untouched(self) -> None:
        alpha = VectorClock("alpha")
        alpha.tick(2)
        beta = VectorClock("beta")
        beta.tick(3)
        before_a = alpha.snapshot()
        before_b = beta.snapshot()

        for bad in (None, 1, "alpha", [], {}, alpha.snapshot()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    alpha.merge(bad)
                with self.assertRaises(TypeError):
                    beta.merge(bad)

        self.assertEqual(alpha.snapshot(), before_a)
        self.assertEqual(beta.snapshot(), before_b)

    def test_failed_from_snapshot_leaves_input_untouched(self) -> None:
        good = VectorClock("alpha")
        good.tick(1)
        for bad in [
            {"replica_id": "a", "clock": {"a": -1}},
            {"replica_id": "", "clock": {}},
            {"replica_id": 1, "clock": {}},
            {"replica_id": "a", "clock": {"": 1}},
            {"replica_id": "a", "clock": []},
            {"replica_id": "a"},
        ]:
            with self.subTest(bad=bad):
                original = copy.deepcopy(bad)
                with self.assertRaises((TypeError, ValueError)):
                    VectorClock.from_snapshot(bad)
                self.assertEqual(bad, original)
        # Sanity: a valid clock still works.
        self.assertEqual(good.components(), {"alpha": 1})


class ExportTests(unittest.TestCase):
    def test_package_level_export(self) -> None:
        import crdt_sync

        self.assertIs(crdt_sync.VectorClock, VectorClock)
        self.assertIn("VectorClock", crdt_sync.__all__)
        # Existing exports are unchanged.
        for name in ("GCounter", "ORSet", "LWWRegister", "RGA", "__version__"):
            self.assertTrue(hasattr(crdt_sync, name))


if __name__ == "__main__":
    unittest.main()
