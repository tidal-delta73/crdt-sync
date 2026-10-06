"""Systematic tests for the public GCounter contract.

Coverage:
* convergence under duplicated, reordered, batched and interleaved delivery,
  including snapshot round trips across a JSON boundary;
* merge algebra: idempotence, commutativity, associativity, receiver return
  value and argument isolation;
* snapshot / from_snapshot independence and JSON serialization;
* zero-value replicas, explicit zero components, varying replica counts and
  large positive increments;
* the documented TypeError / ValueError input contract;
* unchanged ``version`` / ``help`` CLI behavior.

Only the public API is used (``GCounter``, its methods, ``replica_id`` and
``__version__``); no private attributes, dict iteration order, or randomness
is relied upon.
"""

from __future__ import annotations

import copy
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from itertools import permutations

from crdt_sync import GCounter, __version__
from crdt_sync.__main__ import main


def wire_restore(snapshot: dict) -> GCounter:
    """Restore a snapshot after a full JSON round trip (simulated wire)."""
    return GCounter.from_snapshot(json.loads(json.dumps(snapshot)))


def deliver(receiver: GCounter, snapshots) -> GCounter:
    """Merge a sequence of snapshot dicts into ``receiver`` in order."""
    for snapshot in snapshots:
        receiver.merge(wire_restore(snapshot))
    return receiver


def build_scenario():
    """Return ``(history, finals, expected_counts, expected_value)``.

    Three replicas (alpha, beta, gamma) run through several increment rounds;
    some replicas continue from restored snapshots. ``history`` contains
    intermediate snapshots (oldest first), ``finals`` the last snapshot of
    each replica.
    """

    def cap(counter: GCounter) -> dict:
        return json.loads(json.dumps(counter.snapshot()))

    alpha = GCounter("alpha")
    beta = GCounter("beta")
    gamma = GCounter("gamma")

    alpha.increment(3)
    beta.increment(1)
    gamma.increment(7)

    history = [cap(alpha), cap(beta), cap(gamma)]

    # Continue alpha from a restored snapshot, proving a round-tripped replica
    # can keep making local progress.
    alpha2 = wire_restore(alpha.snapshot())
    alpha2.increment(2)
    gamma.increment(4)
    history.extend([cap(alpha2), cap(gamma)])

    beta2 = wire_restore(beta.snapshot())
    beta2.increment(10)
    history.append(cap(beta2))

    finals = [cap(alpha2), cap(beta2), cap(gamma)]
    expected_counts = {"alpha": 5, "beta": 11, "gamma": 11}
    return history, finals, expected_counts, 27


class ConvergenceTests(unittest.TestCase):
    """Duplicate, out-of-order, batched and interleaved delivery."""

    def setUp(self) -> None:
        self.history, self.finals, self.expected_counts, self.expected_value = (
            build_scenario()
        )

    def assertConverged(self, counter: GCounter) -> None:
        self.assertEqual(counter.value(), self.expected_value)
        self.assertEqual(counter.snapshot()["counts"], self.expected_counts)
        # The converged state must survive a JSON wire round trip unchanged.
        self.assertEqual(
            json.loads(json.dumps(counter.snapshot()))["counts"],
            self.expected_counts,
        )

    def test_all_permutations_of_final_states_converge(self) -> None:
        for index, order in enumerate(permutations(self.finals)):
            with self.subTest(order=index):
                receiver = GCounter(f"observer-{index}")
                deliver(receiver, order)
                self.assertConverged(receiver)

    def test_duplicated_delivery_converges(self) -> None:
        fa, fb, fg = self.finals
        duplicate_paths = [
            [fa, fa, fb, fg, fb, fg, fa],
            [fg, fb, fa, fa, fa],
            list(self.finals) * 3,
        ]
        for index, order in enumerate(permutations(self.finals)):
            duplicate_paths.append(
                list(order) + [self.finals[0], self.finals[0], self.finals[2]]
            )
        for index, path in enumerate(duplicate_paths):
            with self.subTest(path=index):
                receiver = deliver(GCounter("observer"), path)
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
                receiver = deliver(GCounter("observer"), path)
                self.assertConverged(receiver)

    def test_receiver_sharing_a_known_replica_id_converges(self) -> None:
        # A receiver whose own id is alpha starts at component 0; merging the
        # alpha state must take it to the observed maximum.
        receiver = GCounter("alpha")
        deliver(receiver, reversed(self.history))
        self.assertConverged(receiver)
        self.assertEqual(receiver.replica_id, "alpha")

    def test_batched_delivery_between_replicas_converges(self) -> None:
        fa, fb, fg = self.finals

        # Split the same states into disjoint batches, then exchange batches.
        x = deliver(GCounter("x"), [fa])
        y = deliver(GCounter("y"), [fb, fg])
        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertConverged(x)
        self.assertConverged(y)
        self.assertEqual(x.snapshot()["counts"], y.snapshot()["counts"])

        # Three-way split merged in a ring, with overlapping batches.
        p = deliver(GCounter("p"), [fa, fb])
        q = deliver(GCounter("q"), [fb, fg])
        r = deliver(GCounter("r"), [fg, fa])
        p.merge(wire_restore(q.snapshot()))
        r.merge(wire_restore(p.snapshot()))
        q.merge(wire_restore(r.snapshot()))
        for counter in (p, q, r):
            self.assertConverged(counter)

    def test_redelivery_of_old_or_same_state_changes_nothing(self) -> None:
        receiver = deliver(GCounter("observer"), self.finals)
        self.assertConverged(receiver)

        before = receiver.snapshot()
        stale_and_duplicate = [
            self.history[0],
            self.history[1],
            self.finals[0],
            self.finals[0],
            self.finals[2],
        ]
        deliver(receiver, stale_and_duplicate)

        self.assertEqual(receiver.snapshot(), before)
        self.assertEqual(receiver.value(), self.expected_value)

    def test_all_converged_receivers_agree_regardless_of_path(self) -> None:
        receivers = []
        paths = [
            self.finals,
            list(reversed(self.finals)) + [self.finals[1]],
            self.history,
            list(reversed(self.history)),
        ]
        for index, path in enumerate(paths):
            receivers.append(deliver(GCounter(f"node-{index}"), path))

        snapshots = [c.snapshot() for c in receivers]
        for snapshot in snapshots[1:]:
            self.assertEqual(snapshot["counts"], snapshots[0]["counts"])
        # replica_id is per receiver and must not be rewritten by merges.
        self.assertEqual(
            [s["replica_id"] for s in snapshots],
            ["node-0", "node-1", "node-2", "node-3"],
        )


class MergeAlgebraTests(unittest.TestCase):
    """Idempotence, commutativity, associativity and object semantics."""

    def setUp(self) -> None:
        s1 = GCounter("a")
        s1.increment(2)

        # s2 also knows about a, so states partially overlap.
        s2 = GCounter("b")
        s2.merge(wire_restore(s1.snapshot()))
        s2.increment(3)

        s3 = GCounter("c")
        s3.increment(4)

        self.states = [s1, s2, s3]

    def merge_order(self, owner_id: str, ordered) -> GCounter:
        receiver = GCounter(owner_id)
        for state in ordered:
            # Merge an independent copy built through the public snapshot API.
            receiver.merge(
                GCounter.from_snapshot(copy.deepcopy(state.snapshot()))
            )
        return receiver

    def test_merge_returns_receiver(self) -> None:
        receiver = GCounter("recv")
        self.assertIs(receiver.merge(self.states[0]), receiver)

    def test_merge_does_not_modify_argument(self) -> None:
        for state in self.states:
            before = state.snapshot()
            receiver = GCounter("recv")
            receiver.merge(state)
            self.assertEqual(state.snapshot(), before)
            self.assertEqual(state.value(), sum(before["counts"].values()))

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
        # The equivalent argument reports its own unchanged state.
        self.assertEqual(duplicate.snapshot()["counts"], before["counts"])

    def test_commutative_orderings(self) -> None:
        baseline = self.merge_order("baseline", self.states)
        for index, order in enumerate(permutations(self.states)):
            with self.subTest(order=index):
                receiver = self.merge_order(f"recv-{index}", order)
                self.assertEqual(
                    receiver.snapshot()["counts"],
                    baseline.snapshot()["counts"],
                )

    def test_associative_groupings(self) -> None:
        s1, s2, s3 = self.states

        # (s1 ⊔ s2) ⊔ s3
        left = self.merge_order("left", [s1, s2])
        left.merge(
            GCounter.from_snapshot(copy.deepcopy(s3.snapshot()))
        )

        # s1 ⊔ (s2 ⊔ s3)
        right_inner = self.merge_order("inner", [s2, s3])
        right = GCounter("right")
        right.merge(GCounter.from_snapshot(copy.deepcopy(s1.snapshot())))
        right.merge(
            GCounter.from_snapshot(copy.deepcopy(right_inner.snapshot()))
        )

        flat = self.merge_order("flat", [s1, s2, s3])
        self.assertEqual(left.snapshot()["counts"], flat.snapshot()["counts"])
        self.assertEqual(right.snapshot()["counts"], flat.snapshot()["counts"])
        self.assertEqual(left.value(), right.value())


class SnapshotIsolationTests(unittest.TestCase):
    def test_snapshots_are_independent_and_json_serializable(self) -> None:
        counter = GCounter("alpha")
        counter.increment(3)

        first = counter.snapshot()
        second = counter.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["counts"], second["counts"])
        self.assertEqual(first, second)

        # Every snapshot must round-trip through JSON unchanged.
        self.assertEqual(
            json.loads(json.dumps(first)),
            {"replica_id": "alpha", "counts": {"alpha": 3}},
        )

        # Mutating a returned snapshot (top level and nested) must not leak back.
        first["replica_id"] = "hacked"
        first["counts"]["alpha"] = 999
        first["counts"]["ghost"] = 1
        first["unexpected"] = True
        self.assertEqual(counter.value(), 3)
        self.assertEqual(
            counter.snapshot(),
            {"replica_id": "alpha", "counts": {"alpha": 3}},
        )
        self.assertEqual(second, {"replica_id": "alpha", "counts": {"alpha": 3}})

    def test_from_snapshot_preserves_id_and_components(self) -> None:
        source = GCounter("alpha")
        source.increment(5)
        other = GCounter("beta")
        other.increment(2)
        source.merge(other)

        data = json.loads(json.dumps(source.snapshot()))
        restored = GCounter.from_snapshot(data)

        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(
            restored.snapshot()["counts"], {"alpha": 5, "beta": 2}
        )
        self.assertEqual(restored.value(), 7)

        # Local increments only move the restored replica's own component.
        restored.increment(4)
        self.assertEqual(
            restored.snapshot()["counts"], {"alpha": 9, "beta": 2}
        )
        self.assertEqual(restored.value(), 11)

    def test_restored_counter_does_not_alias_source_dict(self) -> None:
        data = {"replica_id": "alpha", "counts": {"alpha": 5, "beta": 2}}
        restored = GCounter.from_snapshot(data)

        data["replica_id"] = "mutated"
        data["counts"]["alpha"] = 500
        data["counts"]["gamma"] = 9

        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(
            restored.snapshot()["counts"], {"alpha": 5, "beta": 2}
        )

    def test_zero_value_replica(self) -> None:
        zero = GCounter("zero")
        self.assertEqual(zero.value(), 0)
        self.assertEqual(
            zero.snapshot(), {"replica_id": "zero", "counts": {}}
        )
        json.dumps(zero.snapshot())

        # Merging a zero replica into a populated counter changes nothing.
        populated = GCounter("alpha")
        populated.increment(4)
        before = populated.snapshot()
        populated.merge(zero)
        self.assertEqual(populated.snapshot(), before)

        # The zero replica receives full state without changing the argument.
        zero.merge(populated)
        self.assertEqual(zero.value(), 4)
        self.assertEqual(zero.snapshot()["counts"], {"alpha": 4})
        self.assertEqual(populated.value(), 4)

    def test_explicit_zero_components_are_legal(self) -> None:
        data = {"replica_id": "p", "counts": {"p": 0, "q": 0}}
        restored = GCounter.from_snapshot(json.loads(json.dumps(data)))
        self.assertEqual(restored.value(), 0)
        self.assertEqual(restored.replica_id, "p")
        self.assertEqual(restored.snapshot()["counts"], {"p": 0, "q": 0})

        restored.increment(3)
        self.assertEqual(restored.snapshot()["counts"], {"p": 3, "q": 0})

        # Zero components must not erase a peer's positive component.
        peer = GCounter("q")
        peer.increment(8)
        restored.merge(peer)
        self.assertEqual(restored.snapshot()["counts"], {"p": 3, "q": 8})

    def test_varying_replica_counts_and_increments(self) -> None:
        amounts = [1, 2, 7, 1000]
        for n in range(1, 5):
            with self.subTest(n=n):
                replicas = []
                expected = {}
                total = 0
                for i in range(n):
                    replica = GCounter(f"node-{i}")
                    replica.increment(amounts[i])
                    replica.increment()  # second local round
                    replicas.append(replica)
                    expected[f"node-{i}"] = amounts[i] + 1
                    total += amounts[i] + 1

                # Merge directly and, independently, via JSON snapshots.
                direct = GCounter("direct")
                over_wire = GCounter("wire")
                for replica in replicas:
                    direct.merge(replica)
                    over_wire.merge(wire_restore(replica.snapshot()))

                self.assertEqual(direct.value(), total)
                self.assertEqual(over_wire.value(), total)
                self.assertEqual(direct.snapshot()["counts"], expected)
                self.assertEqual(over_wire.snapshot()["counts"], expected)

    def test_large_positive_integers(self) -> None:
        big = 10**30
        counter = GCounter("alpha")
        counter.increment(big)
        counter.increment(big)
        self.assertEqual(counter.value(), 2 * big)
        restored = wire_restore(counter.snapshot())
        self.assertEqual(restored.value(), 2 * big)
        self.assertEqual(restored.snapshot()["counts"]["alpha"], 2 * big)


class ValidationTests(unittest.TestCase):
    """The current TypeError / ValueError contract must hold."""

    def test_replica_id_validation(self) -> None:
        for bad in (None, 1, 1.0, b"a", [], ("a",), True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    GCounter(bad)
        with self.assertRaises(ValueError):
            GCounter("")

    def test_increment_validation(self) -> None:
        counter = GCounter("alpha")
        counter.increment(1)

        for bad in (True, False, 1.0, "1", None, [1], 1j):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    counter.increment(bad)
        for bad in (0, -1, -1000):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    counter.increment(bad)

        # A rejected increment must leave the counter untouched.
        self.assertEqual(counter.value(), 1)
        self.assertEqual(counter.snapshot()["counts"], {"alpha": 1})

    def test_merge_requires_a_gcounter(self) -> None:
        counter = GCounter("alpha")
        for bad in (None, 1, "alpha", [], {}, counter.snapshot()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    counter.merge(bad)

    def test_from_snapshot_requires_dict(self) -> None:
        for bad in (None, [], "x", 42, 3.14, {1, 2}, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    GCounter.from_snapshot(bad)

    def test_from_snapshot_key_set_must_be_exact(self) -> None:
        bad_snapshots = [
            {},
            {"replica_id": "a"},
            {"counts": {}},
            {"replica_id": "a", "counts": {}, "extra": 1},
            {"id": "a", "counts": {}},
            {"replica_id": "a", "state": {}},
        ]
        for bad in bad_snapshots:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    GCounter.from_snapshot(bad)

    def test_from_snapshot_replica_id_validation(self) -> None:
        for bad_id in ("", None, 0, True, False, [], {}):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(ValueError):
                    GCounter.from_snapshot(
                        {"replica_id": bad_id, "counts": {}}
                    )

    def test_from_snapshot_counts_must_be_dict(self) -> None:
        for bad_counts in (None, [], "", 42, (), {1, 2}):
            with self.subTest(bad_counts=bad_counts):
                with self.assertRaises(ValueError):
                    GCounter.from_snapshot(
                        {"replica_id": "a", "counts": bad_counts}
                    )

    def test_from_snapshot_component_keys_validation(self) -> None:
        for bad_counts in ({"": 1}, {1: 1}, {None: 1}, {True: 1}, {"": 0}):
            with self.subTest(bad_counts=bad_counts):
                with self.assertRaises(ValueError):
                    GCounter.from_snapshot(
                        {"replica_id": "a", "counts": bad_counts}
                    )

    def test_from_snapshot_component_values_validation(self) -> None:
        for bad_value in (True, False, -1, -(10**9), 1.0, "1", None, [1]):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(ValueError):
                    GCounter.from_snapshot(
                        {"replica_id": "a", "counts": {"a": bad_value}}
                    )

    def test_from_snapshot_accepts_valid_shapes(self) -> None:
        valid = [
            {"replica_id": "a", "counts": {}},
            {"replica_id": "a", "counts": {"a": 0}},
            {"replica_id": "a", "counts": {"b": 0, "a": 1}},
            {"replica_id": "x", "counts": {"x": 10**18}},
        ]
        for snapshot in valid:
            with self.subTest(snapshot=snapshot):
                restored = GCounter.from_snapshot(copy.deepcopy(snapshot))
                self.assertEqual(restored.replica_id, snapshot["replica_id"])
                self.assertEqual(restored.snapshot()["counts"], snapshot["counts"])


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
