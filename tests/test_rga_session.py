"""Systematic tests for the public RGASession contract.

Coverage:
* construction with a non-empty ``replica_id``; an empty sequence and an empty
  causal clock to start, mirroring the RGA / VectorClock id validation;
* ``insert`` / ``delete`` / ``values`` semantics identical to RGA, with the
  local vector clock advancing exactly once per *successful* mutation and not
  at all for a rejected (bad type / out of range) call, plus independent
  ``values()`` lists;
* ``snapshot`` packages the session, RGA and vector clock under one shared
  ``replica_id`` as independent, JSON-round-trippable exchange packages;
* ``from_snapshot`` restores all three as a unit and lets editing continue
  with node ids and clock components that never move backward or collide;
* ``merge`` classification (before / after / equal / concurrent) against the
  VectorClock orientation, receiver keeps its id, the argument is untouched,
  and interleaved offline inserts/deletes plus stale, fresh, duplicated and
  reordered packets converge ``values()``, RGA shared records and clock
  components on every replica; re-merging a received state is a snapshot
  no-op;
* atomic failure: non-``RGASession`` -> ``TypeError``; same node id bound to a
  different value/predecessor -> ``ValueError`` (the RGA convention), leaving
  sequence and clock as they were;
* ``from_snapshot`` failure contract: non-dict -> ``TypeError``; missing or
  extra fields, invalid nested snapshots (including non-dict), or disagreeing
  ``replica_id`` values -> ``ValueError`` with no partially restored object;
* snapshot deep isolation through a standard JSON encode/decode boundary;
* ``RGASession`` importable from the package top level while the published
  surfaces of RGA, VectorClock, GCounter, ORSet and LWWRegister are unchanged.

Only the public API is used; no private attributes are relied upon.
"""

from __future__ import annotations

import copy
import json
import unittest
from itertools import permutations

from crdt_sync import GCounter, ORSet, RGA, RGASession, VectorClock


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def wire_restore(snapshot: dict) -> RGASession:
    """Restore a session snapshot after a full JSON round trip (wire)."""
    return RGASession.from_snapshot(json.loads(json.dumps(snapshot)))


def cap(session: RGASession) -> dict:
    """A detached JSON-shape copy of a session's current package."""
    return json.loads(json.dumps(session.snapshot()))


def deliver(receiver: RGASession, packages) -> RGASession:
    """Merge a sequence of session package dicts into ``receiver``."""
    for package in packages:
        receiver.merge(wire_restore(package))
    return receiver


def shared_rga_records(snapshot: dict) -> tuple[str, str]:
    """Replica-independent RGA causal state: node records and tombstones."""
    return (
        json.dumps(snapshot["rga"]["nodes"], sort_keys=True),
        json.dumps(snapshot["rga"]["tombstones"], sort_keys=True),
    )


def clock_components(snapshot: dict) -> dict:
    return snapshot["clock"]["clock"]


def build_offline_scenario():
    """Return ``(history, finals, expected_values, expected_clock)``.

    Three sessions edit offline and exchange only complete packages:

    * alpha builds ``a b c d`` (four local ticks);
    * beta observes alpha, then tombstones ``b``, reinserts ``B`` there and
      inserts ``β0`` at the head (three local ticks);
    * alpha, still offline, tombstones ``c`` and inserts ``A1`` after ``a``
      and ``e`` at the tail (three more local ticks);
    * gamma starts fresh with ``g`` (one local tick).
    """
    alpha = RGASession("alpha")
    alpha.insert(0, "a")
    alpha.insert(1, "b")
    alpha.insert(2, "c")
    alpha.insert(3, "d")
    history = [cap(alpha)]

    beta = RGASession("beta")
    beta.merge(wire_restore(alpha.snapshot()))
    gamma = RGASession("gamma")
    gamma.insert(0, "g")
    history.extend([cap(beta), cap(gamma)])

    beta2 = wire_restore(beta.snapshot())
    beta2.delete(1)  # tombstone b
    beta2.insert(1, "B")
    beta2.insert(0, "β0")
    history.append(cap(beta2))

    alpha.delete(2)  # tombstone c; d stays visible
    alpha.insert(1, "A1")
    alpha.insert(4, "e")
    history.extend([cap(alpha), cap(gamma)])

    finals = [cap(alpha), cap(beta2), cap(gamma)]
    expected_values = ["β0", "g", "a", "B", "A1", "d", "e"]
    expected_clock = {"alpha": 7, "beta": 3, "gamma": 1}
    return history, finals, expected_values, expected_clock


# ---------------------------------------------------------------------------
# Construction and local editing
# ---------------------------------------------------------------------------


class ConstructionTests(unittest.TestCase):
    def test_fresh_session_is_empty_under_its_replica_id(self) -> None:
        session = RGASession("alpha")
        self.assertEqual(session.replica_id, "alpha")
        self.assertEqual(session.values(), [])
        self.assertEqual(
            session.snapshot(),
            {
                "replica_id": "alpha",
                "rga": {
                    "replica_id": "alpha",
                    "counter": 0,
                    "nodes": [],
                    "tombstones": [],
                },
                "clock": {"replica_id": "alpha", "clock": {}},
            },
        )

    def test_replica_id_validation_matches_rga_and_clock(self) -> None:
        for bad in (None, 1, 1.0, b"a", [], ("a",), True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    RGASession(bad)
        with self.assertRaises(ValueError):
            RGASession("")


class InsertDeleteValuesTests(unittest.TestCase):
    def test_insert_and_values_match_rga(self) -> None:
        session = RGASession("r")
        session.insert(0, "a")
        session.insert(1, "b")
        session.insert(0, "c")
        self.assertEqual(session.values(), ["c", "a", "b"])

        reference = RGA("r")
        reference.insert(0, "a")
        reference.insert(1, "b")
        reference.insert(0, "c")
        self.assertEqual(session.values(), reference.values())

    def test_delete_returns_removed_string_and_removes_it(self) -> None:
        session = RGASession("r")
        for value in ("a", "b", "c"):
            session.insert(len(session.values()), value)
        self.assertEqual(session.delete(1), "b")
        self.assertEqual(session.values(), ["a", "c"])
        self.assertEqual(session.delete(0), "a")
        self.assertEqual(session.delete(0), "c")
        self.assertEqual(session.values(), [])

    def test_empty_string_values_round_trip(self) -> None:
        session = RGASession("r")
        session.insert(0, "")
        session.insert(0, "x")
        self.assertEqual(session.values(), ["x", ""])
        self.assertEqual(session.delete(1), "")
        self.assertEqual(session.values(), ["x"])

    def test_values_returns_an_independent_list(self) -> None:
        session = RGASession("r")
        session.insert(0, "a")
        session.insert(1, "b")
        first = session.values()
        first.clear()
        first.append("ghost")
        self.assertEqual(session.values(), ["a", "b"])
        second = session.values()
        self.assertIsNot(second, first)
        second[0] = "mutated"
        self.assertEqual(session.values(), ["a", "b"])


class ClockAdvancesOnceTests(unittest.TestCase):
    def test_successful_insert_and_delete_each_tick_once(self) -> None:
        session = RGASession("alpha")
        session.insert(0, "a")
        self.assertEqual(clock_components(session.snapshot()), {"alpha": 1})
        session.insert(1, "b")
        session.insert(2, "c")
        self.assertEqual(clock_components(session.snapshot()), {"alpha": 3})
        session.delete(0)
        self.assertEqual(clock_components(session.snapshot()), {"alpha": 4})
        session.delete(0)
        session.delete(0)
        self.assertEqual(clock_components(session.snapshot()), {"alpha": 6})

    def test_failed_insert_does_not_advance_clock_or_change_sequence(
        self,
    ) -> None:
        session = RGASession("alpha")
        session.insert(0, "ok")
        before = session.snapshot()

        for bad_index in (1.0, "1", None, [1], True, False):
            with self.subTest(bad_index=bad_index):
                with self.assertRaises(TypeError):
                    session.insert(bad_index, "x")
        for bad_value in (None, 1, 1.0, b"a", [], True, False):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(TypeError):
                    session.insert(0, bad_value)
        for out_of_range in (-1, 2, 100):
            with self.subTest(out_of_range=out_of_range):
                with self.assertRaises(IndexError):
                    session.insert(out_of_range, "x")

        self.assertEqual(session.snapshot(), before)
        self.assertEqual(clock_components(session.snapshot()), {"alpha": 1})
        self.assertEqual(session.values(), ["ok"])

    def test_failed_delete_does_not_advance_clock_or_change_sequence(
        self,
    ) -> None:
        empty = RGASession("alpha")
        empty_before = empty.snapshot()
        for bad_index in (1.0, "1", None, [1], True, False):
            with self.subTest(bad_index=bad_index):
                with self.assertRaises(TypeError):
                    empty.delete(bad_index)
        for out_of_range in (0, -1, 1):
            with self.subTest(out_of_range=out_of_range):
                with self.assertRaises(IndexError):
                    empty.delete(out_of_range)
        self.assertEqual(empty.snapshot(), empty_before)

        session = RGASession("alpha")
        session.insert(0, "ok")
        before = session.snapshot()
        for out_of_range in (1, 2, -1):
            with self.subTest(out_of_range=out_of_range):
                with self.assertRaises(IndexError):
                    session.delete(out_of_range)
        self.assertEqual(session.snapshot(), before)
        self.assertEqual(clock_components(session.snapshot()), {"alpha": 1})


# ---------------------------------------------------------------------------
# Snapshot / from_snapshot
# ---------------------------------------------------------------------------


class SnapshotShapeTests(unittest.TestCase):
    def test_snapshot_bundles_three_states_under_one_replica_id(self) -> None:
        session = RGASession("alpha")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(0)

        snapshot = session.snapshot()
        self.assertEqual(set(snapshot.keys()), {"replica_id", "rga", "clock"})
        self.assertEqual(snapshot["replica_id"], "alpha")
        self.assertEqual(snapshot["rga"]["replica_id"], "alpha")
        self.assertEqual(snapshot["clock"]["replica_id"], "alpha")
        # The nested snapshots are exactly the RGA / clock package formats.
        self.assertEqual(
            snapshot["rga"],
            {
                "replica_id": "alpha",
                "counter": 2,
                "nodes": [
                    {"id": [1, "alpha"], "value": "a", "prev": None},
                    {"id": [2, "alpha"], "value": "b", "prev": [1, "alpha"]},
                ],
                "tombstones": [[1, "alpha"]],
            },
        )
        self.assertEqual(
            snapshot["clock"],
            {"replica_id": "alpha", "clock": {"alpha": 3}},
        )

    def test_snapshot_is_json_serializable_and_canonical(self) -> None:
        session = RGASession("alpha")
        session.insert(0, "a")
        session.merge(RGASession("beta"))  # observing a peer adds no component
        encoded = json.dumps(session.snapshot())
        self.assertEqual(json.loads(encoded), session.snapshot())

    def test_snapshots_are_deeply_independent_of_each_other_and_session(
        self,
    ) -> None:
        session = RGASession("alpha")
        session.insert(0, "a")
        first = session.snapshot()
        second = session.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["rga"], second["rga"])
        self.assertIsNot(first["clock"], second["clock"])
        self.assertIsNot(first["rga"]["nodes"], second["rga"]["nodes"])
        self.assertEqual(first, second)

        first["replica_id"] = "hacked"
        first["rga"]["replica_id"] = "hacked"
        first["rga"]["counter"] = 999
        first["rga"]["nodes"].append(
            {"id": [9, "ghost"], "value": "g", "prev": None}
        )
        first["clock"]["replica_id"] = "hacked"
        first["clock"]["clock"]["ghost"] = 5
        first["unexpected"] = True

        self.assertEqual(session.snapshot(), second)
        self.assertEqual(session.values(), ["a"])
        self.assertEqual(clock_components(session.snapshot()), {"alpha": 1})


class FromSnapshotTests(unittest.TestCase):
    def test_restore_preserves_sequence_and_clock(self) -> None:
        source = RGASession("alpha")
        source.insert(0, "a")
        source.insert(1, "b")
        peer = RGASession("beta")
        peer.insert(0, "c")
        source.merge(peer)
        source.delete(1)  # tombstone a in the merged weave c,a,b

        data = json.loads(json.dumps(source.snapshot()))
        restored = RGASession.from_snapshot(data)
        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.values(), ["c", "b"])
        # alpha made three local changes (insert a, insert b, delete a) and
        # absorbed beta's single insert: {alpha:3, beta:1}.
        self.assertEqual(
            clock_components(restored.snapshot()),
            {"alpha": 3, "beta": 1},
        )

    def test_round_trip_is_canonical_through_json(self) -> None:
        source = RGASession("alpha")
        source.insert(0, "a")
        source.merge(RGASession("zeta"))
        wire = json.loads(json.dumps(source.snapshot()))
        restored = RGASession.from_snapshot(wire)
        self.assertEqual(restored.snapshot(), wire)

    def test_restore_does_not_alias_the_input_package(self) -> None:
        data = {
            "replica_id": "alpha",
            "rga": {
                "replica_id": "alpha",
                "counter": 1,
                "nodes": [
                    {"id": [1, "alpha"], "value": "a", "prev": None}
                ],
                "tombstones": [],
            },
            "clock": {"replica_id": "alpha", "clock": {"alpha": 1}},
        }
        restored = RGASession.from_snapshot(data)
        data["replica_id"] = "mutated"
        data["rga"]["nodes"][0]["value"] = "mutated"
        data["clock"]["clock"]["alpha"] = 999
        self.assertEqual(restored.replica_id, "alpha")
        self.assertEqual(restored.values(), ["a"])
        self.assertEqual(clock_components(restored.snapshot()), {"alpha": 1})

    def test_restored_session_keeps_emitting_forward_only_ids_and_ticks(
        self,
    ) -> None:
        source = RGASession("alpha")
        source.insert(0, "a")
        source.insert(1, "b")
        source.delete(0)
        restored = wire_restore(source.snapshot())  # counter 2, alpha clock 3

        restored.insert(0, "a")
        self.assertEqual(restored.values(), ["a", "b"])
        snapshot = restored.snapshot()
        # The new node id is (3, alpha): no reuse of seq 1 or 2 ...
        origins = sorted(
            node["id"][0]
            for node in snapshot["rga"]["nodes"]
            if node["id"][1] == "alpha"
        )
        self.assertEqual(origins, [1, 2, 3])
        # ... and the clock component only moves forward (3 -> 4).
        self.assertEqual(clock_components(snapshot), {"alpha": 4})

        other = RGASession("beta")
        other.merge(source)
        other.merge(wire_restore(restored.snapshot()))
        restored.merge(wire_restore(other.snapshot()))
        self.assertEqual(other.values(), ["a", "b"])
        self.assertEqual(restored.values(), ["a", "b"])

    def test_restored_session_catches_up_after_observing_a_peer(self) -> None:
        live = RGASession("alpha")
        live.insert(0, "a")
        live.insert(1, "b")
        backup = wire_restore(live.snapshot())

        live.insert(2, "c")  # node (3, alpha), clock alpha 3
        backup.merge(wire_restore(live.snapshot()))
        backup.insert(3, "d")  # must be node (4, alpha), clock 4

        everyone = RGASession("observer")
        everyone.merge(live)
        everyone.merge(backup)
        self.assertEqual(everyone.values(), ["a", "b", "c", "d"])
        origins = sorted(
            node["id"][0]
            for node in everyone.snapshot()["rga"]["nodes"]
            if node["id"][1] == "alpha"
        )
        self.assertEqual(origins, [1, 2, 3, 4])
        self.assertEqual(clock_components(backup.snapshot()), {"alpha": 4})


# ---------------------------------------------------------------------------
# merge: relation, identity, argument isolation
# ---------------------------------------------------------------------------


class MergeRelationTests(unittest.TestCase):
    def test_fresh_sessions_are_equal(self) -> None:
        a = RGASession("a")
        b = RGASession("b")
        self.assertEqual(a.merge(b), "equal")
        self.assertEqual(
            VectorClock.from_snapshot(a.snapshot()["clock"]).compare(
                VectorClock.from_snapshot(b.snapshot()["clock"])
            ),
            "equal",
        )

    def test_independent_edits_are_concurrent(self) -> None:
        a = RGASession("a")
        b = RGASession("b")
        a.insert(0, "a")
        b.insert(0, "b")
        # Freeze both packages before either merge mutates a clock, then ask
        # two independent VectorClocks for the relation.
        a_clock = VectorClock.from_snapshot(cap(a)["clock"])
        b_clock = VectorClock.from_snapshot(cap(b)["clock"])
        self.assertEqual(a_clock.compare(b_clock), "concurrent")
        self.assertEqual(b_clock.compare(a_clock), "concurrent")

        self.assertEqual(a.merge(b), "concurrent")

        # Multiple rounds of purely local offline progress stay concurrent.
        # Both sides keep ticking without exchanging; compare fresh restored
        # copies of each package so one direction's merge can't perturb the
        # other's fork point.
        a_pkg = cap(a)  # {a:1, b:1} after the first exchange
        a_fork = wire_restore(a_pkg)
        a_fork.insert(1, "a2")  # {a:2, b:1}
        b_fork = wire_restore(cap(b))  # {b:1}, still untouched
        b_fork.insert(1, "b2")  # {b:2}
        a_fork_pkg = cap(a_fork)
        b_fork_pkg = cap(b_fork)
        self.assertEqual(
            wire_restore(a_fork_pkg).merge(wire_restore(b_fork_pkg)),
            "concurrent",
        )
        self.assertEqual(
            wire_restore(b_fork_pkg).merge(wire_restore(a_fork_pkg)),
            "concurrent",
        )

    def test_receiver_with_strictly_more_history_is_after(self) -> None:
        # b observes a, then makes no local progress: a then edits once more.
        a = RGASession("a")
        a.insert(0, "a")
        a.insert(1, "a2")
        b = RGASession("b")
        b.merge(wire_restore(a.snapshot()))  # b knows exactly a's old state
        a.insert(2, "a3")
        self.assertEqual(a.merge(b), "after")
        self.assertEqual(b.merge(a), "before")

    def test_observing_then_idle_is_before(self) -> None:
        a = RGASession("a")
        a.insert(0, "a")  # a: {a:1}
        b = RGASession("b")  # b: {} — has seen nothing
        # An empty receiver happens-before the advanced peer it absorbs.
        self.assertEqual(b.merge(a), "before")
        # b adopted a's clock, so re-merging the same state is now equal.
        self.assertEqual(b.merge(a), "equal")

        # Symmetrically, an advanced session merging a fresh empty peer sees
        # itself strictly after it.
        advanced = wire_restore(cap(a))
        self.assertEqual(advanced.merge(RGASession("c")), "after")

    def test_merge_then_local_progress_is_strictly_after(self) -> None:
        a = RGASession("a")
        a.insert(0, "a")
        b = RGASession("b")
        b.merge(wire_restore(cap(a)))  # b observes exactly a: {a:1}
        a.insert(1, "a2")  # a advances to {a:2}, beyond what b knows
        self.assertEqual(a.merge(b), "after")
        self.assertEqual(b.merge(a), "before")

    def test_converged_sessions_compare_equal(self) -> None:
        a = RGASession("a")
        b = RGASession("b")
        a.insert(0, "x")
        b.insert(0, "y")
        a.merge(wire_restore(b.snapshot()))
        b.merge(wire_restore(a.snapshot()))
        self.assertEqual(a.merge(b), "equal")
        self.assertEqual(b.merge(a), "equal")

    def test_merge_keeps_receiver_replica_id_everywhere(self) -> None:
        receiver = RGASession("receiver")
        receiver.insert(0, "r")
        other = RGASession("other")
        other.insert(0, "o")
        receiver.merge(other)
        self.assertEqual(receiver.replica_id, "receiver")
        snapshot = receiver.snapshot()
        self.assertEqual(snapshot["replica_id"], "receiver")
        self.assertEqual(snapshot["rga"]["replica_id"], "receiver")
        self.assertEqual(snapshot["clock"]["replica_id"], "receiver")

    def test_merge_does_not_modify_argument(self) -> None:
        receiver = RGASession("receiver")
        receiver.insert(0, "r")
        other = RGASession("other")
        other.insert(0, "o")
        other.delete(0)
        other.insert(0, "o2")
        before = other.snapshot()
        receiver.merge(other)
        self.assertEqual(other.snapshot(), before)
        self.assertEqual(other.values(), ["o2"])


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class ConvergenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history, self.finals, self.expected_values, self.expected_clock = (
            build_offline_scenario()
        )

    def assertConverged(self, session: RGASession) -> None:
        self.assertEqual(session.values(), self.expected_values)
        snapshot = session.snapshot()
        self.assertEqual(clock_components(snapshot), self.expected_clock)
        # The converged package itself survives a JSON wire round trip.
        again = wire_restore(snapshot)
        self.assertEqual(again.values(), self.expected_values)
        self.assertEqual(clock_components(again.snapshot()), self.expected_clock)

    def test_all_permutations_of_final_packages_converge(self) -> None:
        for index, order in enumerate(permutations(self.finals)):
            with self.subTest(order=index):
                receiver = deliver(RGASession(f"observer-{index}"), order)
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
                receiver = deliver(RGASession("observer"), path)
                self.assertConverged(receiver)

    def test_interleaved_and_stale_packages_converge(self) -> None:
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
                receiver = deliver(RGASession("observer"), path)
                self.assertConverged(receiver)

    def test_all_replicas_share_values_records_and_clock_after_flood(
        self,
    ) -> None:
        fa, fb, fg = self.finals
        x = deliver(RGASession("x"), [fa])
        y = deliver(RGASession("y"), [fb])
        z = deliver(RGASession("z"), [fg])

        # Repeated bidirectional full-package flood until quiescence.
        for _ in range(4):
            for left, right in ((x, y), (y, z), (x, z), (z, x), (y, x)):
                left.merge(wire_restore(right.snapshot()))
                right.merge(wire_restore(left.snapshot()))

        for replica in (x, y, z):
            self.assertEqual(replica.values(), self.expected_values)
        snaps = [replica.snapshot() for replica in (x, y, z)]
        # RGA shared records (nodes + tombstones) agree ...
        records = {shared_rga_records(snap) for snap in snaps}
        self.assertEqual(len(records), 1)
        # ... the adopted Lamport counters agree after full exchange ...
        self.assertEqual(len({snap["rga"]["counter"] for snap in snaps}), 1)
        # ... and the causal clock components agree on every replica, while
        # each session keeps its own replica_id.
        for snap in snaps:
            self.assertEqual(clock_components(snap), self.expected_clock)
        self.assertEqual(
            [snap["replica_id"] for snap in snaps], ["x", "y", "z"]
        )

    def test_stale_packages_cannot_change_a_converged_state(self) -> None:
        receiver = deliver(RGASession("observer"), self.finals)
        before = receiver.snapshot()
        deliver(receiver, self.history * 2)
        self.assertEqual(receiver.snapshot(), before)
        self.assertConverged(receiver)

    def test_re_merging_received_state_keeps_snapshot_identical(self) -> None:
        receiver = deliver(RGASession("observer"), self.finals)
        before = receiver.snapshot()

        # Same live object, JSON-restored copies, repeatedly, any sender.
        for package in self.finals:
            restored = wire_restore(package)
            receiver.merge(restored)
            receiver.merge(wire_restore(package))
        receiver.merge(receiver)
        self.assertEqual(receiver.snapshot(), before)

        # And a converged peer merged back is an equal-relation no-op.
        peer = wire_restore(receiver.snapshot())
        relation = receiver.merge(peer)
        self.assertEqual(relation, "equal")
        self.assertEqual(receiver.snapshot(), before)

    def test_multi_round_bidirectional_reconnect_then_more_edits(self) -> None:
        fa, fb, _fg = self.finals
        x = deliver(RGASession("x"), [fa])
        y = deliver(RGASession("y"), [fb, self.finals[2]])

        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertConverged(x)
        self.assertConverged(y)

        # Keep editing on both sides while disconnected, then reconnect.
        x.insert(0, "x-head")
        y.insert(len(y.values()), "y-tail")
        x.merge(wire_restore(y.snapshot()))
        y.merge(wire_restore(x.snapshot()))
        self.assertEqual(x.values(), y.values())
        self.assertEqual(
            shared_rga_records(x.snapshot()), shared_rga_records(y.snapshot())
        )
        self.assertEqual(
            clock_components(x.snapshot()), clock_components(y.snapshot())
        )
        self.assertIn("x-head", x.values())
        self.assertIn("y-tail", x.values())

    def test_relations_during_offline_reconnect(self) -> None:
        alpha = RGASession("alpha")
        for value in ("a", "b", "c", "d"):
            alpha.insert(len(alpha.values()), value)

        beta = RGASession("beta")
        self.assertEqual(beta.merge(wire_restore(alpha.snapshot())), "before")

        # Both sides then edit offline from the shared fork: beta deletes b
        # and inserts B and β0; alpha deletes c and appends e.
        beta.delete(1)
        beta.insert(1, "B")
        beta.insert(0, "β0")
        alpha.delete(2)
        alpha.insert(len(alpha.values()), "e")

        # Freeze each side's divergent package before either is delivered, so
        # both comparisons see the same fork point.
        alpha_pkg = cap(alpha)
        beta_pkg = cap(beta)

        # Each side has edits the other lacks, so the first exchange is
        # concurrent in both directions.
        self.assertEqual(
            alpha.merge(wire_restore(beta_pkg)), "concurrent"
        )
        self.assertEqual(
            beta.merge(wire_restore(alpha_pkg)), "concurrent"
        )
        # After both exchanges they carry the same causal knowledge: equal.
        self.assertEqual(
            alpha.merge(wire_restore(beta.snapshot())), "equal"
        )
        self.assertEqual(alpha.values(), beta.values())

    def test_converged_receivers_keep_their_own_replica_ids(self) -> None:
        paths = [
            self.finals,
            list(reversed(self.finals)) + [self.finals[1]],
            self.history,
            list(reversed(self.history)),
        ]
        receivers = [
            deliver(RGASession(f"node-{index}"), path)
            for index, path in enumerate(paths)
        ]
        for receiver in receivers:
            self.assertEqual(receiver.values(), self.expected_values)
            self.assertEqual(
                clock_components(receiver.snapshot()), self.expected_clock
            )
        self.assertEqual(
            [r.replica_id for r in receivers],
            ["node-0", "node-1", "node-2", "node-3"],
        )


# ---------------------------------------------------------------------------
# Atomic failure semantics
# ---------------------------------------------------------------------------


def _package(replica_id, nodes, clock, counter=None, tombstones=None):
    if counter is None:
        counter = max((node["id"][0] for node in nodes), default=0)
    return {
        "replica_id": replica_id,
        "rga": {
            "replica_id": replica_id,
            "counter": counter,
            "nodes": nodes,
            "tombstones": tombstones or [],
        },
        "clock": {"replica_id": replica_id, "clock": clock},
    }


class MergeFailureTests(unittest.TestCase):
    def test_merge_requires_an_rga_session(self) -> None:
        session = RGASession("alpha")
        session.insert(0, "a")
        foreign_clock = VectorClock("alpha")
        foreign_clock.tick()
        for bad in (
            None,
            1,
            "alpha",
            [],
            {},
            session.snapshot(),
            json.loads(json.dumps(session.snapshot())),
            RGA("alpha"),
            VectorClock("alpha"),
            foreign_clock,
            GCounter("alpha"),
            ORSet("alpha"),
        ):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    session.merge(bad)
        self.assertEqual(session.values(), ["a"])
        self.assertEqual(clock_components(session.snapshot()), {"alpha": 1})

    def test_same_node_different_value_raises_and_changes_nothing(self) -> None:
        receiver = RGASession.from_snapshot(
            _package("alpha", [{"id": [1, "alpha"], "value": "x",
                                "prev": None}], {"alpha": 1})
        )
        other = RGASession.from_snapshot(
            _package("alpha", [{"id": [1, "alpha"], "value": "y",
                                "prev": None}], {"alpha": 1})
        )
        before_receiver = receiver.snapshot()
        before_other = other.snapshot()
        with self.assertRaises(ValueError):
            receiver.merge(other)
        self.assertEqual(receiver.snapshot(), before_receiver)
        self.assertEqual(other.snapshot(), before_other)
        self.assertEqual(receiver.values(), ["x"])
        self.assertEqual(other.values(), ["y"])

    def test_same_node_different_predecessor_raises_and_changes_nothing(
        self,
    ) -> None:
        receiver = RGASession.from_snapshot(
            _package("alpha", [{"id": [1, "alpha"], "value": "x",
                                "prev": None}], {"alpha": 1})
        )
        other_nodes = [
            {"id": [1, "alpha"], "value": "x", "prev": [7, "zeta"]},
            {"id": [7, "zeta"], "value": "z", "prev": None},
        ]
        other = RGASession.from_snapshot(
            _package("alpha", other_nodes, {"alpha": 1, "zeta": 1})
        )
        before = receiver.snapshot()
        with self.assertRaises(ValueError):
            receiver.merge(other)
        self.assertEqual(receiver.snapshot(), before)
        self.assertEqual(receiver.values(), ["x"])
        # Clock must not have advanced or adopted the peer either.
        self.assertEqual(clock_components(receiver.snapshot()), {"alpha": 1})
        self.assertEqual(other.values(), ["z", "x"])

    def test_same_node_same_record_is_not_a_conflict(self) -> None:
        node = [{"id": [1, "alpha"], "value": "x", "prev": None}]
        receiver = RGASession.from_snapshot(
            _package("alpha", node, {"alpha": 1})
        )
        other = RGASession.from_snapshot(
            _package("alpha", copy.deepcopy(node), {"alpha": 1})
        )
        receiver.merge(other)
        self.assertEqual(receiver.values(), ["x"])

    def test_conflict_leaves_clock_even_when_relations_would_merge(self) -> None:
        # Receiver strictly behind the peer, yet the RGA conflict must abort
        # the whole call so the clock join is also rolled back (never applied).
        receiver = RGASession.from_snapshot(
            _package("alpha", [{"id": [1, "alpha"], "value": "x",
                                "prev": None}], {"alpha": 1})
        )
        other = RGASession.from_snapshot(
            _package(
                "alpha",
                [
                    {"id": [1, "alpha"], "value": "DIFFERENT", "prev": None},
                    {"id": [2, "alpha"], "value": "n", "prev": [1, "alpha"]},
                ],
                {"alpha": 2},
            )
        )
        before = receiver.snapshot()
        with self.assertRaises(ValueError):
            receiver.merge(other)
        self.assertEqual(receiver.snapshot(), before)
        self.assertEqual(receiver.values(), ["x"])
        self.assertEqual(clock_components(receiver.snapshot()), {"alpha": 1})


# ---------------------------------------------------------------------------
# from_snapshot validation
# ---------------------------------------------------------------------------


class FromSnapshotValidationTests(unittest.TestCase):
    GOOD_RGA = {
        "replica_id": "a",
        "counter": 1,
        "nodes": [{"id": [1, "a"], "value": "x", "prev": None}],
        "tombstones": [],
    }
    GOOD_CLOCK = {"replica_id": "a", "clock": {"a": 1}}

    def good_package(self, **overrides) -> dict:
        package = {
            "replica_id": "a",
            "rga": copy.deepcopy(self.GOOD_RGA),
            "clock": copy.deepcopy(self.GOOD_CLOCK),
        }
        package.update(overrides)
        return package

    def test_requires_dict(self) -> None:
        for bad in (None, [], "x", 42, 3.14, {1, 2}, True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    RGASession.from_snapshot(bad)

    def test_key_set_must_be_exact(self) -> None:
        good = self.good_package()
        bad_packages = [
            {},
            {"replica_id": "a"},
            {"replica_id": "a", "rga": self.GOOD_RGA},
            {"replica_id": "a", "clock": self.GOOD_CLOCK},
            {**good, "extra": 1},
            {"id": "a", "rga": self.GOOD_RGA, "clock": self.GOOD_CLOCK},
            {
                "replica_id": "a",
                "sequence": self.GOOD_RGA,
                "clock": self.GOOD_CLOCK,
            },
        ]
        for bad in bad_packages:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RGASession.from_snapshot(copy.deepcopy(bad))

    def test_top_level_replica_id_must_be_non_empty_string(self) -> None:
        for bad_id in ("", None, 0, True, False, [], {}):
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(ValueError):
                    RGASession.from_snapshot(self.good_package(replica_id=bad_id))

    def test_invalid_nested_rga_raises_value_error(self) -> None:
        # A nested RGA package that violates the RGA snapshot contract.
        bad_rga_cases = [
            [],  # not a dict -> normalized to ValueError
            None,
            {**self.GOOD_RGA, "counter": -1},
            {**self.GOOD_RGA, "extra": 1},
            {
                **self.GOOD_RGA,
                "nodes": [{"id": [1, "a"], "value": 3, "prev": None}],
            },
            {
                **self.GOOD_RGA,
                "nodes": [
                    {"id": [1, "a"], "value": "x", "prev": None},
                    {"id": [1, "a"], "value": "y", "prev": None},
                ],
            },
            {**self.GOOD_RGA, "tombstones": [[9, "a"]]},
        ]
        for bad_rga in bad_rga_cases:
            with self.subTest(bad_rga=bad_rga):
                with self.assertRaises(ValueError):
                    RGASession.from_snapshot(self.good_package(rga=bad_rga))

    def test_invalid_nested_clock_raises_value_error(self) -> None:
        bad_clock_cases = [
            [],  # not a dict -> normalized to ValueError
            None,
            {"replica_id": "a", "clock": {"a": -1}},
            {"replica_id": "a", "clock": {1: 1}},
            {"replica_id": "a"},
            {"replica_id": "a", "clock": {}, "extra": 1},
        ]
        for bad_clock in bad_clock_cases:
            with self.subTest(bad_clock=bad_clock):
                with self.assertRaises(ValueError):
                    RGASession.from_snapshot(
                        self.good_package(clock=bad_clock)
                    )

    def test_three_replica_ids_must_agree(self) -> None:
        rga_b = {**copy.deepcopy(self.GOOD_RGA), "replica_id": "b"}
        clock_b = {"replica_id": "b", "clock": {"a": 1}}

        mismatched = [
            # top says a, rga says b
            {"replica_id": "a", "rga": rga_b,
             "clock": copy.deepcopy(self.GOOD_CLOCK)},
            # top says a, clock says b
            {"replica_id": "a", "rga": copy.deepcopy(self.GOOD_RGA),
             "clock": clock_b},
            # rga and clock agree with each other (b) but disagree with top
            {"replica_id": "a", "rga": copy.deepcopy(rga_b),
             "clock": {"replica_id": "b", "clock": {"a": 1}}},
        ]
        for bad in mismatched:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    RGASession.from_snapshot(copy.deepcopy(bad))

    def test_no_partial_object_is_returned(self) -> None:
        # Every malformed package raises; a valid package still restores fine,
        # proving validation failures do not poison any shared/default state.
        for bad in (
            self.good_package(clock={"replica_id": "a", "clock": {"a": -1}}),
            self.good_package(rga=[]),
        ):
            with self.assertRaises(ValueError):
                RGASession.from_snapshot(bad)
        restored = RGASession.from_snapshot(self.good_package())
        self.assertEqual(restored.values(), ["x"])
        self.assertEqual(clock_components(restored.snapshot()), {"a": 1})

    def test_empty_session_package_round_trips_and_keeps_editing(self) -> None:
        fresh = RGASession("zero")
        package = json.loads(json.dumps(fresh.snapshot()))
        restored = RGASession.from_snapshot(package)
        self.assertEqual(restored.values(), [])
        self.assertEqual(clock_components(restored.snapshot()), {})
        restored.insert(0, "later")
        self.assertEqual(restored.values(), ["later"])
        self.assertEqual(clock_components(restored.snapshot()), {"zero": 1})


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


class PackageSurfaceTests(unittest.TestCase):
    def test_rga_session_is_top_level_importable(self) -> None:
        import crdt_sync

        self.assertIs(crdt_sync.RGASession, RGASession)

    def test_existing_package_exports_are_unchanged(self) -> None:
        # The baseline scope contract remains exactly as published.
        import crdt_sync

        self.assertEqual(
            set(crdt_sync.__all__),
            {
                "GCounter",
                "ORSet",
                "LWWRegister",
                "RGA",
                "VectorClock",
                "__version__",
            },
        )

    def test_rga_and_vector_clock_still_work_standalone(self) -> None:
        rga = RGA("a")
        rga.insert(0, "x")
        clock = VectorClock("a")
        clock.tick()
        self.assertEqual(rga.values(), ["x"])
        self.assertEqual(clock.components(), {"a": 1})

    def test_repr_does_not_dump_internals(self) -> None:
        session = RGASession("r")
        session.insert(0, "a")
        text = repr(session)
        self.assertIn("RGASession", text)
        self.assertIn("r", text)
        self.assertIn("a", text)


if __name__ == "__main__":
    unittest.main()
