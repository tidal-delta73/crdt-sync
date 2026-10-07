"""Tests for RGASession: an RGA replica plus its vector clock as one unit."""

from __future__ import annotations

import copy
import json
import unittest

from crdt_sync import RGA, RGASession, VectorClock
from crdt_sync.rga_session import _RETIRED


def wire(snapshot: object) -> object:
    """Round-trip a snapshot through JSON, as a real network boundary would."""
    return json.loads(json.dumps(snapshot))


def normalized(snapshot: object) -> str:
    """Canonical JSON fingerprint of a snapshot."""
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"))


def restore(snapshot: object) -> RGASession:
    return RGASession.from_snapshot(wire(snapshot))


def frontier(session: RGASession) -> VectorClock:
    """A stability frontier equal to the session's observed clock."""
    clock = VectorClock("frontier")
    clock.merge(session._clock)
    return clock


def is_skeleton(session: RGASession, node_id: tuple[int, str]) -> bool:
    record = session._rga._nodes.get(node_id)
    return record is not None and record[0] is _RETIRED


def shared_view(session: RGASession) -> object:
    """Everything about a session snapshot except its own replica ids."""
    snapshot = session.snapshot()
    rga = {key: value for key, value in snapshot["rga"].items()
           if key != "replica_id"}
    clock = snapshot["clock"]["clock"]
    return {"rga": rga, "clock": clock, "compaction": snapshot.get("compaction")}



class ConstructionTests(unittest.TestCase):
    def test_fresh_session_is_empty_with_empty_clock(self) -> None:
        session = RGASession("A")
        self.assertEqual(session.replica_id, "A")
        self.assertEqual(session.values(), [])
        self.assertEqual(
            session.snapshot(),
            {
                "replica_id": "A",
                "rga": RGA("A").snapshot(),
                "clock": VectorClock("A").snapshot(),
            },
        )

    def test_replica_id_must_be_a_non_empty_string(self) -> None:
        for bad in (None, 1, 1.5, ["A"], {"id": "A"}):
            with self.subTest(bad=bad):
                self.assertRaises(TypeError, RGASession, bad)
        self.assertRaises(ValueError, RGASession, "")


class LocalEditTests(unittest.TestCase):
    def test_insert_delete_values_match_rga_semantics(self) -> None:
        session = RGASession("A")
        session.insert(0, "he")
        session.insert(1, "llo")
        session.insert(1, "l")
        self.assertEqual(session.values(), ["he", "l", "llo"])
        self.assertEqual(session.delete(1), "l")
        self.assertEqual(session.values(), ["he", "llo"])

    def test_each_successful_edit_ticks_the_clock_once(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(0)
        clock = session.snapshot()["clock"]
        self.assertEqual(clock, {"replica_id": "A", "clock": {"A": 3}})

    def test_failed_edits_change_neither_sequence_nor_clock(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        before = session.snapshot()
        for call in (
            lambda: session.insert("0", "x"),      # non-integer index
            lambda: session.insert(True, "x"),     # boolean index
            lambda: session.insert(0, 1),          # non-string value
            lambda: session.insert(-1, "x"),       # index out of range
            lambda: session.insert(2, "x"),        # index out of range
            lambda: session.delete("0"),           # non-integer index
            lambda: session.delete(1),             # index out of range
            lambda: session.delete(-1),            # index out of range
        ):
            self.assertRaises((TypeError, IndexError), call)
        self.assertEqual(session.values(), ["a"])
        self.assertEqual(session.snapshot(), before)

    def test_delete_on_empty_session_raises_and_keeps_clock_empty(self) -> None:
        session = RGASession("A")
        self.assertRaises(IndexError, session.delete, 0)
        self.assertEqual(session.snapshot()["clock"]["clock"], {})

    def test_values_returns_an_independent_list(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        result = session.values()
        result.append("intruder")
        result[0] = "mutated"
        self.assertEqual(session.values(), ["a"])


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_json_round_trippable_and_names_one_replica(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(0)
        snapshot = session.snapshot()
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)
        self.assertEqual(snapshot["replica_id"], "A")
        self.assertEqual(snapshot["rga"]["replica_id"], "A")
        self.assertEqual(snapshot["clock"]["replica_id"], "A")

    def test_snapshots_are_deeply_independent(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        first = session.snapshot()
        second = session.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["rga"], second["rga"])
        self.assertIsNot(first["clock"], second["clock"])
        # Mutating every level of a snapshot leaves the session untouched.
        first["rga"]["nodes"].append({"id": [9, "Z"], "value": "z", "prev": None})
        first["rga"]["tombstones"].append([9, "Z"])
        first["rga"]["counter"] = 99
        first["clock"]["clock"]["Z"] = 99
        first["replica_id"] = "Z"
        self.assertEqual(session.snapshot(), second)

    def test_from_snapshot_restores_and_editing_never_reuses_ids(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(0)
        packet = wire(session.snapshot())

        # The restored copy replaces the original and keeps editing.
        restored = RGASession.from_snapshot(packet)
        self.assertEqual(restored.values(), ["b"])
        self.assertEqual(restored.snapshot(), session.snapshot())
        restored.insert(1, "c")
        restored.delete(0)
        self.assertEqual(restored.values(), ["c"])
        # The clock continues from the restored component, never backwards.
        self.assertEqual(restored.snapshot()["clock"]["clock"], {"A": 5})
        # Post-restore inserts mint fresh ids above every pre-restore one.
        ids = [
            tuple(node["id"]) for node in restored.snapshot()["rga"]["nodes"]
        ]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn((3, "A"), ids)

        # A peer that only ever saw the pre-restore packet still merges
        # cleanly with the continued session, and both converge.
        peer = RGASession("B")
        peer.merge(RGASession.from_snapshot(packet))
        peer.merge(restore(restored.snapshot()))
        restored.merge(restore(peer.snapshot()))
        self.assertEqual(peer.values(), restored.values())
        shared = lambda s: normalized(
            {key: s.snapshot()["rga"][key] for key in ("counter", "nodes", "tombstones")}
        )
        self.assertEqual(shared(peer), shared(restored))

    def test_from_snapshot_rejects_non_dict_with_type_error(self) -> None:
        for bad in (None, [], "snapshot", 42):
            with self.subTest(bad=bad):
                self.assertRaises(TypeError, RGASession.from_snapshot, bad)

    def test_from_snapshot_rejects_missing_or_extra_fields(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        good = session.snapshot()
        for key in ("replica_id", "rga", "clock"):
            incomplete = {k: v for k, v in good.items() if k != key}
            with self.subTest(missing=key):
                self.assertRaises(
                    ValueError, RGASession.from_snapshot, wire(incomplete)
                )
        extra = {**wire(good), "extra": True}
        self.assertRaises(ValueError, RGASession.from_snapshot, extra)

    def test_from_snapshot_rejects_invalid_nested_snapshots(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        good = wire(session.snapshot())
        for nested in (None, [], "rga", 7):
            with self.subTest(nested=nested):
                bad = {**good, "rga": nested}
                self.assertRaises(ValueError, RGASession.from_snapshot, bad)
                bad = {**good, "clock": nested}
                self.assertRaises(ValueError, RGASession.from_snapshot, bad)
        # Structurally invalid nested content is rejected as ValueError too.
        bad_rga = {**good, "rga": {**good["rga"], "counter": -1}}
        self.assertRaises(ValueError, RGASession.from_snapshot, bad_rga)
        bad_clock = {**good, "clock": {**good["clock"], "clock": {"A": -1}}}
        self.assertRaises(ValueError, RGASession.from_snapshot, bad_clock)

    def test_from_snapshot_rejects_replica_id_mismatch(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        good = wire(session.snapshot())
        mismatched_top = {**good, "replica_id": "B"}
        self.assertRaises(ValueError, RGASession.from_snapshot, mismatched_top)
        mismatched_rga = {
            **good,
            "rga": {**good["rga"], "replica_id": "B"},
        }
        self.assertRaises(ValueError, RGASession.from_snapshot, mismatched_rga)
        mismatched_clock = {
            **good,
            "clock": {**good["clock"], "replica_id": "B"},
        }
        self.assertRaises(
            ValueError, RGASession.from_snapshot, mismatched_clock
        )


class MergeTests(unittest.TestCase):
    def test_merge_rejects_non_session_with_type_error(self) -> None:
        session = RGASession("A")
        for bad in (None, "B", RGA("B"), VectorClock("B"), 42):
            with self.subTest(bad=bad):
                self.assertRaises(TypeError, session.merge, bad)

    def test_merge_reports_the_peer_relation(self) -> None:
        a, b = RGASession("A"), RGASession("B")
        a.insert(0, "a")
        # B is strictly behind A's state: the peer (A) is ahead of B.
        self.assertEqual(b.merge(restore(a.snapshot())), "after")
        # Merging the already-received state again: the peer is now equal.
        self.assertEqual(b.merge(restore(a.snapshot())), "equal")
        # B edits locally; a stale peer packet is behind the receiver.
        b.insert(1, "b")
        self.assertEqual(b.merge(restore(a.snapshot())), "before")
        # Concurrent edits on both sides report concurrency.
        a.insert(1, "a2")
        self.assertEqual(b.merge(restore(a.snapshot())), "concurrent")

    def test_merge_keeps_receiver_identity_and_leaves_peer_untouched(self) -> None:
        a, b = RGASession("A"), RGASession("B")
        a.insert(0, "a")
        b.insert(0, "b")
        peer_snapshot = a.snapshot()
        b.merge(a)
        self.assertEqual(b.replica_id, "B")
        self.assertEqual(a.replica_id, "A")
        self.assertEqual(a.snapshot(), peer_snapshot)

    def test_conflicting_node_id_fails_atomically(self) -> None:
        # Two live sessions share a replica id and mint colliding node ids.
        a1, a2 = RGASession("A"), RGASession("A")
        a1.insert(0, "x")
        a2.insert(0, "y")
        receiver = RGASession("B")
        receiver.insert(0, "b")
        receiver.merge(restore(a1.snapshot()))
        before = receiver.snapshot()
        self.assertRaises(ValueError, receiver.merge, a2)
        # Neither the sequence nor the clock moved.
        self.assertEqual(receiver.snapshot(), before)
        self.assertEqual(receiver.values(), ["b", "x"])

    def test_offline_sessions_converge_under_messy_delivery(self) -> None:
        a, b, c = RGASession("A"), RGASession("B"), RGASession("C")
        a.insert(0, "a")
        common = a.snapshot()
        b.merge(restore(common))
        c.merge(restore(common))

        # Offline, interleaved edits on all three replicas.
        b.delete(0)
        c.insert(1, "c")
        a.insert(1, "a2")
        b.insert(0, "b")
        c.delete(0)

        # Capture packets at different times; deliver old and new packets in
        # arbitrary order, with duplicates.
        packets = [
            wire(c.snapshot()),
            wire(common),            # stale, pre-edit state of A
            wire(b.snapshot()),
            wire(a.snapshot()),
            wire(c.snapshot()),      # duplicate
            wire(b.snapshot()),      # duplicate
        ]
        for packet in packets:
            for receiver in (a, b, c):
                receiver.merge(restore(packet))

        expected_values = ["b", "c", "a2"]
        expected_clock = {"A": 2, "B": 2, "C": 2}
        for receiver in (a, b, c):
            snapshot = receiver.snapshot()
            self.assertEqual(receiver.values(), expected_values)
            self.assertEqual(snapshot["clock"]["clock"], expected_clock)

        # Shared RGA records and clock components are byte-identical.
        shared = [
            normalized(
                {
                    "rga": {
                        key: snapshot[key]
                        for key in ("counter", "nodes", "tombstones")
                    },
                    "clock": receiver.snapshot()["clock"]["clock"],
                }
            )
            for receiver in (a, b, c)
            for snapshot in [receiver.snapshot()["rga"]]
        ]
        self.assertEqual(shared[0], shared[1])
        self.assertEqual(shared[1], shared[2])

        # Redelivering already-received packets changes nothing.
        for receiver in (a, b, c):
            before = normalized(receiver.snapshot())
            for packet in packets:
                receiver.merge(restore(packet))
            self.assertEqual(normalized(receiver.snapshot()), before)

    def test_double_merge_of_current_state_is_idempotent(self) -> None:
        a, b = RGASession("A"), RGASession("B")
        a.insert(0, "x")
        b.merge(restore(a.snapshot()))
        b.insert(1, "y")
        b.delete(0)
        packet = wire(b.snapshot())
        a.merge(restore(packet))
        before = normalized(a.snapshot())
        a.merge(restore(packet))
        a.merge(restore(a.snapshot()))
        self.assertEqual(normalized(a.snapshot()), before)
        self.assertEqual(a.values(), ["y"])


class CompactValidationTests(unittest.TestCase):
    def test_stable_clock_must_be_a_vector_clock(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.delete(0)
        for bad in (None, 1, 1.5, "clock", [], {}):
            with self.subTest(bad=bad):
                self.assertRaises(TypeError, session.compact, bad)

    def test_empty_session_compacts_to_zero(self) -> None:
        self.assertEqual(RGASession("A").compact(VectorClock("S")), 0)

    def test_frontier_ahead_or_concurrent_raises_and_changes_nothing(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.delete(0)

        ahead = VectorClock("S")
        ahead.merge(session._clock)
        ahead.tick()  # claims a delete the session never observed

        concurrent = VectorClock("S")
        concurrent.merge(session._clock)
        concurrent._components["Q"] = 1
        concurrent._components["A"] = 0

        for bad in (ahead, concurrent):
            with self.subTest(bad=bad.components()):
                before = wire(session.snapshot())
                self.assertRaises(ValueError, session.compact, bad)
                self.assertEqual(session.snapshot(),
                                 RGASession.from_snapshot(before).snapshot())
                self.assertEqual(session.values(), [])

    def test_frontier_equal_or_behind_is_accepted(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.delete(0)
        self.assertEqual(session.compact(frontier(session)), 1)
        # An empty frontier is strictly behind; it is a valid no-op.
        self.assertEqual(session.compact(VectorClock("S")), 0)

    def test_compact_does_not_tick_the_clock_or_change_values(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(1)
        before_clock = session.snapshot()["clock"]
        self.assertEqual(session.compact(frontier(session)), 1)
        self.assertEqual(session.snapshot()["clock"], before_clock)
        self.assertEqual(session.values(), ["a"])

    def test_repeated_compact_with_same_frontier_returns_zero(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(1)
        stable = frontier(session)
        self.assertEqual(session.compact(stable), 1)
        self.assertEqual(session.compact(stable), 0)
        self.assertEqual(session.compact(stable), 0)

    def test_legacy_tombstones_are_never_compaction_eligible(self) -> None:
        legacy = {
            "replica_id": "A",
            "rga": {
                "replica_id": "A",
                "counter": 2,
                "nodes": [
                    {"id": [1, "A"], "value": "a", "prev": None},
                    {"id": [2, "A"], "value": "b", "prev": [1, "A"]},
                ],
                "tombstones": [[2, "A"]],
            },
            "clock": {"replica_id": "A", "clock": {"A": 2}},
        }
        session = RGASession.from_snapshot(legacy)
        self.assertEqual(session.values(), ["a"])
        # The clock knows the delete happened, but its time is not inferred:
        # even the exact observed frontier cannot retire the untagged node.
        self.assertEqual(session.compact(frontier(session)), 0)
        self.assertNotIn("compaction", session.snapshot())
        # And it still blocks the leaf-peeling rule by remaining a record.
        self.assertIn((2, "A"), session._rga._nodes)


class CompactReclamationTests(unittest.TestCase):
    def _three(self) -> RGASession:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.insert(2, "c")
        return session

    def test_leaf_tombstone_is_reclaimed_and_reported(self) -> None:
        session = self._three()
        session.delete(2)
        self.assertEqual(session.compact(frontier(session)), 1)
        self.assertEqual(session.values(), ["a", "b"])
        self.assertTrue(is_skeleton(session, (3, "A")))

    def test_continuous_deleted_branch_peels_from_the_leaf(self) -> None:
        session = self._three()
        session.delete(2)
        session.compact(frontier(session))
        session.delete(1)
        self.assertEqual(session.compact(frontier(session)), 1)
        self.assertTrue(is_skeleton(session, (2, "A")))
        self.assertTrue(is_skeleton(session, (3, "A")))
        # The surviving head stays a full, visible record.
        self.assertEqual(session._rga._nodes[(1, "A")][0], "a")
        self.assertEqual(session.values(), ["a"])

    def test_chain_deleted_together_peels_in_one_call(self) -> None:
        session = self._three()
        session.delete(2)
        session.delete(1)
        self.assertEqual(session.compact(frontier(session)), 2)
        self.assertEqual(session.values(), ["a"])

    def test_live_node_blocks_peeling_through_it(self) -> None:
        session = self._three()
        # Delete the tail, then delete the head: the live middle keeps the
        # dead head as its predecessor and the tail peel stops at the middle.
        session.delete(2)
        session.delete(0)
        self.assertEqual(session.compact(frontier(session)), 1)
        # (1,A) is still a full tombstone record: referenced by live (2,A).
        self.assertNotIn((1, "A"), session._retired_ids)
        self.assertEqual(session.values(), ["b"])
        # Deleting the middle frees the whole branch in a later compaction.
        session.delete(0)
        self.assertEqual(session.compact(frontier(session)), 2)
        self.assertEqual(session.values(), [])

    def test_unstable_delete_is_not_crossed(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(1)  # b deleted at A:2
        stable = frontier(session)
        session.insert(0, "z")
        session.delete(1)  # a deleted later at A:4, beyond the old frontier
        # The tail b was deleted first and is a leaf: only it is stable.
        self.assertEqual(session.compact(stable), 1)
        self.assertTrue(is_skeleton(session, (2, "A")))
        self.assertIn((1, "A"), session._rga._nodes)
        self.assertEqual(session.values(), ["z"])
        # Same stale frontier cannot make progress.
        self.assertEqual(session.compact(stable), 0)
        # A fresh frontier reclaims the rest.
        self.assertEqual(session.compact(frontier(session)), 1)

    def test_retired_skeleton_keeps_late_branch_in_position(self) -> None:
        base = RGASession("A")
        base.insert(0, "a")
        base.insert(1, "b")
        base.insert(2, "c")
        base.insert(3, "d")
        peer = RGASession("E")
        peer.merge(base)

        compactor = RGASession("A")
        compactor.insert(0, "a")
        compactor.insert(1, "b")
        compactor.insert(2, "c")
        compactor.insert(3, "d")
        compactor.delete(3)
        compactor.delete(2)
        self.assertEqual(compactor.compact(frontier(compactor)), 2)

        # The peer still holds full c/d records; merge demotes them but keeps
        # the visible sequence identical.
        peer.merge(compactor)
        self.assertEqual(peer.values(), ["a", "b"])
        self.assertTrue(is_skeleton(peer, (3, "A")))
        self.assertTrue(is_skeleton(peer, (4, "A")))
        self.assertFalse(is_skeleton(peer, (2, "A")))
        self.assertEqual(restore(peer.snapshot()).values(), ["a", "b"])


class CompactSnapshotTests(unittest.TestCase):
    def test_uncompacted_snapshot_keeps_the_three_field_shape(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.delete(0)
        snapshot = session.snapshot()
        self.assertEqual(set(snapshot), {"replica_id", "rga", "clock"})
        # The tagged tombstone rides inside the legacy-shaped RGA snapshot.
        self.assertEqual(
            snapshot["rga"]["tombstones"], [[[1, "A"], [["A", 2]]]]
        )

    def test_full_round_trip_is_deep_independent_and_json_safe(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(1)
        session.compact(frontier(session))
        first = session.snapshot()
        second = session.snapshot()
        self.assertIsNot(first, second)
        self.assertIsNot(first["rga"], second["rga"])
        self.assertIsNot(first["compaction"], second["compaction"])
        self.assertEqual(json.loads(json.dumps(first)), first)

        restored = restore(first)
        self.assertEqual(restored.snapshot(), first)
        self.assertEqual(restored.values(), session.values())

        # Mutating the snapshot must not touch the session.
        first["compaction"]["retired"].append("intruder")
        first["rga"]["nodes"][0]["value"] = "z"
        self.assertEqual(session.snapshot(), second)

    def test_compaction_summary_is_deterministic(self) -> None:
        def make() -> RGASession:
            session = RGASession("A")
            session.insert(0, "a")
            session.insert(1, "b")
            session.delete(1)
            session.compact(frontier(session))
            return session
        self.assertEqual(make().snapshot(), make().snapshot())

    def test_legacy_four_field_rga_snapshot_still_restores(self) -> None:
        legacy = {
            "replica_id": "A",
            "rga": {
                "replica_id": "A",
                "counter": 1,
                "nodes": [{"id": [1, "A"], "value": "a", "prev": None}],
                "tombstones": [],
            },
            "clock": {"replica_id": "A", "clock": {"A": 1}},
        }
        session = restore(legacy)
        self.assertEqual(session.values(), ["a"])
        self.assertEqual(set(session.snapshot()), {"replica_id", "rga", "clock"})
        # Continued editing follows the restored counter and clock.
        session.insert(1, "b")
        self.assertEqual(
            session.snapshot()["clock"]["clock"], {"A": 2}
        )

    def test_restored_compacted_session_edits_without_id_reuse(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(1)
        session.compact(frontier(session))
        restored = restore(session.snapshot())
        restored.insert(1, "c")
        ids = [tuple(node["id"]) for node in restored.snapshot()["rga"]["nodes"]]
        self.assertEqual(ids, [(1, "A"), (3, "A")])
        self.assertEqual(restored.values(), ["a", "c"])

    def test_deletion_dot_not_covered_by_clock_is_rejected(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.delete(0)
        good = wire(session.snapshot())
        good["rga"]["tombstones"] = [[[1, "A"], [["A", 9]]]]
        self.assertRaises(ValueError, RGASession.from_snapshot, good)

    def test_malformed_compaction_summaries_are_rejected(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.delete(0)
        session.compact(frontier(session))
        good = wire(session.snapshot())

        def corrupt(mutate) -> object:
            bad = copy.deepcopy(good)
            mutate(bad)
            return bad

        candidates = [
            lambda b: b["compaction"].pop("retired"),
            lambda b: b["compaction"].pop("retired_dots"),
            lambda b: b["compaction"].pop("retired_clock"),
            lambda b: b["compaction"].update(extra=1),
            lambda b: b["compaction"].__setitem__("retired", []),
            lambda b: b["compaction"].__setitem__("retired_dots", []),
            lambda b: b["compaction"].__setitem__("retired_clock", {}),
            lambda b: b["compaction"]["retired"][0].pop("value"),
            lambda b: b["compaction"]["retired"][0].__setitem__("prev", [9, "Z"]),
            lambda b: b["compaction"]["retired_dots"].append(["A", 9]),
            lambda b: b["compaction"]["retired_clock"].__setitem__("A", 9),
        ]
        for mutate in candidates:
            with self.subTest():
                self.assertRaises(
                    ValueError, RGASession.from_snapshot, corrupt(mutate)
                )


class CompactMergeTests(unittest.TestCase):
    def _story(self):
        compacted = RGASession("A")
        compacted.insert(0, "a")
        compacted.insert(1, "b")
        compacted.insert(2, "c")
        compacted.delete(2)
        compacted.delete(1)
        compacted.compact(frontier(compacted))
        uncompacted = RGASession("A")
        uncompacted.insert(0, "a")
        uncompacted.insert(1, "b")
        uncompacted.insert(2, "c")
        uncompacted.delete(2)
        uncompacted.delete(1)
        return compacted, uncompacted

    def test_compacted_and_uncompacted_converge_both_directions(self) -> None:
        for receiver_first, sender_second in (
            (self._story()[0], self._story()[1]),
            (self._story()[1], self._story()[0]),
        ):
            receiver = RGASession("R")
            receiver.merge(restore(receiver_first.snapshot()))
            receiver.merge(restore(sender_second.snapshot()))
            self.assertEqual(receiver.values(), ["a"])
            self.assertTrue(is_skeleton(receiver, (2, "A")))
            self.assertTrue(is_skeleton(receiver, (3, "A")))

    def test_stale_duplicate_and_out_of_order_packets_never_resurrect(self) -> None:
        compacted, uncompacted = self._story()
        old_packet = wire(uncompacted.snapshot())
        new_packet = wire(compacted.snapshot())

        receiver = RGASession("R")
        for packet in (new_packet, old_packet, new_packet, old_packet,
                       old_packet):
            receiver.merge(restore(packet))
        self.assertEqual(receiver.values(), ["a"])
        self.assertTrue(is_skeleton(receiver, (2, "A")))
        self.assertTrue(is_skeleton(receiver, (3, "A")))
        # A retired node never reappears as a full value record.
        for node_id in ((2, "A"), (3, "A")):
            self.assertTrue(is_skeleton(receiver, node_id))

    def test_compaction_spreads_through_merge(self) -> None:
        compacted, uncompacted = self._story()
        # Uncompacted first receives a live state, then the compacted packet;
        # it adopts the retirement and its output shrinks to match.
        uncompacted.merge(restore(compacted.snapshot()))
        self.assertEqual(
            shared_view(uncompacted)["compaction"],
            shared_view(compacted)["compaction"],
        )
        self.assertEqual(uncompacted.values(), ["a"])

    def test_converged_sessions_agree_on_shared_snapshot(self) -> None:
        compacted, uncompacted = self._story()
        left = RGASession("L")
        right = RGASession("M")
        left.merge(restore(compacted.snapshot()))
        right.merge(restore(uncompacted.snapshot()))
        left.merge(restore(uncompacted.snapshot()))
        right.merge(restore(compacted.snapshot()))
        self.assertEqual(left.values(), right.values())
        self.assertEqual(shared_view(left), shared_view(right))
        self.assertNotEqual(left.replica_id, right.replica_id)

    def test_same_id_conflicting_content_still_raises_after_compaction(self) -> None:
        # Two replicas sharing an id minted colliding records with different
        # content; one side then compacts the loser. The conflict must still
        # surface, atomically.
        first = RGASession("A")
        first.insert(0, "x")
        second = RGASession("A")
        second.insert(0, "y")

        compacted = RGASession("C")
        compacted.merge(first)
        compacted.delete(0)
        self.assertEqual(compacted.compact(frontier(compacted)), 1)

        before = wire(compacted.snapshot())
        self.assertRaises(ValueError, compacted.merge, second)
        self.assertEqual(compacted.snapshot(),
                         RGASession.from_snapshot(before).snapshot())
        self.assertEqual(compacted.values(), [])

    def test_conflicting_retired_predecessor_raises_atomically(self) -> None:
        first = RGASession("A")
        first.insert(0, "x")
        first.insert(1, "y")
        holder = RGASession("H")
        holder.merge(first)

        compacted = RGASession("H")
        compacted.merge(first)
        compacted.delete(1)  # retire the tail y
        compacted.compact(frontier(compacted))

        # Forge a second compacted state claiming the retired id hung off a
        # different predecessor (root instead of x).
        forged = restore(compacted.snapshot())
        forged._retired_prev[(2, "A")] = None
        forged._rga._nodes[(2, "A")] = (_RETIRED, None)
        before = wire(holder.snapshot())
        self.assertRaises(ValueError, holder.merge, forged)
        self.assertEqual(holder.snapshot(),
                         RGASession.from_snapshot(before).snapshot())

    def test_merge_relation_still_comes_from_pre_merge_clocks(self) -> None:
        left = RGASession("A")
        left.insert(0, "a")
        left.delete(0)
        left.compact(frontier(left))

        right = RGASession("B")
        self.assertEqual(right.merge(restore(left.snapshot())), "after")
        right.insert(0, "b")
        self.assertEqual(right.merge(restore(left.snapshot())), "before")
        left.insert(0, "a2")
        self.assertEqual(right.merge(restore(left.snapshot())), "concurrent")

    def test_offline_compaction_converges_under_messy_delivery(self) -> None:
        a, b, c = RGASession("A"), RGASession("B"), RGASession("C")
        a.insert(0, "a")
        common = wire(a.snapshot())
        for peer in (b, c):
            peer.merge(restore(common))

        b.delete(0)
        c.insert(1, "c")
        a.insert(1, "a2")
        b.insert(0, "b")
        c.delete(0)

        packets = [
            wire(c.snapshot()), wire(common), wire(b.snapshot()),
            wire(a.snapshot()), wire(c.snapshot()), wire(b.snapshot()),
        ]
        for packet in packets:
            for receiver in (a, b, c):
                receiver.merge(restore(packet))

        # Fully converged: ["b", "c", "a2"]. Now A deletes the leaf a2 once
        # everyone could have observed it, and compacts against the shared
        # frontier.
        a.delete(2)
        stable = frontier(a)
        self.assertEqual(a.compact(stable), 1)

        # Re-exchange everything — including stale pre-compaction packets
        # and the new compacted one, with duplicates — in arbitrary order.
        packets = [
            wire(a.snapshot()), wire(c.snapshot()), wire(common),
            wire(b.snapshot()), wire(a.snapshot()), wire(c.snapshot()),
        ]
        for packet in packets:
            for receiver in (a, b, c):
                receiver.merge(restore(packet))

        for receiver in (a, b, c):
            self.assertEqual(receiver.values(), ["b", "c"])
        self.assertEqual(shared_view(a), shared_view(b))
        self.assertEqual(shared_view(b), shared_view(c))
        # The retired leaf never returns as a full record anywhere.
        for receiver in (a, b, c):
            self.assertTrue(is_skeleton(receiver, (2, "A")))


if __name__ == "__main__":
    unittest.main()
