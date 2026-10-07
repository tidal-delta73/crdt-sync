"""Tests for RGASession: an RGA replica plus its vector clock as one unit."""

from __future__ import annotations

import json
import unittest

from crdt_sync import RGA, RGASession, VectorClock


def wire(snapshot: object) -> object:
    """Round-trip a snapshot through JSON, as a real network boundary would."""
    return json.loads(json.dumps(snapshot))


def normalized(snapshot: object) -> str:
    """Canonical JSON fingerprint of a snapshot."""
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"))


def restore(snapshot: object) -> RGASession:
    return RGASession.from_snapshot(wire(snapshot))


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


if __name__ == "__main__":
    unittest.main()
