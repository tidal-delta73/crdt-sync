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


def frontier(session: RGASession) -> VectorClock:
    """An independent VectorClock equal to the session's current clock."""
    return VectorClock.from_snapshot(wire(session.snapshot()["clock"]))


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


class DeleteCausalityTests(unittest.TestCase):
    """Every successful delete carries mergeable causal information."""

    def test_delete_stamps_post_tick_clock_context(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")  # A=1
        session.delete(0)       # ticks A=2, then stamps context {A:2}
        rga = session.snapshot()["rga"]
        self.assertEqual(
            rga["deletes"],
            [{"id": [1, "A"], "clock": {"A": 2}}],
        )
        # The tombstone still exists in the classic field.
        self.assertEqual(rga["tombstones"], [[1, "A"]])

    def test_delete_context_merges_across_replicas(self) -> None:
        a = RGASession("A")
        a.insert(0, "a")
        b = RGASession("B")
        b.merge(restore(a.snapshot()))
        b.delete(0)  # B observed A=1 then ticked B=1
        a.merge(restore(b.snapshot()))
        self.assertEqual(
            a.snapshot()["rga"]["deletes"],
            [{"id": [1, "A"], "clock": {"A": 1, "B": 1}}],
        )

    def test_failed_delete_records_no_causal_context(self) -> None:
        session = RGASession("A")
        self.assertRaises(IndexError, session.delete, 0)
        self.assertNotIn("deletes", session.snapshot()["rga"])


class CompactValidationTests(unittest.TestCase):
    """The compact(stable_clock) argument and frontier contract."""

    def test_non_vector_clock_raises_type_error(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.delete(0)
        for bad in (None, 1, 1.5, "clock", [], {}, True, RGA("A")):
            with self.subTest(bad=bad):
                self.assertRaises(TypeError, session.compact, bad)
        # The rejected calls changed nothing.
        self.assertEqual(session.values(), [])
        self.assertIn([1, "A"], session.snapshot()["rga"]["tombstones"])

    def _session_with_leaf_delete(self) -> RGASession:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.delete(1)  # b is a leaf tombstone, clock A=3
        return session

    def test_equal_frontier_is_accepted_and_reclaims(self) -> None:
        session = self._session_with_leaf_delete()
        self.assertEqual(session.compact(frontier(session)), 1)

    def test_behind_frontier_covers_nothing_new(self) -> None:
        session = self._session_with_leaf_delete()
        behind = VectorClock.from_snapshot(
            {"replica_id": "X", "clock": {"A": 2}}
        )  # the delete happened at A=3, so it is not yet stable
        self.assertEqual(session.compact(behind), 0)
        node_ids = [node["id"] for node in session.snapshot()["rga"]["nodes"]]
        self.assertIn([2, "A"], node_ids)

    def test_concurrent_frontier_raises_and_changes_nothing(self) -> None:
        session = self._session_with_leaf_delete()
        other = RGASession("W")
        other.insert(0, "w")  # clock {W:1}, unknown to session -> concurrent
        before = session.snapshot()
        self.assertRaises(ValueError, session.compact, other._clock)
        self.assertEqual(session.snapshot(), before)
        self.assertEqual(session.values(), ["a"])

    def test_ahead_frontier_raises_and_changes_nothing(self) -> None:
        session = self._session_with_leaf_delete()  # clock A=3
        ahead = VectorClock.from_snapshot(
            {"replica_id": "X", "clock": {"A": 99}}
        )
        before = session.snapshot()
        self.assertRaises(ValueError, session.compact, ahead)
        self.assertEqual(session.snapshot(), before)

    def test_compact_never_advances_the_clock_or_values(self) -> None:
        session = self._session_with_leaf_delete()
        clock_before = session.snapshot()["clock"]
        values_before = session.values()
        self.assertEqual(session.compact(frontier(session)), 1)
        self.assertEqual(session.snapshot()["clock"], clock_before)
        self.assertEqual(session.values(), values_before)
        # Even a no-op compact leaves the clock untouched.
        self.assertEqual(session.compact(frontier(session)), 0)
        self.assertEqual(session.snapshot()["clock"], clock_before)


class CompactReclamationTests(unittest.TestCase):
    """Which tombstones compact may peel, leaf end to the frontier."""

    def test_reclaim_one_leaf_and_summarize_it(self) -> None:
        session = RGASession("A")
        session.insert(0, "x")
        session.insert(1, "y")
        session.delete(1)  # y leaf, A=3
        self.assertEqual(session.compact(frontier(session)), 1)
        rga = session.snapshot()["rga"]
        self.assertEqual([node["id"] for node in rga["nodes"]], [[1, "A"]])
        self.assertEqual(rga["tombstones"], [])
        self.assertNotIn("deletes", rga)
        self.assertEqual(
            rga["retired"],
            [
                {
                    "id": [2, "A"],
                    "value": "y",
                    "prev": [1, "A"],
                    "deleted": {"A": 3},
                }
            ],
        )

    def test_repeated_compact_with_same_frontier_returns_zero(self) -> None:
        session = RGASession("A")
        session.insert(0, "x")
        session.delete(0)
        stable = frontier(session)
        self.assertEqual(session.compact(stable), 1)
        self.assertEqual(session.compact(stable), 0)
        self.assertEqual(session.compact(stable), 0)

    def test_whole_stable_deleted_chain_peels_in_one_call(self) -> None:
        session = RGASession("A")
        session.insert(0, "x")
        session.insert(1, "y")
        session.insert(2, "z")
        session.delete(2)  # z leaf
        session.delete(1)  # y now visible-last
        session.delete(0)  # x last
        # All three tombstones are stable; peeling leaves reaches x.
        self.assertEqual(session.compact(frontier(session)), 3)
        self.assertEqual(session.values(), [])
        rga = session.snapshot()["rga"]
        self.assertEqual(rga["nodes"], [])
        self.assertEqual(rga["tombstones"], [])
        self.assertEqual(
            sorted(entry["id"] for entry in rga["retired"]),
            [[1, "A"], [2, "A"], [3, "A"]],
        )

    def test_live_node_blocks_the_peel(self) -> None:
        session = RGASession("A")
        session.insert(0, "x")
        session.insert(1, "y")
        session.insert(2, "z")
        session.delete(2)  # dead leaf z ...
        self.assertEqual(session.compact(frontier(session)), 1)
        # y is live: the peel stops, and a later leaf delete reclaims
        # incrementally without touching y.
        session.insert(2, "w")  # hangs off live y
        session.delete(2)       # w is again a leaf
        self.assertEqual(session.compact(frontier(session)), 1)
        self.assertEqual(session.values(), ["x", "y"])
        ids = sorted(entry["id"] for entry in session.snapshot()["rga"]["retired"])
        self.assertEqual(ids, [[3, "A"], [4, "A"]])

    def test_referenced_tombstone_cannot_be_reclaimed(self) -> None:
        session = RGASession("A")
        session.insert(0, "x")
        session.insert(1, "y")
        session.delete(0)  # x tombstoned but still anchors live y
        self.assertEqual(session.compact(frontier(session)), 0)
        rga = session.snapshot()["rga"]
        self.assertEqual([node["id"] for node in rga["nodes"]], [[1, "A"], [2, "A"]])
        self.assertEqual(rga["tombstones"], [[1, "A"]])
        self.assertNotIn("retired", rga)
        # Once y is deleted too, both peel together, leaf first.
        session.delete(0)
        self.assertEqual(session.compact(frontier(session)), 2)

    def test_unstabilized_delete_blocks_only_that_node(self) -> None:
        a = RGASession("A")
        a.insert(0, "a")
        a.insert(1, "b")
        a.delete(1)  # delete of b at A=3
        # A frontier that knows the insert (A=2) but not the delete (A=3).
        behind = VectorClock.from_snapshot(
            {"replica_id": "X", "clock": {"A": 2}}
        )
        self.assertEqual(a.compact(behind), 0)
        # After everyone observes the delete it reclaims.
        self.assertEqual(a.compact(frontier(a)), 1)
        self.assertEqual(a.values(), ["a"])

    def test_legacy_tombstone_without_context_is_never_eligible(self) -> None:
        legacy = {
            "replica_id": "L",
            "rga": {
                "replica_id": "L",
                "counter": 2,
                "nodes": [
                    {"id": [1, "L"], "value": "l", "prev": None},
                    {"id": [2, "L"], "value": "m", "prev": [1, "L"]},
                ],
                "tombstones": [[1, "L"]],
            },
            "clock": {"replica_id": "L", "clock": {"L": 5}},
        }
        session = restore(legacy)
        self.assertEqual(session.values(), ["m"])
        # Even a frontier equal to the session clock cannot retire node 1:
        # its delete time is unknown and must never be inferred.
        self.assertEqual(session.compact(frontier(session)), 0)
        rga = session.snapshot()["rga"]
        self.assertEqual(rga["tombstones"], [[1, "L"]])
        self.assertNotIn("deletes", rga)
        self.assertNotIn("retired", rga)

    def test_legacy_tombstone_keeps_blocking_a_chain_forever(self) -> None:
        legacy = {
            "replica_id": "L",
            "rga": {
                "replica_id": "L",
                "counter": 2,
                "nodes": [
                    {"id": [1, "L"], "value": "l", "prev": None},
                    {"id": [2, "L"], "value": "m", "prev": [1, "L"]},
                ],
                "tombstones": [[1, "L"]],
            },
            "clock": {"replica_id": "L", "clock": {"L": 2}},
        }
        session = restore(legacy)
        session.delete(0)  # delete leaf m with a real causal context
        # Only m is eligible; legacy node 1 anchors nothing retained now but
        # has no causal delete record and stays put.
        self.assertEqual(session.compact(frontier(session)), 1)
        rga = session.snapshot()["rga"]
        self.assertEqual([node["id"] for node in rga["nodes"]], [[1, "L"]])
        self.assertEqual(rga["tombstones"], [[1, "L"]])
        self.assertEqual(
            [entry["id"] for entry in rga["retired"]], [[2, "L"]]
        )


class RetirementNoResurrectionTests(unittest.TestCase):
    """Old, duplicate and out-of-order packets cannot undo retirement."""

    def setUp(self) -> None:
        live = RGASession("S")
        live.insert(0, "a")
        live.insert(1, "b")
        self.pre_delete = wire(live.snapshot())  # b still visible
        live.delete(1)
        self.pre_compact = wire(live.snapshot())  # b tombstoned
        self.compacted = RGASession("S")
        self.compacted.merge(restore(self.pre_compact))
        self.compacted.compact(frontier(self.compacted))
        self.compacted_packet = wire(self.compacted.snapshot())

    def test_stale_pre_delete_packet_cannot_resurrect(self) -> None:
        receiver = restore(self.compacted_packet)
        receiver.merge(restore(self.pre_delete))
        self.assertEqual(receiver.values(), ["a"])
        rga = receiver.snapshot()["rga"]
        self.assertEqual([n["id"] for n in rga["nodes"]], [[1, "S"]])
        self.assertEqual([e["id"] for e in rga["retired"]], [[2, "S"]])

    def test_pre_compact_packet_and_duplicates_are_consumed(self) -> None:
        receiver = restore(self.compacted_packet)
        for packet in (
            self.pre_compact,
            self.pre_compact,
            self.compacted_packet,
            self.pre_delete,
        ):
            receiver.merge(restore(packet))
        self.assertEqual(receiver.values(), ["a"])
        rga = receiver.snapshot()["rga"]
        self.assertEqual(rga["tombstones"], [])
        self.assertEqual([e["id"] for e in rga["retired"]], [[2, "S"]])

    def test_out_of_order_new_then_old_converges(self) -> None:
        receiver = RGASession("O")
        receiver.merge(restore(self.compacted_packet))
        receiver.merge(restore(self.pre_delete))
        receiver.merge(restore(self.pre_compact))
        self.assertEqual(receiver.values(), ["a"])

    def test_uncompacted_holder_learns_retirement(self) -> None:
        holder = restore(self.pre_compact)  # holds b's node and tombstone
        self.assertIn([2, "S"], holder.snapshot()["rga"]["tombstones"])
        holder.merge(restore(self.compacted_packet))
        rga = holder.snapshot()["rga"]
        self.assertEqual([n["id"] for n in rga["nodes"]], [[1, "S"]])
        self.assertEqual(rga["tombstones"], [])
        self.assertEqual([e["id"] for e in rga["retired"]], [[2, "S"]])
        # Replaying its own old state afterwards changes nothing.
        holder.merge(restore(self.pre_compact))
        self.assertEqual(holder.values(), ["a"])


class CompactConvergenceTests(unittest.TestCase):
    """Compacted and uncompacted replicas converge both ways."""

    @staticmethod
    def _shared_state(session: RGASession) -> str:
        snapshot = session.snapshot()
        return normalized(
            {
                "rga": {
                    key: snapshot["rga"].get(key)
                    for key in (
                        "counter",
                        "nodes",
                        "tombstones",
                        "deletes",
                        "retired",
                    )
                },
                "clock": snapshot["clock"]["clock"],
            }
        )

    @staticmethod
    def _sync(left: RGASession, right: RGASession) -> None:
        """Bidirectional snapshot exchange until nothing new moves."""
        for _ in range(4):
            left.merge(restore(right.snapshot()))
            right.merge(restore(left.snapshot()))

    def test_compacted_and_plain_replicas_converge(self) -> None:
        plain = RGASession("A")
        plain.insert(0, "a")
        plain.insert(1, "b")
        compacted = RGASession("A")
        compacted.merge(restore(plain.snapshot()))
        plain.delete(1)  # b leaf
        compacted.merge(restore(plain.snapshot()))
        self.assertEqual(compacted.compact(frontier(compacted)), 1)
        # compacted retired b; plain still holds the explicit tombstone.
        self._sync(compacted, plain)
        self.assertEqual(compacted.values(), ["a"])
        self.assertEqual(plain.values(), ["a"])
        self.assertEqual(self._shared_state(compacted), self._shared_state(plain))
        self.assertEqual(compacted.replica_id, "A")
        self.assertEqual(
            compacted.snapshot()["clock"]["clock"],
            plain.snapshot()["clock"]["clock"],
        )

    def test_chain_peels_incrementally_as_references_clear(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        session.insert(1, "b")
        session.insert(2, "c")
        peer = RGASession("B")
        peer.merge(restore(session.snapshot()))

        session.delete(2)  # c leaf reclaims immediately
        self.assertEqual(session.compact(frontier(session)), 1)
        # b is live and a is live: nothing else peels.
        self.assertEqual(session.compact(frontier(session)), 0)

        session.delete(1)  # b now the tail tombstone
        self.assertEqual(session.compact(frontier(session)), 1)
        session.delete(0)  # a finally goes
        self.assertEqual(session.compact(frontier(session)), 1)
        self.assertEqual(session.values(), [])

        # The plain peer converges to the fully compacted state.
        self._sync(session, peer)
        self.assertEqual(peer.values(), [])
        self.assertEqual(self._shared_state(session), self._shared_state(peer))
        retired = sorted(
            entry["id"] for entry in peer.snapshot()["rga"]["retired"]
        )
        self.assertEqual(retired, [[1, "A"], [2, "A"], [3, "A"]])

    def test_branch_peel_does_not_cross_a_live_sibling(self) -> None:
        # root holds two sibling branches; deleting one branch's leaves
        # reclaims that branch while the live sibling stays anchored.
        root = RGASession("R")
        root.insert(0, "root")
        alpha = RGASession("A")
        alpha.merge(restore(root.snapshot()))
        beta = RGASession("B")
        beta.merge(restore(root.snapshot()))
        alpha.insert(1, "achild")
        beta.insert(1, "bchild")
        self._sync(alpha, beta)
        root.merge(restore(alpha.snapshot()))

        # Delete the greater-id sibling's leaf (the later weave child) and
        # compact it away; root and the other child remain.
        victim = root.values()[-1]
        root.delete(root.values().index(victim))
        self.assertEqual(root.compact(frontier(root)), 1)
        self.assertEqual(root.values(), ["root", "bchild"])

        beta.merge(restore(root.snapshot()))
        self.assertEqual(beta.values(), ["root", "bchild"])
        self.assertEqual(self._shared_state(root), self._shared_state(beta))

    def test_retired_anchor_keeps_late_concurrent_child_in_position(self) -> None:
        base = RGASession("K")
        base.insert(0, "r")
        peer = RGASession("T")
        peer.merge(restore(base.snapshot()))
        base.insert(1, "mid")
        base.delete(1)
        base.compact(frontier(base))  # mid retired, anchor r survives
        peer.insert(1, "late")  # (2,T) hangs off the shared r node
        compacted = restore(base.snapshot())
        compacted.merge(restore(peer.snapshot()))
        self.assertEqual(compacted.values(), ["r", "late"])
        peer.merge(restore(compacted.snapshot()))
        self.assertEqual(peer.values(), ["r", "late"])
        self.assertEqual(self._shared_state(compacted), self._shared_state(peer))

    def test_continue_insert_delete_offline_reconnect_after_compaction(self) -> None:
        a = RGASession("A")
        a.insert(0, "a")
        a.insert(1, "b")
        b = RGASession("B")
        b.merge(restore(a.snapshot()))
        a.delete(1)
        a.compact(frontier(a))  # b retired on A
        a.insert(1, "tail")  # fresh id, clock moves on
        # Offline edit on the stale side (it still holds live b... it has
        # a,b; prepend head).
        b.insert(0, "head")
        # Reconnect: neither id is reused, the clock never rolls back.
        relation_a = a.merge(restore(b.snapshot()))
        # b has not received a's state yet, so its context is strictly behind
        # a's just-merged clock: the peer (a) is "after" b.
        relation_b = b.merge(restore(a.snapshot()))
        self.assertEqual(relation_a, "concurrent")
        self.assertEqual(relation_b, "after")
        self._sync(a, b)
        self.assertEqual(a.values(), ["head", "a", "tail"])
        self.assertEqual(a.values(), b.values())
        self.assertEqual(self._shared_state(a), self._shared_state(b))
        ids = [tuple(n["id"]) for n in a.snapshot()["rga"]["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))
        # Continued deletes and a second compaction keep working.
        a.delete(0)
        self._sync(a, b)
        self.assertGreaterEqual(a.compact(frontier(a)), 1)

    def test_merge_relation_still_reflects_only_pre_merge_clocks(self) -> None:
        a = RGASession("A")
        a.insert(0, "x")
        a.delete(0)
        a.compact(frontier(a))
        b = RGASession("B")
        self.assertEqual(b.merge(restore(a.snapshot())), "after")
        self.assertEqual(b.merge(restore(a.snapshot())), "equal")
        b.insert(0, "y")
        self.assertEqual(b.merge(restore(a.snapshot())), "before")

    def test_stale_history_replayed_after_compaction_converges(self) -> None:
        a = RGASession("A")
        a.insert(0, "a")
        a.insert(1, "b")
        old_packet = wire(a.snapshot())  # a,b both live
        a.delete(1)
        post_delete = wire(a.snapshot())
        a.compact(frontier(a))

        receiver = restore(a.snapshot())
        # Newest state first, then stale and duplicate packets afterwards.
        receiver.merge(restore(old_packet))
        receiver.merge(restore(post_delete))
        receiver.merge(restore(old_packet))
        receiver.merge(restore(a.snapshot()))
        self.assertEqual(receiver.values(), ["a"])
        self.assertEqual(self._shared_state(receiver), self._shared_state(a))
        self.assertEqual(
            [entry["id"] for entry in receiver.snapshot()["rga"]["retired"]],
            [[2, "A"]],
        )


class CompactConflictTests(unittest.TestCase):
    """Node-id content conflicts still raise, with atomic metadata."""

    def test_conflict_against_retired_node_raises_atomically(self) -> None:
        first = RGASession("X")
        first.insert(0, "hello")
        first.delete(0)
        first.compact(frontier(first))
        receiver = RGASession("R")
        receiver.merge(restore(first.snapshot()))
        before = receiver.snapshot()

        colliding = RGASession("X")
        colliding.insert(0, "world")  # same id (1, X), different value
        self.assertRaises(ValueError, receiver.merge, colliding)
        self.assertEqual(receiver.snapshot(), before)
        self.assertEqual(receiver.values(), [])

    def test_retained_versus_retired_conflict_raises_atomically(self) -> None:
        one = RGASession("A")
        one.insert(0, "same")
        holder = restore(one.snapshot())
        other = RGASession("A")
        other.insert(0, "different")  # colliding (1,A)
        witness = RGASession("W")
        witness.merge(other)
        witness.delete(0)
        witness.compact(frontier(witness))
        before = holder.snapshot()
        self.assertRaises(
            ValueError, holder.merge, restore(witness.snapshot())
        )
        self.assertEqual(holder.snapshot(), before)


class CompactSnapshotTests(unittest.TestCase):
    """Round trips, determinism, independence and legacy acceptance."""

    def _compacted_session(self) -> RGASession:
        session = RGASession("Z")
        session.insert(0, "z")
        session.insert(1, "q")
        session.delete(1)
        session.compact(frontier(session))
        return session

    def test_compacted_snapshot_round_trips_through_json(self) -> None:
        session = self._compacted_session()
        data = wire(session.snapshot())
        restored = RGASession.from_snapshot(data)
        self.assertEqual(restored.snapshot(), data)
        self.assertEqual(restored.values(), ["z"])
        # Retired nodes do not come back after restore.
        self.assertEqual(restored.compact(frontier(restored)), 0)
        restored.merge(RGASession("Z"))  # no-op self-like merge
        self.assertEqual(restored.values(), ["z"])

    def test_snapshots_are_deterministic_and_deeply_independent(self) -> None:
        session = self._compacted_session()
        first = session.snapshot()
        second = session.snapshot()
        self.assertEqual(first, second)
        self.assertIsNot(first["rga"], second["rga"])
        first["rga"]["retired"].append("junk")
        first["rga"]["deletes"] = "tampered"
        first["clock"]["clock"]["?"] = 9
        self.assertEqual(session.snapshot(), second)

    def test_legacy_three_field_session_restores_and_merges(self) -> None:
        legacy = {
            "replica_id": "L",
            "rga": {
                "replica_id": "L",
                "counter": 1,
                "nodes": [{"id": [1, "L"], "value": "l", "prev": None}],
                "tombstones": [],
            },
            "clock": {"replica_id": "L", "clock": {"L": 1}},
        }
        session = RGASession.from_snapshot(wire(legacy))
        self.assertEqual(session.values(), ["l"])
        session.insert(1, "m")
        session.delete(0)
        self.assertEqual(session.values(), ["m"])
        # Continued edits never reuse id 1 or roll the clock back.
        self.assertEqual(session.snapshot()["clock"]["clock"], {"L": 3})
        peer = RGASession("P")
        peer.merge(restore(legacy))
        peer.merge(restore(session.snapshot()))
        session.merge(restore(peer.snapshot()))
        self.assertEqual(peer.values(), session.values())

    def test_legacy_session_can_gain_causal_deletes_by_merge(self) -> None:
        legacy = {
            "replica_id": "L",
            "rga": {
                "replica_id": "L",
                "counter": 2,
                "nodes": [
                    {"id": [1, "L"], "value": "l", "prev": None},
                    {"id": [2, "L"], "value": "n", "prev": [1, "L"]},
                ],
                "tombstones": [[1, "L"]],
            },
            "clock": {"replica_id": "L", "clock": {"L": 2}},
        }
        session = restore(legacy)
        # A fresh causal delete of the visible leaf n; old tombstone 1 stays
        # non-causal.
        session.delete(0)
        self.assertEqual(session.compact(frontier(session)), 1)
        rga = session.snapshot()["rga"]
        self.assertEqual([e["id"] for e in rga["retired"]], [[2, "L"]])
        self.assertEqual(rga["tombstones"], [[1, "L"]])

    def test_malformed_extension_fields_are_rejected(self) -> None:
        good = wire(self._compacted_session().snapshot())["rga"]

        def session_rga(**overrides) -> dict:
            rga = json.loads(json.dumps(good))
            rga.update(overrides)
            return {
                "replica_id": "Z",
                "rga": rga,
                "clock": {"replica_id": "Z", "clock": {"Z": 3}},
            }

        self.assertRaises(
            ValueError,
            RGASession.from_snapshot,
            session_rga(deletes="nope"),
        )
        self.assertRaises(
            ValueError,
            RGASession.from_snapshot,
            session_rga(retired="nope"),
        )
        self.assertRaises(
            ValueError,
            RGASession.from_snapshot,
            session_rga(deletes=[{"id": [2, "Z"]}]),
        )
        self.assertRaises(
            ValueError,
            RGASession.from_snapshot,
            session_rga(
                retired=[
                    {"id": [2, "Z"], "value": "q", "prev": [1, "Z"]}
                ]
            ),
        )
        # Retirement without a non-empty delete context is invalid.
        self.assertRaises(
            ValueError,
            RGASession.from_snapshot,
            session_rga(
                retired=[
                    {
                        "id": [2, "Z"],
                        "value": "q",
                        "prev": [1, "Z"],
                        "deleted": {},
                    }
                ]
            ),
        )

    def test_delete_context_must_belong_to_a_tombstone(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        snapshot = wire(session.snapshot())
        snapshot["rga"]["deletes"] = [{"id": [1, "A"], "clock": {"A": 1}}]
        self.assertRaises(ValueError, RGASession.from_snapshot, snapshot)

    def test_retired_node_must_keep_no_explicit_record(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        snapshot = wire(session.snapshot())
        snapshot["rga"]["retired"] = [
            {
                "id": [1, "A"],
                "value": "a",
                "prev": None,
                "deleted": {"A": 1},
            }
        ]
        # Node 1 is both retained and retired.
        self.assertRaises(ValueError, RGASession.from_snapshot, snapshot)

    def test_retired_anchor_chain_must_resolve(self) -> None:
        session = RGASession("A")
        session.insert(0, "a")
        snapshot = wire(session.snapshot())
        snapshot["rga"]["nodes"] = []
        snapshot["rga"]["retired"] = [
            {
                "id": [1, "A"],
                "value": "a",
                "prev": [9, "A"],  # missing predecessor
                "deleted": {"A": 1},
            }
        ]
        self.assertRaises(ValueError, RGASession.from_snapshot, snapshot)


if __name__ == "__main__":
    unittest.main()
