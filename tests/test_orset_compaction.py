"""Tests for ORSet tombstone compaction (``compact`` + ``compacted`` summary).

Coverage:
* the prefix rule per tag origin: start at sequence 1, retire only while
  every add is present and tombstoned; stop at gaps, live tags, unseen
  history or already-retired bounds; the return value counts dropped adds;
* visibility invariance: ``elements()`` / ``contains()`` and subsequent
  ``add()`` / ``remove()`` results are unchanged by compaction;
* merge semantics of retired bounds: per-origin maximum join, consumption of
  late/duplicated adds and tombstones at or below the bound, no resurrection,
  conflict detection and counter advancement still effective above the bound;
* convergence between compacted, never-compacted, restored and stale replicas
  in arbitrary bidirectional delivery orders, with identical canonical state
  apart from ``replica_id`` (and the local ``counter``);
* snapshot shape: the four-field shape until a summary exists, ``compacted``
  afterwards with sorted keys; old and new shapes load; JSON round trips
  preserve the summary; every malformed summary and every bound/history
  inconsistency is a ``ValueError``; snapshots stay isolated;
* seeded differential fuzzing against an independent reference model that
  retains full history.

Only the public API is used.
"""

from __future__ import annotations

import copy
import json
import random
import unittest
from itertools import permutations

from crdt_sync import ORSet


def wire_restore(snapshot: dict) -> ORSet:
    return ORSet.from_snapshot(json.loads(json.dumps(snapshot)))


def cap(replica: ORSet) -> dict:
    return json.loads(json.dumps(replica.snapshot()))


def canonical(snapshot: dict) -> str:
    """Replica-independent state fingerprint, including the summary."""
    return json.dumps(
        {
            "adds": snapshot["adds"],
            "removes": snapshot["removes"],
            "compacted": snapshot.get("compacted", {}),
        },
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Independent reference: keeps the *full* add/tombstone history forever and
# tracks retired bounds only as a visibility filter / merge artifact.
# ---------------------------------------------------------------------------


class ReferenceORSet:
    def __init__(self, replica_id: str) -> None:
        self.replica_id = replica_id
        self.counter = 0
        self.adds: dict[str, set[tuple[str, int]]] = {}
        self.removes: set[tuple[str, int]] = set()
        self.bounds: dict[str, int] = {}

    def _live(self, tags: set[tuple[str, int]]) -> set[tuple[str, int]]:
        return {
            tag
            for tag in tags
            if tag not in self.removes
            and tag[1] > self.bounds.get(tag[0], 0)
        }

    def add(self, element: str) -> None:
        self.counter += 1
        self.adds.setdefault(element, set()).add((self.replica_id, self.counter))

    def remove(self, element: str) -> bool:
        tags = self.adds.get(element)
        if not tags:
            return False
        live = self._live(tags)
        if not live:
            return False
        self.removes.update(live)
        return True

    def elements(self) -> set[str]:
        return {
            element
            for element, tags in self.adds.items()
            if self._live(tags)
        }

    def compact(self) -> int:
        present: dict[str, set[int]] = {}
        for tags in self.adds.values():
            for origin, sequence in tags:
                present.setdefault(origin, set()).add(sequence)
        dropped = 0
        for origin in set(present) | set(self.bounds):
            old_bound = self.bounds.get(origin, 0)
            sequences = present.get(origin, set())
            bound = old_bound
            candidate = old_bound + 1
            while (
                candidate in sequences
                and (origin, candidate) in self.removes
            ):
                bound = candidate
                candidate += 1
            if bound > old_bound:
                self.bounds[origin] = bound
                # History is retained here; count what production drops.
                for tags in self.adds.values():
                    dropped += sum(
                        1
                        for origin_id, sequence in tags
                        if origin_id == origin and old_bound < sequence <= bound
                    )
        return dropped

    def merge_wire(self, snapshot: dict) -> None:
        """Absorb a *production-shaped* wire snapshot, retaining history."""
        for element, raw_tags in snapshot["adds"].items():
            self.adds.setdefault(element, set()).update(
                tuple(tag) for tag in raw_tags
            )
        for raw_tag in snapshot["removes"]:
            self.removes.add(tuple(raw_tag))
        for origin, bound in snapshot.get("compacted", {}).items():
            if bound > self.bounds.get(origin, 0):
                self.bounds[origin] = bound
        own_bound = self.bounds.get(self.replica_id, 0)
        if own_bound > self.counter:
            self.counter = own_bound
        for tags in self.adds.values():
            for origin, sequence in tags:
                if origin == self.replica_id and sequence > self.counter:
                    self.counter = sequence


# ---------------------------------------------------------------------------
# Hand-built prefix / return-value cases
# ---------------------------------------------------------------------------


class CompactPrefixTests(unittest.TestCase):
    def test_empty_and_fully_live_state_compact_nothing(self) -> None:
        replica = ORSet("r")
        self.assertEqual(replica.compact(), 0)
        self.assertEqual(
            replica.snapshot(),
            {"replica_id": "r", "counter": 0, "adds": {}, "removes": []},
        )

        replica.add("x")
        before = replica.snapshot()
        self.assertEqual(replica.compact(), 0)
        self.assertEqual(replica.snapshot(), before)
        self.assertNotIn("compacted", replica.snapshot())

    def test_single_dead_prefix_is_retired(self) -> None:
        replica = ORSet("a")
        replica.add("x")
        replica.add("y")
        replica.remove("x")  # (a,1) dead; (a,2) live
        before_elements = replica.elements()

        self.assertEqual(replica.compact(), 1)
        self.assertEqual(replica.elements(), before_elements)
        self.assertEqual(replica.elements(), {"y"})
        self.assertTrue(replica.contains("y"))
        self.assertFalse(replica.contains("x"))
        self.assertEqual(
            replica.snapshot(),
            {
                "replica_id": "a",
                "counter": 2,
                "adds": {"y": [["a", 2]]},
                "removes": [],
                "compacted": {"a": 1},
            },
        )
        # Idempotent: the retired prefix is gone, nothing more to retire.
        self.assertEqual(replica.compact(), 0)
        self.assertEqual(replica.compact(), 0)

    def test_long_prefix_retired_in_full_then_extended(self) -> None:
        replica = ORSet("a")
        replica.add("x")
        replica.remove("x")
        replica.add("y")
        replica.remove("y")
        self.assertEqual(replica.compact(), 2)
        self.assertEqual(replica.snapshot()["compacted"], {"a": 2})
        self.assertEqual(replica.snapshot()["adds"], {})
        self.assertEqual(replica.snapshot()["removes"], [])

        # A fresh dead tag beyond the bound extends the summary later.
        replica.add("z")
        replica.remove("z")
        self.assertEqual(replica.compact(), 1)
        self.assertEqual(replica.snapshot()["compacted"], {"a": 3})

    def test_live_tag_blocks_extension_but_not_the_earlier_prefix(self) -> None:
        # (a,1) dead and (a,2) live: the dead prefix up to the live tag is
        # still retired; only crossing *over* the live tag is forbidden.
        replica = ORSet("a")
        replica.add("x")
        replica.remove("x")
        replica.add("y")
        self.assertEqual(replica.elements(), {"y"})
        self.assertEqual(replica.compact(), 1)
        self.assertEqual(replica.snapshot()["compacted"], {"a": 1})
        self.assertEqual(
            replica.snapshot()["adds"], {"y": [["a", 2]]}
        )
        self.assertEqual(replica.elements(), {"y"})
        # Nothing changed while the next tag stays live.
        self.assertEqual(replica.compact(), 0)

        # Once the live tag is removed too, the prefix extends over it.
        replica.remove("y")
        self.assertEqual(replica.compact(), 1)
        self.assertEqual(replica.snapshot()["compacted"], {"a": 2})
        self.assertEqual(replica.snapshot()["adds"], {})

    def test_gap_blocks_prefix_extension(self) -> None:
        alpha = ORSet("alpha")
        alpha.add("x")
        alpha.remove("x")
        observer = wire_restore(alpha.snapshot())
        self.assertEqual(observer.compact(), 1)  # knows alpha:1 only

        # alpha now has alpha:2 dead, alpha:3 live; observer learns both.
        alpha.add("y")
        alpha.remove("y")
        alpha.add("z")
        observer.merge(wire_restore(alpha.snapshot()))
        # alpha:2 is dead and contiguous, alpha:3 is live, so extend to 2
        # but no further — the live tag blocks.
        self.assertEqual(observer.compact(), 1)
        self.assertEqual(observer.snapshot()["compacted"], {"alpha": 2})
        self.assertEqual(observer.elements(), {"z"})
        self.assertEqual(observer.compact(), 0)

    def test_missing_history_blocks_prefix(self) -> None:
        # A snapshot fed by hand: observer knows only alpha:3 (dead); with no
        # alpha:1/2 there is a gap from sequence 1, so nothing retires.
        snapshot = {
            "replica_id": "obs",
            "counter": 0,
            "adds": {"late": [["alpha", 3]]},
            "removes": [["alpha", 3]],
        }
        observer = ORSet.from_snapshot(copy.deepcopy(snapshot))
        self.assertEqual(observer.compact(), 0)
        self.assertEqual(observer.snapshot(), snapshot)

    def test_origins_are_compacted_independently_and_keys_sorted(self) -> None:
        observer = ORSet("obs")
        alpha = ORSet("alpha")
        beta = ORSet("beta")
        gamma = ORSet("gamma")
        alpha.add("a")
        alpha.remove("a")
        beta.add("b")
        beta.remove("b")
        gamma.add("g")  # still live
        for peer in (alpha, beta, gamma):
            observer.merge(wire_restore(peer.snapshot()))
        self.assertEqual(observer.compact(), 2)
        self.assertEqual(
            observer.snapshot()["compacted"], {"alpha": 1, "beta": 1}
        )
        self.assertEqual(observer.elements(), {"g"})
        # JSON key order in the emitted mapping is stable (sorted).
        text = json.dumps(observer.snapshot())
        self.assertLess(text.index('"alpha"'), text.index('"beta"'))

    def test_visibility_never_changes_and_adds_continue(self) -> None:
        replica = ORSet("a")
        replica.add("x")
        replica.add("y")
        replica.remove("x")
        expected = replica.elements()
        self.assertEqual(replica.compact(), 1)
        self.assertEqual(replica.elements(), expected)
        self.assertEqual(replica.compact(), 0)
        replica.add("x")  # re-add after compaction uses a fresh tag
        self.assertEqual(replica.elements(), {"x", "y"})
        self.assertTrue(replica.remove("x"))
        replica.add("x")
        self.assertEqual(replica.elements(), {"x", "y"})


class CompactMergeTests(unittest.TestCase):
    def _built_history(self):
        # alpha: a1 add/remove x, a2 add/remove y, a3 live z.
        alpha = ORSet("alpha")
        alpha.add("x")
        alpha.remove("x")
        alpha.add("y")
        alpha.remove("y")
        alpha.add("z")
        full = cap(alpha)  # pre-compaction, carries all tombstones
        self.assertEqual(alpha.compact(), 2)
        compacted = cap(alpha)  # bound alpha:2, only z explicit
        return alpha, full, compacted

    def test_compacted_state_merges_into_fresh_peer(self) -> None:
        alpha, _full, compacted = self._built_history()
        beta = ORSet("beta")
        beta.merge(wire_restore(compacted))
        self.assertEqual(beta.elements(), {"z"})
        self.assertEqual(canonical(beta.snapshot()), canonical(compacted))
        beta.add("w")
        self.assertEqual(beta.elements(), {"z", "w"})

        # Full bidirectional exchange converges to the compacted form.
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(alpha.elements(), beta.elements())
        self.assertEqual(alpha.elements(), {"z", "w"})
        self.assertEqual(canonical(alpha.snapshot()), canonical(beta.snapshot()))

    def test_full_history_peer_adopts_bound_and_drops_explicit_prefix(self) -> None:
        alpha, full, compacted = self._built_history()
        # A peer that absorbed the *pre-compaction* state keeps all records.
        follower = ORSet("follower")
        follower.merge(wire_restore(full))
        self.assertEqual(follower.snapshot()["removes"], [["alpha", 1], ["alpha", 2]])
        self.assertNotIn("compacted", follower.snapshot())

        # Merging the compacted state certifies the prefix as consumed: the
        # follower's explicit records below the bound disappear too.
        follower.merge(wire_restore(compacted))
        self.assertEqual(canonical(follower.snapshot()), canonical(compacted))
        self.assertEqual(follower.elements(), {"z"})

    def test_stale_pre_compaction_snapshot_cannot_resurrect(self) -> None:
        alpha, full, compacted = self._built_history()
        beta = ORSet("beta")
        beta.merge(wire_restore(compacted))
        before = canonical(beta.snapshot())

        # The old, tombstone-heavy state arrives late, repeatedly, in both
        # directions and after a round trip through JSON.
        for _ in range(3):
            beta.merge(wire_restore(full))
            beta.merge(wire_restore(cap(beta)))
        self.assertEqual(canonical(beta.snapshot()), before)
        self.assertEqual(beta.elements(), {"z"})

        alpha.merge(wire_restore(full))  # self-shaped stale state
        self.assertEqual(canonical(alpha.snapshot()), before)

    def test_arbitrary_delivery_orders_converge(self) -> None:
        _alpha, full, compacted = self._built_history()
        beta = ORSet("beta")
        beta.merge(wire_restore(compacted))
        beta.add("q")
        beta_state = cap(beta)

        gamma = ORSet("gamma")
        gamma.add("g")  # unrelated concurrent activity
        gamma_state = cap(gamma)

        messages = [full, compacted, beta_state, gamma_state]
        baselines = set()
        for order in permutations(messages):
            observer = ORSet("observer")
            for message in order:
                observer.merge(wire_restore(message))
            self.assertEqual(observer.elements(), {"z", "q", "g"})
            # x and y were fully retired and never come back.
            self.assertFalse(observer.contains("x"))
            self.assertFalse(observer.contains("y"))
            baselines.add(canonical(observer.snapshot()))
        self.assertEqual(len(baselines), 1)

    def test_bounds_join_by_maximum_across_chains(self) -> None:
        alpha = ORSet("alpha")
        alpha.add("x")
        alpha.remove("x")
        first = wire_restore(alpha.snapshot())
        self.assertEqual(first.compact(), 1)  # bound alpha:1

        alpha.add("y")
        alpha.remove("y")
        second = wire_restore(alpha.snapshot())
        # second still carries alpha:1 explicitly and no bound yet; merging
        # must not LOWER the bound already known to first.
        snapshot_before = first.snapshot()
        first.merge(second)
        self.assertEqual(first.snapshot()["compacted"], {"alpha": 1})
        # And the contiguous dead alpha:2 can now be retired as well.
        self.assertEqual(first.compact(), 1)
        self.assertEqual(first.snapshot()["compacted"], {"alpha": 2})
        self.assertEqual(first.elements(), set())
        # Redelivering the lower-bounded (already compacted) state is a no-op.
        first.merge(wire_restore(snapshot_before))
        self.assertEqual(first.snapshot()["compacted"], {"alpha": 2})

    def test_conflict_above_bound_still_detected(self) -> None:
        # alpha retired alpha:1; two live replicas sharing id "alpha" then
        # both mint alpha:3 for different elements.
        left = ORSet.from_snapshot(
            {
                "replica_id": "alpha",
                "counter": 3,
                "adds": {"p": [["alpha", 3]]},
                "removes": [],
                "compacted": {"alpha": 2},
            }
        )
        right = ORSet.from_snapshot(
            {
                "replica_id": "alpha",
                "counter": 3,
                "adds": {"q": [["alpha", 3]]},
                "removes": [],
                "compacted": {"alpha": 2},
            }
        )
        before = left.snapshot()
        with self.assertRaises(ValueError):
            left.merge(right)
        self.assertEqual(left.snapshot(), before)

    def test_different_ownership_below_bound_is_consumed_not_conflict(self) -> None:
        # The retired prefix has no observable owner: a stale state that
        # attributes alpha:1 to another element must be ignored, not raise.
        compacted = ORSet.from_snapshot(
            {
                "replica_id": "obs",
                "counter": 0,
                "adds": {},
                "removes": [],
                "compacted": {"alpha": 1},
            }
        )
        stale = wire_restore(
            {
                "replica_id": "alpha",
                "counter": 1,
                "adds": {"ghost": [["alpha", 1]]},
                "removes": [["alpha", 1]],
            }
        )
        before = canonical(compacted.snapshot())
        compacted.merge(stale)
        self.assertEqual(canonical(compacted.snapshot()), before)
        self.assertEqual(compacted.elements(), set())

    def test_bound_advances_local_counter_for_restored_replica(self) -> None:
        alpha = ORSet("alpha")
        alpha.add("x")
        alpha.remove("x")
        alpha.add("y")
        alpha.remove("y")
        self.assertEqual(alpha.compact(), 2)

        # A stale backup of alpha itself learns its own retired prefix via the
        # bound; its next tag must be alpha:3, beyond the bound.
        backup = ORSet("alpha")
        backup.merge(wire_restore(alpha.snapshot()))
        backup.add("d")
        self.assertEqual(backup.elements(), {"d"})

        everyone = ORSet("observer")
        everyone.merge(wire_restore(alpha.snapshot()))
        everyone.merge(wire_restore(backup.snapshot()))
        self.assertEqual(everyone.elements(), {"d"})
        origins = [
            tag
            for tags in everyone.snapshot()["adds"].values()
            for tag in tags
            if tag[0] == "alpha"
        ]
        self.assertEqual([sequence for _, sequence in origins], [3])

    def test_continue_add_remove_after_bidirectional_compaction(self) -> None:
        alpha = ORSet("alpha")
        beta = ORSet("beta")
        alpha.add("x")
        alpha.remove("x")
        alpha.compact()
        beta.merge(wire_restore(alpha.snapshot()))
        alpha.merge(wire_restore(beta.snapshot()))

        alpha.add("x")  # fresh tag for a previously retired element
        beta.add("y")
        beta.remove("y")
        beta.compact()
        alpha.merge(wire_restore(beta.snapshot()))
        beta.merge(wire_restore(alpha.snapshot()))
        self.assertEqual(alpha.elements(), {"x"})
        self.assertEqual(beta.elements(), {"x"})
        self.assertEqual(canonical(alpha.snapshot()), canonical(beta.snapshot()))

        alpha.remove("x")
        alpha.compact()
        beta.merge(wire_restore(alpha.snapshot()))
        alpha.merge(wire_restore(beta.snapshot()))
        self.assertEqual(alpha.elements(), set())
        self.assertEqual(beta.elements(), set())
        self.assertEqual(canonical(alpha.snapshot()), canonical(beta.snapshot()))


# ---------------------------------------------------------------------------
# Snapshot shape / validation
# ---------------------------------------------------------------------------


class CompactSnapshotTests(unittest.TestCase):
    def test_old_four_field_shape_still_loads_and_prints(self) -> None:
        old = {
            "replica_id": "a",
            "counter": 1,
            "adds": {"x": [["a", 1]]},
            "removes": [["a", 1]],
        }
        restored = ORSet.from_snapshot(copy.deepcopy(old))
        self.assertEqual(restored.snapshot(), old)
        self.assertNotIn("compacted", restored.snapshot())

    def test_new_shape_round_trips_through_json(self) -> None:
        snapshot = {
            "replica_id": "a",
            "counter": 3,
            "adds": {"z": [["a", 3], ["b", 2]]},
            "removes": [["b", 2]],
            "compacted": {"a": 2, "b": 1},
        }
        restored = ORSet.from_snapshot(json.loads(json.dumps(snapshot)))
        self.assertEqual(restored.snapshot(), snapshot)
        self.assertEqual(restored.elements(), {"z"})
        restored.add("q")
        self.assertEqual(restored.elements(), {"z", "q"})

    def test_empty_origin_bound_accepted_for_non_local_origin(self) -> None:
        snapshot = {
            "replica_id": "obs",
            "counter": 0,
            "adds": {},
            "removes": [],
            "compacted": {"alpha": 5},
        }
        restored = ORSet.from_snapshot(copy.deepcopy(snapshot))
        self.assertEqual(restored.snapshot(), snapshot)

    def test_compacted_summary_is_independent_of_returned_dict(self) -> None:
        replica = ORSet("a")
        replica.add("x")
        replica.remove("x")
        replica.compact()
        snapshot = replica.snapshot()
        snapshot["compacted"]["a"] = 999
        snapshot["compacted"]["ghost"] = 1
        self.assertEqual(replica.snapshot()["compacted"], {"a": 1})
        second = replica.snapshot()
        self.assertIsNot(second["compacted"], snapshot["compacted"])

    def test_malformed_compacted_summaries(self) -> None:
        base = {
            "replica_id": "a",
            "counter": 0,
            "adds": {},
            "removes": [],
        }

        def expect_error(summary: object, label: str) -> None:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    ORSet.from_snapshot({**base, "compacted": summary})

        expect_error([], "non-dict list")
        expect_error("alpha:1", "non-dict string")
        expect_error(42, "non-dict int")
        expect_error(True, "non-dict bool")
        expect_error(None, "non-dict null")
        # An explicitly present but empty mapping carries no summary and
        # normalizes to the four-field shape.
        empty = ORSet.from_snapshot({**base, "compacted": {}})
        self.assertNotIn("compacted", empty.snapshot())
        expect_error({"": 1}, "empty origin")
        expect_error({1: 1}, "non-string origin")
        expect_error({None: 1}, "null origin")
        expect_error({"a": True}, "bool bound")
        expect_error({"a": False}, "bool bound false")
        expect_error({"a": 0}, "zero bound")
        expect_error({"a": -3}, "negative bound")
        expect_error({"a": 1.0}, "float bound")
        expect_error({"a": "1"}, "string bound")
        expect_error({"a": None}, "null bound")

        with self.assertRaises(ValueError):
            ORSet.from_snapshot({**base, "compacted": {"a": 1}, "extra": 1})

    def test_explicit_history_at_or_below_bound_is_rejected(self) -> None:
        base = {
            "replica_id": "a",
            "counter": 2,
            "removes": [],
            "compacted": {"a": 2},
        }
        with self.assertRaises(ValueError):
            ORSet.from_snapshot({**base, "adds": {"x": [["a", 1]]}})
        with self.assertRaises(ValueError):
            ORSet.from_snapshot({**base, "adds": {"x": [["a", 2]]}})
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {**base, "adds": {"x": [["a", 3]]}, "removes": [["a", 2]]}
            )
        # A per-tag tombstone at the bound without its add is also rejected.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "obs",
                    "counter": 0,
                    "adds": {},
                    "removes": [["a", 1]],
                    "compacted": {"a": 1},
                }
            )

    def test_existing_causal_checks_still_apply_above_bound(self) -> None:
        # Tombstone above the bound with no matching add.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "obs",
                    "counter": 0,
                    "adds": {},
                    "removes": [["alpha", 3]],
                    "compacted": {"alpha": 2},
                }
            )
        # Counter ahead of the local history.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 4,
                    "adds": {"z": [["a", 3]]},
                    "removes": [],
                    "compacted": {"a": 2},
                }
            )
        # Bound for the local origin above the counter would allow tag
        # collisions on the next add.
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 1,
                    "adds": {},
                    "removes": [],
                    "compacted": {"a": 2},
                }
            )
        # Counter behind the local history (explicit a:4 missing).
        with self.assertRaises(ValueError):
            ORSet.from_snapshot(
                {
                    "replica_id": "a",
                    "counter": 4,
                    "adds": {"z": [["a", 3], ["a", 4]], "w": [["a", 5]]},
                    "removes": [],
                    "compacted": {"a": 2},
                }
            )

    def test_valid_counter_shapes_with_bounds(self) -> None:
        # Bound exactly at counter: no explicit own tags.
        restored = ORSet.from_snapshot(
            {
                "replica_id": "a",
                "counter": 2,
                "adds": {},
                "removes": [],
                "compacted": {"a": 2},
            }
        )
        self.assertEqual(restored.elements(), set())
        restored.add("n")
        self.assertEqual(
            restored.snapshot()["adds"], {"n": [["a", 3]]}
        )


# ---------------------------------------------------------------------------
# Seeded differential fuzzing vs the full-history reference
# ---------------------------------------------------------------------------


ELEMENTS = ("x", "y", "z", "w")


class DifferentialFuzzTests(unittest.TestCase):
    def run_trace(self, seed: int, n: int, steps: int) -> None:
        rng = random.Random(seed)
        ids = [f"r{i}" for i in range(n)]
        pros = [ORSet(ident) for ident in ids]
        refs = [ReferenceORSet(ident) for ident in ids]
        # Per-replica log of wire snapshots they have been sent, for late and
        # duplicated replay.
        delivered: list[list[dict]] = [[] for _ in ids]

        def assert_matches(label: str) -> None:
            for i, ident in enumerate(ids):
                self.assertEqual(
                    pros[i].elements(),
                    refs[i].elements(),
                    msg=f"{label}: {ident} elements, seed={seed}",
                )
                json.dumps(pros[i].snapshot())  # always wire-serializable

        def exchange(i: int, j: int, snapshot: dict) -> None:
            pros[j].merge(wire_restore(snapshot))
            refs[j].merge_wire(copy.deepcopy(snapshot))
            delivered[j].append(copy.deepcopy(snapshot))

        for step in range(steps):
            kind = rng.choice(
                (
                    "add",
                    "add",
                    "remove",
                    "compact",
                    "compact",
                    "exchange",
                    "replay",
                    "restore",
                )
            )
            i = rng.randrange(n)

            if kind == "add":
                element = rng.choice(ELEMENTS)
                pros[i].add(element)
                refs[i].add(element)

            elif kind == "remove":
                element = rng.choice(ELEMENTS)
                self.assertEqual(
                    pros[i].remove(element),
                    refs[i].remove(element),
                    msg=f"seed={seed} step={step}",
                )

            elif kind == "compact":
                expected_elements = pros[i].elements()
                got = pros[i].compact()
                want = refs[i].compact()
                self.assertEqual(got, want, msg=f"seed={seed} step={step}")
                self.assertEqual(pros[i].elements(), expected_elements)
                if rng.random() < 0.5:
                    self.assertEqual(pros[i].compact(), 0)
                    self.assertEqual(refs[i].compact(), 0)

            elif kind == "exchange":
                j = rng.randrange(n - 1)
                if j >= i:
                    j += 1
                snapshot = cap(pros[i])
                exchange(i, j, snapshot)
                if rng.random() < 0.5:
                    exchange(i, j, snapshot)  # duplicate
                if rng.random() < 0.5:
                    reverse = cap(pros[j])
                    exchange(j, i, reverse)

            elif kind == "replay":
                log = delivered[i]
                if log:
                    old = log[rng.randrange(len(log))]
                    before = canonical(pros[i].snapshot())
                    pros[i].merge(wire_restore(old))
                    refs[i].merge_wire(copy.deepcopy(old))
                    self.assertEqual(canonical(pros[i].snapshot()), before)

            elif kind == "restore":
                snapshot = cap(pros[i])
                pros[i] = wire_restore(snapshot)
                fresh = ReferenceORSet(ids[i])
                fresh.merge_wire(copy.deepcopy(snapshot))
                refs[i] = fresh

            assert_matches(f"step {step} ({kind})")

        # Reconnect: flood pairs in both directions until quiescence.
        for _ in range(2 * n + 3):
            changed = False
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    snapshot_ij = cap(pros[i])
                    before_j = canonical(pros[j].snapshot())
                    pros[j].merge(wire_restore(snapshot_ij))
                    refs[j].merge_wire(copy.deepcopy(snapshot_ij))
                    snapshot_ji = cap(pros[j])
                    before_i = canonical(pros[i].snapshot())
                    pros[i].merge(wire_restore(snapshot_ji))
                    refs[i].merge_wire(copy.deepcopy(snapshot_ji))
                    if canonical(pros[j].snapshot()) != before_j:
                        changed = True
                    if canonical(pros[i].snapshot()) != before_i:
                        changed = True
            assert_matches("flood")
            if not changed:
                break

        baseline = canonical(pros[0].snapshot())
        for i in range(n):
            self.assertEqual(canonical(pros[i].snapshot()), baseline)
            self.assertEqual(pros[i].elements(), refs[i].elements())

        # Every stale message ever delivered is idempotent at convergence.
        for i in range(n):
            for old in delivered[i][:6]:
                before = canonical(pros[i].snapshot())
                pros[i].merge(wire_restore(old))
                refs[i].merge_wire(copy.deepcopy(old))
                self.assertEqual(canonical(pros[i].snapshot()), before)
            assert_matches("stale replay")

        # The converged set keeps accepting adds and removes and reconverges.
        for i in range(n):
            pros[i].add("final")
            refs[i].add("final")
            if rng.random() < 0.5:
                pros[i].compact()
                refs[i].compact()
        for i in range(n):
            for j in range(n):
                if i != j:
                    pros[i].merge(wire_restore(cap(pros[j])))
                    refs[i].merge_wire(copy.deepcopy(cap(pros[j])))
        assert_matches("post-convergence edits")
        final = canonical(pros[0].snapshot())
        for i in range(1, n):
            self.assertEqual(canonical(pros[i].snapshot()), final)

    def test_many_seeds_three_replicas(self) -> None:
        for seed in range(60):
            with self.subTest(seed=seed):
                self.run_trace(seed, n=3, steps=40)

    def test_many_seeds_four_replicas(self) -> None:
        for seed in range(20):
            with self.subTest(seed=seed):
                self.run_trace(1000 + seed, n=4, steps=50)


if __name__ == "__main__":
    unittest.main()
