import json
import unittest

from crdt_sync import GCounter


class TestConstruction(unittest.TestCase):
    def test_valid_replica_id(self):
        c = GCounter("A")
        self.assertEqual(c.value, 0)

    def test_replica_id_type_error(self):
        for bad in (None, 1, 1.5, b"A", ["A"]):
            with self.assertRaises(TypeError):
                GCounter(bad)

    def test_empty_replica_id_value_error(self):
        with self.assertRaises(ValueError):
            GCounter("")


class TestIncrement(unittest.TestCase):
    def test_default_increment(self):
        c = GCounter("A")
        c.increment()
        self.assertEqual(c.value, 1)

    def test_explicit_amount(self):
        c = GCounter("A")
        c.increment(5)
        self.assertEqual(c.value, 5)

    def test_amount_type_error(self):
        c = GCounter("A")
        for bad in (1.0, "1", None, True, False):
            with self.assertRaises(TypeError, msg=repr(bad)):
                c.increment(bad)

    def test_amount_value_error(self):
        c = GCounter("A")
        for bad in (0, -1, -100):
            with self.assertRaises(ValueError, msg=repr(bad)):
                c.increment(bad)

    def test_failed_increment_leaves_state_unchanged(self):
        c = GCounter("A")
        c.increment(3)
        for bad in (0, -1, True, 1.5):
            try:
                c.increment(bad)
            except (TypeError, ValueError):
                pass
        self.assertEqual(c.value, 3)

    def test_huge_amount_uses_python_int_semantics(self):
        c = GCounter("A")
        c.increment(10**100)
        self.assertEqual(c.value, 10**100)


class TestMerge(unittest.TestCase):
    def test_merge_takes_component_max(self):
        a = GCounter("A")
        b = GCounter("B")
        a.increment(2)
        b.increment(3)
        a.merge(b)
        self.assertEqual(a.value, 5)
        self.assertEqual(a.snapshot()["counts"], {"A": 2, "B": 3})

    def test_merge_returns_self(self):
        a = GCounter("A")
        b = GCounter("B")
        self.assertIs(a.merge(b), a)

    def test_merge_does_not_modify_argument(self):
        a = GCounter("A")
        b = GCounter("B")
        a.increment(5)
        before = b.snapshot()
        b.merge(a)
        b.merge(a)
        a.merge(b)
        self.assertNotEqual(b.snapshot(), before)  # b did merge a
        # now check the argument side: merging b into a must not change b
        snapshot_b = b.snapshot()
        a.merge(b)
        self.assertEqual(b.snapshot(), snapshot_b)

    def test_merge_type_error(self):
        a = GCounter("A")
        for bad in (None, 1, "x", {"A": 1}, object()):
            with self.assertRaises(TypeError):
                a.merge(bad)

    def test_merge_is_idempotent(self):
        a = GCounter("A")
        b = GCounter("B")
        a.increment(4)
        b.increment(7)
        b.merge(a)
        once = b.snapshot()
        b.merge(a)
        b.merge(a)
        self.assertEqual(b.snapshot(), once)
        self.assertEqual(b.value, 11)

    def test_convergence_any_order_with_duplicates(self):
        a = GCounter("A")
        b = GCounter("B")
        c = GCounter("C")
        a.increment(1)
        b.increment(2)
        c.increment(3)

        # order 1: a <- b <- c, then broadcast
        a1, b1, c1 = GCounter.from_snapshot(a.snapshot()), GCounter.from_snapshot(
            b.snapshot()
        ), GCounter.from_snapshot(c.snapshot())
        a1.merge(b1).merge(c1)
        b1.merge(a1).merge(c1)
        c1.merge(a1).merge(b1)

        # order 2: reverse, with duplicate deliveries
        a2, b2, c2 = GCounter.from_snapshot(a.snapshot()), GCounter.from_snapshot(
            b.snapshot()
        ), GCounter.from_snapshot(c.snapshot())
        c2.merge(b2).merge(b2).merge(a2)
        b2.merge(c2).merge(a2).merge(a2)
        a2.merge(c2).merge(b2).merge(c2)

        for group in ((a1, b1, c1), (a2, b2, c2)):
            for counter in group:
                self.assertEqual(counter.value, 6)
                self.assertEqual(
                    counter.snapshot()["counts"], {"A": 1, "B": 2, "C": 3}
                )
        self.assertEqual(a1.snapshot(), a2.snapshot())
        self.assertEqual(b1.snapshot(), b2.snapshot())
        self.assertEqual(c1.snapshot(), c2.snapshot())


class TestSnapshot(unittest.TestCase):
    def test_snapshot_structure(self):
        a = GCounter("A")
        a.increment(2)
        self.assertEqual(a.snapshot(), {"replica_id": "A", "counts": {"A": 2}})

    def test_snapshot_keys_sorted(self):
        a = GCounter("b")
        b = GCounter("a")
        c = GCounter("c")
        a.increment(1)
        b.increment(1)
        c.increment(1)
        a.merge(c).merge(b)
        self.assertEqual(list(a.snapshot()["counts"].keys()), ["a", "b", "c"])

    def test_snapshot_is_json_serializable(self):
        a = GCounter("A")
        a.increment(2)
        round_tripped = json.loads(json.dumps(a.snapshot()))
        self.assertEqual(round_tripped, a.snapshot())

    def test_mutating_snapshot_does_not_affect_counter(self):
        a = GCounter("A")
        a.increment(2)
        snap = a.snapshot()
        snap["counts"]["A"] = 999
        snap["counts"]["ZZZ"] = 1
        snap["replica_id"] = "hacked"
        self.assertEqual(a.value, 2)
        self.assertEqual(a.snapshot(), {"replica_id": "A", "counts": {"A": 2}})

    def test_instances_do_not_share_state(self):
        a = GCounter("A")
        b = GCounter("A")
        a.increment(3)
        self.assertEqual(b.value, 0)
        restored = GCounter.from_snapshot(a.snapshot())
        restored.increment(1)
        self.assertEqual(a.value, 3)
        self.assertEqual(restored.value, 4)


class TestFromSnapshot(unittest.TestCase):
    def test_round_trip(self):
        a = GCounter("A")
        b = GCounter("B")
        a.increment(2)
        b.increment(3)
        a.merge(b)
        restored = GCounter.from_snapshot(a.snapshot())
        self.assertEqual(restored.snapshot(), a.snapshot())
        self.assertEqual(restored.value, 5)

    def test_restored_counter_can_increment(self):
        a = GCounter("A")
        a.increment(2)
        restored = GCounter.from_snapshot(a.snapshot())
        restored.increment()
        self.assertEqual(restored.value, 3)
        self.assertEqual(restored.snapshot()["counts"]["A"], 3)

    def test_zero_components_preserved(self):
        snap = {"replica_id": "A", "counts": {"A": 0, "B": 0}}
        restored = GCounter.from_snapshot(snap)
        self.assertEqual(restored.snapshot()["counts"], {"A": 0, "B": 0})
        self.assertEqual(restored.value, 0)

    def test_type_error_for_non_dict(self):
        for bad in (None, 1, "x", [], (("replica_id", "A"),)):
            with self.assertRaises(TypeError):
                GCounter.from_snapshot(bad)

    def test_value_error_for_missing_or_extra_fields(self):
        with self.assertRaises(ValueError):
            GCounter.from_snapshot({"replica_id": "A"})
        with self.assertRaises(ValueError):
            GCounter.from_snapshot({"counts": {"A": 1}})
        with self.assertRaises(ValueError):
            GCounter.from_snapshot(
                {"replica_id": "A", "counts": {"A": 1}, "extra": 1}
            )
        with self.assertRaises(ValueError):
            GCounter.from_snapshot({})

    def test_value_error_for_bad_replica_id(self):
        for bad in ("", None, 1, b"A"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                GCounter.from_snapshot({"replica_id": bad, "counts": {}})

    def test_value_error_for_bad_counts(self):
        with self.assertRaises(ValueError):
            GCounter.from_snapshot({"replica_id": "A", "counts": [("A", 1)]})
        with self.assertRaises(ValueError):
            GCounter.from_snapshot({"replica_id": "A", "counts": None})

    def test_value_error_for_bad_component_keys(self):
        for bad_key in ("", 1, None, b"A"):
            with self.assertRaises(ValueError, msg=repr(bad_key)):
                GCounter.from_snapshot(
                    {"replica_id": "A", "counts": {bad_key: 1}}
                )

    def test_value_error_for_bad_component_values(self):
        for bad_val in (-1, 1.5, "1", None, True, False):
            with self.assertRaises(ValueError, msg=repr(bad_val)):
                GCounter.from_snapshot(
                    {"replica_id": "A", "counts": {"A": bad_val}}
                )

    def test_huge_values_round_trip(self):
        big = 10**100
        snap = {"replica_id": "A", "counts": {"A": big}}
        restored = GCounter.from_snapshot(snap)
        self.assertEqual(restored.value, big)
        restored.increment(1)
        self.assertEqual(restored.value, big + 1)

    def test_input_dict_not_consumed_by_reference(self):
        counts = {"A": 1}
        restored = GCounter.from_snapshot({"replica_id": "A", "counts": counts})
        counts["A"] = 999
        self.assertEqual(restored.value, 1)


if __name__ == "__main__":
    unittest.main()
