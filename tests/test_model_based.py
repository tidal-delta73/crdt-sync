"""Model-based, seed-reproducible convergence tests for the public CRDT API.

For each CRDT type exposed by the current baseline (``GCounter``, ``ORSet``,
``LWWRegister``, ``RGA``) this suite runs randomized scenarios that:

* start three or more replicas (distinct participant ids) from the same empty
  initial state and let them edit concurrently while "offline";
* record every local operation and every sync message in an explicit trace;
* shuffle cross-replica delivery order, duplicate messages, split batches
  (a snapshot fanned out to several replicas is delivered piecemeal) and
  delay arbitrary messages until a final reconnect phase;
* reconnect by repeatedly applying the public ``merge`` entry point until
  every queued message is drained and a fixed point is reached;
* check the originating replica against an independent abstract model after
  every local operation, and check every receiver after every delivery;
* after all messages are delivered, compare the public value *and* a
  canonical serialization of every replica, and probe idempotence
  (redelivering a stale snapshot and merging the same state twice must be
  no-ops).

The abstract models below are small, self-contained re-implementations
written directly from each type's documented contract; they share no code
with the production merge paths. The ORSet model, for example, tracks
harness-assigned addition ids rather than the production tag scheme.

Everything is driven through the public API only (operations, ``merge``,
``snapshot``/``from_snapshot`` and the public readers); no private fields
are inspected.

Determinism: a scenario is a pure function of ``(crdt_type, seed, n_ops,
replicas)`` — the same seed, initial state and operation count reproduce the
exact same participants, operations and delivery trace. On failure the
report includes the seed, the CRDT type, the original operation and delivery
traces, the first divergence position, and an automatically shrunk event
trace (delta-debugging over the recorded events) that still reproduces a
divergence and can be replayed directly via ``replay_events``.

Compaction: the current baseline exposes no tombstone-compaction entry point
on any of the four types, so — per the task constraints — no compaction
behavior is fabricated or tested here. Existing deterministic example tests
and the public exception contract are untouched.
"""

from __future__ import annotations

import copy
import json
import random
import unittest

from crdt_sync import GCounter, LWWRegister, ORSet, RGA

REPLICAS = ("alpha", "beta", "gamma")
REAL_CLASSES = {
    "gcounter": GCounter,
    "orset": ORSet,
    "lww": LWWRegister,
    "rga": RGA,
}
CRDT_TYPES = tuple(REAL_CLASSES)

ORSET_POOL = ("red", "green", "blue", "cat", "dog")
RGA_POOL = ("a", "b", "c", "d")
LWW_POOL = (
    None,
    True,
    False,
    "x",
    "yz",
    0,
    7,
    2.5,
    [1, "a"],
    {"k": [1, 2]},
    {"a": None},
)

SEEDS_PER_TYPE = 6
OPS_PER_SCENARIO = 60
MAX_SYNC_ROUNDS = 8
SHRINK_BUDGET = 400  # max replay attempts while minimizing a failing trace


# ---------------------------------------------------------------------------
# Independent abstract models (spec re-implementations, no production code).
# ---------------------------------------------------------------------------


class GCounterModel:
    """Per-component counts; merge takes the component-wise maximum."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def increment(self, replica: str, amount: int) -> None:
        self.counts[replica] = self.counts.get(replica, 0) + amount

    def merge(self, other: "GCounterModel") -> None:
        for replica, count in other.counts.items():
            if count > self.counts.get(replica, 0):
                self.counts[replica] = count

    def value(self) -> int:
        return sum(self.counts.values())


class ORSetModel:
    """Adds carry harness-minted addition ids; removes tombstone observed ids."""

    def __init__(self) -> None:
        self.element_of: dict[int, str] = {}
        self.known: set[int] = set()
        self.removed: set[int] = set()

    def add(self, oid: int, element: str) -> None:
        self.element_of[oid] = element
        self.known.add(oid)

    def remove(self, element: str) -> bool:
        live = {
            oid
            for oid in self.known
            if oid not in self.removed and self.element_of[oid] == element
        }
        self.removed |= live
        return bool(live)

    def merge(self, other: "ORSetModel") -> None:
        for oid in other.known - self.known:
            self.element_of[oid] = other.element_of[oid]
        self.known |= other.known
        self.removed |= other.removed

    def elements(self) -> set[str]:
        return {
            self.element_of[oid]
            for oid in self.known
            if oid not in self.removed
        }


class LWWModel:
    """Winner is the greatest (count, replica) timestamp; clock is the max seen."""

    def __init__(self) -> None:
        self.clock = 0
        self.timestamp: tuple[int, str] | None = None
        self.stored: object = None
        self.has = False

    def assign(self, replica: str, value: object) -> None:
        self.clock += 1
        self.timestamp = (self.clock, replica)
        self.stored = copy.deepcopy(value)
        self.has = True

    def merge(self, other: "LWWModel") -> None:
        if other.has and (
            not self.has
            or (other.timestamp is not None and other.timestamp > self.timestamp)
        ):
            self.timestamp = other.timestamp
            self.stored = copy.deepcopy(other.stored)
            self.has = True
        if other.clock > self.clock:
            self.clock = other.clock

    def view(self) -> tuple[bool, object]:
        return (self.has, copy.deepcopy(self.stored) if self.has else None)


class RGAModel:
    """Spec weave: siblings ordered by id descending, branches contiguous."""

    def __init__(self, replica: str) -> None:
        self.replica = replica
        self.counter = 0
        self.nodes: dict[tuple[int, str], tuple[str, tuple[int, str] | None]] = {}
        self.tombstones: set[tuple[int, str]] = set()

    def _visible_ids(self) -> list[tuple[int, str]]:
        children: dict[tuple[int, str] | None, list[tuple[int, str]]] = {}
        for node_id, (_value, predecessor) in self.nodes.items():
            children.setdefault(predecessor, []).append(node_id)
        for siblings in children.values():
            siblings.sort(reverse=True)
        ordered: list[tuple[int, str]] = []
        stack = list(reversed(children.get(None, ())))
        while stack:
            node_id = stack.pop()
            ordered.append(node_id)
            stack.extend(reversed(children.get(node_id, ())))
        return [n for n in ordered if n not in self.tombstones]

    def values(self) -> list[str]:
        return [self.nodes[n][0] for n in self._visible_ids()]

    def insert(self, index: int, value: str) -> None:
        visible = self._visible_ids()
        if index < 0 or index > len(visible):
            raise IndexError("model insert index out of range")
        predecessor = visible[index - 1] if index > 0 else None
        self.counter += 1
        self.nodes[(self.counter, self.replica)] = (value, predecessor)

    def delete(self, index: int) -> str:
        visible = self._visible_ids()
        if index < 0 or index >= len(visible):
            raise IndexError("model delete index out of range")
        node_id = visible[index]
        self.tombstones.add(node_id)
        return self.nodes[node_id][0]

    def merge(self, other: "RGAModel") -> None:
        for node_id, record in other.nodes.items():
            if node_id not in self.nodes:
                self.nodes[node_id] = record
        self.tombstones |= other.tombstones
        if other.counter > self.counter:
            self.counter = other.counter


MODEL_FACTORIES = {
    "gcounter": lambda replica: GCounterModel(),
    "orset": lambda replica: ORSetModel(),
    "lww": lambda replica: LWWModel(),
    "rga": lambda replica: RGAModel(replica),
}


# ---------------------------------------------------------------------------
# Divergence and engine plumbing.
# ---------------------------------------------------------------------------


class Divergence(Exception):
    """A mismatch between the production CRDT and the abstract model."""

    def __init__(self, where: str, detail: str) -> None:
        super().__init__(f"{where}: {detail}")
        self.where = where
        self.detail = detail
        self.events: list[dict] | None = None


class SkipOp(Exception):
    """Raised when a replayed (shrunk) operation is no longer legal."""


class Engine:
    """Holds the real replicas, their models and the pending message queue."""

    def __init__(self, crdt_type: str, replicas) -> None:
        self.type = crdt_type
        self.replicas = tuple(replicas)
        self.real = {r: REAL_CLASSES[crdt_type](r) for r in self.replicas}
        self.model = {r: MODEL_FACTORIES[crdt_type](r) for r in self.replicas}
        # msg id -> [dst, wire snapshot, model payload, remaining copies]
        self.queue: dict[int, list] = {}
        self.next_msg = 0
        self.next_oid = 0


def restore(crdt_type: str, snapshot: dict):
    """Restore a snapshot after a full JSON round trip (simulated wire)."""
    return REAL_CLASSES[crdt_type].from_snapshot(
        json.loads(json.dumps(snapshot))
    )


def public_view(crdt_type: str, real) -> object:
    if crdt_type == "gcounter":
        return real.value()
    if crdt_type == "orset":
        return sorted(real.elements())
    if crdt_type == "lww":
        return (real.has_value(), real.value() if real.has_value() else None)
    return list(real.values())


def model_view(crdt_type: str, model) -> object:
    if crdt_type == "gcounter":
        return model.value()
    if crdt_type == "orset":
        return sorted(model.elements())
    if crdt_type == "lww":
        return model.view()
    return model.values()


def canonical_form(crdt_type: str, real) -> str:
    """Canonical serialization used for cross-replica comparison.

    ``replica_id`` is always replica-local. The ORSet ``counter`` only
    describes the replica's *own* addition history (merges never adopt a
    peer's counter), so it legitimately differs between converged replicas
    and is excluded. The RGA counter and the LWW clock are Lamport-style
    maxima that do converge, so they stay in.
    """
    snap = json.loads(json.dumps(real.snapshot()))
    snap.pop("replica_id", None)
    if crdt_type == "orset":
        snap.pop("counter", None)
    return json.dumps(snap, sort_keys=True)


def check_view(ctx: Engine, replica: str, where: str) -> None:
    expected = model_view(ctx.type, ctx.model[replica])
    actual = public_view(ctx.type, ctx.real[replica])
    if expected != actual:
        raise Divergence(
            where,
            f"replica {replica!r} view diverged: "
            f"expected (model) {expected!r}, actual (crdt) {actual!r}",
        )


def apply_op(ctx: Engine, ev: dict, where: str) -> None:
    replica = ev["replica"]
    real = ctx.real[replica]
    model = ctx.model[replica]
    if ctx.type == "gcounter":
        real.increment(ev["amount"])
        model.increment(replica, ev["amount"])
    elif ctx.type == "orset":
        if ev["op"] == "add":
            real.add(ev["element"])
            model.add(ev["oid"], ev["element"])
        else:
            got = real.remove(ev["element"])
            want = model.remove(ev["element"])
            if got != want:
                raise Divergence(
                    where,
                    f"replica {replica!r} remove({ev['element']!r}) returned "
                    f"{got!r}, model expected {want!r}",
                )
    elif ctx.type == "lww":
        real.assign(copy.deepcopy(ev["value"]))
        model.assign(replica, ev["value"])
    else:  # rga
        length = len(real.values())
        if ev["op"] == "insert":
            if ev["index"] < 0 or ev["index"] > length:
                raise SkipOp
            real.insert(ev["index"], ev["value"])
            try:
                model.insert(ev["index"], ev["value"])
            except IndexError:
                raise Divergence(
                    where,
                    f"replica {replica!r} model rejected insert index "
                    f"{ev['index']} that the crdt accepted",
                )
        else:
            if ev["index"] < 0 or ev["index"] >= length:
                raise SkipOp
            got = real.delete(ev["index"])
            try:
                want = model.delete(ev["index"])
            except IndexError:
                raise Divergence(
                    where,
                    f"replica {replica!r} model rejected delete index "
                    f"{ev['index']} that the crdt accepted",
                )
            if got != want:
                raise Divergence(
                    where,
                    f"replica {replica!r} delete({ev['index']}) returned "
                    f"{got!r}, model expected {want!r}",
                )


def apply_event(ctx: Engine, events: list[dict], i: int) -> None:
    ev = events[i]
    where = f"event {i} {json.dumps(ev, sort_keys=True)}"
    kind = ev["ev"]
    if kind == "op":
        apply_op(ctx, ev, where)
        check_view(ctx, ev["replica"], where)
    elif kind == "send":
        ctx.queue[ev["msg"]] = [
            ev["dst"],
            json.loads(json.dumps(ctx.real[ev["src"]].snapshot())),
            copy.deepcopy(ctx.model[ev["src"]]),
            ev["copies"],
        ]
    else:  # deliver
        entry = ctx.queue.get(ev["msg"])
        if entry is None:
            return  # only possible in shrunk replays
        dst, wire, model_payload, _copies = entry
        ctx.real[dst].merge(restore(ctx.type, wire))
        ctx.model[dst].merge(copy.deepcopy(model_payload))
        entry[3] -= 1
        if entry[3] == 0:
            del ctx.queue[ev["msg"]]
        check_view(ctx, dst, where)


def post_phase(ctx: Engine) -> dict[str, str]:
    """Reconnect: sync to a fixed point, then run the final assertions.

    Returns the per-replica canonical forms (all identical on success).
    """
    replicas = sorted(ctx.real)
    stale = json.loads(json.dumps(ctx.real[replicas[0]].snapshot()))

    converged = False
    for _round in range(MAX_SYNC_ROUNDS):
        for src in replicas:
            wire = json.loads(json.dumps(ctx.real[src].snapshot()))
            model_payload = copy.deepcopy(ctx.model[src])
            for dst in replicas:
                if src == dst:
                    continue
                ctx.real[dst].merge(restore(ctx.type, wire))
                ctx.model[dst].merge(copy.deepcopy(model_payload))
        canonicals = {r: canonical_form(ctx.type, ctx.real[r]) for r in replicas}
        if len(set(canonicals.values())) == 1:
            converged = True
            break
    if not converged:
        detail = "; ".join(
            f"{r}: {c}" for r, c in sorted(canonicals.items())
        )
        raise Divergence(
            "post:converge",
            f"replicas did not converge after {MAX_SYNC_ROUNDS} sync "
            f"rounds: {detail}",
        )

    for replica in replicas:
        check_view(ctx, replica, "post:view")

    # Idempotence: merging the same state twice in a row changes nothing.
    for replica in replicas:
        before = canonical_form(ctx.type, ctx.real[replica])
        snap = json.loads(json.dumps(ctx.real[replica].snapshot()))
        ctx.real[replica].merge(restore(ctx.type, snap))
        ctx.real[replica].merge(restore(ctx.type, snap))
        after = canonical_form(ctx.type, ctx.real[replica])
        if after != before:
            raise Divergence(
                "post:idempotence",
                f"replica {replica!r} changed after merging the same state "
                f"twice: before {before}, after {after}",
            )

    # Stale redelivery: an old pre-reconnect snapshot must be a no-op.
    for replica in replicas:
        before = canonical_form(ctx.type, ctx.real[replica])
        ctx.real[replica].merge(restore(ctx.type, stale))
        after = canonical_form(ctx.type, ctx.real[replica])
        if after != before:
            raise Divergence(
                "post:stale-redelivery",
                f"replica {replica!r} changed after redelivering a stale "
                f"snapshot: before {before}, after {after}",
            )

    return {r: canonical_form(ctx.type, ctx.real[r]) for r in replicas}


# ---------------------------------------------------------------------------
# Scenario generation (seeded) and replay (seed-free, trace-driven).
# ---------------------------------------------------------------------------


def _gen_op(rng: random.Random, ctx: Engine, replica: str) -> dict:
    if ctx.type == "gcounter":
        return {
            "ev": "op",
            "op": "inc",
            "replica": replica,
            "amount": rng.randint(1, 5),
        }
    if ctx.type == "orset":
        visible = sorted(ctx.real[replica].elements())
        if rng.random() < 0.4:
            if visible and rng.random() < 0.75:
                element = rng.choice(visible)
            else:
                element = rng.choice(ORSET_POOL)
            return {"ev": "op", "op": "remove", "replica": replica,
                    "element": element}
        oid = ctx.next_oid
        ctx.next_oid += 1
        return {"ev": "op", "op": "add", "replica": replica,
                "element": rng.choice(ORSET_POOL), "oid": oid}
    if ctx.type == "lww":
        return {"ev": "op", "op": "assign", "replica": replica,
                "value": copy.deepcopy(rng.choice(LWW_POOL))}
    # rga
    values = ctx.real[replica].values()
    if values and rng.random() < 0.35:
        return {"ev": "op", "op": "delete", "replica": replica,
                "index": rng.randrange(len(values))}
    return {"ev": "op", "op": "insert", "replica": replica,
            "index": rng.randint(0, len(values)),
            "value": rng.choice(RGA_POOL)}


def generate_and_run(
    crdt_type: str, seed: int, n_ops: int, replicas=REPLICAS
) -> tuple[list[dict], dict[str, str]]:
    """Generate a seeded scenario, execute it with checks, return its trace.

    The trace plus the post-phase fully determine the final state, so the
    returned ``(events, canonicals)`` pair is reproducible from the seed.
    """
    rng = random.Random(
        f"model-based|{crdt_type}|{seed}|{n_ops}|{','.join(replicas)}"
    )
    ctx = Engine(crdt_type, replicas)
    events: list[dict] = []
    try:
        for _step in range(n_ops):
            roll = rng.random()
            if roll < 0.55 or (roll >= 0.8 and not ctx.queue):
                ev = _gen_op(rng, ctx, rng.choice(ctx.replicas))
                events.append(ev)
                apply_event(ctx, events, len(events) - 1)
            elif roll < 0.8:
                src = rng.choice(ctx.replicas)
                others = [r for r in ctx.replicas if r != src]
                # One fan-out batch; delivery below splits it piecemeal.
                for dst in rng.sample(others, rng.randint(1, len(others))):
                    copies = (
                        1
                        + int(rng.random() < 0.35)
                        + int(rng.random() < 0.15)
                    )
                    ev = {
                        "ev": "send",
                        "src": src,
                        "dst": dst,
                        "msg": ctx.next_msg,
                        "copies": copies,
                    }
                    ctx.next_msg += 1
                    events.append(ev)
                    apply_event(ctx, events, len(events) - 1)
            else:
                ids = sorted(ctx.queue)
                for msg in rng.sample(ids, rng.randint(1, min(3, len(ids)))):
                    ev = {"ev": "deliver", "msg": msg}
                    events.append(ev)
                    apply_event(ctx, events, len(events) - 1)
        # Reconnect: drain every delayed message, in random order.
        while ctx.queue:
            ev = {"ev": "deliver", "msg": rng.choice(sorted(ctx.queue))}
            events.append(ev)
            apply_event(ctx, events, len(events) - 1)
        canonicals = post_phase(ctx)
    except Divergence as div:
        div.events = events
        raise
    return events, canonicals


def replay_events(
    crdt_type: str, events, replicas=REPLICAS
) -> dict[str, str]:
    """Replay a recorded trace (no randomness) and run all checks.

    Operations that became illegal through trace shrinking are skipped;
    everything else must behave exactly as in the original run.
    """
    ctx = Engine(crdt_type, replicas)
    events = list(events)
    for i in range(len(events)):
        try:
            apply_event(ctx, events, i)
        except SkipOp:
            continue
    return post_phase(ctx)


# ---------------------------------------------------------------------------
# Shrinking: delta-debug the recorded trace down to a shorter failing one.
# ---------------------------------------------------------------------------


def minimize(crdt_type: str, replicas, events: list[dict]) -> list[dict]:
    """Return a 1-minimal-ish subsequence of ``events`` that still diverges."""
    attempts = [0]

    def fails(candidate: list[dict]) -> bool:
        if not candidate or attempts[0] >= SHRINK_BUDGET:
            return False
        attempts[0] += 1
        try:
            replay_events(crdt_type, candidate, replicas=replicas)
        except Divergence:
            return True
        return False

    events = list(events)
    if not fails(events):
        return events
    n = 2
    while len(events) >= 2:
        chunk = max(1, len(events) // n)
        reduced = False
        i = 0
        while i < len(events):
            candidate = events[:i] + events[i + chunk:]
            if fails(candidate):
                events = candidate
                reduced = True
                break
            i += chunk
        if attempts[0] >= SHRINK_BUDGET:
            break
        if reduced:
            n = max(2, n - 1)
        else:
            if chunk == 1:
                break
            n = min(len(events), n * 2)
    return events


# ---------------------------------------------------------------------------
# Failure reporting.
# ---------------------------------------------------------------------------


def _render_events(events: list[dict]) -> str:
    lines = ["  operation trace:"]
    for i, ev in enumerate(events):
        if ev["ev"] == "op":
            lines.append(f"    {i:4d} {json.dumps(ev, sort_keys=True)}")
    lines.append("  delivery trace:")
    for i, ev in enumerate(events):
        if ev["ev"] != "op":
            lines.append(f"    {i:4d} {json.dumps(ev, sort_keys=True)}")
    return "\n".join(lines)


def build_report(
    crdt_type: str, seed: int, n_ops: int, replicas, div: Divergence
) -> str:
    events = div.events or []
    shrunk = minimize(crdt_type, replicas, events)
    try:
        replay_events(crdt_type, shrunk, replicas=replicas)
        shrunk_outcome = "<shrunk trace did not diverge on replay>"
    except Divergence as div2:
        shrunk_outcome = f"{div2.where}: {div2.detail}"
    return "\n".join(
        [
            "model-based convergence failure",
            f"seed: {seed}",
            f"crdt type: {crdt_type}",
            f"operations requested: {n_ops}",
            f"first divergence: {div.where}: {div.detail}",
            "",
            f"original trace ({len(events)} events):",
            _render_events(events),
            "",
            f"shrunk failing trace ({len(shrunk)} events):",
            _render_events(shrunk),
            "",
            f"shrunk divergence: {shrunk_outcome}",
            "",
            "replay the shrunk trace with:",
            f"  replay_events({crdt_type!r}, "
            f"{json.dumps(shrunk, sort_keys=True)})",
        ]
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


class ModelBasedScenarioTests(unittest.TestCase):
    def _run_type(self, crdt_type: str) -> None:
        for seed in range(SEEDS_PER_TYPE):
            with self.subTest(crdt=crdt_type, seed=seed):
                try:
                    generate_and_run(
                        crdt_type, seed, OPS_PER_SCENARIO, REPLICAS
                    )
                except Divergence as div:
                    self.fail(
                        build_report(
                            crdt_type, seed, OPS_PER_SCENARIO, REPLICAS, div
                        )
                    )

    def test_gcounter_scenarios(self) -> None:
        self._run_type("gcounter")

    def test_orset_scenarios(self) -> None:
        self._run_type("orset")

    def test_lww_scenarios(self) -> None:
        self._run_type("lww")

    def test_rga_scenarios(self) -> None:
        self._run_type("rga")

    def test_same_seed_reproduces_trace(self) -> None:
        for crdt_type in CRDT_TYPES:
            with self.subTest(crdt=crdt_type):
                first, _ = generate_and_run(crdt_type, 101, 40, REPLICAS)
                second, _ = generate_and_run(crdt_type, 101, 40, REPLICAS)
                self.assertEqual(first, second)

    def test_different_seeds_diverge(self) -> None:
        first, _ = generate_and_run("rga", 1, 40, REPLICAS)
        second, _ = generate_and_run("rga", 2, 40, REPLICAS)
        self.assertNotEqual(first, second)

    def test_scenarios_exercise_delivery_and_duplication(self) -> None:
        # Guard against scenario-generation regressions: every type must
        # produce operations, sends and deliveries, and across the seeds at
        # least one message must be queued with duplicate copies.
        for crdt_type in CRDT_TYPES:
            with self.subTest(crdt=crdt_type):
                saw_duplicates = False
                for seed in range(SEEDS_PER_TYPE):
                    events, _ = generate_and_run(
                        crdt_type, seed, OPS_PER_SCENARIO, REPLICAS
                    )
                    kinds = {ev["ev"] for ev in events}
                    self.assertEqual(kinds, {"op", "send", "deliver"})
                    saw_duplicates |= any(
                        ev["ev"] == "send" and ev["copies"] > 1
                        for ev in events
                    )
                self.assertTrue(saw_duplicates)

    def test_replay_reproduces_final_state(self) -> None:
        for crdt_type in CRDT_TYPES:
            with self.subTest(crdt=crdt_type):
                events, generated = generate_and_run(crdt_type, 7, 40, REPLICAS)
                replayed = replay_events(crdt_type, events)
                self.assertEqual(generated, replayed)


if __name__ == "__main__":
    unittest.main()
