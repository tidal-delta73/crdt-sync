"""Deterministic state-model tests for the public GCounter / ORSet /
LWWRegister / RGA contracts.

These tests drive the four CRDTs only through their published surface —
constructors, the local mutators, the single sync entry the baseline
publishes (``merge``), and ``snapshot`` / ``from_snapshot`` — and judge
correctness solely through public values and normalized (canonical JSON)
snapshots. Correctness is decided against *abstract models* defined in
this module. The models state the plain mathematical semantics (G-Counter
component join, observed-remove sets, last-writer-wins with the documented
timestamp tie-break, the RGA tree weave) independently of the production
merge implementation, so the two cross-check each other.

What is exercised
-----------------
* at least three replicas with distinct, seed-generated ids;
* local operations issued in offline bursts so several replicas edit
  concurrently while partitioned;
* after each burst the replicas' states are captured as JSON-round-tripped
  snapshots and put on a per-receiver queue; the queue randomly reorders
  messages across replicas, duplicates arbitrary messages (the duplicate may
  arrive much later), splits one fan-out across rounds and delays messages;
* between bursts some rounds sync only partially, simulating a still-torn
  network; reconnect then delivers everything and repeatedly floods pair
  merges until no message is left and nothing changes anymore;
* after every local op the origin is checked against its model; after every
  delivered message receiver and model are compared; after convergence every
  replica's public value and normalized shared serialization agrees with the
  independent model, duplicate messages and back-to-back merges of the same
  state change nothing.

Generation uses an explicit ``random.Random(seed)``: the same seed, initial
state and op scale produce the same participants, a fixed *script* (ops +
fault plan) and hence the identical actual delivery trace. On failure the
assertion text reports seed, CRDT type, the raw op trace, the actual
delivery trace, the first divergence, and a shrunk JSON script that can be
replayed directly via :func:`replay_script_by_name`.

Tombstone compaction is published only on ORSet; GCounter, LWWRegister and
RGA still expose no compaction entry. ``CompactionScopeTests`` pins that
scope so extending compaction to another CRDT is a deliberate decision.
"""

from __future__ import annotations

import json
import random
import unittest
from typing import Any, Callable, Optional

from crdt_sync import GCounter, LWWRegister, ORSet, RGA

# ---------------------------------------------------------------------------
# Wire helpers / canonical serialization
# ---------------------------------------------------------------------------


def wire(snapshot: object) -> object:
    """Round-trip a snapshot through JSON, as a real network boundary would."""
    return json.loads(json.dumps(snapshot))


def normalized(snapshot: object) -> str:
    """Canonical JSON fingerprint of a snapshot or a shared-view dict."""
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Abstract models — written independently of the production merge code
# ---------------------------------------------------------------------------


class GCModel:
    """G-Counter: per-replica cumulative totals joined by component max."""

    def __init__(self, replica: str) -> None:
        self.replica = replica
        self.counts: dict[str, int] = {}

    def increment(self, amount: int) -> None:
        self.counts[self.replica] = self.counts.get(self.replica, 0) + amount

    def merge(self, other: "GCModel") -> None:
        for rid, amount in other.counts.items():
            if amount > self.counts.get(rid, 0):
                self.counts[rid] = amount

    def value(self) -> int:
        return sum(self.counts.values())

    def snapshot(self) -> dict[str, object]:
        return {
            "replica_id": self.replica,
            "counts": {k: self.counts[k] for k in sorted(self.counts)},
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "GCModel":
        model = cls(snapshot["replica_id"])
        model.counts = dict(snapshot["counts"])
        return model


class ORModel:
    """Observed-remove set: unique add tags, tombstones observed adds only."""

    def __init__(self, replica: str) -> None:
        self.replica = replica
        self.counter = 0
        self.added: dict[str, set[tuple[str, int]]] = {}
        self.removed: set[tuple[str, int]] = set()

    def add(self, element: str) -> None:
        self.counter += 1
        self.added.setdefault(element, set()).add((self.replica, self.counter))

    def remove(self, element: str) -> bool:
        tags = self.added.get(element, frozenset())
        live = {tag for tag in tags if tag not in self.removed}
        if not live:
            return False
        self.removed.update(live)
        return True

    def elements(self) -> set[str]:
        return {
            element
            for element, tags in self.added.items()
            if any(tag not in self.removed for tag in tags)
        }

    def merge(self, other: "ORModel") -> None:
        for element, tags in other.added.items():
            self.added.setdefault(element, set()).update(tags)
        self.removed.update(other.removed)
        # Future local tags must stay unique even if our own tag history
        # arrives back relayed through another replica.
        for tags in self.added.values():
            for origin, sequence in tags:
                if origin == self.replica and sequence > self.counter:
                    self.counter = sequence

    def snapshot(self) -> dict[str, object]:
        return {
            "replica_id": self.replica,
            "counter": self.counter,
            "adds": {
                element: [[origin, seq] for origin, seq in sorted(tags)]
                for element, tags in sorted(self.added.items())
            },
            "removes": [[origin, seq] for origin, seq in sorted(self.removed)],
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "ORModel":
        model = cls(snapshot["replica_id"])
        model.counter = snapshot["counter"]
        model.added = {
            element: {tuple(tag) for tag in tags}
            for element, tags in snapshot["adds"].items()
        }
        model.removed = {tuple(tag) for tag in snapshot["removes"]}
        return model


class LWWModel:
    """Last-writer-wins register: (count, replica-id) timestamp, clock max."""

    def __init__(self, replica: str) -> None:
        self.replica = replica
        self.clock = 0
        self.winner: Optional[tuple[int, str]] = None
        self.stored: Any = None
        self.has = False

    def assign(self, value: Any) -> None:
        self.clock += 1
        self.winner = (self.clock, self.replica)
        self.stored = json.loads(json.dumps(value))
        self.has = True

    def has_value(self) -> bool:
        return self.has

    def value(self) -> Any:
        return json.loads(json.dumps(self.stored))

    def merge(self, other: "LWWModel") -> None:
        if other.has and (not self.has or other.winner > self.winner):
            self.winner = other.winner
            self.stored = json.loads(json.dumps(other.stored))
            self.has = True
        if other.clock > self.clock:
            self.clock = other.clock

    def snapshot(self) -> dict[str, object]:
        entry: object = None
        if self.has:
            entry = {"timestamp": [self.winner[0], self.winner[1]], "value": self.stored}
        return {
            "replica_id": self.replica,
            "clock": self.clock,
            "entry": json.loads(json.dumps(entry)),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "LWWModel":
        model = cls(snapshot["replica_id"])
        model.clock = snapshot["clock"]
        entry = snapshot["entry"]
        if entry is not None:
            count, origin = entry["timestamp"]
            model.winner = (count, origin)
            model.stored = json.loads(json.dumps(entry["value"]))
            model.has = True
        return model


class RGAModel:
    """RGA: node graph off predecessors, tombstones, descending-id weave."""

    ROOT: Optional[tuple[int, str]] = None

    def __init__(self, replica: str) -> None:
        self.replica = replica
        self.counter = 0
        self.nodes: dict[tuple[int, str], tuple[str, Optional[tuple[int, str]]]] = {}
        self.tombstones: set[tuple[int, str]] = set()

    def _visible(self) -> list[tuple[int, str]]:
        children: dict[Optional[tuple[int, str]], list[tuple[int, str]]] = {}
        for nid, (_value, prev) in self.nodes.items():
            children.setdefault(prev, []).append(nid)
        for sibs in children.values():
            sibs.sort(reverse=True)
        ordered: list[tuple[int, str]] = []
        stack = list(reversed(children.get(self.ROOT, ())))
        while stack:
            nid = stack.pop()
            ordered.append(nid)
            stack.extend(reversed(children.get(nid, ())))
        return [nid for nid in ordered if nid not in self.tombstones]

    def values(self) -> list[str]:
        return [self.nodes[nid][0] for nid in self._visible()]

    def insert(self, index: int, value: str) -> None:
        visible = self._visible()
        if index < 0 or index > len(visible):
            raise IndexError("insert index out of range")
        prev = visible[index - 1] if index > 0 else self.ROOT
        self.counter += 1
        self.nodes[(self.counter, self.replica)] = (value, prev)

    def delete(self, index: int) -> str:
        visible = self._visible()
        if index < 0 or index >= len(visible):
            raise IndexError("delete index out of range")
        nid = visible[index]
        value = self.nodes[nid][0]
        self.tombstones.add(nid)
        return value

    def merge(self, other: "RGAModel") -> None:
        for nid, record in other.nodes.items():
            if nid not in self.nodes:
                self.nodes[nid] = record
        self.tombstones.update(other.tombstones)
        if other.counter > self.counter:
            self.counter = other.counter

    def snapshot(self) -> dict[str, object]:
        return {
            "replica_id": self.replica,
            "counter": self.counter,
            "nodes": [
                {
                    "id": [sequence, origin],
                    "value": self.nodes[(sequence, origin)][0],
                    "prev": (
                        None
                        if self.nodes[(sequence, origin)][1] is None
                        else list(self.nodes[(sequence, origin)][1])
                    ),
                }
                for sequence, origin in sorted(self.nodes)
            ],
            "tombstones": [list(nid) for nid in sorted(self.tombstones)],
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "RGAModel":
        model = cls(snapshot["replica_id"])
        model.counter = snapshot["counter"]
        model.nodes = {
            tuple(node["id"]): (node["value"], None if node["prev"] is None else tuple(node["prev"]))
            for node in snapshot["nodes"]
        }
        model.tombstones = {tuple(nid) for nid in snapshot["tombstones"]}
        return model


# ---------------------------------------------------------------------------
# Harness: script generation (models + seeded RNG only) and replay (CRDT +
# model driven through the exact same wire messages)
# ---------------------------------------------------------------------------

ID_POOL = ["alpha", "beta", "gamma", "delta", "epsilon"]
LWW_VALUES = [None, True, False, 0, 1, 3.5, "s", "t", [1, 2, None], {"k": "v"}]
RGA_TOKENS = ["a", "b", "c", "d", "e"]
OR_ELEMENTS = ["x", "y", "z"]


class Harness:
    name = "base"
    crdt_cls: type
    model_cls: type

    def __init__(
        self,
        seed: int,
        n: int = 3,
        scale: int = 20,
        crdt_cls: Optional[type] = None,
        reset_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        self.seed = seed
        self.n = n
        self.scale = scale
        if crdt_cls is not None:
            self.crdt_cls = crdt_cls
        self.reset_hook = reset_hook
        self.delivery_trace: list[list] = []

    # -- per-type hooks --------------------------------------------------

    def make_crdt(self, ident: str) -> Any:
        return self.crdt_cls(ident)

    def make_model(self, ident: str) -> Any:
        return self.model_cls(ident)

    def gen_op(self, rng: random.Random, model: Any) -> list:
        raise NotImplementedError

    def public(self, obj: Any) -> Any:
        """Public-value view; identical method names on CRDT and model."""
        raise NotImplementedError

    def crdt_shared(self, snapshot: dict) -> object:
        raise NotImplementedError

    def model_shared(self, model: Any) -> object:
        raise NotImplementedError

    # -- generation ------------------------------------------------------

    def generate(self) -> dict:
        """Return a fully determined, JSON-serializable execution script."""
        rng = random.Random(self.seed)
        pool = list(ID_POOL)
        rng.shuffle(pool)
        ids = pool[: self.n]
        models = [self.make_model(ident) for ident in ids]
        steps: list[dict] = []
        queue: list[list] = []  # [round, sender, receiver, model-snapshot]
        produced = 0
        burst = 0
        while produced < self.scale:
            actors = rng.sample(range(self.n), k=rng.randint(2, self.n))
            for _ in range(rng.randint(1, 3)):
                for _ in range(rng.randint(1, 2)):
                    if produced >= self.scale:
                        break
                    actor = rng.choice(actors)
                    op = self.gen_op(rng, models[actor])
                    self._model_call(models[actor], op)
                    steps.append({"k": "op", "b": burst, "r": actor, "op": op})
                    produced += 1

            # Broadcast one captured state per actor; then inject the faults.
            net_msgs: list[list] = []
            for sender in actors:
                for receiver in range(self.n):
                    if receiver == sender:
                        continue
                    delays = [0]
                    if rng.random() < 0.18:
                        delays.append(rng.randint(1, 3))  # duplicated, later
                    if rng.random() < 0.08:
                        delays.append(rng.randint(1, 3))  # triplicated
                    if rng.random() < 0.25:
                        delays[0] = rng.randint(1, 3)  # primary delayed
                    for delay in delays:
                        net_msgs.append([sender, receiver, delay])
            rng.shuffle(net_msgs)  # cross-replica reordering / batch splits
            senders = list(dict.fromkeys(msg[0] for msg in net_msgs))
            captured = {sender: models[sender].snapshot() for sender in senders}
            for sender, receiver, delay in net_msgs:
                queue.append([burst + 1 + delay, sender, receiver, captured[sender]])
            steps.append({"k": "net", "b": burst, "msgs": net_msgs})

            if rng.random() < 0.75:
                steps.append({"k": "sync", "b": burst})
                queue = self._gen_drain(queue, burst + 1, models)

            burst += 1

        # Generator side ends connected: flush whatever is still queued.
        self._gen_deliver(queue, models, lambda _r: True)
        return {"crdt": self.name, "ids": ids, "steps": steps}

    def _gen_drain(self, queue: list[list], due: int, models: list) -> list[list]:
        return self._gen_deliver(queue, models, lambda round_no: round_no <= due)

    def _gen_deliver(self, queue: list[list], models: list, due) -> list[list]:
        remaining = []
        for round_no, sender, receiver, snapshot in queue:
            if due(round_no):
                models[receiver].merge(self.model_cls.from_snapshot(snapshot))
            else:
                remaining.append([round_no, sender, receiver, snapshot])
        return remaining

    # -- replay ----------------------------------------------------------

    def replay(self, spec: dict) -> Optional[dict]:
        """Execute a script; return None on success or a failure report dict."""
        if self.reset_hook is not None:
            self.reset_hook()
        ids = spec["ids"]
        crdts = [self.make_crdt(ident) for ident in ids]
        models = [self.make_model(ident) for ident in ids]
        queue: list[list] = []  # [round, sender, receiver, wire snapshot]
        received: list[list] = [[] for _ in ids]
        self.delivery_trace = []
        delivery_index = 0
        effective_steps: list[dict] = []

        def failure(kind: str, where: str, expected: object, actual: object) -> dict:
            return {
                "kind": kind,
                "where": where,
                "expected": expected,
                "actual": actual,
                "spec": spec,
                "effective": {**spec, "steps": list(effective_steps)},
                "delivery": list(self.delivery_trace),
                "seed": self.seed,
                "n": self.n,
                "scale": self.scale,
            }

        for step_index, step in enumerate(spec["steps"]):
            kind = step["k"]
            burst_no = step["b"]

            if kind == "op":
                actor = step["r"]
                op = step["op"]
                try:
                    model_result = self._model_call(models[actor], op)
                except IndexError:
                    # Under a shrunk prefix this op is no longer legal; drop it
                    # from the effective script instead of forcing it in.
                    continue
                try:
                    crdt_result = self._crdt_call(crdts[actor], op)
                except Exception as exc:
                    return failure(
                        "local-exception",
                        f"step {step_index} {ids[actor]}.{op[0]}",
                        "no exception",
                        f"{type(exc).__name__}: {exc}",
                    )
                if crdt_result != model_result:
                    return failure(
                        "local-return",
                        f"step {step_index} {ids[actor]}.{op[0]}{tuple(op[1:])}",
                        model_result,
                        crdt_result,
                    )
                mismatch = self._compare(
                    crdts[actor],
                    models[actor],
                    f"step {step_index} after {ids[actor]}.{op[0]}",
                    failure,
                )
                if mismatch is not None:
                    return mismatch
                effective_steps.append(step)

            elif kind == "net":
                senders = list(dict.fromkeys(msg[0] for msg in step["msgs"]))
                captures = {
                    sender: wire(crdts[sender].snapshot()) for sender in senders
                }
                for sender, receiver, delay in step["msgs"]:
                    queue.append(
                        [burst_no + 1 + delay, sender, receiver, captures[sender]]
                    )
                effective_steps.append(step)

            elif kind == "sync":
                problem, delivery_index = self._drain(
                    queue,
                    burst_no + 1,
                    crdts,
                    models,
                    ids,
                    received,
                    delivery_index,
                    failure,
                    due_only=True,
                )
                if problem is not None:
                    return problem
                effective_steps.append(step)

            else:  # pragma: no cover - guarded by construction
                return failure("bad-script", f"step {step_index}", "known kind", kind)

        # Reconnect: every delayed/duplicated message, whatever its round.
        problem, delivery_index = self._drain(
            queue,
            None,
            crdts,
            models,
            ids,
            received,
            delivery_index,
            failure,
            due_only=False,
        )
        if problem is not None:
            return problem

        problem = self._flood(crdts, models, ids, failure)
        if problem is not None:
            return problem

        # Final cross-replica convergence against each other and the models.
        baseline = normalized(self.crdt_shared(crdts[0].snapshot()))
        for k in range(self.n):
            mismatch = self._compare(
                crdts[k], models[k], f"final convergence: replica {ids[k]}", failure
            )
            if mismatch is not None:
                return mismatch
            if normalized(self.crdt_shared(crdts[k].snapshot())) != baseline:
                return failure(
                    "convergence",
                    f"replica {ids[k]} normalized serialization",
                    baseline,
                    normalized(self.crdt_shared(crdts[k].snapshot())),
                )

        # Duplicate old messages must be idempotent ...
        for k, ident in enumerate(ids):
            log = received[k]
            picks = []
            if log:
                picks = [log[0], log[-1]]
                if len(log) > 2:
                    picks.append(log[len(log) // 2])
            for old in picks:
                for _ in range(2):
                    before = normalized(self.crdt_shared(crdts[k].snapshot()))
                    crdts[k].merge(self.crdt_cls.from_snapshot(wire(old)))
                    models[k].merge(self.model_cls.from_snapshot(old))
                    after = normalized(self.crdt_shared(crdts[k].snapshot()))
                    if after != before:
                        return failure(
                            "idempotence",
                            f"duplicate old message redelivered to {ident}",
                            before,
                            after,
                        )
                    mismatch = self._compare(
                        crdts[k], models[k], f"redelivery to {ident}", failure
                    )
                    if mismatch is not None:
                        return mismatch

        # ... and merging the same current state twice in a row changes nothing.
        for receiver, ident in enumerate(ids):
            for sender in range(self.n):
                snapshot = wire(crdts[sender].snapshot())
                before = normalized(self.crdt_shared(crdts[receiver].snapshot()))
                for _ in range(2):
                    crdts[receiver].merge(self.crdt_cls.from_snapshot(wire(snapshot)))
                    models[receiver].merge(
                        self.model_cls.from_snapshot(wire(snapshot))
                    )
                after = normalized(self.crdt_shared(crdts[receiver].snapshot()))
                if after != before:
                    return failure(
                        "idempotence",
                        f"{ident}: double merge of {ids[sender]} current state",
                        before,
                        after,
                    )
            mismatch = self._compare(
                crdts[receiver], models[receiver], f"post-idempotence {ident}", failure
            )
            if mismatch is not None:
                return mismatch
        return None

    def _drain(
        self,
        queue: list[list],
        due: Optional[int],
        crdts: list,
        models: list,
        ids: list[str],
        received: list[list],
        delivery_index: int,
        failure: Callable,
        due_only: bool,
    ):
        remaining = []
        for round_no, sender, receiver, snapshot in queue:
            if due_only and round_no > due:
                remaining.append([round_no, sender, receiver, snapshot])
                continue
            crdts[receiver].merge(self.crdt_cls.from_snapshot(wire(snapshot)))
            models[receiver].merge(self.model_cls.from_snapshot(snapshot))
            received[receiver].append(snapshot)
            self.delivery_trace.append(
                [
                    delivery_index,
                    round_no,
                    ids[sender],
                    ids[receiver],
                    normalized(self.crdt_shared(crdts[receiver].snapshot())),
                ]
            )
            mismatch = self._compare(
                crdts[receiver],
                models[receiver],
                f"delivery #{delivery_index} (round {round_no}, "
                f"{ids[sender]} -> {ids[receiver]})",
                failure,
            )
            if mismatch is not None:
                return mismatch, delivery_index
            delivery_index += 1
        queue[:] = remaining
        return None, delivery_index

    def _flood(
        self,
        crdts: list,
        models: list,
        ids: list[str],
        failure: Callable,
    ) -> Optional[dict]:
        quiet = 0
        for pass_no in range(2 * self.n + 4):
            changed = False
            order = [(pass_no + offset) % self.n for offset in range(self.n)]
            for i in order:
                for j in order:
                    if i == j:
                        continue
                    snap_i = wire(crdts[i].snapshot())
                    snap_j = wire(crdts[j].snapshot())
                    before_j = normalized(self.crdt_shared(crdts[j].snapshot()))
                    crdts[j].merge(self.crdt_cls.from_snapshot(wire(snap_i)))
                    models[j].merge(self.model_cls.from_snapshot(snap_i))
                    before_i = normalized(self.crdt_shared(crdts[i].snapshot()))
                    crdts[i].merge(self.crdt_cls.from_snapshot(wire(snap_j)))
                    models[i].merge(self.model_cls.from_snapshot(snap_j))
                    if normalized(self.crdt_shared(crdts[j].snapshot())) != before_j:
                        changed = True
                    if normalized(self.crdt_shared(crdts[i].snapshot())) != before_i:
                        changed = True
                    for k in (i, j):
                        mismatch = self._compare(
                            crdts[k], models[k], f"flood pass {pass_no} on {ids[k]}", failure
                        )
                        if mismatch is not None:
                            return mismatch
            quiet = 0 if changed else quiet + 1
            if quiet >= 2:
                return None
        return failure(
            "reconnect",
            "flood did not reach quiescence",
            "two quiet passes",
            f"{2 * self.n + 4} passes exhausted",
        )

    def _compare(self, crdt: Any, model: Any, where: str, failure: Callable):
        actual_public = self.public(crdt)
        expected_public = self.public(model)
        if actual_public != expected_public:
            return failure("public-value", where, expected_public, actual_public)
        actual_norm = normalized(self.crdt_shared(crdt.snapshot()))
        expected_norm = normalized(self.model_shared(model))
        if actual_norm != expected_norm:
            return failure(
                "normalized-snapshot", where, expected_norm, actual_norm
            )
        return None

    def _crdt_call(self, crdt: Any, op: list) -> Any:
        return getattr(crdt, op[0])(*op[1:])

    def _model_call(self, model: Any, op: list) -> Any:
        return getattr(model, op[0])(*op[1:])

    # -- shrinking -------------------------------------------------------

    def shrink(self, spec: dict) -> Optional[dict]:
        """Greedily reduce a failing script to a smaller one failing the same way.

        A candidate is accepted only while the first divergence keeps the same
        signature (failure kind and lifecycle stage), so shrinking cannot turn
        the fault into a trivially different failure such as "replicas never
        synced because every message was deleted".
        """
        first_report = self.replay(spec)
        if first_report is None:
            return None
        signature = failure_signature(first_report)
        current = self.last_effective(spec)

        def fails(candidate: dict) -> bool:
            candidate_report = self.replay(candidate)
            return (
                candidate_report is not None
                and failure_signature(candidate_report) == signature
            )

        changed = True
        while changed:
            changed = False
            index = 0
            while index < len(current["steps"]):
                trial = {
                    **current,
                    "steps": current["steps"][:index] + current["steps"][index + 1 :],
                }
                if fails(trial):
                    current = self.last_effective(trial)
                    changed = True
                    index = 0
                    continue
                index += 1

            # Within a net step, drop individual fan-out messages / duplicates.
            for step_index, step in enumerate(current["steps"]):
                if step["k"] != "net" or len(step["msgs"]) <= 1:
                    continue
                msg_index = 0
                while msg_index < len(step["msgs"]):
                    msgs = step["msgs"][:msg_index] + step["msgs"][msg_index + 1 :]
                    trial_steps = [
                        dict(s, msgs=msgs) if s is step else s
                        for s in current["steps"]
                    ]
                    trial = {**current, "steps": trial_steps}
                    if fails(trial):
                        current = self.last_effective(trial)
                        changed = True
                        step = current["steps"][step_index]
                        msg_index = 0
                        continue
                    msg_index += 1

            # Remove all delays: everything arrives in the first possible round.
            flattened_steps = []
            for step in current["steps"]:
                if step["k"] == "net":
                    flattened_steps.append(
                        {**step, "msgs": [[s, r, 0] for s, r, _d in step["msgs"]]}
                    )
                else:
                    flattened_steps.append(step)
            trial = {**current, "steps": flattened_steps}
            if normalized_script(trial) != normalized_script(current) and fails(trial):
                current = self.last_effective(trial)
                changed = True
        return current

    def last_effective(self, spec: dict) -> dict:
        result = self.replay(spec)
        if result is not None:
            return result["effective"]
        return spec


def normalized_script(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def failure_signature(report: dict) -> tuple:
    """Coarse identity of a failure: kind plus the lifecycle stage in ``where``.

    Stages are strings like "delivery #3 (round 2, x -> y)" or
    "final convergence: replica z"; only the stage token matters, so a
    shortened trace may still point at a different delivery index.
    """
    where = report["where"]
    for cut, char in enumerate(where):
        if char.isdigit() or char in "#:":
            where = where[:cut]
            break
    return report["kind"], where.strip()


# ---------------------------------------------------------------------------
# Concrete harnesses
# ---------------------------------------------------------------------------


class GCHarness(Harness):
    name = "GCounter"
    crdt_cls = GCounter
    model_cls = GCModel

    def gen_op(self, rng: random.Random, model: GCModel) -> list:
        return ["increment", rng.choice((1, 1, 1, 2, 3, 8))]

    def public(self, obj: Any) -> int:
        return obj.value()

    def crdt_shared(self, snapshot: dict) -> object:
        return {"counts": snapshot["counts"]}

    def model_shared(self, model: GCModel) -> object:
        return {"counts": {k: model.counts[k] for k in sorted(model.counts)}}


class ORHarness(Harness):
    name = "ORSet"
    crdt_cls = ORSet
    model_cls = ORModel

    def gen_op(self, rng: random.Random, model: ORModel) -> list:
        element = rng.choice(OR_ELEMENTS)
        if rng.random() < 0.55:
            return ["add", element]
        return ["remove", element]

    def public(self, obj: Any) -> list[str]:
        return sorted(obj.elements())

    def crdt_shared(self, snapshot: dict) -> object:
        return {"adds": snapshot["adds"], "removes": snapshot["removes"]}

    def model_shared(self, model: ORModel) -> object:
        return {
            "adds": {
                element: [[origin, seq] for origin, seq in sorted(tags)]
                for element, tags in sorted(model.added.items())
            },
            "removes": [[origin, seq] for origin, seq in sorted(model.removed)],
        }


class LWWHarness(Harness):
    name = "LWWRegister"
    crdt_cls = LWWRegister
    model_cls = LWWModel

    def gen_op(self, rng: random.Random, model: LWWModel) -> list:
        return ["assign", rng.choice(LWW_VALUES)]

    def public(self, obj: Any) -> list:
        # String-encode the value so Python's True == 1 / 1 == 1.0 equivalences
        # cannot mask a real difference in the chosen JSON value.
        if not obj.has_value():
            return [False, None]
        return [True, json.dumps(obj.value(), sort_keys=True)]

    def crdt_shared(self, snapshot: dict) -> object:
        return {"clock": snapshot["clock"], "entry": snapshot["entry"]}

    def model_shared(self, model: LWWModel) -> object:
        return {
            "clock": model.clock,
            "entry": (
                None
                if not model.has
                else {
                    "timestamp": [model.winner[0], model.winner[1]],
                    "value": json.loads(json.dumps(model.stored)),
                }
            ),
        }


class RGAHarness(Harness):
    name = "RGA"
    crdt_cls = RGA
    model_cls = RGAModel

    def gen_op(self, rng: random.Random, model: RGAModel) -> list:
        length = len(model.values())
        if length == 0 or rng.random() < 0.6:
            return ["insert", rng.randint(0, length), rng.choice(RGA_TOKENS)]
        return ["delete", rng.randrange(length)]

    def public(self, obj: Any) -> list[str]:
        return obj.values()

    def crdt_shared(self, snapshot: dict) -> object:
        return {
            "counter": snapshot["counter"],
            "nodes": snapshot["nodes"],
            "tombstones": snapshot["tombstones"],
        }

    def model_shared(self, model: RGAModel) -> object:
        return {
            "counter": model.counter,
            "nodes": [
                {
                    "id": [sequence, origin],
                    "value": model.nodes[(sequence, origin)][0],
                    "prev": (
                        None
                        if model.nodes[(sequence, origin)][1] is None
                        else list(model.nodes[(sequence, origin)][1])
                    ),
                }
                for sequence, origin in sorted(model.nodes)
            ],
            "tombstones": [list(nid) for nid in sorted(model.tombstones)],
        }


HARNESS_BY_NAME = {
    "GCounter": GCHarness,
    "ORSet": ORHarness,
    "LWWRegister": LWWHarness,
    "RGA": RGAHarness,
}


def replay_script_by_name(name: str, spec: dict) -> Optional[dict]:
    """Replay a JSON script; entry point referenced by failure reports."""
    seed = spec.get("seed", 0) if isinstance(spec, dict) else 0
    harness = HARNESS_BY_NAME[name](seed)
    return harness.replay(spec)


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------


def raw_op_trace(spec: dict) -> list[str]:
    ids = spec["ids"]
    lines = []
    for step in spec["steps"]:
        if step["k"] != "op":
            continue
        args = ", ".join(repr(arg) for arg in step["op"][1:])
        lines.append(f"  burst {step['b']}: {ids[step['r']]}.{step['op'][0]}({args})")
    return lines


def render_failure(report: dict, reduced: Optional[dict]) -> str:
    spec = report["spec"]
    lines = [
        "STATE-MODEL FAILURE",
        f"crdt: {spec['crdt']}",
        f"seed: {report['seed']}  replicas: {report['n']}  scale: {report['scale']}",
        f"replica ids: {spec['ids']}",
        f"first divergence ({report['kind']}) at {report['where']}",
        f"  expected (abstract model): {report['expected']!r}",
        f"  actual   (published CRDT): {report['actual']!r}",
        "raw op trace (burst: replica.op):",
    ]
    lines.extend(raw_op_trace(spec) or ["  <no local operations>"])
    lines.append("actual delivery trace (#, round, from -> to, state after):")
    for index, round_no, sender, receiver, after in report["delivery"]:
        lines.append(f"  #{index} r{round_no} {sender} -> {receiver}  {after}")
    if not report["delivery"]:
        lines.append("  <no deliveries before divergence>")
    if reduced is not None:
        original_ops = sum(1 for s in spec["steps"] if s["k"] == "op")
        reduced_ops = sum(1 for s in reduced["steps"] if s["k"] == "op")
        original_msgs = sum(
            len(s["msgs"]) for s in spec["steps"] if s["k"] == "net"
        )
        reduced_msgs = sum(
            len(s["msgs"]) for s in reduced["steps"] if s["k"] == "net"
        )
        lines.append(
            f"shrunk: {original_ops} -> {reduced_ops} ops, "
            f"{original_msgs} -> {reduced_msgs} messages"
        )
        encoded = json.dumps(reduced)
        lines.append("BEGIN_REPLAY_SPEC")
        lines.append(encoded)
        lines.append("END_REPLAY_SPEC")
        lines.append(
            f'replay: replay_script_by_name("{spec["crdt"]}", <spec above>)'
        )
    return "\n".join(lines)


def assert_script_passes(
    harness: Harness, spec: dict, shrink_on_failure: bool = True
) -> None:
    report = harness.replay(spec)
    if report is None:
        return
    reduced = harness.shrink(spec) if shrink_on_failure else None
    raise AssertionError(render_failure(report, reduced))


# ---------------------------------------------------------------------------
# Property tests: many explicit seeds per CRDT type
# ---------------------------------------------------------------------------


class StateModelPropertyTests(unittest.TestCase):
    def _run_many(
        self,
        harness_cls: type,
        seeds: range,
        n: int = 3,
        scale: int = 20,
    ) -> None:
        for seed in seeds:
            with self.subTest(seed=seed, n=n, scale=scale):
                harness = harness_cls(seed, n=n, scale=scale)
                spec = harness.generate()
                assert_script_passes(harness, spec)

    def test_gcounter_model(self) -> None:
        self._run_many(GCHarness, range(40))

    def test_orset_model(self) -> None:
        self._run_many(ORHarness, range(40))

    def test_lww_model(self) -> None:
        self._run_many(LWWHarness, range(40))

    def test_rga_model(self) -> None:
        self._run_many(RGAHarness, range(40))

    def test_four_replicas(self) -> None:
        for harness_cls in (GCHarness, ORHarness, LWWHarness, RGAHarness):
            with self.subTest(crdt=harness_cls.name):
                self._run_many(harness_cls, range(8), n=4, scale=24)

    def test_larger_scale(self) -> None:
        for harness_cls in (GCHarness, ORHarness, LWWHarness, RGAHarness):
            with self.subTest(crdt=harness_cls.name):
                self._run_many(harness_cls, range(4), scale=48)


# ---------------------------------------------------------------------------
# Determinism: explicit seeds fix participants, ops and delivery traces
# ---------------------------------------------------------------------------


class DeterminismTests(unittest.TestCase):
    def test_same_seed_produces_same_script_and_trace(self) -> None:
        for harness_cls in (GCHarness, ORHarness, LWWHarness, RGAHarness):
            with self.subTest(crdt=harness_cls.name):
                first = harness_cls(12345)
                second = harness_cls(12345)
                spec_a = first.generate()
                spec_b = second.generate()
                self.assertEqual(
                    normalized_script(spec_a), normalized_script(spec_b)
                )
                self.assertIsNone(first.replay(spec_a))
                self.assertIsNone(second.replay(spec_b))
                self.assertEqual(first.delivery_trace, second.delivery_trace)

    def test_different_seeds_differ(self) -> None:
        scripts = {
            seed: normalized_script(RGAHarness(seed).generate())
            for seed in range(10)
        }
        self.assertEqual(len(scripts), len(set(scripts.values())))

    def test_replay_is_trace_pure(self) -> None:
        # Replaying the same fixed script twice, on fresh harness objects, must
        # yield the byte-identical delivery trace independent of object reuse.
        harness = RGAHarness(777)
        spec = harness.generate()
        traces = []
        for _ in range(3):
            runner = RGAHarness(777)
            self.assertIsNone(runner.replay(spec))
            traces.append([row[:4] for row in runner.delivery_trace])
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[1], traces[2])


# ---------------------------------------------------------------------------
# Hand-built, literal concurrency scenarios (fixed, readable regression cases)
# ---------------------------------------------------------------------------


def _restore(cls: type, snapshot: dict) -> Any:
    return cls.from_snapshot(wire(snapshot))


class HandBuiltConcurrencyTests(unittest.TestCase):
    def test_gcounter_duplicated_reordered_deliveries(self) -> None:
        a, b, c = GCounter("A"), GCounter("B"), GCounter("C")
        a.increment(3)
        a.increment(2)
        b.increment(5)
        c.increment(7)
        model = GCModel("observer")
        observer = GCounter("observer")
        path = [a.snapshot(), c.snapshot(), c.snapshot(), b.snapshot(), a.snapshot()]
        for snapshot in path:
            observer.merge(_restore(GCounter, snapshot))
            model.merge(GCModel.from_snapshot(snapshot))
        self.assertEqual(observer.value(), 17)
        self.assertEqual(observer.value(), model.value())
        self.assertEqual(
            observer.snapshot()["counts"], {"A": 5, "B": 5, "C": 7}
        )
        # A second consecutive merge of the same state changes nothing.
        before = observer.snapshot()
        observer.merge(_restore(GCounter, c.snapshot()))
        observer.merge(_restore(GCounter, c.snapshot()))
        self.assertEqual(observer.snapshot(), before)

    def test_orset_remove_only_hits_observed_adds(self) -> None:
        a, b, c = ORSet("A"), ORSet("B"), ORSet("C")
        a.add("x")
        a_state = a.snapshot()
        b.merge(_restore(ORSet, a_state))
        c.merge(_restore(ORSet, a_state))
        common = a_state

        # B has observed A's add and removes it; C concurrently re-adds "x"
        # with a fresh tag B's remove cannot know about.
        self.assertTrue(b.remove("x"))
        c.add("x")
        b.merge(_restore(ORSet, c.snapshot()))
        c.merge(_restore(ORSet, b.snapshot()))
        a.merge(_restore(ORSet, b.snapshot()))
        a.merge(_restore(ORSet, c.snapshot()))

        model = ORModel("model")
        for snapshot in [common, b.snapshot(), c.snapshot()]:
            model.merge(ORModel.from_snapshot(snapshot))
        for replica in (a, b, c):
            self.assertEqual(replica.elements(), model.elements())
            self.assertIn("x", replica.elements())

        # A stale snapshot of B taken *before* its remove cannot resurrect
        # anything, and delivering it repeatedly is idempotent.
        before = normalized(a.snapshot())
        for _ in range(3):
            a.merge(_restore(ORSet, common))
        self.assertEqual(normalized(a.snapshot()), before)
        self.assertIn("x", a.elements())

    def test_orset_observed_remove_is_stable_under_late_delivery(self) -> None:
        a, b, c = ORSet("A"), ORSet("B"), ORSet("C")
        a.add("x")
        b.merge(_restore(ORSet, a.snapshot()))
        c.merge(_restore(ORSet, a.snapshot()))
        pre_remove = b.snapshot()

        b.remove("x")
        c.add("y")  # concurrent activity on another element
        for left, right in [(b, c), (c, b), (a, b), (a, c)]:
            left.merge(_restore(ORSet, right.snapshot()))
            right.merge(_restore(ORSet, left.snapshot()))
        for replica in (a, b, c):
            self.assertEqual(replica.elements(), {"y"})
        # B's old pre-remove state arriving late never brings "x" back.
        a.merge(_restore(ORSet, pre_remove))
        self.assertEqual(a.elements(), {"y"})

    def test_lww_timestamp_and_tie_rule(self) -> None:
        a, b, c = LWWRegister("A"), LWWRegister("B"), LWWRegister("C")
        a.assign("a")
        b.assign("b")
        c.assign("c")  # all at count 1; tie breaks on replica id: C wins
        for target, source in [(a, b), (a, c), (b, a), (b, c), (c, a), (c, b)]:
            target.merge(_restore(LWWRegister, source.snapshot()))
        for replica in (a, b, c):
            self.assertEqual(replica.value(), "c")
        model = LWWModel("m")
        for snapshot in (a.snapshot(), b.snapshot(), c.snapshot()):
            model.merge(LWWModel.from_snapshot(snapshot))
        self.assertEqual(model.value(), "c")

        # After observing the winner, A's next write is at count 2 and wins.
        a.assign("a2")
        late = b.snapshot()  # stale "c" at (1, C), captured before a2 propagates
        for replica in (b, c):
            replica.merge(_restore(LWWRegister, a.snapshot()))
        b.merge(_restore(LWWRegister, late))  # stale arrival must not roll back
        c.merge(_restore(LWWRegister, c.snapshot()))  # self-merge no-op
        for replica in (a, b, c):
            self.assertEqual(replica.value(), "a2")
        twice = c.snapshot()
        a.merge(_restore(LWWRegister, twice))
        a.merge(_restore(LWWRegister, twice))
        self.assertEqual(a.value(), "a2")

    def test_rga_concurrent_root_inserts_order_deterministically(self) -> None:
        a, b, c = RGA("A"), RGA("B"), RGA("C")
        a.insert(0, "a")
        b.insert(0, "b")
        c.insert(0, "c")  # three concurrent siblings off the root, seq all 1
        for left, right in [(a, b), (b, c), (c, a), (a, c), (b, a)]:
            left.merge(_restore(RGA, right.snapshot()))
            right.merge(_restore(RGA, left.snapshot()))
        expected = ["c", "b", "a"]  # descending (seq, replica-id) order
        for replica in (a, b, c):
            self.assertEqual(replica.values(), expected)
        model = RGAModel("m")
        for snapshot in (a.snapshot(), b.snapshot(), c.snapshot()):
            model.merge(RGAModel.from_snapshot(snapshot))
        self.assertEqual(model.values(), expected)

    def test_rga_sibling_branch_is_contiguous(self) -> None:
        a, b = RGA("A"), RGA("B")
        a.insert(0, "a")
        a.insert(1, "a2")  # hangs off "a"
        b.insert(0, "b")  # concurrent sibling off the root
        a.merge(_restore(RGA, b.snapshot()))
        b.merge(_restore(RGA, a.snapshot()))
        # Root siblings in descending id order: (1,B) then (1,A); A's own
        # branch node (2,A) follows A immediately.
        self.assertEqual(a.values(), ["b", "a", "a2"])
        self.assertEqual(b.values(), ["b", "a", "a2"])

    def test_rga_delete_vs_concurrent_unobserved_insert(self) -> None:
        a, b, c = RGA("A"), RGA("B"), RGA("C")
        a.insert(0, "a")
        common = a.snapshot()
        for replica in (b, c):
            replica.merge(_restore(RGA, common))
        b.delete(0)  # observed remove of "a"
        c.insert(1, "c")  # concurrently append after the node B is deleting
        b_pre_delete = common  # B's state while "a" was still live
        b.merge(_restore(RGA, c.snapshot()))
        c.merge(_restore(RGA, b.snapshot()))
        a.merge(_restore(RGA, b.snapshot()))
        a.merge(_restore(RGA, c.snapshot()))
        for replica in (a, b, c):
            self.assertEqual(replica.values(), ["c"])
        # B's old live-"a" snapshot, delivered after convergence, never
        # resurrects the tombstoned node, and duplicate delivery is stable.
        before = normalized(a.snapshot())
        a.merge(_restore(RGA, b_pre_delete))
        a.merge(_restore(RGA, b_pre_delete))
        self.assertEqual(a.values(), ["c"])
        self.assertEqual(normalized(a.snapshot()), before)

    def test_rga_double_merge_is_idempotent(self) -> None:
        a, b = RGA("A"), RGA("B")
        a.insert(0, "x")
        b.merge(_restore(RGA, a.snapshot()))
        b.insert(1, "y")
        b.delete(0)
        snap = b.snapshot()
        before = a.snapshot()
        a.merge(_restore(RGA, snap))
        a.merge(_restore(RGA, snap))
        self.assertEqual(a.values(), ["y"])
        # Merging an equivalent copy twice more changes nothing.
        a.merge(_restore(RGA, a.snapshot()))
        self.assertEqual(a.values(), ["y"])
        self.assertNotEqual(a.snapshot(), before)


# ---------------------------------------------------------------------------
# Published scope: tombstone compaction exists only on ORSet
# ---------------------------------------------------------------------------


class CompactionScopeTests(unittest.TestCase):
    BANNED = ("compact", "compaction", "compress", "prune", "purge", "gc")

    def test_compaction_is_published_only_on_orset(self) -> None:
        for cls in (GCounter, LWWRegister, RGA):
            with self.subTest(crdt=cls.__name__):
                public_names = [
                    name for name in dir(cls) if not name.startswith("_")
                ]
                offenders = [
                    name
                    for name in public_names
                    if any(word in name.lower() for word in self.BANNED)
                ]
                self.assertEqual(offenders, [])
        # ORSet is the single CRDT that publishes a compaction entry.
        self.assertTrue(hasattr(ORSet, "compact"))
        self.assertTrue(callable(getattr(ORSet, "compact")))

    def test_package_exports_only_the_crdts_clock_and_version(self) -> None:
        import crdt_sync

        self.assertEqual(
            set(crdt_sync.__all__),
            {
                "GCounter",
                "ORSet",
                "LWWRegister",
                "RGA",
                "RGASession",
                "VectorClock",
                "__version__",
            },
        )

    def test_abstract_models_exercise_merge_independently(self) -> None:
        # Sanity pin: the RGA model used as oracle really is a separate
        # implementation (tree weave), not production RGA in disguise.
        self.assertIsNot(RGAModel, RGA)
        self.assertIsNot(ORModel, ORSet)
        model = RGAModel("A")
        model.insert(0, "a")
        self.assertEqual(model.values(), ["a"])


# ---------------------------------------------------------------------------
# Diagnostic machinery: a deliberately faulty CRDT subclass must be detected,
# reported with every required field, and shrunk to a smaller failing script.
# ---------------------------------------------------------------------------


class NarrowMergeGCounter(GCounter):
    """Test double: accepts only components from replica ids <= its own.

    State-based join is therefore partial: a higher-id peer's increments are
    ignored forever, so convergence cannot be reached. The faulty behavior is
    injected purely through published surface (snapshot / from_snapshot /
    merge); no production internals are read.
    """

    def merge(self, other: "GCounter") -> "GCounter":  # type: ignore[override]
        snap = other.snapshot()
        filtered = {
            "replica_id": snap["replica_id"],
            "counts": {
                rid: amount
                for rid, amount in snap["counts"].items()
                if rid <= self.replica_id
            },
        }
        return super().merge(GCounter.from_snapshot(wire(filtered)))


class TombstoneDeafORSet(ORSet):
    """Test double: unions adds on merge but never adopts tombstones."""

    def merge(self, other: "ORSet") -> "ORSet":  # type: ignore[override]
        snap = other.snapshot()
        blind = {**snap, "removes": []}
        return super().merge(ORSet.from_snapshot(wire(blind)))


def _find_failing(harness_cls: type, crdt_cls: type, seeds: range, scale: int):
    for seed in seeds:
        generator = harness_cls(seed, scale=scale)
        spec = generator.generate()
        runner = harness_cls(seed, scale=scale, crdt_cls=crdt_cls)
        report = runner.replay(spec)
        if report is not None:
            return seed, spec, runner, report
    raise AssertionError("no seed exposed the injected fault")


class FailureReportingTests(unittest.TestCase):
    def _check_report(self, harness: Harness, spec: dict, report: dict,
                      faulty_cls: type) -> dict:
        reduced = harness.shrink(spec)
        self.assertIsNotNone(reduced)
        text = render_failure(report, reduced)
        for token in (
            f"crdt: {spec['crdt']}",
            f"seed: {harness.seed}",
            "raw op trace",
            "actual delivery trace",
            "first divergence",
            "BEGIN_REPLAY_SPEC",
            "END_REPLAY_SPEC",
        ):
            self.assertIn(token, text)

        original_steps = len(spec["steps"])
        self.assertLessEqual(len(reduced["steps"]), original_steps)
        self.assertLess(
            len(normalized_script(reduced)), len(normalized_script(spec))
        )

        begin = text.index("BEGIN_REPLAY_SPEC") + len("BEGIN_REPLAY_SPEC")
        end = text.index("END_REPLAY_SPEC")
        embedded = json.loads(text[begin:end].strip())
        # The embedded script is plain JSON; replaying it through the same
        # faulty double must still fail, and through the correct CRDT pass.
        faulty_runner = harness.__class__(
            harness.seed, n=harness.n, scale=harness.scale, crdt_cls=faulty_cls
        )
        self.assertIsNotNone(faulty_runner.replay(embedded))
        healthy = harness.__class__(harness.seed, n=harness.n, scale=harness.scale)
        self.assertIsNone(healthy.replay(embedded))
        return reduced

    def test_faulty_gcounter_is_detected_reported_and_shrunk(self) -> None:
        seed, spec, runner, report = _find_failing(
            GCHarness, NarrowMergeGCounter, range(40), 18
        )
        reduced = self._check_report(runner, spec, report, NarrowMergeGCounter)
        # The shrunk script still fails on the faulty class ...
        rerun = GCHarness(seed, scale=18, crdt_cls=NarrowMergeGCounter)
        self.assertIsNotNone(rerun.replay(reduced))
        # ... and passes against the correct implementation.
        healthy = GCHarness(seed, scale=18)
        self.assertIsNone(healthy.replay(reduced))

    def test_faulty_orset_is_detected_reported_and_shrunk(self) -> None:
        seed, spec, runner, report = _find_failing(
            ORHarness, TombstoneDeafORSet, range(60), 24
        )
        self._check_report(runner, spec, report, TombstoneDeafORSet)

    def test_shrunk_trace_is_deterministic_and_directly_replayable(self) -> None:
        seed, spec, runner, report = _find_failing(
            GCHarness, NarrowMergeGCounter, range(40), 18
        )
        reduced_a = runner.shrink(spec)
        reduced_b = runner.shrink(spec)
        self.assertEqual(
            normalized_script(reduced_a), normalized_script(reduced_b)
        )


if __name__ == "__main__":
    unittest.main()
