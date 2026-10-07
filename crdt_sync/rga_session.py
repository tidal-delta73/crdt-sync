"""Session bundling an RGA sequence replica with its causal context.

An :class:`RGASession` couples one :class:`~crdt_sync.rga.RGA` replica and
one :class:`~crdt_sync.vector_clock.VectorClock` under a single
``replica_id`` so the pair travels as one offline-editing and reconnect
unit. Local ``insert``/``delete`` edit the sequence exactly as on a bare
RGA and additionally tick the session's clock once per successful edit, so
the clock always counts the edits this replica has observed. A delete is
also recorded under the clock dot it just minted, so a session snapshot
carries mergeable deletion *causality* — which delete events tombstoned a
node — rather than a bare tombstone set. ``snapshot`` packages the
sequence state and causal context into one JSON-serializable exchange
unit, and ``merge`` absorbs a peer session's sequence and clock atomically
after reporting how the peer's causal context relates to the receiver's.

``compact(stable_clock)`` performs causal, safe sequence compaction
against a stability frontier the caller derives from the acknowledgements
of every participant currently syncing with the session. It reclaims only
tombstone nodes with a deletion event covered by that frontier which no
longer predecessor any node surviving as a full record, peeling whole
stable deleted branches inward from the leaf end; it never crosses a live
node, an unstabilized delete or an untagged legacy tombstone. A reclaimed
node's value record disappears (that is the reclaimed count), but a
permanent *retired summary* remains: the node id with its predecessor edge
(a tombstoned "skeleton" that keeps a late-arriving branch weaving in the
right place), its deletion dots and a per-origin retired clock prefix.
Consequently a late, duplicated or reordered pre-compaction snapshot can
never resurrect a retired node, and compacted and uncompacted replicas
converge in both directions. The bare :class:`~crdt_sync.rga.RGA` gains no
compaction entry point and keeps its original four-field snapshots and
behavior; all compaction state lives in the session layer.
"""

from __future__ import annotations

from .rga import RGA, _parse_node_id, _validate_index
from .vector_clock import VectorClock

NodeId = tuple[int, str]
Dot = tuple[str, int]

# Value stored for a node demoted to a retired predecessor skeleton. The
# skeleton always remains tombstoned, so RGA weave code never renders this
# placeholder; it only keeps predecessor links reachable for branches that
# still hang off a reclaimed node.
_RETIRED = object()


def _dominates(clock: dict[str, int], other: dict[str, int]) -> bool:
    """Whether every component of ``other`` is met by ``clock`` (>=)."""
    return all(
        clock.get(replica_id, 0) >= value for replica_id, value in other.items()
    )


class RGASession:
    """An RGA sequence replica plus its vector clock, under one identity."""

    def __init__(self, replica_id: str) -> None:
        self._rga = RGA(replica_id)
        self._clock = VectorClock(replica_id)
        # Tombstoned, still-stored node id -> all known deletion event dots.
        # Several replicas may each delete an already-dead node they have
        # observed, so a node can carry several dots. Tombstones lacking any
        # dot (restored from a legacy snapshot) are absent and never become
        # compaction-eligible.
        self._deleted_at: dict[NodeId, set[Dot]] = {}
        # Ids of nodes whose value records were reclaimed by compaction.
        self._retired_ids: set[NodeId] = set()
        # Permanent predecessor edges of retired nodes; a retired node stays
        # a tombstoned skeleton so descendants still hang off it correctly.
        self._retired_prev: dict[NodeId, NodeId | None] = {}
        # Shadow values of retired nodes, kept solely so a merge with a peer
        # that still holds the full (pre-compaction) record can still detect
        # a same-id/different-content collision as ValueError.
        self._retired_value: dict[NodeId, str] = {}
        # All deletion dots of reclaimed nodes, union-merged.
        self._retired_dots: set[Dot] = set()
        # Per-origin prefix covered by the retired dots (their
        # component-wise maximum); a deterministic summary of retirement.
        self._retired_clock: dict[str, int] = {}

    @property
    def replica_id(self) -> str:
        return self._rga.replica_id

    def values(self) -> list[str]:
        """Return a fresh list of the visible strings in sequence order."""
        return self._rga.values()

    def insert(self, index: int, value: str) -> None:
        """Insert ``value`` at the visible zero-based ``index``.

        Same contract as :meth:`RGA.insert`; a successful insert also
        advances the session clock's local component once. A failed call
        (``TypeError``/``IndexError``) changes neither the sequence nor the
        clock.
        """
        self._rga.insert(index, value)
        self._clock.tick()

    def delete(self, index: int) -> str:
        """Tombstone the visible element at ``index`` and return its string.

        Same contract as :meth:`RGA.delete`; a successful delete also
        advances the session clock's local component once and records that
        clock dot as the delete's causal tag. A failed call
        (``TypeError``/``IndexError``) changes neither the sequence nor the
        clock.
        """
        # Validate exactly like the RGA before touching anything, so a
        # boolean or otherwise invalid index raises TypeError rather than
        # silently behaving as an integer subscript.
        index = _validate_index(index)
        # Resolve the id before the RGA tombstones it: the visible weave is
        # identical up to that call, so visible[index] is exactly the node
        # delete() removes. Mirror the RGA's own range check first so a
        # negative or out-of-range index raises IndexError without touching
        # the sequence or clock.
        visible = self._rga._visible_nodes()
        if index < 0 or index >= len(visible):
            raise IndexError("delete index out of range")
        node_id = visible[index]
        removed = self._rga.delete(index)
        self._clock.tick()
        dot = (self.replica_id, self._clock.components()[self.replica_id])
        self._deleted_at.setdefault(node_id, set()).add(dot)
        return removed

    def compact(self, stable_clock: VectorClock) -> int:
        """Reclaim stable tombstone branches below a confirmed frontier.

        ``stable_clock`` is the vector-clock frontier the caller derives
        from the acknowledgements of every participant currently syncing
        with this session: a delete is stable once one of its deletion
        events is covered by the frontier, certifying every participant has
        observed that tombstone. Only tombstone nodes which carry such a
        covered deletion dot, are not restored from a legacy snapshot
        (untagged tombstones are never eligible) and no longer predecessor
        any node surviving as a full record can be reclaimed. Eligible
        leaves are peeled repeatedly, so a whole contiguous branch of
        stable deleted nodes comes away in one call — a peeled node becomes
        a retired skeleton whose own edge never blocks its parent — while
        compaction stops at a live node, an unstable delete, or an
        untagged node still referenced as a predecessor.

        The call neither advances the session clock nor changes
        :meth:`values`. A reclaimed node leaves a retired skeleton (id and
        predecessor edge) plus its deletion dots and a per-origin retired
        clock prefix, so old, duplicated or reordered snapshots cannot
        resurrect it. Returns the number of RGA node value records removed
        by this call; calling again with the same frontier returns ``0``.

        Raises ``TypeError`` when ``stable_clock`` is not a
        :class:`VectorClock`. Raises ``ValueError`` — leaving the session
        completely unchanged, clock included — when ``stable_clock`` is
        concurrent with or strictly ahead of this session's current clock,
        i.e. it claims stability for edits this replica has not observed.
        """
        if not isinstance(stable_clock, VectorClock):
            raise TypeError("stable_clock must be a VectorClock")
        frontier = stable_clock.components()
        current = self._clock.components()
        # The frontier may be equal to or behind the session clock, never
        # concurrent with it or ahead: every component it names must be
        # observed here.
        if not _dominates(current, frontier):
            raise ValueError(
                "stable_clock must not be concurrent with or ahead of the "
                "session clock"
            )

        nodes = self._rga._nodes

        def stable_deleted(node_id: NodeId) -> bool:
            dots = self._deleted_at.get(node_id)
            if not dots:
                # Untagged (legacy) tombstone or a live node: not eligible.
                return False
            return any(
                frontier.get(origin, 0) >= counter for origin, counter in dots
            )

        # Peel eligible leaves. Only a node that survives as a FULL record
        # keeps its predecessor alive: a node peeled this round contributes
        # no reference, and an already-retired skeleton's edge extends into
        # a skeleton chain rather than blocking. Recomputing after each peel
        # lets a stable-deleted parent become a leaf once its child is gone.
        removed: set[NodeId] = set()
        while True:
            referenced: set[NodeId] = set()
            for node_id, (_value, predecessor) in nodes.items():
                if node_id in removed or node_id in self._retired_ids:
                    continue
                if predecessor is not None:
                    referenced.add(predecessor)
            leaves = [
                node_id
                for node_id in nodes
                if node_id not in removed
                and node_id not in self._retired_ids
                and stable_deleted(node_id)
                and node_id not in referenced
            ]
            if not leaves:
                break
            removed.update(leaves)

        if not removed:
            return 0

        # Commit. The value record leaves the RGA weave but its content
        # shadow joins the retired summary: it never becomes visible or
        # editable again, yet a later merge with a peer still holding the
        # full pre-compaction record can detect a same-id content conflict.
        # The id/predecessor skeleton stays, so an unseen branch hanging off
        # the node still weaves after a late merge. The node remains a
        # tombstone while its live deletion tags join the retired summary.
        for node_id in removed:
            value, predecessor = nodes[node_id]
            nodes[node_id] = (_RETIRED, predecessor)
            self._retired_prev[node_id] = predecessor
            self._retired_value[node_id] = value
            self._retired_dots.update(self._deleted_at.pop(node_id))
        self._retired_ids.update(removed)
        for origin, counter in self._retired_dots:
            if counter > self._retired_clock.get(origin, 0):
                self._retired_clock[origin] = counter
        return len(removed)

    def merge(self, other: "RGASession") -> str:
        """Absorb ``other``'s sequence and clock; return the causal relation.

        The relation — ``before``/``after``/``equal``/``concurrent`` — is
        decided from both clocks *before* any state moves and describes
        ``other`` relative to this receiver. The sequence and the causal
        context then merge together; the receiver keeps its own
        ``replica_id`` and ``other`` is never modified. Repeated, reordered
        or stale merges converge to the same sequence and clock.

        Deletion causality and the retired summary merge with the sequence:
        a node retired on either side is consumed history whose full record
        can never be resurrected, while its predecessor skeleton is adopted
        so branches still hanging off it keep their positions. Raises
        ``TypeError`` when ``other`` is not an ``RGASession``. Raises
        ``ValueError`` — leaving this session's sequence, clock and
        compaction metadata exactly as they were — when the same RGA node id
        carries a different value or predecessor on the two sides.
        """
        if not isinstance(other, RGASession):
            raise TypeError("can only merge with another RGASession")
        relation = other._clock.compare(self._clock)

        joined_retired = self._retired_ids | other._retired_ids

        # Planned post-merge node map, built in scratch so every conflict is
        # seen before any receiver state moves.
        planned: dict[NodeId, tuple[object, NodeId | None]] = {}
        for node_id, record in self._rga._nodes.items():
            planned[node_id] = record
        planned_retired_value = dict(self._retired_value)
        for node_id in other._retired_ids:
            # The peer retired a record we still hold in full: its content
            # and predecessor edge are immutable per id and must agree. A
            # mismatch — two replicas sharing an id minted different
            # histories — raises before anything moves, exactly as a live
            # node conflict does.
            their_prev = other._retired_prev[node_id]
            their_value = other._retired_value[node_id]
            local = planned.get(node_id)
            if local is not None and local[0] is not _RETIRED:
                if local[0] != their_value or local[1] != their_prev:
                    raise ValueError(
                        "the same node id is bound to a different value or "
                        "predecessor"
                    )
            else:
                shadow = planned_retired_value.get(node_id)
                if shadow is not None and (
                    shadow != their_value
                    or self._retired_prev.get(node_id) != their_prev
                ):
                    raise ValueError(
                        "the same node id is bound to a different value or "
                        "predecessor"
                    )
            planned[node_id] = (_RETIRED, their_prev)
            planned_retired_value[node_id] = their_value
        for node_id, record in other._rga._nodes.items():
            if node_id in joined_retired:
                # A full record on the wire for an id we retired means the
                # peer never compacted; compare its immutable content
                # against our shadow so a compacted replica still detects a
                # same-id collision. The peer's own retired skeletons (value
                # already _RETIRED) carry no content to compare here — their
                # edges were checked above.
                if record[0] is not _RETIRED and node_id in self._retired_ids:
                    shadow_value = planned_retired_value.get(node_id)
                    shadow_prev = self._retired_prev.get(node_id)
                    if (
                        shadow_value is not None
                        and (record[0] != shadow_value or record[1] != shadow_prev)
                    ):
                        raise ValueError(
                            "the same node id is bound to a different value "
                            "or predecessor"
                        )
                continue
            local = planned.get(node_id)
            if local is not None and local[0] is not _RETIRED:
                if local != record:
                    raise ValueError(
                        "the same node id is bound to a different value or "
                        "predecessor"
                    )
                continue
            planned[node_id] = record

        # Every predecessor edge must resolve to a planned node (full record
        # or skeleton), and the functional predecessor graph must be acyclic.
        for node_id, (_value, predecessor) in planned.items():
            if predecessor is not None and predecessor not in planned:
                raise ValueError("node predecessor is missing")
        state: dict[NodeId, int] = {}  # 1 = on path, 2 = settled
        for start in planned:
            if state.get(start) == 2:
                continue
            path: list[NodeId] = []
            cursor: NodeId | None = start
            while cursor is not None and state.get(cursor) != 2:
                if state.get(cursor) == 1:
                    raise ValueError("node predecessor cycle")
                state[cursor] = 1
                path.append(cursor)
                cursor = planned[cursor][1]
            for visited in path:
                state[visited] = 2

        planned_tombstones = set(self._rga._tombstones)
        for node_id in other._rga._tombstones:
            if node_id not in joined_retired and node_id not in self._retired_ids:
                planned_tombstones.add(node_id)
        planned_tombstones |= joined_retired

        # Merge deletion causality: union dots for retained tombstones; dots
        # of retired nodes join the retired summary. An untagged (legacy)
        # tombstone becomes tagged once a tagged copy of it arrives.
        planned_deleted: dict[NodeId, set[Dot]] = {
            node_id: set(dots) for node_id, dots in self._deleted_at.items()
        }
        planned_retired_dots = set(self._retired_dots)
        for node_id, dots in other._deleted_at.items():
            if node_id in joined_retired:
                planned_retired_dots.update(dots)
            else:
                planned_deleted.setdefault(node_id, set()).update(dots)
        planned_retired_dots.update(other._retired_dots)
        # Peers retiring a node we still tag move our tags with it.
        for node_id in other._retired_ids:
            tags = planned_deleted.pop(node_id, None)
            if tags:
                planned_retired_dots.update(tags)

        planned_counter = max(self._rga._counter, other._rga._counter)
        planned_clock = VectorClock(self.replica_id)
        planned_clock.merge(self._clock)
        planned_clock.merge(other._clock)

        planned_retired_prev = {
            node_id: planned[node_id][1] for node_id in joined_retired
        }
        planned_retired_clock: dict[str, int] = {}
        for origin, counter in planned_retired_dots:
            if counter > planned_retired_clock.get(origin, 0):
                planned_retired_clock[origin] = counter

        # All checks passed; commit.
        self._rga._nodes = planned
        self._rga._tombstones = planned_tombstones
        self._rga._counter = planned_counter
        self._deleted_at = planned_deleted
        self._retired_ids = joined_retired
        self._retired_prev = planned_retired_prev
        self._retired_value = planned_retired_value
        self._retired_dots = planned_retired_dots
        self._retired_clock = planned_retired_clock
        self._clock = planned_clock
        return relation

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of the whole session.

        Until any node is retired the package has exactly ``replica_id``,
        ``rga`` and ``clock`` (the legacy three-field session shape; the
        nested RGA snapshot keeps its original four fields). Once
        compaction has retired nodes it additionally carries a
        deterministic ``compaction`` summary with the retired ids and their
        predecessor edges, the retired deletion dots and the per-origin
        retired clock prefix. Deletion causality rides on the tombstones: a
        tagged tombstone serializes as ``[id, [[origin, counter], ...]]``
        while an untagged legacy tombstone stays the plain
        ``[sequence, origin]`` id. Every level is rebuilt on each call, so
        the result is deeply independent of the session, of other
        snapshots and safe to round-trip through JSON.
        """
        raw_rga = self._rga.snapshot()
        retired = self._retired_ids

        nodes = [
            node for node in raw_rga["nodes"] if tuple(node["id"]) not in retired
        ]

        tombstones: list[object] = []
        for raw_id in raw_rga["tombstones"]:
            node_id = (raw_id[0], raw_id[1])
            if node_id in retired:
                continue
            dots = self._deleted_at.get(node_id)
            if dots:
                tombstones.append([raw_id, [[o, c] for o, c in sorted(dots)]])
            else:
                tombstones.append(raw_id)

        rga_snapshot = dict(raw_rga)
        rga_snapshot["nodes"] = nodes
        rga_snapshot["tombstones"] = tombstones

        snapshot: dict[str, object] = {
            "replica_id": self.replica_id,
            "rga": rga_snapshot,
            "clock": self._clock.snapshot(),
        }
        if retired:
            snapshot["compaction"] = {
                "retired": [
                    {
                        "id": [sequence, origin],
                        "value": self._retired_value[(sequence, origin)],
                        "prev": (
                            None
                            if predecessor is None
                            else [predecessor[0], predecessor[1]]
                        ),
                    }
                    for sequence, origin in sorted(retired)
                    for predecessor in [self._retired_prev[(sequence, origin)]]
                ],
                "retired_dots": [
                    [origin, counter]
                    for origin, counter in sorted(self._retired_dots)
                ],
                "retired_clock": {
                    origin: self._retired_clock[origin]
                    for origin in sorted(self._retired_clock)
                },
            }
        return snapshot

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "RGASession":
        """Restore a session from a :meth:`snapshot`-compatible dict.

        Accepts the original three-field shape (including a legacy
        four-field RGA snapshot with plain tombstone ids) and the shape that
        additionally carries a ``compaction`` summary. Tombstones restored
        in the legacy shape carry no deletion causality and are treated as
        permanently ineligible for compaction; their deletion time is never
        inferred.

        Raises ``TypeError`` when ``snapshot`` is not a dict. Raises
        ``ValueError`` — without producing a partially restored session —
        when fields are missing or unexpected, a nested snapshot is
        invalid, the three ``replica_id`` values disagree, a deletion dot
        is not covered by the clock, or the compaction summary is
        internally inconsistent. A restored session keeps its Lamport
        counter and clock components, so continued editing mints fresh node
        ids and never reuses or rolls back.
        """
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        keys = set(snapshot.keys())
        base_keys = {"replica_id", "rga", "clock"}
        if keys == base_keys:
            has_compaction = False
        elif keys == base_keys | {"compaction"}:
            has_compaction = True
        else:
            raise ValueError(
                "snapshot must contain exactly 'replica_id', 'rga' and "
                "'clock', and optionally 'compaction'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        rga_snapshot = snapshot["rga"]
        clock_snapshot = snapshot["clock"]
        if not isinstance(rga_snapshot, dict):
            raise ValueError("rga snapshot must be a dict")
        if not isinstance(clock_snapshot, dict):
            raise ValueError("clock snapshot must be a dict")

        clock = VectorClock.from_snapshot(clock_snapshot)
        components = clock.components()

        # Peel deletion dots off enriched tombstone entries, reducing them
        # to plain legacy-shaped ids for the strict RGA validator.
        rga_snapshot = dict(rga_snapshot)
        raw_tombstones = rga_snapshot.get("tombstones")
        if not isinstance(raw_tombstones, list):
            raise ValueError("tombstones must be a list")
        deleted_at: dict[NodeId, set[Dot]] = {}
        plain_tombstones: list[object] = []
        for raw_entry in raw_tombstones:
            node_id, dots = cls._parse_tombstone_entry(raw_entry)
            if node_id in deleted_at:
                raise ValueError("duplicate tombstone entry")
            plain_tombstones.append([node_id[0], node_id[1]])
            if dots is not None:
                deleted_at[node_id] = set(dots)
        rga_snapshot["tombstones"] = plain_tombstones

        retired_ids: set[NodeId] = set()
        retired_prev: dict[NodeId, NodeId | None] = {}
        retired_value: dict[NodeId, str] = {}
        retired_dots: set[Dot] = set()
        retired_clock: dict[str, int] = {}
        placeholder_nodes: list[dict[str, object]] = []
        if has_compaction:
            raw_nodes = rga_snapshot.get("nodes")
            if not isinstance(raw_nodes, list):
                raise ValueError("nodes must be a list")
            # Only structurally plausible ids are collected; malformed node
            # dicts are left for RGA.from_snapshot to reject below.
            stored_ids: set[NodeId] = set()
            for raw_node in raw_nodes:
                if isinstance(raw_node, dict) and isinstance(
                    raw_node.get("id"), list
                ):
                    stored_ids.add(_parse_node_id(raw_node["id"]))
            (
                retired_ids,
                retired_prev,
                retired_value,
                retired_dots,
                retired_clock,
            ) = cls._parse_compaction(
                snapshot["compaction"], stored_ids, components
            )
            # Install skeletons as tombstoned placeholder records before the
            # RGA validator runs, so a retained node hanging off a retired
            # predecessor sees the edge resolve. The placeholders are
            # demoted to real skeletons immediately after construction.
            for node_id in sorted(retired_ids):
                predecessor = retired_prev[node_id]
                placeholder_nodes.append(
                    {
                        "id": [node_id[0], node_id[1]],
                        "value": "",
                        "prev": (
                            None
                            if predecessor is None
                            else [predecessor[0], predecessor[1]]
                        ),
                    }
                )
                plain_tombstones.append([node_id[0], node_id[1]])
            rga_snapshot["nodes"] = list(raw_nodes) + placeholder_nodes
            rga_snapshot["tombstones"] = plain_tombstones

        rga = RGA.from_snapshot(rga_snapshot)
        if rga.replica_id != replica_id or clock.replica_id != replica_id:
            raise ValueError(
                "replica_id must match across 'replica_id', 'rga' and "
                "'clock'"
            )

        for node_id, dots in deleted_at.items():
            if node_id not in rga._tombstones:
                raise ValueError("deletion dot tags a node that is not deleted")
            for origin, counter in dots:
                if counter > components.get(origin, 0):
                    raise ValueError("deletion dot is not covered by the clock")

        # Demote the placeholders to genuine skeletons (their empty-string
        # value must never be observable or serializable as a real record).
        for node_id in retired_ids:
            _value, predecessor = rga._nodes[node_id]
            rga._nodes[node_id] = (_RETIRED, predecessor)

        session = cls(replica_id)
        session._rga = rga
        session._clock = clock
        session._deleted_at = deleted_at
        session._retired_ids = retired_ids
        session._retired_prev = retired_prev
        session._retired_value = retired_value
        session._retired_dots = retired_dots
        session._retired_clock = retired_clock
        return session

    @staticmethod
    def _parse_dot(raw_dot: object) -> Dot:
        if not isinstance(raw_dot, list) or len(raw_dot) != 2:
            raise ValueError("deletion dot must be [origin, counter]")
        origin, counter = raw_dot
        if not isinstance(origin, str) or origin == "":
            raise ValueError("deletion dot origin must be a non-empty string")
        if (
            not isinstance(counter, int)
            or isinstance(counter, bool)
            or counter < 1
        ):
            raise ValueError("deletion dot counter must be a positive integer")
        return origin, counter

    @classmethod
    def _parse_tombstone_entry(
        cls, raw_entry: object
    ) -> tuple[NodeId, list[Dot] | None]:
        """Parse plain ``[seq, origin]`` or tagged ``[id, [dot, ...]]``."""
        if (
            isinstance(raw_entry, list)
            and len(raw_entry) == 2
            and isinstance(raw_entry[0], list)
            and isinstance(raw_entry[1], list)
        ):
            node_id = _parse_node_id(raw_entry[0])
            raw_dots = raw_entry[1]
            if not raw_dots:
                raise ValueError("tagged tombstone must carry at least one dot")
            dots: list[Dot] = []
            for raw_dot in raw_dots:
                dot = cls._parse_dot(raw_dot)
                if dot in dots:
                    raise ValueError("duplicate deletion dot for one node")
                dots.append(dot)
            return node_id, dots
        return _parse_node_id(raw_entry), None

    @classmethod
    def _parse_compaction(
        cls,
        raw: object,
        stored_ids: set[NodeId],
        components: dict[str, int],
    ) -> tuple[
        set[NodeId],
        dict[NodeId, NodeId | None],
        dict[NodeId, str],
        set[Dot],
        dict[str, int],
    ]:
        if not isinstance(raw, dict):
            raise ValueError("compaction must be a dict")
        if set(raw.keys()) != {"retired", "retired_dots", "retired_clock"}:
            raise ValueError(
                "compaction must contain exactly 'retired', 'retired_dots' "
                "and 'retired_clock'"
            )

        raw_retired = raw["retired"]
        if not isinstance(raw_retired, list) or not raw_retired:
            raise ValueError("compaction.retired must be a non-empty list")
        retired_ids: set[NodeId] = set()
        retired_prev: dict[NodeId, NodeId | None] = {}
        retired_value: dict[NodeId, str] = {}
        for raw_entry in raw_retired:
            if not isinstance(raw_entry, dict):
                raise ValueError("each retired entry must be a dict")
            if set(raw_entry.keys()) != {"id", "value", "prev"}:
                raise ValueError(
                    "each retired entry must contain exactly 'id', 'value' "
                    "and 'prev'"
                )
            node_id = _parse_node_id(raw_entry["id"])
            if node_id in retired_ids:
                raise ValueError("duplicate retired node id")
            if node_id in stored_ids:
                raise ValueError("a retired node must not remain a stored record")
            value = raw_entry["value"]
            if not isinstance(value, str):
                raise ValueError("retired node value must be a string")
            raw_prev = raw_entry["prev"]
            predecessor: NodeId | None
            predecessor = None if raw_prev is None else _parse_node_id(raw_prev)
            retired_ids.add(node_id)
            retired_prev[node_id] = predecessor
            retired_value[node_id] = value

        raw_dots = raw["retired_dots"]
        if not isinstance(raw_dots, list) or not raw_dots:
            raise ValueError("compaction.retired_dots must be a non-empty list")
        retired_dots: set[Dot] = set()
        for raw_dot in raw_dots:
            dot = cls._parse_dot(raw_dot)
            if dot in retired_dots:
                raise ValueError("duplicate retired deletion dot")
            origin, counter = dot
            if counter > components.get(origin, 0):
                raise ValueError("retired dot is not covered by the clock")
            retired_dots.add(dot)

        raw_clock = raw["retired_clock"]
        if not isinstance(raw_clock, dict) or not raw_clock:
            raise ValueError("compaction.retired_clock must be a non-empty dict")
        retired_clock: dict[str, int] = {}
        for origin, value in raw_clock.items():
            if not isinstance(origin, str) or origin == "":
                raise ValueError("retired_clock keys must be non-empty strings")
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("retired_clock values must be positive ints")
            retired_clock[origin] = value

        # The prefix is exactly the per-origin maximum of the retired dots.
        expected: dict[str, int] = {}
        for origin, counter in retired_dots:
            if counter > expected.get(origin, 0):
                expected[origin] = counter
        if expected != retired_clock:
            raise ValueError("retired_clock is inconsistent with retired_dots")

        # Every skeleton edge must target a stored record or another
        # skeleton, and skeleton chains must not cycle. (Full structural
        # validation over the merged graph is repeated by RGA.from_snapshot
        # once the placeholder records are installed.)
        targets = stored_ids | retired_ids
        for node_id, predecessor in retired_prev.items():
            if predecessor is not None and predecessor not in targets:
                raise ValueError("retired node predecessor is missing")
        state: dict[NodeId, int] = {}
        for start in retired_ids:
            if state.get(start) == 2:
                continue
            path: list[NodeId] = []
            cursor: NodeId | None = start
            while (
                cursor is not None
                and cursor in retired_prev
                and state.get(cursor) != 2
            ):
                if state.get(cursor) == 1:
                    raise ValueError("retired predecessor cycle")
                state[cursor] = 1
                path.append(cursor)
                cursor = retired_prev[cursor]
            for visited in path:
                state[visited] = 2

        return retired_ids, retired_prev, retired_value, retired_dots, retired_clock

    def __repr__(self) -> str:
        return (
            f"RGASession(replica_id={self.replica_id!r}, "
            f"values={self.values()!r})"
        )
