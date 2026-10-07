"""RGA variant backing :class:`~crdt_sync.rga_session.RGASession`.

This subclass keeps the bare :class:`~crdt_sync.rga.RGA` insertion history and
tombstone semantics but adds the two pieces of causal metadata that safe
sequence compaction needs:

* ``_deletes`` maps every tombstoned node to the vector-clock components the
  deleting replica had observed when it deleted the node. The contexts travel
  with snapshots and merge component-wise, so a delete's whole causal
  ancestry is knowable on every replica without changing how deletes apply:
  a node is deleted as soon as anyone has observed it deleted.
* ``_retired`` is the retirement summary. A compacted-away node keeps a
  permanent entry here — its value, predecessor link and the delete causal
  context certified at retirement — so the node's insert record and
  tombstone can leave the state while (a) a late, duplicated or
  out-of-order pre-compaction packet can never resurrect it, (b) a
  concurrent insert that only arrives later still has its anchor to weave
  against, and (c) a node id that ever carries conflicting content still
  raises, even when one side remembers the node only through the summary.
  Entries merge by component-wise maximum of the delete context.

Tombstones whose delete ancestry is unknown (the historical four-field RGA
snapshot shape) carry no delete context and are therefore never eligible for
compaction; their deletion time is never guessed.
"""

from __future__ import annotations

from .rga import RGA, NodeId, _validate_index

Components = dict[str, int]
# Retired node -> (value at insert time, predecessor link, delete context).
RetiredRecord = tuple[str, NodeId | None, Components]


def _parse_components(raw: object) -> Components:
    """Parse one JSON-shape vector-clock component mapping."""
    if not isinstance(raw, dict):
        raise ValueError("causal clock context must be a dict")
    components: Components = {}
    for replica_id, value in raw.items():
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("causal clock keys must be non-empty strings")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                "causal clock values must be non-negative integers"
            )
        components[replica_id] = value
    return components


def _merge_components(left: Components, right: Components) -> Components:
    """Component-wise maximum of two causal-context mappings."""
    merged = dict(left)
    for replica_id, value in right.items():
        if value > merged.get(replica_id, 0):
            merged[replica_id] = value
    return merged


def _parse_node_id(raw_id: object) -> NodeId:
    """Parse a JSON-shape ``[sequence, origin_id]`` node id into a tuple."""
    if not isinstance(raw_id, list) or len(raw_id) != 2:
        raise ValueError("node ids must be [sequence, origin_id] lists")
    sequence, origin = raw_id
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise ValueError("node id sequence must be an integer")
    if sequence < 1:
        raise ValueError("node id sequence must be greater than zero")
    if not isinstance(origin, str) or origin == "":
        raise ValueError("node id origin must be a non-empty string")
    return sequence, origin


def _encode_predecessor(predecessor: NodeId | None) -> list[object] | None:
    if predecessor is None:
        return None
    return [predecessor[0], predecessor[1]]


def _parse_predecessor(raw_prev: object) -> NodeId | None:
    return None if raw_prev is None else _parse_node_id(raw_prev)


class SessionRGA(RGA):
    """An :class:`RGA` carrying per-delete causal context and retirement data."""

    def __init__(self, replica_id: str) -> None:
        super().__init__(replica_id)
        # Tombstoned node id -> components observed by its deleter. A
        # tombstone missing here has unknown delete ancestry (legacy shape)
        # and can never be compacted.
        self._deletes: dict[NodeId, Components] = {}
        # Retired (compacted away) node id -> (value, predecessor, context).
        # Only ever grows; merge joins the context by component-wise max.
        self._retired: dict[NodeId, RetiredRecord] = {}

    # -- visibility ---------------------------------------------------------

    def _visible_nodes(self) -> list[NodeId]:
        """Weave the RGA tree, treating retired nodes as invisible anchors.

        Retired nodes keep no insert record and can never become visible
        again, but their predecessor link survives in the retirement summary
        so a retained node that hangs off a retired anchor (a concurrent
        child that arrived after the anchor was compacted away) is still
        woven in its anchor's exact position. Tombstones behave the same way
        in the ordinary RGA weave: invisible themselves, but their branches
        descend. Retired anchors participate in sibling ordering just like
        every other id, so the global id ordering is preserved.
        """
        # Parent id (None for the implicit root) -> every node directly
        # linked to it. Both retained and retired nodes are registered so a
        # retired anchor keeps its slot among its predecessor's siblings.
        children: dict[NodeId | None, list[NodeId]] = {}
        for node_id, (_value, predecessor) in self._nodes.items():
            children.setdefault(predecessor, []).append(node_id)
        for node_id, (_value, predecessor, _components) in self._retired.items():
            children.setdefault(predecessor, []).append(node_id)
        for siblings in children.values():
            # Descending id order, exactly like the bare RGA weave. Ids are
            # globally unique so retained and retired ids mix safely.
            siblings.sort(reverse=True)

        def invisible(node_id: NodeId) -> bool:
            return node_id in self._tombstones or node_id in self._retired

        ordered: list[NodeId] = []
        stack: list[NodeId] = list(reversed(children.get(None, ())))
        while stack:
            node_id = stack.pop()
            if not invisible(node_id):
                ordered.append(node_id)
            # Tombstoned and retired anchors alike still descend into their
            # branches; only live nodes contribute a visible entry.
            stack.extend(reversed(children.get(node_id, ())))
        return ordered

    # -- edits --------------------------------------------------------------

    def delete_with_node(self, index: int) -> tuple[NodeId, str]:
        """Tombstone the visible element at ``index``; return id and string.

        Like :meth:`RGA.delete` but also yields the tombstoned node id, so
        the owning session can stamp the delete with its causal context.
        Raises and leaves state untouched under the same conditions as
        :meth:`RGA.delete`.
        """
        index = _validate_index(index)
        visible = self._visible_nodes()
        if index < 0 or index >= len(visible):
            raise IndexError("delete index out of range")
        node_id = visible[index]
        value = self._nodes[node_id][0]
        self._tombstones.add(node_id)
        return node_id, value

    def record_delete_context(
        self, node_id: NodeId, components: Components
    ) -> None:
        """Attach the deleter's observed clock components to a tombstone."""
        existing = self._deletes.get(node_id)
        if existing is None:
            self._deletes[node_id] = dict(components)
        else:
            self._deletes[node_id] = _merge_components(existing, components)

    def retire(
        self,
        node_id: NodeId,
        value: str,
        predecessor: NodeId | None,
        components: Components,
    ) -> None:
        """Move one tombstoned leaf node into the retirement summary.

        The caller guarantees the node is a tombstoned leaf of the retained
        graph; this method only reconciles an already-known summary entry
        (which must agree on value and predecessor).
        """
        existing = self._retired.get(node_id)
        if existing is not None:
            old_value, old_predecessor, old_components = existing
            if old_value != value or old_predecessor != predecessor:
                raise ValueError(
                    "the same node id is bound to a different value or "
                    "predecessor"
                )
            components = _merge_components(old_components, components)
        self._retired[node_id] = (value, predecessor, components)
        self._nodes.pop(node_id, None)
        self._tombstones.discard(node_id)
        self._deletes.pop(node_id, None)

    # -- merge --------------------------------------------------------------

    def merge(self, other: "RGA") -> "SessionRGA":
        """Union ``other``'s history, tombstones and compaction metadata.

        Insert records and tombstones follow the ordinary RGA union rules.
        Retirement summaries are joined first: a node a joined summary
        certifies as retired is consumed history — any explicit record of it
        on either side is dropped, so a late, duplicated or pre-compaction
        packet can never resurrect it, while its value still protects
        against content conflicts and its predecessor anchor still positions
        concurrent children. Raises ``TypeError`` for a non-RGA argument and
        ``ValueError`` — before touching any state — when the same node id
        carries a different value or predecessor, or when the resulting
        predecessor graph is malformed.
        """
        if not isinstance(other, RGA):
            raise TypeError("can only merge with another RGA")

        other_deletes: dict[NodeId, Components] = getattr(
            other, "_deletes", {}
        )
        other_retired: dict[NodeId, RetiredRecord] = getattr(
            other, "_retired", {}
        )

        # Join the retirement summaries up front; every check below filters
        # on the joined view, mirroring how ORSet joins bounds first.
        retired: dict[NodeId, RetiredRecord] = {}
        for node_id, (value, predecessor, components) in self._retired.items():
            retired[node_id] = (value, predecessor, dict(components))
        for node_id, (value, predecessor, components) in other_retired.items():
            local = retired.get(node_id)
            if local is not None and (
                local[0] != value or local[1] != predecessor
            ):
                raise ValueError(
                    "the same node id is bound to a different value or "
                    "predecessor"
                )
            joined_context = _merge_components(
                local[2] if local is not None else {}, components
            )
            retired[node_id] = (value, predecessor, joined_context)

        # Explicit records the joined summary certifies are consumed history
        # and take no part in conflict checks or unions.
        incoming = {
            node_id: record
            for node_id, record in other._nodes.items()
            if node_id not in retired
        }

        # Validate every shared retained id before mutating anything so a
        # conflict leaves the whole merge atomic.
        for node_id, (value, predecessor) in incoming.items():
            local = self._nodes.get(node_id)
            if local is not None and local != (value, predecessor):
                raise ValueError(
                    "the same node id is bound to a different value or "
                    "predecessor"
                )

        # An explicit record whose id the joined summary certifies as
        # retired is consumed history only while it agrees with the
        # certified value and anchor; a colliding id with different content
        # still raises, exactly as with two retained records.
        for node_id, record in other._nodes.items():
            summary = retired.get(node_id)
            if summary is not None and record != (summary[0], summary[1]):
                raise ValueError(
                    "the same node id is bound to a different value or "
                    "predecessor"
                )

        # An explicit record this side still holds must agree with the value
        # and anchor the other side's retirement summary certifies.
        for node_id, (value, predecessor, _components) in retired.items():
            held = self._nodes.get(node_id)
            if held is not None and held != (value, predecessor):
                raise ValueError(
                    "the same node id is bound to a different value or "
                    "predecessor"
                )

        # Structural validity of the prospective combined graph: every
        # predecessor edge must resolve and the graph must be acyclic. It is
        # functional (one predecessor per node), so a three-color walk
        # settles both questions in linear time.
        graph: dict[NodeId, NodeId | None] = {
            node_id: self._nodes[node_id][1]
            for node_id in self._nodes
            if node_id not in retired
        }
        for node_id, (_value, predecessor) in incoming.items():
            graph.setdefault(node_id, predecessor)
        for node_id, (_value, predecessor, _components) in retired.items():
            graph.setdefault(node_id, predecessor)
        for node_id, predecessor in graph.items():
            if predecessor is not None and predecessor not in graph:
                raise ValueError("node predecessor is missing")
        state: dict[NodeId, int] = {}  # 1 = on path, 2 = settled
        for start in graph:
            if state.get(start) == 2:
                continue
            path: list[NodeId] = []
            cursor: NodeId | None = start
            while cursor is not None and state.get(cursor) != 2:
                if state.get(cursor) == 1:
                    raise ValueError("node predecessor cycle")
                state[cursor] = 1
                path.append(cursor)
                cursor = graph[cursor]
            for visited in path:
                state[visited] = 2

        # --- the merge is committed below this point ----------------------
        self._retired = {
            node_id: (value, predecessor, dict(components))
            for node_id, (value, predecessor, components) in retired.items()
        }

        # A higher summary learned from ``other`` may retire records this
        # replica still held explicitly; fold the richest known delete
        # context into the summary entry so contexts converge too.
        for node_id in list(self._nodes):
            if node_id in self._retired:
                value, predecessor, components = self._retired[node_id]
                richer = self._deletes.get(node_id)
                if richer is not None:
                    self._retired[node_id] = (
                        value,
                        predecessor,
                        _merge_components(components, richer),
                    )
                del self._nodes[node_id]
        self._tombstones = {
            node_id for node_id in self._tombstones if node_id not in retired
        }
        self._deletes = {
            node_id: context
            for node_id, context in self._deletes.items()
            if node_id not in retired
        }

        # Union the surviving records above the joined retirement summary.
        for node_id, record in incoming.items():
            if node_id not in self._nodes:
                self._nodes[node_id] = record
        for node_id in other._tombstones:
            if node_id not in retired and node_id in self._nodes:
                self._tombstones.add(node_id)

        # Join delete causal context: with a retained tombstone it enriches
        # the tombstone; with a retired anchor it enriches the summary.
        for node_id, components in other_deletes.items():
            if node_id in self._tombstones:
                self.record_delete_context(node_id, components)
            elif node_id in self._retired:
                value, predecessor, certified = self._retired[node_id]
                self._retired[node_id] = (
                    value,
                    predecessor,
                    _merge_components(certified, components),
                )

        if other._counter > self._counter:
            self._counter = other._counter
        return self

    # -- snapshots ----------------------------------------------------------

    def session_snapshot(self) -> dict[str, object]:
        """Return the session RGA snapshot shape.

        The classic four RGA fields are always present; ``deletes`` and
        ``retired`` are appended together only while at least one of them is
        non-empty, so a session that never causally deleted anything keeps
        serializing like the classic shape. Every level is rebuilt and
        sorted, so the result is deterministic, JSON-round-trippable and
        independent of the live state.
        """
        snapshot = self.snapshot()
        if self._deletes:
            snapshot["deletes"] = [
                {
                    "id": [node_id[0], node_id[1]],
                    "clock": {
                        replica_id: self._deletes[node_id][replica_id]
                        for replica_id in sorted(self._deletes[node_id])
                    },
                }
                for node_id in sorted(self._deletes)
            ]
        if self._retired:
            snapshot["retired"] = [
                {
                    "id": [node_id[0], node_id[1]],
                    "value": self._retired[node_id][0],
                    "prev": _encode_predecessor(self._retired[node_id][1]),
                    "deleted": {
                        replica_id: self._retired[node_id][2][replica_id]
                        for replica_id in sorted(self._retired[node_id][2])
                    },
                }
                for node_id in sorted(self._retired)
            ]
        return snapshot

    @classmethod
    def session_from_snapshot(cls, snapshot: object) -> "SessionRGA":
        """Restore from a classic four-field or session-extended snapshot.

        The classic RGA shape restores with empty causal metadata, leaving
        every tombstone without a delete context and therefore ineligible
        for compaction; ``deletes`` (causal tombstones) and ``retired``
        (the compaction summary) may each appear independently once
        sessions use them.
        """
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        base_keys = {"replica_id", "counter", "nodes", "tombstones"}
        extension_keys = {"deletes", "retired"}
        keys = set(snapshot.keys())
        if not (base_keys <= keys <= base_keys | extension_keys):
            raise ValueError(
                "snapshot must contain 'replica_id', 'counter', 'nodes' and "
                "'tombstones', and may additionally carry 'deletes' and "
                "'retired'"
            )
        has_extension = bool(keys & extension_keys)

        # Parse the ordinary RGA fields through the strict base validator
        # first, so the classic structural rules all apply unchanged.
        base = RGA.from_snapshot({key: snapshot[key] for key in base_keys})

        deletes: dict[NodeId, Components] = {}
        retired: dict[NodeId, RetiredRecord] = {}
        if has_extension:
            raw_deletes = snapshot.get("deletes", [])
            raw_retired = snapshot.get("retired", [])
            if not isinstance(raw_deletes, list):
                raise ValueError("deletes must be a list")
            if not isinstance(raw_retired, list):
                raise ValueError("retired must be a list")
            for entry in raw_deletes:
                if not isinstance(entry, dict) or set(entry.keys()) != {
                    "id",
                    "clock",
                }:
                    raise ValueError(
                        "each delete entry must contain exactly 'id' and "
                        "'clock'"
                    )
                node_id = _parse_node_id(entry["id"])
                if node_id in deletes:
                    raise ValueError("duplicate delete entry")
                deletes[node_id] = _parse_components(entry["clock"])
            for entry in raw_retired:
                if not isinstance(entry, dict) or set(entry.keys()) != {
                    "id",
                    "value",
                    "prev",
                    "deleted",
                }:
                    raise ValueError(
                        "each retired entry must contain exactly 'id', "
                        "'value', 'prev' and 'deleted'"
                    )
                node_id = _parse_node_id(entry["id"])
                if node_id in retired:
                    raise ValueError("duplicate retired entry")
                value = entry["value"]
                if not isinstance(value, str):
                    raise ValueError("retired node value must be a string")
                predecessor = _parse_predecessor(entry["prev"])
                components = _parse_components(entry["deleted"])
                if not components:
                    raise ValueError(
                        "retirement summary must carry the delete causal "
                        "context"
                    )
                retired[node_id] = (value, predecessor, components)

        nodes = base._nodes
        tombstones = base._tombstones

        # Causal metadata membership rules.
        for node_id in deletes:
            if node_id not in tombstones:
                raise ValueError(
                    "delete context must belong to a tombstoned node"
                )
        for node_id, (_value, _predecessor, _components) in retired.items():
            if node_id in nodes:
                # An explicit record of a certified-retired node is malformed
                # history, exactly like an explicit tag below an ORSet bound.
                raise ValueError(
                    "a retired node must keep no insert record"
                )
            if node_id in tombstones:
                raise ValueError("a retired node must keep no tombstone")
            if node_id in deletes:
                raise ValueError("a retired node must keep no delete record")

        # The counter must dominate every id the summary still knows about,
        # including retired ones, so continued inserts never reuse a seq.
        if retired and base._counter < max(
            node_id[0] for node_id in retired
        ):
            raise ValueError(
                "counter must not be below the greatest retired node "
                "sequence"
            )

        # Validate the combined predecessor graph (retained nodes plus
        # retired anchors): every edge resolves and the graph is acyclic.
        graph: dict[NodeId, NodeId | None] = {
            node_id: nodes[node_id][1] for node_id in nodes
        }
        for node_id, (_value, predecessor, _components) in retired.items():
            graph[node_id] = predecessor
        for node_id, predecessor in graph.items():
            if predecessor is not None and predecessor not in graph:
                raise ValueError("node predecessor is missing")
        mark: dict[NodeId, int] = {}  # 1 = on path, 2 = settled
        for start in graph:
            if mark.get(start) == 2:
                continue
            path: list[NodeId] = []
            cursor: NodeId | None = start
            while cursor is not None and mark.get(cursor) != 2:
                if mark.get(cursor) == 1:
                    raise ValueError("node predecessor cycle")
                mark[cursor] = 1
                path.append(cursor)
                cursor = graph[cursor]
            for visited in path:
                mark[visited] = 2

        restored = cls(base.replica_id)
        restored._counter = base._counter
        restored._nodes = nodes
        restored._tombstones = tombstones
        restored._deletes = deletes
        restored._retired = retired
        return restored
