"""Replicated growable array (RGA) CRDT for an editable string sequence.

Every local ``insert(index, value)`` mints a unique node id
``(sequence, replica_id)`` from a Lamport-style counter: the counter starts at
zero, is advanced to the greatest sequence observed via every merge, and is
incremented once per insert. Nodes hang off their predecessor (the visible
element immediately to the left at insert time, or the implicit head root).
When several inserts share a predecessor they are siblings and are ordered by
their id in *descending* order — greater sequence first, with the Unicode code
point order of ``replica_id`` breaking ties — and each sibling's own branch is
emitted before the next sibling. Because a freshly minted id is greater than
every id already known on this replica, an insert always lands exactly at the
requested zero-based visible position.

``delete`` is an observed remove: it tombstones only elements currently
visible on this replica. The node keeps its identity in the tombstone set, so
redelivery never errors or resurrects it; concurrent inserts never observed by
the remover are unaffected and a later insert at the same slot is never
swallowed by the old tombstone. The node map, predecessor links and tombstone
set only grow; merging unions them, which makes delivery idempotent,
commutative and associative.
"""

from __future__ import annotations

NodeId = tuple[int, str]

# Sentinel predecessor for nodes inserted at index 0 (the implicit head root).
ROOT: NodeId | None = None


def _validate_replica_id(replica_id: object) -> str:
    if not isinstance(replica_id, str):
        raise TypeError("replica_id must be a non-empty string")
    if replica_id == "":
        raise ValueError("replica_id must be a non-empty string")
    return replica_id


def _validate_index(index: object) -> int:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("index must be an integer")
    return index


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


class RGA:
    """A replicated editable string sequence identified by a ``replica_id``."""

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        # Lamport-style greatest sequence known on this replica. Local
        # inserts use counter + 1, so a new id always dominates every id that
        # was visible when the insertion position was chosen.
        self._counter = 0
        # node id -> (stored string, predecessor id or None for the root).
        self._nodes: dict[NodeId, tuple[str, NodeId | None]] = {}
        self._tombstones: set[NodeId] = set()

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def _visible_nodes(self) -> list[NodeId]:
        """Return all live node ids in RGA order (the tree weave)."""
        children: dict[NodeId | None, list[NodeId]] = {}
        for node_id, (_value, predecessor) in self._nodes.items():
            children.setdefault(predecessor, []).append(node_id)
        for siblings in children.values():
            # Descending id order: greater sequence first, then the greater
            # replica id in Unicode order. Tuple ordering is exactly that.
            siblings.sort(reverse=True)

        ordered: list[NodeId] = []
        # Stack holds nodes deepest-next first: siblings are pushed in
        # reverse sorted order so pop() takes the greatest id. The root
        # children follow the same rule as nested children.
        stack: list[NodeId] = list(reversed(children.get(ROOT, ())))
        while stack:
            node_id = stack.pop()
            ordered.append(node_id)
            # The depth-first walk keeps each sibling's branch contiguous.
            stack.extend(reversed(children.get(node_id, ())))
        return [
            node_id for node_id in ordered if node_id not in self._tombstones
        ]

    def values(self) -> list[str]:
        """Return a fresh list of the visible strings in sequence order."""
        return [self._nodes[node_id][0] for node_id in self._visible_nodes()]

    def insert(self, index: int, value: str) -> None:
        """Insert ``value`` at the visible zero-based ``index``.

        ``index`` may equal the current length to append. The new node gets a
        unique id minted from the local counter and this replica's id.
        Raises ``TypeError`` for a non-integer index (booleans included) or a
        non-string value, ``IndexError`` when the index is out of range; a
        failed call leaves the state unchanged.
        """
        index = _validate_index(index)
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        visible = self._visible_nodes()
        if index < 0 or index > len(visible):
            raise IndexError("insert index out of range")
        predecessor = visible[index - 1] if index > 0 else ROOT
        self._counter += 1
        node_id = (self._counter, self._replica_id)
        self._nodes[node_id] = (value, predecessor)

    def delete(self, index: int) -> str:
        """Tombstone the visible element at ``index`` and return its string.

        Only an element currently observed on this replica can be deleted; a
        concurrent insert never seen here is unaffected. Raises ``TypeError``
        for a non-integer index (booleans included) and ``IndexError`` when the
        sequence is empty or the index is out of range; a failed call leaves
        the state unchanged.
        """
        index = _validate_index(index)
        visible = self._visible_nodes()
        if index < 0 or index >= len(visible):
            raise IndexError("delete index out of range")
        node_id = visible[index]
        value = self._nodes[node_id][0]
        self._tombstones.add(node_id)
        return value

    def merge(self, other: "RGA") -> "RGA":
        """Absorb ``other``'s state in place and return ``self``.

        Unions the node histories, predecessor links and tombstones and adopts
        the greatest observed counter; ``other`` is never modified. Repeated,
        reordered or stale merges converge to the same sequence.

        Raises ``ValueError`` — leaving both states untouched — when the very
        same node id carries a different string or a different predecessor on
        the two sides, which means two live replicas share a ``replica_id``
        and minted colliding ids.
        """
        if not isinstance(other, RGA):
            raise TypeError("can only merge with another RGA")

        # Validate every shared id before mutating anything so a conflict
        # leaves both states unchanged.
        for node_id, (value, predecessor) in other._nodes.items():
            local = self._nodes.get(node_id)
            if local is not None and local != (value, predecessor):
                raise ValueError(
                    "the same node id is bound to a different value or "
                    "predecessor"
                )

        for node_id, record in other._nodes.items():
            if node_id not in self._nodes:
                self._nodes[node_id] = record
        self._tombstones.update(other._tombstones)
        if other._counter > self._counter:
            self._counter = other._counter
        return self

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this sequence."""
        nodes = [
            {
                "id": [sequence, origin],
                "value": self._nodes[(sequence, origin)][0],
                "prev": self._encode_predecessor(
                    self._nodes[(sequence, origin)][1]
                ),
            }
            for sequence, origin in sorted(self._nodes)
        ]
        tombstones = [
            [sequence, origin] for sequence, origin in sorted(self._tombstones)
        ]
        return {
            "replica_id": self._replica_id,
            "counter": self._counter,
            "nodes": nodes,
            "tombstones": tombstones,
        }

    @staticmethod
    def _encode_predecessor(predecessor: NodeId | None) -> list[object] | None:
        if predecessor is None:
            return None
        return [predecessor[0], predecessor[1]]

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "RGA":
        """Restore an RGA from a :meth:`snapshot`-compatible dict."""
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        if set(snapshot.keys()) != {
            "replica_id",
            "counter",
            "nodes",
            "tombstones",
        }:
            raise ValueError(
                "snapshot must contain exactly 'replica_id', 'counter', "
                "'nodes' and 'tombstones'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        counter = snapshot["counter"]
        if not isinstance(counter, int) or isinstance(counter, bool):
            raise ValueError("counter must be a non-negative integer")
        if counter < 0:
            raise ValueError("counter must be a non-negative integer")

        raw_nodes = snapshot["nodes"]
        if not isinstance(raw_nodes, list):
            raise ValueError("nodes must be a list")

        nodes: dict[NodeId, tuple[str, NodeId | None]] = {}
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                raise ValueError("each node must be a dict")
            if set(raw_node.keys()) != {"id", "value", "prev"}:
                raise ValueError(
                    "each node must contain exactly 'id', 'value' and 'prev'"
                )
            node_id = _parse_node_id(raw_node["id"])
            if node_id in nodes:
                raise ValueError("duplicate node id")
            value = raw_node["value"]
            if not isinstance(value, str):
                raise ValueError("node value must be a string")
            raw_prev = raw_node["prev"]
            predecessor: NodeId | None
            if raw_prev is None:
                predecessor = None
            else:
                predecessor = _parse_node_id(raw_prev)
            nodes[node_id] = (value, predecessor)

        raw_tombstones = snapshot["tombstones"]
        if not isinstance(raw_tombstones, list):
            raise ValueError("tombstones must be a list")
        tombstones: set[NodeId] = set()
        for raw_id in raw_tombstones:
            node_id = _parse_node_id(raw_id)
            if node_id in tombstones:
                raise ValueError("duplicate tombstone entry")
            tombstones.add(node_id)

        # Structural validity, first pass: every predecessor must exist.
        for _node_id, (_value, predecessor) in nodes.items():
            if predecessor is not None and predecessor not in nodes:
                raise ValueError("node predecessor is missing")

        # Second pass: predecessor edges form a functional graph (each node
        # points at at most one predecessor), so three-color walking detects
        # cycles in linear time overall.
        state: dict[NodeId, int] = {}  # 1 = on the current path, 2 = settled
        for start in nodes:
            if state.get(start) == 2:
                continue
            path: list[NodeId] = []
            cursor: NodeId | None = start
            while cursor is not None and state.get(cursor) != 2:
                if state.get(cursor) == 1:
                    raise ValueError("node predecessor cycle")
                state[cursor] = 1
                path.append(cursor)
                cursor = nodes[cursor][1]
            for visited in path:
                state[visited] = 2

        # A tombstone may only refer to an observed node...
        if not tombstones <= set(nodes):
            raise ValueError("tombstone references a node with no insert record")
        # ...and the counter must be at least the greatest observed sequence.
        # A Lamport clock is only ever advanced, never shared out with gaps, so
        # every counter at or above the maximum observed sequence is reachable
        # (this replica may simply have observed no insert of its own yet).
        if nodes and counter < max(sequence for sequence, _origin in nodes):
            raise ValueError(
                "counter must not be below the greatest observed node sequence"
            )

        restored = cls(replica_id)
        restored._counter = counter
        restored._nodes = nodes
        restored._tombstones = tombstones
        return restored

    def __repr__(self) -> str:
        return (
            f"RGA(replica_id={self._replica_id!r}, "
            f"values={self.values()!r})"
        )
