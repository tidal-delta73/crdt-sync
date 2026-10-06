"""Replicated growable array (RGA) CRDT for an editable, deletable string
sequence.

Every local ``insert`` mints a unique identifier ``(replica_id, counter)`` and
links the new node after the node that currently precedes the insertion point
(or at the virtual root when inserting at the front). Concurrent inserts after
the same predecessor are siblings; they are ordered by the greater identifier
first — counter first, then the Unicode code point order of the originating
``replica_id`` — and each node's own descendants immediately follow it, so any
set of observed inserts always determines one unique sequence (the classical
RGA / TreedFS order).

A ``delete`` records the deleted node's identifier in a tombstone set, which
only grows. Because tombstones identify nodes rather than positions, replayed
or duplicated deletes can neither raise nor resurrect anything, a concurrent
insert the deleter had not observed is unaffected, and a fresh insert after a
delete is never swallowed by the historical tombstone. Both the node map and
the tombstone set only grow, so merging (their union, after a consistency
check) is idempotent, commutative and associative regardless of delivery
order.
"""

from __future__ import annotations

NodeId = tuple[str, int]
ROOT: NodeId | None = None


def _validate_replica_id(replica_id: object) -> str:
    if not isinstance(replica_id, str):
        raise TypeError("replica_id must be a non-empty string")
    if replica_id == "":
        raise ValueError("replica_id must be a non-empty string")
    return replica_id


def _validate_index(index: object, length: int) -> int:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("index must be an integer")
    if index < 0 or index > length:
        raise IndexError("index out of range")
    return index


def _parse_node_id(raw_id: object) -> NodeId:
    """Parse a JSON-shape ``[origin_id, counter]`` identifier into a tuple."""
    if not isinstance(raw_id, list) or len(raw_id) != 2:
        raise ValueError("node ids must be [origin_id, counter] lists")
    origin, counter = raw_id
    if not isinstance(origin, str) or origin == "":
        raise ValueError("node id origin must be a non-empty string")
    if not isinstance(counter, int) or isinstance(counter, bool):
        raise ValueError("node id counter must be an integer")
    if counter < 1:
        raise ValueError("node id counter must be greater than zero")
    return origin, counter


class RGA:
    """A replicated sequence of strings identified by a ``replica_id``."""

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        # Greatest local counter ever minted / observed for this replica id.
        self._counter = 0
        # node id -> (value, predecessor id); predecessor is None for a node
        # inserted at the front (a child of the virtual root).
        self._nodes: dict[NodeId, tuple[str, NodeId | None]] = {}
        self._tombstones: set[NodeId] = set()

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def _visible_ids(self) -> list[NodeId]:
        """Return the ids of all live nodes in canonical RGA order."""
        # Children grouped by predecessor; sibling groups sorted by the RGA
        # tie-break (greater counter, then greater origin id) so the result is
        # independent of dict iteration order. Each sibling subtree is read
        # before the next sibling ("each element's successors immediately
        # follow its branch").
        children: dict[NodeId | None, list[NodeId]] = {}
        for node_id, (_value, predecessor) in self._nodes.items():
            children.setdefault(predecessor, []).append(node_id)
        for siblings in children.values():
            siblings.sort(key=lambda node_id: (node_id[1], node_id[0]), reverse=True)

        # Iterative pre-order depth-first walk: a node's whole subtree is read
        # before its next sibling ("each element's successors immediately
        # follow its branch"). Siblings are pushed in reverse so the first one
        # stays on top of the stack.
        ordered: list[NodeId] = []
        stack: list[NodeId] = list(reversed(children.get(ROOT, ())))
        while stack:
            node_id = stack.pop()
            ordered.append(node_id)
            stack.extend(reversed(children.get(node_id, ())))
        return [node_id for node_id in ordered if node_id not in self._tombstones]

    def values(self) -> list[str]:
        """Return a fresh list of the visible values in sequence order."""
        return [self._nodes[node_id][0] for node_id in self._visible_ids()]

    def insert(self, index: int, value: str) -> None:
        """Insert ``value`` at the visible zero-based ``index``.

        ``index`` may equal the current length to append. The new node is
        linked after the node currently preceding the insertion point and
        receives a fresh unique ``(replica_id, counter)`` identifier.
        """
        visible = self._visible_ids()
        index = _validate_index(index, len(visible))
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        predecessor = visible[index - 1] if index > 0 else ROOT
        self._counter += 1
        node_id = (self._replica_id, self._counter)
        self._nodes[node_id] = (value, predecessor)

    def delete(self, index: int) -> str:
        """Delete and return the visible value currently at ``index``.

        The node is retained as a tombstone: repeated deletes of the same node
        (delivered via merge) never raise or resurrect it, while concurrent
        inserts and later inserts at the same position stay visible.
        """
        visible = self._visible_ids()
        index = _validate_index(index, len(visible))
        if index >= len(visible):
            # _validate_index allows the one-past-the-end position; deletion
            # has no such position.
            raise IndexError("index out of range")
        node_id = visible[index]
        value = self._nodes[node_id][0]
        self._tombstones.add(node_id)
        return value

    def merge(self, other: "RGA") -> "RGA":
        """Absorb ``other``'s nodes and tombstones in place.

        Returns ``self``; ``other`` is never modified. The node map and
        tombstone set only grow, so delivery is idempotent, commutative and
        associative under reordering, duplication and stale replays.

        Raises ``ValueError`` — leaving both states untouched — when the same
        node identifier is bound to a different value or predecessor on the
        two sides, which means two live replicas share a ``replica_id`` and
        minted colliding identifiers.
        """
        if not isinstance(other, RGA):
            raise TypeError("can only merge with another RGA")

        # Resolve conflicts before touching any state so a rejected merge
        # leaves both sequences unchanged.
        for node_id, (value, predecessor) in other._nodes.items():
            local = self._nodes.get(node_id)
            if local is not None and local != (value, predecessor):
                raise ValueError(
                    "the same node id is bound to a different value or "
                    "predecessor"
                )

        for node_id, record in other._nodes.items():
            self._nodes.setdefault(node_id, record)
        self._tombstones.update(other._tombstones)
        # A state carrying our own replica id (e.g. a restored backup of this
        # replica) may know about more of our own inserts; advance the counter
        # past every observed node of ours so future identifiers stay unique.
        for origin, counter in self._nodes:
            if origin == self._replica_id and counter > self._counter:
                self._counter = counter
        return self

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this sequence."""
        nodes = [
            {
                "id": [origin, counter],
                "value": value,
                "predecessor": (
                    [predecessor[0], predecessor[1]]
                    if predecessor is not None
                    else None
                ),
            }
            for (origin, counter), (value, predecessor) in sorted(
                self._nodes.items()
            )
        ]
        tombstones = [
            [origin, counter] for origin, counter in sorted(self._tombstones)
        ]
        return {
            "replica_id": self._replica_id,
            "counter": self._counter,
            "nodes": nodes,
            "tombstones": tombstones,
        }

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
            if set(raw_node.keys()) != {"id", "value", "predecessor"}:
                raise ValueError(
                    "each node must contain exactly 'id', 'value' and "
                    "'predecessor'"
                )
            node_id = _parse_node_id(raw_node["id"])
            if node_id in nodes:
                raise ValueError("duplicate node id")
            value = raw_node["value"]
            if not isinstance(value, str):
                raise ValueError("node value must be a string")
            raw_predecessor = raw_node["predecessor"]
            if raw_predecessor is None:
                predecessor: NodeId | None = ROOT
            else:
                predecessor = _parse_node_id(raw_predecessor)
            nodes[node_id] = (value, predecessor)

        raw_tombstones = snapshot["tombstones"]
        if not isinstance(raw_tombstones, list):
            raise ValueError("tombstones must be a list")
        tombstones: set[NodeId] = set()
        for raw_id in raw_tombstones:
            node_id = _parse_node_id(raw_id)
            if node_id in tombstones:
                raise ValueError("duplicate tombstone")
            tombstones.add(node_id)

        # Causal validity: a tombstone names an observed node, every
        # predecessor names an observed node, and the predecessor graph is
        # acyclic (each node must be reachable from the virtual root).
        if not tombstones <= set(nodes):
            raise ValueError("tombstone references a node with no insert record")

        def reachable_from_root() -> set[NodeId]:
            children: dict[NodeId | None, list[NodeId]] = {}
            for node_id, (_value, predecessor) in nodes.items():
                children.setdefault(predecessor, []).append(node_id)
            seen: set[NodeId] = set()
            stack: list[NodeId] = list(children.get(ROOT, ()))
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(children.get(current, ()))
            return seen

        reachable = reachable_from_root()
        if len(reachable) != len(nodes):
            # Either a predecessor dangles (points at an unknown node) or a
            # predecessor cycle keeps nodes unreachable from the root.
            raise ValueError(
                "predecessor references must form an acyclic graph rooted at "
                "the virtual root"
            )

        # The local counter must describe exactly this replica's own insert
        # history (1..counter, no gaps, no unknown future counters).
        own_counters = sorted(
            node_counter
            for origin, node_counter in nodes
            if origin == replica_id
        )
        if own_counters != list(range(1, counter + 1)):
            raise ValueError(
                "counter is inconsistent with the replica's insert history"
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
