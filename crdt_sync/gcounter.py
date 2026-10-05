"""Grow-only counter (GCounter) CRDT.

A GCounter is a state-based CRDT: each replica increments only its own
component, and replicas converge by merging pairwise maxima per component.
Merge is commutative, associative and idempotent, so replicas that have
received the same set of states agree on the value regardless of order or
duplication of deliveries.
"""
from __future__ import annotations

__all__ = ["GCounter"]

_SNAPSHOT_FIELDS = {"replica_id", "counts"}


def _validate_replica_id(replica_id: object) -> str:
    if not isinstance(replica_id, str):
        raise TypeError(
            f"replica_id must be a str, got {type(replica_id).__name__}"
        )
    if not replica_id:
        raise ValueError("replica_id must be a non-empty string")
    return replica_id


class GCounter:
    """A grow-only counter identified by a non-empty ``replica_id``."""

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        self._counts: dict[str, int] = {replica_id: 0}

    @property
    def replica_id(self) -> str:
        return self._replica_id

    @property
    def value(self) -> int:
        """Sum of all known per-replica components."""
        return sum(self._counts.values())

    def increment(self, amount: int = 1) -> "GCounter":
        """Increment this replica's own component; returns self."""
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise TypeError(
                f"amount must be an int, got {type(amount).__name__}"
            )
        if amount <= 0:
            raise ValueError("amount must be a positive integer")
        self._counts[self._replica_id] = (
            self._counts.get(self._replica_id, 0) + amount
        )
        return self

    def merge(self, other: "GCounter") -> "GCounter":
        """Merge ``other`` into self (per-component max); returns self.

        ``other`` is never modified.
        """
        if not isinstance(other, GCounter):
            raise TypeError(
                f"can only merge another GCounter, got {type(other).__name__}"
            )
        for rid, count in other._counts.items():
            if count > self._counts.get(rid, 0):
                self._counts[rid] = count
        return self

    def snapshot(self) -> dict:
        """Return a fresh JSON-serializable dict of the current state.

        The returned dict shares no mutable state with this counter.
        """
        return {
            "replica_id": self._replica_id,
            "counts": {rid: self._counts[rid] for rid in sorted(self._counts)},
        }

    @classmethod
    def from_snapshot(cls, data: object) -> "GCounter":
        """Restore a counter from a dict produced by :meth:`snapshot`."""
        if not isinstance(data, dict):
            raise TypeError(
                f"snapshot must be a dict, got {type(data).__name__}"
            )
        if set(data) != _SNAPSHOT_FIELDS:
            raise ValueError(
                "snapshot must contain exactly the fields "
                "'replica_id' and 'counts'"
            )
        replica_id = data["replica_id"]
        if not isinstance(replica_id, str) or not replica_id:
            raise ValueError("snapshot replica_id must be a non-empty string")
        counts = data["counts"]
        if not isinstance(counts, dict):
            raise ValueError("snapshot counts must be a dict")
        validated: dict[str, int] = {}
        for rid, count in counts.items():
            if not isinstance(rid, str) or not rid:
                raise ValueError("snapshot counts keys must be non-empty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    "snapshot counts values must be non-negative integers"
                )
            validated[rid] = count
        counter = cls(replica_id)
        counter._counts = validated
        return counter
