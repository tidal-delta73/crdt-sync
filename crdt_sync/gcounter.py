"""Grow-only counter CRDT.

Each replica keeps a per-replica non-negative integer component. The local
replica only increments its own component; merging takes the component-wise
maximum across replicas.
"""

from __future__ import annotations


def _validate_replica_id(replica_id: object) -> str:
    if not isinstance(replica_id, str):
        raise TypeError("replica_id must be a non-empty string")
    if replica_id == "":
        raise ValueError("replica_id must be a non-empty string")
    return replica_id


def _validate_amount(amount: object) -> int:
    if not isinstance(amount, int) or isinstance(amount, bool):
        raise TypeError("amount must be an integer")
    if amount <= 0:
        raise ValueError("amount must be greater than zero")
    return amount


class GCounter:
    """A grow-only counter identified by a local ``replica_id``."""

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        self._counts: dict[str, int] = {}

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def increment(self, amount: int = 1) -> None:
        """Increase the local replica's component by ``amount`` (default 1)."""
        amount = _validate_amount(amount)
        self._counts[self._replica_id] = (
            self._counts.get(self._replica_id, 0) + amount
        )

    def value(self) -> int:
        """Return the sum of all known replica components."""
        return sum(self._counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        """Take the component-wise maximum with ``other``, in place.

        Returns ``self``; ``other`` is never modified. Repeated or reordered
        merges converge to the same state.
        """
        if not isinstance(other, GCounter):
            raise TypeError("can only merge with another GCounter")
        for replica_id, other_count in other._counts.items():
            current = self._counts.get(replica_id, 0)
            if other_count > current:
                self._counts[replica_id] = other_count
        return self

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this counter."""
        counts = {key: self._counts[key] for key in sorted(self._counts)}
        return {"replica_id": self._replica_id, "counts": counts}

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "GCounter":
        """Restore a counter from a :meth:`snapshot`-compatible dict."""
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        keys = set(snapshot.keys())
        if keys != {"replica_id", "counts"}:
            raise ValueError(
                "snapshot must contain exactly 'replica_id' and 'counts'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        counts = snapshot["counts"]
        if not isinstance(counts, dict):
            raise ValueError("counts must be a dict")

        restored: dict[str, int] = {}
        for key, component in counts.items():
            if not isinstance(key, str) or key == "":
                raise ValueError("counts keys must be non-empty strings")
            if (
                not isinstance(component, int)
                or isinstance(component, bool)
                or component < 0
            ):
                raise ValueError(
                    "counts values must be non-negative integers"
                )
            restored[key] = component

        counter = cls(replica_id)
        counter._counts = restored
        return counter

    def __repr__(self) -> str:
        return f"GCounter(replica_id={self._replica_id!r}, value={self.value()})"
