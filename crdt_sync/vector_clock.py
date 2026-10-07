"""Vector clock for causally ordering offline operations and reconnect sync.

Each clock keeps a per-replica non-negative integer component keyed by
``replica_id``. The local replica only advances its own component via
:meth:`tick`; merging takes the component-wise maximum, and :meth:`compare`
decides the happens-before relation treating unknown components as zero.
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


class VectorClock:
    """A vector clock identified by a local ``replica_id``."""

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        self._components: dict[str, int] = {}

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def components(self) -> dict[str, int]:
        """Return an independent copy of all known replica components.

        Mutating the returned mapping never affects the clock. Keys are
        produced in stable (sorted) order.
        """
        return {key: self._components[key] for key in sorted(self._components)}

    def tick(self, amount: int = 1) -> None:
        """Advance the local replica's component by ``amount`` (default 1)."""
        amount = _validate_amount(amount)
        self._components[self._replica_id] = (
            self._components.get(self._replica_id, 0) + amount
        )

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Take the component-wise maximum with ``other``, in place.

        Returns ``self``; the local ``replica_id`` is kept and ``other`` is
        never modified. Repeated, reordered or relayed merges converge to the
        same components.
        """
        if not isinstance(other, VectorClock):
            raise TypeError("can only merge with another VectorClock")
        for replica_id in set(self._components) | set(other._components):
            merged = max(
                self._components.get(replica_id, 0),
                other._components.get(replica_id, 0),
            )
            self._components[replica_id] = merged
        return self

    def compare(self, other: "VectorClock") -> str:
        """Compare causal order, returning ``before``/``after``/``equal``/``concurrent``.

        Components missing on either side count as zero. Every component less
        than or equal with at least one strictly less is ``before``; the
        reverse is ``after``; identical components are ``equal``; components
        strictly greater on both sides are ``concurrent``.
        """
        if not isinstance(other, VectorClock):
            raise TypeError("can only compare with another VectorClock")
        earlier = False
        later = False
        for replica_id in set(self._components) | set(other._components):
            local = self._components.get(replica_id, 0)
            remote = other._components.get(replica_id, 0)
            if local < remote:
                earlier = True
            elif local > remote:
                later = True
        if earlier and later:
            return "concurrent"
        if earlier:
            return "before"
        if later:
            return "after"
        return "equal"

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this clock."""
        clock = {key: self._components[key] for key in sorted(self._components)}
        return {"replica_id": self._replica_id, "clock": clock}

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "VectorClock":
        """Restore a clock from a :meth:`snapshot`-compatible dict."""
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        keys = set(snapshot.keys())
        if keys != {"replica_id", "clock"}:
            raise ValueError(
                "snapshot must contain exactly 'replica_id' and 'clock'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        clock = snapshot["clock"]
        if not isinstance(clock, dict):
            raise ValueError("clock must be a dict")

        restored: dict[str, int] = {}
        for replica, component in clock.items():
            if not isinstance(replica, str) or replica == "":
                raise ValueError("clock keys must be non-empty strings")
            if (
                not isinstance(component, int)
                or isinstance(component, bool)
                or component < 0
            ):
                raise ValueError(
                    "clock values must be non-negative integers"
                )
            restored[replica] = component

        vector_clock = cls(replica_id)
        vector_clock._components = restored
        return vector_clock

    def __repr__(self) -> str:
        return (
            f"VectorClock(replica_id={self._replica_id!r}, "
            f"components={self.components()!r})"
        )
