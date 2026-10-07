"""Vector clock for causal ordering across replicas.

Each replica keeps a per-replica non-negative integer component. The local
replica only advances its own component via :meth:`VectorClock.tick`; merging
takes the component-wise maximum across replicas. :meth:`VectorClock.compare`
orders two clocks as ``before``, ``after``, ``equal`` or ``concurrent``,
treating missing components as zero.
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
        self._clock: dict[str, int] = {}

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def components(self) -> dict[str, int]:
        """Return an independent copy of all known replica components.

        A fresh clock reports an empty dict; mutating the returned mapping
        never affects the clock.
        """
        return dict(self._clock)

    def tick(self, amount: int = 1) -> None:
        """Advance the local replica's component by ``amount`` (default 1)."""
        amount = _validate_amount(amount)
        self._clock[self._replica_id] = (
            self._clock.get(self._replica_id, 0) + amount
        )

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Take the component-wise maximum with ``other``, in place.

        Returns ``self``; the ``replica_id`` and ``other`` are never modified.
        Repeated or reordered merges converge to the same state.
        """
        if not isinstance(other, VectorClock):
            raise TypeError("can only merge with another VectorClock")
        for replica_id, other_component in other._clock.items():
            current = self._clock.get(replica_id, 0)
            if other_component > current:
                self._clock[replica_id] = other_component
        return self

    def compare(self, other: "VectorClock") -> str:
        """Compare causal history with ``other``.

        Missing components are treated as zero. Returns ``"before"`` when
        every component is less than or equal to ``other``'s with at least
        one strictly smaller, ``"after"`` for the reverse, ``"equal"`` when
        the clocks agree, and ``"concurrent"`` when each has events the
        other has not seen.
        """
        if not isinstance(other, VectorClock):
            raise TypeError("can only compare with another VectorClock")
        has_smaller = False
        has_greater = False
        for replica_id in set(self._clock) | set(other._clock):
            local = self._clock.get(replica_id, 0)
            remote = other._clock.get(replica_id, 0)
            if local < remote:
                has_smaller = True
            elif local > remote:
                has_greater = True
        if has_smaller and has_greater:
            return "concurrent"
        if has_smaller:
            return "before"
        if has_greater:
            return "after"
        return "equal"

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this clock."""
        clock = {key: self._clock[key] for key in sorted(self._clock)}
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
        vector_clock._clock = restored
        return vector_clock

    def __repr__(self) -> str:
        return (
            f"VectorClock(replica_id={self._replica_id!r}, "
            f"components={self.components()!r})"
        )
