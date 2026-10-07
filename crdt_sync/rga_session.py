"""Session bundling an RGA sequence replica with its causal context.

An :class:`RGASession` couples one :class:`~crdt_sync.rga.RGA` replica and
one :class:`~crdt_sync.vector_clock.VectorClock` under a single
``replica_id`` so the pair travels as one offline-editing and reconnect
unit. Local ``insert``/``delete`` edit the sequence exactly as on a bare
RGA and additionally tick the session's clock once per successful edit, so
the clock always counts the edits this replica has observed. ``snapshot``
packages the sequence state and the causal context — both stamped with the
same ``replica_id`` — into one JSON-serializable exchange unit, and
``merge`` absorbs a peer session's sequence and clock atomically after
reporting how the peer's causal context relates to the receiver's.
"""

from __future__ import annotations

from .rga import RGA
from .vector_clock import VectorClock


class RGASession:
    """An RGA sequence replica plus its vector clock, under one identity."""

    def __init__(self, replica_id: str) -> None:
        self._rga = RGA(replica_id)
        self._clock = VectorClock(replica_id)

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
        advances the session clock's local component once. A failed call
        (``TypeError``/``IndexError``) changes neither the sequence nor the
        clock.
        """
        removed = self._rga.delete(index)
        self._clock.tick()
        return removed

    def merge(self, other: "RGASession") -> str:
        """Absorb ``other``'s sequence and clock; return the causal relation.

        The relation — ``before``/``after``/``equal``/``concurrent`` — is
        decided from both clocks *before* any state moves and describes
        ``other`` relative to this receiver: ``before`` when every edit the
        sender knew is already observed here, ``after`` when the sender is
        strictly ahead, ``equal`` when both contexts match, ``concurrent``
        when each side holds edits the other has not seen. The sequence and
        the causal context are then merged together; the receiver keeps its
        own ``replica_id`` and ``other`` is never modified. Repeated,
        reordered or stale merges converge to the same sequence and clock.

        Raises ``TypeError`` when ``other`` is not an ``RGASession``.
        Raises ``ValueError`` — leaving this session's sequence *and* clock
        exactly as they were — when the same RGA node id carries a
        different string or predecessor on the two sides.
        """
        if not isinstance(other, RGASession):
            raise TypeError("can only merge with another RGASession")
        relation = other._clock.compare(self._clock)
        # The RGA merge validates every shared node id before mutating, so
        # a conflict raises here with the sequence untouched — and the
        # clock merge below is never reached, keeping the clock untouched
        # too. The whole merge is therefore atomic.
        self._rga.merge(other._rga)
        self._clock.merge(other._clock)
        return relation

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of the whole session.

        The package carries exactly ``replica_id``, ``rga`` and ``clock``;
        all three name the same replica. Every level is rebuilt on each
        call, so mutating the result never affects the session and two
        snapshots are independent of each other.
        """
        return {
            "replica_id": self.replica_id,
            "rga": self._rga.snapshot(),
            "clock": self._clock.snapshot(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "RGASession":
        """Restore a session from a :meth:`snapshot`-compatible dict.

        Raises ``TypeError`` when ``snapshot`` is not a dict. Raises
        ``ValueError`` — without producing a partially restored session —
        when fields are missing or unexpected, when either nested snapshot
        is invalid, or when the three ``replica_id`` values disagree. A
        restored session keeps its Lamport counter and clock components, so
        continued editing mints fresh node ids and never reuses or rolls
        back a clock component.
        """
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        if set(snapshot.keys()) != {"replica_id", "rga", "clock"}:
            raise ValueError(
                "snapshot must contain exactly 'replica_id', 'rga' and "
                "'clock'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        rga_snapshot = snapshot["rga"]
        if not isinstance(rga_snapshot, dict):
            raise ValueError("rga snapshot must be a dict")
        clock_snapshot = snapshot["clock"]
        if not isinstance(clock_snapshot, dict):
            raise ValueError("clock snapshot must be a dict")

        rga = RGA.from_snapshot(rga_snapshot)
        clock = VectorClock.from_snapshot(clock_snapshot)
        if rga.replica_id != replica_id or clock.replica_id != replica_id:
            raise ValueError(
                "replica_id must match across 'replica_id', 'rga' and "
                "'clock'"
            )

        session = cls(replica_id)
        session._rga = rga
        session._clock = clock
        return session

    def __repr__(self) -> str:
        return (
            f"RGASession(replica_id={self.replica_id!r}, "
            f"values={self.values()!r})"
        )
