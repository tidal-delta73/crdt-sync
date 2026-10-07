"""An offline editing session pairing an RGA sequence with its causal context.

``RGASession`` bundles one :class:`~crdt_sync.rga.RGA` sequence copy with the
:class:`~crdt_sync.vector_clock.VectorClock` that tracks its causal history as
a single unit of offline editing and reconnect sync. The sequence and the
clock belong to the same non-empty ``replica_id``; a freshly created session
holds an empty sequence and an empty clock.

The local mutators mirror the RGA contract exactly: ``insert(index, value)``
inserts at a visible zero-based position and ``delete(index)`` tombstones the
visible element there, returning its string; ``values()`` returns a fresh
independent list. Every *successful* insert or delete advances the local
vector clock component exactly once. A call rejected with ``TypeError``,
``IndexError`` or ``ValueError`` changes neither the sequence nor the clock.

``snapshot`` emits one complete, JSON-serializable exchange package carrying
the session ``replica_id`` together with the nested RGA and vector clock
snapshots, and ``from_snapshot`` restores all three as a unit, refusing any
package whose nested state is invalid or whose three ``replica_id`` values
disagree. Restored sessions keep editing with node ids and clock components
that only move forward, thanks to the underlying RGA Lamport counter and
vector clock merge.

``merge`` accepts another ``RGASession`` (never a bare snapshot), first
classifies the peer's clock against the receiver as ``before``, ``after``,
``equal`` or ``concurrent`` — the same orientation as
``VectorClock.compare`` — and then unions both the sequence and the causal
context, returning that relation. The receiver keeps its own ``replica_id``
and the peer is never modified. Sequence union and clock join are both
idempotent, commutative and associative, so interleaved offline edits, stale,
duplicated or arbitrarily reordered packets all converge; merging an already
received state leaves the snapshot byte-identical. A non-session argument
raises ``TypeError`` and the existing RGA node-id conflict raises
``ValueError``, in both cases with the receiver exactly as it was before the
call.
"""

from __future__ import annotations

from .rga import RGA
from .vector_clock import VectorClock


class RGASession:
    """An RGA sequence and its vector clock under one shared ``replica_id``."""

    def __init__(self, replica_id: str) -> None:
        # Both members validate replica_id identically; construct the clock
        # from the RGA's accepted id so the two can never diverge.
        self._rga = RGA(replica_id)
        self._clock = VectorClock(self._rga.replica_id)

    @property
    def replica_id(self) -> str:
        return self._rga.replica_id

    def insert(self, index: int, value: str) -> None:
        """Insert ``value`` at the visible zero-based ``index``.

        Mirrors :meth:`RGA.insert`; on success the local vector clock advances
        exactly once. A rejected call (bad index type, out-of-range index or
        non-string value) changes neither the sequence nor the clock.
        """
        self._rga.insert(index, value)
        self._clock.tick()

    def delete(self, index: int) -> str:
        """Tombstone the visible element at ``index`` and return its string.

        Mirrors :meth:`RGA.delete`; on success the local vector clock advances
        exactly once. A rejected call (bad index type, out-of-range index on a
        possibly empty sequence) changes neither the sequence nor the clock.
        """
        removed = self._rga.delete(index)
        self._clock.tick()
        return removed

    def values(self) -> list[str]:
        """Return a fresh list of the visible strings in sequence order.

        The list is independent of the session: mutating it never affects the
        session, just like :meth:`RGA.values`.
        """
        return self._rga.values()

    def merge(self, other: "RGASession") -> str:
        """Absorb another session's sequence and causal context in place.

        Returns the receiver-clock relation to ``other`` — ``before`` when the
        receiver happens-before the peer, ``after`` when the peer
        happens-before the receiver, ``equal`` when their clocks agree,
        ``concurrent`` when each has edits the other has not — exactly the
        orientation of :meth:`VectorClock.compare`.

        The RGA histories, tombstones and counter union first; the vector
        clocks then take their component-wise maximum. The receiver keeps its
        ``replica_id`` and ``other`` is never modified. Repeated, stale or
        reordered merges converge, and merging an already received state
        leaves :meth:`snapshot` unchanged.

        Raises ``TypeError`` for a non-``RGASession`` argument and, reusing
        the RGA contract, ``ValueError`` when the same node id carries a
        different value or predecessor on the two sides. Either failure leaves
        the sequence and the clock exactly as they were before the call.
        """
        if not isinstance(other, RGASession):
            raise TypeError("can only merge with another RGASession")

        # Classify before mutating anything: the relation describes the two
        # clocks as they stood when the call began. compare is read-only.
        relation = self._clock.compare(other._clock)

        # RGA.merge validates every shared node id before applying any change,
        # so a conflict raises with both sequences untouched; the clock merge
        # that follows has no failure path of its own. Together that gives the
        # session whole-call atomic failure semantics.
        self._rga.merge(other._rga)
        self._clock.merge(other._clock)
        return relation

    def snapshot(self) -> dict[str, object]:
        """Return a fresh, JSON-serializable exchange package.

        Bundles the session ``replica_id`` with independent nested RGA and
        vector clock snapshots under keys ``rga`` and ``clock``. Mutating the
        returned package or any part of it never affects the session.
        """
        return {
            "replica_id": self._rga.replica_id,
            "rga": self._rga.snapshot(),
            "clock": self._clock.snapshot(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "RGASession":
        """Restore a session from a :meth:`snapshot`-compatible dict.

        Raises ``TypeError`` when ``snapshot`` is not a dict. Raises
        ``ValueError`` — without returning a partially restored session — when
        a field is missing or extra, the top-level ``replica_id`` is invalid,
        either nested snapshot is invalid, or the session, RGA and vector
        clock ``replica_id`` values do not all agree.
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

        # Build both members fully before touching any returned object, so a
        # rejected package can never leak a half-restored session. The nested
        # restorers raise TypeError only for a non-dict payload; normalize that
        # to ValueError here since a malformed nested snapshot is bad package
        # content, not the wrong top-level type.
        try:
            rga = RGA.from_snapshot(snapshot["rga"])
            clock = VectorClock.from_snapshot(snapshot["clock"])
        except TypeError as exc:
            raise ValueError("nested snapshot must be a dict") from exc

        if rga.replica_id != replica_id or clock.replica_id != replica_id:
            raise ValueError(
                "session, RGA and vector clock must share one replica_id"
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
