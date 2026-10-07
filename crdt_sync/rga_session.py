"""Session bundling an RGA sequence replica with its causal context.

An :class:`RGASession` couples one :class:`~crdt_sync.rga.RGA` replica (held
as a :class:`~crdt_sync.session_rga.SessionRGA` carrying per-delete causal
context) and one :class:`~crdt_sync.vector_clock.VectorClock` under a single
``replica_id`` so the pair travels as one offline-editing and reconnect unit.
Local ``insert``/``delete`` edit the sequence exactly as on a bare RGA and
additionally tick the session's clock once per successful edit, so the clock
always counts the edits this replica has observed. A successful delete is
stamped with the clock components observed *after* its tick — the delete
event's own causal timestamp — and that timestamp rides along in snapshots.

``snapshot`` packages the sequence state and the causal context — both
stamped with the same ``replica_id`` — into one JSON-serializable exchange
unit, and ``merge`` absorbs a peer session's sequence and clock atomically
after reporting how the peer's causal context relates to the receiver's.

``compact(stable_clock)`` reclaims tombstone nodes whose delete event is
covered by the caller-supplied stability frontier, leaf to frontier, without
ever advancing the session clock or changing :meth:`values`. Reclaimed nodes
survive as a permanent retirement summary in the snapshot, so older packets,
duplicates and out-of-order delivery cannot bring them back.
"""

from __future__ import annotations

from .session_rga import SessionRGA
from .vector_clock import VectorClock


class RGASession:
    """An RGA sequence replica plus its vector clock, under one identity."""

    def __init__(self, replica_id: str) -> None:
        self._rga = SessionRGA(replica_id)
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
        advances the session clock's local component once and stamps the
        tombstone with the post-tick clock components as its delete causal
        context. A failed call (``TypeError``/``IndexError``) changes
        neither the sequence nor the clock.
        """
        node_id, removed = self._rga.delete_with_node(index)
        self._clock.tick()
        self._rga.record_delete_context(node_id, self._clock.components())
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
        reordered or stale merges converge to the same sequence and clock,
        and a compacted receiver filters pre-compaction packets through its
        retirement summary instead of resurrecting anything.

        Raises ``TypeError`` when ``other`` is not an ``RGASession``.
        Raises ``ValueError`` — leaving this session's sequence *and* clock
        exactly as they were — when the same RGA node id carries a
        different string or predecessor on the two sides.
        """
        if not isinstance(other, RGASession):
            raise TypeError("can only merge with another RGASession")
        relation = other._clock.compare(self._clock)
        # The sequence merge validates every shared node id and the combined
        # predecessor graph before mutating, so a conflict raises here with
        # the sequence untouched — and the clock merge below is never
        # reached, keeping the clock untouched too. The whole merge is
        # therefore atomic.
        self._rga.merge(other._rga)
        self._clock.merge(other._clock)
        return relation

    def compact(self, stable_clock: VectorClock) -> int:
        """Reclaim tombstone nodes covered by the ``stable_clock`` frontier.

        ``stable_clock`` is the stability frontier the caller derives from
        the acknowledgements of every replica taking part in sync: the
        greatest clock all participants have observed. A tombstone is
        reclaimable only when (a) its delete causal context is known and
        dominated by the frontier — deletes carried by legacy snapshots
        without causal context are never eligible and their deletion time
        is never guessed — and (b) it is no longer the predecessor of any
        retained node. Reclamation walks leaf-first, so a whole run of
        stable deleted branches is retired in one call, but never crosses a
        live node, an unstabilized delete or a still-referenced node.

        Reclaimed nodes move into the retirement summary, which preserves
        their value, predecessor anchor and delete context; the call never
        advances the session clock and never changes :meth:`values`.
        Returns the number of RGA node records actually reclaimed by this
        call; repeating the call (or reusing an already-satisfied frontier)
        returns ``0``.

        Raises ``TypeError`` when ``stable_clock`` is not a
        :class:`VectorClock`. Raises ``ValueError`` — leaving the session
        completely unchanged — when the frontier is concurrent with this
        session's clock or ahead of it, since such a frontier cannot
        certify deletes against state this replica actually holds.
        """
        if not isinstance(stable_clock, VectorClock):
            raise TypeError("stable_clock must be a VectorClock")
        # The frontier may only sit at or behind this session's clock: this
        # replica must itself have observed everything the frontier claims
        # is stable. A concurrent frontier includes unobserved edits, and a
        # future one cannot be checked against local history at all.
        relation = stable_clock.compare(self._clock)
        if relation in ("concurrent", "after"):
            raise ValueError(
                "stable clock frontier must be dominated by this session's "
                "clock"
            )

        stable = stable_clock.components()
        rga = self._rga

        def covered(node_id) -> bool:
            context = rga._deletes.get(node_id)
            # A tombstone with no recorded delete context (a legacy
            # snapshot) is permanently ineligible; its delete time is never
            # inferred.
            if context is None:
                return False
            return all(
                stable.get(replica_id, 0) >= component
                for replica_id, component in context.items()
            )

        reclaimed = 0
        while True:
            # Predecessors still referenced by retained records anchor the
            # visible graph (or a not-yet-stable deleted branch) and cannot
            # be removed. Retired anchors live in the summary instead and do
            # not block their own predecessor from retiring later.
            referenced = {
                predecessor
                for _value, predecessor in rga._nodes.values()
                if predecessor is not None
            }
            candidates = [
                node_id
                for node_id in rga._tombstones
                if node_id not in referenced and covered(node_id)
            ]
            if not candidates:
                break
            # Peeling eligible leaves confluently reaches the same maximal
            # reclaimable set in any order; sort only for deterministic
            # application.
            candidates.sort()
            node_id = candidates[0]
            value, predecessor = rga._nodes[node_id]
            rga.retire(
                node_id,
                value,
                predecessor,
                rga._deletes[node_id],
            )
            reclaimed += 1
        return reclaimed

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of the whole session.

        The package carries exactly ``replica_id``, ``rga`` and ``clock``;
        all three name the same replica. The nested sequence snapshot keeps
        its classic four fields and additionally carries the per-node
        delete causal context (``deletes``) and retirement summary
        (``retired``) once they exist, so delete causality and compaction
        round-trip completely. Every level is rebuilt on each call, so
        mutating the result never affects the session and two snapshots are
        independent of each other.
        """
        return {
            "replica_id": self.replica_id,
            "rga": self._rga.session_snapshot(),
            "clock": self._clock.snapshot(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "RGASession":
        """Restore a session from a :meth:`snapshot`-compatible dict.

        Raises ``TypeError`` when ``snapshot`` is not a dict. Raises
        ``ValueError`` — without producing a partially restored session —
        when fields are missing or unexpected, when either nested snapshot
        is invalid, or when the three ``replica_id`` values disagree. The
        historical three-field session package with a plain four-field RGA
        snapshot is accepted unchanged: its tombstones carry no delete
        causal context and are therefore never eligible for compaction, but
        merge, continued editing and the exception semantics all work as
        before. A restored session keeps its Lamport counter and clock
        components, so continued editing mints fresh node ids and never
        reuses or rolls back a clock component.
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

        rga = SessionRGA.session_from_snapshot(rga_snapshot)
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
