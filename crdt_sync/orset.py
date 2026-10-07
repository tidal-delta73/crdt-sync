"""Observed-remove set (ORSet) CRDT for removable string sets.

Every local ``add`` mints a unique tag ``(replica_id, counter)`` attached to
the added element. An element is visible as long as it owns at least one tag
that has not been observed by a ``remove``: ``remove(e)`` tombstones exactly
the tags visible on this replica at call time. Both the per-element tag map
and the tombstone set only ever grow, and merging takes their union, so
delivery is idempotent, commutative and associative, stale or duplicated
snapshots can never resurrect a removed add, and a fresh ``add`` after a
remove is never swallowed by the historical tombstone.

``compact`` bounds that growth without changing any visible state. For each
tag origin it finds the longest prefix ``1..n`` whose every add is known
locally *and* already tombstoned, moves that fully-dead tag history out of
the snapshot, and records ``n`` as that origin's retired bound in the
``compacted`` summary. Merging takes the per-origin maximum bound; add and
tombstone records at or below either side's bound are treated as already
consumed history, so a late or duplicated pre-compaction snapshot can never
resurrect a compacted element.
"""

from __future__ import annotations

Tag = tuple[str, int]


def _validate_replica_id(replica_id: object) -> str:
    if not isinstance(replica_id, str):
        raise TypeError("replica_id must be a non-empty string")
    if replica_id == "":
        raise ValueError("replica_id must be a non-empty string")
    return replica_id


def _validate_element(element: object) -> str:
    if not isinstance(element, str):
        raise TypeError("element must be a non-empty string")
    if element == "":
        raise ValueError("element must be a non-empty string")
    return element


def _parse_tag(raw_tag: object) -> Tag:
    """Parse a JSON-shape ``[origin_id, sequence]`` tag into a tuple."""
    if not isinstance(raw_tag, list):
        raise ValueError("tags must be [origin_id, sequence] lists")
    if len(raw_tag) != 2:
        raise ValueError("tags must be [origin_id, sequence] lists")
    origin, sequence = raw_tag
    if not isinstance(origin, str) or origin == "":
        raise ValueError("tag origin must be a non-empty string")
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise ValueError("tag sequence must be an integer")
    if sequence < 1:
        raise ValueError("tag sequence must be greater than zero")
    return origin, sequence


def _parse_bound(value: object) -> int:
    """Parse one positive-integer retired bound from a JSON snapshot."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("compacted bounds must be positive integers")
    if value < 1:
        raise ValueError("compacted bounds must be positive integers")
    return value


class ORSet:
    """An observed-remove set of strings identified by a ``replica_id``."""

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        self._counter = 0
        self._adds: dict[str, set[Tag]] = {}
        self._removes: set[Tag] = set()
        # Per-origin greatest sequence retired by compaction; absent origins
        # have no retired prefix. The bound only ever grows (merge takes max).
        self._compacted: dict[str, int] = {}

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def add(self, element: str) -> None:
        """Record one new addition of ``element`` under a fresh unique tag."""
        element = _validate_element(element)
        self._counter += 1
        tag = (self._replica_id, self._counter)
        self._adds.setdefault(element, set()).add(tag)

    def remove(self, element: str) -> bool:
        """Tombstone the currently visible additions of ``element``.

        Returns ``True`` when at least one visible addition was withdrawn,
        ``False`` (without touching state) when the element is not visible.
        Only additions already observed by this replica are affected; a
        concurrent add of the same element on another replica stays visible.
        """
        element = _validate_element(element)
        tags = self._adds.get(element)
        if not tags:
            return False
        live = {tag for tag in tags if tag not in self._removes}
        if not live:
            return False
        self._removes.update(live)
        return True

    def contains(self, element: str) -> bool:
        """Return whether ``element`` currently has a visible addition."""
        element = _validate_element(element)
        tags = self._adds.get(element)
        if not tags:
            return False
        return any(tag not in self._removes for tag in tags)

    def elements(self) -> set[str]:
        """Return a fresh set of all currently visible elements."""
        return {
            element
            for element, tags in self._adds.items()
            if any(tag not in self._removes for tag in tags)
        }

    def compact(self) -> int:
        """Retire the fully-deleted contiguous tag prefix of every origin.

        For each tag origin, retire the longest prefix ``1..n`` such that
        every one of its adds is present on this replica *and* already
        covered by a tombstone. Compaction stops at the first gap (a missing
        add), a still-visible tag, or a tombstone whose add was never
        observed; it never crosses such a point.

        The retired adds and their per-tag tombstones leave the snapshot
        state and are summarized by the origin's retired bound; visible
        elements and add/remove semantics are unchanged. Returns the number
        of explicit add records removed by this call, which is ``0`` when
        nothing new can be retired, so a repeated call is a no-op.
        """
        # Full locally observed add history, mapped to its owning element.
        owner: dict[Tag, str] = {}
        for element, tags in self._adds.items():
            for tag in tags:
                owner[tag] = element

        retired_total = 0
        for origin in sorted({tag_origin for tag_origin, _ in owner}):
            start = self._compacted.get(origin, 0)
            sequence = start + 1
            # Advance only while the next contiguous add exists locally and
            # is already tombstoned; a gap or a live tag ends the prefix.
            while (origin, sequence) in owner and (
                origin,
                sequence,
            ) in self._removes:
                sequence += 1
            bound = sequence - 1
            if bound <= start:
                continue
            for dead in range(start + 1, bound + 1):
                tag = (origin, dead)
                element = owner[tag]
                tags = self._adds[element]
                tags.remove(tag)
                if not tags:
                    del self._adds[element]
                self._removes.remove(tag)
            self._compacted[origin] = bound
            retired_total += bound - start
        return retired_total

    def merge(self, other: "ORSet") -> "ORSet":
        """Union the tag map and tombstone set with ``other``, in place.

        Returns ``self``; ``other`` is never modified. Repeated or reordered
        merges converge to the same state.

        Per-origin retired bounds are joined by maximum; add and tombstone
        records at or below the joined bound are already-consumed history and
        are skipped on both sides, so a late pre-compaction snapshot can
        never resurrect a retired element. Records above the bound follow the
        normal union rules.

        Raises ``ValueError`` — leaving both states untouched — when the
        same tag is bound to different elements on the two sides. Such a
        conflict means two live replicas share a ``replica_id`` and minted
        colliding tags; unioning them would make the tag's ownership
        ambiguous and the merged state unrestorable.
        """
        if not isinstance(other, ORSet):
            raise TypeError("can only merge with another ORSet")

        # Join retired bounds first (component-wise maximum).
        bounds: dict[str, int] = dict(self._compacted)
        for origin, bound in other._compacted.items():
            if bound > bounds.get(origin, 0):
                bounds[origin] = bound

        def above_bound(tag: Tag) -> bool:
            origin, sequence = tag
            return sequence > bounds.get(origin, 0)

        # A tag identifies one add of one element; verify against the full
        # retained add history of both sides that no tag changes ownership.
        # Tags at or below a bound are consumed history and take no part.
        # The check is total, so its outcome never depends on iteration
        # order, and running it before any mutation keeps failures atomic.
        ownership: dict[Tag, str] = {}
        for element, tags in self._adds.items():
            for tag in tags:
                if above_bound(tag):
                    ownership[tag] = element
        conflicts: set[Tag] = set()
        incoming: list[tuple[str, Tag]] = []
        for element, tags in other._adds.items():
            for tag in tags:
                if not above_bound(tag):
                    continue
                owner = ownership.get(tag)
                if owner is not None and owner != element:
                    conflicts.add(tag)
                incoming.append((element, tag))
        if conflicts:
            raise ValueError(
                "the same tag is bound to different elements: "
                + ", ".join(repr(tag) for tag in sorted(conflicts))
            )

        self._compacted = bounds

        # A higher bound learned from ``other`` may retire records this
        # replica still held explicitly; consume them just like a local
        # compaction would, so joins converge on one canonical state.
        for element in list(self._adds):
            tags = self._adds[element]
            dead = {tag for tag in tags if not above_bound(tag)}
            if not dead:
                continue
            tags.difference_update(dead)
            if not tags:
                del self._adds[element]
        self._removes = {tag for tag in self._removes if above_bound(tag)}

        # Union the surviving records of ``other`` above the joined bound.
        for element, tag in incoming:
            self._adds.setdefault(element, set()).add(tag)
        for tag in other._removes:
            if above_bound(tag):
                self._removes.add(tag)

        # The retired prefix certifies this much of our own add history, and
        # a state carrying our replica id (e.g. a restored backup) may know
        # about more of our own adds; advance the counter past every known
        # own tag so future tags stay unique.
        own_bound = bounds.get(self._replica_id, 0)
        if own_bound > self._counter:
            self._counter = own_bound
        for tags in self._adds.values():
            for origin, sequence in tags:
                if origin == self._replica_id and sequence > self._counter:
                    self._counter = sequence
        return self

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this set.

        The classic four-field shape is used while no compaction summary
        exists; once at least one origin is retired, a fifth ``compacted``
        mapping of origin to positive retired bound is appended with keys in
        stable sorted order.
        """
        adds = {
            element: [[origin, sequence] for origin, sequence in sorted(tags)]
            for element, tags in sorted(self._adds.items())
        }
        removes = [
            [origin, sequence] for origin, sequence in sorted(self._removes)
        ]
        snapshot: dict[str, object] = {
            "replica_id": self._replica_id,
            "counter": self._counter,
            "adds": adds,
            "removes": removes,
        }
        if self._compacted:
            snapshot["compacted"] = {
                origin: self._compacted[origin]
                for origin in sorted(self._compacted)
            }
        return snapshot

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "ORSet":
        """Restore an ORSet from a :meth:`snapshot`-compatible dict.

        Both the original four-field shape and the shape that additionally
        carries a ``compacted`` summary are accepted.
        """
        base_keys = {"replica_id", "counter", "adds", "removes"}
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        keys = set(snapshot.keys())
        if keys == base_keys:
            has_compacted = False
        elif keys == base_keys | {"compacted"}:
            has_compacted = True
        else:
            raise ValueError(
                "snapshot must contain exactly 'replica_id', 'counter', "
                "'adds' and 'removes', and optionally 'compacted'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        counter = snapshot["counter"]
        if not isinstance(counter, int) or isinstance(counter, bool):
            raise ValueError("counter must be a non-negative integer")
        if counter < 0:
            raise ValueError("counter must be a non-negative integer")

        compacted: dict[str, int] = {}
        if has_compacted:
            raw_compacted = snapshot["compacted"]
            if not isinstance(raw_compacted, dict):
                raise ValueError("compacted must be a dict")
            for origin, raw_bound in raw_compacted.items():
                if not isinstance(origin, str) or origin == "":
                    raise ValueError(
                        "compacted keys must be non-empty origin strings"
                    )
                compacted[origin] = _parse_bound(raw_bound)

        raw_adds = snapshot["adds"]
        if not isinstance(raw_adds, dict):
            raise ValueError("adds must be a dict")

        adds: dict[str, set[Tag]] = {}
        tag_owner: dict[Tag, str] = {}
        for element, raw_tags in raw_adds.items():
            if not isinstance(element, str) or element == "":
                raise ValueError("adds keys must be non-empty strings")
            if not isinstance(raw_tags, list):
                raise ValueError("adds values must be lists of tags")
            if not raw_tags:
                raise ValueError("adds values must not be empty tag lists")
            tags: set[Tag] = set()
            for raw_tag in raw_tags:
                tag = _parse_tag(raw_tag)
                owner = tag_owner.get(tag)
                if owner is not None and owner != element:
                    raise ValueError(
                        "the same tag is bound to different elements"
                    )
                tag_owner[tag] = element
                tags.add(tag)
            adds[element] = tags

        raw_removes = snapshot["removes"]
        if not isinstance(raw_removes, list):
            raise ValueError("removes must be a list of tags")
        removes: set[Tag] = set()
        for raw_tag in raw_removes:
            removes.add(_parse_tag(raw_tag))

        # Explicit records at or below a retired bound are already-consumed
        # history and must have been removed, not shipped again.
        for origin, sequence in (*tag_owner, *removes):
            if sequence <= compacted.get(origin, 0):
                raise ValueError(
                    "explicit tags must lie above the compacted bound"
                )

        # Causal validity: a tombstone may only refer to an observed add or
        # to the certified compacted prefix (known history without records).
        def observed(tag: Tag) -> bool:
            origin, sequence = tag
            return tag in tag_owner or sequence <= compacted.get(origin, 0)

        if not all(observed(tag) for tag in removes):
            raise ValueError("removes references a tag with no matching add")

        # The local counter must describe exactly this replica's own add
        # history: the retired prefix 1..bound plus the explicit tags
        # bound+1..counter, with no gaps and no unknown future tags.
        own_bound = compacted.get(replica_id, 0)
        if own_bound > counter:
            raise ValueError(
                "counter is inconsistent with the replica's add history"
            )
        own_sequences = sorted(
            sequence
            for origin, sequence in tag_owner
            if origin == replica_id
        )
        if own_sequences != list(range(own_bound + 1, counter + 1)):
            raise ValueError(
                "counter is inconsistent with the replica's add history"
            )

        restored = cls(replica_id)
        restored._counter = counter
        restored._adds = adds
        restored._removes = removes
        restored._compacted = compacted
        return restored

    def __repr__(self) -> str:
        return (
            f"ORSet(replica_id={self._replica_id!r}, "
            f"elements={sorted(self.elements())!r})"
        )
