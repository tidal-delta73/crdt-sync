"""Observed-remove set (ORSet) CRDT for removable string sets.

Every local ``add`` mints a unique tag ``(replica_id, counter)`` attached to
the added element. An element is visible as long as it owns at least one tag
that has not been observed by a ``remove``: ``remove(e)`` tombstones exactly
the tags visible on this replica at call time. Both the per-element tag map
and the tombstone set only ever grow, and merging takes their union, so
delivery is idempotent, commutative and associative, stale or duplicated
snapshots can never resurrect a removed add, and a fresh ``add`` after a
remove is never swallowed by the historical tombstone.
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


class ORSet:
    """An observed-remove set of strings identified by a ``replica_id``."""

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        self._counter = 0
        self._adds: dict[str, set[Tag]] = {}
        self._removes: set[Tag] = set()

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

    def merge(self, other: "ORSet") -> "ORSet":
        """Union the tag map and tombstone set with ``other``, in place.

        Returns ``self``; ``other`` is never modified. Repeated or reordered
        merges converge to the same state.

        Raises ``ValueError`` — leaving both states untouched — when the two
        states bind the same tag to different elements, which only happens
        when distinct replicas were created with the same ``replica_id``.
        """
        if not isinstance(other, ORSet):
            raise TypeError("can only merge with another ORSet")
        # A tag minted by one (replica_id, counter) pair names exactly one
        # element. If the two states bind the same tag to different elements
        # (e.g. two replicas were created with the same replica_id and each
        # minted the tag independently), merging would make the tag
        # ambiguous and the result unrestorable, so reject it up front
        # without touching either state. The full owner map of ``self`` is
        # built before checking, so detection never depends on dict order.
        tag_owner: dict[Tag, str] = {}
        for element, tags in self._adds.items():
            for tag in tags:
                tag_owner[tag] = element
        for element, tags in other._adds.items():
            for tag in tags:
                owner = tag_owner.get(tag)
                if owner is not None and owner != element:
                    raise ValueError(
                        "the same tag is bound to different elements"
                    )
        for element, tags in other._adds.items():
            self._adds.setdefault(element, set()).update(tags)
        self._removes.update(other._removes)
        # A state carrying our own replica id (e.g. a restored backup of this
        # replica) may know about more of our own adds; advance the counter
        # past every observed tag of ours so future tags stay unique.
        for tags in self._adds.values():
            for origin, sequence in tags:
                if origin == self._replica_id and sequence > self._counter:
                    self._counter = sequence
        return self

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this set."""
        adds = {
            element: [[origin, sequence] for origin, sequence in sorted(tags)]
            for element, tags in sorted(self._adds.items())
        }
        removes = [
            [origin, sequence] for origin, sequence in sorted(self._removes)
        ]
        return {
            "replica_id": self._replica_id,
            "counter": self._counter,
            "adds": adds,
            "removes": removes,
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "ORSet":
        """Restore an ORSet from a :meth:`snapshot`-compatible dict."""
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        if set(snapshot.keys()) != {
            "replica_id",
            "counter",
            "adds",
            "removes",
        }:
            raise ValueError(
                "snapshot must contain exactly 'replica_id', 'counter', "
                "'adds' and 'removes'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        counter = snapshot["counter"]
        if not isinstance(counter, int) or isinstance(counter, bool):
            raise ValueError("counter must be a non-negative integer")
        if counter < 0:
            raise ValueError("counter must be a non-negative integer")

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

        # Causal validity: a tombstone may only refer to an observed add...
        if not removes <= set(tag_owner):
            raise ValueError("removes references a tag with no matching add")
        # ...and the local counter must describe exactly this replica's own
        # observed add history (1..counter, no gaps, no unknown future tags).
        own_sequences = sorted(
            sequence
            for origin, sequence in tag_owner
            if origin == replica_id
        )
        if own_sequences != list(range(1, counter + 1)):
            raise ValueError(
                "counter is inconsistent with the replica's add history"
            )

        restored = cls(replica_id)
        restored._counter = counter
        restored._adds = adds
        restored._removes = removes
        return restored

    def __repr__(self) -> str:
        return (
            f"ORSet(replica_id={self._replica_id!r}, "
            f"elements={sorted(self.elements())!r})"
        )
