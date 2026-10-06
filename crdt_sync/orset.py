"""Observed-remove set (OR-Set) CRDT for string elements.

Every local ``add`` mints a unique tag (``"<replica_id>#<n>"``) and attaches
it to the element. An element is visible while it has at least one added tag
that has not been removed. ``remove`` only tombstones the tags the local
replica has observed at call time, so a concurrent add on another replica
survives until it has been observed.

Merging takes the set-wise union of both the add tags and the remove tags.
Both collections only grow, so delivery may be duplicated, reordered or
stale without ever resurrecting a removed tag or swallowing a newer add.
"""

from __future__ import annotations

_SNAPSHOT_KEYS = {"replica_id", "adds", "removes", "counter"}


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


def _split_tag(tag: object) -> tuple[str, int] | None:
    """Split a ``"<replica_id>#<seq>"`` tag, or return ``None`` if malformed."""
    if not isinstance(tag, str) or "#" not in tag:
        return None
    prefix, _, digits = tag.rpartition("#")
    if prefix == "" or not (digits.isascii() and digits.isdigit()):
        return None
    return prefix, int(digits)


class ORSet:
    """An observed-remove set of strings identified by a local ``replica_id``."""

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        self._adds: dict[str, set[str]] = {}
        self._removes: dict[str, set[str]] = {}
        # Sequence number of the next locally minted tag.
        self._counter = 0

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def _mint_tag(self) -> str:
        # Merges may reveal tags minted by this replica on another device (for
        # example after restoring an old snapshot); never mint a colliding tag.
        prefix = self._replica_id + "#"
        observed = 0
        for tags in self._adds.values():
            for tag in tags:
                if tag.startswith(prefix):
                    parsed = _split_tag(tag)
                    if parsed is not None and parsed[0] == self._replica_id:
                        observed = max(observed, parsed[1] + 1)
        seq = max(self._counter, observed)
        self._counter = seq + 1
        return f"{prefix}{seq}"

    def add(self, element: str) -> None:
        """Record one new addition of ``element``.

        Every call mints a fresh unique tag, so an element removed in the
        past can be added again and the new addition is never swallowed by
        the historical removal.
        """
        element = _validate_element(element)
        tag = self._mint_tag()
        self._adds.setdefault(element, set()).add(tag)

    def remove(self, element: str) -> bool:
        """Remove the tags currently observed for ``element``.

        Return ``True`` when at least one addition was visible and has been
        tombstoned. Return ``False`` (without changing state) when no visible
        addition exists.
        """
        element = _validate_element(element)
        observed = self._adds.get(element)
        if not observed:
            return False
        already_removed = self._removes.get(element)
        if already_removed is not None and observed <= already_removed:
            return False
        self._removes[element] = set(observed) | (
            set(already_removed) if already_removed is not None else set()
        )
        return True

    def contains(self, element: str) -> bool:
        """Return whether ``element`` currently has a visible addition."""
        element = _validate_element(element)
        observed = self._adds.get(element)
        if observed is None:
            return False
        removed = self._removes.get(element)
        if removed is None:
            return bool(observed)
        return bool(observed - removed)

    def elements(self) -> set[str]:
        """Return a fresh set of every currently visible element."""
        visible: set[str] = set()
        for element, added in self._adds.items():
            removed = self._removes.get(element)
            if removed is None or added - removed:
                visible.add(element)
        return visible

    def merge(self, other: "ORSet") -> "ORSet":
        """Union add tags and remove tags with ``other``, in place.

        Returns ``self``; ``other`` is never modified. Tag collections only
        grow, so the merge is idempotent, commutative and associative, and
        stale or duplicated snapshots cannot undo removals.
        """
        if not isinstance(other, ORSet):
            raise TypeError("can only merge with another ORSet")
        for element, tags in other._adds.items():
            self._adds.setdefault(element, set()).update(tags)
        for element, tags in other._removes.items():
            self._removes.setdefault(element, set()).update(tags)
        # Learn about tags this replica may have minted elsewhere (for example
        # on a device restored from an older snapshot) so the next local add
        # stays unique.
        highest = self._counter - 1
        for tags in other._adds.values():
            for tag in tags:
                parsed = _split_tag(tag)
                if parsed is not None and parsed[0] == self._replica_id:
                    highest = max(highest, parsed[1])
        self._counter = max(self._counter, highest + 1)
        return self

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this set."""
        adds = {
            element: sorted(tags)
            for element, tags in sorted(self._adds.items())
        }
        removes = {
            element: sorted(tags)
            for element, tags in sorted(self._removes.items())
        }
        return {
            "replica_id": self._replica_id,
            "adds": adds,
            "removes": removes,
            "counter": self._counter,
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "ORSet":
        """Restore an ORSet from a :meth:`snapshot`-compatible dict."""
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        if set(snapshot.keys()) != _SNAPSHOT_KEYS:
            raise ValueError(
                "snapshot must contain exactly 'replica_id', 'adds', "
                "'removes' and 'counter'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        adds, tag_owner = cls._validate_tag_map(snapshot["adds"], "adds")

        removes, _ = cls._validate_tag_map(snapshot["removes"], "removes")

        # Causal validity: a tombstone may only reference a tag that was
        # actually observed for the same element.
        for element, tags in removes.items():
            added_here = adds.get(element)
            if added_here is None or any(tag not in added_here for tag in tags):
                raise ValueError(
                    "removes may only tombstone tags present in adds "
                    "for the same element"
                )

        counter = snapshot["counter"]
        if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
            raise ValueError("counter must be a non-negative integer")
        required = 0
        for tag in tag_owner:
            parsed = _split_tag(tag)
            assert parsed is not None
            if parsed[0] == replica_id:
                required = max(required, parsed[1] + 1)
        if counter < required:
            raise ValueError(
                "counter must be greater than every local tag sequence"
            )

        restored = cls(replica_id)
        restored._adds = {element: set(tags) for element, tags in adds.items()}
        restored._removes = {
            element: set(tags) for element, tags in removes.items()
        }
        restored._counter = counter
        return restored

    @staticmethod
    def _validate_tag_map(value: object, name: str) -> tuple[
        dict[str, list[str]], dict[str, str]
    ]:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a dict")
        maps: dict[str, list[str]] = {}
        owner: dict[str, str] = {}
        for element, tags in value.items():
            if not isinstance(element, str) or element == "":
                raise ValueError(f"{name} keys must be non-empty strings")
            if not isinstance(tags, list) or not tags:
                raise ValueError(
                    f"{name} values must be non-empty lists of tag strings"
                )
            element_tags: list[str] = []
            for tag in tags:
                parsed = _split_tag(tag)
                if parsed is None:
                    raise ValueError(f"malformed tag in {name}: {tag!r}")
                if tag in element_tags:
                    raise ValueError(f"duplicate tag in {name}: {tag!r}")
                previous_owner = owner.get(tag)
                if previous_owner is not None and previous_owner != element:
                    raise ValueError(
                        f"tag {tag!r} used for multiple elements in {name}"
                    )
                owner[tag] = element
                element_tags.append(tag)
            maps[element] = element_tags
        return maps, owner

    def __repr__(self) -> str:
        return (
            f"ORSet(replica_id={self._replica_id!r}, "
            f"elements={sorted(self.elements())!r})"
        )
