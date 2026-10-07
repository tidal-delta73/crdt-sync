"""Observed-remove set (ORSet) CRDT for removable string sets.

Every local ``add`` mints a unique tag ``(replica_id, counter)`` attached to
the added element. An element is visible as long as it owns at least one tag
that has not been observed by a ``remove``: ``remove(e)`` tombstones exactly
the tags visible on this replica at call time. Both the per-element tag map
and the tombstone set only ever grow, and merging takes their union, so
delivery is idempotent, commutative and associative, stale or duplicated
snapshots can never resurrect a removed add, and a fresh ``add`` after a
remove is never swallowed by the historical tombstone.

``compact()`` bounds the otherwise-unbounded history: per tag origin it
retires the longest prefix ``1..n`` whose adds are all observed *and* all
tombstoned, dropping those add records and per-tag tombstones in favour of a
single retired-up-to bound. The bound is a causal summary, not a mutation of
the observed-remove semantics: elements and their visibility are unchanged,
merging takes the per-origin maximum of the bounds, and any add or tombstone
at or below a known bound arriving (late or duplicated) is treated as
already-consumed history rather than redelivered state.
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
        # Origin -> greatest sequence whose add (and per-tag tombstone) has
        # been retired into this causal summary. Only prefixes 1..bound are
        # ever retired, so a tag at or below the bound is known history.
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
        """Retire fully-deleted tag-history prefixes into causal summaries.

        For each tag origin, find the greatest ``n`` such that every add
        ``(origin, 1)`` .. ``(origin, n)`` is present here *and* covered by a
        tombstone. Stop the prefix at the first gap, still-visible tag, or
        known-but-unreached sequence. Remove those adds and their per-tag
        tombstones, recording ``n`` as the origin's retired-up-to bound
        (merged with the bound already known).

        Visible state never changes: only adds that were already tombstoned
        are retired. Returns the number of explicit add records removed by
        this call; calling again immediately returns ``0``.
        """
        # Per origin: the set of observed add sequences (ownership is
        # irrelevant to prefix eligibility, every sequence is counted once).
        present: dict[str, set[int]] = {}
        for tags in self._adds.values():
            for origin, sequence in tags:
                present.setdefault(origin, set()).add(sequence)

        retired = 0
        # Origins with history to consider: those carrying adds here or
        # already known through a bound (so a bound can only ever advance).
        origins = set(present) | set(self._compacted)
        for origin in origins:
            old_bound = self._compacted.get(origin, 0)
            sequences = present.get(origin, set())
            bound = old_bound
            candidate = old_bound + 1
            while candidate in sequences and (origin, candidate) in self._removes:
                bound = candidate
                candidate += 1
            if bound == old_bound:
                continue

            # Drop the retired prefix's adds ...
            def retired_tag(element_tags: set[Tag]) -> set[Tag]:
                return {
                    tag
                    for tag in element_tags
                    if tag[0] == origin and old_bound < tag[1] <= bound
                }

            for element in list(self._adds):
                stale = retired_tag(self._adds[element])
                if stale:
                    self._adds[element] -= stale
                    retired += len(stale)
                    if not self._adds[element]:
                        del self._adds[element]
            # ... and the per-tag tombstones the bound now stands in for.
            self._removes -= {
                (origin, sequence)
                for sequence in range(old_bound + 1, bound + 1)
            }
            self._compacted[origin] = bound
        return retired

    def merge(self, other: "ORSet") -> "ORSet":
        """Union the tag map and tombstone set with ``other``, in place.

        Returns ``self``; ``other`` is never modified. Repeated or reordered
        merges converge to the same state. Retired-prefix bounds join by
        per-origin maximum; adds or tombstones at or below a joined bound are
        already-consumed history and are skipped, so a late or duplicated
        snapshot carrying only the pre-compaction history can never resurrect
        a retired add.

        Raises ``ValueError`` — leaving both states untouched — when the
        same tag is bound to different elements on the two sides. Such a
        conflict means two live replicas share a ``replica_id`` and minted
        colliding tags; unioning them would make the tag's ownership
        ambiguous and the merged state unrestorable.
        """
        if not isinstance(other, ORSet):
            raise TypeError("can only merge with another ORSet")

        # Join the retired bounds first; a bound consumes every add and
        # tombstone at or below it on both sides during this very merge.
        joined_bounds = dict(self._compacted)
        for origin, bound in other._compacted.items():
            if bound > joined_bounds.get(origin, 0):
                joined_bounds[origin] = bound

        def is_consumed(tag: Tag) -> bool:
            origin, sequence = tag
            return sequence <= joined_bounds.get(origin, 0)

        # A tag identifies one add of one element; before unioning, verify
        # against the full, un-consumed add history of both sides that no tag
        # changes ownership across the two states. Retired prefixes have no
        # owner anywhere by construction, so they cannot participate. The
        # check is total, so its outcome never depends on dict iteration
        # order.
        ownership: dict[Tag, str] = {}
        for element, tags in self._adds.items():
            for tag in tags:
                if not is_consumed(tag):
                    ownership[tag] = element
        conflicts: set[Tag] = set()
        for element, tags in other._adds.items():
            for tag in tags:
                if is_consumed(tag):
                    continue
                owner = ownership.get(tag)
                if owner is not None and owner != element:
                    conflicts.add(tag)
        if conflicts:
            raise ValueError(
                "the same tag is bound to different elements: "
                + ", ".join(repr(tag) for tag in sorted(conflicts))
            )

        for element, tags in other._adds.items():
            live = {tag for tag in tags if not is_consumed(tag)}
            if live:
                self._adds.setdefault(element, set()).update(live)
        for origin, sequence in other._removes:
            if sequence > joined_bounds.get(origin, 0):
                self._removes.add((origin, sequence))
        # A newly learned bound may retire history this replica still held
        # explicitly (adds and per-tag tombstones); the summary replaces it.
        for element in list(self._adds):
            remaining = {tag for tag in self._adds[element] if not is_consumed(tag)}
            if remaining:
                self._adds[element] = remaining
            else:
                del self._adds[element]
        self._removes = {tag for tag in self._removes if not is_consumed(tag)}
        self._compacted = joined_bounds

        # A state carrying our own replica id (e.g. a restored backup of this
        # replica) may know about more of our own adds; advance the counter
        # past every observed tag of ours — including retired ones, whose
        # existence is witnessed by the bound — so future tags stay unique.
        own_bound = self._compacted.get(self._replica_id, 0)
        if own_bound > self._counter:
            self._counter = own_bound
        for tags in self._adds.values():
            for origin, sequence in tags:
                if origin == self._replica_id and sequence > self._counter:
                    self._counter = sequence
        return self

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this set.

        The snapshot keeps the four-field shape until any compaction has
        happened; once a retired-prefix summary exists it additionally
        carries ``compacted`` mapping each origin to its positive retired
        sequence bound, with keys in sorted order.
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

        Accepts both the original four-field shape and the compacted shape,
        which differs only by an additional ``compacted`` mapping of origins
        to positive retired sequence bounds.
        """
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        keys = set(snapshot.keys())
        allowed = {"replica_id", "counter", "adds", "removes"}
        if keys != allowed and keys != allowed | {"compacted"}:
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
        if "compacted" in snapshot:
            raw_compacted = snapshot["compacted"]
            if not isinstance(raw_compacted, dict):
                raise ValueError("compacted must be a dict")
            for origin, bound in raw_compacted.items():
                if not isinstance(origin, str) or origin == "":
                    raise ValueError("compacted keys must be non-empty strings")
                if not isinstance(bound, int) or isinstance(bound, bool):
                    raise ValueError("compacted bounds must be positive integers")
                if bound <= 0:
                    raise ValueError("compacted bounds must be positive integers")
                compacted[origin] = bound

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
                # Retired prefixes live only in the summary: an explicit tag
                # at or below its origin's bound is stale, duplicated history.
                if tag[1] <= compacted.get(tag[0], 0):
                    raise ValueError(
                        "adds contains a tag already retired by its compacted bound"
                    )
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
            tag = _parse_tag(raw_tag)
            # Likewise, a per-tag tombstone the bound already subsumes must
            # not be carried explicitly.
            if tag[1] <= compacted.get(tag[0], 0):
                raise ValueError(
                    "removes contains a tag already retired by its compacted bound"
                )
            removes.add(tag)

        # Causal validity: a tombstone may only refer to an observed add; a
        # retired prefix counts as known history, so tombstones above a bound
        # must still match an explicit add.
        if not removes <= set(tag_owner):
            raise ValueError("removes references a tag with no matching add")
        # ...and the local counter must describe exactly this replica's own
        # observed add history (1..counter, no gaps, no unknown future tags),
        # treating the retired prefix as already-known history.
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
        expected = list(range(own_bound + 1, counter + 1))
        if own_sequences != expected:
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
