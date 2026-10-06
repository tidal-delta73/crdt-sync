"""Last-writer-wins register CRDT for offline-writable JSON values.

Each local ``assign`` first advances a logical clock and timestamps the write
with ``(logical_count, replica_id)``. Merging two registers keeps the entry
with the greater timestamp — count first, then the Unicode code point order
of the writing ``replica_id`` to break ties — so delivery is idempotent,
commutative and associative. Regardless of whose write wins, the receiver
also remembers the greatest logical count it has observed, so its next local
assignment is always later than every write seen so far.
"""

from __future__ import annotations

import copy
import math

Timestamp = tuple[int, str]


def _validate_replica_id(replica_id: object) -> str:
    if not isinstance(replica_id, str):
        raise TypeError("replica_id must be a non-empty string")
    if replica_id == "":
        raise ValueError("replica_id must be a non-empty string")
    return replica_id


def _is_json_value(value: object, stack: frozenset[int] = frozenset()) -> bool:
    """Return whether ``value`` is an assignable / JSON-round-trippable value.

    Accepts ``None``, booleans, strings, finite numbers and recursive lists
    and string-keyed dicts thereof. Rejects everything else (including
    non-finite floats and self-referential containers). Non-cyclic sharing of
    a container (e.g. ``[x, x]``) stays acceptable, as for ``json.dumps``.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, dict)):
        marker = id(value)
        if marker in stack:
            return False
        stack = stack | {marker}
        if isinstance(value, list):
            return all(_is_json_value(item, stack) for item in value)
        return all(
            isinstance(key, str) and _is_json_value(item, stack)
            for key, item in value.items()
        )
    return False


def _same_value(left: object, right: object) -> bool:
    """Structurally compare two already-validated stored values.

    Booleans are distinct from numbers (mirroring the JSON ``true``/``false``
    vs number distinction), while ``int`` and ``float`` compare by numeric
    equality. Lists and dicts recurse.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left
        )
    return False


class LWWRegister:
    """A last-writer-wins register identified by a ``replica_id``.

    A fresh instance holds no value: :meth:`has_value` is ``False`` and
    :meth:`value` raises ``LookupError`` until the first :meth:`assign`.
    """

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        # Greatest logical count ever observed (locally or via merge).
        self._clock = 0
        # Winning timestamp and its stored value, or None / unset before any
        # observed write.
        self._timestamp: Timestamp | None = None
        self._value: object = None
        self._has_value = False

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def has_value(self) -> bool:
        """Return whether this register currently carries a written value."""
        return self._has_value

    def value(self) -> object:
        """Return an independent deep copy of the current value.

        Raises ``LookupError`` when no write has been observed yet.
        """
        if not self._has_value:
            raise LookupError("LWWRegister has no value yet")
        return copy.deepcopy(self._value)

    def assign(self, value: object) -> None:
        """Locally assign ``value`` under a fresh, strictly greater timestamp."""
        if not _is_json_value(value):
            raise TypeError(
                "value must be a JSON value: null, bool, str, finite number, "
                "or a recursive list / string-keyed dict of such values"
            )
        self._clock += 1
        self._timestamp = (self._clock, self._replica_id)
        # Deep copy so later caller mutations of the input cannot leak in.
        self._value = copy.deepcopy(value)
        self._has_value = True

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        """Absorb ``other``'s state in place, keeping the winning entry.

        Returns ``self``; ``other`` is never modified. The winner is chosen by
        timestamp: greater count wins, with the Unicode code point order of
        the writing ``replica_id`` breaking ties. Either way, the greatest
        observed logical count is remembered.

        Raises ``ValueError`` — leaving both registers untouched — when the
        very same timestamp is bound to two structurally different values,
        which means two live replicas share a ``replica_id`` and minted
        colliding timestamps.
        """
        if not isinstance(other, LWWRegister):
            raise TypeError("can only merge with another LWWRegister")

        # Resolve the outcome before touching any state so a conflict leaves
        # both registers unchanged.
        adopt_value: object = None
        adopt_timestamp: Timestamp | None = None
        conflict = False
        if other._has_value:
            if not self._has_value:
                adopt_timestamp = other._timestamp
                adopt_value = other._value
            else:
                assert self._timestamp is not None
                assert other._timestamp is not None
                if other._timestamp > self._timestamp:
                    adopt_timestamp = other._timestamp
                    adopt_value = other._value
                elif other._timestamp == self._timestamp:
                    conflict = not _same_value(other._value, self._value)

        if conflict:
            raise ValueError(
                "the same timestamp is bound to structurally different values"
            )

        if other._clock > self._clock:
            self._clock = other._clock
        if adopt_timestamp is not None:
            self._timestamp = adopt_timestamp
            self._value = copy.deepcopy(adopt_value)
            self._has_value = True
        return self

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this register."""
        if not self._has_value:
            entry: object = None
        else:
            assert self._timestamp is not None
            entry = {
                "timestamp": [self._timestamp[0], self._timestamp[1]],
                "value": copy.deepcopy(self._value),
            }
        return {
            "replica_id": self._replica_id,
            "clock": self._clock,
            "entry": entry,
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "LWWRegister":
        """Restore a register from a :meth:`snapshot`-compatible dict."""
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        if set(snapshot.keys()) != {"replica_id", "clock", "entry"}:
            raise ValueError(
                "snapshot must contain exactly 'replica_id', 'clock' and "
                "'entry'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        clock = snapshot["clock"]
        if not isinstance(clock, int) or isinstance(clock, bool):
            raise ValueError("clock must be a non-negative integer")
        if clock < 0:
            raise ValueError("clock must be a non-negative integer")

        raw_entry = snapshot["entry"]
        timestamp: Timestamp | None = None
        value: object = None
        if raw_entry is not None:
            if not isinstance(raw_entry, dict):
                raise ValueError("entry must be null or a timestamped entry")
            if set(raw_entry.keys()) != {"timestamp", "value"}:
                raise ValueError(
                    "entry must contain exactly 'timestamp' and 'value'"
                )

            raw_timestamp = raw_entry["timestamp"]
            if not isinstance(raw_timestamp, list) or len(raw_timestamp) != 2:
                raise ValueError("timestamp must be a [count, origin_id] list")
            count, origin = raw_timestamp
            if not isinstance(count, int) or isinstance(count, bool):
                raise ValueError("timestamp count must be an integer")
            if count < 1:
                raise ValueError("timestamp count must be greater than zero")
            if not isinstance(origin, str) or origin == "":
                raise ValueError("timestamp origin must be a non-empty string")
            if count > clock:
                raise ValueError("timestamp count must not exceed the clock")

            raw_value = raw_entry["value"]
            if not _is_json_value(raw_value):
                raise ValueError("entry value must be a JSON value")

            timestamp = (count, origin)
            value = copy.deepcopy(raw_value)

        restored = cls(replica_id)
        restored._clock = clock
        restored._timestamp = timestamp
        restored._value = value
        restored._has_value = raw_entry is not None
        return restored

    def __repr__(self) -> str:
        if not self._has_value:
            return f"LWWRegister(replica_id={self._replica_id!r}, <no value>)"
        return (
            f"LWWRegister(replica_id={self._replica_id!r}, "
            f"timestamp={self._timestamp!r}, value={self._value!r})"
        )
