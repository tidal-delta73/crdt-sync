"""Last-writer-wins register (LWW-Register) CRDT.

Each local ``assign`` first advances a per-replica logical counter and stamps
the new value with a ``(counter, replica_id)`` timestamp. A merge adopts the
entry with the larger counter, breaking ties by the Unicode ordering of the
writer's ``replica_id``; regardless of which entry wins, the receiver also
adopts the largest logical counter either side has observed, so its next
local write is guaranteed to be newer than every write observed so far.

Values are JSON-shaped: ``None``, booleans, strings, finite numbers, and
recursive lists / string-keyed dicts thereof. Stored values and the values
returned by :meth:`value` are always detached copies, so callers can mutate
neither the argument nor the result into the register.
"""

from __future__ import annotations

import json
import math

Timestamp = tuple[int, str]


def _validate_replica_id(replica_id: object) -> str:
    if not isinstance(replica_id, str):
        raise TypeError("replica_id must be a non-empty string")
    if replica_id == "":
        raise ValueError("replica_id must be a non-empty string")
    return replica_id


def _clone_json_value(value: object, error_type: type[Exception]) -> object:
    """Validate a JSON-shaped value and return a detached copy of it.

    Booleans are accepted in their own right even though ``bool`` subclasses
    ``int``; floats must be finite; object keys must be strings. Raises
    ``error_type`` (TypeError for user input, ValueError for snapshots) on
    anything outside that shape. Because the copy is built fresh and only
    returned when fully valid, a raised call aliases nothing and mutates
    nothing the caller still holds.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type("numbers must be finite")
        return value
    if isinstance(value, list):
        return [_clone_json_value(item, error_type) for item in value]
    if isinstance(value, dict):
        cloned: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise error_type("object keys must be strings")
            cloned[key] = _clone_json_value(item, error_type)
        return cloned
    raise error_type(
        "value must be null, a boolean, string, finite number, "
        "or a recursive list / string-keyed dict of such"
    )


def _same_json_value(left: object, right: object) -> bool:
    """Structural JSON equality (``True`` differs from ``1``, ``1`` from 1.0).

    Both arguments are already-validated stored values, so serializing them
    can only fail on a bug in this module; comparing the canonical JSON forms
    makes the type distinction JSON itself draws explicit.
    """
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


class LWWRegister:
    """A last-writer-wins register identified by a non-empty ``replica_id``.

    A fresh instance holds no value: :meth:`has_value` reports ``False`` and
    :meth:`value` raises ``LookupError`` until the first :meth:`assign`.
    """

    def __init__(self, replica_id: str) -> None:
        self._replica_id = _validate_replica_id(replica_id)
        self._clock = 0
        self._entry: tuple[Timestamp, object] | None = None

    @property
    def replica_id(self) -> str:
        return self._replica_id

    def has_value(self) -> bool:
        """Return whether this register currently holds a value."""
        return self._entry is not None

    def value(self) -> object:
        """Return an independent deep copy of the current value.

        Raises ``LookupError`` before the register has ever been assigned.
        """
        if self._entry is None:
            raise LookupError("register has no value yet")
        return _clone_json_value(self._entry[1], ValueError)

    def assign(self, value: object) -> None:
        """Stamp and store ``value`` under a fresh local timestamp.

        Accepts null, booleans, strings, finite numbers and recursive lists /
        string-keyed dicts thereof; anything else raises ``TypeError`` and
        leaves the register (including its logical clock) untouched. The
        logical counter is advanced only after the value is fully validated,
        and a detached copy is stored.
        """
        stored = _clone_json_value(value, TypeError)
        self._clock += 1
        self._entry = ((self._clock, self._replica_id), stored)

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        """Absorb ``other``'s state in place.

        Returns ``self``; ``other`` is never modified. The held entry is the
        one with the larger ``(counter, replica_id)`` timestamp — counter
        first, then the Unicode ordering of the writer id. Whatever the
        outcome, the largest logical counter observed on either side is
        adopted, so the next local :meth:`assign` postdates every observed
        write.

        Raises ``ValueError`` — leaving both states untouched — when equal
        timestamps carry structurally different values: two live replicas
        share a ``replica_id`` and minted colliding timestamps, so no winner
        exists.
        """
        if not isinstance(other, LWWRegister):
            raise TypeError("can only merge with another LWWRegister")

        # Detect the equal-timestamp conflict before mutating anything, so a
        # rejected merge cannot advance the clock on either side.
        if self._entry is not None and other._entry is not None:
            self_timestamp, self_value = self._entry
            other_timestamp, other_value = other._entry
            if self_timestamp == other_timestamp and not _same_json_value(
                self_value, other_value
            ):
                raise ValueError(
                    "the same timestamp carries structurally different values"
                )

        self._clock = max(self._clock, other._clock)
        if other._entry is not None:
            other_timestamp, other_value = other._entry
            if self._entry is None or other_timestamp > self._entry[0]:
                # Detach the absorbed value so later caller-side mutation on
                # either side cannot leak across.
                self._entry = (
                    other_timestamp,
                    _clone_json_value(other_value, ValueError),
                )
        return self

    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-serializable snapshot of this register.

        An unassigned register carries ``entry: null``; an assigned one
        carries ``{"timestamp": [count, origin], "value": value}``.
        """
        entry: object = None
        if self._entry is not None:
            (count, origin), value = self._entry
            entry = {
                "timestamp": [count, origin],
                "value": _clone_json_value(value, ValueError),
            }
        return {
            "replica_id": self._replica_id,
            "clock": self._clock,
            "entry": entry,
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "LWWRegister":
        """Restore a register from a :meth:`snapshot`-compatible dict.

        Non-dict input raises ``TypeError``; every structural problem —
        missing or extra fields, a bad id, a negative or boolean clock, a
        malformed entry, a timestamp count above the clock, or a non-JSON
        value — raises ``ValueError``. The result copies every container out
        of the input, so mutating ``snapshot`` afterwards cannot reach it.
        """
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")

        if set(snapshot.keys()) != {"replica_id", "clock", "entry"}:
            raise ValueError(
                "snapshot must contain exactly 'replica_id', 'clock' "
                "and 'entry'"
            )

        replica_id = snapshot["replica_id"]
        if not isinstance(replica_id, str) or replica_id == "":
            raise ValueError("replica_id must be a non-empty string")

        clock = snapshot["clock"]
        if not isinstance(clock, int) or isinstance(clock, bool) or clock < 0:
            raise ValueError("clock must be a non-negative integer")

        raw_entry = snapshot["entry"]
        entry: tuple[Timestamp, object] | None = None
        if raw_entry is not None:
            if not isinstance(raw_entry, dict):
                raise ValueError("entry must be null or an object")
            if set(raw_entry.keys()) != {"timestamp", "value"}:
                raise ValueError(
                    "entry must contain exactly "
                    "'timestamp' and 'value'"
                )

            raw_timestamp = raw_entry["timestamp"]
            if (
                not isinstance(raw_timestamp, list)
                or len(raw_timestamp) != 2
            ):
                raise ValueError("timestamp must be a [count, origin] list")
            count, origin = raw_timestamp
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
            ):
                raise ValueError("timestamp count must be a positive integer")
            if not isinstance(origin, str) or origin == "":
                raise ValueError(
                    "timestamp origin must be a non-empty string"
                )
            if count > clock:
                raise ValueError("timestamp count must not exceed clock")

            # Rebuilding the value both validates its JSON shape and detaches
            # it from the caller's containers.
            value = _clone_json_value(raw_entry["value"], ValueError)
            entry = ((count, origin), value)

        register = cls(replica_id)
        register._clock = clock
        register._entry = entry
        return register

    def __repr__(self) -> str:
        if self._entry is None:
            return (
                f"LWWRegister(replica_id={self._replica_id!r}, <no value>)"
            )
        return (
            f"LWWRegister(replica_id={self._replica_id!r}, "
            f"value={self._entry[1]!r})"
        )
