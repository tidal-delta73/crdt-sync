"""Shared deterministic helpers for the GCounter test-suite.

Tests only touch the public surface: ``GCounter``, ``replica_id``,
``increment``, ``value``, ``merge``, ``snapshot`` and ``from_snapshot``.
No internal attributes are read or relied upon.
"""

from __future__ import annotations

import copy

from crdt_sync import GCounter


def replica_history(replica_id: str, amounts: list[int]):
    """Drive a fresh replica through ``amounts`` one increment at a time.

    Returns ``(counter, history)`` where ``history`` holds independent deep
    copies of the snapshot observed after each increment, oldest first. These
    let tests deliver genuine intermediate states (including stale ones).
    """
    counter = GCounter(replica_id)
    history: list[dict] = []
    for amount in amounts:
        counter.increment(amount)
        history.append(copy.deepcopy(counter.snapshot()))
    return counter, history


def merge_states(receiver: GCounter, states) -> GCounter:
    """Merge a sequence of snapshot dicts into ``receiver`` via public API."""
    for state in states:
        receiver.merge(GCounter.from_snapshot(copy.deepcopy(state)))
    return receiver


def counts_of(snapshot: dict) -> dict:
    return snapshot["counts"]
