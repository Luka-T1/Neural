"""Deterministic spike event queue (heapq)."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

from .interfaces import SpikeEvent


@dataclass(frozen=True)
class Delivery:
    arrival_ms: float
    target: str
    event: SpikeEvent


class EventQueue:
    def __init__(self) -> None:
        self._h: list = []
        self._seq = 0

    def schedule(self, event: SpikeEvent, delay_ms: float, target: str) -> None:
        if delay_ms < 0:
            raise ValueError("delay_ms must be >= 0")
        d = Delivery(event.t_ms + delay_ms, target, event)
        heapq.heappush(self._h, (d.arrival_ms, target, event.source_id, event.index, self._seq, d))
        self._seq += 1

    def pop_due(self, t_end_ms: float) -> list[Delivery]:
        out = []
        while self._h and self._h[0][0] < t_end_ms:
            out.append(heapq.heappop(self._h)[-1])
        return out

    def peek_time(self) -> float | None:
        return self._h[0][0] if self._h else None

    def __len__(self) -> int:
        return len(self._h)
