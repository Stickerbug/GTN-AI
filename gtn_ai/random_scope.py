from __future__ import annotations

import copy
import random
import threading
from contextlib import contextmanager
from typing import Iterator


_GLOBAL_RUNTIME_LOCK = threading.RLock()


class EngineRuntimeScope:
    """Own RNG and card-instance counters without changing the game server."""

    def __init__(self, seed: int = 0):
        seeded = random.Random(int(seed))
        self._state = seeded.getstate()
        self._instance_counter = 0

    def __deepcopy__(self, memo):
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        clone._state = copy.deepcopy(self._state, memo)
        clone._instance_counter = int(self._instance_counter)
        return clone

    @contextmanager
    def activate(self) -> Iterator[None]:
        import cards

        with _GLOBAL_RUNTIME_LOCK:
            outer_state = random.getstate()
            outer_instance_counter = int(getattr(cards, "_next_instance_id", 0))
            random.setstate(self._state)
            cards._next_instance_id = int(self._instance_counter)
            try:
                yield
            finally:
                self._state = random.getstate()
                self._instance_counter = int(getattr(cards, "_next_instance_id", 0))
                random.setstate(outer_state)
                cards._next_instance_id = outer_instance_counter
