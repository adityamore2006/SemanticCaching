"""
A concurrency limit that refuses instead of waiting.

    limit = ConcurrencyLimit(24)
    with limit.slot():      # raises Overloaded at once if 24 are already inside
        ...

Why refuse rather than queue: the server has a fixed pool of worker threads
(40 by default) shared by every request. A request that waits for a slot is
still holding one of those threads, so waiting would spend the very capacity
the limit exists to protect. Refusing frees the thread within milliseconds.

The lock guards a counter and is held for microseconds.
"""

import threading
from contextlib import contextmanager
from typing import Optional


class Overloaded(RuntimeError):
    """Too many requests are already in the protected section."""


class ConcurrencyLimit:
    def __init__(self, limit: Optional[int]):
        """`limit` of None means unlimited. Anything below 1 is a mistake:
        it would refuse every request."""
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be at least 1 or None, got {limit}")
        self.limit = limit
        self._lock = threading.Lock()
        self._in_use = 0

    @contextmanager
    def slot(self):
        with self._lock:
            if self.limit is not None and self._in_use >= self.limit:
                raise Overloaded(f"{self._in_use} of {self.limit} slots in use")
            self._in_use += 1
        try:
            yield
        finally:
            with self._lock:
                self._in_use -= 1

    def in_use(self) -> int:
        with self._lock:
            return self._in_use
