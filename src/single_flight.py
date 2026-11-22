"""
Single-flight: when several threads ask for the same key at once, one of
them runs the work and the rest wait for its result and share it.

Used for model calls. Twenty users asking the same new question at the same
moment should cost one model call, not twenty.

How it works:
  - A table maps each key that is being worked on right now to a "flight".
  - The first caller for a key finds no flight, creates one and becomes the
    leader. It runs the work.
  - Callers that arrive while the flight exists are followers. They wait on
    it and return the leader's result, or raise the leader's exception.
  - When the work ends, the leader removes the flight, so the next caller
    starts a fresh one. This is not a cache: it remembers nothing once the
    work is done, and it does not remember failures.

The lock guards only the table lookup, so it is held for microseconds.
It is never held while the work runs, so different keys proceed in
parallel and a slow model call blocks no one else.
"""

import threading


class _Flight:
    def __init__(self):
        self.done = threading.Event()
        self.value = None
        self.error = None


class SingleFlight:
    def __init__(self):
        self._lock = threading.Lock()
        self._flights = {}

    def do(self, key, work):
        """Run `work()` once per key at a time. Returns (result, led), where
        `led` is True for the caller that ran the work and False for callers
        that shared its result. If the work raised, every caller raises the
        same exception."""
        with self._lock:
            flight = self._flights.get(key)
            led = flight is None
            if led:
                flight = self._flights[key] = _Flight()

        if led:
            try:
                flight.value = work()
            except BaseException as exc:  # noqa: BLE001 - followers must hear about it
                flight.error = exc
            finally:
                # Remove the flight before waking followers, so a caller
                # arriving after this point starts a fresh one instead of
                # joining a flight that has already ended.
                with self._lock:
                    del self._flights[key]
                flight.done.set()
        else:
            flight.done.wait()

        if flight.error is not None:
            raise flight.error
        return flight.value, led

    def in_progress(self):
        """Number of keys being worked on right now."""
        with self._lock:
            return len(self._flights)
