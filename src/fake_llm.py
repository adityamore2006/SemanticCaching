"""
A deterministic LLM for tests and load testing. No network, no model.

Counts its calls, because "how many times was the model actually called?"
is the question a cache exists to answer. `latency` simulates a slow
model and `error` simulates a broken one.
"""

import threading
import time
from typing import Optional

from llm import LLM


class FakeLLM(LLM):
    def __init__(self, latency: float = 0.0, error: Optional[Exception] = None):
        self.latency = latency
        self.error = error
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        return self._calls

    def __call__(self, query: str) -> str:
        with self._lock:
            self._calls += 1
        if self.latency:
            time.sleep(self.latency)
        if self.error is not None:
            raise self.error
        return f"fake answer to: {query}"
