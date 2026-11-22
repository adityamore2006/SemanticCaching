"""
Wraps any LLM and counts what it does.

"How many times was the model really called?" is the number a cache exists
to shrink, so it is counted here, at the one place a model is called,
rather than inferred from hit/miss totals. The wrapper is the same shape as
the LLM it wraps, so nothing else has to know it is there.
"""

import time

import metrics as m
from llm import LLM, LLMTimeout


class MeteredLLM(LLM):
    def __init__(self, inner, metrics):
        self.inner = inner
        self.metrics = metrics

    def __call__(self, query: str) -> str:
        self.metrics.incr(m.MODEL_CALLS)
        started = time.perf_counter()
        try:
            return self.inner(query)
        except LLMTimeout:
            self.metrics.incr(m.MODEL_TIMEOUTS)
            raise
        except Exception:
            self.metrics.incr(m.MODEL_ERRORS)
            raise
        finally:
            self.metrics.observe("model", (time.perf_counter() - started) * 1000)
