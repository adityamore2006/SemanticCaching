"""
The LLM contract: the model that writes an answer when the cache misses.

Same seam as VectorIndex, Embedder and CacheStore: the router depends on
this interface, never on a provider. Switching providers is a change in
llm_factory.py, not in the router.

This is a different model from the Embedder. The embedder runs on every
request and decides hit or miss, so it is tied to the similarity threshold.
The LLM only runs on a miss, and the threshold does not depend on it.

An LLM is just a callable: query in, answer text out. Keeping it that
plain is deliberate, so a test can pass a lambda.

Backends raise LLMError (or LLMTimeout) on any failure and never return a
placeholder. A cache stores whatever it is given and replays it forever, so
failing loudly is correct; an invented answer would become permanent. They
do not retry either, retries are a policy that belongs above the backend.
"""

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """The model call failed: unreachable, bad response, truncated answer."""


class LLMTimeout(LLMError):
    """The model did not answer within the backend's timeout."""


class LLM(ABC):
    @abstractmethod
    def __call__(self, query: str) -> str:
        """Return the answer text for `query`, or raise LLMError."""
