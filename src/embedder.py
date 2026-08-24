"""
The embedder contract.

Third instance of the same seam already used for VectorIndex (linear vs
hnsw) and CacheStore (in-memory vs DynamoDB): callers depend on this
interface, never on a concrete embedding backend, so swapping backends is
a change at one factory call (see embedder_factory.py) rather than at
every call site.

Phase 6 is what forced this abstraction to exist. Until now there was
exactly one implementation (sentence-transformers, running locally), so
an interface would have been decoration. Deploying to Lambda added a
second: sentence-transformers plus torch is ~2GB, far past a zip-packaged
Lambda's limit, so the deployed system embeds via Bedrock instead.

The critical thing this interface does NOT promise, and the reason it's
worth stating here rather than discovering later: two implementations of
this contract do not produce comparable vectors. Different models mean
different dimensionality and a different vector space entirely, so an
index built with one embedder is meaningless to another, and the
operating threshold (knowledge/learned.md section 11) has to be re-derived
per model rather than carried over. This was already documented as a known
limit in section 9 before Phase 6 made it a live concern.
"""

from abc import ABC, abstractmethod
from typing import List, Sequence


class Embedder(ABC):
    """Turns text into fixed-dimensionality vectors."""

    # Set by implementations at construction: dim is the vector length the
    # paired VectorIndex must be built with, model_name identifies which
    # vector space those vectors live in (and therefore which threshold
    # applies).
    dim: int
    model_name: str

    @abstractmethod
    def embed(self, text: str) -> Sequence[float]:
        """Embed a single string, returns a (dim,) vector."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> List[Sequence[float]]:
        """Embed many strings at once. Implementations should make this
        faster than looping embed(); callers may assume it's worth using."""
