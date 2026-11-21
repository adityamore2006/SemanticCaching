"""
The embedder contract: text in, fixed-length vector out.

Same seam as VectorIndex, CacheStore and LLM: callers depend on this
interface, never on a concrete backend (see embedder_factory.py).

What this interface does NOT promise: two embedders do not produce
comparable vectors. A different model means a different dimension and a
different vector space, so an index built with one is meaningless to
another, and the operating threshold (knowledge/learned.md section 11) has
to be re-derived per model, not carried over (section 9).
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
