"""
Index factory: the single place that knows which concrete index exists.

Everything downstream (cache router, eval harness) asks for an index by
name and gets back something typed as VectorIndex. This is the hot-swap
point: flip "linear" to "hnsw" in one config value and the rest of the
system is unchanged. It also keeps the eval harness honest, it can run the
exact same code path against both indexes to compute recall.
"""

from typing import Dict, Type

from vector_index import VectorIndex
from linear_search import LinearIndex
from hnsw import HNSWIndex

# name -> implementation. Add new indexes here and nowhere else.
_REGISTRY: Dict[str, Type[VectorIndex]] = {
    "linear": LinearIndex,
    "hnsw": HNSWIndex,
}


def create_index(kind: str, *, dim: int, **params) -> VectorIndex:
    """
    Build an index by name.

    kind:   "linear" (Phase 1, exact) or "hnsw" (Phase 3, approximate).
    dim:    vector dimensionality, fixed for the life of the index.
    params: implementation-specific knobs (e.g. HNSW's M / ef_*), ignored
            by implementations that don't take them.
    """
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown index kind {kind!r}; available: {available}")
    return cls(dim=dim, **params)


def available_indexes():
    """Names accepted by create_index, for CLIs / config validation."""
    return sorted(_REGISTRY)
