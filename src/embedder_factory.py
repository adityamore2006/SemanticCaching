"""
Embedder factory: the single place that knows which concrete embedding
backend exists. Same role factory.py plays for indexes.

One deliberate difference from factory.py: that one imports both index
classes at module level, this one imports the concrete backend only when
it is asked for, so a caller only pays for the backend it uses (importing
local_embedder pulls in sentence-transformers and torch).
"""

from embedder import Embedder


def _load_local():
    from local_embedder import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder


# name -> a loader that imports and returns the class. Add new backends
# here and nowhere else.
_REGISTRY = {
    "local": _load_local,
}


def create_embedder(kind: str, **params) -> Embedder:
    """
    Build an embedder by name.

    kind:   "local" (sentence-transformers, what the eval numbers were
            measured on).
    params: backend-specific knobs (model_name).
    """
    try:
        loader = _REGISTRY[kind]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown embedder kind {kind!r}; available: {available}")
    return loader()(**params)


def available_embedders():
    """Names accepted by create_embedder, for CLIs / config validation."""
    return sorted(_REGISTRY)
