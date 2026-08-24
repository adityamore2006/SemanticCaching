"""
Embedder factory: the single place that knows which concrete embedding
backend exists. Same role factory.py plays for indexes.

One deliberate difference from factory.py, worth explaining because it
looks like inconsistency otherwise: that one imports both index classes at
module level, this one imports the concrete backend only when it's asked
for. The reason is a real deployment constraint, not style. Importing
local_embedder pulls in sentence-transformers and torch, which are not
installed in the Lambda package (that's the whole reason BedrockEmbedder
exists, see bedrock_embedder.py). An eager import would make this module
un-importable in the deployed environment; a lazy one means each caller
only pays for the backend it actually asked for.
"""

from embedder import Embedder


def _load_local():
    from local_embedder import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder


def _load_bedrock():
    from bedrock_embedder import BedrockEmbedder
    return BedrockEmbedder


# name -> a loader that imports and returns the class. Add new backends
# here and nowhere else.
_REGISTRY = {
    "local": _load_local,
    "bedrock": _load_bedrock,
}


def create_embedder(kind: str, **params) -> Embedder:
    """
    Build an embedder by name.

    kind:   "local" (sentence-transformers, what the eval numbers were
            measured on) or "bedrock" (Titan V2, what the Lambda deploys).
    params: backend-specific knobs (model_name, and for bedrock also
            dimensions / region_name / max_workers).
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
