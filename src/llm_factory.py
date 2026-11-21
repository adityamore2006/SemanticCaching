"""
LLM factory: the single place that knows which LLM backends exist.
Same role as factory.py (indexes) and embedder_factory.py (embedders).
"""

from fake_llm import FakeLLM
from llm import LLM
from openai_compat_llm import OpenAICompatLLM

# name -> class. Add new backends here and nowhere else.
_REGISTRY = {
    "openai_compat": OpenAICompatLLM,  # Ollama by default
    "fake": FakeLLM,
}


def create_llm(kind: str, **params) -> LLM:
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown llm kind {kind!r}; available: {available}")
    return cls(**params)
