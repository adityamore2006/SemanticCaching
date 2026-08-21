"""
Text embedding wrapper (Phase 2).

Turns text into vectors the VectorIndex implementations can store and
search. This is a core system component, not a one-off eval helper: the
eval harness uses it to embed the paraphrase/near-miss test set, and the
Phase 5 cache router will use the exact same class to embed real incoming
queries. One embedding implementation, multiple consumers, same pattern as
the index contract having multiple implementations behind it.

Wrapped rather than called inline because loading the model is the
expensive part (reads weights into memory once); embedding a sentence with
an already-loaded model is fast. Wrapping it means callers never have to
think about that lifecycle, they just build one Embedder and reuse it.
"""

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Loads a sentence-transformers model once and embeds text with it."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str):
        """Embed a single string, returns a (dim,) numpy array."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts):
        """Embed a list of strings in one batched call, faster than looping embed()."""
        return self.model.encode(texts, convert_to_numpy=True)


if __name__ == "__main__":
    # Small runnable check: two paraphrases should score high, an unrelated
    # pair should score low. No test set needed to sanity-check the wiring.
    embedder = Embedder()
    print(f"model: {embedder.model_name}, dim: {embedder.dim}")

    a = embedder.embed("How do I reset my password?")
    b = embedder.embed("I forgot my password, how do I get back into my account?")
    c = embedder.embed("Does this integrate with Slack?")

    import numpy as np

    def cosine(x, y):
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    print("paraphrase similarity:", cosine(a, b))
    print("unrelated similarity:", cosine(a, c))
