"""
Local (sentence-transformers) embedding backend -- the one every phase
before 6 used, and still the one the eval harness runs against.

Wrapped rather than called inline because loading the model is the
expensive part (reads weights into memory once); embedding a sentence with
an already-loaded model is fast. Wrapping it means callers never have to
think about that lifecycle, they just build one embedder and reuse it.

Was src/embedding.py's `Embedder` through Phase 5; renamed when Phase 6
added a second backend (BedrockEmbedder) and `Embedder` became the
abstract contract both implement.
"""

from sentence_transformers import SentenceTransformer

from embedder import Embedder

# The model every locked number in this project was measured on: the
# four-model comparison (knowledge/learned.md section 8) and the 0.80
# operating threshold (section 11). Deliberately NOT all-MiniLM-L6-v2,
# which was the default through Phase 5 and silently mismatched the
# threshold anything constructing this with no arguments was using.
DEFAULT_MODEL = "all-mpnet-base-v2"


class SentenceTransformerEmbedder(Embedder):
    """Loads a sentence-transformers model once and embeds text with it."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)


if __name__ == "__main__":
    # Small runnable check: two paraphrases should score high, an unrelated
    # pair should score low. No test set needed to sanity-check the wiring.
    embedder = SentenceTransformerEmbedder()
    print(f"model: {embedder.model_name}, dim: {embedder.dim}")

    a = embedder.embed("How do I reset my password?")
    b = embedder.embed("I forgot my password, how do I get back into my account?")
    c = embedder.embed("Does this integrate with Slack?")

    import numpy as np

    def cosine(x, y):
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    print("paraphrase similarity:", cosine(a, b))
    print("unrelated similarity:", cosine(a, c))
