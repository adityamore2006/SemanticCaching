"""
Local embedding backend (sentence-transformers), the one every locked number
in this project was measured on.

Loading the model is the expensive part, so build one embedder and reuse it.
"""

from sentence_transformers import SentenceTransformer

from embedder import Embedder

# The model every locked number in this project was measured on: the
# four-model comparison (knowledge/learned.md section 8) and the 0.80
# operating threshold (section 11).
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
