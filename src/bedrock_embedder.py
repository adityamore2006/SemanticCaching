"""
Bedrock embedding backend -- what the deployed Lambda uses.

Why this exists at all, since every measured number in the project so far
came from the local sentence-transformers backend: all-mpnet-base-v2 plus
torch is ~2GB, well past the 250MB unzipped limit for a zip-packaged
Lambda. Bundling it would have forced a container-image Lambda (Docker +
ECR + slower cold starts on top of an already-slow graph rebuild). Calling
a managed embedding model instead keeps the deployment artifact small and
the cold start cheap.

The cost of that choice, stated plainly rather than buried: this is a
different vector space than mpnet's, so the 0.80 operating threshold does
not transfer. It gets re-derived against this model over the same
194-pair eval set (knowledge/learned.md section 9 already flagged that a
threshold is model-bound, section 11 is the original derivation this
mirrors).

The second, subtler cost: with a local model, embedding was free and the
only billed call was the LLM on a miss. Now embedding is a billed network
call on EVERY request, hit or miss, because checking the cache requires
the query's vector before hit/miss is even known. The project's cost claim
is still real, a hit skips the far more expensive generative call, but
"Bedrock is only called on a miss" is no longer literally true and
shouldn't be repeated as-is.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np

from embedder import Embedder

# Amazon's own embedding model, chosen over a third-party one on Bedrock
# so the deployment needs model access granted for one fewer vendor.
DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"

# Titan V2 supports 1024 / 512 / 256. Full width by default -- the index
# is small enough (1k-10k, section 16) that the memory saved by truncating
# isn't worth spending recall on.
DEFAULT_DIMENSIONS = 1024

# Titan's InvokeModel takes one string per call, so a batch is N calls.
# Run them concurrently: the 194-pair eval set is ~250 calls and latency
# here is almost entirely network wait, not local CPU. Modest on purpose,
# high concurrency just trades a slow sweep for Bedrock throttling.
DEFAULT_MAX_WORKERS = 8


class BedrockEmbedder(Embedder):
    """Embeds text via Bedrock's Titan Text Embeddings V2."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        region_name: str = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        client=None,
    ):
        self.model_name = model_name
        self.dim = dimensions
        self.max_workers = max_workers
        # Injectable for tests; in Lambda the default client picks up the
        # execution role's credentials and region automatically.
        if client is None:
            client = boto3.client("bedrock-runtime", region_name=region_name)
        self.client = client

    def embed(self, text):
        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps({
                "inputText": text,
                "dimensions": self.dim,
                # Unit-normalize at the source. The index normalizes on
                # insert anyway, so this is belt-and-braces rather than
                # load-bearing, but it keeps raw dot products meaningful
                # for anything inspecting vectors outside the index.
                "normalize": True,
            }),
        )
        payload = json.loads(response["body"].read())
        return np.array(payload["embedding"], dtype=np.float32)

    def embed_batch(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # ThreadPoolExecutor.map preserves input order, which callers rely
        # on -- the eval harness zips results back against the pair list.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            vectors = list(pool.map(self.embed, texts))
        return np.vstack(vectors)
