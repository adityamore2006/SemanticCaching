"""
Cache routing -- Phase 5. The actual decision logic the whole project
exists to build: ties together the embedder (Phase 2), the vector index
(Phase 1/3, hot-swappable via factory.py), the cache store (this phase),
and the locked operating threshold (0.80, see knowledge/learned.md
section 11) into one hit/miss decision.

Flow, matching the project brief exactly:
  1. Embed the incoming query.
  2. Search the index for the nearest previously-seen query.
  3. If a match exists and its similarity >= threshold: HIT. Return the
     stored response, no LLM call.
  4. Otherwise: MISS. Call the LLM, insert the new query's vector into the
     index and its response into the cache store, return the fresh
     response.

The LLM call is stubbed (call_llm below) on purpose. Phase 5's job is
proving the routing logic itself is correct -- cheap and fast to iterate
on. Swapping in real Bedrock is Phase 6's deliberate, separate step (see
knowledge/learned.md section 16 for why that sequencing matters): pass a
different callable as `llm=` when that step happens, nothing else here
changes.

index_kind has no default on purpose -- HNSW only wins on speed once the
cache has warmed up past a few thousand entries (knowledge/learned.md
section 17), so which index a router uses should be a stated choice at
construction time, not a silently assumed one.
"""

from dataclasses import dataclass
from typing import Callable, Hashable, Optional

from cache_store import CacheStore, InMemoryCacheStore
from embedding import Embedder
from factory import create_index

OPERATING_THRESHOLD = 0.80  # locked in Phase 2, see knowledge/learned.md section 11


def call_llm(query: str) -> str:
    """Phase 5 stub -- deterministic, free, fast to test. Real Bedrock
    wiring is Phase 6, deliberately kept out of this file until then."""
    return f"[stub response for: {query}]"


@dataclass
class RouteResult:
    response: str
    hit: bool
    matched_id: Optional[Hashable] = None
    # Similarity of the best match found, if the index wasn't empty --
    # None only when this was the very first query. On a miss with a
    # non-None similarity, this is how close the closest existing entry
    # was to clearing the threshold.
    similarity: Optional[float] = None


class CacheRouter:
    def __init__(
        self,
        index_kind: str,
        embedder: Embedder = None,
        cache_store: CacheStore = None,
        threshold: float = OPERATING_THRESHOLD,
        llm: Callable[[str], str] = call_llm,
        **index_params,
    ):
        self.embedder = embedder or Embedder()
        self.index = create_index(index_kind, dim=self.embedder.dim, **index_params)
        self.cache_store = cache_store or InMemoryCacheStore()
        self.threshold = threshold
        self.llm = llm
        self._next_id = 0

    def route(self, query: str) -> RouteResult:
        vector = self.embedder.embed(query)

        matched_id = None
        similarity = None
        if len(self.index) > 0:
            matched_id, similarity = self.index.search(vector, k=1)[0]
            if similarity >= self.threshold:
                return RouteResult(
                    response=self.cache_store.get(matched_id),
                    hit=True,
                    matched_id=matched_id,
                    similarity=similarity,
                )

        # Miss: either the index was empty, or the best match didn't
        # clear the threshold.
        response = self.llm(query)
        new_id = f"q_{self._next_id}"
        self._next_id += 1
        self.index.insert(new_id, vector)
        self.cache_store.put(new_id, response)
        return RouteResult(response=response, hit=False, matched_id=matched_id, similarity=similarity)


if __name__ == "__main__":
    # Runnable end-to-end demo with the real embedding model (tests use a
    # fake one for speed) -- a genuine miss, a paraphrase that should hit,
    # and an unrelated query that should miss again.
    router = CacheRouter("hnsw", threshold=OPERATING_THRESHOLD)

    for query in [
        "How do I reset my password?",
        "I forgot my password, how do I get back into my account?",
        "Does this integrate with Slack?",
    ]:
        result = router.route(query)
        sim = f"{result.similarity:.4f}" if result.similarity is not None else "n/a"
        print(f"{'HIT ' if result.hit else 'MISS'}  sim={sim}  {query!r}")
        print(f"      -> {result.response}")
