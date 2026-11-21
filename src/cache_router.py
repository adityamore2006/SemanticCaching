"""
Cache routing: the hit/miss decision the whole project builds toward.

Flow:
  1. Embed the incoming query.
  2. Search the index for the nearest previously-seen query.
  3. If the best match's similarity >= threshold: HIT. Return the stored
     response, no LLM call.
  4. Otherwise: MISS. Call the LLM, store the response and insert the
     query's vector into the index, return the fresh response.

`llm`, `embedder` and `index_kind` have no defaults on purpose. A silent
default is how this project once ran one embedding model against another
model's threshold (knowledge/learned.md sections 9 and 24), and how a stub
LLM once had its placeholder answer cached and served as a confident hit.
The threshold (0.80) was calibrated on all-mpnet-base-v2 (section 11); a
different embedder needs its own threshold.
"""

from dataclasses import dataclass
from typing import Callable, Hashable, Optional

from cache_store import CacheStore, InMemoryCacheStore
from embedder import Embedder
from factory import create_index

OPERATING_THRESHOLD = 0.80

# Entry ids are "q_0", "q_1", ... Defined once because route() mints them
# and restore() reads the counter back out of persisted ones.
ID_PREFIX = "q_"

# Provenance stored with every entry: the curated starter corpus, or an
# answer a model actually produced.
SOURCE_SEED = "seed"
SOURCE_LLM = "llm"


class EmptyLLMResponse(RuntimeError):
    """The LLM returned nothing usable. Raised instead of caching, because a
    cached empty answer is served as a confident HIT to every paraphrase."""


@dataclass
class RouteResult:
    response: str
    hit: bool
    matched_id: Optional[Hashable] = None
    # Similarity of the closest existing entry, or None if the index was
    # empty. On a miss, how close the nearest entry came to the threshold.
    similarity: Optional[float] = None


class CacheRouter:
    def __init__(
        self,
        index_kind: str,
        embedder: Embedder,
        llm: Callable[[str], str],
        cache_store: CacheStore = None,
        threshold: float = OPERATING_THRESHOLD,
        **index_params,
    ):
        self.embedder = embedder
        self.index = create_index(index_kind, dim=embedder.dim, **index_params)
        # `is None`, not `or`: stores define __len__, so an empty store is
        # falsy and `or` would silently discard it.
        self.cache_store = InMemoryCacheStore() if cache_store is None else cache_store
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
                cached = self.cache_store.get(matched_id)
                if cached is None:
                    # _add writes the store before the index, and restore
                    # builds the index from the store, so a matched id with
                    # no stored response means something outside those paths
                    # changed the data. Serving None would be a confident
                    # wrong answer, and papering over it hides the bug, so
                    # fail loudly.
                    raise RuntimeError(
                        f"index has {matched_id!r} but the store has no response "
                        "for it; the cache is inconsistent and needs a reset"
                    )
                return RouteResult(cached, hit=True, matched_id=matched_id, similarity=similarity)

        response = self.llm(query)

        # Validate before writing: an empty answer would be stored and then
        # served as a permanent HIT.
        if not response or not str(response).strip():
            raise EmptyLLMResponse("llm returned an empty response; refusing to cache it")

        self._add(query_vector=vector, response=response, source=SOURCE_LLM)
        return RouteResult(response, hit=False, matched_id=matched_id, similarity=similarity)

    def seed(self, pairs) -> int:
        """Populate the cache from (question, answer) pairs without calling
        the LLM. Skips a question that already matches an entry above the
        threshold, so re-running is safe. Returns how many were added."""
        added = 0
        for question, answer in pairs:
            if not answer or not str(answer).strip():
                raise EmptyLLMResponse(f"seed answer for {question!r} is empty; refusing to cache it")

            vector = self.embedder.embed(question)
            if len(self.index) > 0:
                _, similarity = self.index.search(vector, k=1)[0]
                if similarity >= self.threshold:
                    continue
            self._add(query_vector=vector, response=answer, source=SOURCE_SEED)
            added += 1
        return added

    def restore(self, entries) -> int:
        """Rebuild the index from persisted (id, vector) pairs, returning how
        many were restored. Also resumes the id counter past the highest
        persisted id; otherwise a restarted router would reissue ids that
        already exist and overwrite earlier entries."""
        restored = 0
        highest = -1
        for entry_id, vector in entries:
            self.index.insert(entry_id, vector)
            restored += 1
            if isinstance(entry_id, str) and entry_id.startswith(ID_PREFIX):
                suffix = entry_id[len(ID_PREFIX):]
                if suffix.isdigit():
                    highest = max(highest, int(suffix))

        self._next_id = max(self._next_id, highest + 1)
        return restored

    def _add(self, query_vector, response, source) -> None:
        """Mint an id, write the store first, then the index. If the write
        fails nothing is half-added; if the process dies in between, the
        row is rebuilt into the index at the next start."""
        new_id = f"{ID_PREFIX}{self._next_id}"
        self._next_id += 1
        self.cache_store.put(new_id, response, query_vector, source=source)
        self.index.insert(new_id, query_vector)
