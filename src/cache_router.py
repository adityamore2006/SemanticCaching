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

`llm` is required and has no default. Through Phase 5 it defaulted to a
stub returning "[stub response for: ...]", which was harmless while nothing
persisted and actively wrong once things did: a miss cached the
placeholder, and re-asking the same question served it back as a confident
HIT. The cache accumulated answers that had never been answered. A miss
with no model available is a request that cannot be fulfilled, and the
router now says so rather than inventing something to store.

index_kind has no default on purpose -- HNSW only wins on speed once the
cache has warmed up past a few thousand entries (knowledge/learned.md
section 17), so which index a router uses should be a stated choice at
construction time, not a silently assumed one.

`embedder` has no default for the same reason, and for a concrete one:
it used to default to sentence-transformers' own default model
(all-MiniLM-L6-v2), which is NOT the model the 0.80 threshold below was
calibrated on (all-mpnet-base-v2, section 11). Every local run that didn't
pass an embedder explicitly was quietly measuring one model against
another model's threshold. Now that a Bedrock backend exists too, with its
own separately-derived threshold, which vector space a router operates in
is far too load-bearing to leave implicit.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Callable, Hashable, Optional

from cache_store import CacheStore, InMemoryCacheStore
from embedder import Embedder
from embedder_factory import create_embedder
from factory import create_index

OPERATING_THRESHOLD = 0.80  # locked in Phase 2, see knowledge/learned.md section 11

# Cache entry ids are "q_0", "q_1", ... Defined once here because two
# places depend on the format: route() minting new ids, and restore()
# reading the counter back out of persisted ones.
ID_PREFIX = "q_"

# Provenance written with every entry. "seed" is the curated starter
# corpus, "llm" is an answer a model actually produced. See cache_store.py
# for why the distinction is worth persisting.
SOURCE_SEED = "seed"
SOURCE_LLM = "llm"

# Bumped whenever the snapshot payload's shape changes, so an old file is
# rejected and rebuilt rather than unpickled into a subtly wrong state.
SNAPSHOT_VERSION = 1


class EmptyLLMResponse(RuntimeError):
    """The LLM callable returned nothing usable.

    Raised instead of caching, because a cached empty answer is served as a
    confident HIT to every future paraphrase.
    """


class LLMNotConfigured(RuntimeError):
    """A miss needs an answer and no model is available to produce one.

    Raised instead of substituting placeholder text. Through Phase 5 the
    default here WAS a stub that returned "[stub response for: ...]", which
    was fine while nothing persisted and wrong the moment things did: a
    miss cached the placeholder, and asking the same question again served
    it back as a confident HIT. The cache filled up with answers that had
    never been answered.

    A miss with no model is a request the system cannot fulfil, and saying
    so is the honest outcome. Cached questions keep working, because a hit
    never needed the model.
    """


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
        embedder: Embedder,
        llm: Callable[[str], str],
        cache_store: CacheStore = None,
        threshold: float = OPERATING_THRESHOLD,
        **index_params,
    ):
        self.embedder = embedder
        self.index = create_index(index_kind, dim=self.embedder.dim, **index_params)
        # `is None`, not `or`. CacheStore implements __len__, so an empty
        # store is falsy -- `cache_store or InMemoryCacheStore()` silently
        # discarded any store that happened to be empty at construction,
        # which is exactly the state a real one is in on first deploy.
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
                if cached is not None:
                    return RouteResult(
                        response=cached,
                        hit=True,
                        matched_id=matched_id,
                        similarity=similarity,
                    )
                # The index and the store disagree: the index holds a
                # vector under this id but the store has no response for
                # it. Returning the cached value regardless would answer a
                # user's question with None at high confidence, which is
                # the project's central failure mode reached through
                # bookkeeping rather than a bad threshold.
                #
                # The two can drift apart for real reasons: the graph is
                # snapshotted to disk while responses live in a separate
                # store, so a snapshot restored beside a store that lost
                # or never had those rows leaves exactly this state. Rather
                # than trust one side, regenerate and heal the entry in
                # place below.

        # Miss: the index was empty, the best match didn't clear the
        # threshold, or it cleared it but had no stored response.
        response = self.llm(query)

        # `llm` is a caller-supplied callable, so the router validates what
        # comes back rather than trusting it. An empty or whitespace-only
        # answer is not a minor quality issue here: it gets written to the
        # store and, because it is not None, passes the orphan check above
        # on every future query, so one bad answer becomes a permanent
        # empty HIT. Same principle as refusing to serve a None found in
        # the store (knowledge/learned.md section 21) -- fail loudly at the
        # boundary instead of persisting something unusable.
        if not response or not str(response).strip():
            raise EmptyLLMResponse(
                "llm returned an empty response; refusing to cache it"
            )

        if matched_id is not None and similarity is not None and similarity >= self.threshold:
            # Heal the orphaned entry under its existing id instead of
            # minting a new one. The index already holds a vector for this
            # neighborhood, so adding a second would leave the orphan
            # behind to fail the same way on every future query.
            self.cache_store.put(matched_id, response, vector, source=SOURCE_LLM)
            return RouteResult(response=response, hit=False, matched_id=matched_id, similarity=similarity)

        new_id = f"{ID_PREFIX}{self._next_id}"
        self._next_id += 1
        self.index.insert(new_id, vector)
        self.cache_store.put(new_id, response, vector, source=SOURCE_LLM)
        return RouteResult(response=response, hit=False, matched_id=matched_id, similarity=similarity)

    def seed(self, pairs):
        """
        Populate the cache from (question, answer) pairs without calling the
        LLM. Returns how many entries were added.

        A cache that starts empty misses on its first query no matter what,
        so the behaviour the system exists to demonstrate never happens on a
        fresh deployment. Seeding a verified starter corpus fixes that and,
        unlike generating the same answers, costs nothing and needs no model
        access.

        Uses the same embed-and-insert path route() does, so a seeded entry
        is identical in every respect to one the router created -- except
        `source`, which records that it was curated rather than generated.
        That single field is what keeps the two tellable apart later.

        Skips a question that already matches something cached above the
        threshold, so re-running is safe and does not create near-duplicate
        entries competing for the same neighborhood.
        """
        added = 0
        for question, answer in pairs:
            if not answer or not str(answer).strip():
                raise EmptyLLMResponse(
                    f"seed answer for {question!r} is empty; refusing to cache it"
                )

            vector = self.embedder.embed(question)
            if len(self.index) > 0:
                _, similarity = self.index.search(vector, k=1)[0]
                if similarity >= self.threshold:
                    continue

            new_id = f"{ID_PREFIX}{self._next_id}"
            self._next_id += 1
            self.index.insert(new_id, vector)
            self.cache_store.put(new_id, answer, vector, source=SOURCE_SEED)
            added += 1
        return added

    def restore(self, entries):
        """
        Rebuild in-memory index state from previously persisted entries,
        returning how many were restored.

        Exists for deployments whose process doesn't outlive the cache:
        Lambda containers are stateless between invocations, so a cold
        start replays every stored (id, vector) through the ordinary
        insert() path rather than deserializing a saved graph
        (knowledge/learned.md section 19).

        Resuming the id counter is the subtle part, and the reason this is
        the router's job rather than the caller's. Ids come from a counter
        that restarts at zero in a fresh process, so without this a
        restored router would hand out ids that already exist in storage
        and silently overwrite earlier cache entries -- a wrong answer
        served with no failure signal, the exact hazard this project is
        built to avoid.
        """
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

    def save_snapshot(self, path) -> None:
        """
        Write the built index and its id counter to a local file.

        Why both, and why in one call: the id counter is not decoration,
        it's an invariant of the graph. A snapshot holding the index alone
        would restore entries q_0..q_5 while the counter restarted at 0,
        reissuing live ids and serving stored answers to unrelated queries
        at similarity 1.0 (knowledge/learned.md section 22). Keeping them
        in one payload makes that desync unrepresentable rather than
        merely discouraged.

        Written to a temp file and renamed, because os.replace is atomic:
        a crash mid-write leaves the previous good snapshot intact instead
        of a truncated file that would fail to load on next boot.
        """
        payload = {
            "version": SNAPSHOT_VERSION,
            # Vectors only mean anything in the space that produced them
            # (section 9), so the snapshot records which model that was
            # and load_snapshot refuses a mismatch.
            "model_name": self.embedder.model_name,
            "dim": self.embedder.dim,
            "index": self.index,
            "next_id": self._next_id,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load_snapshot(self, path) -> bool:
        """
        Restore index + id counter from a snapshot. Returns True if it was
        loaded, False if there was nothing usable.

        Every rejection path returns False rather than raising, because
        the caller always has a correct fallback: rebuilding from the
        durable store. A snapshot is an optimization, never the source of
        truth, so a bad one should cost startup time and nothing else.
        """
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
            return False

        if not isinstance(payload, dict):
            return False
        if payload.get("version") != SNAPSHOT_VERSION:
            return False
        # A snapshot from a different embedding model is not stale, it's
        # meaningless: the stored vectors live in a space this embedder
        # never produces, so every similarity against them would be
        # nonsense. Reject rather than silently serve wrong matches.
        if payload.get("model_name") != self.embedder.model_name:
            return False
        if payload.get("dim") != self.embedder.dim:
            return False

        index = payload.get("index")
        next_id = payload.get("next_id")
        if index is None or not isinstance(next_id, int):
            return False

        self.index = index
        self._next_id = next_id
        return True


def _offline_llm(query: str) -> str:
    """What the local demos pass as `llm`. Refuses rather than inventing an
    answer, so a miss stays a miss instead of poisoning the cache with
    placeholder text."""
    raise LLMNotConfigured(
        "no model configured, so this question cannot be answered or cached"
    )


def _load_seed_answers():
    """The curated starter corpus, so the demos have something to hit."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "data", "seed_answers.json")
    with open(path) as f:
        return list(json.load(f)["answers"].items())


def _run_fixed_demo():
    # Runnable end-to-end demo with the real embedding model (tests use a
    # fake one for speed) -- a genuine miss, a paraphrase that should hit,
    # and an unrelated query that should miss again.
    #
    # The pair below is drawn from the verified eval set rather than
    # invented, and deliberately: at 0.80 on all-mpnet-base-v2 only 22% of
    # real paraphrases clear the threshold (knowledge/learned.md section
    # 11), so a demo pair has to be one that actually does. An earlier
    # version of this demo used a password-reset pair that scores 0.783 on
    # mpnet and therefore misses; it only appeared to pass because the
    # demo was silently running a different model than the threshold was
    # calibrated on (section 24).
    router = CacheRouter(
        "hnsw",
        embedder=create_embedder("local"),
        threshold=OPERATING_THRESHOLD,
        llm=_offline_llm,
    )
    router.seed(_load_seed_answers())

    for query in [
        "Can I merge two accounts into one?",
        "Is it possible to combine my two separate accounts?",
        "Does this integrate with Slack?",
    ]:
        result = router.route(query)
        sim = f"{result.similarity:.4f}" if result.similarity is not None else "n/a"
        print(f"{'HIT ' if result.hit else 'MISS'}  sim={sim}  {query!r}")
        print(f"      -> {result.response}")


def _run_interactive():
    # Type your own queries and watch the router decide hit/miss live,
    # against a cache that accumulates across the session -- try a query,
    # then a paraphrase of it, then something unrelated.
    router = CacheRouter(
        "hnsw",
        embedder=create_embedder("local"),
        threshold=OPERATING_THRESHOLD,
        llm=_offline_llm,
    )
    router.seed(_load_seed_answers())
    print(f"interactive mode, threshold={OPERATING_THRESHOLD}. empty line or ctrl-d to quit.\n")

    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            print()
            break
        if not query:
            break

        result = router.route(query)
        sim = f"{result.similarity:.4f}" if result.similarity is not None else "n/a"
        print(f"  {'HIT ' if result.hit else 'MISS'}  sim={sim}  -> {result.response}")
        print(f"  (cache now has {len(router.index)} entries)\n")


if __name__ == "__main__":
    import sys
    if "--interactive" in sys.argv:
        _run_interactive()
    else:
        _run_fixed_demo()
