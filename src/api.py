"""
HTTP entry point -- the deployed system.

Replaces the Lambda handler this phase started with. The reason is
measured, not stylistic: Lambda recycles idle containers on its own
schedule, and this system's cold start is unusually expensive (load an
embedding model, then rebuild an HNSW graph held in RAM). That meant an
unpredictable multi-second stall on an arbitrary request, which is exactly
wrong for a cache whose entire pitch is latency. Keeping a Lambda warm
costs more than the small instance this now runs on. Full reasoning in
knowledge/learned.md section 23.

What changes as a result: this is a long-lived process, so the router is
built once at startup and simply stays there. The "rebuild on every cold
start" logic becomes ordinary startup work, paid once per boot on your
schedule rather than repeatedly on AWS's.

Startup order is a deliberate fallback chain:
  1. Load the graph from a local snapshot on the instance's disk (fast).
  2. If there isn't a usable one, rebuild it from DynamoDB (slower, but
     always correct -- DynamoDB is the source of truth, the snapshot is
     only ever an optimization).
Shutdown writes a fresh snapshot so the next boot takes path 1.

Run it:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cache_router import (
    CacheRouter,
    EmptyLLMResponse,
    LLMNotConfigured,
    OPERATING_THRESHOLD,
)
from local_embedder import SentenceTransformerEmbedder
from usage_limiter import DailyLimitReached, UsageLimiter

# Configuration is read here, at the entry point, rather than inside the
# components, so the components stay usable from the eval harness and
# local scripts with no environment set up.
CACHE_TABLE_NAME = os.environ.get("CACHE_TABLE_NAME")
SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", "/var/lib/semantic-cache/graph.pkl")
THRESHOLD = float(os.environ.get("OPERATING_THRESHOLD", OPERATING_THRESHOLD))
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID")

# Enables POST /admin/reset. Unset means the endpoint does not exist at
# all, which is the right default for an API reachable from anywhere:
# forgetting to configure it fails closed rather than leaving an
# unauthenticated cache-wipe endpoint exposed.
RESET_TOKEN = os.environ.get("RESET_TOKEN")

# Populate the curated starter corpus when the cache comes up empty, so a
# fresh deployment can demonstrate a hit immediately instead of having to
# be warmed by hand. Set to 0 to bring it up genuinely cold.
SEED_ON_EMPTY = os.environ.get("SEED_ON_EMPTY", "1") != "0"
SEED_ANSWERS_PATH = os.environ.get(
    "SEED_ANSWERS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "seed_answers.json"),
)

# Set at startup. Module-level because a single long-lived process is the
# whole point: this survives every request, unlike a Lambda container.
state = {
    "router": None, "hits": 0, "misses": 0,
    "started_at": None, "boot_seconds": None, "limiter": None,
}


class QueryRequest(BaseModel):
    # Bounded on purpose. Every query that misses becomes a permanent cache
    # entry, so an empty or enormous one is not just a bad request, it is
    # persistent pollution. 2000 characters is far past any real support
    # question and well inside DynamoDB's 400KB item limit alongside a
    # 768-float vector.
    query: str = Field(..., min_length=1, max_length=2000)


def _build_store():
    """DynamoDB when a table is configured, in-memory otherwise so the app
    can be run locally with no AWS account at all."""
    if not CACHE_TABLE_NAME:
        from cache_store import InMemoryCacheStore
        return InMemoryCacheStore(), False
    from dynamodb_cache_store import DynamoDBCacheStore
    return DynamoDBCacheStore(CACHE_TABLE_NAME), True


def _refuse(query: str) -> str:
    """Stands in for the model when none is configured.

    Refuses rather than returning placeholder text. Returning a stub here
    is what made a miss cache "[stub response for: ...]" and then serve it
    back as a confident HIT on the next identical question, filling the
    cache with answers that were never answered.

    The cache is still fully demonstrable without a model: the seeded
    corpus serves real hits, and near-misses are still correctly refused.
    The only thing that stops working is answering something genuinely new,
    which is exactly the operation that needs a model.
    """
    raise LLMNotConfigured(
        "no model is configured, so this question cannot be answered or cached; "
        "questions already in the cache still work"
    )


def _build_llm():
    """The real Bedrock call when configured, otherwise a callable that
    refuses. Never a placeholder."""
    if not LLM_MODEL_ID:
        return _refuse
    from bedrock_llm import BedrockLLM
    return BedrockLLM(model=LLM_MODEL_ID)


def build_router():
    """Construct the router and populate its index, preferring the local
    snapshot and falling back to the durable store."""
    embedder = SentenceTransformerEmbedder()
    store, durable = _build_store()
    llm = _build_llm()

    # Only cap a real model. Wrapping the refusing placeholder would count
    # calls that never happen and burn the day's allowance on requests that
    # cost nothing.
    if durable and LLM_MODEL_ID:
        limiter = UsageLimiter(store.table)
        state["limiter"] = limiter
        llm = limiter.wrap(llm)

    router = CacheRouter(
        "hnsw", embedder=embedder, llm=llm, cache_store=store, threshold=THRESHOLD
    )

    # The snapshot only holds the index. Responses live in the store, so
    # restoring one without the other leaves ids in the graph that have no
    # answer behind them. With a durable store both come back together;
    # with the in-memory store the responses are gone, so loading a
    # snapshot would guarantee that mismatch. CacheRouter.route() recovers
    # from it either way, but there's no reason to manufacture it.
    if durable and router.load_snapshot(SNAPSHOT_PATH):
        source = "snapshot"
    elif durable:
        router.restore(
            (entry_id, vector) for entry_id, vector, _ in store.all_items()
        )
        source = "dynamodb"
    else:
        source = "empty (no CACHE_TABLE_NAME, nothing persists)"

    # A cache that starts empty misses on its first query by definition, so
    # the behaviour worth demonstrating never happens on a fresh deployment.
    # Only seeds when the index is genuinely empty, so this runs once on a
    # new table and never touches a warm one.
    if SEED_ON_EMPTY and len(router.index) == 0:
        pairs = load_seed_answers()
        if pairs:
            added = router.seed(pairs)
            source = f"{source} + seeded {added}"

    return router, source


def load_seed_answers(path=SEED_ANSWERS_PATH):
    """(question, answer) pairs from the curated starter corpus, or [] if
    the file is missing. Absence is not an error: the service runs fine
    against an empty cache, it just starts cold."""
    try:
        with open(path) as f:
            answers = json.load(f)["answers"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"no seed corpus loaded from {path}: {exc}")
        return []
    return list(answers.items())


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.time()
    router, source = build_router()
    state["router"] = router
    state["started_at"] = time.time()
    state["boot_seconds"] = round(time.time() - started, 2)
    # These describe the current run alongside uptime_seconds, so startup
    # zeroes them. Only observable when the app starts twice in one
    # process (tests do; a deployed service gets a fresh module each time).
    state["hits"] = state["misses"] = 0
    print(
        f"ready in {state['boot_seconds']}s  "
        f"restored_from={source}  entries={len(router.index)}  threshold={THRESHOLD}"
    )

    yield

    # Best effort: a failed snapshot costs the next boot some time, it
    # never costs correctness, since DynamoDB still holds everything.
    try:
        os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
        router.save_snapshot(SNAPSHOT_PATH)
        print(f"snapshot written to {SNAPSHOT_PATH} ({len(router.index)} entries)")
    except OSError as exc:
        print(f"snapshot failed, next boot will rebuild from the store: {exc}")


app = FastAPI(title="Semantic Cache", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    """The demo page, served by the same process as the API.

    Deliberately not S3 + CloudFront. The page is useless without this
    backend, so splitting them across two origins buys CORS configuration
    and a page that loads but errors whenever the instance is stopped,
    which is most of the time by design. It also adds a service that
    demonstrates nothing the rest of the stack does not already cover.
    Same reasoning that ruled out OpenSearch Serverless and Lambda
    (knowledge/learned.md sections 16, 19, 23): a service has to earn its
    place.
    """
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.post("/admin/reset")
def reset(x_reset_token: str = Header(default=None)):
    """Restore the cache to exactly the curated corpus.

    Lives inside the service because the in-memory index is the thing that
    actually needs clearing, and only this process holds it. Purging the
    durable store from outside leaves the running index untouched, so the
    cache keeps serving entries that no longer exist in storage.

    Three steps, and the order matters:
      1. Clear the store (reserved rows, like the usage counter, survive).
      2. Delete the snapshot. Skipping this is the subtle failure: the
         reset looks like it worked, then the next restart reloads the old
         graph from disk and silently undoes all of it.
      3. Rebuild the router and re-seed from the canonical corpus.

    Disabled entirely unless RESET_TOKEN is set. This API is deliberately
    reachable from anywhere, and an unauthenticated endpoint that wipes the
    cache is not something to leave on by accident, so the default is off
    rather than open.
    """
    if not RESET_TOKEN:
        # 404 rather than 403: an endpoint that isn't enabled shouldn't
        # advertise that it exists.
        raise HTTPException(status_code=404, detail="Not Found")
    if x_reset_token != RESET_TOKEN:
        raise HTTPException(status_code=401, detail="bad or missing X-Reset-Token")

    removed = state["router"].cache_store.clear()

    snapshot_removed = False
    try:
        os.remove(SNAPSHOT_PATH)
        snapshot_removed = True
    except OSError:
        pass  # absent is the desired end state either way

    router, source = build_router()
    state["router"] = router
    state["hits"] = state["misses"] = 0

    print(f"reset: removed {removed}, restored {len(router.index)} ({source})")
    return {
        "removed": removed,
        "restored": len(router.index),
        "snapshot_deleted": snapshot_removed,
        "source": source,
    }


@app.post("/query")
def query(request: QueryRequest):
    router = state["router"]
    started = time.time()

    try:
        result = router.route(request.query)
    except DailyLimitReached as exc:
        # 429, not 503: this is a deliberate policy decision, not a
        # malfunction, and the caller should know the difference. Cache
        # hits keep working while this is in effect, since they never
        # touch the LLM.
        raise HTTPException(status_code=429, detail=str(exc))
    except LLMNotConfigured as exc:
        # 501, not 503: the service is healthy and cached questions still
        # work. What is missing is a capability, not availability.
        raise HTTPException(status_code=501, detail=str(exc))
    except EmptyLLMResponse as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        # Anything else from the miss path is upstream: throttling, auth,
        # a network failure. Return a useful message rather than a stack
        # trace, and log the real error where it can be found.
        #
        # Nothing was cached in any of these branches, which is the point.
        # route() writes only after the LLM returns something usable, so a
        # failure leaves the cache exactly as it was.
        print(f"route failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"upstream model call failed ({type(exc).__name__}); nothing was cached",
        )

    latency_ms = (time.time() - started) * 1000

    state["hits" if result.hit else "misses"] += 1

    return {
        "response": result.response,
        "hit": result.hit,
        "similarity": result.similarity,
        "matched_id": result.matched_id,
        "latency_ms": round(latency_ms, 2),
    }


@app.get("/health")
def health():
    return {"ok": state["router"] is not None}


@app.get("/stats")
def stats():
    """What the cache is actually doing, for the demo. Every hit is a
    Bedrock call that provably did not happen, which is the project's
    cost-savings claim made countable rather than estimated."""
    hits, misses = state["hits"], state["misses"]
    total = hits + misses
    limiter = state["limiter"]
    return {
        "llm_calls_remaining_today": limiter.remaining() if limiter else None,
        "daily_llm_limit": limiter.limit if limiter else None,
        "entries": len(state["router"].index),
        "requests": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total else None,
        "threshold": THRESHOLD,
        "boot_seconds": state["boot_seconds"],
        "uptime_seconds": round(time.time() - state["started_at"], 1),
    }
