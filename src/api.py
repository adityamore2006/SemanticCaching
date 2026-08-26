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

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cache_router import CacheRouter, EmptyLLMResponse, OPERATING_THRESHOLD
from local_embedder import SentenceTransformerEmbedder
from usage_limiter import DailyLimitReached, UsageLimiter

# Configuration is read here, at the entry point, rather than inside the
# components, so the components stay usable from the eval harness and
# local scripts with no environment set up.
CACHE_TABLE_NAME = os.environ.get("CACHE_TABLE_NAME")
SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", "/var/lib/semantic-cache/graph.pkl")
THRESHOLD = float(os.environ.get("OPERATING_THRESHOLD", OPERATING_THRESHOLD))
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID")

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


def _build_llm():
    """The real Bedrock call when configured; otherwise the Phase 5 stub,
    so the cache is fully demonstrable without Bedrock access. Returns
    None to mean 'use CacheRouter's default stub'."""
    if not LLM_MODEL_ID:
        return None
    from bedrock_llm import BedrockLLM
    return BedrockLLM(model=LLM_MODEL_ID)


def build_router():
    """Construct the router and populate its index, preferring the local
    snapshot and falling back to the durable store."""
    embedder = SentenceTransformerEmbedder()
    store, durable = _build_store()
    llm = _build_llm()

    kwargs = {"cache_store": store, "threshold": THRESHOLD}
    if llm is not None:
        # Only meaningful with a durable store, since the counter lives in
        # the same table. Without one there is no real LLM configured
        # either, so there is nothing to cap.
        if durable:
            limiter = UsageLimiter(store.table)
            state["limiter"] = limiter
            llm = limiter.wrap(llm)
        kwargs["llm"] = llm
    router = CacheRouter("hnsw", embedder=embedder, **kwargs)

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

    return router, source


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.time()
    router, source = build_router()
    state["router"] = router
    state["started_at"] = time.time()
    state["boot_seconds"] = round(time.time() - started, 2)
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
