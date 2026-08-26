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

from fastapi import FastAPI
from pydantic import BaseModel

from cache_router import CacheRouter, OPERATING_THRESHOLD
from local_embedder import SentenceTransformerEmbedder

# Configuration is read here, at the entry point, rather than inside the
# components, so the components stay usable from the eval harness and
# local scripts with no environment set up.
CACHE_TABLE_NAME = os.environ.get("CACHE_TABLE_NAME")
SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", "/var/lib/semantic-cache/graph.pkl")
THRESHOLD = float(os.environ.get("OPERATING_THRESHOLD", OPERATING_THRESHOLD))
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID")

# Set at startup. Module-level because a single long-lived process is the
# whole point: this survives every request, unlike a Lambda container.
state = {"router": None, "hits": 0, "misses": 0, "started_at": None, "boot_seconds": None}


class QueryRequest(BaseModel):
    query: str


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


@app.post("/query")
def query(request: QueryRequest):
    router = state["router"]
    started = time.time()
    result = router.route(request.query)
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
    return {
        "entries": len(state["router"].index),
        "requests": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total else None,
        "threshold": THRESHOLD,
        "boot_seconds": state["boot_seconds"],
        "uptime_seconds": round(time.time() - state["started_at"], 1),
    }
