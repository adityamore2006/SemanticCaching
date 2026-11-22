"""
HTTP entry point.

A long-lived process: the router (embedder, index, store, LLM) is built once
at startup and stays in memory. Entries persist in a SQLite file, and the
HNSW index is rebuilt from it at startup.

Run it:
    cd src && ../.venv/bin/python -m uvicorn api:app --port 8000

Configuration (environment variables, all optional):
    CACHE_DB_PATH        SQLite file            default data/cache.db
    LLM_BACKEND          openai_compat | fake   default openai_compat
    LLM_MODEL            model name             default qwen2.5:7b-instruct
    LLM_BASE_URL         API base URL           default http://localhost:11434/v1 (Ollama)
    LLM_API_KEY          bearer token, if the server wants one
    OPERATING_THRESHOLD  similarity threshold   default 0.80
    RESET_TOKEN          enables POST /admin/reset (404 if unset)
    SEED_ON_EMPTY        false to start with an empty cache   default true
"""

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import metrics as m
from cache_router import CacheRouter, EmptyLLMResponse, OPERATING_THRESHOLD
from llm import LLMError, LLMTimeout
from llm_factory import create_llm
from local_embedder import SentenceTransformerEmbedder
from metered_llm import MeteredLLM
from metrics import Metrics
from sqlite_cache_store import SqliteCacheStore

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def parse_bool(raw, name):
    """Strict boolean parse for an env var. Anything outside the known
    spellings is an error, so a typo stops the boot instead of silently
    meaning the opposite of what was intended."""
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean; use one of "
        "true/false, 1/0, yes/no, on/off"
    )


def env_bool(name, default):
    """Read a boolean env var, falling back to `default` when unset."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return parse_bool(raw, name)


# Configuration is read here, at the entry point, rather than inside the
# components, so the components stay usable from the eval harness and
# local scripts with no environment set up.
THRESHOLD = float(os.environ.get("OPERATING_THRESHOLD", OPERATING_THRESHOLD))
CACHE_DB_PATH = os.environ.get(
    "CACHE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "cache.db"),
)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openai_compat")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY")

# Enables POST /admin/reset. Unset means the endpoint does not exist at
# all, which is the right default for an API reachable from anywhere:
# forgetting to configure it fails closed rather than leaving an
# unauthenticated cache-wipe endpoint exposed.
RESET_TOKEN = os.environ.get("RESET_TOKEN")

# Populate the curated starter corpus when the cache comes up empty, so a
# fresh deployment can demonstrate a hit immediately instead of having to
# be warmed by hand. Set SEED_ON_EMPTY=false to bring it up genuinely cold.
SEED_ON_EMPTY = env_bool("SEED_ON_EMPTY", default=True)
SEED_ANSWERS_PATH = os.environ.get(
    "SEED_ANSWERS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "seed_answers.json"),
)

# Set at startup. Module-level because a single long-lived process is the
# whole point: this survives every request.
state = {
    "router": None, "metrics": None,
    "started_at": None, "boot_seconds": None,
}


class QueryRequest(BaseModel):
    # Bounded on purpose. Every query that misses becomes a permanent cache
    # entry, so an empty or enormous one is not just a bad request, it is
    # persistent pollution. 2000 characters is far past any real support
    # question.
    query: str = Field(..., min_length=1, max_length=2000)


def _build_llm():
    if LLM_BACKEND == "fake":
        return create_llm("fake")
    return create_llm(
        LLM_BACKEND, model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY
    )


def build_router(store=None, embedder=None, metrics=None):
    """Construct the router: open the store, rebuild the index from it, and
    seed the curated corpus if the cache is empty. When `metrics` is given,
    every model call is counted."""
    embedder = embedder or SentenceTransformerEmbedder()
    if store is None:
        store = SqliteCacheStore(CACHE_DB_PATH, embedder_id=embedder.model_name)

    llm = _build_llm()
    if metrics is not None:
        llm = MeteredLLM(llm, metrics)
    router = CacheRouter(
        "hnsw", embedder=embedder, llm=llm, cache_store=store, threshold=THRESHOLD
    )
    restored = router.restore((id, vector) for id, vector, _ in store.all_items())

    seeded = 0
    if SEED_ON_EMPTY and len(router.index) == 0:
        seeded = router.seed(load_seed_answers())
    return router, f"restored {restored}, seeded {seeded}"


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
    # A fresh Metrics per startup: the numbers describe the current run
    # alongside uptime_seconds. Only observable when the app starts twice
    # in one process (tests do; a deployed service gets a fresh module).
    state["metrics"] = Metrics()
    router, source = build_router(metrics=state["metrics"])
    state["router"] = router
    state["started_at"] = time.time()
    state["boot_seconds"] = round(time.time() - started, 2)
    print(
        f"ready in {state['boot_seconds']}s  "
        f"{source}  entries={len(router.index)}  threshold={THRESHOLD}"
    )

    yield

    state["router"].cache_store.close()


app = FastAPI(title="Semantic Cache", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    """The demo page, served by the same process as the API."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.post("/admin/reset")
def reset(x_reset_token: str = Header(default=None)):
    """Restore the cache to exactly the curated corpus.

    Disabled entirely (404) unless RESET_TOKEN is set, so forgetting to
    configure it fails closed rather than leaving an open cache-wipe
    endpoint.
    """
    if not RESET_TOKEN:
        raise HTTPException(status_code=404, detail="Not Found")
    if x_reset_token != RESET_TOKEN:
        raise HTTPException(status_code=401, detail="bad or missing X-Reset-Token")

    store = state["router"].cache_store
    removed = store.clear()
    router, source = build_router(store, state["router"].embedder, state["metrics"])
    state["router"] = router
    state["metrics"].reset()

    print(f"reset: removed {removed}, restored {len(router.index)} ({source})")
    return {"removed": removed, "restored": len(router.index), "source": source}


async def _stamp_arrival(request: Request):
    """Record when the request arrived. Async on purpose: it runs on the
    event loop immediately, before the sync handler is queued for a worker
    thread, so (handler start - arrival) is exactly the time spent waiting
    for a free thread."""
    request.state.arrived = time.perf_counter()


@app.post("/query", dependencies=[Depends(_stamp_arrival)])
def query(request: QueryRequest, http_request: Request):
    router = state["router"]
    metrics = state["metrics"]
    arrived = http_request.state.arrived

    metrics.observe("queue_wait", (time.perf_counter() - arrived) * 1000)
    metrics.incr(m.REQUESTS)

    outcome = "error"
    try:
        with metrics.in_flight():
            result = router.route(request.query)
        outcome = "hit" if result.hit else "coalesced" if result.coalesced else "miss"
    except LLMTimeout as exc:
        metrics.incr(m.ERRORS_504)
        raise HTTPException(status_code=504, detail=f"{exc}; nothing was cached")
    except (LLMError, EmptyLLMResponse) as exc:
        metrics.incr(m.ERRORS_503)
        raise HTTPException(status_code=503, detail=f"{exc}; nothing was cached")
    except Exception:
        metrics.incr(m.ERRORS_500)
        raise
    finally:
        # Measured from arrival, so it includes any wait for a worker
        # thread: this is the latency a client actually experiences.
        latency_ms = (time.perf_counter() - arrived) * 1000
        metrics.observe(outcome, latency_ms)

    metrics.incr({"hit": m.HITS, "coalesced": m.COALESCED, "miss": m.MISSES}[outcome])

    return {
        "response": result.response,
        "hit": result.hit,
        "coalesced": result.coalesced,
        "similarity": result.similarity,
        "matched_id": result.matched_id,
        "latency_ms": round(latency_ms, 2),
    }


# async on purpose: these run on the event loop, not in the worker-thread
# pool, so they still answer when every worker thread is busy with a slow
# model call. A health check that queues behind the overload it is
# supposed to report is no use.
@app.get("/health")
async def health():
    return {"ok": state["router"] is not None}


@app.get("/metrics")
async def metrics_endpoint():
    """Counters, in-flight gauge and recent latency percentiles as JSON."""
    snapshot = state["metrics"].snapshot()
    snapshot["entries"] = len(state["router"].index)
    return snapshot


@app.get("/stats")
def stats():
    """What the cache is actually doing, for the demo. Every hit is an
    LLM call that provably did not happen, which is the project's
    cost-savings claim made countable rather than estimated. The fuller
    operational picture is at /metrics."""
    metrics = state["metrics"]
    hits, misses, coalesced = metrics.get(m.HITS), metrics.get(m.MISSES), metrics.get(m.COALESCED)
    total = hits + misses + coalesced
    return {
        "entries": len(state["router"].index),
        "requests": total,
        "hits": hits,
        "misses": misses,
        "coalesced": coalesced,
        "hit_rate": round(hits / total, 4) if total else None,
        "threshold": THRESHOLD,
        "boot_seconds": state["boot_seconds"],
        "uptime_seconds": round(time.time() - state["started_at"], 1),
    }
