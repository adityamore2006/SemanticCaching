"""
Lambda entry point -- the deployed system.

The problem this file exists to solve: CacheRouter holds its HNSW graph in
plain Python memory, which is correct for a long-running local process and
wrong for Lambda, where containers are stateless between invocations and
recycled without warning. Deploying the Phase 5 code unchanged would lose
the entire cache on every cold start, so every query would miss forever
and the cache would never actually be a cache.

The fix (decided in knowledge/learned.md section 19, chosen over
serializing the graph to S3 and over paying for provisioned concurrency):
persist the vectors in DynamoDB, and on cold start replay them through
the existing insert() to rebuild the graph. No serialization format, no
new index code -- the rebuild is just the insert path that's already
tested.

This is only affordable because of an earlier, separate decision: capping
the realistic cache size at 1k-10k entries (section 16). Rebuilding a
10,000-node graph takes ~16.5s, which is a real cold-start cost, paid once
per container rather than per request. That number is the honest tradeoff
of this design, not a footnote.
"""

import json
import os
import time

import bedrock_embedder
import bedrock_llm
from bedrock_embedder import BedrockEmbedder
from bedrock_llm import BedrockLLM
from cache_router import CacheRouter
from dynamodb_cache_store import DynamoDBCacheStore

# All deployment configuration is read here, in the entry point, rather
# than inside the components. That keeps BedrockEmbedder / BedrockLLM
# usable from the eval harness and local scripts without any environment
# set up, and keeps "what this deployment is configured with" answerable
# by reading one file.
#
# The threshold is NOT the 0.80 derived in Phase 2 unless it happens to
# coincide -- that number belongs to all-mpnet-base-v2, and this deploys
# against Titan, a different vector space entirely (knowledge/learned.md
# section 9). It comes from re-running the same threshold sweep on Titan.
OPERATING_THRESHOLD = float(os.environ.get("OPERATING_THRESHOLD", "0.80"))
TABLE_NAME = os.environ["CACHE_TABLE_NAME"]
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", bedrock_embedder.DEFAULT_MODEL)
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", bedrock_llm.DEFAULT_MODEL)

# Survives across invocations for as long as AWS keeps this container
# alive. Rebuilt only when it doesn't (a cold start).
_router = None


def _build_router():
    """Construct the router and replay every persisted vector into its
    index. Returns the router plus how long the rebuild took, so the cost
    this whole design trades against is measurable rather than assumed."""
    started = time.time()

    store = DynamoDBCacheStore(TABLE_NAME)
    router = CacheRouter(
        # Always hnsw. LinearIndex stays what it has always been in this
        # project: the exact ground-truth answer key HNSW is graded
        # against locally, never a deployed component (section 19).
        "hnsw",
        embedder=BedrockEmbedder(model_name=EMBEDDING_MODEL_ID),
        cache_store=store,
        threshold=OPERATING_THRESHOLD,
        llm=BedrockLLM(model=LLM_MODEL_ID),
    )

    restored = router.restore(
        (entry_id, vector) for entry_id, vector, _response in store.all_items()
    )

    return router, restored, time.time() - started


def handler(event, context):
    global _router

    if _router is None:
        _router, restored, rebuild_seconds = _build_router()
        print(json.dumps({
            "event": "cold_start",
            "restored_entries": restored,
            "rebuild_seconds": round(rebuild_seconds, 3),
        }))

    query = _extract_query(event)
    if not query:
        return _response(400, {"error": "missing 'query' in request body"})

    started = time.time()
    result = _router.route(query)
    latency_ms = (time.time() - started) * 1000

    # One structured line per request. CloudWatch Logs Insights can turn
    # these into the hit rate and latency the dashboard reports, without
    # needing a metrics library in the request path.
    print(json.dumps({
        "event": "route",
        "hit": result.hit,
        "similarity": result.similarity,
        "latency_ms": round(latency_ms, 2),
        "cache_size": len(_router.index),
    }))

    return _response(200, {
        "response": result.response,
        "hit": result.hit,
        "similarity": result.similarity,
        "matched_id": result.matched_id,
        "latency_ms": round(latency_ms, 2),
    })


def _extract_query(event):
    body = event.get("body")
    if not body:
        return None
    try:
        return json.loads(body).get("query")
    except (json.JSONDecodeError, AttributeError):
        return None


def _response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }
