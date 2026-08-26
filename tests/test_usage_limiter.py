import boto3
import numpy as np
import pytest
from moto import mock_aws

from cache_router import CacheRouter
from dynamodb_cache_store import DynamoDBCacheStore
from usage_limiter import USAGE_ID_PREFIX, DailyLimitReached, UsageLimiter

TABLE_NAME = "semantic-cache-test"


@pytest.fixture
def table():
    with mock_aws():
        yield boto3.resource("dynamodb", region_name="us-east-1").create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )


class FakeEmbedder:
    dim = 2
    model_name = "fake-embedder"

    def embed(self, text):
        # Distinct vectors per query so nothing accidentally hits.
        seed = sum(ord(c) for c in text)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=2).astype(np.float32)
        return v / np.linalg.norm(v)


def test_counts_start_at_zero_and_increment(table):
    limiter = UsageLimiter(table, limit=5)
    assert limiter.count() == 0
    assert limiter.record() == 1
    assert limiter.record() == 2
    assert limiter.count() == 2
    assert limiter.remaining() == 3


def test_check_raises_once_the_limit_is_reached(table):
    limiter = UsageLimiter(table, limit=2)
    limiter.check()          # 0 used, fine
    limiter.record()
    limiter.check()          # 1 used, still fine
    limiter.record()
    with pytest.raises(DailyLimitReached):
        limiter.check()      # 2 used, done
    assert limiter.remaining() == 0


def test_the_count_persists_across_a_new_limiter_instance(table):
    # The point of counting in DynamoDB rather than memory: a restart must
    # not reset the cap, or anyone able to restart the service could bypass
    # it entirely.
    UsageLimiter(table, limit=5).record()
    UsageLimiter(table, limit=5).record()
    assert UsageLimiter(table, limit=5).count() == 2


def test_each_day_gets_its_own_counter(table):
    limiter = UsageLimiter(table, limit=5)
    limiter.record()
    assert limiter.count() == 1
    # A different day's key is untouched by today's usage.
    assert limiter.count(day="2020-01-01") == 0


def test_the_usage_row_is_invisible_to_the_graph_rebuild(table):
    # The counter shares a table with cache entries. It is written without
    # a vector so all_items() skips it, which is what keeps the cold-start
    # rebuild from trying to insert a counter into the HNSW graph.
    store = DynamoDBCacheStore(TABLE_NAME, table=table)
    store.put("q_0", "a real answer", np.array([1.0, 0.0], dtype=np.float32))
    UsageLimiter(table, limit=5).record()

    ids = [item[0] for item in store.all_items()]

    assert ids == ["q_0"]
    assert not any(i.startswith(USAGE_ID_PREFIX) for i in ids)


def test_wrapped_llm_blocks_the_miss_path_once_the_cap_is_hit(table):
    calls = []

    def llm(query):
        calls.append(query)
        return f"answer to {query}"

    store = DynamoDBCacheStore(TABLE_NAME, table=table)
    limiter = UsageLimiter(table, limit=2)
    router = CacheRouter(
        "linear", embedder=FakeEmbedder(), cache_store=store, llm=limiter.wrap(llm)
    )

    router.route("first question")
    router.route("second question")
    assert len(calls) == 2

    with pytest.raises(DailyLimitReached):
        router.route("third question")

    # The refused call never reached the model, and nothing was cached for
    # it -- a blocked request must cost nothing.
    assert len(calls) == 2
    assert len(router.index) == 2


def test_cache_hits_still_work_after_the_cap_is_reached(table):
    # Hits never touch the LLM, so a spent allowance must not take the
    # cache offline. This is the difference between a spend guard and an
    # outage.
    def llm(query):
        return f"answer to {query}"

    store = DynamoDBCacheStore(TABLE_NAME, table=table)
    limiter = UsageLimiter(table, limit=1)
    router = CacheRouter(
        "linear", embedder=FakeEmbedder(), cache_store=store, llm=limiter.wrap(llm)
    )

    first = router.route("a question")   # spends the single allowed call
    with pytest.raises(DailyLimitReached):
        router.route("a different question")

    repeat = router.route("a question")  # identical, so a hit
    assert repeat.hit is True
    assert repeat.response == first.response
