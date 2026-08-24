import boto3
import numpy as np
import pytest
from moto import mock_aws

from cache_store import InMemoryCacheStore
from dynamodb_cache_store import DynamoDBCacheStore

TABLE_NAME = "semantic-cache-test"


@pytest.fixture
def dynamo_store():
    """
    A DynamoDBCacheStore backed by moto's in-process DynamoDB mock, so
    these tests exercise the real boto3 call shapes (including the Decimal
    conversion and Scan pagination) without touching a live table or
    incurring any cost.
    """
    with mock_aws():
        table = boto3.resource("dynamodb", region_name="us-east-1").create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield DynamoDBCacheStore(TABLE_NAME, table=table)


def test_put_then_get_roundtrips():
    store = InMemoryCacheStore()
    store.put("a", "response for a")
    assert store.get("a") == "response for a"


def test_get_missing_id_returns_none():
    store = InMemoryCacheStore()
    assert store.get("nope") is None


def test_put_overwrites_existing_entry():
    store = InMemoryCacheStore()
    store.put("a", "first")
    store.put("a", "second")
    assert store.get("a") == "second"


def test_len_tracks_distinct_ids():
    store = InMemoryCacheStore()
    assert len(store) == 0
    store.put("a", "r1")
    store.put("b", "r2")
    assert len(store) == 2
    store.put("a", "overwritten")
    assert len(store) == 2


def test_in_memory_store_accepts_and_ignores_a_vector():
    # The vector argument exists for persistent stores; the in-memory one
    # must still accept it so both satisfy one contract.
    store = InMemoryCacheStore()
    store.put("a", "response for a", np.array([0.1, 0.2], dtype=np.float32))
    assert store.get("a") == "response for a"


# --- DynamoDBCacheStore: same contract, real persistence ---


def test_dynamo_put_then_get_roundtrips(dynamo_store):
    dynamo_store.put("a", "response for a")
    assert dynamo_store.get("a") == "response for a"


def test_dynamo_get_missing_id_returns_none(dynamo_store):
    assert dynamo_store.get("nope") is None


def test_dynamo_put_overwrites_existing_entry(dynamo_store):
    dynamo_store.put("a", "first")
    dynamo_store.put("a", "second")
    assert dynamo_store.get("a") == "second"


def test_dynamo_len_tracks_distinct_ids(dynamo_store):
    assert len(dynamo_store) == 0
    dynamo_store.put("a", "r1")
    dynamo_store.put("b", "r2")
    assert len(dynamo_store) == 2
    dynamo_store.put("a", "overwritten")
    assert len(dynamo_store) == 2


def test_dynamo_roundtrips_the_vector_through_decimal_conversion(dynamo_store):
    # The cold-start rebuild is only correct if vectors survive storage
    # intact -- DynamoDB has no float type, so put() converts to Decimal
    # and all_items() converts back.
    vector = np.array([0.1, -0.25, 0.7071068], dtype=np.float32)
    dynamo_store.put("a", "response for a", vector)

    items = list(dynamo_store.all_items())

    assert len(items) == 1
    stored_id, stored_vector, stored_response = items[0]
    assert stored_id == "a"
    assert stored_response == "response for a"
    assert stored_vector.dtype == np.float32
    np.testing.assert_allclose(stored_vector, vector, rtol=1e-6)


def test_dynamo_all_items_skips_entries_with_no_vector(dynamo_store):
    # A response stored without a vector can't be replayed into the graph;
    # it should be skipped, not crash the whole rebuild.
    dynamo_store.put("has_vector", "r1", np.array([1.0, 0.0], dtype=np.float32))
    dynamo_store.put("no_vector", "r2")

    ids = [item[0] for item in dynamo_store.all_items()]

    assert ids == ["has_vector"]


def test_dynamo_all_items_and_len_follow_scan_pagination(dynamo_store):
    # DynamoDB caps a Scan page at 1MB and returns LastEvaluatedKey rather
    # than erroring, so a store that ignores pagination silently rebuilds a
    # partial graph -- and would report a wrong count -- with no failure
    # signal. 60 full-width (1024-dim, matching Titan V2) random vectors is
    # enough to spill past one page; an earlier version of this test used
    # np.ones, which serializes to "1.0" and fit in a single page, making
    # the test pass without ever exercising pagination at all.
    rng = np.random.default_rng(0)
    for i in range(60):
        dynamo_store.put(f"q_{i}", f"response {i}", rng.random(1024).astype(np.float32))

    # Guard the premise: if this ever stops spanning pages, the assertions
    # below stop testing what they claim to.
    assert "LastEvaluatedKey" in dynamo_store.table.scan()

    items = list(dynamo_store.all_items())
    assert len(items) == 60
    assert {item[0] for item in items} == {f"q_{i}" for i in range(60)}
    assert len(dynamo_store) == 60
