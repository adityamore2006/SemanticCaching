"""
DynamoDB-backed cache storage -- the persistent half of the deployed
system.

Implements the same three-method CacheStore contract InMemoryCacheStore
does, so CacheRouter needs no knowledge of which one it's holding. Kept in
its own module rather than added to cache_store.py so that importing the
in-memory store never drags in boto3.

Item shape, one per cached entry:
    id       (String, partition key) -- the router's "q_0", "q_1", ...
    response (String)                -- what the LLM answered
    vector   (List<Number>)          -- the query embedding

Storing the vector is what makes a stateless Lambda viable at all. The
HNSW graph itself is never serialized; it's rebuilt by replaying insert()
over these vectors on cold start (knowledge/learned.md section 19 chose
this over S3 graph snapshots and over provisioned concurrency). That's
also why all_items() exists below and isn't part of the CacheStore ABC.
"""

from decimal import Decimal

import boto3
import numpy as np

from cache_store import CacheStore


class DynamoDBCacheStore(CacheStore):
    def __init__(self, table_name: str, region_name: str = None, table=None):
        # `table` is injectable so tests can hand in a moto-backed resource;
        # in Lambda the default picks up the execution role automatically.
        if table is None:
            table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)
        self.table = table
        self.table_name = table_name

    def put(self, id, response, vector=None, source=None):
        item = {"id": id, "response": response}
        if source is not None:
            item["source"] = source
        if vector is not None:
            # DynamoDB has no float type -- Number is arbitrary-precision
            # decimal, and boto3 rejects raw floats rather than silently
            # rounding them. float32 -> str -> Decimal keeps the exact
            # decimal repr of the stored float instead of the long binary
            # expansion Decimal(float) would produce.
            item["vector"] = [Decimal(str(float(x))) for x in vector]
        self.table.put_item(Item=item)

    def get(self, id):
        item = self.table.get_item(Key={"id": id}).get("Item")
        return item["response"] if item else None

    def __len__(self):
        # Scan with COUNT rather than DescribeTable's ItemCount, which AWS
        # only refreshes roughly every six hours and would report stale
        # counts immediately after a write.
        total = 0
        kwargs = {"Select": "COUNT"}
        while True:
            page = self.table.scan(**kwargs)
            total += page["Count"]
            if "LastEvaluatedKey" not in page:
                return total
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def count_by_source(self):
        """Return {source: count} across the table, so a caller can tell how
        much of the cache is curated seed data and how much the model
        actually produced. Entries written before `source` existed count as
        "unknown" rather than being silently attributed to either."""
        counts = {}
        kwargs = {"ProjectionExpression": "id, #s", "ExpressionAttributeNames": {"#s": "source"}}
        while True:
            page = self.table.scan(**kwargs)
            for item in page["Items"]:
                if str(item["id"]).startswith("__"):
                    continue  # reserved rows (usage counters), not cache entries
                counts[item.get("source", "unknown")] = counts.get(item.get("source", "unknown"), 0) + 1
            if "LastEvaluatedKey" not in page:
                return counts
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def all_items(self):
        """
        Yield (id, vector, response) for every stored entry, following
        pagination to the end.

        Deliberately NOT part of the CacheStore ABC: it exists solely for
        the Lambda cold-start rebuild, and InMemoryCacheStore has no
        equivalent need (its paired index never lost its state in the first
        place). Putting it in the shared contract would force a meaningless
        implementation onto the in-memory side just to satisfy the
        interface.
        """
        kwargs = {}
        while True:
            page = self.table.scan(**kwargs)
            for item in page["Items"]:
                vector = item.get("vector")
                if vector is None:
                    # Written before vectors were persisted, or by something
                    # other than the router. Nothing to replay into the
                    # graph, so skip rather than crash the whole rebuild.
                    continue
                yield (
                    item["id"],
                    np.array([float(x) for x in vector], dtype=np.float32),
                    item["response"],
                )
            if "LastEvaluatedKey" not in page:
                return
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
