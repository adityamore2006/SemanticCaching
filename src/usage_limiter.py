"""
A daily ceiling on how many LLM calls the service will make.

Why this exists: the API is deliberately reachable from anywhere so the
demo URL always works, and every cache *miss* costs a real Bedrock call.
Without a ceiling, a stranger sending unique queries spends the account's
money, and cache hits offer no protection because a novel query misses by
definition.

Counted in DynamoDB rather than memory, and that choice is load-bearing:
an in-memory counter resets whenever the service restarts, so anyone who
could make the process restart could reset the cap. A guard that a restart
clears is not a guard.

The counter row lives in the same table as cache entries, under a reserved
id, and is written **without a `vector` attribute**. That single detail is
what keeps it invisible to DynamoDBCacheStore.all_items(), which already
skips vector-less items so the cold-start rebuild never tries to insert a
counter into the HNSW graph. Worth stating explicitly because it looks
incidental and is not.

This is the innermost of three layers. The nightly auto-stop and the $5
budget action (knowledge/learned.md section 23c) bound cost from the
outside; this bounds it per-day from the inside, before spend happens
rather than hours after billing data catches up.
"""

import datetime
import os

DEFAULT_DAILY_LIMIT = int(os.environ.get("DAILY_LLM_LIMIT", "100"))

# Prefix chosen so a counter row sorts and greps distinctly from the
# "q_<n>" cache entries, and so a human scanning the table can tell at a
# glance that it is not a cached answer.
USAGE_ID_PREFIX = "__usage__"


class DailyLimitReached(RuntimeError):
    """The service has made its allowed number of LLM calls for the day."""


def _today():
    # UTC, not local: the instance's timezone is incidental and a local
    # date would shift the reset point if the region ever changed.
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


class UsageLimiter:
    """Counts LLM calls per UTC day and refuses once the limit is hit."""

    def __init__(self, table, limit: int = DEFAULT_DAILY_LIMIT):
        self.table = table
        self.limit = limit

    def _key(self, day=None):
        return f"{USAGE_ID_PREFIX}{day or _today()}"

    def count(self, day=None) -> int:
        item = self.table.get_item(Key={"id": self._key(day)}).get("Item")
        return int(item["count"]) if item else 0

    def check(self) -> None:
        """Raise if today's allowance is already spent. Called before the
        LLM, so a refused request costs nothing."""
        used = self.count()
        if used >= self.limit:
            raise DailyLimitReached(
                f"daily limit of {self.limit} LLM calls reached ({used} used); resets at UTC midnight"
            )

    def record(self) -> int:
        """Increment today's count and return the new total.

        ADD is an atomic server-side update, so two concurrent requests
        cannot read the same value and both write back n+1. Read-then-write
        would silently undercount under exactly the burst of traffic the
        limit exists to stop.
        """
        result = self.table.update_item(
            Key={"id": self._key()},
            UpdateExpression="ADD #c :one",
            ExpressionAttributeNames={"#c": "count"},
            ExpressionAttributeValues={":one": 1},
            ReturnValues="UPDATED_NEW",
        )
        return int(result["Attributes"]["count"])

    def remaining(self) -> int:
        return max(0, self.limit - self.count())

    def wrap(self, llm):
        """Return `llm` with the limit enforced around it.

        Wrapping the callable rather than teaching CacheRouter about
        limits keeps the router unchanged and unaware, which is the same
        seam that let the LLM be a stub for five phases. The router still
        just calls something that takes a query and returns a string.

        Counts before calling, not after. That over-counts by one when a
        call fails in transport and was never billed, which is the safe
        direction to be wrong in for a spend guard: it can only stop
        sooner, never later. Counting afterwards would undercount every
        call that was billed but then raised, which is the direction that
        costs money.
        """
        def limited(query: str) -> str:
            self.check()
            self.record()
            return llm(query)
        return limited
