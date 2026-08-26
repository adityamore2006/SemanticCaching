"""
Empty the cache table.

Exists for one specific moment: the switch from the stubbed LLM to real
Bedrock. Every entry stored while the stub was active holds
"[stub response for: ...]" rather than an answer, and those entries do not
go away when the real model is wired in. They keep being served as cache
hits, which look like success, so nothing surfaces the problem. Purge, then
re-seed, so the cache holds real answers.

Also useful for resetting between demos: negative tests are one-shot,
because a query that misses is stored and hits on the next ask.

    python scripts/purge_cache.py --table <name>          # dry run
    python scripts/purge_cache.py --table <name> --yes    # actually delete

Deliberately requires --yes. This deletes everything in the table and the
only recovery is to re-seed and re-pay for the answers.
"""

import argparse
import os
import sys

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from usage_limiter import USAGE_ID_PREFIX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=os.environ.get("CACHE_TABLE_NAME"),
                        help="defaults to $CACHE_TABLE_NAME")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--yes", action="store_true", help="actually delete")
    parser.add_argument("--keep-usage", action="store_true",
                        help="leave the daily LLM-call counter alone")
    args = parser.parse_args()

    if not args.table:
        parser.error("no table given; pass --table or set CACHE_TABLE_NAME")

    table = boto3.resource("dynamodb", region_name=args.region).Table(args.table)

    ids, kwargs = [], {"ProjectionExpression": "id"}
    while True:
        page = table.scan(**kwargs)
        ids.extend(item["id"] for item in page["Items"])
        if "LastEvaluatedKey" not in page:
            break
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    usage = [i for i in ids if i.startswith(USAGE_ID_PREFIX)]
    entries = [i for i in ids if not i.startswith(USAGE_ID_PREFIX)]

    # The usage counter shares this table but is not a cache entry, and
    # wiping it silently resets the daily spend cap. Keeping that opt-in
    # means a routine purge cannot quietly remove a spend guard.
    targets = entries if args.keep_usage else ids

    print(f"table {args.table}: {len(entries)} cache entries, {len(usage)} usage rows")
    if not args.yes:
        print(f"dry run. {len(targets)} items would be deleted. re-run with --yes")
        return

    with table.batch_writer() as batch:
        for item_id in targets:
            batch.delete_item(Key={"id": item_id})

    print(f"deleted {len(targets)} items")
    print("the running service still holds the old graph in memory; "
          "restart it (or stop/start the instance) before re-seeding")


if __name__ == "__main__":
    main()
