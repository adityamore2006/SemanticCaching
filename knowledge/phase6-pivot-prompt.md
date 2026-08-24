# Handoff prompt: pivot Phase 6 from Bedrock embeddings to local mpnet

Copy everything below the line into a fresh chat.

---

I'm working on `/Users/adityamore/Desktop/SemanticCaching`, a semantic caching layer for LLM APIs. Read `knowledge/PLAN.md`, `knowledge/Progress.md`, and `knowledge/learned.md` first — especially learned.md sections 16, 19, 20, 21, and 22, which cover the AWS architecture decisions and what building Phase 6 already taught.

## Where things stand

Phases 1-5 are complete: a from-scratch HNSW index, a 194-pair eval set, and a locked operating threshold of **0.80 on `all-mpnet-base-v2`**, derived with an explicit asymmetric cost model (learned.md section 11). Phase 6 is fully built and committed — `DynamoDBCacheStore`, a Lambda handler with cold-start graph rebuild, `bedrock_llm.py`, and a SAM template. 46/46 tests pass and the whole handler path was verified end to end against a mocked DynamoDB, including a simulated container recycle.

## Why I'm pivoting

Phase 6 currently embeds via Bedrock's Titan Text Embeddings V2, chosen because `all-mpnet-base-v2` plus torch is ~2GB, past a zip-packaged Lambda's 250MB limit. That is blocked and not by anything in the repo:

- Model access is fully granted (`agreementAvailability: AVAILABLE`, `authorizationStatus: AUTHORIZED` for both Titan and Claude Haiku 4.5).
- But **every Bedrock `InvokeModel` call returns `ThrottlingException` (429) on the first request in a fresh process**, in us-east-1 and us-west-2, for both Titan and every Claude model. That is not real rate limiting — it is what AWS returns when provisioned throughput is zero on a new account.
- Titan's on-demand quota (`L-26C560CE`) is `Adjustable: False`, so it cannot be requested, only waited out. Claude Haiku 4.5's are adjustable and also 0 (`L-CCA5DF70` requests/min, `L-58BE175A` tokens/min).
- This did not clear after a day of waiting.

The deciding factor is not just the block: switching to Titan **invalidates the locked 0.80 threshold**, because a threshold is bound to its embedding model's vector space (learned.md section 9). Going back to mpnet keeps all of Phase 2's eval work — the 194-pair set, the four-model comparison, the cost model — valid and unchanged.

## The task

Pivot the deployed embedder from Titan back to local `all-mpnet-base-v2`, running in a **container-image Lambda** (10GB limit, comfortably fits torch) instead of a zip package.

**Do not revert any commits.** I already checked: only about 6 lines of real code are Titan-coupled (`src/lambda_handler.py` lines 31 and 68; `template.yaml` lines 26, 28, 76, 100). Everything else from Phase 6 is embedder-agnostic and worth keeping, including two bug fixes that apply equally under the container design — the falsy-empty-cache-store bug (learned.md section 21) and the id-collision-on-restart bug (section 22).

**Keep `src/bedrock_embedder.py` and the factory's `"bedrock"` entry.** Having two working backends behind one interface is the point of the seam, and the Bedrock path becomes usable again the moment quota appears. This is the same reason `LinearIndex` was kept after HNSW existed.

Concretely:
1. `Dockerfile` based on `public.ecr.aws/lambda/python:3.12`, installing sentence-transformers + torch and baking the model weights into the image at build time (do NOT download them at cold start — that would add a large download to every cold container).
2. `template.yaml`: switch the function to `PackageType: Image` with the SAM `Metadata`/`Dockerfile` block, drop the `EmbeddingModelId` parameter and its env var, and remove Titan's ARN from the IAM policy. Keep the Claude ARNs — the LLM call still goes to Bedrock. Raise `MemorySize` (torch needs more than 1024MB) and re-check `Timeout` against the now-slower cold start.
3. `src/lambda_handler.py`: construct `SentenceTransformerEmbedder` instead of `BedrockEmbedder`.
4. `src/requirements.txt`: add sentence-transformers; note the size implication in the comment, which currently claims the artifact is ~102MB and will no longer be true.
5. Set the template's `OperatingThreshold` back to **0.80**, which is now correct again rather than a placeholder.

## The miss-path LLM call

The generative call still targets Claude on Bedrock and is still quota-blocked. Do not silently leave it broken. Either keep the Phase 5 stub for now so the infrastructure can be proven end to end, or wire it to the Anthropic API directly with an API key (bypassing AWS quota entirely) — flag the tradeoff and let me choose. Whichever way, the cache itself (embed → search → threshold → hit/miss → store) will work fully, since that path no longer touches Bedrock at all.

## How I want you to work

This is the methodology that has worked across this project, and the reason `knowledge/learned.md` is worth reading before writing code:

- **Verify, do not assume.** Every performance and correctness claim in this repo is backed by a measurement that was actually run. If you fix something, re-measure and show the number. If a test passes, check that it is testing what it claims — one pagination test in this repo passed for a while without ever exercising pagination.
- **Never trust an aggregate.** Drop to the raw rows. Two of this project's real bugs were found that way (learned.md sections 6 and 21) and hidden by summary metrics.
- **Say what a change costs, not just what it buys.** Sections 14, 17, and 20 all lead with the unfavorable side. If the container image makes cold starts materially worse, measure it and write the number down rather than glossing it.
- **Explain reasoning in plain language and flag tradeoffs before committing to them.** If a decision has a real alternative, name it and say why it lost.
- **Update `knowledge/Progress.md` and add a `knowledge/learned.md` section** for anything non-obvious. learned.md is written as interview talking points, not a changelog: what decision was made, what alternatives were considered, what it traded away.
- **I do not use em dashes in my writing.** Keep that out of anything drafted for me.
- Keep the scope to this pivot. Do not expand into the CloudWatch dashboard until the deploy is proven.

## Verification

- `pytest` — all 46 existing tests must stay green.
- `sam validate --lint` and `sam build` must both succeed. `sam build` will now need Docker running; it was not running last session, so start Docker Desktop first.
- Re-run the local end-to-end check with the real embedding model (`python src/cache_router.py`) and confirm the paraphrase still hits around 0.826 and the unrelated query still misses around 0.099, matching learned.md section 18. That is the direct evidence the 0.80 threshold is valid again.
- Exercise `lambda_handler.handler` against a mocked DynamoDB including a simulated cold start (set `lambda_handler._router = None` between calls) and confirm `restored_entries` is non-zero on the rebuild. That check is what caught the falsy-store bug.
- Do not run `sam deploy` without asking me first — it creates real billed AWS resources.
