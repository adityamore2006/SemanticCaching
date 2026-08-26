# Progress

## Current status

- Phase 1 through Phase 5 are complete and pushed. **Phase 6 is built but not yet deployed or measured**, blocked on an external AWS constraint, not on code -- see "Phase 6 status" below.
- The baseline implementation in src/linear_search.py is in place and validated with focused tests.
- Scaffolding is decoupled around a shared index contract so linear search and HNSW are hot-swappable (swap happens at one factory call, nothing downstream changes).
- The eval harness (eval/threshold_sweep.py) is built, verified against a 194-pair hand-authored test set, and has produced a documented operating threshold. Full reasoning lives in knowledge/learned.md.

## Completed

- Confirmed the project brief and execution roadmap in PLAN.md.
- Built the brute-force cosine-similarity search index with insert + search behavior.
- Added correctness tests covering self-similarity, ordering, extreme values, empty inputs, and dimension errors.
- Introduced the VectorIndex abstract contract (src/vector_index.py): the seam that lets the cache router and eval harness depend on an interface, never a concrete index.
- LinearIndex now implements VectorIndex; added src/hnsw.py as a contract-conforming Phase 3 stub and src/factory.py (create_index) as the single hot-swap point.
- Added an index-agnostic contract test suite (tests/test_index_contract.py) that HNSW will inherit for free once implemented. Full suite: 18 passing.
- Set up a local venv + .gitignore; deps (numpy, pytest, sentence-transformers) installed and green.
- Built src/embedding.py (Embedder) and eval/threshold_sweep.py, the Phase 2 harness: embeds a hand-authored test set, builds a shared index of previously-seen queries, sweeps similarity thresholds, reports hit rate and wrong-match rate.
- Compared four embedding configurations (MiniLM, mpnet, bge-large unprefixed, bge-large prefixed) on the same eval; mpnet won on the actual numbers, including a case where a theoretically-better model (BGE, hard-negative trained) lost in practice. See knowledge/learned.md section 8.
- Scaled the eval set from 45 to 194 pairs (67 topic anchors) and built eval/verify_pairs.py, an automated collision checker that embeds the whole set and checks every pair against every anchor, not just its own. Caught and fixed three real authoring mistakes before trusting the numbers. See knowledge/learned.md section 10.
- Chose the final operating threshold (0.80 on all-mpnet-base-v2) using an explicit asymmetric cost model, safety weighted over raw hit rate, with the reasoning for why 0.80 and not stricter documented in knowledge/learned.md section 11.
- Built HNSWIndex from scratch (src/hnsw.py): probabilistic layer assignment, greedy multi-layer descent for search, SELECT-NEIGHBORS-SIMPLE for insert-time neighbor selection. Conforms to VectorIndex, so it dropped straight into the existing contract test suite (tests/test_index_contract.py now runs all 6 invariant checks against both "linear" and "hnsw", 23/23 passing, stable across 10 repeated runs despite randomized layer assignment). Design reasoning in knowledge/learned.md section 12.
- Validated HNSW against LinearIndex on the real 194-pair eval set (not synthetic data): 194/194 top-1 matches identical, similarity scores agreeing to 6 decimal places, and the threshold_sweep table matching the locked linear baseline exactly at every reference threshold (0.70/0.75/0.80/0.85). Caveat documented, not overclaimed: at 67 anchors this is expected since ef_search=50 sees nearly the whole graph; the real approximation/speed tradeoff is Phase 4's job once the dataset is large enough to matter.
- Added 50 new cross-domain seed anchors (data/phase4_new_anchors.json: recipe app, fitness tracker, budgeting app) alongside the original 67, verified zero cross-anchor collisions across the combined 117 (max cross-domain similarity 0.638). Kept separate from data/eval_pairs.json so Phase 2's locked threshold decision stays untouched.
- Built eval/scale_dataset.py: generates synthetic datasets by perturbing the 117 real anchors with calibrated Gaussian noise (sigma=0.018, empirically tuned to land parent-similarity at 0.85-0.95, matching real paraphrase pairs) instead of using unrepresentative random vectors.
- Built eval/recall_latency.py and ran it at n=1,000/10,000/50,000: recall@1 degrades 98.0% -> 64.5% -> 54.0% as ef_search=50 becomes a shrinking fraction of the graph (expected, honestly reported); query latency crosses over between linear and HNSW between 1k and 10k, reaching HNSW 4.3x faster than linear at 50k (linear query time grew 41x over the 50x data increase, HNSW's only 2x). Full results and reasoning in knowledge/learned.md section 14.
- Found and fixed a real O(n^2) bug in LinearIndex.insert (np.vstack reallocating the whole array every insert) that Phase 4's scale testing exposed but Phase 1/2's small-n tests never could. Fixed via lazy-cached matrix rebuild; insert time at n=50,000 dropped from 443.8s to 0.37s, full test suite unaffected (23/23 still passing).
- Diagnosed and fixed a real HNSW recall problem at n=50,000: an ef_search sweep (50->1600) showed recall flat, ruling out "beam too narrow"; a targeted diagnostic then showed 60.9% of misses landed in a completely different anchor cluster, pointing at unrecoverable greedy routing through the upper layers (ef=1, per the paper's own algorithm) as the real bottleneck. Added ef_upper (default 8, a deliberate deviation from the paper) to both insert's phase A and search()'s upper-layer descent. Recall@1 at n=50,000 improved 54.0% -> 65.5% with no latency regression (still ~3.3x faster than linear); re-verified small-scale exact correctness still holds (194/194). Full investigation in knowledge/learned.md section 15.
- Re-scoped the benchmark's realistic operating range to 1k-10k (down from 50k), matching what a real cache and real hosting costs actually look like; n=50,000 stays documented as the stress test that found the O(n^2) and routing bugs, not the ongoing target scale. Chose a fully serverless AWS architecture for the eventual deployment (API Gateway + Lambda + DynamoDB on-demand + Bedrock + CloudWatch), explicitly avoiding OpenSearch Serverless due to its non-trivial idle billing floor. Full reasoning and the canonical linear-vs-HNSW comparison table in knowledge/learned.md sections 16-17.
- Removed the unused, contract-contradicting metadata dict from both LinearIndex and HNSWIndex, then built Phase 5 for real: src/cache_store.py (CacheStore ABC + InMemoryCacheStore) and src/cache_router.py (CacheRouter: embed -> search -> threshold check -> hit/miss -> stubbed LLM call -> insert). index_kind has no default deliberately; the LLM call is stubbed, real Bedrock wiring is Phase 6. 34/34 tests passing (7 new router tests against a fast FakeEmbedder, 4 new store tests), plus a real end-to-end run with the actual embedding model: paraphrase hit at similarity 0.826, unrelated query missed at 0.099. Added an --interactive mode to cache_router.py so hit/miss behavior can be verified by hand, not just via automated tests. Full reasoning in knowledge/learned.md section 18.
- Designed (not built) Phase 6's persistence strategy: rebuild the HNSW graph from DynamoDB on every Lambda cold start, chosen over S3 graph-snapshotting (more complex, deferred until proven necessary) and provisioned concurrency (rejected outright, a standing 24/7 cost that contradicts the near-zero-idle-cost architecture). Estimated at $0-1/month at realistic demo traffic, with cold-start latency (~16.5s at n=10,000) named as the honest non-dollar cost. Reaffirmed LinearIndex never gets deployed -- index_kind is always "hnsw" for the Lambda handler. Full reasoning in knowledge/learned.md section 19. Debugging story in knowledge/learned.md section 13.

## Architectural decisions

- Single-responsibility index: the index maps vectors <-> ids only. Response storage (cache), text embedding, and hit/miss decisions are separate components, so each stays swappable on its own.
- Hot-swap via registry + factory: linear vs hnsw is one config value; the eval harness can run the identical code path against both to compute recall.
- Linear search is kept permanently as the exact ground-truth answer key for grading HNSW's approximate results (Phase 3/4).

## Next milestone

- Re-derive the operating threshold against Titan, then deploy. Both are blocked on Bedrock quota, see "Phase 6 status" below.
- Keep notes brief and evidence-backed.

## Phase 6 status (revised: EC2, not Lambda)

**The platform changed.** The Lambda/container design was built, then replaced with a single small EC2 instance running FastAPI. Reasoning and the cost comparison are in learned.md section 23; the short version is that Lambda's cold start is recurring and externally scheduled, which a latency-sensitive cache cannot absorb, and keeping a Lambda warm costs more than just running the instance.

**Deleted:** `src/lambda_handler.py`, `template.yaml`, `src/requirements.txt`, and with them the whole container path (Docker, ECR, SAM packaging). All of it existed to work around Lambda's 250MB limit.

**Added:** `src/api.py` (FastAPI: `/query`, `/health`, `/stats`), plus `CacheRouter.save_snapshot()` / `load_snapshot()` so the built graph persists to the instance's disk and reloads at boot instead of being rebuilt every time.

**Embedding is local again** (`all-mpnet-base-v2`), so the locked 0.80 threshold and all of Phase 2's eval work stay valid. No re-derivation needed. `bedrock_embedder.py` and the factory's `"bedrock"` entry are kept deliberately, since having two backends behind one interface is what made this switch a small change.

**A fourth bug, found by the switch:** the end-to-end demo had been passing on the wrong embedding model for two phases, so learned.md section 18's recorded numbers were MiniLM's, not mpnet's. Corrected, with the full story in section 24. The demo pair was re-chosen from the verified eval set to one that genuinely clears the threshold on the correct model.

**Verified locally:** 51/51 tests pass. The API was run end to end with the real model across two boots, confirming the snapshot round-trips (`restored_from=snapshot`, hit at similarity 1.0 on the second boot) and that a fresh boot with no snapshot and no DynamoDB still works.

### DEPLOYED (2026-08-26)

Live on AWS and verified end to end, then stopped. Measured numbers in learned.md section 23b.

- **Stack:** `semantic-cache` (CloudFormation). Instance `i-0066a19f7fa7614cb`, `m7i-flex.large`, us-east-1. Table `semantic-cache-CacheTable-1H90CFPXOWNLO`, 67 seeded entries persisted.
- **Verified:** seeded 67/67 anchors over HTTP, ran the full behaviour tour against the live URL with similarity scores identical to local, then stopped and restarted to confirm the snapshot path (`restored_from=snapshot entries=73`, ready in 1.6s) and that the restored cache still serves hits.
- **Cost so far: $0.00.** Instance is stopped; only the EBS volume accrues.

**Spending guards (deployed and tested, learned.md section 23c):** a nightly EventBridge schedule stops the instance at 3am ET, a $5 budget action stops it if spend crosses, and the budget emails at 50/80/100% plus a forecast alert. The schedule was fired deliberately to confirm it works rather than assumed; that test is what caught `cloudformation deploy` silently reusing a previously-set parameter while reporting "No changes to deploy". Both guard roles are scoped to `ec2:StopInstances` only, so neither can ever cost money. Worst case for a forgotten instance is about $2.30.

**Console locations:** EventBridge > Scheduler > Schedules for the auto-stop (not Rules, which is the older mechanism and will be empty); Billing and Cost Management > Budgets for the limit, alerts, and the action; CloudFormation > semantic-cache > Resources for everything at once. Change things through the template, not the console: a hand edit gets silently reverted by the next deploy.

**Operating it:**
```
aws ec2 start-instances --instance-ids i-0066a19f7fa7614cb
aws ec2 stop-instances  --instance-ids i-0066a19f7fa7614cb
```
The public IP changes on every start. Read it with `aws ec2 describe-instances --instance-ids i-0066a19f7fa7614cb --query 'Reservations[0].Instances[0].PublicIpAddress' --output text`. systemd starts the API automatically on boot, so starting the instance is the only manual step.

### Remaining

1. **Bedrock quota** for Claude Haiku 4.5 (`L-CCA5DF70` requests/min, `L-58BE175A` tokens/min, both adjustable and currently 0). Until it clears, `LLM_MODEL_ID` stays unset and the miss path uses the Phase 5 stub, so everything that makes this a cache is demonstrable without it. Once approved: set `LLM_MODEL_ID` in `/etc/systemd/system/semantic-cache.service`, restart, and re-seed to replace stub answers with real ones.
2. **Optional, deferred deliberately:** a stable address (Elastic IP, ~$3.65/mo) so the demo URL survives restarts; HTTPS via Caddy; and the CloudWatch dashboard. None are needed to demo.
3. **Open lead, not blocking:** the orphaned-node hypothesis in learned.md section 22b, which may explain the remaining half of section 15's recall ceiling.

---

## Superseded: the Lambda build (kept for the reasoning)

Read learned.md sections 16 and 19 (the decisions) plus 20, 21, and 22 (what building it actually taught) to pick this up cold.

**Built and locally verified:**
- `src/embedder.py` / `local_embedder.py` / `bedrock_embedder.py` / `embedder_factory.py`: the embedding backend is now an interface with two implementations, same seam as VectorIndex and CacheStore. Forced by the deployment (mpnet + torch is ~2GB, past Lambda's zip limit), reasoning in learned.md section 20.
- `src/dynamodb_cache_store.py`: DynamoDBCacheStore against the existing CacheStore contract, plus `all_items()` for the cold-start rebuild. Tested against moto, including Scan pagination.
- `src/lambda_handler.py`: cold-start rebuild + routing + structured JSON logs for the eventual dashboard.
- `src/bedrock_llm.py`: the real Claude call, replacing the Phase 5 stub.
- `template.yaml`: HTTP API + Lambda + on-demand DynamoDB, IAM scoped to the one table and to Bedrock model ARNs. `sam validate --lint` passes and `sam build` succeeds (~102MB artifact, correct x86_64 linux wheels).
- 46/46 tests passing. The full handler path, including a simulated container recycle, was exercised end to end against a mocked DynamoDB.
- Two real bugs found and fixed in the process, both of which would have shipped silently: a falsy empty cache store being discarded by an `or` default (learned.md section 21), and id-counter collisions across a restart serving wrong answers at similarity 1.0 (section 22).

**The Bedrock finding that came out of it, still relevant:** the AWS account is new. Account verification cleared and models show `AUTHORIZED` / `AVAILABLE`, but **every Bedrock on-demand quota on the account is 0 requests/minute**, so `InvokeModel` returns `ThrottlingException` in every region tested, for Titan and every Claude model, and it did not clear after a day. Titan's quota (`L-26C560CE`) is `Adjustable: False`, so it is AWS-side provisioning rather than something to request. Claude Haiku 4.5's (`L-CCA5DF70`, `L-58BE175A`) are adjustable and unrequested. This is what made a Bedrock-embedding design untenable and is why embedding moved back to local mpnet.

**What was dropped with the Lambda design and does not need doing:** re-deriving the operating threshold against Titan. Local mpnet keeps the 0.80 threshold and all of Phase 2's eval work valid, so that whole workstream disappeared rather than being deferred.

## Guardrail note

- No rabbit-hole exploration beyond the active phase unless it directly advances the approved build plan or is explicitly approved as a scope change.
