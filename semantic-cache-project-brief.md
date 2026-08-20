# Semantic Caching Layer for LLM APIs — Project Brief

## Who I am (context for whoever is helping with this)

Rising junior, Computer Science + Data Science at UNC Chapel Hill, 3.9 GPA, BS/MS combined track. AWS Certified Cloud Practitioner (studying for SAA-C03 next). Completed a DealCloud implementation internship at Intapp (enterprise CRM for private capital/legal clients, hands-on with Bedrock, Lambda, DynamoDB, OpenSearch Serverless, API Gateway) and an earlier data/product internship at an investment bank (Integrus), where I built pipelines that fixed recurring data reconciliation problems by hand. Also built ThreadMind, a Socratic AI tutor deployed in a real UNC course with an evaluation pipeline across 700+ interaction logs, headed toward a SIGCSE paper.

Comfortable in Python, have shipped real AWS infra before, not new to Bedrock/Lambda/DynamoDB.

**Target roles:** AI Engineering / Forward Deployed Engineer / AWS ProServe / Solutions Architect style internships at top companies (not classic algorithms-heavy SWE). This matters architecturally, see "Why this project" below.

**Style note:** I don't use em dashes in my writing, prefer that carried through in anything drafted for me.

## Timeline

**About a week, roughly 20 hours of build time total.** Scope has already been triaged, see "Build priority" below. Do not suggest scope expansions without flagging the time cost first.

## Why this project (the reasoning trail, so the architecture isn't second-guessed)

I went through several project ideas before landing here, each rejected for a specific, real reason:

1. **CardSense** (an "agentic" credit card recommender on Bedrock, Lambda, DynamoDB, OpenSearch) — killed because the core task (which card maximizes expected value for a purchase) is a fully deterministic lookup/calculation. Wrapping it in an LLM agent orchestrating tool calls was decorative, not functional. Diagnosed using Anthropic's own workflow-vs-agent framework: agents are only justified when the correct path can't be hardcoded in advance and you can still verify progress. A card EV calculation can be hardcoded. This is the standing test applied to every idea since.

2. **Multi-agent AgentCore systems** (a research analyst, then an IB diligence-reconciliation agent) — architecturally sound and genuinely tied to FDE/SA interview formats (which test live architecture discussion and decomposition of ambiguous problems, not LeetCode), but ultimately shelved for this cycle because of scope and the "am I proving anything that's mine" concern below.

3. **A verifiable coding agent** (SWE-bench style, agent fixes real bugs, graded by hidden pre-existing tests) — technically legitimate (the verification design, sandboxing, and anti-cheating harness are real engineering), but the actual "impressive" moment in the demo (reading code, finding the bug, writing a fix) is the model's capability, not mine. Correct pattern, wrong fit for what I actually want to prove about myself.

4. **The real pivot**: what do SWE portfolio projects that land FAANG-style offers actually look like? Not isolated algorithms (implementing a hashmap), but scaled-down real **systems** with multiple interacting components and real engineering tradeoffs (mini Redis, mini distributed KV store). The AI-engineering equivalent of that same pattern, a system where the "hard part" is unambiguously my own code, not the model being smart, is infrastructure that AI teams operate in production: an LLM inference engine, a vector search engine, or a semantic cache. Landed on the last one because it combines the other two.

**This project is the synthesis:** a semantic caching layer requires building a real ANN vector index from scratch (the systems/algorithms work, fully mine, same category as implementing a database index) and applying it to a genuine, well-precedented AI infrastructure problem (cutting redundant LLM API calls), which is exactly the kind of cost/latency concern that comes up in AI engineering and Solutions Architect conversations with real customers.

## Design principles (non-negotiable, apply to any implementation decisions)

- **The interesting/hard part must be code I write**, not an LLM call doing the interesting part. The LLM here is just the thing being called, it has no agency or reasoning role in the system itself.
- **Build it as a system, not a single algorithm.** Multiple real components with tradeoffs between them (embedding, index, decision logic, storage, routing), not one isolated function.
- **Needs a real, measured evaluation story**, not "it works." The threshold tuning tradeoff (below) is the actual deliverable, not a side note.
- **No customers/users required to prove value.** Self-generated benchmark traffic and measured numbers are sufficient and by design.
- **Ties to AWS** where possible (existing skill investment, relevant to Solutions Architect target roles), but AWS deployment is explicitly lower priority than the working core system, see timeline.

## Project description

A proxy layer that sits between an application and an LLM API (Claude via Bedrock). Instead of caching only exact repeated prompts, it recognizes when a new query means the same thing as one already answered, even if worded differently, and serves the cached response instead of making a new API call.

**Flow:**
1. Incoming query gets embedded.
2. Embedding is checked against a self-built HNSW (hierarchical navigable small world) vector index of previously-seen queries.
3. If the nearest neighbor's similarity is above a threshold: cache hit, return the stored response immediately, no API call.
4. If below threshold: cache miss, call the real LLM, store the query embedding + response in the index and cache, return the fresh response.

**The core engineering problem, and the actual point of the project:** where to set the similarity threshold. Too loose, the cache confidently serves a wrong answer to a question that only sounded like a cached one, with no visible failure signal. Too strict, and it barely saves anything. This is a precision/recall tradeoff that has to be measured, not guessed, using a deliberately constructed test set of true paraphrase pairs (should match) and deceptive near-miss pairs (should not match), swept across threshold values to find where hit rate and wrong-answer rate cross.

**Real-world precedent** (for grounding/comparison, not to copy): GPTCache (Zilliz) is the standard open-source implementation of this pattern. A published academic version of the same architecture (embed query → Redis-backed ANN index → serve on hit, call + cache on miss) reported reducing API calls by up to 68.8%, with cache hit rates of 61.6-68.8% and positive (correct) hit rates exceeding 97%. Industry sources note roughly 31% of production LLM traffic is semantically similar/duplicated, which is the underlying reason this pattern is worth building at all. Anthropic and OpenAI both run related caching mechanisms (prefix caching) natively at the infrastructure level for cost reasons.

## Architecture

**Core (must-have, local, no cloud dependency required to work):**
- Embedding generation (call an embedding API or use a small local model)
- HNSW index, built from scratch (not FAISS/a library, this is the part that has to be mine), supporting insert and approximate-nearest-neighbor search
- Similarity threshold decision logic
- Cache storage (in-memory dict or SQLite is fine for the core build)
- Request routing: hit → return cached response, miss → call real LLM, store result

**Stretch (AWS phase, cut first if time runs short):**
- API Gateway + Lambda as the proxy layer, sitting in front of Bedrock
- DynamoDB or ElastiCache as the cache store
- CloudWatch dashboard tracking hit rate, latency saved, cost saved over time
- Optional: compare the from-scratch HNSW index against OpenSearch Serverless's native vector search, as a "built it myself vs. the managed service" comparison point

## Evaluation plan (this is the deliverable, treat it as core, not optional)

1. Build a test set: pairs of queries that are true paraphrases (should cache-hit) and pairs that are superficially similar but actually different (should cache-miss).
2. Run the system across a range of similarity thresholds.
3. Plot/report hit rate vs. wrong-answer rate at each threshold.
4. Identify and justify the chosen operating threshold with the numbers, not intuition.
5. If AWS phase happens: report cost saved and latency saved on top of the accuracy tradeoff.

## Build priority / time budget (~20 hours total, 1 week)

Rough estimate from a prior planning pass, treat as a guide not gospel:

| Component | Est. hours | Priority |
|---|---|---|
| HNSW from scratch (correct implementation) | 6-8 | Must-have, do not cut |
| Embedding + cache hit/miss logic | 1-2 | Must-have |
| Test set construction + threshold sweep (eval story) | 3-4 | Must-have, this is the actual proof of skill |
| AWS wiring (API Gateway, Lambda, DynamoDB, IAM) | 5-8 | Cut first if time is short |
| Dashboard | 2-4 | Cut second if time is short |

**If time runs out: a fully working local Python version (HNSW + cache logic + a real threshold sweep with numbers) is a complete, defensible project on its own.** AWS deployment and the dashboard are valuable additions but not the thing an interviewer will actually probe on, the threshold decision and why is the likely question.
