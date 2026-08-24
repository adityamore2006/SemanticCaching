"""
The real LLM call -- what a cache miss actually costs, and what every
cache hit avoids paying.

Phase 5 deliberately left this as a stub (cache_router.call_llm) so the
routing logic could be built and tested for free. CacheRouter takes `llm`
as a constructor parameter precisely so swapping the stub for this is
passing a different callable, with no change to the routing logic itself.

Two different Bedrock surfaces are in play in this project, which is worth
not confusing:
  - Embeddings (bedrock_embedder.py) call Titan through boto3's
    bedrock-runtime, because Titan is an Amazon model with its own
    request format.
  - This module calls Claude through Anthropic's own Bedrock client,
    which speaks the Messages API. The raw bedrock-runtime InvokeModel
    path also works but is the legacy one for Claude models.
"""

from anthropic import AnthropicBedrockMantle

# Haiku on purpose. Every other cost decision in this project (on-demand
# DynamoDB, no provisioned concurrency, no OpenSearch Serverless -- see
# knowledge/learned.md section 16) was made to keep idle cost near zero,
# and the cache's whole value proposition is measured in avoided calls to
# this model. The cheapest current-gen model keeps the demo honest and the
# bill small; override via env var if a deployment wants more capability.
DEFAULT_MODEL = "anthropic.claude-haiku-4-5"

# Short by design, not to lowball the model. These responses get stored as
# DynamoDB items and replayed verbatim on every future cache hit, so a
# rambling answer is one that gets served hundreds of times.
DEFAULT_MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are the support assistant for a SaaS product. "
    "Answer the user's question directly and concisely, in at most a short paragraph."
)


class BedrockLLM:
    """Callable that answers a query via Claude on Bedrock.

    Implements the same one-argument callable shape as the stub it
    replaces, so CacheRouter can't tell the difference.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        region_name: str = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # Injectable for tests; in Lambda the default client picks up the
        # execution role's credentials automatically.
        if client is None:
            client = AnthropicBedrockMantle(aws_region=region_name)
        self.client = client

    def __call__(self, query: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
