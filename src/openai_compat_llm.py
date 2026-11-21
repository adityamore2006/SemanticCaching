"""
An LLM backend for any server that speaks the OpenAI chat-completions API.

That covers Ollama (the default, local and free), LM Studio, vLLM and
OpenAI itself. Only base_url, model and api_key differ, so switching
provider is configuration, not code.

Uses the standard library only, so there is no SDK dependency.
"""

import json
import socket
import urllib.error
import urllib.request
from typing import Optional

from llm import LLM, LLMError, LLMTimeout

DEFAULT_SYSTEM_PROMPT = (
    "You are a customer support assistant for a software product. "
    "Answer the user's question directly and concisely, in two to four sentences."
)


class OpenAICompatLLM(LLM):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",  # Ollama
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_tokens: int = 400,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def __call__(self, query: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers=headers
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (socket.timeout, TimeoutError) as exc:
            raise LLMTimeout(f"no answer from {self.url} within {self.timeout}s") from exc
        except urllib.error.HTTPError as exc:
            raise LLMError(f"{self.url} returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise LLMTimeout(f"no answer from {self.url} within {self.timeout}s") from exc
            raise LLMError(f"could not reach {self.url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"{self.url} returned a body that is not JSON") from exc

        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape from {self.url}") from exc

        # A truncated answer would be cached and replayed as if complete.
        if choice.get("finish_reason") == "length":
            raise LLMError("answer was cut off at max_tokens; refusing to use it")
        return text
