import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fake_llm import FakeLLM
from llm import LLMError, LLMTimeout
from llm_factory import create_llm
from openai_compat_llm import OpenAICompatLLM


# --- FakeLLM ---


def test_fake_llm_is_deterministic_and_counts_calls():
    llm = FakeLLM()
    assert llm("hello") == "fake answer to: hello"
    assert llm("hello") == "fake answer to: hello"
    assert llm.calls == 2


def test_fake_llm_can_fail_on_demand():
    llm = FakeLLM(error=LLMError("boom"))
    with pytest.raises(LLMError):
        llm("anything")
    assert llm.calls == 1


def test_fake_llm_counts_calls_correctly_across_threads():
    llm = FakeLLM()
    threads = [threading.Thread(target=lambda: [llm("q") for _ in range(100)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert llm.calls == 800


# --- factory ---


def test_factory_builds_by_name_and_rejects_unknown():
    assert isinstance(create_llm("fake"), FakeLLM)
    assert isinstance(create_llm("openai_compat", model="m"), OpenAICompatLLM)
    with pytest.raises(ValueError):
        create_llm("nope")


# --- OpenAICompatLLM, against a throwaway local HTTP server ---


def serve(handler):
    """Run `handler(request_body) -> (status, payload, delay)` on a free port."""
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            H.last_request = (body, dict(self.headers))
            status, payload = handler(body)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, H, f"http://127.0.0.1:{server.server_port}/v1"


def ok(text="an answer", finish="stop"):
    return lambda body: (200, {"choices": [{"message": {"content": text}, "finish_reason": finish}]})


def test_openai_compat_returns_the_answer_and_sends_the_right_request():
    server, H, base = serve(ok("use the forgot password link"))
    try:
        llm = OpenAICompatLLM(model="m", base_url=base, api_key="secret")
        assert llm("how do I reset my password?") == "use the forgot password link"
        body, headers = H.last_request
        assert body["model"] == "m"
        assert body["messages"][-1] == {"role": "user", "content": "how do I reset my password?"}
        assert headers["Authorization"] == "Bearer secret"
    finally:
        server.shutdown()


def test_openai_compat_refuses_a_truncated_answer():
    server, _, base = serve(ok("half an ans", finish="length"))
    try:
        with pytest.raises(LLMError, match="cut off"):
            OpenAICompatLLM(model="m", base_url=base)("q")
    finally:
        server.shutdown()


def test_openai_compat_maps_http_errors_and_bad_shapes_to_llm_error():
    server, _, base = serve(lambda body: (500, {"error": "overloaded"}))
    try:
        with pytest.raises(LLMError, match="HTTP 500"):
            OpenAICompatLLM(model="m", base_url=base)("q")
    finally:
        server.shutdown()

    server, _, base = serve(lambda body: (200, {"unexpected": True}))
    try:
        with pytest.raises(LLMError, match="unexpected response shape"):
            OpenAICompatLLM(model="m", base_url=base)("q")
    finally:
        server.shutdown()


def test_openai_compat_unreachable_server_is_llm_error():
    with pytest.raises(LLMError, match="could not reach"):
        OpenAICompatLLM(model="m", base_url="http://127.0.0.1:1/v1", timeout=2)("q")


def test_openai_compat_times_out():
    import time
    server, _, base = serve(lambda body: (time.sleep(1.5), (200, {}))[1])
    try:
        with pytest.raises(LLMTimeout):
            OpenAICompatLLM(model="m", base_url=base, timeout=0.3)("q")
    finally:
        server.shutdown()
