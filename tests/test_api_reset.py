"""
Reset behaviour at the HTTP boundary.

Exercises api.py through TestClient rather than calling CacheRouter
directly, because the parts most likely to break are the ones only the
endpoint touches: the auth gate, the snapshot file, and rebuilding the
router in place.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api

CORPUS = [
    ("How do I reset my password?", "Use the forgot password link."),
    ("Can I merge two accounts into one?", "Settings then Account then Merge."),
    ("How do I cancel my subscription?", "Settings then Billing then Cancel."),
]


class FakeEmbedder:
    """Deterministic per-text vectors, so distinct questions never collide
    and identical ones always match exactly."""

    dim = 16
    model_name = "fake-embedder"

    def embed(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=self.dim).astype(np.float32)
        return v / np.linalg.norm(v)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """An app wired to a fake embedder, an answering LLM, and a snapshot
    path under tmp_path, so tests never touch the real model or disk."""
    monkeypatch.setattr(api, "SentenceTransformerEmbedder", FakeEmbedder)
    monkeypatch.setattr(api, "load_seed_answers", lambda *a, **k: list(CORPUS))
    monkeypatch.setattr(api, "SNAPSHOT_PATH", str(tmp_path / "graph.pkl"))
    monkeypatch.setattr(api, "RESET_TOKEN", "test-token")
    monkeypatch.setattr(api, "_build_llm", lambda: lambda q: f"fresh answer to {q}")
    with TestClient(api.app) as c:
        yield c


def entries(client):
    return client.get("/stats").json()["entries"]


def test_starts_seeded_from_the_canonical_corpus(client):
    assert entries(client) == len(CORPUS)


def test_reset_removes_junk_and_restores_canonical(client):
    # Pollute the way a demo does: ask things that aren't cached, each of
    # which gets answered and stored.
    for junk in ["what is the best language", "write me a haiku", "weather today"]:
        assert client.post("/query", json={"query": junk}).json()["hit"] is False
    assert entries(client) == len(CORPUS) + 3

    body = client.post("/admin/reset", headers={"X-Reset-Token": "test-token"}).json()

    assert body["restored"] == len(CORPUS)
    assert entries(client) == len(CORPUS)

    # The junk is gone rather than merely uncounted: asking again misses.
    assert client.post("/query", json={"query": "what is the best language"}).json()["hit"] is False


def test_reset_keeps_the_corpus_answerable(client):
    client.post("/query", json={"query": "junk"})
    client.post("/admin/reset", headers={"X-Reset-Token": "test-token"})

    result = client.post("/query", json={"query": CORPUS[0][0]}).json()

    assert result["hit"] is True
    assert result["response"] == CORPUS[0][1]


def test_reset_deletes_the_snapshot_so_a_restart_cannot_undo_it(client, tmp_path):
    # The subtle failure this guards: reset clears storage and memory, but
    # leaves the on-disk graph. The next boot loads that stale snapshot and
    # silently restores everything the reset removed.
    snapshot = tmp_path / "graph.pkl"
    client.post("/query", json={"query": "junk that should not survive"})
    api.state["router"].save_snapshot(str(snapshot))
    assert snapshot.exists()

    body = client.post("/admin/reset", headers={"X-Reset-Token": "test-token"}).json()

    assert body["snapshot_deleted"] is True
    assert not snapshot.exists()


def test_reset_requires_the_token(client):
    client.post("/query", json={"query": "junk"})
    polluted = entries(client)

    assert client.post("/admin/reset").status_code == 401
    assert client.post("/admin/reset", headers={"X-Reset-Token": "wrong"}).status_code == 401
    assert entries(client) == polluted  # nothing happened


def test_reset_is_invisible_when_no_token_is_configured(client, monkeypatch):
    # Unset means the endpoint does not exist, rather than existing and
    # refusing. An API reachable from anywhere should not advertise a
    # cache-wipe endpoint it isn't willing to serve.
    monkeypatch.setattr(api, "RESET_TOKEN", None)
    assert client.post("/admin/reset", headers={"X-Reset-Token": "anything"}).status_code == 404


def test_reset_zeroes_the_hit_counters(client):
    client.post("/query", json={"query": CORPUS[0][0]})   # a hit
    client.post("/query", json={"query": "something new"})  # a miss
    assert client.get("/stats").json()["requests"] == 2

    client.post("/admin/reset", headers={"X-Reset-Token": "test-token"})

    stats = client.get("/stats").json()
    assert stats["requests"] == 0
    assert stats["hits"] == 0 and stats["misses"] == 0
