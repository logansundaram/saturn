"""
Ollama-locality boundary — a remote OLLAMA_HOST is network egress, never "local".

The local-inference story (posture line, /policy) keys on
egress.ollama_is_local(): when the Ollama endpoint is off-machine, chat models are wrapped in
the network boundary proxy (ledger-recorded), embeddings go through the embeddings
boundary, the air-gap refuses both, and egress._inference classifies the bindings
"remote" so no surface can claim the words were computed on this machine.
"""

import pytest

from trust import egress


# ── endpoint classification ─────────────────────────────────────────────────────────────────


def test_ollama_is_local_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert egress.ollama_is_local() is True
    assert egress.ollama_endpoint() == "http://127.0.0.1:11434"


@pytest.mark.parametrize(
    "host,expected",
    [
        ("http://127.0.0.1:11434", True),
        ("localhost:11434", True),
        ("127.0.0.1", True),
        ("0.0.0.0:11434", True),
        ("http://192.168.1.50:11434", False),
        ("gpu-box.local:11434", False),
        ("https://ollama.example.com", False),
    ],
)
def test_ollama_is_local_endpoint_forms(monkeypatch, host, expected):
    monkeypatch.setenv("OLLAMA_HOST", host)
    assert egress.ollama_is_local() is expected


# ── the one locality classifier ─────────────────────────────────────────────────────────────


def test_inference_classifies_remote_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.50:11434")
    from trust.egress import _inference

    inf = _inference()
    assert inf["all_local"] is False
    assert inf["remote_ollama"] == "http://192.168.1.50:11434"
    # No binding may read "local" when the daemon is off-machine.
    assert all(b["locality"] == "remote" for b in inf["bindings"])
    # The embedder runs through Ollama too, so it classifies remote with the rest.
    embedder = [b for b in inf["bindings"] if b["role"] == "embedder"]
    assert embedder and embedder[0]["locality"] == "remote"


def test_inference_no_remote_marker_on_loopback(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    from trust.egress import _inference

    inf = _inference()
    assert "remote_ollama" not in inf
    assert all(b["locality"] != "remote" for b in inf["bindings"])


# ── model factory boundary ──────────────────────────────────────────────────────────────────


def test_build_wraps_remote_ollama_only(monkeypatch):
    from core import llms

    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.50:11434")
    m = llms._build("qwen3.5:9b")
    assert isinstance(m, llms._NetworkBoundaryModel)
    assert "192.168.1.50" in m._host

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    m2 = llms._build("qwen3.5:9b")
    assert not isinstance(m2, llms._NetworkBoundaryModel)


def test_build_sends_keep_alive_from_config(monkeypatch):
    """runtime.keep_alive rides the client (default 30m): the daemon's 5-minute default unloads
    the weights between turns and the next turn pays the load again."""
    from core import llms
    from config import get_config

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    cfg = get_config()
    monkeypatch.setattr(cfg, "get", lambda key, default=None: {"runtime.keep_alive": "2h"}.get(key, default))
    assert cfg.keep_alive == "2h"
    assert llms._build("qwen3.5:9b").keep_alive == "2h"
    monkeypatch.setattr(cfg, "get", lambda key, default=None: {"runtime.keep_alive": -1}.get(key, default))
    assert llms._build("qwen3.5:9b").keep_alive == -1
    monkeypatch.setattr(cfg, "get", lambda key, default=None: {"runtime.keep_alive": None}.get(key, default))
    assert cfg.keep_alive is None
    assert llms._build("qwen3.5:9b").keep_alive is None   # the daemon's own default


def test_get_model_refuses_remote_ollama_under_airgap(monkeypatch, isolated_paths):
    from config import get_config
    from core import llms

    rt = get_config()._data.setdefault("runtime", {})
    monkeypatch.setitem(rt, "airgap", True)
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.50:11434")
    llms.reset_models()
    mark = egress.next_seq()
    try:
        with pytest.raises(RuntimeError, match="OLLAMA_HOST"):
            llms.get_model()
    finally:
        llms.reset_models()
    blocked = [e for e in egress.events_since(mark) if e.status == egress.BLOCKED]
    assert blocked and blocked[0].channel == "llm"
    assert "192.168.1.50" in blocked[0].host


# ── embeddings boundary ─────────────────────────────────────────────────────────────────────


class _FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(("docs", list(texts)))
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        self.calls.append(("query", text))
        return [0.0]


def test_embeddings_boundary_records_egress(monkeypatch, isolated_paths):
    from config import get_config
    from core.llms import _EmbeddingsBoundary

    rt = get_config()._data.setdefault("runtime", {})
    monkeypatch.setitem(rt, "airgap", False)
    inner = _FakeEmbedder()
    b = _EmbeddingsBoundary(inner, "qwen3-embedding:8b", "ollama @ http://192.168.1.50:11434")

    mark = egress.next_seq()
    b.embed_documents(["hello", "world"])
    evs = egress.events_since(mark)
    assert [e.channel for e in evs] == ["embedding"]
    assert evs[0].n_bytes == len("hello") + len("world")
    assert evs[0].status == egress.SENT
    assert inner.calls  # the embed actually ran after the record


def test_embeddings_boundary_refuses_under_airgap(monkeypatch, isolated_paths):
    from config import get_config
    from core.llms import _EmbeddingsBoundary

    rt = get_config()._data.setdefault("runtime", {})
    monkeypatch.setitem(rt, "airgap", True)
    inner = _FakeEmbedder()
    b = _EmbeddingsBoundary(inner, "qwen3-embedding:8b", "ollama @ http://192.168.1.50:11434")

    mark = egress.next_seq()
    with pytest.raises(RuntimeError, match="Air-gap"):
        b.embed_query("document text that must not leave")
    evs = egress.events_since(mark)
    assert evs and evs[0].status == egress.BLOCKED
    assert not inner.calls  # nothing crossed the boundary


def test_get_embeddings_unwrapped_on_loopback(monkeypatch):
    from core import llms

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert not isinstance(llms.get_embeddings(), llms._EmbeddingsBoundary)
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.50:11434")
    assert isinstance(llms.get_embeddings(), llms._EmbeddingsBoundary)


# ── network-boundary byte accounting ──────────────────────────────────────────────────────────


def test_boundary_records_the_bytes_it_sends(monkeypatch, isolated_paths):
    """The ledger records what crossed the boundary, and the boundary sends the messages
    unchanged (the redact mode that rewrote them was cut 2026-09-29)."""
    from langchain_core.messages import HumanMessage

    from config import get_config
    from core.llms import _NetworkBoundaryModel, _approx_bytes

    rt = get_config()._data.setdefault("runtime", {})
    monkeypatch.setitem(rt, "airgap", False)
    msgs = [HumanMessage(content="please use sk-ant-" + "a" * 60 + " for this")]

    b = _NetworkBoundaryModel(inner=object(), model="qwen3.5:9b", host="ollama @ http://10.0.0.5:11434")
    mark = egress.next_seq()
    assert b._outgoing(msgs) is msgs

    evs = egress.events_since(mark)
    assert [e.channel for e in evs] == ["llm"] and evs[0].n_bytes == _approx_bytes(msgs)


def test_posture_line_names_remote_endpoint(monkeypatch):
    from trust import receipt

    monkeypatch.setattr(
        egress,
        "_inference",
        lambda: {
            "all_local": False,
            "remote_ollama": "http://192.168.1.50:11434",
        },
    )
    spans = receipt.posture_spans()
    assert (
        "inference off-machine: ollama @ http://192.168.1.50:11434",
        "warn",
    ) in spans


# ── the network boundary covers every send path ───────────────────────────────────────────────


class _FakeResp:
    content = "ok"


class _FakeInner:
    def __init__(self):
        self.invoked = []

    def invoke(self, messages, *a, **k):
        self.invoked.append(messages)
        return _FakeResp()


def test_network_boundary_batch_routes_through_the_boundary():
    """batch() must cross the boundary one input at a time (each recorded) — the
    inner model's batch would take the whole list past it in one unobserved call."""
    from core.llms import _NetworkBoundaryModel

    inner = _FakeInner()
    wrapped = _NetworkBoundaryModel(inner, "m", host="remote:11434")
    out = wrapped.batch([[], []])
    assert len(out) == 2 and len(inner.invoked) == 2


def test_network_boundary_refuses_unguarded_send_paths():
    """__getattr__ used to hand generate/transform/… back bound to the INNER model — an
    unrecorded send. They fail closed now; benign attributes still delegate."""
    from core.llms import _NetworkBoundaryModel

    inner = _FakeInner()
    wrapped = _NetworkBoundaryModel(inner, "m", host="remote:11434")
    for name in ("generate", "agenerate", "transform", "abatch_as_completed"):
        with pytest.raises(AttributeError):
            getattr(wrapped, name)
    assert wrapped.invoked == []  # non-network attributes still delegate to the inner model


def test_embeddings_boundary_async_paths_gate_airgap(monkeypatch):
    """The Embeddings base-class async default runs against the INNER object, skipping the
    air-gap raise and the ledger — the explicit aembed_* overrides must gate first."""
    import asyncio

    from core import llms

    class _E:
        async def aembed_query(self, text):  # pragma: no cover — must never be reached
            raise AssertionError("the air-gap must block before the inner send")

    monkeypatch.setattr(egress, "airgap_on", lambda: True)
    boundary = llms._EmbeddingsBoundary(_E(), "emb", "remote:11434")
    with pytest.raises(RuntimeError, match="Air-gap"):
        asyncio.run(boundary.aembed_query("hello"))
