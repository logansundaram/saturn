"""
Ollama-locality boundary — a remote OLLAMA_HOST is network egress, never "local".

The local-inference story (posture line, Glass Box attestation, /privacy) keys on
egress.ollama_is_local(): when the Ollama endpoint is off-machine, chat models are wrapped in
the cloud boundary proxy (redacted + ledger-recorded), embeddings go through the embeddings
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
    assert all(b["locality"] in ("remote", "cloud") for b in inf["bindings"])
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
    m = llms._build("ollama", "qwen3.5:9b")
    assert isinstance(m, llms._CloudBoundaryModel)
    assert "192.168.1.50" in m._host

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    m2 = llms._build("ollama", "qwen3.5:9b")
    assert not isinstance(m2, llms._CloudBoundaryModel)


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
            llms.get_model("planner")
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


# ── cloud-boundary byte accounting ──────────────────────────────────────────────────────────


def test_boundary_records_post_redaction_bytes(monkeypatch, isolated_paths):
    """The ledger must record what actually crossed the boundary: in redact mode that is the
    redacted copy, smaller than the original by exactly the stripped secret."""
    from langchain_core.messages import HumanMessage

    from config import get_config
    from core.llms import _CloudBoundaryModel, _approx_bytes

    rt = get_config()._data.setdefault("runtime", {})
    monkeypatch.setitem(rt, "airgap", False)
    monkeypatch.setitem(rt, "redaction", "redact")
    secret = "sk-ant-" + "a" * 60
    msgs = [HumanMessage(content=f"please use {secret} for this")]

    b = _CloudBoundaryModel(inner=object(), provider="anthropic", model="claude-x")
    mark = egress.next_seq()
    to_send = b._outgoing(msgs)

    evs = egress.events_since(mark)
    assert evs and evs[0].redactions == 1
    assert evs[0].n_bytes == _approx_bytes(to_send)
    assert evs[0].n_bytes < _approx_bytes(msgs)  # the secret never counted as "sent"


# ── posture surface ─────────────────────────────────────────────────────────────────────────


def test_posture_line_names_remote_endpoint(monkeypatch):
    from trust import receipt

    monkeypatch.setattr(
        egress,
        "_inference",
        lambda: {
            "all_local": False,
            "cloud_providers": [],
            "remote_ollama": "http://192.168.1.50:11434",
        },
    )
    spans = receipt.posture_spans()
    assert (
        "inference off-machine: ollama @ http://192.168.1.50:11434",
        "warn",
    ) in spans
