"""The 2026-07-04 consolidation pass: shared helpers that replaced per-site copies, and the
network-boundary hardening. One test per drift hazard so a re-rolled copy or a boundary bypass
fails loudly instead of drifting silently."""

import asyncio

import pytest


# ── textutil.mask_secret — THE one masking rule ───────────────────────────────────────────────
def test_mask_secret_one_envelope():
    from textutil import mask_secret

    assert mask_secret("short") == "****"  # ≤8 chars: show nothing at all
    long = "sk-abcdefghijklmnop"
    m = mask_secret(long)
    assert long not in m and "…" in m
    assert m.startswith(long[:4]) and m.endswith(long[-2:])
    assert mask_secret("") == "" and mask_secret(None) == ""


def test_both_mask_surfaces_delegate_to_it():
    """env_keys' key listing and trust/redaction's findings once had different exposure
    envelopes (3+4 vs 6+2 visible chars); both must render THE one rule now."""
    import env_keys
    from textutil import mask_secret
    from trust import redaction

    secret = "tvly-0123456789abcdef"
    assert mask_secret(secret) in env_keys.mask(secret)
    assert redaction._mask(secret) == mask_secret(secret)
    # Short secrets show nothing on either surface.
    assert env_keys.mask("tiny") == "****"
    assert redaction._mask("tiny") == "****"


# ── the [source: …] provenance-marker pair ────────────────────────────────────────────────────
def test_doc_source_marker_round_trip():
    """tools/knowledge builds the marker, synthesize parses it back — one builder + one parser
    (the CALL_RESULT_SEP treatment), so the round trip must be lossless and dedupe in order."""
    from textutil import doc_source_label, parse_doc_sources

    obs = (
        doc_source_label("a.md") + "\nchunk one\n\n"
        + doc_source_label("b.pdf", 3) + "\nchunk two\n\n"
        + doc_source_label("a.md") + "\nchunk three"
    )
    assert parse_doc_sources(obs) == ["a.md", "b.pdf, page 3"]
    assert parse_doc_sources("no markers here") == []
    assert doc_source_label(None) == "[source: unknown]"


# ── /trace run-selector grammar (was five drifting hand-rolled copies) ────────────────────────
def test_parse_run_selector_grammar(capsys):
    from commands.trace import _parse_run_selector

    assert _parse_run_selector(["#7"]) == (7, None, False)
    assert _parse_run_selector(["-r", "9"]) == (9, None, False)
    assert _parse_run_selector(["12"]) == (12, None, False)  # bare digits are RUN IDS…
    assert _parse_run_selector(["-l", "20"]) == (None, 20, True)  # …except as the list COUNT
    assert _parse_run_selector(["ls"]) == (None, None, True)
    assert _parse_run_selector([]) == (None, None, False)
    assert _parse_run_selector(["garbage"]) == (None, None, False)
    assert "ignoring" in capsys.readouterr().out


def test_parse_run_selector_consume_hook():
    from commands.trace import _parse_run_selector

    seen = {}

    def consume(low, a, it):
        if low == "--md":
            seen["md"] = True
            return True
        return False

    assert _parse_run_selector(["--md", "#3"], consume=consume) == (3, None, False)
    assert seen == {"md": True}


# ── the network boundary covers every send path ───────────────────────────────────────────────
class _FakeResp:
    content = "ok"


class _FakeInner:
    def __init__(self):
        self.invoked = []

    def invoke(self, messages, *a, **k):
        self.invoked.append(messages)
        return _FakeResp()


def test_cloud_boundary_batch_routes_through_the_boundary():
    """batch() must cross the boundary one input at a time (each redacted + recorded) — the
    inner model's batch would take the whole list past it in one unobserved call."""
    from core.llms import _CloudBoundaryModel

    inner = _FakeInner()
    wrapped = _CloudBoundaryModel(inner, "ollama", "m", host="remote:11434")
    out = wrapped.batch([[], []])
    assert len(out) == 2 and len(inner.invoked) == 2


def test_cloud_boundary_refuses_unguarded_send_paths():
    """__getattr__ used to hand generate/transform/… back bound to the INNER model — an
    unredacted, unrecorded send. They fail closed now; benign attributes still delegate."""
    from core.llms import _CloudBoundaryModel

    inner = _FakeInner()
    wrapped = _CloudBoundaryModel(inner, "ollama", "m", host="remote:11434")
    for name in ("generate", "agenerate", "transform", "abatch_as_completed"):
        with pytest.raises(AttributeError):
            getattr(wrapped, name)
    assert wrapped.invoked == []  # non-network attributes still delegate to the inner model


def test_embeddings_boundary_async_paths_gate_airgap(monkeypatch):
    """The Embeddings base-class async default runs against the INNER object, skipping the
    air-gap raise and the ledger — the explicit aembed_* overrides must gate first."""
    from core import llms
    from trust import egress

    class _E:
        async def aembed_query(self, text):  # pragma: no cover — must never be reached
            raise AssertionError("the air-gap must block before the inner send")

    monkeypatch.setattr(egress, "airgap_on", lambda: True)
    boundary = llms._EmbeddingsBoundary(_E(), "emb", "remote:11434")
    with pytest.raises(RuntimeError, match="Air-gap"):
        asyncio.run(boundary.aembed_query("hello"))


# ── document_registry's mtime-validated memo ──────────────────────────────────────────────────
def test_manifest_memo_is_mtime_honest(isolated_paths, monkeypatch):
    """The manifest memo saves the per-file re-reads but must never trust itself blindly: a
    hand-edited manifest (new mtime) is re-read, not served stale from memory."""
    import time

    from stores import document_registry as dr

    monkeypatch.setattr(dr, "_summarize", lambda content, filename: "a summary")
    dr.register_workspace_file("a.txt", "hello")
    text1 = dr.read_workspace_manifest()
    assert "### a.txt" in text1

    time.sleep(0.01)  # ensure a distinct mtime on coarse filesystems
    p = dr._workspace_manifest()
    p.write_text(text1 + "\n### hand-added\nmanual entry\n", encoding="utf-8")
    assert "hand-added" in dr.read_workspace_manifest()
