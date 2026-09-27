"""Review fixes 2026-08-21 — one section per finding, each pinning the failure the review
reproduced on the perf/consolidation working-tree wave (see documentation.md's changelog entry).
"""

import math
import types

import pytest


# ── plan editor: no-tool markers mean a reasoning step (parity with /draft) ───────────────────
# resolve_tool claimed parity with /draft's normalization but skipped the _NO_TOOL_MARKERS half,
# so `add Summarize the findings ::none` minted intended_tool="none" — an unknown-tool ERROR
# incident at execute — while the identical spelling in /draft made a genuine reasoning step.


# ── confidence: grade_start walks to a real run boundary (not a fixed margin) ─────────────────
# A run OPENS on _MIN_RUN tokens under enter but EXTENDS indefinitely through hysteresis, so a
# fixed entry margin dropped the red tail of any run longer than the margin: the live tail and
# the final render disagreed about the model's most uncertain passage.


def _entries(probs):
    return [
        {"start": 2 * i, "end": 2 * i + 2, "logprob": math.log(p)}
        for i, p in enumerate(probs)
    ]


def test_grade_start_covers_a_long_hysteresis_run():
    from core import confidence

    n = 600
    text = "qx" * n
    probs = [0.05] * 3 + [0.25] * (n - 3)  # opens on 3 enter-low, extends via hysteresis only
    entries = _entries(probs)
    full = confidence.low_runs(entries, text, threshold_p=0.2, exit_p=0.3)
    assert full == [(0, 2 * n)]

    pos = 2 * (n - 50)  # the visible window starts far past any fixed margin from the opening
    start = confidence.grade_start(entries, text, pos, threshold_p=0.2, exit_p=0.3)
    assert start == 0  # no breaker between the run's opening and the window
    assert confidence.low_runs(entries[start:], text, threshold_p=0.2, exit_p=0.3) == full


def test_grade_start_stops_at_a_confident_breaker():
    from core import confidence

    n = 20
    text = "qx" * n
    probs = [0.9] * 10 + [0.05] * 10  # a confident stretch closes every possible run
    entries = _entries(probs)
    start = confidence.grade_start(entries, text, 2 * 15, threshold_p=0.2, exit_p=0.3)
    assert start == 10  # just past the last confident content token
    assert confidence.low_runs(entries[start:], text, threshold_p=0.2, exit_p=0.3) == [
        (20, 40)
    ]


def test_grade_start_tolerates_empty_and_garbage():
    from core import confidence

    assert confidence.grade_start([], "", 0, threshold_p=0.2, exit_p=0.3) == 0
    junk = [{"start": 0, "end": 2, "logprob": "?"}, {"start": 2, "end": 4, "logprob": -3.0}]
    # The malformed entry is a safe boundary (low_runs closes every run there).
    assert confidence.grade_start(junk, "qxqx", 3, threshold_p=0.2, exit_p=0.3) == 1


# ── rag: the racy-clean guard on the stat fast path ───────────────────────────────────────────
# Size+mtime alone would skip a file edited within the same mtime tick as its verification (same
# size, coarse-timestamp filesystem) FOREVER — search_knowledge_base kept citing stale vectors.
# The stat is trusted only once the file's mtime is a full coarse tick older than the recorded
# verification moment; a legacy entry (no indexed_at_ns) always re-hashes.


def test_unchanged_requires_an_aged_verification():
    from stores import rag

    mtime = 1_000_000_000_000_000_000
    stat = {"size": 10, "mtime_ns": mtime}
    legacy = {"hash": "h", "size": 10, "mtime_ns": mtime}  # pre-guard entry: no indexed_at_ns
    assert rag._unchanged(legacy, stat) is False
    racy = {**legacy, "indexed_at_ns": mtime + 1}  # verified inside the coarse tick
    assert rag._unchanged(racy, stat) is False
    aged = {**legacy, "indexed_at_ns": mtime + rag._RACY_WINDOW_NS}
    assert rag._unchanged(aged, stat) is True
    assert rag._unchanged(aged, {"size": 11, "mtime_ns": mtime}) is False
    assert rag._unchanged(aged, None) is False
    assert rag._unchanged(None, stat) is False


# ── glassbox: a record truncated mid-footer still has the footer stripped ─────────────────────
# end_run's write-time cap appends its truncation marker AFTER the cut, which fails the strict
# shared parser; the per-source answer analysis must never run over source labels, so stripping
# falls back to the lenient trailing strip in exactly that case.


def test_strip_footer_strict_shape():
    from trust import glassbox

    ans = "The answer.\n\nSources:\n  [1] read_file(x)"
    assert glassbox._strip_footer(ans) == "The answer."


def test_strip_footer_truncated_record_still_stripped():
    from trust import glassbox

    ans = ("The answer.\n\nSources:\n  [1] read_file(x)\n"
           "… [recorded answer truncated at 4000 chars]")
    assert glassbox._strip_footer(ans) == "The answer."


def test_strip_footer_prose_mention_is_not_a_footer():
    from trust import glassbox

    ans = "Discussing the Sources: section of a paper is fun."
    assert glassbox._strip_footer(ans) == ans


# ── approval: a persist grant that could not be persisted is logged as session ────────────────
# The audit entry used to be written BEFORE set_risk_override could fail, so grant_log claimed a
# durable grant while the next process started from the declared tier.


def test_failed_persist_grant_logged_as_session(isolated_paths, monkeypatch):
    from config import get_config
    from nodes.approval import _apply_always_grants
    from trust import policy

    monkeypatch.setitem(get_config()._data["runtime"], "grant_scope", "persist")
    fake = types.SimpleNamespace(TOOL_RISK={"write_file": "side_effecting"})
    monkeypatch.setattr("tools.registry", fake, raising=False)

    def boom(_name, _tier):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(policy, "set_risk_override", boom)
    _apply_always_grants({"approved": True, "tools": ["write_file"], "shell_grants": []})
    assert fake.TOOL_RISK["write_file"] == "read_only"  # the live drop stands
    last = policy.grant_log()[-1]
    assert last["event"] == "grant" and last["tool"] == "write_file"
    assert last["scope"] == "session"  # the lifetime it actually got — never claimed durable


def test_successful_persist_grant_logged_as_persist(isolated_paths, monkeypatch):
    from config import get_config
    from nodes.approval import _apply_always_grants
    from trust import policy

    monkeypatch.setitem(get_config()._data["runtime"], "grant_scope", "persist")
    fake = types.SimpleNamespace(TOOL_RISK={"write_file": "side_effecting"})
    monkeypatch.setattr("tools.registry", fake, raising=False)
    _apply_always_grants({"approved": True, "tools": ["write_file"], "shell_grants": []})
    last = policy.grant_log()[-1]
    assert last["event"] == "grant" and last["scope"] == "persist"
    assert policy.risk_overrides() == {"write_file": "read_only"}


# ── egress: check_or_raise is a view of check(), and the refusal reaches the ledger ───────────
# The raising twin used to re-implement check()'s body, so a future rung added inside check()
# (the queued egress budget) would have silently bypassed every LLM/embedder/continuation exit.


def test_check_or_raise_records_blocked_and_raises(isolated_paths, monkeypatch):
    from trust import egress

    monkeypatch.setattr(egress, "airgap_on", lambda: True)
    before = len(egress.events())
    with pytest.raises(RuntimeError, match="air-gap"):
        egress.check_or_raise("llm", "ollama @ http://10.0.0.5:11434", "planner → m",
                              provider="ollama", subject="role 'planner' (m)")
    evs = egress.events()
    assert len(evs) == before + 1
    assert evs[-1].status == egress.BLOCKED
    assert evs[-1].provider == "ollama"


def test_check_or_raise_delegates_to_check(monkeypatch):
    from trust import egress

    seen = {}

    def fake_check(channel, host, detail="", *, provider=""):
        seen["args"] = (channel, host, detail, provider)
        return None

    monkeypatch.setattr(egress, "check", fake_check)
    egress.check_or_raise("llm", "h", "d", provider="p")  # allowed: returns without raising
    assert seen["args"] == ("llm", "h", "d", "p")


# ── tui/ui/trace.py: nothing shadows the module-level textutil.clip import ────────────────────
# Two locals named `clip` survived the rename that made room for the import; a future call to
# clip() inside those bodies would raise TypeError only on the --preview/non-full path.


def test_trace_module_does_not_shadow_textutil_clip():
    import inspect
    import re

    from tui.ui import trace as trace_mod

    src = inspect.getsource(trace_mod)
    assert not re.search(r"^\s+clip\s*=", src, re.M)
