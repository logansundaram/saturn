"""
Trace replay (commands.trace.export_rows / render_export) and the /source citation drill-down
(commands.source.lookup_source) — the pure halves of both features. Plus the two stdout-honesty
guards: the attestation caption tracks the verify VERDICT (a tampered export never prints a
trust-affirming caption), and `/trace export -o` refuses a missing/flag-shaped path.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trust import signing
from commands.source import lookup_source
from commands.trace import _canonical_digest, _export, export_rows, render_export


# --- replay ----------------------------------------------------------------------------------

def _payload():
    return {
        "saturn_trace_export": 1,
        "saturn_version": "0.1.0",
        "exported_at": "2026-06-10T12:00:00",
        "run": {
            "run_id": 7, "query": "what is 2+2", "started_at": "2026-06-10T11:59:00",
            "ended_at": "2026-06-10T11:59:30", "status": "ok", "response": "4",
        },
        "events": [
            {"seq": 1, "ts": "t1", "node": "plan", "summary": "plan", "data": {"plan": []}},
            {"seq": 2, "ts": "t2", "node": "tools", "summary": "tools", "data": None},
        ],
        "llm_calls": [],
    }


def test_export_rows_shapes_match_show_run():
    run, rows = export_rows(_payload())
    assert run == (7, "what is 2+2", "2026-06-10T11:59:00", "2026-06-10T11:59:30", "ok", "4")
    assert len(rows) == 2
    seq, ts, node, summary, data = rows[0]
    assert (seq, node) == (1, "plan")
    assert json.loads(data) == {"plan": []}   # re-encoded so show_run's decode_json works
    assert rows[1][4] is None                 # None data stays None


def test_render_export_verifies_and_renders(tmp_path, capsys):
    payload = _payload()
    payload["integrity"] = {"algorithm": "sha256", "digest": _canonical_digest(_payload())}
    f = tmp_path / "run_7.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    assert render_export(str(f)) is True
    out = capsys.readouterr().out
    assert "integrity verified" in out
    assert "run #7" in out


def test_render_export_flags_tampering(tmp_path, capsys):
    payload = _payload()
    payload["integrity"] = {"algorithm": "sha256", "digest": _canonical_digest(_payload())}
    payload["run"]["response"] = "5"  # tamper after digest
    f = tmp_path / "run_7.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    assert render_export(str(f)) is True  # still renders, loudly
    captured = capsys.readouterr()
    # The failure banner is a DIAGNOSTIC: stderr, so a piped stdout stays the rendered run.
    assert "INTEGRITY FAILURE" in captured.err
    assert "INTEGRITY FAILURE" not in captured.out


def test_render_export_rejects_non_exports(tmp_path, capsys):
    f = tmp_path / "junk.json"
    f.write_text("{\"foo\": 1}", encoding="utf-8")
    assert render_export(str(f)) is False
    assert render_export(str(tmp_path / "missing.json")) is False


# --- the attestation caption tracks the verify VERDICT ----------------------------------------
# The integrity/signature failure banners go to STDERR; the attestation caption is STDOUT. A
# tampered export must therefore downgrade the caption itself — otherwise a piped transcript
# contains only the trust-affirming header over a record that just FAILED verification.

def _fake_signature_block():
    """A well-formed signature block that cannot verify (garbage key + signature). Deterministic
    with or without `cryptography` installed: verify_signature returns False either way."""
    return {
        "algorithm": "ed25519", "signed": "sha256-digest",
        "public_key": "ab" * 32, "key_id": "deadbeefdeadbeef", "signature": "ab" * 64,
    }


def _attested_payload():
    payload = _payload()
    payload["answer_attestation"] = {"answer": "4", "sources": [], "complete": True}
    return payload


def test_render_export_tampered_signed_caption_downgrades(tmp_path, capsys):
    """A TAMPERED signed export: stdout carries the UNVERIFIED downgrade, never 'committed by'."""
    payload = _attested_payload()
    payload["integrity"] = {"algorithm": "sha256", "digest": _canonical_digest(payload)}
    payload["signature"] = _fake_signature_block()
    payload["run"]["response"] = "5"  # tamper AFTER digest + signature
    f = tmp_path / "run_7.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    assert render_export(str(f)) is True  # render-anyway philosophy holds
    captured = capsys.readouterr()
    assert "INTEGRITY FAILURE" in captured.err
    assert "UNVERIFIED answer attestation" in captured.out
    assert "FAILED verification" in captured.out
    assert "do not trust these claims" in captured.out
    assert "committed by" not in captured.out  # neither trust-affirming caption printed


def test_render_export_invalid_signature_caption_downgrades(tmp_path, capsys):
    """Digest intact but the signature does not verify: still the downgrade, not 'signed'."""
    payload = _attested_payload()
    payload["integrity"] = {"algorithm": "sha256", "digest": _canonical_digest(payload)}
    payload["signature"] = _fake_signature_block()
    f = tmp_path / "run_7.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    assert render_export(str(f)) is True
    captured = capsys.readouterr()
    assert "UNVERIFIED answer attestation" in captured.out
    assert "committed by" not in captured.out


def test_render_export_digest_only_caption_unsigned(tmp_path, capsys):
    """An intact unsigned export keeps the digest-only caption — never the downgrade."""
    payload = _attested_payload()
    payload["integrity"] = {"algorithm": "sha256", "digest": _canonical_digest(payload)}
    f = tmp_path / "run_7.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    assert render_export(str(f)) is True
    out = capsys.readouterr().out
    assert "committed by this export's integrity digest; unsigned" in out
    assert "UNVERIFIED answer attestation" not in out


def test_render_export_signed_verified_caption(isolated_paths, tmp_path, capsys):
    """digest_ok AND signature_ok → the signed caption (a real signature from a tmp-minted key)."""
    if not signing.available():
        pytest.skip("cryptography not installed — signing unavailable")
    payload = _attested_payload()
    digest = _canonical_digest(payload)
    payload["integrity"] = {"algorithm": "sha256", "digest": digest}
    payload["signature"] = signing.sign_digest(digest)
    assert payload["signature"]  # the tmp key minted + signed
    f = tmp_path / "run_7.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    assert render_export(str(f)) is True
    out = capsys.readouterr().out
    assert "signed answer attestation — committed by this export's digest + signature" in out
    assert "UNVERIFIED answer attestation" not in out


# --- /trace export -o argument guard -----------------------------------------------------------
# A dangling -o used to silently write the default path; `-o --md` wrote a JSON file literally
# named '--md'. Both now refuse with a usage error BEFORE any DB/file work.

def test_export_dash_o_missing_path_refuses(tmp_path, capsys):
    ctx = SimpleNamespace(db_path=str(tmp_path / "none.sqlite"))
    _export(ctx, ["-o"])
    out = capsys.readouterr().out
    assert "needs a path" in out and "nothing written" in out
    assert not (tmp_path / "none.sqlite").exists()  # refused before touching the DB


def test_export_dash_o_flag_shaped_value_refuses(tmp_path, capsys):
    ctx = SimpleNamespace(db_path=str(tmp_path / "none.sqlite"))
    _export(ctx, ["-o", "--md"])
    out = capsys.readouterr().out
    assert "needs a path" in out and "nothing written" in out
    assert not Path("--md").exists()                # the literally-named file is never written
    assert not (tmp_path / "none.sqlite").exists()


# --- /source ---------------------------------------------------------------------------------

_STATE = {
    "tool_results": ["web_search(query='x') -> first full result text"],
    "documents_retrieved": ["[source: notes.md] full passage text"],
}


def test_lookup_source_numbers_match_build_sources():
    label, text = lookup_source(_STATE, 1)
    assert label.startswith("web_search(")
    assert text == "web_search(query='x') -> first full result text"
    label2, text2 = lookup_source(_STATE, 2)
    assert "notes.md" in label2
    assert text2 == "[source: notes.md] full passage text"


def test_lookup_source_out_of_range():
    assert lookup_source(_STATE, 0) is None
    assert lookup_source(_STATE, 3) is None
    assert lookup_source({}, 1) is None
