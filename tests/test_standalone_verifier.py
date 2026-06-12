"""
The zero-Saturn verifier (utilities/saturn_verify.py) — the spec's reference implementation.

Loaded IN-PROCESS via importlib (it is deliberately not a package member) and run against REAL
artifacts built by commands.trace.export_run: the verifier must agree with Saturn
byte-for-byte — same canonicalization, same digest, same ed25519 layering (including the
vendored pure-Python verify path), same egress-chain walk — while importing nothing from
Saturn. The tail-truncation test is the whole point of the egress anchor: a clean truncation
verifies as an intact chain on its own, and only the signed artifact's anchored tip exposes it.
"""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from trust import egress
from trust import signing
from commands import trace as trace_cmd
from config import get_config

_ROOT = Path(__file__).resolve().parents[1]
_VERIFIER = _ROOT / "utilities" / "saturn_verify.py"

_spec = importlib.util.spec_from_file_location("saturn_verify", _VERIFIER)
sv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sv)

needs_crypto = pytest.mark.skipif(
    not signing.available(), reason="cryptography not installed — no signature to cross-check"
)


def _seed_run_db(tmp_path) -> str:
    db = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (run_id INTEGER PRIMARY KEY, query TEXT, started_at TEXT,
                           ended_at TEXT, status TEXT, response TEXT);
        CREATE TABLE events (id INTEGER PRIMARY KEY, run_id INTEGER, seq INTEGER, ts TEXT,
                             node TEXT, summary TEXT, data TEXT);
        INSERT INTO runs VALUES (1, 'hello', '2026-06-11T00:00:00', '2026-06-11T00:00:01',
                                 'ok', 'hi — ünïcode there');
        INSERT INTO events VALUES (1, 1, 1, '2026-06-11T00:00:00', 'plan', 'planned', NULL);
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def test_verifier_imports_nothing_from_saturn():
    # The moat claim is third-party-verifiable WITHOUT Saturn — the file must stay standalone.
    src = _VERIFIER.read_text(encoding="utf-8")
    for token in ("import signing", "from signing", "import egress", "from egress",
                  "import config", "from config", "import diag", "import glassbox",
                  "import trust_report"):
        assert token not in src, f"saturn_verify.py must not contain {token!r}"


def test_canonical_scheme_matches_saturn():
    # The verifier TRANSCRIBES the canonical scheme — it must agree with signing.py on every
    # byte, including unicode (ensure_ascii=False) and separators.
    payload = {"b": [1, 2, {"c": "é — ünïcode ✓"}], "a": "x", "n": 3.5, "t": True, "z": None}
    assert sv.canonical_json(payload) == signing.canonical_json(payload)
    assert sv.canonical_digest(payload) == signing.canonical_digest(payload)


# --- artifact mode --------------------------------------------------------------------------


@needs_crypto
def test_intact_signed_export_exits_0(isolated_paths, tmp_path, capsys):
    db = _seed_run_db(tmp_path)
    dest = tmp_path / "run_1.json"
    trace_cmd.export_run(db, 1, dest=dest)
    rc = sv.main([str(dest)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "digest ok" in out and "signature valid" in out


def test_byte_flip_exits_1(isolated_paths, tmp_path, capsys):
    db = _seed_run_db(tmp_path)
    dest = tmp_path / "run_1.json"
    trace_cmd.export_run(db, 1, dest=dest)
    payload = json.loads(dest.read_text(encoding="utf-8"))
    payload["run"]["response"] = "tampered"
    dest.write_text(json.dumps(payload), encoding="utf-8")
    assert sv.main([str(dest)]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_attestation_field_tamper_exits_1(isolated_paths, tmp_path, capsys):
    # The answer attestation rides INSIDE the signed body — editing it must break the digest.
    db = _seed_run_db(tmp_path)
    dest = tmp_path / "run_1.json"
    _, payload = trace_cmd.export_run(db, 1, dest=dest)
    assert isinstance(payload["answer_attestation"], dict)
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    on_disk["answer_attestation"]["forged_claim"] = "all local, honest"
    dest.write_text(json.dumps(on_disk), encoding="utf-8")
    assert sv.main([str(dest)]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_unsigned_export_passes_with_unsigned_verdict(isolated_paths, tmp_path, capsys):
    db = _seed_run_db(tmp_path)
    dest = tmp_path / "run_1.json"
    get_config().set("runtime.sign_exports", False)
    try:
        trace_cmd.export_run(db, 1, dest=dest)
    finally:
        get_config().set("runtime.sign_exports", True)
    rc = sv.main([str(dest)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "unsigned" in out  # intact-but-unsigned passes, PRINTED as such — never overclaimed


def test_bom_tolerated(isolated_paths, tmp_path):
    db = _seed_run_db(tmp_path)
    dest = tmp_path / "run_1.json"
    trace_cmd.export_run(db, 1, dest=dest)
    dest.write_text(dest.read_text(encoding="utf-8"), encoding="utf-8-sig")  # add a BOM
    assert sv.main([str(dest)]) == 0


def test_not_an_artifact_and_unreadable_exit_2(tmp_path, capsys):
    bad = tmp_path / "notes.json"
    bad.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    assert sv.main([str(bad)]) == 2
    assert sv.main([str(tmp_path / "missing.json")]) == 2
    capsys.readouterr()


def test_usage_errors_exit_2(tmp_path, capsys):
    assert sv.main([]) == 2
    assert sv.main(["--expect-tip", "abc"]) == 2
    art = tmp_path / "a.json"
    art.write_text("{}", encoding="utf-8")
    assert sv.main([str(art), "--egress-log", str(art)]) == 2
    capsys.readouterr()


# --- the vendored pure-Python ed25519 path ----------------------------------------------------


@needs_crypto
def test_pure_python_ed25519_cross_checks_saturn_signature(isolated_paths, monkeypatch):
    # Force the vendored RFC 8032 path and check it against a signature signing.py produced —
    # the two implementations must agree on valid, wrong-message, and corrupted signatures.
    digest = signing.canonical_digest({"x": 1})
    block = signing.sign_digest(digest)
    monkeypatch.setattr(sv, "_try_cryptography", lambda *a, **k: None)
    pub = bytes.fromhex(block["public_key"])
    sig = bytes.fromhex(block["signature"])
    msg = digest.encode("utf-8")
    assert sv.ed25519_verify(pub, msg, sig) is True
    assert sv.ed25519_verify(pub, b"a different message", sig) is False
    forged = bytearray(sig)
    forged[0] ^= 0xFF
    assert sv.ed25519_verify(pub, msg, bytes(forged)) is False
    assert sv.ed25519_verify(pub, msg, sig[:63]) is False  # bad length → False, never a crash


@needs_crypto
def test_full_artifact_verifies_on_pure_python_path(isolated_paths, tmp_path, monkeypatch, capsys):
    db = _seed_run_db(tmp_path)
    dest = tmp_path / "run_1.json"
    trace_cmd.export_run(db, 1, dest=dest)
    monkeypatch.setattr(sv, "_try_cryptography", lambda *a, **k: None)
    assert sv.main([str(dest)]) == 0
    assert "signature valid" in capsys.readouterr().out


# --- egress-chain mode -------------------------------------------------------------------------


def test_chain_intact_exits_0(isolated_paths, capsys):
    egress.clear()
    for i in range(4):
        egress.record("web_search", f"h{i}.example.com")
    log = isolated_paths / "database" / "egress.log"
    assert sv.main(["--egress-log", str(log)]) == 0
    assert "chain intact" in capsys.readouterr().out
    # The standalone walk and Saturn's agree.
    assert egress.verify_log()["ok"] is True


def test_chain_middle_edit_exits_1(isolated_paths, capsys):
    egress.clear()
    for i in range(4):
        egress.record("web_search", f"h{i}.example.com")
    log = isolated_paths / "database" / "egress.log"
    lines = log.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["host"] = "evil.example.com"
    lines[1] = json.dumps(row)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert sv.main(["--egress-log", str(log)]) == 1
    assert "BROKEN at line 2" in capsys.readouterr().out
    assert egress.verify_log()["ok"] is False  # Saturn's walk agrees


def test_garbled_line_breaks_chain(isolated_paths, capsys):
    egress.clear()
    egress.record("web_search", "a.example.com")
    log = isolated_paths / "database" / "egress.log"
    with open(log, "ab") as fh:
        fh.write(b"garbage not json\n")
    assert sv.main(["--egress-log", str(log)]) == 1
    assert "unparseable" in capsys.readouterr().out


def test_tail_truncation_caught_by_signed_anchor(isolated_paths, tmp_path, capsys):
    # THE anchor scenario end to end: a clean tail truncation still verifies as an intact
    # chain on its own — only the tip a signed export committed exposes it.
    egress.clear()
    for i in range(3):
        egress.record("web_search", f"h{i}.example.com")
    db = _seed_run_db(tmp_path)
    dest = tmp_path / "run_1.json"
    _, payload = trace_cmd.export_run(db, 1, dest=dest)
    anchor = payload["egress_anchor"]
    assert anchor["tip_hash"] == egress.read_log()[-1]["h"]
    assert anchor["line_count"] == 3
    log = isolated_paths / "database" / "egress.log"

    # Untruncated: the chain reaches the anchored tip.
    assert sv.main(["--egress-log", str(log), "--expect-tip", anchor["tip_hash"]]) == 0
    capsys.readouterr()

    # Clean truncation: drop the last line. Alone, the shortened chain STILL verifies…
    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert sv.main(["--egress-log", str(log)]) == 0
    capsys.readouterr()
    # …but the anchored tip is gone — exactly the tamper the anchor exists to catch.
    assert sv.main(["--egress-log", str(log), "--expect-tip", anchor["tip_hash"]]) == 1
    assert "NOT reached" in capsys.readouterr().out


def test_export_anchor_absent_without_log(isolated_paths, tmp_path):
    # No durable log in this fresh tree → no egress_anchor field, never a fake value.
    db = _seed_run_db(tmp_path)
    _, payload = trace_cmd.export_run(db, 1, dest=tmp_path / "r.json")
    assert "egress_anchor" not in payload
    assert payload["format"] == signing.ARTIFACT_FORMAT
