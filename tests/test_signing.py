"""
Signed audit exports (signing.py) — the provenance layer on top of the export digest.

Covers: keypair mint + persistence, the sign/verify roundtrip, tamper + forgery detection,
signer-identity matching, graceful degradation when `cryptography` is absent, and the full
/trace export -> /trace verify path producing a record whose digest AND signature both verify.
"""

import json

import pytest

from trust import signing


def test_available_when_cryptography_installed():
    # The dev/test environment ships cryptography (a hard requirement nowhere, but present here).
    assert signing.available() is True


def test_keypair_minted_and_persisted(isolated_paths):
    pub1 = signing.public_key_hex()
    assert pub1 and len(bytes.fromhex(pub1)) == 32  # raw ed25519 public key is 32 bytes
    key_file = isolated_paths / "database" / "signing_key.json"
    assert key_file.exists()
    doc = json.loads(key_file.read_text(encoding="utf-8"))
    assert doc["algorithm"] == "ed25519"
    assert "private_key" in doc and "public_key" in doc
    # Stable across calls — no re-mint once the file exists.
    assert signing.public_key_hex() == pub1
    assert signing.fingerprint() == doc["key_id"]


def test_sign_verify_roundtrip(isolated_paths):
    digest = "a" * 64
    block = signing.sign_digest(digest)
    assert block and block["algorithm"] == "ed25519"
    assert signing.verify_signature(digest, block) is True
    assert signing.signer_matches_local(block) is True


def test_tampered_digest_fails_verification(isolated_paths):
    block = signing.sign_digest("a" * 64)
    # A different digest (content changed) must not verify against the same signature.
    assert signing.verify_signature("b" * 64, block) is False


def test_forged_signature_fails(isolated_paths):
    block = signing.sign_digest("a" * 64)
    forged = dict(block)
    forged["signature"] = ("0" * len(block["signature"]))
    assert signing.verify_signature("a" * 64, forged) is False


def test_signer_matches_local_false_for_other_key(isolated_paths):
    block = signing.sign_digest("a" * 64)
    other = dict(block)
    other["public_key"] = "f" * 64  # a different (well-formed) public key
    assert signing.signer_matches_local(other) is False
    # And a signature whose public key doesn't match its signature bytes won't verify.
    assert signing.verify_signature("a" * 64, other) is False


def test_graceful_without_cryptography(isolated_paths, monkeypatch):
    monkeypatch.setattr(signing, "_crypto", lambda: None)
    assert signing.available() is False
    assert signing.sign_digest("a" * 64) is None
    assert signing.verify_signature("a" * 64, {"algorithm": "ed25519"}) is False
    assert signing.key_info() == {"available": False}


def test_verify_payload_roundtrip_and_tamper(isolated_paths):
    # THE shared verify flow: pops integrity+signature off a COPY before recomputing the digest.
    payload = {"saturn_trace_export": 1, "run": {"run_id": 1, "response": "hi"}}
    digest = signing.canonical_digest(payload)
    payload["integrity"] = {"algorithm": "sha256", "digest": digest}
    payload["signature"] = signing.sign_digest(digest)

    v = signing.verify_payload(payload)
    assert v["has_integrity"] and v["digest_ok"]
    assert v["signed"] and v["signature_ok"] and v["signer_is_local"]
    assert "integrity" in payload and "signature" in payload  # the input is not mutated

    tampered = dict(payload)
    tampered["run"] = {"run_id": 1, "response": "edited"}
    v2 = signing.verify_payload(tampered)
    assert v2["digest_ok"] is False
    assert v2["signature_ok"] is True  # the signature still covers the STORED digest

    assert signing.verify_payload({"no": "blocks"})["has_integrity"] is False


def test_key_loaded_once_per_path(isolated_paths, monkeypatch):
    # The keypair is immutable for the process lifetime: after the first load, signing and
    # fingerprinting must not re-read the key file.
    signing.public_key_hex()  # mint + cache
    reads = []
    real_read = type(isolated_paths).read_text

    def counting_read(self, *a, **k):
        if self.name == "signing_key.json":
            reads.append(self)
        return real_read(self, *a, **k)

    monkeypatch.setattr(type(isolated_paths), "read_text", counting_read)
    signing.sign_digest("a" * 64)
    signing.fingerprint()
    signing.public_key_hex()
    assert reads == []


def test_artifact_format_marker_roundtrip_and_markerless_compat(isolated_paths):
    # The format marker pins the standalone verify spec (utilities/VERIFY_SPEC.md); verify_payload
    # must accept BOTH a marked and an old marker-less artifact — the digest simply covers
    # whatever body is there (backward compatible by construction).
    assert signing.ARTIFACT_FORMAT == "saturn-artifact/1"

    marked = {"saturn_trace_export": 1, "format": signing.ARTIFACT_FORMAT, "run": {"x": 1}}
    digest = signing.canonical_digest(marked)
    marked["integrity"] = {"algorithm": "sha256", "digest": digest}
    marked["signature"] = signing.sign_digest(digest)
    v = signing.verify_payload(marked)
    assert v["digest_ok"] and v["signature_ok"]

    legacy = {"saturn_trace_export": 1, "run": {"x": 1}}  # pre-marker artifact (before 2026-06-11)
    d2 = signing.canonical_digest(legacy)
    legacy["integrity"] = {"algorithm": "sha256", "digest": d2}
    assert signing.verify_payload(legacy)["digest_ok"] is True


def _seed_export_db(root):
    import sqlite3

    db_path = root / "database" / "db.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs (run_id INTEGER PRIMARY KEY, query TEXT, started_at TEXT,
                           ended_at TEXT, status TEXT, response TEXT);
        CREATE TABLE events (id INTEGER PRIMARY KEY, run_id INTEGER, seq INTEGER, ts TEXT,
                             node TEXT, summary TEXT, data TEXT);
        INSERT INTO runs VALUES (1, 'hello', '2026-06-10T00:00:00', '2026-06-10T00:00:01',
                                 'ok', 'hi there');
        INSERT INTO events VALUES (1, 1, 1, '2026-06-10T00:00:00', 'plan', 'planned', NULL);
        """
    )
    conn.commit()
    conn.close()
    return str(db_path)


def test_export_warns_when_signing_fails(isolated_paths, capsys, monkeypatch):
    # runtime.sign_exports on AND signing.available() True but the signature came back None
    # (key unreadable etc.): the export path must SAY the artifact is unsigned — silence would
    # let CI 'verify' a should-have-been-signed artifact as merely-unsigned.
    from commands import trace as trace_cmd

    db = _seed_export_db(isolated_paths)
    monkeypatch.setattr(signing, "available", lambda: True)
    monkeypatch.setattr(signing, "sign_digest", lambda d: None)
    dest = isolated_paths / "run_1.json"
    trace_cmd.export_run(db, 1, dest=dest)
    assert "UNSIGNED — signing failed" in capsys.readouterr().err
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert "signature" not in payload
    assert payload["integrity"]["algorithm"] == "sha256"  # digest-only fallback intact


def test_full_export_verify_roundtrip(isolated_paths, capsys):
    """A run exported through commands.trace carries a digest + signature that /trace verify
    confirms; corrupting the file makes verify fail loudly."""
    from commands import trace as trace_cmd

    db_path_str = _seed_export_db(isolated_paths)

    class Ctx:
        db_path = db_path_str

    out_file = isolated_paths / "run_1.json"
    trace_cmd._export(Ctx(), ["#1", "-o", str(out_file)])
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["integrity"]["algorithm"] == "sha256"
    assert payload["signature"]["algorithm"] == "ed25519"
    assert payload["format"] == signing.ARTIFACT_FORMAT  # the marker rides the signed body

    capsys.readouterr()  # clear
    trace_cmd._verify(Ctx(), [str(out_file)])
    out = capsys.readouterr().out
    assert "verifies" in out and "signature valid" in out

    # Tamper: flip the recorded response, re-verify -> both digest and (independently) the
    # signature-over-stored-digest still flags the change via the digest mismatch.
    payload["run"]["response"] = "tampered"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    capsys.readouterr()
    trace_cmd._verify(Ctx(), [str(out_file)])
    out = capsys.readouterr().out
    assert "DOES NOT verify" in out
