"""
The integrity digest (trust/digest.py) — the tamper-evidence layer behind `/trace export`,
`verify`, and `replay` after the ed25519 signing/provenance layer was shelved. Pure/offline:
stdlib only, no LLM/network/DB.
"""

from trust import digest


def test_canonical_json_is_stable_across_key_order():
    a = {"b": 1, "a": [3, 2], "z": {"y": True, "x": None}}
    b = {"z": {"x": None, "y": True}, "a": [3, 2], "b": 1}
    assert digest.canonical_json(a) == digest.canonical_json(b)
    # tight separators, sorted keys, raw unicode
    assert digest.canonical_json({"k": "café"}) == '{"k":"café"}'


def test_canonical_digest_changes_with_content():
    base = {"run": {"response": "hi"}, "events": []}
    d1 = digest.canonical_digest(base)
    assert len(d1) == 64 and int(d1, 16) >= 0          # sha256 hex
    base["run"]["response"] = "bye"
    assert digest.canonical_digest(base) != d1


def _artifact():
    body = {
        "saturn_trace_export": 1,
        "format": digest.ARTIFACT_FORMAT,
        "run": {"run_id": 1, "response": "hi"},
        "events": [],
    }
    body["integrity"] = {"algorithm": "sha256", "digest": digest.canonical_digest(body)}
    return body


def test_verify_payload_intact_and_never_signed():
    v = digest.verify_payload(_artifact())
    assert v["has_integrity"] and v["digest_ok"]
    assert v["signed"] is False                         # provenance layer is shelved


def test_verify_payload_detects_tampering():
    body = _artifact()
    body["run"]["response"] = "tampered after the digest was taken"
    v = digest.verify_payload(body)
    assert v["has_integrity"] and v["digest_ok"] is False
    assert v["stored_digest"] != v["computed_digest"]


def test_verify_payload_ignores_a_legacy_signature_block():
    # An artifact minted before the signing layer was shelved still carries a `signature` block;
    # it is popped off the copy and never affects the digest check — the digest still verifies.
    body = _artifact()
    body["signature"] = {"algorithm": "ed25519", "signature": "ab" * 64}
    assert digest.verify_payload(body)["digest_ok"] is True


def test_verify_payload_without_integrity_block():
    v = digest.verify_payload({"saturn_trace_export": 1, "run": {}})
    assert v["has_integrity"] is False and v["digest_ok"] is False
    assert digest.verify_payload(None)["has_integrity"] is False
