"""
Trust report (trust_report.py) — the signed attestation that gathers every trust surface.

Covers: the report assembles the expected sections, signing wraps it with a verifiable
digest+signature, verify_report confirms an untouched report and flags a tampered one.
"""

from trust import egress
from trust import trust_report


def test_build_report_shape(isolated_paths):
    egress.clear()
    r = trust_report.build_report()
    assert r["saturn_trust_report"] == trust_report.REPORT_VERSION
    for key in ("inference", "policy", "boundary", "egress_session", "egress_durable", "signing"):
        assert key in r
    assert isinstance(r["inference"]["bindings"], list)
    assert "auto_approve" in r["policy"]
    assert "airgap" in r["boundary"]


def test_report_reflects_egress(isolated_paths):
    egress.clear()
    egress.record("web_search", "api.example.com", n_bytes=100)
    r = trust_report.build_report()
    assert r["egress_session"]["sent"] == 1
    assert r["egress_durable"]["sent"] == 1
    assert r["egress_durable"]["chain_ok"] is True


def test_sign_and_verify_report(isolated_paths):
    egress.clear()
    report = trust_report.build_report()
    signed = trust_report.sign_report(report)
    assert signed["integrity"]["algorithm"] == "sha256"
    assert signed["signature"]["algorithm"] == "ed25519"

    v = trust_report.verify_report(signed)
    assert v["is_report"] is True
    assert v["digest_ok"] is True
    assert v["signature_ok"] is True


def test_tampered_report_fails_digest(isolated_paths):
    egress.clear()
    signed = trust_report.sign_report(trust_report.build_report())
    signed["boundary"]["airgap"] = not signed["boundary"]["airgap"]  # flip a posture field
    v = trust_report.verify_report(signed)
    assert v["digest_ok"] is False


def test_verify_rejects_non_report(isolated_paths):
    assert trust_report.verify_report({"not": "a report"}) == {"is_report": False}


def test_report_discloses_a_cleared_session_ledger(isolated_paths, monkeypatch):
    """A `/privacy egress clear` empties the in-memory ledger the report attests as
    egress_session — the signed body must carry the `cleared` marker so a post-clear report
    can never attest 'sent: 0' over a session that sent. (The never-cleared case pins
    _CLEARED_AT = 0 explicitly: the suites use egress.clear() as a test-isolation reset, so
    without the pin every report here would read cleared=True.)"""
    egress.clear()
    monkeypatch.setattr(egress, "_CLEARED_AT", 0)
    egress.record("web_search", "api.example.com", n_bytes=100)
    fresh = trust_report.build_report()
    assert fresh["egress_session"]["cleared"] is False
    assert fresh["egress_session"]["sent"] == 1

    egress.clear()
    r = trust_report.build_report()
    assert r["egress_session"]["cleared"] is True
    assert r["egress_session"]["sent"] == 0  # understated — the marker qualifies it

    # The marker rides INSIDE the signed body: digest + signature still verify over it.
    signed = trust_report.sign_report(r)
    v = trust_report.verify_report(signed)
    assert v["digest_ok"] is True and v.get("signature_ok") is True


def test_report_attests_effective_quarantine_mode(isolated_paths, monkeypatch):
    """boundary.quarantine must be the mode IN FORCE (quarantine.mode() — invalid values run as
    'gate', case is normalized), never the raw config string: a report attesting 'none' over a
    system actually running 'gate' misstates the posture it signs."""
    from config import get_config

    egress.clear()
    rt = get_config()._data.setdefault("runtime", {})
    monkeypatch.setitem(rt, "quarantine", "none")  # invalid → the system runs gated
    assert trust_report.build_report()["boundary"]["quarantine"] == "gate"
    monkeypatch.setitem(rt, "quarantine", "OFF")   # case variant → effective off
    assert trust_report.build_report()["boundary"]["quarantine"] == "off"


def test_report_carries_format_marker(isolated_paths):
    # The versioned artifact-format marker the standalone spec pins (utilities/VERIFY_SPEC.md).
    from trust import signing

    egress.clear()
    r = trust_report.build_report()
    assert r["format"] == signing.ARTIFACT_FORMAT


def test_report_anchor_present_inside_signed_body(isolated_paths):
    # The egress-chain anchor rides INSIDE the signed body: the report's signature commits the
    # chain head, so a later tail-truncation of egress.log contradicts the report.
    import json

    egress.clear()
    egress.record("web_search", "api.example.com", n_bytes=10)
    r = trust_report.build_report()
    assert r["egress_anchor"] == egress.log_tip()
    assert r["egress_anchor"]["tip_hash"] == egress.read_log()[-1]["h"]
    assert r["egress_anchor"]["line_count"] == 1

    signed = trust_report.sign_report(r)
    assert trust_report.verify_report(signed)["digest_ok"] is True
    tampered = json.loads(json.dumps(signed))
    tampered["egress_anchor"]["tip_hash"] = "0" * 64
    assert trust_report.verify_report(tampered)["digest_ok"] is False


def test_report_anchor_absent_without_log(isolated_paths):
    # No durable log in this fresh tree → the field is ABSENT, never a fake value.
    egress.clear()
    r = trust_report.build_report()
    assert "egress_anchor" not in r


def test_trace_verify_accepts_trust_report_file(isolated_paths, capsys):
    # `/privacy report -o` advertises the artifact as verifiable — /trace verify must accept it
    # (same digest+signature layering as a trace export), not bounce it as "not a trace export".
    import json

    from commands import trace as trace_cmd

    egress.clear()
    signed = trust_report.sign_report(trust_report.build_report())
    out_file = isolated_paths / "trust.json"
    out_file.write_text(json.dumps(signed, indent=2), encoding="utf-8")

    class Ctx:
        db_path = str(isolated_paths / "database" / "db.sqlite")

    capsys.readouterr()
    trace_cmd._verify(Ctx(), [str(out_file)])
    out = capsys.readouterr().out
    assert "verifies (trust report)" in out
    assert "signature valid" in out


def test_offmachine_destinations_is_the_one_where_list():
    # The posture line, /privacy's verdict, and the trust-report render all print this assembly —
    # one helper, so the three surfaces can never name different destination sets.
    from trust.trust_report import offmachine_destinations, remote_ollama_label

    assert offmachine_destinations({"cloud_providers": ["anthropic"]}) == ["anthropic"]
    inf = {"cloud_providers": ["anthropic"], "remote_ollama": "10.0.0.5:11434"}
    assert offmachine_destinations(inf) == ["anthropic", "ollama @ 10.0.0.5:11434"]
    assert remote_ollama_label(inf) == "ollama @ 10.0.0.5:11434"
    assert offmachine_destinations({"cloud_providers": []}) == []
