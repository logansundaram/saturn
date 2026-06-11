"""
Trust report (trust_report.py) — the signed attestation that gathers every trust surface.

Covers: the report assembles the expected sections, signing wraps it with a verifiable
digest+signature, verify_report confirms an untouched report and flags a tampered one.
"""

import egress
import trust_report


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
