"""
Trust report — the single signed document that summarizes this session's trust posture.

Every transparency surface Saturn already has answers one question: the policy gate (what runs
without asking), the egress ledger (what left the machine), the inference bindings (local vs
cloud), redaction (what was stripped), the air-gap (is the boundary sealed). This module gathers
all of them into ONE record and signs it — the artifact a security-conscious user (or their
auditor) actually wants: a portable, verifiable attestation of "here is exactly how this agent was
allowed to behave, and here is what it actually did."

It is pure assembly over leaf modules (config, egress, policy, quarantine, redaction, signing) —
no new state,
no I/O of its own beyond what those modules already do. `build_report()` returns a plain dict (so
it is trivially testable and renderable); `sign_report()` wraps it with the same digest+signature
layering as a trace export, so `/trace verify`-style checks apply to it too.
"""

from __future__ import annotations

from datetime import datetime

from trust import egress
from trust import policy
from trust import quarantine
from trust import redaction
from trust import signing
from config import MODEL_ROLES, get_config

REPORT_VERSION = 1


def _inference() -> dict:
    """THE locality classifier (posture line, Glass Box attestation, /privacy all read this —
    never re-roll it). 'local' means the words are computed ON THIS MACHINE: an Ollama binding
    only earns it when the endpoint is loopback — a remote OLLAMA_HOST is network inference and
    classifies 'remote' (off-machine, like 'cloud'), reported with the endpoint so the reader
    can see where."""
    cfg = get_config()
    ollama_local = egress.ollama_is_local()
    ollama_loc = "local" if ollama_local else "remote"
    bindings = []
    for role in MODEL_ROLES:
        try:
            spec = cfg.model_for_role(role)
        except KeyError:
            continue
        bindings.append({
            "role": role,
            "provider": spec.provider,
            "model": spec.model,
            "locality": ollama_loc if spec.provider == "ollama" else "cloud",
        })
    try:
        bindings.append({"role": "embedder", "provider": "ollama",
                         "model": cfg.embedder_model, "locality": ollama_loc})
    except Exception:
        pass
    cloud = sorted({b["provider"] for b in bindings if b["locality"] == "cloud"})
    remote = any(b["locality"] == "remote" for b in bindings)
    out = {"bindings": bindings, "cloud_providers": cloud,
           "all_local": not cloud and not remote}
    if remote:
        out["remote_ollama"] = egress.ollama_endpoint()
    return out


def remote_ollama_label(inf: dict) -> str:
    """The display label for a remote-Ollama destination (`ollama @ <endpoint>`) — one spelling
    for every surface that names it (posture line, /privacy tables, the report render)."""
    return f"ollama @ {inf.get('remote_ollama', '?')}"


def offmachine_destinations(inf: "dict | None" = None) -> list[str]:
    """The off-machine inference destinations as display labels: cloud providers plus a remote
    Ollama endpoint (`remote_ollama_label`). THE one assembly of the where-list — the session
    posture line (receipt.posture_spans), /privacy's verdict, and the trust-report render all
    print this, so they can never name different destination sets for the identical posture.
    Takes the classifier's dict (or computes it fresh); empty when everything is local."""
    if inf is None:
        inf = _inference()
    where = list(inf.get("cloud_providers") or [])
    if inf.get("remote_ollama"):
        where.append(remote_ollama_label(inf))
    return where


def build_report() -> dict:
    """Assemble the full trust posture + activity for this session into one plain dict.

    Carries `egress_anchor` (egress.log_tip() — the durable log's chain head at report time)
    INSIDE the body sign_report signs: the signature commits the chain head, so a later
    tail-truncation of egress.log — the one tamper the self-contained chain can't expose —
    contradicts this report (a verifier confirms the anchored tip still appears in the chain).
    The field is simply ABSENT when there is no durable log or the knob is off — never a fake
    value."""
    cfg = get_config()
    # One read of the durable log feeds the summary; verify_log walks the raw lines itself (it
    # must hash them byte-for-byte, so it can't share parsed rows).
    durable = egress.log_summary(egress.read_log())
    chain = egress.verify_log()

    key = signing.key_info()
    # The trust report never carries the private key (key_info doesn't expose it either) — only the
    # public key + fingerprint, which are safe to publish.
    signing_info = {
        "available": key.get("available", False),
        "key_id": key.get("key_id"),
        "public_key": key.get("public_key"),
    }

    report = {
        "saturn_trust_report": REPORT_VERSION,
        # The versioned artifact-format marker the standalone verify spec pins
        # (utilities/VERIFY_SPEC.md); verify flows accept reports with and without it.
        "format": signing.ARTIFACT_FORMAT,
        "saturn_version": signing.saturn_version(),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_id": egress._SESSION_ID,
        "inference": _inference(),
        "policy": {
            "auto_approve": policy.tier(),
            "gate_off": policy.gate_off(),
            "risk_overrides": policy.risk_overrides(),
            "shell_allow": policy.shell_allow(),
        },
        "boundary": {
            "airgap": bool(cfg.get("runtime.airgap", False)),
            "redaction": redaction.mode(),
            # The EFFECTIVE mode (mirrors redaction.mode() above): quarantine.mode() normalizes
            # case and falls back to "gate" on an invalid value — a signed attestation must
            # state the mode in force, never echo a raw string the system ignored.
            "quarantine": quarantine.mode(),
            "dry_run": bool(cfg.get("runtime.dry_run", False)),
        },
        "egress_session": egress.summary(),
        "egress_durable": {
            "lines": durable["lines"],
            "sent": durable["sent"],
            "blocked": durable["blocked"],
            "bytes": durable["bytes"],
            "sessions": len(durable["sessions"]),
            "hosts": durable["hosts"],
            "chain_ok": chain.get("ok", True),
            "chain_broken_at": chain.get("broken_at"),
        },
        "signing": signing_info,
    }
    # Egress-chain anchor (see the docstring): inside the signed body, absent when unavailable.
    anchor = egress.log_tip()
    if anchor:
        report["egress_anchor"] = anchor
    return report


def sign_report(report: dict) -> dict:
    """Return a copy of `report` with an integrity digest and (when signing is available) an
    ed25519 signature attached — the portable, verifiable artifact. The digest is
    signing.canonical_digest — the SAME scheme as a trace export, so the report's integrity is
    checked exactly the same way."""
    payload = dict(report)
    digest = signing.canonical_digest(payload)
    payload["integrity"] = {"algorithm": "sha256", "digest": digest}
    if bool(get_config().get("runtime.sign_exports", True)):
        sig = signing.sign_digest(digest)
        if sig:
            payload["signature"] = sig
    return payload


def verify_report(payload: dict) -> dict:
    """Check a signed report's digest + signature (the same `signing.verify_payload` flow
    `/trace verify` runs — which also accepts trust-report files directly). Returns
    {is_report, digest_ok, signed, signature_ok, key_id}."""
    if not isinstance(payload, dict) or payload.get("saturn_trust_report") != REPORT_VERSION:
        return {"is_report": False}
    v = signing.verify_payload(payload)
    out = {"is_report": True, "digest_ok": v["digest_ok"], "signed": v["signed"]}
    if "signature_ok" in v:
        out["signature_ok"] = v["signature_ok"]
        out["key_id"] = v.get("key_id")
    return out
