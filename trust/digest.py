"""
Canonical integrity digest for trace exports — tamper-evidence, not provenance.

A `/trace export` carries a sha256 *integrity* digest over a canonical serialization of its body:
proof the record was not altered after export. `verify`/`replay` recompute the digest and report
whether the content matches. This is self-contained tamper-evidence — it answers "was this record
changed?", not "who produced it" (the ed25519 signing/attestation layer that answered the latter
was shelved to the phase-3/audit-crypto branch).

`canonical_json`/`canonical_digest` are THE one byte stream every Saturn integrity check commits
to — one home so a tweak to separators/ensure_ascii can't silently break verification. Imports
only stdlib, so any layer may use it without a cycle.
"""

from __future__ import annotations

import hashlib
import json

# Versioned name of the export FORMAT (layout + canonicalization + digest), embedded inside the
# body of every trace export so a reader has a stable name to pin. Verify flows accept artifacts
# with AND without the marker (older records carry none).
ARTIFACT_FORMAT = "saturn-artifact/1"


def canonical_json(payload: dict) -> str:
    """Canonical JSON of `payload`: sorted keys, tight separators, raw unicode."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_digest(payload: dict) -> str:
    """sha256 hex over canonical_json(payload). `integrity` must not be in `payload`."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def saturn_version() -> str:
    """The running Saturn version for stamping exports — read off the already-loaded agent module
    (importing agent.py here would be heavy and double-imports under `python agent.py`)."""
    import sys

    for name in ("__main__", "agent"):
        v = getattr(sys.modules.get(name), "__version__", None)
        if v:
            return str(v)
    return "unknown"


def verify_payload(payload: dict) -> dict:
    """Verify a trace export's integrity digest. THE one implementation of the fragile rule every
    verifier shares: the `integrity` block must be popped off a COPY before recomputing the digest
    (it is not part of the canonical bytes the digest covers). A legacy `signature` block (from an
    artifact produced before the signing layer was shelved) is likewise popped and ignored — its
    presence never breaks the digest check. Returns:
      {has_integrity, stored_digest, computed_digest, digest_ok, signed}
    `signed` is always False now: this build verifies integrity, not provenance."""
    body = dict(payload or {})
    integrity = body.pop("integrity", None)
    body.pop("signature", None)  # legacy provenance block — not part of the canonical bytes
    has_integrity = isinstance(integrity, dict) and "digest" in integrity
    computed = canonical_digest(body)
    return {
        "has_integrity": has_integrity,
        "stored_digest": integrity.get("digest") if has_integrity else None,
        "computed_digest": computed,
        "digest_ok": has_integrity and computed == integrity.get("digest"),
        "signed": False,
    }
