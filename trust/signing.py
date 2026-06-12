"""
Ed25519 signatures for audit artifacts — turning Saturn's transparency claims into
third-party-verifiable PROVENANCE.

A `/trace export` already carries a sha256 *integrity* digest: tamper-evidence (the record was
not altered after export). A signature adds the next rung — proof the record was produced by the
holder of THIS machine's private key. That is the one assurance a cloud agent can never give you,
because the signing key exists only on hardware you own. It is also the seam to the v2.0 team
layer (a signed audit trail an auditor can verify against your published key).

Design:
  - One Ed25519 keypair, generated lazily on first use and stored at `paths.signing_key`
    (database/signing_key.json). The PRIVATE key never leaves the machine; the PUBLIC key is
    embedded in every signed artifact and printed by `/trace key` so it can be published.
  - We sign the export's sha256 digest (a 64-char hex string), not the whole payload — the digest
    already commits to every byte (collision-resistant), so signing it is equivalent and keeps the
    two mechanisms cleanly layered: the digest proves "content unchanged", the signature proves
    "signed by this key". Both must pass for an authentic record.
  - Verification needs only the public key (carried in the signature block), so anyone you hand a
    signed export to can check it offline — and confirm it is YOURS by comparing the key
    fingerprint to your published one.

Degrades safely: if the `cryptography` library is absent, signing is simply unavailable —
`available()` reports False and callers fall back to the existing digest-only behavior unchanged.
No outbound dependency, no network: a private-key file and a hashing/signing primitive. Imports
only config + diag, so any layer may use it without a cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import diag
from config import get_config

# Block identifier embedded in every signature so a verifier knows the scheme without guessing.
SCHEME = "ed25519"
_KEY_FILE_VERSION = 1

# Versioned name of the signed-artifact FORMAT (layout + canonicalization + digest + signature
# layering), embedded inside the signed body of every trace export and trust report so the
# standalone verify spec (utilities/VERIFY_SPEC.md) and the zero-dependency verifier
# (utilities/saturn_verify.py) have a stable name to pin. Every verify flow accepts artifacts
# with AND without the marker (records written before 2026-06-11 carry none) — bump the suffix
# only when the scheme itself changes, and update the spec in the same change.
ARTIFACT_FORMAT = "saturn-artifact/1"


# --- the canonical digest scheme -------------------------------------------------------------
# THE byte stream every Saturn integrity check commits to. One home (here, the provenance leaf
# everything already imports) so trace exports, trust reports, and the egress hash chain can never
# drift onto different canonicalizations — a one-copy tweak to separators/ensure_ascii would
# silently break verification of every artifact the other copies produced. Mirrored, by exact
# transcription, in utilities/VERIFY_SPEC.md + utilities/saturn_verify.py (the no-Saturn verify
# path): a change here is a FORMAT change — bump ARTIFACT_FORMAT and update both.


def canonical_json(payload: dict) -> str:
    """Canonical JSON of `payload`: sorted keys, tight separators, raw unicode."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_digest(payload: dict) -> str:
    """sha256 hex over canonical_json(payload). `integrity`/`signature` must not be in `payload`."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def saturn_version() -> str:
    """The running Saturn version for stamping signed artifacts — read off the already-loaded
    agent module (importing agent.py here would be heavy and double-imports under
    `python agent.py`)."""
    import sys

    for name in ("__main__", "agent"):
        v = getattr(sys.modules.get(name), "__version__", None)
        if v:
            return str(v)
    return "unknown"


def _crypto():
    """The cryptography Ed25519 primitives, or None if the library isn't installed. Imported lazily
    so the module loads (and `available()` answers) even on a machine without `cryptography`."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives import serialization

        return Ed25519PrivateKey, Ed25519PublicKey, serialization
    except Exception:
        return None


def available() -> bool:
    """Whether signing is possible on this machine (the `cryptography` library is importable)."""
    return _crypto() is not None


def _key_path() -> Path:
    """Location of the private-key file. Falls back to database/signing_key.json when the
    config predates the `paths.signing_key` key, so an older config.yaml still works."""
    try:
        return get_config().path("signing_key")
    except KeyError:
        return get_config().path("database") / "signing_key.json"


def _fingerprint_of(public_hex: str) -> str:
    """A short, stable key id: the first 16 hex chars of sha256 over the raw public-key bytes.
    Short enough to read aloud / publish, long enough to be unambiguous in practice."""
    try:
        raw = bytes.fromhex(public_hex)
    except ValueError:
        raw = public_hex.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# (key-file path, private key, public-key hex) — the keypair is immutable for the process
# lifetime, so one disk read serves every sign/verify/fingerprint call. Keyed by path so a
# config change (or the test suite's per-test path isolation) reloads naturally.
_KEY_CACHE: "tuple[str, object, str] | None" = None


def _load_or_create():
    """Return the Ed25519 private key, generating + persisting a fresh keypair on first use.
    Returns None if signing is unavailable or the key file can't be created. Cached after the
    first load (the keypair never changes within a process). Best-effort: any failure is logged
    and degrades to no-signing rather than raising into an export."""
    global _KEY_CACHE
    crypto = _crypto()
    if crypto is None:
        return None
    Ed25519PrivateKey, _Pub, serialization = crypto
    path = _key_path()
    if _KEY_CACHE and _KEY_CACHE[0] == str(path):
        return _KEY_CACHE[1]

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            priv_hex = data["private_key"]
            key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
            pub_hex = key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ).hex()
            _KEY_CACHE = (str(path), key, pub_hex)
            return key
        except Exception as exc:
            diag.log(f"signing: could not load key at {path}: {exc}")
            return None

    # First use — mint a keypair.
    try:
        key = Ed25519PrivateKey.generate()
        priv_raw = key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_raw = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        doc = {
            "saturn_signing_key": _KEY_FILE_VERSION,
            "algorithm": SCHEME,
            "private_key": priv_raw.hex(),
            "public_key": pub_raw.hex(),
            "key_id": _fingerprint_of(pub_raw.hex()),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        # Best-effort: lock the private key down to the owner on POSIX. (Windows ACLs differ; the
        # file already lives under the user's own data dir, so this is defense in depth.)
        try:
            os.chmod(path, 0o600)
        except (OSError, NotImplementedError):
            pass
        diag.log(f"signing: generated new ed25519 keypair at {path} (key_id {doc['key_id']})")
        _KEY_CACHE = (str(path), key, pub_raw.hex())
        return key
    except Exception as exc:
        diag.log(f"signing: could not create key at {path}: {exc}")
        return None


def public_key_hex() -> "str | None":
    """Hex of this machine's raw public key, or None if signing is unavailable / key creation
    failed. This is the value you publish so others can verify your signed exports."""
    if _load_or_create() is None:
        return None
    return _KEY_CACHE[2] if _KEY_CACHE else None


def fingerprint() -> "str | None":
    """The short key id of this machine's signing key (None if unavailable)."""
    pub = public_key_hex()
    return _fingerprint_of(pub) if pub else None


def sign_digest(digest_hex: str) -> "dict | None":
    """Sign an export's sha256 digest. Returns the signature block to embed in the export, or None
    if signing is unavailable (caller then emits a digest-only export, exactly as before).

    The block carries everything a verifier needs offline: the scheme, the signature, and the
    public key + its fingerprint (so the verifier can both check the signature AND identify whose
    key it is)."""
    crypto = _crypto()
    key = _load_or_create()
    if crypto is None or key is None:
        return None
    try:
        sig = key.sign(digest_hex.encode("utf-8"))
        pub = public_key_hex()
        return {
            "algorithm": SCHEME,
            "signed": "sha256-digest",
            "public_key": pub,
            "key_id": _fingerprint_of(pub) if pub else "",
            "signature": sig.hex(),
        }
    except Exception as exc:
        diag.log(f"signing: sign_digest failed: {exc}")
        return None


def verify_signature(digest_hex: str, sig_block: dict) -> bool:
    """True iff `sig_block`'s signature is a valid Ed25519 signature over `digest_hex`, made by the
    public key the block carries. Verification needs only the public key, so this works on any
    machine — including one that never created its own key. Returns False on any malformation."""
    crypto = _crypto()
    if crypto is None or not isinstance(sig_block, dict):
        return False
    _Priv, Ed25519PublicKey, _ser = crypto
    try:
        if sig_block.get("algorithm") != SCHEME:
            return False
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(sig_block["public_key"]))
        pub.verify(bytes.fromhex(sig_block["signature"]), digest_hex.encode("utf-8"))
        return True
    except Exception:
        return False


def verify_payload(payload: dict) -> dict:
    """Verify a signed Saturn artifact (trace export, trust report — anything carrying the
    `integrity` + `signature` blocks this module writes). THE one implementation of the fragile
    rule every verifier shares: BOTH blocks must be popped off a copy before recomputing the
    digest (neither is part of the canonical bytes the digest covers). Returns:
      {has_integrity, stored_digest, computed_digest, digest_ok,
       signed, signature_ok?, key_id?, public_key?, signer_is_local?}
    `signature_ok` and friends are present only when a signature block exists."""
    body = dict(payload or {})
    integrity = body.pop("integrity", None)
    signature = body.pop("signature", None)
    has_integrity = isinstance(integrity, dict) and "digest" in integrity
    computed = canonical_digest(body)
    out: dict = {
        "has_integrity": has_integrity,
        "stored_digest": integrity.get("digest") if has_integrity else None,
        "computed_digest": computed,
        "digest_ok": has_integrity and computed == integrity.get("digest"),
        "signed": bool(signature),
    }
    if signature and has_integrity:
        out["signature_ok"] = verify_signature(integrity["digest"], signature)
        out["key_id"] = signature.get("key_id", "?")
        out["public_key"] = signature.get("public_key")
        out["signer_is_local"] = signer_matches_local(signature)
    return out


def signer_matches_local(sig_block: dict) -> bool:
    """Whether the signature was made by THIS machine's key (the public keys match). Lets a
    verifier distinguish 'signed by me' from 'signed by some other valid key' without external
    state. False if signing is unavailable here."""
    local = public_key_hex()
    return bool(local) and isinstance(sig_block, dict) and sig_block.get("public_key") == local


def key_info() -> dict:
    """Display-ready facts about this machine's signing key for `/trace key` and the trust report.
    Never raises: reports `available=False` when signing isn't possible."""
    if not available():
        return {"available": False}
    path = _key_path()
    info: dict = {
        "available": True,
        "path": str(path),
        "exists": path.exists(),
    }
    pub = public_key_hex()
    info["public_key"] = pub
    info["key_id"] = _fingerprint_of(pub) if pub else None
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            info["created_at"] = doc.get("created_at")
        except Exception:
            pass
    return info
