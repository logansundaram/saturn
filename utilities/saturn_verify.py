#!/usr/bin/env python3
"""
saturn_verify.py — the zero-Saturn, zero-dependency verifier for Saturn audit artifacts.

Saturn's moat claim is that its transparency artifacts are THIRD-PARTY-verifiable. That claim is
only true if verification does not require Saturn itself — this single file is the proof. It
verifies, with nothing but the Python standard library:

  1. a signed artifact (a `/trace export` run record or a `/privacy report -o` trust report):

         python saturn_verify.py run_7.json

     recomputes the sha256 integrity digest over the canonical bytes and checks the ed25519
     signature over that digest with the public key the artifact itself carries.

  2. a Saturn egress log (the durable, hash-chained record of what left the machine):

         python saturn_verify.py --egress-log egress.log [--expect-tip <hash>]

     walks the hash chain line by line. `--expect-tip` takes the `egress_anchor.tip_hash` from a
     signed artifact and confirms the chain still REACHES that tip — the check that catches a
     clean tail-truncation, the one tamper a self-contained chain cannot expose on its own.

Exit codes (mirroring `saturn verify`):
  0  intact — digest matches and the signature (when present) is valid; an unsigned-but-intact
     artifact prints "unsigned" and still passes. Chain mode: chain intact (+ anchored tip found).
  1  tampered/forged — digest mismatch, invalid signature, missing integrity block, a broken
     chain, or an anchored tip the chain never reaches.
  2  usage or read errors (unreadable file, not a Saturn artifact).

This file reimplements the scheme from the spec (utilities/VERIFY_SPEC.md, format
"saturn-artifact/1") — it imports NOTHING from Saturn, so it can be copied next to an artifact
and run anywhere. If the `cryptography` library happens to be importable it is used for the
ed25519 check (fast); otherwise a vendored pure-Python ed25519 VERIFY fallback runs (slow —
~a second — which is fine for one signature).
"""

import argparse
import hashlib
import json
import sys

ARTIFACT_FORMAT = "saturn-artifact/1"  # the format this verifier implements


# --- canonicalization + digest (transcribed from the spec; one rule, never drift) -------------
# Canonical JSON: sorted keys, tight separators, raw (non-ascii-escaped) unicode, encoded utf-8.
# The digest never covers the `integrity` or `signature` blocks — both are popped off a COPY
# before recomputing.


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_digest(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# --- ed25519 verification ----------------------------------------------------------------------


def _try_cryptography(public: bytes, msg: bytes, sig: bytes):
    """Verify via the `cryptography` library when importable: True/False, or None when the
    library is absent (the caller then falls back to the pure-Python path)."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except Exception:
        return None
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(sig, msg)
        return True
    except Exception:
        return False


# --- vendored pure-Python ed25519 (VERIFY ONLY) -------------------------------------------------
# Transcribed from the Ed25519 reference implementation published in RFC 8032, Section 6
# ("Edwards-Curve Digital Signature Algorithm (EdDSA)", IETF, January 2017,
# https://www.rfc-editor.org/rfc/rfc8032#section-6). Sign/keygen paths removed; bad-length inputs
# return False instead of raising. NOT constant-time and slow — both irrelevant for verifying one
# public signature offline.

_P = 2 ** 255 - 19
_Q = 2 ** 252 + 27742317777372353535851937790883648493


def _modp_inv(x: int) -> int:
    return pow(x, _P - 2, _P)


_D = -121665 * _modp_inv(121666) % _P
_MODP_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _sha512_modq(s: bytes) -> int:
    return int.from_bytes(hashlib.sha512(s).digest(), "little") % _Q


def _point_add(P, Q):
    A = (P[1] - P[0]) * (Q[1] - Q[0]) % _P
    B = (P[1] + P[0]) * (Q[1] + Q[0]) % _P
    C = 2 * P[3] * Q[3] * _D % _P
    D = 2 * P[2] * Q[2] % _P
    E, F, G, H = B - A, D - C, D + C, B + A
    return (E * F % _P, G * H % _P, F * G % _P, E * H % _P)


def _point_mul(s: int, P):
    Q = (0, 1, 1, 0)  # the neutral element
    while s > 0:
        if s & 1:
            Q = _point_add(Q, P)
        P = _point_add(P, P)
        s >>= 1
    return Q


def _point_equal(P, Q) -> bool:
    # x1 / z1 == x2 / z2  <==>  x1 * z2 == x2 * z1 (and likewise for y)
    if (P[0] * Q[2] - Q[0] * P[2]) % _P != 0:
        return False
    if (P[1] * Q[2] - Q[1] * P[2]) % _P != 0:
        return False
    return True


def _recover_x(y: int, sign: int):
    if y >= _P:
        return None
    x2 = (y * y - 1) * _modp_inv(_D * y * y + 1)
    if x2 == 0:
        if sign:
            return None
        return 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _MODP_SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_G_Y = 4 * _modp_inv(5) % _P
_G_X = _recover_x(_G_Y, 0)
_G = (_G_X, _G_Y, 1, _G_X * _G_Y % _P)


def _point_decompress(s: bytes):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _ed25519_verify_pure(public: bytes, msg: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    A = _point_decompress(public)
    if not A:
        return False
    Rs = signature[:32]
    R = _point_decompress(Rs)
    if not R:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _Q:
        return False
    h = _sha512_modq(Rs + public + msg)
    sB = _point_mul(s, _G)
    hA = _point_mul(h, A)
    return _point_equal(sB, _point_add(R, hA))


def ed25519_verify(public: bytes, msg: bytes, sig: bytes) -> bool:
    """One ed25519 verification: `cryptography` when importable, vendored pure Python otherwise."""
    via = _try_cryptography(public, msg, sig)
    if via is not None:
        return via
    return _ed25519_verify_pure(public, msg, sig)


# --- artifact verification ----------------------------------------------------------------------


def verify_artifact(payload: dict) -> dict:
    """Verify one signed Saturn artifact dict. The spec's rule: pop `integrity` + `signature` off
    a COPY, recompute the canonical digest, compare to the stored one; the signature (when
    present) is checked over the STORED digest's hex string as utf-8 bytes. Returns
    {has_integrity, stored_digest, computed_digest, digest_ok, signed, signature_ok?, key_id?}."""
    body = dict(payload or {})
    integrity = body.pop("integrity", None)
    signature = body.pop("signature", None)
    has_integrity = isinstance(integrity, dict) and "digest" in integrity
    computed = canonical_digest(body)
    out = {
        "has_integrity": has_integrity,
        "stored_digest": integrity.get("digest") if has_integrity else None,
        "computed_digest": computed,
        "digest_ok": has_integrity and computed == integrity.get("digest"),
        "signed": bool(signature),
    }
    if signature and has_integrity:
        ok = False
        if isinstance(signature, dict) and signature.get("algorithm") == "ed25519":
            try:
                ok = ed25519_verify(
                    bytes.fromhex(signature["public_key"]),
                    str(integrity["digest"]).encode("utf-8"),
                    bytes.fromhex(signature["signature"]),
                )
            except Exception:
                ok = False
        out["signature_ok"] = ok
        out["key_id"] = signature.get("key_id", "?") if isinstance(signature, dict) else "?"
    return out


def _is_saturn_artifact(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("saturn_trace_export") == 1:
        return True
    if payload.get("saturn_trust_report") is not None:
        return True
    # Future artifact kinds carry only the versioned format marker.
    fmt = payload.get("format")
    return isinstance(fmt, str) and fmt.startswith("saturn-artifact/")


def _cli_artifact(path: str) -> int:
    try:
        # utf-8-sig: tolerate a BOM (PowerShell 5.1 redirection writes one).
        with open(path, "r", encoding="utf-8-sig") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"error: could not read {path}: {e}", file=sys.stderr)
        return 2
    if not _is_saturn_artifact(payload):
        print(f"error: {path} is not a Saturn artifact (trace export / trust report).",
              file=sys.stderr)
        return 2

    if payload.get("saturn_trace_export") == 1:
        kind = "trace export"
    elif payload.get("saturn_trust_report") is not None:
        kind = "trust report"
    else:
        kind = str(payload.get("format"))

    v = verify_artifact(payload)
    if not v["has_integrity"]:
        print(f"{path}: no integrity digest — nothing to verify (only JSON exports carry one).")
        return 1
    ok = True
    if v["digest_ok"]:
        print(f"{path}: digest ok ({kind}) — sha256 {v['stored_digest']}")
    else:
        ok = False
        print(f"{path}: digest MISMATCH — the record was modified after export.")
        print(f"  stored   {v['stored_digest']}")
        print(f"  computed {v['computed_digest']}")
    if v["signed"]:
        if v.get("signature_ok"):
            print(f"signature valid — ed25519 key {v.get('key_id', '?')}")
        else:
            ok = False
            print(f"signature INVALID — does not match the embedded key {v.get('key_id', '?')} "
                  "(forged, corrupted, or the digest was altered).")
    else:
        print("unsigned — sha256 integrity digest only (no signature block).")

    anchor = payload.get("egress_anchor")
    if isinstance(anchor, dict) and anchor.get("tip_hash"):
        print(f"egress anchor — chain tip {anchor['tip_hash']} at line "
              f"{anchor.get('line_count', '?')}; check the log still reaches it with:")
        print(f"  python {sys.argv[0] if sys.argv else 'saturn_verify.py'} "
              f"--egress-log <egress.log> --expect-tip {anchor['tip_hash']}")
    return 0 if ok else 1


# --- egress-log chain verification ---------------------------------------------------------------
# Transcribed from the spec (the same walk Saturn's egress.verify_log performs): every raw
# non-empty line must parse as a JSON object, recompute its hash — sha256 of (prev +
# canonical_json(payload)) where payload is the object minus `prev`/`h` — and link to its
# predecessor. An unparseable/garbled line is a BROKEN chain, never silently skipped. The first
# line's `prev` is "".


def verify_chain(lines, expect_tip: str = "") -> dict:
    """Walk an egress log's hash chain. `lines` is an iterable of raw text lines. Returns
    {ok, lines, broken_at, error, tip, tip_found?, tip_line?} — `tip_found` only when
    `expect_tip` was given (the anchored tip must appear as some intact line's `h`)."""
    raw = [ln for ln in lines if ln.strip()]
    prev = ""
    tip_found = False
    tip_line = None
    for i, line in enumerate(raw, start=1):
        try:
            row = json.loads(line)
        except Exception:
            return {"ok": False, "lines": len(raw), "broken_at": i, "error": "unparseable line",
                    "tip": prev}
        if not isinstance(row, dict):
            return {"ok": False, "lines": len(raw), "broken_at": i, "error": "not a record",
                    "tip": prev}
        payload = {k: v for k, v in row.items() if k not in ("prev", "h")}
        h = hashlib.sha256((prev + canonical_json(payload)).encode("utf-8")).hexdigest()
        if row.get("prev", "") != prev or row.get("h") != h:
            return {"ok": False, "lines": len(raw), "broken_at": i, "error": "hash mismatch",
                    "tip": prev}
        prev = row["h"]
        if expect_tip and prev == expect_tip:
            tip_found = True
            tip_line = i
    out = {"ok": True, "lines": len(raw), "broken_at": None, "error": None, "tip": prev}
    if expect_tip:
        out["tip_found"] = tip_found
        out["tip_line"] = tip_line
    return out


def _cli_chain(path: str, expect_tip: str) -> int:
    try:
        # utf-8-sig: tolerate a BOM a copy/redirect may have added.
        with open(path, "r", encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        print(f"error: could not read {path}: {e}", file=sys.stderr)
        return 2

    v = verify_chain(lines, expect_tip or "")
    if not v["ok"]:
        print(f"{path}: chain BROKEN at line {v['broken_at']} ({v['error']}) — the log was "
              "edited, reordered, or corrupted.")
        return 1
    print(f"{path}: chain intact — {v['lines']} line(s), tip {v['tip'] or '(empty)'}")
    if expect_tip:
        if v.get("tip_found"):
            print(f"anchored tip reached at line {v['tip_line']} — no tail truncation before "
                  "the anchor.")
        else:
            print("anchored tip NOT reached — the log was TRUNCATED (or rewritten) since the "
                  "artifact carrying this anchor was signed.")
            return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="saturn_verify",
        description="Verify Saturn audit artifacts and egress logs with no Saturn install "
                    "(spec: utilities/VERIFY_SPEC.md, format saturn-artifact/1).",
    )
    parser.add_argument("artifact", nargs="?", default=None,
                        help="a signed Saturn artifact (.json): /trace export or /privacy report")
    parser.add_argument("--egress-log", metavar="FILE", default=None,
                        help="walk a Saturn egress log's hash chain instead")
    parser.add_argument("--expect-tip", metavar="HASH", default=None,
                        help="with --egress-log: a signed artifact's egress_anchor.tip_hash — "
                             "the chain must still reach it (catches tail truncation)")
    args = parser.parse_args(argv)

    if args.egress_log:
        if args.artifact:
            print("error: pass either an artifact or --egress-log, not both.", file=sys.stderr)
            return 2
        return _cli_chain(args.egress_log, args.expect_tip or "")
    if args.expect_tip:
        print("error: --expect-tip only applies to --egress-log.", file=sys.stderr)
        return 2
    if not args.artifact:
        parser.print_usage(sys.stderr)
        return 2
    return _cli_artifact(args.artifact)


if __name__ == "__main__":
    sys.exit(main())
