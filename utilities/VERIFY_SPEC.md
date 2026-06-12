# Saturn artifact verification spec — `saturn-artifact/1`

This page is the complete, self-contained recipe for verifying Saturn's audit artifacts **without
Saturn installed**. It is versioned by the `format` field embedded in every signed artifact
(`"format": "saturn-artifact/1"`); artifacts written before 2026-06-11 carry no marker but follow
the identical scheme. A zero-dependency reference implementation ships next to this file:
`utilities/saturn_verify.py`.

## Artifact layout

A Saturn artifact is one JSON document (UTF-8, possibly with a BOM — strip it). Two kinds exist
today, distinguished by a top-level type marker:

| kind         | type marker                  | written by                          |
|--------------|------------------------------|-------------------------------------|
| trace export | `"saturn_trace_export": 1`   | `/trace export`, `saturn -p ... --export` |
| trust report | `"saturn_trust_report": <n>` | `/privacy report -o <path>`         |

Verification blocks, appended after the body is final:

- `"integrity"`: `{"algorithm": "sha256", "digest": "<64 lowercase hex chars>"}`
- `"signature"` (optional — absent when `runtime.sign_exports` is off or the `cryptography`
  library was unavailable at export time):
  `{"algorithm": "ed25519", "signed": "sha256-digest", "public_key": "<64 hex chars — the raw
  32-byte ed25519 public key>", "key_id": "<16 hex chars>", "signature": "<128 hex chars — the
  raw 64-byte signature>"}`

The public key is **embedded in the artifact**, so verification needs no external material; to
know *whose* key it is, compare `key_id` (or the full `public_key`) against the fingerprint the
producer published (`/trace key`). `key_id` = the first 16 hex chars of
`sha256(raw 32 public-key bytes)`.

## Canonicalization (exact)

The digest covers the artifact **body**: the document with the `integrity` and `signature` keys
**removed from a copy** (everything else, in full — including `answer_attestation` and
`egress_anchor` when present). Serialize the body as canonical JSON:

```python
json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
```

— sorted keys, no whitespace, non-ASCII characters raw (not `\uXXXX`-escaped) — then encode UTF-8.

## Digest

`digest = sha256(canonical_bytes).hexdigest()` — 64 lowercase hex chars. The artifact is **intact**
iff this equals `integrity.digest`.

## Signature layering

The ed25519 signature signs **the digest's hex string as UTF-8 bytes** — i.e. the 64 ASCII
characters of `integrity.digest`, *not* the raw 32 digest bytes and *not* the body itself (the
digest already commits every byte of the body, so signing it is equivalent and keeps the two
layers separate):

```python
ed25519_verify(public_key = bytes.fromhex(signature["public_key"]),
               message    = integrity["digest"].encode("utf-8"),
               signature  = bytes.fromhex(signature["signature"]))
```

The signature is checked over the **stored** digest. Both checks must pass for an authentic
record: digest match = content unchanged; signature valid = produced by the holder of that key.

## Worked 4-step manual verify

```bash
# 1. strip the verification blocks off a copy; keep the stored digest
python -c "import json,sys; a=json.load(open(sys.argv[1],encoding='utf-8-sig')); i=a.pop('integrity'); a.pop('signature',None); open('body.json','w',encoding='utf-8').write(json.dumps(a,sort_keys=True,ensure_ascii=False,separators=(',',':'))); print(i['digest'])" run_7.json
# 2. recompute the digest over the canonical bytes
python -c "import hashlib; print(hashlib.sha256(open('body.json','rb').read()).hexdigest())"
# 3. compare: the two hex strings must be identical (content unchanged)
# 4. check the signature over the stored digest (pip install cryptography, or use saturn_verify.py)
python -c "import json,sys; from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey as K; a=json.load(open(sys.argv[1],encoding='utf-8-sig')); s=a['signature']; K.from_public_bytes(bytes.fromhex(s['public_key'])).verify(bytes.fromhex(s['signature']), a['integrity']['digest'].encode()); print('signature valid — key', s['key_id'])" run_7.json
```

Or simply: `python saturn_verify.py run_7.json` (exit 0 = intact + signature valid, or intact +
unsigned, printed as such; 1 = tampered/forged; 2 = unreadable/not an artifact).

## Egress log: line format, chain rule, anchor

Saturn's durable egress log (`database/egress.log`) is append-only JSONL. Each line is one JSON
object: the event payload (`ts`, `channel`, `host`, `detail`, `provider`, `n_bytes`,
`redactions`, `status`, `session`) plus two chain fields:

- `prev` — the previous line's `h` (`""` on the first line)
- `h` — `sha256((prev + canonical_json(payload)).encode("utf-8")).hexdigest()`, where `payload`
  is the line's object **minus** `prev` and `h`, canonicalized exactly as above.

**Chain rule:** walk every raw non-empty line in order. Each must (a) parse as a JSON object,
(b) carry `prev` equal to the previous line's `h`, and (c) recompute to its own `h`. An
unparseable or garbled line is a **broken chain** — never skip it. This detects any edit,
reorder, or deletion of the middle.

**Anchor semantics:** a clean *tail truncation* is the one tamper a self-contained chain cannot
expose — the shortened log still verifies. Signed artifacts close that gap: trace exports and
trust reports embed, **inside the signed body**,

```json
"egress_anchor": {"tip_hash": "<h of the last line at export time>", "line_count": <n>}
```

The field is absent (never faked) when no durable log existed. To check: verify the chain, and
confirm `tip_hash` appears as some line's `h` (`saturn_verify.py --egress-log egress.log
--expect-tip <tip_hash>`). If the chain never reaches the anchored tip, the log was truncated or
rewritten *after* that artifact was signed. An anchor proves the log back to its own tip only;
events after it are covered by later anchors.

## Versioning

`saturn-artifact/1` names everything on this page: the layout, the canonicalization, the digest,
the signature layering, and the chain rule. Any change to any of them bumps the suffix; verifiers
should accept marker-less artifacts as `/1`.
