"""
Outbound redaction — strip secrets from text before it leaves the machine to an off-machine model.

An off-machine inference endpoint is the credibility gap in a privacy-first agent: the moment
prompts + context cross the network, the boundary needs a guard. Today that boundary is a REMOTE
Ollama (`OLLAMA_HOST` off-machine) plus the http/sse MCP arg scan and the gate's secret warning —
cloud providers are SHELVED (2026-07-03), and when they return this module guards them again
unchanged. It scans outgoing message content for things that should never leave — API keys, bearer
tokens, private-key blocks, JWTs, emails — and, depending on `runtime.redaction`, either reports
them or replaces them with a `[REDACTED:<kind>]` placeholder before the send.

  off     no scan (the default — most users run fully local, nothing leaves anyway).
  warn    scan and COUNT (the count is recorded to the egress ledger so it's visible in
          /privacy egress), but send the text unmodified — visibility without altering what the
          model sees.
  redact  scan and REPLACE each match with a placeholder, then send the redacted text.

Wired in `llms.py`: every cloud model is wrapped so `process_messages` runs at the boundary, the
ONE place all nodes funnel through (so a secret can't leak via the planner, agent, judge, or
synthesizer independently). Local (Ollama) models are never wrapped — there is no boundary to
guard. The mode is configured via `/config runtime.redaction` (a trust key — persists only with
an explicit --save); the `/privacy redact` command front end was CUT 2026-07-16 as dormant since
the cloud shelve.

Patterns are deliberately conservative (high-signal prefixes, length floors) to avoid false
positives that would mangle a legitimate prompt — this strips obvious secrets, it is not a DLP
engine. Imports only config, so it's a safe leaf for llms.py to import.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import get_config
from textutil import iter_strings

_MODES = ("off", "warn", "redact")


@dataclass(frozen=True)
class _Pattern:
    kind: str
    regex: "re.Pattern[str]"


# Ordered most-specific-first. Each is anchored on a high-signal prefix or structure with a length
# floor, so ordinary prose doesn't trip it. The credit-card / generic-number space is deliberately
# omitted — too many false positives to be worth mangling real prompts.
_PATTERNS: list[_Pattern] = [
    _Pattern("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    # Negative lookahead so an Anthropic key (sk-ant-…) isn't ALSO counted as an OpenAI key — the
    # `redact` path already avoids the double-hit (it substitutes anthropic first), but `scan`
    # tests patterns independently, so exclude the overlap explicitly.
    _Pattern("openai-key", re.compile(r"sk-(?!ant-)(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    _Pattern("tavily-key", re.compile(r"tvly-[A-Za-z0-9_\-]{16,}")),
    _Pattern("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    _Pattern("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    _Pattern("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    _Pattern("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    _Pattern("private-key", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    _Pattern("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    _Pattern("bearer-token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}")),
    _Pattern("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
]


@dataclass(frozen=True)
class Finding:
    kind: str
    preview: str  # a masked snippet, safe to display (the secret itself is never shown in full)


def mode() -> str:
    """The active redaction mode (`runtime.redaction`): off | warn | redact. Read live."""
    m = str(get_config().get("runtime.redaction", "off") or "off").lower()
    return m if m in _MODES else "off"


def active() -> bool:
    """Whether redaction does anything this turn (mode is warn or redact)."""
    return mode() != "off"


def _mask(match: str) -> str:
    """A display-safe preview of a matched secret — textutil.mask_secret, THE one masking rule
    (shared with env_keys' key listing, so a tightening decision reaches both surfaces)."""
    from textutil import mask_secret

    return mask_secret(match)


def scan(text: str) -> list[Finding]:
    """All secret-like spans in `text`, as display-safe findings (never the raw secret)."""
    if not text:
        return []
    out: list[Finding] = []
    for pat in _PATTERNS:
        for m in pat.regex.finditer(text):
            out.append(Finding(kind=pat.kind, preview=_mask(m.group(0))))
    return out


def scan_args(args) -> list[Finding]:
    """Secret-like values anywhere inside a tool call's arguments. Used by the approval gate to
    warn when the call the user is about to approve would carry a secret out (an MCP call's
    args, a run_shell command with a token inline). Walks the args tree with
    `textutil.iter_strings` — THE one recursive string-leaf walker over call args — so every
    args scan agrees about what counts as argument content. Display-safe findings only, like
    `scan`."""
    return [f for s in iter_strings(args) for f in scan(s)]


def redact(text: str) -> "tuple[str, list[Finding]]":
    """Replace every secret-like span in `text` with `[REDACTED:<kind>]`. Returns the new text and
    the findings. Applied in `redact` mode at the cloud boundary."""
    if not text:
        return text, []
    findings: list[Finding] = []

    def _sub_factory(kind: str):
        def _sub(m: "re.Match[str]") -> str:
            findings.append(Finding(kind=kind, preview=_mask(m.group(0))))
            return f"[REDACTED:{kind}]"
        return _sub

    new = text
    for pat in _PATTERNS:
        new = pat.regex.sub(_sub_factory(pat.kind), new)
    return new, findings


def process_messages(messages: list) -> "tuple[list, int]":
    """Apply the active mode to a list of LangChain messages bound for a cloud model.

    Returns (messages_to_send, n_findings). In `off` it's a pure pass-through (0). In `warn` it
    scans every string content and returns the COUNT but the original messages (visibility only).
    In `redact` it returns copies with secrets replaced by placeholders. Message objects are never
    mutated in place — redacted copies are made via the pydantic model_copy path so the scratchpad
    the rest of the loop holds is untouched."""
    m = mode()
    if m == "off" or not messages:
        return messages, 0

    total = 0
    if m == "warn":
        for msg in messages:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                total += len(scan(content))
        return messages, total

    # redact
    out = []
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            new_content, findings = redact(content)
            if findings:
                total += len(findings)
                msg = _with_content(msg, new_content)
        out.append(msg)
    return out, total


def _with_content(msg, new_content: str):
    """Return a copy of a LangChain message with replaced content, tolerant across pydantic
    versions. Falls back to mutating a shallow copy, then to the original, so redaction can never
    crash a turn."""
    for attempt in ("model_copy", "copy"):
        fn = getattr(msg, attempt, None)
        if fn is None:
            continue
        try:
            return fn(update={"content": new_content})
        except Exception:
            continue
    try:
        import copy as _copy
        clone = _copy.copy(msg)
        clone.content = new_content
        return clone
    except Exception:
        return msg
