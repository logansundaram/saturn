"""
The interrupt-and-correct continuation engine (token steering, Stages A+B).

Interrupt-and-correct is exactly three moves: **stop** generation cleanly, **edit** the
assistant-message prefix (delete bad text, splice in the user's words), **re-prime** the model to
continue that exact prefix — not start a new turn. Because generation is prefix continuation, the
model cannot distinguish authored-by-model from authored-by-human characters in its own prefix,
so the resumed stream is seamless *by construction*. This module owns the two capabilities the
feature reduces to:

  - `continue_from(...)` — the continuation primitive: render the history + edited prefix
    through the ONE raw-prompt chokepoint (core/chat_template.py), stream it through Ollama
    `/api/generate` with `raw: true`, halting at the family's end-of-turn `stop` string. The
    whole prefix is re-tokenized by the daemon on every resume (Stage A/B reprocess the prefix;
    the KV-cache fast path is explicitly deferred), so character-level edits can never produce
    an invalid token sequence.
  - `FreezeController` — the stop signal: the one latch the console reader (tui/typeahead) sets
    and the synthesize node's streaming loops poll. Armed only while an answer is actually
    streaming from a template-supported model; a freeze on a disarmed latch reports False so the
    Esc key falls back to its plan-review meaning.

Trust boundary: a LOOPBACK Ollama is not egress — nothing is recorded, exactly like the chat
path (`llms._wrap_ollama`). Behind a remote `OLLAMA_HOST` the same rules as
`llms._CloudBoundaryModel` apply: air-gap refuses the call outright, and the outgoing prompt is
run through `trust.redaction` (warn counts, redact rewrites) before `trust.egress.record` logs
the send. `num_ctx` rides every request explicitly — the daemon otherwise front-truncates
silently at its own default (see the pinned-behavior notes in core/chat_template.py).
"""

from __future__ import annotations

import json
import threading
from typing import Iterator, Optional

import httpx

from config import get_config
from core import chat_template
from trust import egress, redaction


def supports(model: str) -> bool:
    """Whether interrupt-and-correct is offered for `model` (it has a template entry; the
    contract test — utilities/continuation_contract.py — defines the officially-supported set)."""
    return chat_template.supported(model)


def _endpoint() -> str:
    """The Ollama base URL, normalized to carry a scheme (OLLAMA_HOST may be a bare host:port)."""
    ep = egress.ollama_endpoint()
    return ep if "://" in ep else "http://" + ep


class ContinuationStream:
    """One cancellable raw-mode generation: iterate it for text chunks; `close()` (or just
    breaking out of the loop — closing is idempotent and also runs on GC) stops requesting
    tokens and tears the HTTP stream down. After exhaustion, `meta` carries the daemon's final
    stats (eval_count/eval_duration/prompt_eval_count) and `done_reason` why it stopped —
    the same numbers the chat path reads for the tok/s and context gauges.

    The interface deliberately assumes nothing beyond "an iterator of text the owner may stop
    pulling from": a later llama-cpp backend maps onto it without touching any caller."""

    def __init__(self, url: str, body: dict, timeout: "httpx.Timeout | None"):
        self._url = url
        self._body = body
        self._timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._response = None
        self.meta: dict = {}
        self.done_reason: str = ""

    def __iter__(self) -> Iterator[str]:
        self._client = httpx.Client(timeout=self._timeout)
        try:
            with self._client.stream("POST", self._url, json=self._body) as r:
                self._response = r
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        raise RuntimeError(f"Ollama raw generation failed: {data['error']}")
                    chunk = data.get("response") or ""
                    if data.get("done"):
                        self.meta = data
                        self.done_reason = str(data.get("done_reason") or "")
                        if chunk:
                            yield chunk
                        break
                    if chunk:
                        yield chunk
        finally:
            self.close()

    def close(self) -> None:
        """Stop the generation: close the HTTP stream (the daemon stops decoding when the
        client goes away) and the client. Idempotent; never raises."""
        for obj, attr in ((self._response, "close"), (self._client, "close")):
            try:
                if obj is not None:
                    getattr(obj, attr)()
            except Exception:
                pass
        self._response = None
        self._client = None


def continue_from(model: str, messages: list, edited_prefix: str,
                  options: "dict | None" = None) -> ContinuationStream:
    """THE continuation primitive: re-prime `model` to continue `edited_prefix` as its own
    in-progress assistant turn over `messages` (the same history the first pass saw). Returns a
    cancellable token stream; raises chat_template.UnsupportedModel for models outside the
    registry and RuntimeError when the air-gap blocks a remote daemon.

    `edited_prefix` is a STRING — the daemon re-tokenizes the whole rendered prompt, so an edit
    that cut mid-token simply re-tokenizes to a valid sequence (never token-index surgery).
    `options` overlays the request options (the contract test pins temperature with it)."""
    template = chat_template.template_for(model)
    prompt = template.render(chat_template.normalize_turns(messages), edited_prefix)

    # The trust boundary, mirroring llms._CloudBoundaryModel for the one raw-mode caller:
    # loopback = no boundary; remote = air-gap refusal, then redaction, then the egress ledger.
    if not egress.ollama_is_local():
        host = f"ollama @ {egress.ollama_endpoint()}"
        if egress.airgap_on():
            egress.record("llm", host, f"continuation → {model}",
                          provider="ollama", status=egress.BLOCKED)
            raise RuntimeError(
                f"Air-gap is ON — OLLAMA_HOST points off this machine ({egress.ollama_endpoint()}), "
                f"so continuing the answer on {model} would cross the network. Unset OLLAMA_HOST "
                f"or turn the air-gap off with `/privacy airgap off`."
            )
        n_red = 0
        if redaction.mode() == "redact":
            prompt, findings = redaction.redact(prompt)
            n_red = len(findings)
        elif redaction.mode() == "warn":
            n_red = len(redaction.scan(prompt))
        egress.record("llm", host, f"continuation → {model}", provider="ollama",
                      n_bytes=len(prompt), redactions=n_red)

    opts = {"num_ctx": get_config().num_ctx_for(model)}
    opts.update(options or {})
    body = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": True,
        "options": opts,
        "stop": list(template.stop),
    }
    t = get_config().llm_timeout
    timeout = httpx.Timeout(t, connect=min(10.0, t)) if t else None
    return ContinuationStream(f"{_endpoint()}/api/generate", body, timeout)


def extract_tok_per_sec(meta: dict) -> float:
    """tokens/second from a finished ContinuationStream's meta (same fields the chat path's
    response_metadata carries); 0.0 when unavailable."""
    count = meta.get("eval_count") or 0
    dur = meta.get("eval_duration") or 0
    return count / (dur / 1e9) if dur else 0.0


# ── the freeze latch ───────────────────────────────────────────────────────────────────────────
# The stop half of stop→edit→continue. The synthesize node ARMS the latch around each streaming
# segment (first pass and every continuation) — and only when the synthesizer model is
# template-supported, so Esc never promises an editor that can't resume. The console reader
# (typeahead.InputQueue) calls freeze() on Esc: True consumed the keypress as a freeze; False
# means "not streaming right now" and Esc falls through to its usual pause/steer meaning. The
# streaming loop polls requested() per chunk and stops pulling tokens — that, plus
# ContinuationStream.close(), IS the clean stop (§5.2's generation controller).
#
# Same singleton rationale as plan_ops.PauseController: the CLI runs exactly one turn at a time,
# and the latch must be reachable from both the reader thread and the node without riding
# checkpointed state.


class FreezeController:
    """Thread-safe freeze latch: armed while an answer is streaming; `freeze()` requests the
    stop; the streaming loop polls `requested()` and the node `clear()`s once handled."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._armed = False
        self._requested = False

    def arm(self) -> None:
        with self._lock:
            self._armed = True
            self._requested = False

    def disarm(self) -> None:
        with self._lock:
            self._armed = False
            self._requested = False

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def freeze(self) -> bool:
        """Request a freeze. True only when armed (the caller's Esc was consumed); False lets
        the caller fall back to the pause/steer meaning of the key."""
        with self._lock:
            if not self._armed:
                return False
            self._requested = True
            return True

    def requested(self) -> bool:
        with self._lock:
            return self._requested

    def clear(self) -> None:
        with self._lock:
            self._requested = False


_controller = FreezeController()


def get_freeze_controller() -> FreezeController:
    return _controller
