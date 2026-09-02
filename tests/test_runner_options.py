"""
The runner-option chokepoint (2026-09-02): every chat request Saturn sends to Ollama carries the
same `draft_num_predict` setting while confidence grading is on — whoever the caller is and
whether or not it passes its own `options`.

Why: Ollama 0.33.2 treats `draft_num_predict` as a runner LOAD option. A request that carries
`draft_num_predict: 0` relaunches llama-server without its `--spec-type draft-mtp` flags; the
next request without the key relaunches it with them. Applying the key on the logprob sites
only (the synthesizer's stream) made every qwen3.8:27b turn reload the weights at least twice.
The model built by `core.llms._build` is the one place all chat traffic funnels through
(`ChatOllama._chat_params` serves invoke, stream and their async twins, and survives
`bind_tools` / `with_structured_output`), so it is where the option lives.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from core import confidence, llms


class _FakeClient:
    """Stands in for ChatOllama's ollama.Client: records the request, answers one done chunk."""

    def __init__(self):
        self.requests: list[dict] = []

    def chat(self, **params):
        self.requests.append(params)
        chunk = {
            "model": params["model"], "created_at": "2026-09-02T00:00:00Z",
            "message": {"role": "assistant", "content": "hi"},
            "done": True, "done_reason": "stop",
        }
        if params.get("stream", True):
            return iter([chunk])
        return chunk


@pytest.fixture
def model(monkeypatch, isolated_paths):
    monkeypatch.setattr(llms.egress, "ollama_is_local", lambda: True)
    m = llms._build("ollama", "fake:1b")
    client = _FakeClient()
    m._client = client
    return m, client


def test_a_bare_invoke_turns_drafting_off_and_keeps_num_ctx(model, monkeypatch):
    """The callers that pass no options at all (the compaction summary, the knowledge draft,
    the synthesizer's blocking fallback) get the constructor's num_ctx PLUS the runner option."""
    m, client = model
    monkeypatch.setattr(confidence, "enabled", lambda: True)

    m.invoke([HumanMessage(content="hi")])

    opts = client.requests[0]["options"]
    assert opts["draft_num_predict"] == 0
    assert opts.get("num_ctx", 0) > 0


def test_per_call_options_carry_the_runner_option_and_keep_the_callers_keys(model, monkeypatch):
    """`_invoke_kwargs` builds a FULL per-call options dict (temperature, num_ctx, num_predict);
    the runner option rides along without dropping any of it."""
    m, client = model
    monkeypatch.setattr(confidence, "enabled", lambda: True)

    list(m.stream([HumanMessage(content="hi")],
                  options={"temperature": 0.2, "num_ctx": 4096, "num_predict": 64}))

    opts = client.requests[0]["options"]
    assert opts == {"temperature": 0.2, "num_ctx": 4096, "num_predict": 64,
                    "draft_num_predict": 0}


def test_with_grading_off_no_request_carries_the_runner_option(model, monkeypatch):
    """Off means the runner is loaded with the daemon's defaults for EVERY request — the point
    is that the setting never differs between two requests of one process."""
    m, client = model
    monkeypatch.setattr(confidence, "enabled", lambda: False)

    m.invoke([HumanMessage(content="hi")])
    m.invoke([HumanMessage(content="hi")], options={"temperature": 0.2, "num_ctx": 4096})

    assert all("draft_num_predict" not in r["options"] for r in client.requests)
    assert client.requests[1]["options"] == {"temperature": 0.2, "num_ctx": 4096}
