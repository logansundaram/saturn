from config import get_config
from core import continuation, provenance
from core.plan_context import WRITE_TOOLS
from core.state import AgentState, incident_steps, unfinished_steps
from textutil import clip, parse_doc_sources, split_call_result
from core.llms import (
    get_model,
    model_id,
    extract_tok_per_sec,
    extract_prompt_tokens,
)
from core.messages import synthesize_sys_msg
from langchain.messages import HumanMessage, AIMessage, ToolMessage


# ── answer provenance (runtime.citations, default on) ─────────────────────────────────────────
# The gathered material handed to the synthesizer is numbered ([1], [2], …) so the model can cite
# the matching marker inline, and a "Sources" footer mapping each number back to the tool call /
# document that produced it is appended to the final answer. The footer is built mechanically from
# the trace accumulators — it is a receipt of what informed the answer, present whether or not the
# model cited inline, so the answer always carries its evidence (the same provenance /trace can
# reconstruct, but on the answer itself). `/config runtime.citations false` restores the exact
# pre-citation prompt sections and an unadorned answer.

_MAX_SOURCE_LABEL = 100

# Caps for the plan-outcomes block: a tool step's full observation already rides the numbered
# "Tool results" section, so its row here is a short pointer; a reasoning step's result exists
# ONLY on the step, so it gets the full (but still bounded) text.
_TOOL_STEP_RESULT_CAP = 400
_REASONING_STEP_RESULT_CAP = 2000

# Read-back cap for the verified-writes ground-truth block.
_VERIFY_CAP = 300


def _label_clamp(label: str, cap: int = _MAX_SOURCE_LABEL) -> str:
    return clip(label, cap)


def _tool_source_label(result) -> str:
    """Provenance label for one tool_results entry. Entries are `name(args) -> result` strings
    (nodes/tools.py pairs them on purpose); the call repr before the arrow is the label — split
    via textutil.split_call_result, THE one parser of that serialization."""
    return _label_clamp(split_call_result(result)[0])


def _doc_source_label(observation) -> str:
    """Provenance label for one retrieval observation: the distinct `[source: …]` names inside it
    (one search_knowledge_base call returns several chunks, usually from a handful of files).
    Parsed via textutil.parse_doc_sources — the parse half of the marker pair knowledge.py
    builds with doc_source_label, so the two sides can't drift."""
    names = parse_doc_sources(observation)
    if names:
        return _label_clamp("knowledge base: " + ", ".join(names))
    return "knowledge base passage"


def build_sources(tool_results, documents_retrieved):
    """Number everything the synthesizer is given, in the order it sees it.

    Returns (numbered_tool_results, numbered_docs, sources) where the numbered lists are the
    prompt-ready `[n] …` strings and `sources` is the [(n, label)] registry the footer renders.
    One shared numbering across both sections so an inline `[4]` is unambiguous."""
    sources: list[tuple[int, str]] = []
    numbered_tools: list[str] = []
    for r in tool_results or []:
        n = len(sources) + 1
        sources.append((n, _tool_source_label(r)))
        numbered_tools.append(f"[{n}] {r}")
    numbered_docs: list[str] = []
    for d in documents_retrieved or []:
        n = len(sources) + 1
        sources.append((n, _doc_source_label(d)))
        numbered_docs.append(f"[{n}] {d}")
    return numbered_tools, numbered_docs, sources


def sources_footer(sources) -> str:
    """The `Sources:` block appended to the answer — the mechanical map from each inline [n] to
    the tool call / document behind it. Empty string when nothing was gathered."""
    if not sources:
        return ""
    return "Sources:\n" + "\n".join(f"  [{n}] {label}" for n, label in sources)


def _gathered_section(items, numbered, citations, name):
    """One gathered-material prompt section ("Tool results" / "Retrieved documents"), or None
    when nothing was gathered. Keyed on the section NAME only — the citation-instruction suffix
    is deliberately byte-identical across both sections, and the `numbered` list must come from
    build_sources so the prompt's [n] markers and the Sources footer stay in lockstep."""
    if not items:
        return None
    if citations:
        return HumanMessage(
            content=f"{name} (numbered — cite the matching [n] after claims drawn from them):\n"
            + "\n\n".join(numbered)
        )
    return HumanMessage(content=f"{name}:\n" + "\n\n".join(map(str, items)))


def plan_outcomes_block(plan) -> str:
    """The completed plan as a step -> outcome narrative — the engine's data bus rendered for
    the synthesizer. A reasoning step's result exists ONLY here (it never rode tool_results), so
    it gets the fuller cap; a tool step's row is a bounded pointer to the numbered sections.
    A step that never ran says so explicitly."""
    lines = []
    for s in plan or []:
        result = s.get("result")
        if result is None:
            outcome = "(never ran — the turn ended before this step)"
        else:
            cap = (
                _REASONING_STEP_RESULT_CAP
                if not s.get("intended_tool")
                else _TOOL_STEP_RESULT_CAP
            )
            outcome = clip(" ".join(str(result).split()), cap)
        lines.append(f"- {s.get('label')} -> {outcome}")
    return "\n".join(lines) or "(no steps were run)"


def incidents_block(plan) -> list[str]:
    """One line per incident the answer must disclose: steps that were skipped, blocked,
    errored, cancelled — or never ran at all (iteration cap / abort)."""
    out = [
        f"step {s.get('step_id')} ({s.get('label')}): {s.get('result')}"
        for s in incident_steps(plan)
    ]
    out += [
        f"step {s.get('step_id')} ({s.get('label')}): never ran — the turn ended before it"
        for s in unfinished_steps(plan)
    ]
    return out


def verify_writes(state: AgentState) -> str:
    """Ground truth for the answer's file claims: re-read every file this turn actually wrote
    (successful write_file/edit_file calls, from tool_events) and quote what it NOW contains —
    so the answer describes real file contents, not the step log's intentions. Best-effort:
    an unreadable file just drops out of the block."""
    lines: list[str] = []
    seen: set = set()
    for ev in state.get("tool_events") or []:
        if not isinstance(ev, dict) or not ev.get("ok"):
            continue
        if ev.get("name") not in WRITE_TOOLS:
            continue
        args = ev.get("args") or {}
        path = args.get("file_path")
        if not path or path in seen:
            continue
        # The tools report refusals as ordinary strings (ok=True), so check the result text for
        # the EXACT success markers (tools/files.py return strings). A loose "File " prefix
        # would also match edit_file's "File not found: …" failure and quote a file the failed
        # edit never touched.
        preview = str(ev.get("result") or "")
        if not preview.startswith(
            (
                "File overwritten successfully",
                "File created successfully",
                "Content appended",
                "Edited ",
            )
        ):
            continue
        seen.add(path)
        try:
            from tools.registry import tools_by_name

            content = str(tools_by_name["read_file"].invoke({"file_path": path})).strip()
        except Exception:
            continue
        lines.append(f"- {path} now contains: {clip(content, _VERIFY_CAP)!r}")
    return "\n".join(lines)


def cancel_orphaned_calls(last) -> list:
    """Cancellation ToolMessages for a trailing AIMessage's unanswered tool_calls (empty when
    there are none). Nothing can have answered a TRAILING message's calls, so every call gets
    one. Pure helper so the orphan guard is testable without an LLM."""
    if not isinstance(last, AIMessage):
        return []
    return [
        ToolMessage(
            content=(
                "Not executed — the turn ended (iteration limit) before this "
                "call could run."
            ),
            tool_call_id=tc["id"],
            name=tc.get("name", ""),
        )
        for tc in (getattr(last, "tool_calls", None) or [])
    ]


# ── interrupt-and-correct plumbing (token steering) ───────────────────────────────────────────
# The answer streams into a provenance-tagged buffer (core/provenance.py). While it streams, the
# freeze latch (core/continuation.FreezeController — set by Esc via tui/typeahead) is ARMED; a
# freeze stops the stream cleanly and routes to the answer_gate edit interrupt
# (route_after_synthesize below). The gate hands back the human-edited buffer and this node runs
# again: prompt assembly is deterministic, so the re-entry rebuilds the identical history and
# CONTINUES the edited prefix through the raw-mode continuation primitive
# (core/continuation.continue_from) — never a fresh answer. The latch arms only for
# template-supported synthesizer models, so Esc never promises an editor that can't resume.


def _token_sink():
    """The custom-stream writer continuation tokens ride to the UI: the raw-mode stream is not a
    LangChain chat call, so LangGraph's messages mode never sees it — run_turn streams the
    "custom" channel instead and forwards `{"answer_token": …}` payloads to the same on_token.
    No-op outside a streaming graph context (unit tests, /retry's direct node call)."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return None
    if writer is None:
        return None

    def sink(text: str) -> None:
        try:
            writer({"answer_token": text})
        except Exception:
            pass

    return sink


def _stream_first_pass(llm_input, freeze):
    """The chat-path stream (tokens reach the UI via LangGraph messages mode, unchanged),
    polling the freeze latch per chunk. Returns (buffer, frozen, aggregated_message)."""
    model = get_model("synthesizer")
    buf = provenance.new_buffer()
    aggregated = None
    frozen = False
    gen = model.stream(llm_input)
    try:
        for chunk in gen:
            aggregated = chunk if aggregated is None else aggregated + chunk
            text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            if text:
                buf = provenance.append_model(buf, text)
            if freeze is not None and freeze.requested():
                frozen = True  # stop pulling tokens; closing the generator stops the decode
                break
    finally:
        try:
            gen.close()
        except Exception:
            pass
    if aggregated is None and not frozen:  # a model that streamed nothing — blocking fallback
        aggregated = model.invoke(llm_input)
        text = aggregated.content if isinstance(aggregated.content, str) else str(aggregated.content)
        buf = provenance.append_model(buf, text)
    return buf, frozen, aggregated


def _stream_continuation(model_name: str, llm_input, buf: dict, freeze):
    """Raw-mode prefix continuation over the SAME assembled history: the model resumes the
    (human-edited) buffer text as its own in-progress turn. Tokens go out through the custom
    stream channel; the freeze latch is polled per chunk (the user may freeze again). Returns
    (buffer, frozen, meta) — meta is the daemon's final stats for the tok/s + context gauges."""
    sink = _token_sink()
    stream = continuation.continue_from(model_name, llm_input, buf.get("text", ""))
    frozen = False
    try:
        for text in stream:
            buf = provenance.append_model(buf, text)
            if sink is not None:
                sink(text)
            if freeze is not None and freeze.requested():
                frozen = True
                break
    finally:
        stream.close()
    return buf, frozen, stream.meta


def _final_updates(buf: dict, incidents, sources, cancelled, *,
                   tok_per_sec: float, context_tokens: int,
                   response_metadata=None, usage_metadata=None) -> dict:
    """The turn's terminal state delta: the buffer text becomes the recorded AIMessage (with the
    mechanical incidents note + Sources footer appended — trailers live on the MESSAGE, never in
    the buffer, so the provenance spans keep indexing the prose exactly), and the buffer itself
    is kept on state as `complete` so the answer render and the trace carry the human spans."""
    content = buf.get("text", "")

    # A mechanical incidents note under the answer, mirroring the prompt-level disclosure: the
    # user sees what could not be completed even when the model soft-pedals it.
    if incidents and content.strip():
        content = content.rstrip() + "\n\nNote — the following could not be completed:\n" + "\n".join(
            f"- {i}" for i in incidents
        )

    # Append the provenance footer to the RECORDED answer (state/trace/autosave/headless all carry
    # it). The live token stream has already rendered without it, so the loop re-renders the final
    # message on finish (ui.ResponseStream.finish(final_text) — see app/repl.py) and the footer
    # appears there. Skipped when nothing was gathered — a pure-knowledge answer has no sources.
    footer = sources_footer(sources)
    if footer and content.strip():
        content = content.rstrip() + "\n\n" + footer

    msg_kwargs = {"response_metadata": response_metadata or {}}
    if usage_metadata:
        msg_kwargs["usage_metadata"] = usage_metadata
    llm_response = AIMessage(content=content, **msg_kwargs)

    return {
        "messages": [*cancelled, llm_response],
        "answer_buffer": {**buf, "state": "complete"},
        "tok_per_sec": tok_per_sec,
        "context_tokens": context_tokens,
    }


def synthesize_node(state: AgentState):
    query = state["current_query"]
    context = state["context"]
    plan = state.get("plan", [])
    tool_results = state.get("tool_results", [])
    documents_retrieved = state.get("documents_retrieved", [])

    citations = bool(get_config().get("runtime.citations", True))
    sources: list[tuple[int, str]] = []
    numbered_tools: list[str] = []
    numbered_docs: list[str] = []
    if citations:
        numbered_tools, numbered_docs, sources = build_sources(
            tool_results, documents_retrieved
        )

    llm_input = [synthesize_sys_msg]

    if context:
        llm_input.append(HumanMessage(content=f"Relevant context:\n{context}"))

    # The completed plan — the data bus — as a step -> outcome narrative. Reasoning-step results
    # live ONLY here; tool observations are pointed at the numbered sections below.
    llm_input.append(
        HumanMessage(content="Completed steps and results:\n" + plan_outcomes_block(plan))
    )

    # Tools first, then documents — build_sources numbers in exactly this order, so the section
    # order is load-bearing for the inline [n] markers.
    section = _gathered_section(tool_results, numbered_tools, citations, "Tool results")
    if section is not None:
        llm_input.append(section)
    # tool_node stores retrieval results as pre-formatted strings (source + text),
    # not Document objects.
    section = _gathered_section(
        documents_retrieved, numbered_docs, citations, "Retrieved documents"
    )
    if section is not None:
        llm_input.append(section)

    # Ground truth for written files: what they ACTUALLY contain now — the answer must describe
    # file contents from this, never from the step log's intentions.
    verified = verify_writes(state)
    if verified:
        llm_input.append(
            HumanMessage(
                content="Ground truth — files written during the plan ACTUALLY contain "
                "the following now; describe file contents from this, not from "
                "the step log:\n" + verified
            )
        )

    # Plan-review vetoes: steps the USER removed at the plan-review editor (state["plan_vetoes"],
    # via plan_gate). The answer must describe that work as skipped at the user's own request —
    # never as a failure, missing work, or something to apologize for.
    vetoes = [str(v).strip() for v in state.get("plan_vetoes") or [] if str(v).strip()]
    if vetoes:
        llm_input.append(
            HumanMessage(
                content="Plan-review note: the user themselves REMOVED these planned steps at "
                "the plan-review prompt, so they were deliberately not done at the "
                "user's request. If relevant, describe them as skipped at the user's "
                "request — never as failures or missing work:\n"
                + "\n".join(f"- {v}" for v in vetoes)
            )
        )

    # Incidents: actions that did NOT complete (gate rejections, write-gate skips, errors,
    # cancellations, never-ran steps). The answer must state these plainly — an incident does
    # not cancel the rest of the request, but it must never be presented as done.
    incidents = incidents_block(plan)
    if incidents:
        llm_input.append(
            HumanMessage(
                content="INCIDENTS — these actions did NOT complete. State plainly what "
                "was not done and why; do NOT claim any of these succeeded. An "
                "incident does not cancel the rest of the request: still answer "
                "it from the results of the steps that DID complete:\n"
                + "\n".join(incidents)
            )
        )

    # Forced landing mid-decision: the iteration cap can route here while the trailing AIMessage
    # still carries an unanswered tool_call. Close each orphaned call with a cancellation
    # ToolMessage now, or the carried conversation (and its autosave) holds an assistant
    # tool_use with no tool_result — a hard 400 on the next cloud-provider turn and a /resume
    # that reproduces it.
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    cancelled = cancel_orphaned_calls(last)

    llm_input.append(HumanMessage(content=f"Current user query:\n{query}"))

    # ── interrupt-and-correct: which pass is this? ────────────────────────────────────────────
    # The prompt assembly above is deterministic, so every re-entry after an answer_gate edit
    # rebuilds the identical history — only the generation below differs by the buffer's state:
    #   None / anything else  first pass: chat-path stream into a fresh provenance buffer
    #   "resume"              continue the (human-edited) buffer text via raw-mode continuation
    #   "done"                the user accepted the frozen text as the answer — no generation
    buf = state.get("answer_buffer")
    buf_state = buf.get("state") if isinstance(buf, dict) else None
    model_name = model_id("synthesizer")
    supported = continuation.supports(model_name)

    if buf_state == "done" or (buf_state == "resume" and not supported):
        # "resume" without template support only happens if the model was swapped mid-turn —
        # finalize with the text as it stands rather than fabricate a continuation path.
        return _final_updates(
            dict(buf), incidents, sources, cancelled,
            tok_per_sec=float(state.get("tok_per_sec", 0.0) or 0.0),
            context_tokens=int(state.get("context_tokens", 0) or 0),
        )

    # Stream the answer so the UI can render it token-by-token: LangGraph surfaces the chat
    # path's chunks via stream_mode="messages" (filtered to this node in app/turn.run_turn ->
    # on_token); the continuation path rides the "custom" channel (_token_sink). We still
    # aggregate the whole text into the buffer and return a complete AIMessage, so
    # state["messages"] / the trace / autosave see exactly what the old .invoke() path produced
    # — the streaming is purely additive. The freeze latch is armed only around the stream
    # (and only for supported models), so Esc anywhere else keeps its pause/steer meaning.
    freeze = continuation.get_freeze_controller() if supported else None
    if freeze is not None:
        freeze.arm()
    try:
        if buf_state == "resume":
            buf, frozen, meta = _stream_continuation(model_name, llm_input, dict(buf), freeze)
            tok_per_sec = continuation.extract_tok_per_sec(meta)
            context_tokens = int(meta.get("prompt_eval_count") or 0)
            response_metadata = {k: meta[k] for k in
                                 ("eval_count", "eval_duration", "prompt_eval_count", "done_reason")
                                 if k in meta}
            usage_metadata = None
        else:
            buf, frozen, aggregated = _stream_first_pass(llm_input, freeze)
            tok_per_sec = extract_tok_per_sec(aggregated)
            context_tokens = extract_prompt_tokens(aggregated)
            response_metadata = getattr(aggregated, "response_metadata", {}) or {}
            usage_metadata = getattr(aggregated, "usage_metadata", None)
    finally:
        if freeze is not None:
            freeze.disarm()  # also clears a request that landed after the stream ended

    if frozen:
        # Stop was clean; hand the buffer to the answer_gate edit interrupt
        # (route_after_synthesize). No message lands yet — the turn is mid-answer.
        return {"answer_buffer": {**buf, "state": "frozen", "edited": False}}

    return _final_updates(
        buf, incidents, sources, cancelled,
        tok_per_sec=tok_per_sec, context_tokens=context_tokens,
        response_metadata=response_metadata, usage_metadata=usage_metadata,
    )


def route_after_synthesize(state: AgentState) -> str:
    """After synthesize: a frozen buffer routes to the answer_gate edit interrupt (the user
    pressed Esc mid-stream); anything else ends the turn."""
    buf = state.get("answer_buffer")
    if isinstance(buf, dict) and buf.get("state") == "frozen":
        return "answer_gate"
    return "end"
