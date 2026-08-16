from types import SimpleNamespace

import diag
from config import get_config
from core import confidence, continuation, provenance
from core.plan_context import WRITE_TOOLS, authorization_basis
from textutil import figure_literals, untraceable_figures
from core.state import AgentState, incident_steps, unfinished_steps
from textutil import clip, parse_doc_sources, split_call_result
from core.llms import (
    get_model,
    generate,
    model_id,
    stream as llm_stream,
    extract_tok_per_sec,
    extract_prompt_tokens,
)
from core.structured import _invoke_kwargs, _model_tag
from core.messages import COMPUTED_CORRECTIVE, GROUNDING_CORRECTIVE, synthesize_sys_msg
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


# What the recorded answer says when the model generated nothing. A statement of fact (never
# invented prose) so the turn still reports what happened through the trailers below instead of
# returning a blank message — the provenance buffer keeps the empty text the model produced.
NO_ANSWER_TEXT = "No answer text was produced for this turn."

# The mechanical incidents note's header — one producer (this module), read by the tests.
INCIDENTS_NOTE_HEADER = "Note — the following could not be completed:"

# The groundedness gate's disclosure (2026-08-15, from the engine isolate): figures the answer
# states that survived a corrective regeneration without becoming traceable. The answer is not
# suppressed — it is MARKED, so an unverifiable number never reaches the user looking gathered.
GROUNDING_NOTE_HEADER = "Note — these figures could not be traced to any gathered result:"

# The inverse: a figure a tool PRODUCED that the answer then dropped, stated as a fact of the run.
COMPUTED_NOTE_HEADER = "Computed this turn, from the plan's own calculation step:"


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
    # Lazy imports (the execute-node pattern): the markers are the PRODUCER's own constants
    # (tools/files.WRITE_SUCCESS_MARKERS — one producer, one parser, like DECLINE_TEXT), so a
    # rewording of a files.py return string can never silently empty this block again.
    from tools.files import WRITE_SUCCESS_MARKERS
    from tools.registry import tools_by_name

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
        # the EXACT success markers. A loose "File " prefix would also match edit_file's
        # "File not found: …" failure and quote a file the failed edit never touched.
        preview = str(ev.get("result") or "")
        if not preview.startswith(WRITE_SUCCESS_MARKERS):
            continue
        seen.add(path)
        try:
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
    "custom" channel instead and forwards `{"answer_token": …}` payloads (plus each chunk's raw
    `logprobs`, for the live confidence marking) to the same on_token. No-op outside a streaming
    graph context (e.g. unit tests)."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return None
    if writer is None:
        return None

    def sink(text: str, logprobs=None) -> None:
        try:
            payload = {"answer_token": text}
            if logprobs:
                payload["logprobs"] = logprobs
            writer(payload)
        except Exception:
            pass

    return sink


def _stream_first_pass(llm_input, freeze):
    """The chat-path stream (tokens reach the UI via LangGraph messages mode, unchanged),
    polling the freeze latch per chunk. Returns (buffer, frozen, response_metadata,
    usage_metadata) — the metadata folded per chunk with a plain dict update.

    Deliberately NO AIMessageChunk aggregation: `aggregated + chunk` re-merges the whole
    accumulated message per token — and with logprobs on, merge_dicts re-copies the entire
    accumulated token table per token (quadratic in answer length), only for the merged
    logprobs to be stripped at the end. Ollama's stats ride the final chunk, so last-wins
    dict folding (logprobs excluded at the source) keeps everything the callers read.
    The buffer likewise grows IN PLACE (provenance.extend_model) — it stays private to this
    loop, so the copy-and-return contract isn't needed until it lands on state.

    Confidence grading (runtime.confidence): `logprobs=True` rides the call as a per-call kwarg
    (NOT inside `options`, so the constructor's num_ctx is untouched); each intermediate chunk
    then carries its token logprobs in response_metadata, aligned onto the buffer's confidence
    overlay as it lands. A daemon that doesn't answer just leaves the overlay empty."""
    model = get_model("synthesizer")
    buf = provenance.new_buffer()
    resp_meta: dict = {}
    usage = None
    got_chunk = False
    frozen = False
    seeker = continuation.FreezeSeeker(freeze)
    # The serving layer's per-task decisions ride the stream too: think OFF for the answer,
    # a num_predict bound, num_ctx (never a partial options dict); logprobs as a per-call kwarg.
    stream_kwargs = dict(_invoke_kwargs("synthesizer", None, 0.7, task="answer"))
    if confidence.enabled():
        stream_kwargs["logprobs"] = True
    gen = llm_stream(model, llm_input, tag=_model_tag("synthesizer"), **stream_kwargs)
    try:
        for chunk in gen:
            got_chunk = True
            md = getattr(chunk, "response_metadata", None) or {}
            lp = md.get("logprobs")
            for k, v in md.items():
                if k != "logprobs":
                    resp_meta[k] = v
            um = getattr(chunk, "usage_metadata", None)
            if um:
                usage = um
            text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            # The word-boundary freeze seek (continuation.FreezeSeeker): a freeze request lands
            # on a word boundary — a whitespace-led chunk closes the word and is NOT appended;
            # otherwise up to FREEZE_GRACE more chunks may land; a second Esc forces the cut.
            if text and seeker.before_chunk(buf.get("text", ""), text):
                frozen = True
                break
            if text:
                provenance.extend_model(
                    buf, text, confidence.align_chunk(text, lp) if lp else None
                )
            if seeker.after_chunk(buf.get("text", "")):
                frozen = True  # stop pulling tokens; closing the generator stops the decode
                break
    finally:
        try:
            gen.close()
        except Exception:
            pass
    if not got_chunk and not frozen:  # a model that streamed nothing — blocking fallback
        resp = model.invoke(llm_input)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        provenance.extend_model(buf, text)
        resp_meta = {
            k: v
            for k, v in (getattr(resp, "response_metadata", {}) or {}).items()
            if k != "logprobs"
        }
        usage = getattr(resp, "usage_metadata", None)
    return buf, frozen, resp_meta, usage


def _stream_continuation(model_name: str, llm_input, buf: dict, freeze):
    """Raw-mode prefix continuation over the SAME assembled history: the model resumes the
    (human-edited) buffer text as its own in-progress turn. Tokens go out through the custom
    stream channel; the freeze latch is polled per chunk (the user may freeze again). Returns
    (buffer, frozen, meta) — meta is the daemon's final stats for the tok/s + context gauges."""
    sink = _token_sink()
    # Clone once, extend in place per chunk (the caller's frozen buffer stays intact) — the
    # per-chunk copy-and-return append_model made a long resume quadratic in answer length.
    buf = provenance.clone(buf)
    stream = continuation.continue_from(model_name, llm_input, buf.get("text", ""))
    frozen = False
    seeker = continuation.FreezeSeeker(freeze)  # the same word-boundary seek as the first pass
    try:
        for text in stream:
            lp = stream.last_logprobs  # this chunk's logprobs (see ContinuationStream)
            if text and seeker.before_chunk(buf.get("text", ""), text):
                frozen = True  # a whitespace-led chunk closed the word: cut here, unappended
                break
            provenance.extend_model(
                buf, text, confidence.align_chunk(text, lp) if lp else None
            )
            if sink is not None:
                sink(text, lp)
            if seeker.after_chunk(buf.get("text", "")):
                frozen = True
                break
    finally:
        stream.close()
    return buf, frozen, stream.meta


def _final_updates(buf: dict, incidents, sources, cancelled, *,
                   tok_per_sec: float, context_tokens: int,
                   response_metadata=None, usage_metadata=None,
                   ungrounded=(), dropped=()) -> dict:
    """The turn's terminal state delta: the buffer text becomes the recorded AIMessage (with the
    mechanical incidents note + Sources footer appended — trailers live on the MESSAGE, never in
    the buffer, so the provenance spans keep indexing the prose exactly), and the buffer itself
    is kept on state as `complete` so the answer render and the trace carry the human spans."""
    content = buf.get("text", "")

    # An EMPTY generation must not silence the engine's own disclosures. Every trailer below used
    # to be guarded on `content.strip()` — on the model having written something — so a model
    # that returned nothing produced a turn with no answer, no statement that a rejected write
    # had not happened, and no sources: a fail-OPEN on the one path this node promises to be
    # fail-closed (measured live in the engine isolate: `held.guard.reject_write` returned ''
    # on one model while passing on two others). Each trailer is now gated on its OWN trigger,
    # and an empty answer becomes a stated fact — the buffer keeps the empty text (no invented
    # prose ever enters the provenance record).
    if not content.strip():
        content = NO_ANSWER_TEXT

    # The computed-value and groundedness disclosures (fail-closed by MARKING — the answer stays,
    # the unverifiable figure never passes as gathered).
    if dropped:
        content = content.rstrip() + f"\n\n{COMPUTED_NOTE_HEADER} " + ", ".join(dropped)
    if ungrounded:
        content = content.rstrip() + f"\n\n{GROUNDING_NOTE_HEADER} " + ", ".join(ungrounded)

    # A mechanical incidents note under the answer, mirroring the prompt-level disclosure: the
    # user sees what could not be completed even when the model soft-pedals it.
    if incidents:
        content = content.rstrip() + f"\n\n{INCIDENTS_NOTE_HEADER}\n" + "\n".join(
            f"- {i}" for i in incidents
        )

    # Append the provenance footer to the RECORDED answer (state/trace/autosave/headless all carry
    # it). The live token stream has already rendered without it, so the loop re-renders the final
    # message on finish (ui.ResponseStream.finish(final_text) — see app/repl.py) and the footer
    # appears there. Skipped when nothing was gathered — a pure-knowledge answer has no sources.
    footer = sources_footer(sources)
    if footer:
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


# ── the groundedness gate + the computed-value check (from the engine isolate, 2026-08-15) ─────
#
# Every other LLM boundary in this engine verifies its own output and retries with a specific
# corrective — core.structured on a parse failure, nodes.execute on a rejected tool call.
# Synthesis was the one that did neither, which is why a figure the model computed in prose
# reached the user indistinguishable from one a tool returned. Check → correct ONCE → disclose.
# The correction runs on a normal first pass only; on a RESUME the human edited the prefix and
# regenerating would discard their edit (their text outranks the engine's self-correction) — so
# the resume pass DETECTS and marks without regenerating.


def observation_pool(state: AgentState, query: str) -> str:
    """Everything a figure in the answer may legitimately have come from: the human's words and
    the OBSERVATIONS this turn gathered. Reasoning-step results are deliberately excluded — a
    "none" step's text is model text, and admitting it would let a fabricated figure launder
    itself (the reasoning step invents 515, the answer repeats it, 515 is now "traceable")."""
    parts = [str(query or "")]
    parts += [str(r) for r in state.get("tool_results") or []]
    parts += [str(d) for d in state.get("documents_retrieved") or []]
    parts += [
        str(s.get("result") or "")
        for s in state.get("plan") or []
        if s.get("intended_tool") and s.get("result") is not None
    ]
    return "\n".join(parts)


def gate_applies(state: AgentState) -> bool:
    """The gate runs only when the turn actually OBSERVED something: where nothing was gathered
    there is no ground truth to override, and a general-knowledge answer ("a 256-bit hash") draws
    on the model's own knowledge exactly as it should."""
    return any(
        s.get("intended_tool") and s.get("result") is not None
        for s in state.get("plan") or []
    )


def ungrounded_figures(buf: dict, state: AgentState, query: str) -> tuple:
    """Figures the answer states that trace to nothing this turn gathered (the DETECTION half)."""
    if not gate_applies(state):
        return ()
    return tuple(untraceable_figures(buf.get("text", ""), observation_pool(state, query)))


def required_figures(state: AgentState) -> tuple:
    """The figures this turn COMPUTED that the answer is expected to state — only the LAST
    completed computing step counts (intermediate arithmetic legitimately stays out), and OFF
    whenever the turn has incidents (a disclosure outranks a figure)."""
    from nodes.rectify import COMPUTE_TOOLS

    plan = state.get("plan") or []
    if incidents_block(plan):
        return ()
    computed = [
        s for s in plan
        if s.get("intended_tool") in COMPUTE_TOOLS and s.get("status") == "done"
        and s.get("result") is not None
    ]
    if not computed:
        return ()
    return tuple(lit for _v, lit in figure_literals(computed[-1].get("result")))


def unstated_computed_figures(buf: dict, state: AgentState) -> tuple:
    """Figures this turn COMPUTED that the answer does not state (detection half)."""
    required = required_figures(state)
    if not required:
        return ()
    return tuple(untraceable_figures(" ".join(required), buf.get("text", "")))


def _regenerate(model, llm_input, corrective: str) -> str:
    """ONE corrective regeneration — a plain resample reproduces the same arithmetic, so the retry
    carries the specific complaint. '' when the model errors (the ladder falls back to disclosing)."""
    try:
        resp = generate(model, list(llm_input) + [HumanMessage(content=corrective)],
                        tag=_model_tag("synthesizer"),
                        **_invoke_kwargs("synthesizer", None, 0.7, task="correction"))
        text = getattr(resp, "content", "")
        return text if isinstance(text, str) else str(text)
    except Exception as exc:
        diag.log(f"synthesize_node : corrective regeneration failed ({exc})")
        return ""


def _rewritten(buf: dict, text: str) -> dict:
    """The corrected answer as a fresh all-model buffer (a first-pass buffer carries no human
    spans; the confidence overlay of the discarded draft does not describe the new text)."""
    return {**provenance.append_model(provenance.new_buffer(), text),
            **{k: buf[k] for k in ("state",) if k in buf}}


def _ground_answer(buf: dict, model, llm_input, state: AgentState, query: str):
    """Check, correct once, disclose. Returns (buffer, ungrounded_literals) — an empty tuple means
    the answer is fully traceable. A failed retry keeps the original answer and discloses:
    suppressing the answer would lose the parts that WERE grounded."""
    untraceable = ungrounded_figures(buf, state, query)
    if not untraceable:
        return buf, ()
    text = _regenerate(model, llm_input, GROUNDING_CORRECTIVE.format(bad=", ".join(untraceable)))
    if text.strip():
        still = untraceable_figures(text, observation_pool(state, query))
        return _rewritten(buf, text), tuple(still)
    return buf, tuple(untraceable)


def _state_computed(buf: dict, model, llm_input, state: AgentState):
    """The inverse ladder: the answer must state what the turn computed; correct once, disclose."""
    missing = unstated_computed_figures(buf, state)
    if not missing:
        return buf, ()
    text = _regenerate(model, llm_input, COMPUTED_CORRECTIVE.format(missing=", ".join(missing)))
    if text.strip():
        still = untraceable_figures(" ".join(required_figures(state)), text)
        return _rewritten(buf, text), tuple(still)
    return buf, tuple(missing)


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
            buf, frozen, meta = _stream_continuation(model_name, llm_input, buf, freeze)
            tok_per_sec = continuation.extract_tok_per_sec(meta)
            context_tokens = int(meta.get("prompt_eval_count") or 0)
            response_metadata = {k: meta[k] for k in
                                 ("eval_count", "eval_duration", "prompt_eval_count", "done_reason")
                                 if k in meta}
            usage_metadata = None
        else:
            # response_metadata comes back logprobs-free at the source (the buffer's confidence
            # overlay is the one canonical carrier of the token table — it must never ride the
            # recorded AIMessage into state/autosave/trace).
            buf, frozen, response_metadata, usage_metadata = _stream_first_pass(llm_input, freeze)
            # A tiny attribute shim keeps core/llms' extractors as THE one stats parser.
            stats = SimpleNamespace(
                response_metadata=response_metadata, usage_metadata=usage_metadata
            )
            tok_per_sec = extract_tok_per_sec(stats)
            context_tokens = extract_prompt_tokens(stats)
    finally:
        if freeze is not None:
            freeze.disarm()  # also clears a request that landed after the stream ended

    if frozen:
        # Stop was clean; hand the buffer to the answer_gate edit interrupt
        # (route_after_synthesize). No message lands yet — the turn is mid-answer.
        return {"answer_buffer": {**buf, "state": "frozen", "edited": False}}

    # The groundedness gate + the computed-value check. CORRECTION on a normal first pass only;
    # on a RESUME the human edited the prefix — regenerating would discard their edit, so that
    # pass DETECTS and marks (the notes are trailers on the message, never edits to the answer).
    # Grounding first, then the computed-value check: the grounding corrective may REPLACE the
    # whole answer, so asking "does it state 551" before that would test a draft nobody sees.
    basis = authorization_basis(state)
    if buf_state == "resume":
        ungrounded = ungrounded_figures(buf, state, basis)
        dropped = unstated_computed_figures(buf, state)
    else:
        model = get_model("synthesizer")
        buf, ungrounded = _ground_answer(buf, model, llm_input, state, basis)
        buf, dropped = _state_computed(buf, model, llm_input, state)

    return _final_updates(
        buf, incidents, sources, cancelled,
        tok_per_sec=tok_per_sec, context_tokens=context_tokens,
        response_metadata=response_metadata, usage_metadata=usage_metadata,
        ungrounded=ungrounded, dropped=dropped,
    )


def route_after_synthesize(state: AgentState) -> str:
    """After synthesize: a frozen buffer routes to the answer_gate edit interrupt (the user
    pressed Esc mid-stream); anything else ends the turn."""
    buf = state.get("answer_buffer")
    if isinstance(buf, dict) and buf.get("state") == "frozen":
        return "answer_gate"
    return "end"
