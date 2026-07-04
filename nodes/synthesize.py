import re

from config import get_config
from core.state import AgentState, incident_steps, unfinished_steps
from textutil import clip, split_call_result
from core.llms import (
    get_model,
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
# The `[source: name, page N]` markers search_knowledge_base prepends to each retrieved chunk
# (tools/knowledge.py) — the provenance labels for retrieval observations.
_DOC_SOURCE_RE = re.compile(r"\[source: ([^\]]+)\]")

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
    (one search_knowledge_base call returns several chunks, usually from a handful of files)."""
    names: list[str] = []
    for m in _DOC_SOURCE_RE.finditer(str(observation)):
        name = m.group(1).strip()
        if name and name not in names:
            names.append(name)
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
        if ev.get("name") not in ("write_file", "edit_file"):
            continue
        args = ev.get("args") or {}
        path = args.get("file_path")
        if not path or path in seen:
            continue
        # The tools report refusals as ordinary strings (ok=True), so check the result text for
        # a success marker before quoting the file as "written".
        preview = str(ev.get("result") or "")
        if not (preview.startswith("File ") or preview.startswith("Content appended")
                or preview.startswith("Edited ")):
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

    # Stream the answer so the UI can render it token-by-token: LangGraph surfaces these chunks via
    # stream_mode="messages" (filtered to this node in agent.run_turn -> on_token). We still aggregate
    # the whole message here and return it, so state["messages"] / the trace / autosave see a complete
    # AIMessage exactly as the old .invoke() path did — the streaming is purely additive. Ollama
    # attaches eval/usage metadata to the final aggregated chunk, so the tok/s + context gauges keep
    # working off it unchanged.
    model = get_model("synthesizer")
    aggregated = None
    for chunk in model.stream(llm_input):
        aggregated = chunk if aggregated is None else aggregated + chunk
    if aggregated is None:  # a model that streamed nothing — fall back to a blocking call
        aggregated = model.invoke(llm_input)

    # Normalize the aggregated chunk to a plain AIMessage so every downstream type matches the old
    # invoke() path exactly (add_messages, _compact_history's isinstance checks, autosave).
    content = aggregated.content if isinstance(aggregated.content, str) else str(aggregated.content)

    # A mechanical incidents note under the answer, mirroring the prompt-level disclosure: the
    # user sees what could not be completed even when the model soft-pedals it.
    if incidents and content.strip():
        content = content.rstrip() + "\n\nNote — the following could not be completed:\n" + "\n".join(
            f"- {i}" for i in incidents
        )

    # Append the provenance footer to the RECORDED answer (state/trace/autosave/headless all carry
    # it). The live token stream has already rendered without it, so the loop re-renders the final
    # message on finish (ui.ResponseStream.finish(final_text) — see agent.main) and the footer
    # appears there. Skipped when nothing was gathered — a pure-knowledge answer has no sources.
    footer = sources_footer(sources)
    if footer and content.strip():
        content = content.rstrip() + "\n\n" + footer

    msg_kwargs = {"response_metadata": getattr(aggregated, "response_metadata", {}) or {}}
    if getattr(aggregated, "usage_metadata", None):
        msg_kwargs["usage_metadata"] = aggregated.usage_metadata
    llm_response = AIMessage(content=content, **msg_kwargs)

    return {
        "messages": [*cancelled, llm_response],
        "tok_per_sec": extract_tok_per_sec(aggregated),
        "context_tokens": extract_prompt_tokens(aggregated),
    }
