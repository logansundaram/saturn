import re

from core import budget
from config import get_config
from core.state import AgentState, unrun_planned_tools
from textutil import clip, split_call_result
from core.llms import (
    get_model,
    extract_tok_per_sec,
    extract_prompt_tokens,
    extract_total_tokens,
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


def _label_clamp(label: str, cap: int = _MAX_SOURCE_LABEL) -> str:
    return clip(label, cap)


def _tool_source_label(result) -> str:
    """Provenance label for one tool_results entry. Entries are `name(args) -> result` strings
    (nodes/tools.py pairs them on purpose); the call repr before the arrow is the label — split
    via textutil.split_call_result, THE one parser of that serialization (the Glass Box splits
    the same strings for its taint corpus; two hand-rolled splits would drift)."""
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


def cancel_orphaned_calls(last) -> list:
    """Cancellation ToolMessages for a trailing AIMessage's unanswered tool_calls (empty when
    there are none). Nothing can have answered a TRAILING message's calls, so every call gets
    one. Pure helper so the orphan guard is testable without an LLM."""
    if not isinstance(last, AIMessage):
        return []
    return [
        ToolMessage(
            content=(
                "Not executed — the turn ended (iteration or token-budget limit) before this "
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

    # The agent's draft answer — its final no-tool-call message. This is the exact text the
    # replan judge verified for groundedness, so the shipped answer must BUILD ON it rather than
    # be re-derived blind: regenerating from scratch paid a second full generation for nothing
    # and could introduce new claims the judge never saw. Absent on the paths that never produced
    # a draft (abort at the plan gate, iteration cap hit mid-tool-round) — those synthesize from
    # the gathered material alone, exactly as before.
    msgs = state.get("messages", [])
    last = msgs[-1] if msgs else None
    draft = ""
    if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
        draft = str(last.content).strip()

    # Forced landing mid-decision: the iteration cap or token budget routed here while the
    # trailing AIMessage still carries unanswered tool_calls (route_after_agent checks those
    # bounds before has_tool_calls, deliberately — the bound must stop NEW tool rounds). Close
    # each orphaned call with a cancellation ToolMessage now, or the carried conversation (and
    # its autosave) holds an assistant tool_use with no tool_result — a hard 400 on the next
    # cloud-provider turn and a /resume that reproduces it.
    cancelled = cancel_orphaned_calls(last)
    if draft:
        llm_input.append(
            HumanMessage(
                content=(
                    "Draft answer from the reasoning loop (already checked against the gathered "
                    "results — build on it, do not contradict it):\n" + draft
                )
            )
        )

    # If we arrive here with a planned gathering step still un-run (the agent gave up and the
    # nudge budget was exhausted), be honest about the gap instead of asserting the information
    # doesn't exist — the failure mode this whole guard exists to avoid.
    incomplete = unrun_planned_tools(state.get("plan", []), state.get("tools_called", []))
    if incomplete:
        labels = "; ".join(f"{s.get('label')} (needed `{s.get('intended_tool')}`)" for s in incomplete)
        llm_input.append(
            HumanMessage(
                content=(
                    "NOTE: the plan included information-gathering step(s) that were not "
                    f"completed this turn: {labels}. If the gathered results above are not "
                    "sufficient to answer, say plainly that you were unable to complete that "
                    "lookup — do NOT state that the information does not exist or that nothing "
                    "is available, since the lookup was not actually carried out."
                )
            )
        )

    # Dry-run: nothing was actually executed (tool_node stubbed every call). Tell the synthesizer
    # to report the intended actions as a preview, not to answer as though they had run.
    if bool(get_config().get("runtime.dry_run", False)):
        llm_input.append(
            HumanMessage(
                content=(
                    "DRY-RUN MODE: No tool was actually executed this turn — every tool result above "
                    "is a `[DRY RUN] would execute …` placeholder. Do NOT answer as if the actions "
                    "were performed or report their results as real. Instead, summarize for the user "
                    "exactly what you WOULD do to answer their request: the plan and each tool call "
                    "you intended to make (with its arguments), in order. Make clear nothing was run."
                )
            )
        )

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

    budget.add(extract_total_tokens(aggregated))

    # Normalize the aggregated chunk to a plain AIMessage so every downstream type matches the old
    # invoke() path exactly (add_messages, _compact_history's isinstance checks, autosave).
    content = aggregated.content if isinstance(aggregated.content, str) else str(aggregated.content)

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
