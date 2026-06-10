import re
import time

import budget
from config import get_config
from state import AgentState, unrun_planned_tools
from llms import (
    get_model,
    extract_tok_per_sec,
    extract_prompt_tokens,
    extract_total_tokens,
)
from messages import synthesize_sys_msg
from langchain.messages import HumanMessage, AIMessage


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
# (tool_registry/knowledge.py) — the provenance labels for retrieval observations.
_DOC_SOURCE_RE = re.compile(r"\[source: ([^\]]+)\]")


def _label_clamp(label: str, cap: int = _MAX_SOURCE_LABEL) -> str:
    label = " ".join(str(label).split())
    return label if len(label) <= cap else label[: cap - 1] + "…"


def _tool_source_label(result) -> str:
    """Provenance label for one tool_results entry. Entries are `name(args) -> result` strings
    (node_registry/tools.py pairs them on purpose); the call repr before the arrow is the label."""
    return _label_clamp(str(result).split(" -> ", 1)[0])


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


def synthesize_node(state: AgentState):
    start = time.perf_counter()
    query = state["current_query"]
    context = state["context"]
    tool_results = state.get("tool_results", [])
    documents_retrieved = state.get("documents_retrieved", [])

    citations = bool(get_config().get("runtime.citations", True))
    sources: list[tuple[int, str]] = []
    if citations:
        numbered_tools, numbered_docs, sources = build_sources(
            tool_results, documents_retrieved
        )

    llm_input = [synthesize_sys_msg]

    if context:
        llm_input.append(HumanMessage(content=f"Relevant context:\n{context}"))

    if tool_results:
        if citations:
            body = "\n\n".join(numbered_tools)
            header = "Tool results (numbered — cite the matching [n] after claims drawn from them):\n"
        else:
            body = "\n\n".join(map(str, tool_results))
            header = "Tool results:\n"
        llm_input.append(HumanMessage(content=header + body))

    if documents_retrieved:
        # tool_node stores retrieval results as pre-formatted strings (source + text),
        # not Document objects.
        if citations:
            body = "\n\n".join(numbered_docs)
            header = (
                "Retrieved documents (numbered — cite the matching [n] after claims drawn "
                "from them):\n"
            )
        else:
            body = "\n\n".join(str(doc) for doc in documents_retrieved)
            header = "Retrieved documents:\n"
        llm_input.append(HumanMessage(content=header + body))

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
        "messages": [llm_response],
        "tok_per_sec": extract_tok_per_sec(aggregated),
        "context_tokens": extract_prompt_tokens(aggregated),
    }
