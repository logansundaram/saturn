import time
from state import AgentState, unrun_planned_tools
from llms import get_model, extract_tok_per_sec, extract_prompt_tokens
from messages import synthesize_sys_msg
from langchain.messages import HumanMessage, AIMessage


def synthesize_node(state: AgentState):
    start = time.perf_counter()
    query = state["current_query"]
    context = state["context"]
    tool_results = state.get("tool_results", [])

    llm_input = [synthesize_sys_msg]

    if context:
        llm_input.append(HumanMessage(content=f"Relevant context:\n{context}"))

    if tool_results:
        llm_input.append(
            HumanMessage(
                content="Tool results:\n" + "\n\n".join(map(str, tool_results))
            )
        )

    if documents_retrieved := state.get("documents_retrieved", []):
        # tool_node stores retrieval results as pre-formatted strings (source + text),
        # not Document objects.
        llm_input.append(
            HumanMessage(
                content="Retrieved documents:\n"
                + "\n\n".join(str(doc) for doc in documents_retrieved)
            )
        )

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

    # Normalize the aggregated chunk to a plain AIMessage so every downstream type matches the old
    # invoke() path exactly (add_messages, _compact_history's isinstance checks, autosave).
    content = aggregated.content if isinstance(aggregated.content, str) else str(aggregated.content)
    msg_kwargs = {"response_metadata": getattr(aggregated, "response_metadata", {}) or {}}
    if getattr(aggregated, "usage_metadata", None):
        msg_kwargs["usage_metadata"] = aggregated.usage_metadata
    llm_response = AIMessage(content=content, **msg_kwargs)

    return {
        "messages": [llm_response],
        "tok_per_sec": extract_tok_per_sec(aggregated),
        "context_tokens": extract_prompt_tokens(aggregated),
    }
