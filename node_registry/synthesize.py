import time
from state import AgentState, unrun_planned_tools
from llms import get_model, extract_tok_per_sec, extract_prompt_tokens
from messages import synthesize_sys_msg
from langchain.messages import HumanMessage


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

    llm_response = get_model("synthesizer").invoke(llm_input)
    return {
        "current_response": llm_response,
        "messages": [llm_response],
        "tok_per_sec": extract_tok_per_sec(llm_response),
        "context_tokens": extract_prompt_tokens(llm_response),
    }
