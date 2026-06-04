from state import AgentState
from llms import get_model, extract_tok_per_sec
from messages import synthesize_sys_msg
from langchain.messages import HumanMessage, ToolMessage


def synthesize_node(state: AgentState):
    query = state["current_query"]
    context = state["context"]
    tool_results = state.get("tool_results", [])

    # Iteration-cap guard: route_after_agent sends us here even when the agent's last message
    # still carries unanswered tool_calls (the loop hit max_iterations). Those calls need
    # ToolMessages or the message history is invalid for the next turn's model call (an
    # AIMessage with tool_calls and no matching ToolMessage is rejected by most providers).
    # Emit a neutral "not executed" reply for each so history stays well-formed.
    pending_calls = getattr(state["messages"][-1], "tool_calls", None) or []
    orphan_replies = [
        ToolMessage(
            content="Not executed: the agent reached its iteration limit before running this "
            "call. Answer with the information already gathered.",
            tool_call_id=tc["id"],
            name=tc["name"],
        )
        for tc in pending_calls
    ]

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

    llm_input.append(HumanMessage(content=f"Current user query:\n{query}"))

    llm_response = get_model("synthesizer").invoke(llm_input)
    return {
        "messages": orphan_replies + [llm_response],
        "tok_per_sec": extract_tok_per_sec(llm_response),
    }
