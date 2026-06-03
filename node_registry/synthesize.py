import time
from state import AgentState
from llms import llm
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

    llm_input.append(HumanMessage(content=f"Current user query:\n{query}"))

    llm_response = llm.invoke(llm_input)
    print(f"synthesize_node : {time.perf_counter() - start:.4f}s")
    return {
        "current_response": llm_response,
        "messages": [llm_response],
    }
