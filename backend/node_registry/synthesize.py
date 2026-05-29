import time
from state import AgentState
from llms import llm
from messages import synthesize_system_msg
from langchain.messages import SystemMessage, HumanMessage


def synthesize_node(
    state: AgentState,
    system_prompt: SystemMessage = synthesize_system_msg,
):
    start = time.perf_counter()
    query = state["current_query"]
    context = state.get("context", [])
    tool_results = state.get("tool_results", [])

    llm_input = [system_prompt]

    if context:
        llm_input.append(
            HumanMessage(content="Relevant context:\n" + "\n\n".join(context))
        )

    if tool_results:
        llm_input.append(
            HumanMessage(
                content="Tool results:\n" + "\n\n".join(map(str, tool_results))
            )
        )

    llm_input.append(HumanMessage(content=f"Current user query:\n{query}"))

    llm_response = llm.invoke(llm_input)
    print(f"synthesize_node : {time.perf_counter() - start:.4f}s")
    return {
        "current_response": llm_response,
        "messages": [llm_response],
    }
