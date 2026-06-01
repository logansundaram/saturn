import time
from state import AgentState
from llms import llm_with_structued_output


# need to alter this node to access to relevant context
def plan_node(state: AgentState):
    # plan node does not have access to tools avaible nor document metadata in the rag previous messages
    # should have context -> plan
    start = time.perf_counter()
    llm_response = llm_with_structued_output.invoke(str("context: " + state["context"]))
    print(f"plan_node : {time.perf_counter() - start:.4f}s")
    # need to add the plan to the state
    # plan = PlanStep.model_validate_json(llm_response.content)
    # plan = plan.model_dump()
    # print(plan)
    return {
        "tools_necessary": llm_response.tools_necessary,
        "rag_necessary": llm_response.rag_necessary,
        "messages_relevant": llm_response.messages_relevant,
    }
