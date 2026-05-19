# file to add the system messages for the different nodes
from langchain.messages import SystemMessage

light_llm_msg = SystemMessage(
    content="Answer the users requests using the available tools, if necessary. If you don't know the answer, say so."
)

call_tool_msg = SystemMessage(
    content="Call the relevante tools based on the user request"
)

fetch_docs_msg = SystemMessage(
    content="Fetch the relevant documents based on the user request"
)

synthesize_output_msg = SystemMessage(
    content="Synthesize the output based on the user request"
)
