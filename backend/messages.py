# file to add the system messages for the different nodes
from langchain.messages import SystemMessage

call_tool_msg = SystemMessage(
    content="Call the relevante tools based on the user request"
)

fetch_docs_msg = SystemMessage(
    content="Fetch the relevant documents based on the user request"
)

synthesize_output_msg = SystemMessage(
    content="Synthesize the output based on the user request"
)
