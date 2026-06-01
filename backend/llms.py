from langchain_ollama import ChatOllama
from registry import tool
from pydantic import BaseModel, Field


llm = ChatOllama(model="gemma4:e4b")


llm_with_tools = llm.bind_tools(tool)


# move the planoutput to somewhere else


class PlanOutput(BaseModel):
    tools_necessary: bool = Field(description="determine if the query needs tools")
    rag_necessary: bool = Field(
        description="determine if the query needs sepcific local docs"
    )
    messages_relevant: bool = Field(
        description="determine if the previous messages are relevant to the query"
    )


llm_with_structued_output = llm.with_structured_output(PlanOutput)
