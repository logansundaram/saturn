# deep research tool
from langchain.tools import tool
from dotenv import load_dotenv
import os
from tavily import TavilyClient

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
print(tavily_api_key)

# web tool suite

# docstring of the tool is automatically exposed to the llm, that is how it determines what tool to call


@tool
def deep_research(query: str):
    """Performs a deep research on the given query. Returns a comprehensive research report. This tool is used for in-depth research on a topic. It is a more advanced version of the web search tool. It is used to get a comprehensive understanding of a topic. It is used to get a detailed analysis"""
    tavily_client = TavilyClient(api_key=tavily_api_key)
    response = tavily_client.research(query)
    return response
