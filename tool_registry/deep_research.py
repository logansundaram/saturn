import time
from langchain.tools import tool
from dotenv import load_dotenv
import os
from tavily import TavilyClient

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

POLL_INTERVAL = 3  # seconds between status checks


@tool
def deep_research(query: str):
    """Performs a deep research on the given query. Returns a comprehensive research report. This tool is used for in-depth research on a topic. It is a more advanced version of the web search tool. It is used to get a comprehensive understanding of a topic. It is used to get a detailed analysis"""
    start = time.perf_counter()
    tavily_client = TavilyClient(api_key=tavily_api_key)

    job = tavily_client.research(input=query, model="pro")
    request_id = job["request_id"]

    while True:
        status_response = tavily_client.research_get(request_id)
        if status_response["status"] == "completed":
            print(f"deep_research : {time.perf_counter() - start:.4f}s")
            return status_response["response"]
        time.sleep(POLL_INTERVAL)
