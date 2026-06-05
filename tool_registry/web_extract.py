import time
from langchain.tools import tool
from dotenv import load_dotenv
import os
from tavily import TavilyClient

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")


@tool
def web_extract(url: str):
    """Extract web page content from one or more specified URLs using Tavily Extract."""
    start = time.perf_counter()
    tavily_client = TavilyClient(api_key=tavily_api_key)
    response = tavily_client.extract(url)
    print(f"web_extract : {time.perf_counter() - start:.4f}s")
    return response
