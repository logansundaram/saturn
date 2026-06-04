from langchain.tools import tool

from tool_registry._tavily import get_tavily_client

# web_search finds URLs; web_extract pulls the actual page content out of them.
# Keep ownership of planning, source selection, and synthesis; outsource the scraping.


@tool
def web_extract(urls: list[str], include_images: bool = False):
    """Extract the full page content from one or more web URLs.

    Use this after web_search to read the actual text of a promising result, or
    whenever the user gives you a specific URL to read. Pass one URL or several.

    Args:
        urls: One or more fully-qualified URLs (http/https) to fetch and extract.
        include_images: If True, also return image URLs found on each page.

    Returns the extracted text per URL, plus a note for any URL that failed.
    """
    if isinstance(urls, str):
        urls = [urls]
    if not urls:
        return "No URLs provided to extract."

    response = get_tavily_client().extract(urls=urls, include_images=include_images)

    results = response.get("results", [])
    failed = response.get("failed_results", [])

    if not results and not failed:
        return "No content could be extracted from the provided URLs."

    blocks = []
    for r in results:
        url = r.get("url", "(unknown url)")
        content = r.get("raw_content") or r.get("content") or ""
        block = f"## {url}\n\n{content.strip()}"
        if include_images and r.get("images"):
            block += "\n\nImages:\n" + "\n".join(f"- {img}" for img in r["images"])
        blocks.append(block)

    for f in failed:
        url = f.get("url", "(unknown url)")
        error = f.get("error", "unknown error")
        blocks.append(f"## {url}\n\n[failed to extract: {error}]")

    return "\n\n---\n\n".join(blocks)
