import time

from langchain.tools import tool

from memory_registry import add_memory


@tool
def remember(fact: str, category: str = "general"):
    """Save a durable fact about the user or their preferences to persistent memory so it is
    remembered in future sessions. Use this when the user shares a lasting preference, a fact
    about themselves, or explicitly asks you to remember something (e.g. "I prefer terse
    answers", "my timezone is PST"). `fact` is a single concise statement. `category` is an
    optional label such as preference, identity, or project. Do NOT use this for one-off,
    conversation-specific details."""
    start = time.perf_counter()
    try:
        return add_memory(fact, category)
    finally:
        print(f"remember : {time.perf_counter() - start:.4f}s")
