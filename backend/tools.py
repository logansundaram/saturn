# file to add the tools for the agents to use
from langchain.tools import tool


@tool
def addition(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def subtraction(a: int, b: int) -> int:
    """Subtracts b from a."""
    return a - b


@tool
def multiplication(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


@tool
def division(a: int, b: int) -> int:
    """Divides a by b."""
    return a / b


tool = [addition, subtraction, multiplication, division]

tools_by_name = {t.name: t for t in tool}
