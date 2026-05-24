from langchain.tools import tool


@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result as a string.
    Supports basic arithmetic (+, -, *, /), exponentiation (**), modulo (%),
    and standard math functions (abs, round, min, max, pow, sum).
    Input should be a valid Python math expression as a string, e.g. '2 + 3 * 4' or 'round(3.14159, 2)'."""
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"
