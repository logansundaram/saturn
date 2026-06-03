import time
from langchain.tools import tool


@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result as a string.
    Supports basic arithmetic (+, -, *, /), exponentiation (**), modulo (%),
    and standard math functions (abs, round, min, max, pow, sum).
    Input should be a valid Python math expression as a string, e.g. '2 + 3 * 4' or 'round(3.14159, 2)'."""
    start = time.perf_counter()
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
        # Tame float noise (e.g. 672.3499999999999 -> 672.35) and render whole-number
        # floats as ints (37.0 -> 37) so the agent reports clean values.
        if isinstance(result, float):
            result = round(result, 4)
            if result.is_integer():
                result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"
    finally:
        print(f"calculate : {time.perf_counter() - start:.4f}s")
