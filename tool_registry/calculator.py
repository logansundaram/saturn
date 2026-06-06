import time
import diag
from toolspec import register_tool


@register_tool("read_only")
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
        # Tame binary-float epsilon (672.3499999999999 -> 672.35) WITHOUT capping real precision:
        # rounding to 12 significant figures removes the representation artifact while preserving
        # legitimate decimals (1/3 -> 0.333333333333, not 0.3333). Only apply it when the value is
        # small enough that 12 sig figs still spans the whole integer part (abs < 1e12); above that
        # the same formatting would truncate real integer digits and return a WRONG value
        # (2.0**47 -> 140737488355000 instead of ...328), so leave large magnitudes untouched.
        # Whole-number floats render as ints either way (37.0 -> 37).
        if isinstance(result, float):
            if abs(result) < 1e12:
                result = float(f"{result:.12g}")
            if result.is_integer():
                result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"
    finally:
        diag.log(f"calculate : {time.perf_counter() - start:.4f}s")
