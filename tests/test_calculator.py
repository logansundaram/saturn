"""tool_registry/calculator.py — float-epsilon taming (gotcha #3) and the sandboxed eval."""

from tool_registry.calculator import calculate


def _calc(expr: str) -> str:
    return calculate.invoke({"expression": expr})


def test_basic_arithmetic():
    assert _calc("2 + 3 * 4") == "14"


def test_epsilon_tamed():
    # 0.1 + 0.2 is the canonical binary-float artifact (0.30000000000000004).
    assert _calc("0.1 + 0.2") == "0.3"


def test_real_precision_kept():
    assert _calc("1/3") == "0.333333333333"


def test_whole_floats_render_as_ints():
    assert _calc("37.0 * 2") == "74"


def test_large_magnitude_not_truncated():
    # Above 1e12 the 12-sig-fig rounding would WRONGLY truncate integer digits — it must not run.
    assert _calc("2.0**47") == "140737488355328"


def test_division_by_zero():
    assert _calc("1/0") == "Error: division by zero"


def test_builtins_unreachable():
    out = _calc("__import__('os').getcwd()")
    assert out.startswith("Error")


def test_allowed_functions():
    assert _calc("round(3.14159, 2)") == "3.14"
    assert _calc("max(1, 5, 3)") == "5"
