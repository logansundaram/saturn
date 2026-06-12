"""tools/calculator.py — float-epsilon taming (gotcha #3) and the whitelisted AST
evaluator (a security surface: calculate is read_only, so anything it can execute bypasses
the approval gate entirely)."""

from tools.calculator import calculate


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
    assert _calc("sum([1, 2, 3])") == "6"
    assert _calc("round(3.14159, ndigits=2)") == "3.14"


def test_dunder_traversal_escape_refused():
    """The eval() escape the AST evaluator exists to close: dunder traversal from a literal up
    to object subclasses (and from there to os/subprocess) must never execute."""
    for expr in (
        "().__class__.__bases__[0].__subclasses__()",
        "(1).__class__.__mro__[1].__subclasses__()",
        "abs.__self__",
        "getattr(1, '__class__')",
        "eval('1')",
        "open('x')",
    ):
        assert _calc(expr).startswith("Error"), expr


def test_huge_exponent_bounded():
    """9**9**9**9 must error fast, not hang/OOM the turn."""
    assert _calc("9**9**9**9").startswith("Error")
    assert _calc("2**10**6").startswith("Error")
    # Legitimate large-but-sane powers still work.
    assert _calc("2**64") == "18446744073709551616"


def test_non_numeric_operands_refused():
    """Sequence repetition ([1] * 10**9, 'a' * 10**9) is a memory bomb, not math."""
    assert _calc("[1, 2] * 1000000").startswith("Error")
    assert _calc("'a' * 1000000").startswith("Error")


def test_expression_length_capped():
    assert _calc("1+" * 600 + "1").startswith("Error")
