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


def test_nested_pow_result_size_bounded():
    """Bounding the exponent alone was bypassable by NESTING: (9**9999)**9999 passes both
    per-exponent checks while computing a ~3×10^8-bit integer (minutes of CPU / OOM on an
    ungated read_only tool). The result-size bound (bit_length(base)×exp) refuses it fast."""
    assert _calc("(9**9999)**9999").startswith("Error")
    assert _calc("((9**999)**999)**999").startswith("Error")
    assert _calc("pow(9**9999, 9999)").startswith("Error")
    # Big-but-sane powers stay legal (final results are separately capped at str() time by
    # Python's own 4300-digit int→str limit — a graceful error, not a hang).
    assert not _calc("9**4000").startswith("Error")
    # Huge INTERMEDIATES under the bit bound stay legal too — only the final value is str'ed.
    assert _calc("9**9999 % 97") == str(9**9999 % 97)


def test_pow_builtin_exponent_bounded():
    """pow() is the ** operator with a function-call spelling — it must share the _MAX_POW_EXP
    bound (pow(9, 99999999) was a guard bypass: tiny expression, ~95-million-digit result).
    The exponent can arrive positionally or as a keyword, so every spelling is bounded."""
    assert _calc("pow(9, 20000)").startswith("Error")
    assert _calc("pow(9, exp=20000)").startswith("Error")
    assert _calc("pow(base=9, exp=20000)").startswith("Error")
    # Legitimate small exponents still work, in both spellings.
    assert _calc("pow(2, 10)") == "1024"
    assert _calc("pow(2, exp=10)") == "1024"
    # The 3-arg modular form is cheap at any exponent (modular exponentiation) — stays unbounded.
    expected = str(pow(9, 99999999, 1000))
    assert _calc("pow(9, 99999999, 1000)") == expected
    assert _calc("pow(9, 99999999, mod=1000)") == expected


def test_non_numeric_operands_refused():
    """Sequence repetition ([1] * 10**9, 'a' * 10**9) is a memory bomb, not math."""
    assert _calc("[1, 2] * 1000000").startswith("Error")
    assert _calc("'a' * 1000000").startswith("Error")


def test_expression_length_capped():
    assert _calc("1+" * 600 + "1").startswith("Error")
