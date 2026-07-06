"""
Deterministic local-compute tools — facts the model must never make up from memory.

  calculate    — arithmetic via a whitelisted AST evaluator (never eval()).
  current_time — time grounding from the machine's own clock.

Both are read_only and pure-local: the answer is computed, not recalled, and nothing leaves
the machine. (current_time lived in tools/clock.py until the 2026-06-11 leaf consolidation.)
"""

import ast
import operator
from datetime import datetime, timezone

from tools.toolspec import register_tool

# Whitelisted AST evaluator — NOT eval(). eval with an empty __builtins__ dict is an escapable
# sandbox (dunder traversal reaches os/subprocess), which would make this read_only tool an
# approval-gate bypass. Here only the node types below ever execute; everything else is refused.
_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sum": sum,
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Resource bounds: a math helper must never hang or OOM the turn (9**9**9**9).
_MAX_EXPR_LEN = 1000
_MAX_POW_EXP = 10_000
# Result-SIZE bound for integer exponentiation: the exponent cap alone lets a nested
# (9**9999)**9999 pass both checks — each exponent ≤ 10k — while computing a ~3×10^8-bit
# integer (minutes of CPU / OOM) on this ungated read_only tool. bit_length(base)×exp bounds
# the result cheaply before any work happens; 1M bits keeps every sane calculation legal.
_MAX_POW_BITS = 1_000_000


def _check_pow(base, exp) -> None:
    """Both halves of the pow resource bound (shared by the ** operator and the pow() builtin)."""
    if abs(exp) > _MAX_POW_EXP:
        raise ValueError(f"exponent too large (limit {_MAX_POW_EXP})")
    if isinstance(base, int) and isinstance(exp, int) and exp > 1:
        if base.bit_length() * exp > _MAX_POW_BITS:
            raise ValueError(
                f"result too large (base of {base.bit_length()} bits raised to {exp} "
                f"exceeds the {_MAX_POW_BITS}-bit result limit)"
            )


def _number(value):
    """Operands must be plain numbers — refuses e.g. list * int memory bombs."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric operand: {value!r}")
    return value


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return _number(node.value)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            base = _number(_eval_node(node.left))
            exp = _number(_eval_node(node.right))
            _check_pow(base, exp)
            return base ** exp
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return fn(_number(_eval_node(node.left)), _number(_eval_node(node.right)))
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return fn(_number(_eval_node(node.operand)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("only these functions are supported: " + ", ".join(sorted(_ALLOWED_FUNCS)))
        args = [_eval_node(a) for a in node.args]
        kwargs = {kw.arg: _eval_node(kw.value) for kw in node.keywords if kw.arg}
        if node.func.id == "pow":
            # The pow() builtin is the ** operator with a function-call spelling — it must share
            # the _check_pow bounds, or pow(9, 99999999) / pow(9**9999, 9999) recreate the exact
            # resource bombs the ast.Pow branch above refuses. The base/exponent may arrive
            # positionally OR as keywords (pow takes base/exp/mod), so resolve both forms. The
            # 3-arg modular form stays unbounded on purpose: modular exponentiation is cheap at
            # any exponent, and it is the one reason pow() exists here at all (2-arg use is
            # already covered by **).
            base = args[0] if args else kwargs.get("base")
            exp = args[1] if len(args) > 1 else kwargs.get("exp")
            has_mod = len(args) > 2 or "mod" in kwargs
            if exp is not None and not has_mod:
                _check_pow(_number(base) if base is not None else 0, _number(exp))
        return _ALLOWED_FUNCS[node.func.id](*args, **kwargs)
    if isinstance(node, (ast.Tuple, ast.List)):
        # Argument sequences for sum/min/max, e.g. sum([1, 2, 3]).
        return [_number(_eval_node(e)) for e in node.elts]
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def _safe_eval(expression: str):
    if len(expression) > _MAX_EXPR_LEN:
        raise ValueError(f"expression too long (limit {_MAX_EXPR_LEN} characters)")
    return _eval_node(ast.parse(expression, mode="eval").body)


@register_tool("read_only")
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result as a string.
    Supports basic arithmetic (+, -, *, /), exponentiation (**), modulo (%),
    and standard math functions (abs, round, min, max, pow, sum).
    Input should be a valid Python math expression as a string, e.g. '2 + 3 * 4' or 'round(3.14159, 2)'."""
    try:
        result = _safe_eval(expression)
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


# --- time grounding ---------------------------------------------------------------------------
# `calculate` exists so the model never does arithmetic from memory; `current_time` is the same
# idea applied to time. Local models confabulate dates constantly ("today" resolved against a
# training cutoff), and without this tool the only cure was a pointless web_search.
@register_tool("read_only")
def current_time():
    """The current local date and time, with timezone, UTC equivalent, and weekday. Use this
    whenever the answer depends on 'today', 'now', or any relative date — never guess the
    current date from memory."""
    now = datetime.now().astimezone()
    return {
        "local": now.isoformat(timespec="seconds"),
        "utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "timezone": str(now.tzinfo),
        "weekday": now.strftime("%A"),
    }
