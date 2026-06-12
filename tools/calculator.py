import ast
import operator

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
            if abs(exp) > _MAX_POW_EXP:
                raise ValueError(f"exponent too large (limit {_MAX_POW_EXP})")
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
