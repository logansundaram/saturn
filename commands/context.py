from commands._framework import command, _print
from commands._utils import _ROLES

_MIN_NUM_CTX = 256  # below this Ollama can't fit the system prompts; reject obvious typos


@command(
    "context",
    "Show the model context window and how full it is; resize it.",
    aliases=("ctx",),
    usage="/context [size|auto]",
    details="""
With no args, shows the active context window (num_ctx), how full it was on the last LLM call
(a fill bar + token count), and the per-role windows. The same fill gauge rides the bottom
status bar live during a turn, colored green→yellow→red as it fills.

With a size, sets the Ollama context window for every local role at once (session only) and
rebuilds the models so it takes effect next turn — num_ctx is fixed when a model is built, so
the cache is dropped. With `auto`, clears the override so each model uses its capability
context_window from config.yaml.

Note: without an explicit window Ollama silently caps at 2048 tokens; this binds the full
window so the gauge is truthful. Edit runtime.num_ctx in config.yaml to persist.

Examples:
  /context              show the window + current fill
  /context 16384        resize every local role to 16k tokens
  /context auto         back to per-model capability windows
""",
)
def _context(ctx, args):
    from config import get_config
    from llms import reset_models, active_context_window, model_id
    from tui import ui

    cfg = get_config()

    if not args:
        window = active_context_window()
        used = int(ctx.state.get("context_tokens", 0) or 0)
        if cfg.num_ctx_override:
            source = "override · runtime.num_ctx"
        else:
            source = f"auto · {model_id('tool_caller')} capability"
        per_role = {role: cfg.num_ctx_for(model_id(role)) for role in _ROLES}
        ui.show_context(window, used, source, per_role)
        return

    arg = args[0].lower()
    if arg in ("auto", "default", "reset", "off"):
        cfg.set("runtime.num_ctx", None)
        reset_models()
        _print("  context window -> auto (each model uses its capability window; session only).")
        _print("  models rebuild on next use; edit runtime.num_ctx in config.yaml to persist.")
        return

    try:
        n = int(arg)
    except ValueError:
        _print(f"  not a size: {args[0]!r} — usage: /context <size>|auto")
        return
    if n < _MIN_NUM_CTX:
        _print(f"  num_ctx too small: {n} (minimum {_MIN_NUM_CTX}).")
        return

    cfg.set("runtime.num_ctx", n)
    reset_models()
    _print(f"  context window -> {n:,} tokens for all local roles (session only).")
    _print("  models rebuild on next use; edit runtime.num_ctx in config.yaml to persist.")
