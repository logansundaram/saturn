from commands._framework import command, _print
from commands._utils import _ROLES, split_save_flag

_MIN_NUM_CTX = 256  # below this Ollama can't fit the system prompts; reject obvious typos


@command(
    "context",
    "Runtime readout: context window + fill, CPU/RAM/GPU; resize the window.",
    aliases=("ctx",),
    usage="/context [size|auto [--save]]",
    details="""
With no args, the combined runtime readout: the active context window (num_ctx), how full it was
on the last LLM call (a fill bar + token count), the per-role windows, and a point-in-time
CPU/RAM/GPU/VRAM snapshot (the same gauges the status bar streams live during a turn, colored
green→yellow→red by load). (/system was folded in here — one runtime readout, not two.)

With a size, sets the Ollama context window for every local role at once (session only) and
rebuilds the models so it takes effect next turn — num_ctx is fixed when a model is built, so
the cache is dropped. With `auto`, clears the override so each model uses its capability
context_window from config.yaml. Add --save to also persist runtime.num_ctx to config.yaml so
the setting survives a restart.

Note: without an explicit window Ollama silently caps at 2048 tokens; this binds the full
window so the gauge is truthful.

To free the window up rather than resize it, /compact runs the LLM summary over older turns.

Examples:
  /context                show the window + current fill
  /context 16384          resize every local role to 16k tokens (session only)
  /context 16384 --save   resize AND persist to config.yaml
  /context auto           back to per-model capability windows
  /context --save         persist the CURRENT window setting unchanged (like /config persist)
""",
)
def _context(ctx, args):
    from config import get_config
    from core.llms import reset_models, active_context_window, model_id
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
        # Session token budget (runtime.token_budget) — shown only when one is set, with the
        # enforcement consequence spelled out once it has actually been spent.
        from core import budget

        if budget.limit():
            pct = budget.spent() / budget.limit() * 100
            line = (
                f"  token budget: {budget.spent():,} / {budget.limit():,} session tokens"
                f" ({pct:.0f}%)"
            )
            if budget.exceeded():
                line += " — SPENT: turns answer without new tool calls"
            _print(line)
        # The hardware half of the runtime readout (absorbed from the old /system).
        from tui.system_monitor import get_system_metrics

        ui.show_system_metrics(get_system_metrics())
        return

    args, save = split_save_flag(args)
    if not args:
        if save:  # `/context --save` persists the CURRENT window setting (like /config persist)
            from commands.config import _persist_key
            _persist_key(cfg, "runtime.num_ctx")
            return
        _print("  usage: /context <size>|auto [--save]")
        return

    arg = args[0].lower()
    if arg in ("compact", "summarize"):
        _print("  /context compact was removed — use /compact to summarize older turns.")
        return

    if arg in ("auto", "default", "reset", "off"):
        cfg.set("runtime.num_ctx", None)
        reset_models()
        _print("  context window -> auto (each model uses its capability window).")
        _print("  models rebuild on next use.")
        if save:
            from commands.config import _persist_key
            _persist_key(cfg, "runtime.num_ctx")
        else:
            _print("  (session only; add --save to persist to config.yaml.)")
        return

    try:
        n = int(arg)
    except ValueError:
        _print(f"  not a size: {args[0]!r} — usage: /context <size>|auto [--save]")
        return
    if n < _MIN_NUM_CTX:
        _print(f"  num_ctx too small: {n} (minimum {_MIN_NUM_CTX}).")
        return

    cfg.set("runtime.num_ctx", n)
    reset_models()
    _print(f"  context window -> {n:,} tokens for all local roles.")
    _print("  models rebuild on next use.")
    if save:
        from commands.config import _persist_key
        _persist_key(cfg, "runtime.num_ctx")
    else:
        _print("  (session only; add --save to persist to config.yaml.)")
