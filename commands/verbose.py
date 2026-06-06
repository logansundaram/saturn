from commands._framework import command, _print


@command(
    "verbose",
    "Control the live trace: off / on (normal) / full (verbose).",
    usage="/verbose off|on|full",
    details="""
Controls the live execution trace — the dim node/plan rail streamed during a turn. Three levels:

  off     only the final response prints (turns run quietly)
  on      normal: plan · agent · tools · synthesize, with the plumbing nodes (ground, update_plan)
          folded out and metrics dimmed — the default
  full    verbose: every node line, including the folded plumbing, plus full-precision timings

The trace is always written to the trace DB regardless (see /trace and /calls), so this only
affects what scrolls live, not what's recorded. With no argument, flips the trace on/off and
leaves the detail level untouched.

Examples:
  /verbose off
  /verbose on
  /verbose full
  /verbose        toggle on/off
""",
)
def _verbose(ctx, args):
    from tui import ui

    arg = args[0].lower() if args else ""
    if not arg:
        ctx.show_ui = not ctx.show_ui
    elif arg in ("off", "quiet", "compact", "false", "no", "0"):
        ctx.show_ui = False
    elif arg in ("on", "normal", "true", "yes", "1"):
        ctx.show_ui = True
        ui.set_verbosity("normal")
    elif arg in ("full", "verbose", "detailed", "all", "debug"):
        ctx.show_ui = True
        ui.set_verbosity("verbose")
    else:
        _print(
            f"  usage: /verbose off|on|full   (trace {'on' if ctx.show_ui else 'off'}, "
            f"detail {ui.verbosity()})"
        )
        return

    if not ctx.show_ui:
        _print("  live trace off — only the final response prints.")
    else:
        level = ui.verbosity()
        detail = (
            "every node + full timings" if level == "verbose"
            else "plan · agent · tools · synthesize (plumbing folded)"
        )
        _print(f"  live trace on — {level}: {detail}.")
