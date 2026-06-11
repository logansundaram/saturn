from commands._framework import command, _print
from commands._utils import _parse_toggle


@command(
    "plan",
    "Show the plan; control review/pause/lockstep; save + run plan recipes.",
    usage="/plan | /plan review [on|off] | /plan pause | /plan lockstep [on|off] | "
          "/plan save <name> | /plan recipes [rm <name>] | /plan run <name> [query]",
    details="""
The plan is the agent's living checklist. With no args, renders the most recent one — every step
with its status glyph and intended tool (empty until you've run at least one turn).

Status glyphs:  · pending   ▸ active   ✓ done   ⨯ skipped

This command also controls the human-in-the-loop plan-review architecture:

  /plan review [on|off]   Persistent review mode. When on, EVERY turn pauses at the first step
                          boundary so you can inspect and edit the plan before any tool runs. With
                          no on/off, toggles. Off by default.

  /plan pause             Arm a ONE-SHOT pause: the next turn pauses at its first step boundary for
                          review, then runs normally afterwards. While a turn is running, Esc on an
                          empty line pauses for review at the next step; type a correction first and
                          THEN press Esc to STEER the running turn — the text is injected at the next
                          step boundary so the agent adjusts course without losing the turn. (Plain
                          typing + Enter still queues a follow-up to run after the turn finishes.)

  /plan lockstep [on|off] Lockstep execution. When on (the default), the agent works the plan one
                          step at a time, strongly directed to the current step, so the plan is
                          followed closely. When off, it free-runs with only a soft next-step
                          pointer. Sets runtime.lockstep (session only; edit config.yaml to persist).

When a turn pauses, you get an interactive editor. Its verbs (also usable live):
  add <label> [::tool] · edit <id> <label> · tool <id> <name|none>
  status <id> <status> · move <id> <pos>   · drop <id>
  go / <enter> to run the edited plan, abort to stop the turn.

Plan recipes — a vetted plan captured as a reusable template (database/recipes/):

  /plan save <name>          save the LAST turn's plan (steps + query) as a recipe
  /plan recipes              list saved recipes;  /plan recipes rm <name>  deletes one
  /plan run <name> [query]   re-run a recipe: the next turn's planner uses the saved steps
                             (fresh approval gates, normal lockstep + trace); the saved query
                             runs unless you give a new one after the name

Examples:
  /plan                    show the current plan
  /plan review on          vet every plan before it runs
  /plan pause              review just the next turn's plan
  /plan lockstep off       let the agent free-run the plan
  /plan save weekly-brief  capture the plan that just ran
  /plan run weekly-brief   run it again, same steps, fresh gates
""",
)
def _plan(ctx, args):
    from tui import ui

    if not args:
        _print("  current plan:")
        ui.render_plan(ctx.state.get("plan", []))
        mode = "on" if ctx.review_plan else "off"
        from config import get_config
        lock = "on" if get_config().lockstep else "off"
        _print(f"  review mode: {mode}  ·  lockstep: {lock}   (see /plan --help)")
        return

    sub = args[0].lower()

    if sub == "review":
        new = _parse_toggle(args[1:], ctx.review_plan)
        if new is None:
            _print(f"  usage: /plan review on|off   (currently {'on' if ctx.review_plan else 'off'})")
            return
        ctx.review_plan = new
        if new:
            _print("  plan review ON — every turn pauses at the first step so you can edit the plan.")
        else:
            _print("  plan review off — turns run without the pre-execution review pause.")
        return

    if sub == "pause":
        from interrupts import get_pause_controller
        get_pause_controller().request("user", "one-shot: review the plan before it runs")
        _print("  armed — the next turn will pause at its first step boundary for plan review.")
        _print("  (during a running turn: Esc on an empty line pauses for review at the next step;")
        _print("   type a correction first, then Esc, to steer the running turn instead.)")
        return

    if sub == "lockstep":
        from config import get_config
        cfg = get_config()
        new = _parse_toggle(args[1:], cfg.lockstep)
        if new is None:
            _print(f"  usage: /plan lockstep on|off   (currently {'on' if cfg.lockstep else 'off'})")
            return
        cfg.set("runtime.lockstep", new)
        _print(
            f"  lockstep {'on' if new else 'off'} — "
            + ("the agent follows the plan one step at a time." if new
               else "the agent free-runs the plan with a soft pointer.")
            + " (session only; edit config.yaml to persist.)"
        )
        return

    if sub == "save":
        from stores.recipes import save_recipe
        from textutil import clip

        if len(args) < 2:
            _print("  usage: /plan save <name>   (saves the last turn's plan as a recipe)")
            return
        plan = ctx.state.get("plan", [])
        if not plan:
            _print("  no plan to save yet — run a turn first, then /plan save <name>.")
            return
        try:
            path = save_recipe(args[1], ctx.state.get("current_query", ""), plan)
        except ValueError as exc:
            _print(f"  could not save: {exc}")
            return
        _print(f"  recipe saved -> {path.name}  ({len(plan)} step(s); "
               f"query: \"{clip(ctx.state.get('current_query', ''), 50)}\")")
        _print(f"  run it again any time: /plan run {path.stem}")
        return

    if sub == "recipes":
        from stores.recipes import list_recipes, delete_recipe
        from textutil import clip

        if len(args) >= 3 and args[1].lower() in ("rm", "remove", "delete", "drop"):
            name = args[2]
            if delete_recipe(name):
                _print(f"  recipe removed: {name}")
            else:
                _print(f"  no recipe named {name!r} — /plan recipes lists them.")
            return
        recipes = list_recipes()
        if not recipes:
            _print("  (no recipes saved — after a turn whose plan you liked: /plan save <name>)")
            return
        _print("  saved plan recipes  (/plan run <name> to re-run one):")
        for r in recipes:
            steps = r.get("steps") or []
            when = (r.get("saved_at") or "")[:10]
            _print(f"    {r.get('name'):<20} {len(steps)} step(s)  {when}  "
                   f"\"{clip(r.get('query', ''), 44)}\"")
        return

    if sub == "run":
        from node_registry.plan import seed_next_plan
        from stores.recipes import load_recipe

        if len(args) < 2:
            _print("  usage: /plan run <name> [query override]")
            return
        recipe = load_recipe(args[1])
        if recipe is None:
            _print(f"  no recipe named {args[1]!r} — /plan recipes lists them.")
            return
        # Resolve the query BEFORE arming the seed (a no-query bail must not leave a seed armed),
        # and arm WITH it: plan_node consumes the seed only for this exact query, so a turn that
        # dies before planning can't leak the seed onto the next, unrelated question.
        query = " ".join(args[2:]).strip() or str(recipe.get("query") or "").strip()
        if not query:
            _print("  the recipe stored no query — give one: /plan run <name> <query>")
            return
        n = seed_next_plan(recipe.get("steps") or [], query)
        if not n:
            _print(f"  recipe {args[1]!r} has no usable steps.")
            return
        _print(f"  running recipe {recipe.get('name')!r}: {n} step(s) seeded, fresh gates apply.")
        ctx.requeue = query  # the REPL loop runs it as the next agent turn immediately
        return

    _print(f"  unknown /plan subcommand: {sub!r} — try: review, pause, lockstep, save, recipes, "
           "run (or /plan --help)")
