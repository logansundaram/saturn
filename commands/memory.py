from commands._framework import command, _print


@command(
    "memory",
    "List, add, or delete the agent's persistent memory (the remember/recall facts).",
    aliases=("mem",),
    usage="/memory [add <fact> | forget <n>]",
    details="""
The transparency surface for durable memory: the facts saved via the `remember` tool (or your
own "remember that ..." requests) are loaded into the agent's context EVERY turn, so what is
stored here quietly shapes every answer. This command lets you see and manage that store
without hand-editing database/memory/memory.md.

  /memory               numbered list of every stored fact
  /memory add <fact>    save a fact directly (same dedup as the remember tool)
  /memory forget <n>    delete fact n (the number shown by /memory)

The store is a plain markdown file (paths.memory in config.yaml) — still safe to hand-edit;
this is just the in-app view of it.

Examples:
  /memory
  /memory add I prefer answers in metric units
  /memory forget 3
""",
)
def _memory(ctx, args):
    from stores.memory_registry import add_memory, list_memory, remove_memory
    from tui import ui

    if not args:
        facts = list_memory()
        if not facts:
            ui.note("no persistent memory yet — say `remember that ...` or use /memory add.")
            return
        ui.section(
            "memory",
            f"{len(facts)} fact(s) · loaded into context every turn · /memory forget <n> deletes",
        )
        ui.table([((f"{i}", "accent"), fact) for i, fact in enumerate(facts, start=1)])
        return

    sub = args[0].lower()
    if sub == "add":
        fact = " ".join(args[1:]).strip()
        if not fact:
            _print("  usage: /memory add <fact>")
            return
        _print(f"  {add_memory(fact)}")
        return

    if sub in ("forget", "rm", "delete", "del"):
        if len(args) < 2 or not args[1].isdigit():
            _print("  usage: /memory forget <n>   (the number shown by /memory)")
            return
        removed = remove_memory(int(args[1]))
        if removed is None:
            n = len(list_memory())
            _print(f"  no fact #{args[1]} — /memory lists {n} fact(s).")
        else:
            _print(f"  forgot: {removed}")
        return

    _print(f"  unknown subcommand '{args[0]}' — usage: /memory [add <fact> | forget <n>]")
