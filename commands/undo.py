from commands._framework import command, _print


@command(
    "undo",
    "Revert the file changes the last turn made to the workspace.",
    usage="/undo [list]",
    details="""
Restores the workspace files touched by the most recent turn that wrote anything, using the
pre-write snapshots taken automatically by write_file / edit_file (stores/snapshots.py). A file
the turn created is deleted; a file it overwrote or edited is restored to its turn-start bytes.
Each /undo pops one batch, so repeating it walks further back (up to the retained history).

  /undo         revert the most recent batch of file changes
  /undo list    show the stored snapshot batches (newest first) without restoring

Scope: only the file tools snapshot. run_shell can touch anything, so its effects are NOT
undoable — the approval gate showing the exact command is its safety boundary. The conversation
itself is not rewound, only the files.
""",
)
def _undo(ctx, args):
    from stores import snapshots

    if args and args[0].lower() == "list":
        batches = snapshots.list_batches()
        if not batches:
            _print("  no snapshots stored — no turn has written to the workspace yet.")
            return
        _print(f"  {len(batches)} snapshot batch(es), newest first:")
        for i, b in enumerate(batches, 1):
            query = f' — "{b["query"]}"' if b["query"] else ""
            _print(f"    {i}. {b['created'] or b['id']}{query}")
            for path in b["files"]:
                _print(f"         {path}")
        _print("  /undo restores #1 (each /undo pops one batch).")
        return

    try:
        summary, actions = snapshots.undo_last()
    except RuntimeError as exc:
        _print(f"  {exc}")
        return
    _print(f"  undid file changes from turn {summary}:")
    for line in actions:
        _print(f"    {line}")
    if not actions:
        _print("    (the batch recorded no file changes)")
