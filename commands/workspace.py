from commands._framework import command, _print


@command(
    "workspace",
    "List files in the read/write workspace sandbox.",
    aliases=("ws",),
    usage="/workspace",
    details="""
Lists files in the read/write workspace sandbox — the directory the read_file, write_file, and
list_directory tools are confined to — with sizes (directories marked <dir>). Dotfiles,
including the internal .manifest.md, are hidden.

This is distinct from the RAG corpus (see /docs): the workspace is scratch space the agent
writes to, the corpus is the knowledge base it searches.

Example:
  /workspace
""",
)
def _workspace(ctx, args):
    from config import get_config

    ws = get_config().path("workspace")
    _print(f"  workspace: {ws}")
    if not ws.exists():
        _print("  (workspace directory does not exist yet)")
        return

    entries = sorted(p for p in ws.iterdir() if not p.name.startswith("."))
    if not entries:
        _print("  (empty)")
        return

    for p in entries:
        if p.is_dir():
            _print(f"    {p.name + '/':<32} <dir>")
        else:
            size = p.stat().st_size
            _print(f"    {p.name:<32} {size:>9,} B")
