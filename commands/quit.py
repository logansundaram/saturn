from commands._framework import command, _print
from commands._session import write_autosave


@command(
    "quit",
    "Exit the agent.",
    aliases=("exit", "q"),
    details="""
Ends the interactive session and returns you to the shell. In-process conversation
memory is discarded; the trace DB and RAG corpus on disk are untouched.

Example:
  /quit
""",
)
def _quit(ctx, args):
    if write_autosave(ctx.state):
        _print("  session autosaved — type /resume next launch to continue.")
    ctx.should_quit = True
