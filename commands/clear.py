from commands._framework import command


@command(
    "clear",
    "Clear the terminal screen.",
    aliases=("cls",),
    details="""
Clears the visible terminal (runs `cls` on Windows, `clear` elsewhere). Affects only the
screen — conversation state, message history, and the trace are all left intact. Use
/reset to actually clear the conversation.

Example:
  /clear
""",
)
def _clear(ctx, args):
    import subprocess
    import sys

    subprocess.run("cls" if sys.platform == "win32" else "clear", shell=True, check=False)
