from commands._framework import command


@command(
    "animation",
    "Play the Saturn ring animation in a loop until interrupted (Ctrl+C).",
    aliases=("anim",),
    details="""
Plays the Saturn ring animation in a seamless continuous loop. Press Ctrl+C to stop;
it then settles on the resting frame in place.

Skipped automatically on non-terminal stdout or when the terminal is too narrow.

Example:
  /animation
""",
)
def _animation(ctx, args):
    from tui import ui

    ui.play_animation()
