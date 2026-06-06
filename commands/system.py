from commands._framework import command


@command(
    "system",
    "Show live CPU, RAM, and GPU metrics.",
    aliases=("sys",),
    details="""
Renders a point-in-time readout of CPU, RAM, and (when available) GPU/VRAM usage as colored
bars in the trace-rail style — green/yellow/red by load. A snapshot, not a live monitor; run
it again for a fresh reading.

Example:
  /system
""",
)
def _system(ctx, args):
    from tui.system_monitor import get_system_metrics
    from tui import ui

    ui.show_system_metrics(get_system_metrics())
