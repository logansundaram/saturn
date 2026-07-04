"""The presentation layer (terminal UI). Nothing in here holds product logic — it renders
state the engine and trust stack already recorded:

    ui/                the screen surfaces, split by concern and re-exported flat
                       (`from tui import ui` — see tui/ui/__init__.py for the module map)
    typeahead.py       InputQueue: the single console reader live during a turn — type-ahead
                       queueing, Esc steering / plan-review pause
    system_monitor.py  CPU/RAM/GPU metrics for the status bar's hardware zone
"""
