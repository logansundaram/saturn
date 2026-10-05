"""
What the user is pointing at — read_browser_tab, finder_selection.

"Summarize the page I'm on" and "rename these" name things by where the user's attention is,
not by a path or a URL. Two readers over AppleScript (`tools/applescript.py`):

  read_browser_tab   the URL, title and visible text of the front tab of the browser the
                     user was last in (front-to-back window order, not a fixed list: a
                     browser stays running after its window is forgotten). Safari exposes
                     the page text as a plain property of the tab (Safari.sdef, read
                     2026-10-01), so a page the user is logged in to is read LOCALLY — no
                     fetch, no egress. Chrome-family browsers only give text through
                     `execute … javascript`, which needs View > Developer > Allow JavaScript
                     from Apple Events; without it the tool returns the URL and title and says
                     so. `untrusted=True`: a web page is the canonical injection source.
  finder_selection   the files selected in Finder, as paths the file tools accept. Paths
                     outside the reachable folders are returned with the /add-dir that would
                     allow them — the containment check stays where it is.

A browser that is not running is never launched (the scripts run without `app=`): opening
someone's browser to answer a question is not a read. Nothing here is egress.
"""

from __future__ import annotations

import re
import subprocess

from core import workspace as _ws
from tools import applescript
from tools.applescript import AS_RS, AS_US, AppleScriptError, RS
from tools.toolspec import ToolError, register_tool

# The browsers read. When the user names none, the one they were in last is asked first
# (`_front_to_back`); this order only breaks a tie. The Chromium family shares one dictionary.
_BROWSERS = ("Safari", "Google Chrome", "Arc", "Brave Browser", "Microsoft Edge")
_ALIASES = {"chrome": "Google Chrome", "brave": "Brave Browser", "edge": "Microsoft Edge"}
_MAX_SELECTED = 200

_SAFARI = f"""
tell application "Safari"
  if (count of windows) is 0 then return ""
  set t to current tab of front window
  set u to URL of t
  if u is missing value then set u to ""
  set body to text of t
  if body is missing value then set body to ""
  return u & {AS_US} & (name of t) & {AS_US} & body
end tell"""

_CHROMIUM = f"""
tell application "{{app}}"
  if (count of windows) is 0 then return ""
  set t to active tab of front window
  set body to ""
  try
    set body to execute t javascript "document.body.innerText"
    if body is missing value then set body to ""
  end try
  return (URL of t) & {AS_US} & (title of t) & {AS_US} & body
end tell"""

_NO_TEXT = ("the page text cannot be read from this browser (it needs View > Developer > Allow "
            "JavaScript from Apple Events); use web_extract on the url if the page is public")


def _is_running(app: str) -> bool:
    try:
        return subprocess.run(["pgrep", "-x", app], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _front_to_back() -> list[str]:
    """The visible apps' names, frontmost first (`lsappinfo visibleProcessList`, which writes
    a space as `_`), or [] when that cannot be asked. The terminal is in front while the user
    types here, so "the page I'm on" is the first BROWSER in this order."""
    try:
        proc = subprocess.run(["lsappinfo", "visibleProcessList"], capture_output=True,
                              text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [name.replace("_", " ") for name in re.findall(r'"([^"]+)"', proc.stdout or "")]


@register_tool("read_only", untrusted=True)
def read_browser_tab(browser: str = ""):
    """Read the web page the user has open: the URL, title and visible text of the front tab of
    their browser (the one they used last; `browser` may name Safari, Chrome, Arc, Brave or
    Edge). Use it for "the page I'm on", "this article", "summarize this tab" — it reads the
    page as the user sees it, logged-in pages included, without fetching anything."""
    applescript.mac_only()
    wanted = str(browser or "").strip()
    if wanted:
        app = _ALIASES.get(wanted.lower()) or next((b for b in _BROWSERS if b.lower() == wanted.lower()), None)
        if app is None:
            raise ToolError(f"{wanted!r} is not a supported browser; use one of: " + ", ".join(_BROWSERS))
        if not _is_running(app):
            raise ToolError(f"{app} is not running")
        candidates = [app]
    else:
        candidates = [b for b in _BROWSERS if _is_running(b)]
        if not candidates:
            raise ToolError("no supported browser is running (" + ", ".join(_BROWSERS) + ")")
        if len(candidates) > 1:
            # A browser stays running long after its last window closed: the one the user is
            # reading in is the frontmost, and one without a window passes to the next.
            order = _front_to_back()
            candidates.sort(key=lambda b: order.index(b) if b in order else len(order))
    for app in candidates:
        script = _SAFARI if app == "Safari" else _CHROMIUM.format(app=app)
        try:
            out = applescript.run(script, timeout=30.0)
        except AppleScriptError as exc:
            raise ToolError(str(exc)) from exc
        parts = out.split(applescript.US, 2)
        if len(parts) == 3:
            break
    else:
        raise ToolError(" and ".join(candidates) + (" has" if len(candidates) == 1 else " have")
                        + " no open window")
    result = {"browser": app, "url": parts[0], "title": parts[1], "text": parts[2].strip()}
    if not result["text"]:
        result["note"] = _NO_TEXT
    return result


@register_tool("read_only")
def finder_selection():
    """List the files and folders the user has selected in Finder, as paths the file tools
    accept. Use it when the user says "these", "this file" or "the selected files" without
    naming them."""
    applescript.mac_only()
    script = f"""
tell application "Finder"
  set out to ""
  repeat with f in (selection as alias list)
    set out to out & (POSIX path of f) & {AS_RS}
  end repeat
  return out
end tell"""
    try:
        out = applescript.run(script, timeout=30.0)
    except AppleScriptError as exc:
        raise ToolError(str(exc)) from exc
    paths = [p for p in out.split(RS) if p.strip()][:_MAX_SELECTED]
    if not paths:
        return "Nothing is selected in Finder."
    selected, unreachable = [], []
    for raw in paths:
        target, refusal = _ws.resolve(raw)
        selected.append(_ws.relative(target))
        if refusal is not None:
            folder = target if target.is_dir() else target.parent
            if folder not in unreachable:
                unreachable.append(folder)
    result = {"selected": selected}
    if unreachable:
        result["note"] = ("some of these are outside the folders Saturn can reach; ask the user "
                          "to run " + " and ".join(f"/add-dir {_ws.display(f)}" for f in unreachable)
                          + " before working on them")
    return result
