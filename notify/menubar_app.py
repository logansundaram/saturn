"""
The menu bar item — the AppKit half. `python -m notify.menubar_app`.

Runs as its own process under the `com.saturn.menubar` LaunchAgent (see menubar.py for the
lifecycle, pidfile, menu model, and quit logic — all of it tested offline; this file only
draws). An accessory app (no Dock tile, no window): one NSStatusItem carrying a ringed-planet
template image derived from the splash motif (a dark banded disc under one tilted ring,
tui/ui/art.py), monochrome so macOS tints it for light and dark menu bars. The menu is rebuilt
from `menu_model()` every time it opens, so it always shows the plists launchd will fire.

Requires pyobjc-framework-Cocoa (a macOS-only dependency in pyproject.toml).
"""

from __future__ import annotations

import math
import sys

import diag
import notify
from notify import menubar


# ── the icon ─────────────────────────────────────────────────────────────────────────────────

def make_icon():
    """An 18-pt template NSImage: a filled disc with a tilted elliptical ring whose near half
    passes in front of the planet. The proportions echo tui/ui/art.py (ring semi-axes ≈ 4:1,
    a slight skew) — the splash motif at menu-bar scale."""
    from AppKit import NSAffineTransform, NSBezierPath, NSColor, NSGraphicsContext, NSImage
    from Foundation import NSMakeRect, NSMakeSize

    size = 18.0
    cx = cy = size / 2
    r = 4.4                      # planet radius
    a, b = 8.6, 2.4              # ring semi-axes (the splash's 24:5.7, scaled)
    tilt = -math.degrees(0.12) * 3   # the splash's 0.12-rad skew, exaggerated for 18 px

    img = NSImage.alloc().initWithSize_(NSMakeSize(size, size))
    img.lockFocus()
    NSColor.blackColor().set()

    spin = NSAffineTransform.transform()
    spin.translateXBy_yBy_(cx, cy)
    spin.rotateByDegrees_(tilt)
    spin.translateXBy_yBy_(-cx, -cy)

    ring = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx - a, cy - b, 2 * a, 2 * b))
    ring.transformUsingAffineTransform_(spin)
    ring.setLineWidth_(1.4)
    ring.stroke()                                            # the whole ring (far half shows)

    NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx - r, cy - r, 2 * r, 2 * r)).fill()

    NSGraphicsContext.saveGraphicsState()                    # the near half, back on top
    front = NSBezierPath.bezierPathWithRect_(NSMakeRect(cx - size, cy - size, 2 * size, size))
    front.transformUsingAffineTransform_(spin)
    front.addClip()
    ring.stroke()
    NSGraphicsContext.restoreGraphicsState()

    img.unlockFocus()
    img.setTemplate_(True)
    return img


# ── the app ──────────────────────────────────────────────────────────────────────────────────

def _make_delegate_class():
    """Built lazily so importing this module never needs AppKit (tests import `menubar`,
    not this)."""
    from AppKit import (
        NSAlert, NSAlertFirstButtonReturn, NSApp, NSMenu, NSMenuItem, NSObject, NSStatusBar,
        NSVariableStatusItemLength,
    )

    class SaturnMenuBar(NSObject):
        def applicationDidFinishLaunching_(self, _note):
            self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
            button = self.item.button()
            button.setImage_(make_icon())
            button.setToolTip_("Saturn")
            self.menu = NSMenu.alloc().init()
            self.menu.setAutoenablesItems_(False)
            self.menu.setDelegate_(self)
            self.item.setMenu_(self.menu)
            diag.log("menubar: up")

        # NSMenuDelegate — rebuild from the model each time the menu opens.
        def menuNeedsUpdate_(self, menu):
            menu.removeAllItems()
            for row in menubar.menu_model():
                if row.kind == "sep":
                    menu.addItem_(NSMenuItem.separatorItem())
                    continue
                item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(row.label, None, "")
                if row.kind == "pending":
                    item.setTarget_(self)
                    item.setAction_("cancelNotification:")
                    item.setRepresentedObject_(row.id)
                    item.setToolTip_("Click to cancel this notification")
                elif row.kind == "test":
                    item.setTarget_(self)
                    item.setAction_("sendTest:")
                elif row.kind == "quit":
                    item.setTarget_(self)
                    item.setAction_("quitSaturn:")
                    item.setKeyEquivalent_("q")
                else:
                    item.setEnabled_(False)
                menu.addItem_(item)

        def cancelNotification_(self, sender):
            try:
                notify.backend().cancel(str(sender.representedObject()))
            except notify.NotifyError as exc:
                diag.log(f"menubar: cancel failed: {exc}")

        def sendTest_(self, _sender):
            try:
                notify.backend().fire_now("Saturn", "notifications are working")
            except notify.NotifyError as exc:
                diag.log(f"menubar: test failed: {exc}")

        def quitSaturn_(self, _sender):
            try:
                n_pending = len(notify.backend().pending())
            except notify.NotifyError:
                n_pending = 0
            running = menubar.agent_pid() is not None
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Quit Saturn?")
            alert.setInformativeText_(menubar.quit_summary(n_pending, running))
            alert.addButtonWithTitle_("Quit")
            alert.addButtonWithTitle_("Cancel")
            NSApp.activateIgnoringOtherApps_(True)
            if alert.runModal() != NSAlertFirstButtonReturn:
                return
            summary = menubar.quit_all()          # its last step unloads this very process
            diag.log(f"menubar: quit {summary}")
            NSApp.terminate_(None)

    return SaturnMenuBar


def main() -> int:
    if sys.platform != "darwin":
        print("the Saturn menu bar item is macOS-only", file=sys.stderr)
        return 1
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    except ImportError:
        print("pyobjc-framework-Cocoa is not installed in this Python; "
              "reinstall saturn (pip install -e .) to get the menu bar item", file=sys.stderr)
        return 1
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = _make_delegate_class().alloc().init()
    app.setDelegate_(delegate)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
