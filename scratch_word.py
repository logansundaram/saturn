"""Scratch: hand-built 5-row block font for the 'saturday ai' wordmark."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

B = "█"  # full block
F = {
    "s": ["████", "█   ", "███ ", "   █", "████"],
    "a": ["████", "█  █", "████", "█  █", "█  █"],
    "t": ["████", " █  ", " █  ", " █  ", " █  "],
    "u": ["█  █", "█  █", "█  █", "█  █", "████"],
    "r": ["███ ", "█  █", "███ ", "█ █ ", "█  █"],
    "d": ["███ ", "█  █", "█  █", "█  █", "███ "],
    "y": ["█  █", "█  █", " ██ ", " █  ", "█   "],
    "i": ["█", "█", "█", "█", "█"],
    " ": [" ", " ", " ", " ", " "],
}


def render(text, gap=1):
    rows = ["", "", "", "", ""]
    for ch in text:
        g = F[ch]
        w = max(len(s) for s in g)
        for i in range(5):
            cell = g[i].ljust(w)
            rows[i] += cell.replace("█", B) + " " * gap
    return "\n".join(r.rstrip() for r in rows)


if __name__ == "__main__":
    print(render("saturday ai"))
    print()
    print(render("saturday"))
