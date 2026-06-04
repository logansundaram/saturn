"""Scratch v2: large textured Saturn + multi-band ring + rocket sprite, tuned as plain text."""
import sys, math
sys.stdout.reconfigure(encoding="utf-8")

W, H = 90, 26
PCY, PCX = 13.0, 44.0
P_RR, P_RC = 11.0, 21.0          # planet radii (rows, cols)
RAMP = " .,:;irsXA253hMHGS#9B&@"  # dark -> light, plenty of texture
L = (-0.5, -0.62, 0.6)

# ring bands: (semi-major cols, semi-minor rows, glyph). Flat (B/A ~0.18) so the disc reads as a
# tilted plane crossing the equator; the gap between band 2 and 3 is the Cassini division.
RINGS = [
    (26.0, 4.7, ":"),
    (30.0, 5.4, "#"),
    (37.0, 6.7, "#"),
    (41.0, 7.4, ":"),
    (44.0, 7.9, "·"),
]


def _norm(v):
    m = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / m for c in v)


LN = _norm(L)


def _hash(r, c):
    h = (r * 73856093) ^ (c * 19349663)
    return ((h >> 8) & 0xFFFF) / 0xFFFF


def planet_char(r, c):
    nx = (c - PCX) / P_RC
    ny = (r - PCY) / P_RR
    if nx * nx + ny * ny > 1.0:
        return None
    nz = math.sqrt(max(0.0, 1.0 - nx * nx - ny * ny))
    diff = max(0.0, nx * LN[0] + ny * LN[1] + nz * LN[2])
    band = 0.88 + 0.12 * math.sin(ny * 7.5 + 0.6 * math.sin(nx * 3))   # gas-giant latitude bands
    # ambient floor keeps the unlit side a full textured disc (not sparse dust); jitter adds grain
    b = 0.15 + diff * band * 0.9 + (_hash(r, c) - 0.5) * 0.08
    b = max(0.0, min(1.0, b))
    idx = int(b * (len(RAMP) - 1) + 0.5)
    return RAMP[idx] if idx > 0 else None


def build():
    grid = [[" "] * W for _ in range(H)]

    def put(r, c, ch):
        if 0 <= r < H and 0 <= c < W:
            grid[r][c] = ch

    def ring(front_pass):
        for A, B, ch in RINGS:
            n = int(A * 7)
            for i in range(n):
                t = 2 * math.pi * i / n
                front = math.sin(t) > 0
                if front != front_pass:
                    continue
                rr = int(round(PCY + B * math.sin(t)))
                cc = int(round(PCX + A * math.cos(t)))
                put(rr, cc, ch)

    ring(front_pass=False)               # back rings (behind body)
    for r in range(H):                   # planet
        for c in range(W):
            pc = planet_char(r, c)
            if pc:
                put(r, c, pc)
    ring(front_pass=True)                # front rings (over body)

    # rocket sprite riding the right ansa. Clear a 1-cell-padded cutout first so the ring
    # doesn't speckle through the gaps in the sprite.
    rocket = [
        "   /\\   ",
        "  /  \\  ",
        "  |oo|  ",
        "  |  |  ",
        " /|  |\\ ",
        "/_|__|_\\",
        "  )/\\(  ",
    ]
    ry, rx = 7, 73
    rw = max(len(s) for s in rocket)
    for r in range(ry - 1, ry + len(rocket) + 1):
        for c in range(rx - 1, rx + rw + 1):
            put(r, c, " ")
    for dr, line in enumerate(rocket):
        for dc, ch in enumerate(line):
            if ch != " ":
                put(ry + dr, rx + dc, ch)

    return "\n".join("".join(row).rstrip() for row in grid)


FONT = {
    "s": ["████", "█   ", "███ ", "   █", "████"],
    "a": ["████", "█  █", "████", "█  █", "█  █"],
    "t": ["████", " █  ", " █  ", " █  ", " █  "],
    "u": ["█  █", "█  █", "█  █", "█  █", "████"],
    "r": ["███ ", "█  █", "███ ", "█ █ ", "█  █"],
    "d": ["███ ", "█  █", "█  █", "█  █", "███ "],
    "y": ["█  █", "█  █", " ██ ", " █  ", "█   "],
    "i": ["█", "█", "█", "█", "█"],
    " ": ["  ", "  ", "  ", "  ", "  "],
}


def word(text, gap=1):
    rows = ["", "", "", "", ""]
    for ch in text:
        g = FONT[ch]
        w = max(len(s) for s in g)
        for i in range(5):
            rows[i] += g[i].ljust(w) + " " * gap
    return [r.rstrip() for r in rows]


if __name__ == "__main__":
    art = build()
    print(art)
    print()
    wm = word("saturday ai")
    pad = (W - max(len(r) for r in wm)) // 2
    for r in wm:
        print(" " * pad + r)
    print()
    tag = "terminal-first agent  ·  local by default"
    print(" " * ((W - len(tag)) // 2) + tag)
    div = "──────◯──────"
    print(" " * ((W - len(div)) // 2) + div)

