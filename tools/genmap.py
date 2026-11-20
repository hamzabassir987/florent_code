"""Generate random symmetric .map26 maps for local testing (no login needed).

Usage: python tools/genmap.py OUT_DIR [count] [seed]
"""
import random
import sys
from pathlib import Path

EMPTY, WALL, TI, VOLT, VOID = 0, 1, 2, 4, 5


def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _f_int(field: int, v: int) -> bytes:
    return _varint(field << 3) + _varint(v)


def _f_bytes(field: int, data: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(data)) + data


def encode(width, height, grid, core_a, core_b, symmetry, pit=None) -> bytes:
    out = _f_int(1, width) + _f_int(2, height)
    for row in grid:
        packed = b"".join(_varint(t) for t in row)
        out += _f_bytes(3, _f_bytes(1, packed))
    for cid, team, (x, y) in ((1, 0, core_a), (2, 1, core_b)):
        pos = _f_int(1, x) + _f_int(2, y)
        out += _f_bytes(4, _f_int(1, cid) + _f_int(2, team) + _f_bytes(3, pos))
    out += _f_int(5, symmetry)
    if pit:
        axis, at, w = pit
        out += _f_bytes(8, _f_int(1, axis) + _f_int(2, at) + _f_int(3, w))
    return out


def mirror(x, y, w, h, sym):
    if sym == 0:  # rotational
        return w - 1 - x, h - 1 - y
    if sym == 1:  # horizontal (mirror across vertical axis)
        return w - 1 - x, y
    return x, h - 1 - y  # vertical


def gen(rng: random.Random, pit=False):
    w = rng.randint(16, 30)
    h = rng.randint(16, 30)
    sym = rng.choice([1, 2]) if pit else rng.choice([0, 1, 2])
    grid = [[EMPTY] * w for _ in range(h)]
    pit_spec = None
    if pit:
        pw = rng.randint(1, 3)
        if sym == 1:  # left/right islands: void columns in the middle
            if (w - pw) % 2:
                pw += 1
            at = (w - pw) // 2
            for y in range(h):
                for x in range(at, at + pw):
                    grid[y][x] = VOID
            pit_spec = (0, at, pw)
        else:
            if (h - pw) % 2:
                pw += 1
            at = (h - pw) // 2
            for x in range(w):
                for y in range(at, at + pw):
                    grid[y][x] = VOID
            pit_spec = (1, at, pw)

    def put(x, y, v):
        grid[y][x] = v
        mx, my = mirror(x, y, w, h, sym)
        grid[my][mx] = v

    # core A in the "first half"
    if sym == 2:
        cx, cy = rng.randint(2, w - 4), rng.randint(2, h // 2 - 5)
    else:
        cx, cy = rng.randint(2, w // 2 - 5), rng.randint(2, h - 4)
    core_a = (cx, cy)
    # 2x2 footprint mirrored: other corner
    mx, my = mirror(cx, cy, w, h, sym)
    mx2, my2 = mirror(cx + 1, cy + 1, w, h, sym)
    core_b = (min(mx, mx2), min(my, my2))
    reserved = set()
    for c in (core_a, core_b):
        for dx in range(-2, 4):
            for dy in range(-2, 4):
                reserved.add((c[0] + dx, c[1] + dy))

    def free(x, y):
        return 0 <= x < w and 0 <= y < h and (x, y) not in reserved and grid[y][x] == EMPTY

    # walls: random segments
    for _ in range(rng.randint(2, 8)):
        x, y = rng.randrange(w), rng.randrange(h)
        dx, dy = rng.choice([(1, 0), (0, 1)])
        for i in range(rng.randint(2, 6)):
            if free(x + dx * i, y + dy * i) and free(*mirror(x + dx * i, y + dy * i, w, h, sym)):
                put(x + dx * i, y + dy * i, WALL)
    # titanium clusters
    for _ in range(rng.randint(3, 7)):
        x, y = rng.randrange(w), rng.randrange(h)
        for _ in range(rng.randint(2, 5)):
            xx, yy = x + rng.randint(-1, 1), y + rng.randint(-1, 1)
            if free(xx, yy) and free(*mirror(xx, yy, w, h, sym)):
                put(xx, yy, TI)
    # titanium near each core
    for _ in range(2):
        xx, yy = cx + rng.randint(-5, 6), cy + rng.randint(-5, 6)
        if free(xx, yy) and free(*mirror(xx, yy, w, h, sym)):
            put(xx, yy, TI)
    # voltium near centre
    for _ in range(rng.randint(1, 3)):
        xx, yy = w // 2 + rng.randint(-3, 3), h // 2 + rng.randint(-3, 3)
        if free(xx, yy) and free(*mirror(xx, yy, w, h, sym)):
            put(xx, yy, VOLT)
    return encode(w, h, grid, core_a, core_b, sym, pit_spec)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(n):
        (out / f"gen{i:02d}.map26").write_bytes(gen(rng))
    pits = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    for i in range(pits):
        (out / f"pit{i:02d}.map26").write_bytes(gen(rng, pit=True))
