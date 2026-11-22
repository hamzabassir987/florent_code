import time, random
B = 32                     # bits per slot
HALF = 1 << (B - 1)

class Grid:
    def __init__(s, W, H):
        s.W, s.H, s.S = W, H, W + 2          # 1-slot zero border each side
        s.n = s.S * (H + 2)
        s.ONES = sum(1 << (i * B) for i in range(s.n))
        inner = 0
        for y in range(1, H + 1):
            for x in range(1, W + 1):
                inner |= 1 << ((y * s.S + x) * B)
        s.INNER = inner
        s.OFF = s.ONES * HALF                 # per-slot offset
        s.LOW = s.ONES * (HALF - 1)           # low B-1 bits of every slot
    def pack(s, vals):                        # vals: flat W*H non-negative ints
        v = 0
        for y in range(s.H):
            for x in range(s.W):
                v |= vals[y * s.W + x] << (((y + 1) * s.S + x + 1) * B)
        return v
    def unpack(s, v):
        m = (1 << B) - 1
        return [(v >> (((y + 1) * s.S + x + 1) * B)) & m for y in range(s.H) for x in range(s.W)]

def conv_relu(g, xs, Wp, Wn, cout, q):
    """xs: list of packed channels (>=0). Wp/Wn[co][ci][k]: non-negative int weights (pos/neg parts).
    Returns relu((sum) >> q) per output channel, packed."""
    S = g.S
    shifts = [((k // 3 - 1) * S + (k % 3 - 1)) * B for k in range(9)]
    # neighbour copies of every input channel, computed once
    nb = []
    for x in xs:
        row = []
        for sh in shifts:
            row.append(x >> sh if sh > 0 else x << -sh)
        nb.append(row)
    inner = g.INNER * ((1 << B) - 1)
    outs = []
    for co in range(cout):
        p = g.OFF
        n = 0
        wp, wn = Wp[co], Wn[co]
        for ci, row in enumerate(nb):
            a, b = wp[ci], wn[ci]
            for k in range(9):
                if a[k]: p += a[k] * row[k]
                if b[k]: n += b[k] * row[k]
        r = (p - n) & ((1 << (g.n * B)) - 1)
        sign = (r >> (B - 1)) & g.ONES        # 1 where value >= 0
        r = r & (sign * (HALF - 1)) & g.LOW   # relu, strip offset
        if q:
            r = (r >> q) & (g.ONES * ((1 << (B - 1 - q)) - 1))
        outs.append(r & inner)
    return outs

def naive(vals, W, H, Wp, Wn, cin, cout, q):
    out = []
    for co in range(cout):
        o = []
        for y in range(H):
            for x in range(W):
                s = 0
                for ci in range(cin):
                    for k in range(9):
                        yy, xx = y + k // 3 - 1, x + k % 3 - 1
                        if 0 <= yy < H and 0 <= xx < W:
                            s += (Wp[co][ci][k] - Wn[co][ci][k]) * vals[ci][yy * W + xx]
                o.append(max(0, s) >> q)
        out.append(o)
    return out

def weights(cin, cout):
    Wp = [[[0] * 9 for _ in range(cin)] for _ in range(cout)]
    Wn = [[[0] * 9 for _ in range(cin)] for _ in range(cout)]
    for co in range(cout):
        for ci in range(cin):
            for k in range(9):
                w = random.randint(-127, 127)
                (Wp if w > 0 else Wn)[co][ci][k] = abs(w)
    return Wp, Wn

random.seed(1)
# correctness
W = H = 7; c = 3
g = Grid(W, H)
vals = [[random.randint(0, 255) for _ in range(W * H)] for _ in range(c)]
Wp, Wn = weights(c, c)
got = [g.unpack(o) for o in conv_relu(g, [g.pack(v) for v in vals], Wp, Wn, c, 7)]
print("correct:", got == naive(vals, W, H, Wp, Wn, c, c, 7))

for W, c in ((30, 4), (30, 8), (30, 16), (15, 8), (15, 16), (15, 32)):
    g = Grid(W, W)
    xs = [g.pack([random.randint(0, 255) for _ in range(W * W)]) for _ in range(c)]
    Wp, Wn = weights(c, c)
    t = time.perf_counter()
    for _ in range(5): conv_relu(g, xs, Wp, Wn, c, 7)
    dt = (time.perf_counter() - t) / 5
    print(f"{W}x{W} {c:2d}->{c:2d}: {dt*1000:6.2f} ms  ({W*W*9*c*c/dt/1e6:.0f} M MAC/s)")
