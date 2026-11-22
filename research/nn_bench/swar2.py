import time, random

def make(W, H, B):
    S = W + 2; n = S * (H + 2)
    ONES = sum(1 << (i * B) for i in range(n))
    INNER = 0
    for y in range(1, H + 1):
        for x in range(1, W + 1):
            INNER |= ((1 << B) - 1) << ((y * S + x) * B)
    return dict(W=W, H=H, B=B, S=S, n=n, ONES=ONES, INNER=INNER, HALF=1 << (B - 1))

def pack(g, vals):
    v = 0
    for y in range(g['H']):
        for x in range(g['W']):
            v |= vals[y * g['W'] + x] << (((y + 1) * g['S'] + x + 1) * g['B'])
    return v

def unpack(g, v):
    B, m = g['B'], (1 << g['B']) - 1
    return [(v >> (((y + 1) * g['S'] + x + 1) * B)) & m for y in range(g['H']) for x in range(g['W'])]

def finish(g, p, n, q):
    B, HALF, ONES = g['B'], g['HALF'], g['ONES']
    r = (p + ONES * HALF - n) & ((1 << (g['n'] * B)) - 1)
    sign = (r >> (B - 1)) & ONES
    r &= sign * (HALF - 1)
    if q:
        r = (r >> q) & (ONES * ((1 << (B - 1 - q)) - 1))
    return r & g['INNER']

def sh(x, s):
    return x >> s if s > 0 else x << -s

# --- baseline: 9 shifted copies, one multiply per (co,ci,k)
def conv_base(g, xs, Wp, Wn, cout, q):
    B, S = g['B'], g['S']
    shifts = [((k // 3 - 1) * S + (k % 3 - 1)) * B for k in range(9)]
    nb = [[sh(x, s) for s in shifts] for x in xs]
    out = []
    for co in range(cout):
        p = n = 0
        for ci, row in enumerate(nb):
            a, b = Wp[co][ci], Wn[co][ci]
            for k in range(9):
                if a[k]: p += a[k] * row[k]
                if b[k]: n += b[k] * row[k]
        out.append(finish(g, p, n, q))
    return out

# --- trick 2: 3 horizontal copies; accumulate per row-offset, shift vertically once per output
def conv_rows(g, xs, Wp, Wn, cout, q):
    B, S = g['B'], g['S']
    hs = [(dx) * B for dx in (-1, 0, 1)]
    hb = [[sh(x, s) for s in hs] for x in xs]
    vs = [(dy * S) * B for dy in (-1, 0, 1)]
    out = []
    for co in range(cout):
        P = N = 0
        for r in range(3):
            p = n = 0
            for ci, row in enumerate(hb):
                a, b = Wp[co][ci], Wn[co][ci]
                for c in range(3):
                    k = r * 3 + c
                    if a[k]: p += a[k] * row[c]
                    if b[k]: n += b[k] * row[c]
            P += sh(p, vs[r]); N += sh(n, vs[r])
        out.append(finish(g, P, N, q))
    return out

# --- trick 3: G output channels per big multiply (weights stacked in one multiplier)
def build_multi(g, Wp, Wn, cin, cout, G):
    L = g['n'] * g['B'] + 64           # bits per stacked channel (with gap)
    groups = []
    for g0 in range(0, cout, G):
        cos = list(range(g0, min(cout, g0 + G)))
        Cp = [[sum(Wp[co][ci][k] << (j * L) for j, co in enumerate(cos)) for k in range(9)] for ci in range(cin)]
        Cn = [[sum(Wn[co][ci][k] << (j * L) for j, co in enumerate(cos)) for k in range(9)] for ci in range(cin)]
        groups.append((cos, Cp, Cn))
    return L, groups

def conv_multi(g, xs, L, groups, q):
    B, S = g['B'], g['S']
    shifts = [((k // 3 - 1) * S + (k % 3 - 1)) * B for k in range(9)]
    mask = (1 << L) - 1
    # pad shifted copies left by 32 slots so left-shifts stay positive inside the L window
    pad = 0
    nb = [[sh(x, s) for s in shifts] for x in xs]
    out = []
    for cos, Cp, Cn in groups:
        p = n = 0
        for ci, row in enumerate(nb):
            a, b = Cp[ci], Cn[ci]
            for k in range(9):
                if a[k]: p += a[k] * row[k]
                if b[k]: n += b[k] * row[k]
        for j in range(len(cos)):
            out.append(finish(g, (p >> (j * L)) & mask, (n >> (j * L)) & mask, q))
    return out

def weights(cin, cout, wmax):
    Wp = [[[0] * 9 for _ in range(cin)] for _ in range(cout)]
    Wn = [[[0] * 9 for _ in range(cin)] for _ in range(cout)]
    for co in range(cout):
        for ci in range(cin):
            for k in range(9):
                w = random.randint(-wmax, wmax)
                (Wp if w > 0 else Wn)[co][ci][k] = abs(w)
    return Wp, Wn

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

def timeit(f, *a):
    f(*a); t = time.perf_counter()
    for _ in range(5): f(*a)
    return (time.perf_counter() - t) / 5 * 1000

random.seed(2)
# correctness, both precisions
for B, amax, wmax, q in ((32, 255, 127, 7), (16, 15, 7, 4)):
    W = H = 7; c = 6
    g = make(W, H, B)
    vals = [[random.randint(0, amax) for _ in range(W * H)] for _ in range(c)]
    Wp, Wn = weights(c, c, wmax)
    ref = naive(vals, W, H, Wp, Wn, c, c, q)
    xs = [pack(g, v) for v in vals]
    L, groups = build_multi(g, Wp, Wn, c, c, 4)
    print(f"B={B}: base {[unpack(g,o) for o in conv_base(g,xs,Wp,Wn,c,q)]==ref}, "
          f"rows {[unpack(g,o) for o in conv_rows(g,xs,Wp,Wn,c,q)]==ref}, "
          f"multi {[unpack(g,o) for o in conv_multi(g,xs,L,groups,q)]==ref}")

print(f"{'layer':16s} {'32b base':>9s} {'16b base':>9s} {'16b rows':>9s} {'16b G=4':>9s} {'16b G=8':>9s}")
for W, c in ((30, 8), (30, 16), (15, 16), (15, 32)):
    g32, g16 = make(W, W, 32), make(W, W, 16)
    Wp, Wn = weights(c, c, 7)
    x32 = [pack(g32, [random.randint(0, 15) for _ in range(W * W)]) for _ in range(c)]
    x16 = [pack(g16, [random.randint(0, 15) for _ in range(W * W)]) for _ in range(c)]
    r = [timeit(conv_base, g32, x32, Wp, Wn, c, 4), timeit(conv_base, g16, x16, Wp, Wn, c, 4),
         timeit(conv_rows, g16, x16, Wp, Wn, c, 4)]
    for G in (4, 8):
        L, groups = build_multi(g16, Wp, Wn, c, c, G)
        r.append(timeit(conv_multi, g16, x16, L, groups, 4))
    print(f"{W}x{W} {c:2d}->{c:2d}     " + " ".join(f"{t:7.2f}ms" for t in r))
