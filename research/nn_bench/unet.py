"""Quantised U-Net in pure Python using big-integer SWAR. Activations 0..15, weights -7..7."""
import time, random
from array import array

AMAX = 15

class G:
    """A packed grid level: W x H with a 1-slot zero border, B bits per slot."""
    def __init__(s, W, H, B):
        s.W, s.H, s.B, s.S = W, H, B, W + 2
        s.n = s.S * (H + 2)
        s.nbytes = s.n * B // 8
        s.code = 'H' if B == 16 else 'I'
        s.ONES = int.from_bytes(array(s.code, [1] * s.n).tobytes(), 'little')
        inner = array(s.code, [0] * s.n)
        for y in range(1, H + 1):
            for x in range(1, W + 1):
                inner[y * s.S + x] = (1 << B) - 1
        s.INNER = int.from_bytes(inner.tobytes(), 'little')
        s.HALF = 1 << (B - 1)
        s.OFF = s.ONES * s.HALF
        s.LOWM = s.ONES * (s.HALF - 1)
        s.FULL = (1 << (s.n * B)) - 1
        s.shifts = [((k // 3 - 1) * s.S + (k % 3 - 1)) * B for k in range(9)]
    def to_arr(s, v):
        return array(s.code, v.to_bytes(s.nbytes, 'little'))
    def from_arr(s, a):
        return int.from_bytes(a.tobytes(), 'little')
    def pack(s, vals):
        a = array(s.code, [0] * s.n)
        for y in range(s.H):
            a[(y + 1) * s.S + 1:(y + 1) * s.S + 1 + s.W] = array(s.code, vals[y * s.W:(y + 1) * s.W])
        return s.from_arr(a)
    def unpack(s, v):
        a = s.to_arr(v)
        out = []
        for y in range(s.H):
            out.extend(a[(y + 1) * s.S + 1:(y + 1) * s.S + 1 + s.W])
        return out

def ge_mask(g, a, b):
    """per-slot 1 where a >= b (both < HALF)."""
    return ((a + g.OFF - b) >> (g.B - 1)) & g.ONES

def conv(g, xs, Wp, Wn, q, relu=True, k1=False):
    """3x3 conv (or 1x1 if k1) + bias-free ReLU + >>q + clamp to AMAX."""
    sh = g.shifts
    ks = (4,) if k1 else range(9)
    nb = [[(x >> sh[k] if sh[k] > 0 else x << -sh[k]) for k in range(9)] if not k1 else [None]*4 + [x] for x in xs]
    out = []
    for co in range(len(Wp)):
        p = n = 0
        wp, wn = Wp[co], Wn[co]
        for ci in range(len(xs)):
            row, a, b = nb[ci], wp[ci], wn[ci]
            for k in ks:
                if a[k]: p += a[k] * row[k]
                if b[k]: n += b[k] * row[k]
        r = (p + g.OFF - n) & g.FULL
        if not relu:            # raw signed output, kept in offset form (value + HALF)
            out.append(r & g.INNER)
            continue
        if relu:
            sign = (r >> (g.B - 1)) & g.ONES
            r &= sign * (g.HALF - 1)
        if q:
            r = (r >> q) & (g.ONES * ((1 << (g.B - 1 - q)) - 1))
        over = ge_mask(g, r, g.ONES * (AMAX + 1))
        r = (r & ~(over * ((1 << g.B) - 1))) | (over * AMAX)
        out.append(r & g.INNER)
    return out

def maxpool(g, g2, xs):
    """2x2 max pool, stride 2 (g2 = ceil size). Vectorised max, then strided gather with array slicing."""
    out = []
    S, B = g.S, g.B
    for x in xs:
        right = x >> B
        m = ge_mask(g, x, right); m *= (1 << B) - 1
        x = (x & m) | (right & ~m)
        down = x >> (S * B)
        m = ge_mask(g, x, down); m *= (1 << B) - 1
        x = (x & m) | (down & ~m)
        a = g.to_arr(x)
        b = array(g2.code, [0] * g2.n)
        for y2 in range(g2.H):
            y = 2 * y2 + 1
            row = a[y * S + 1: y * S + 1 + g.W: 2]
            b[(y2 + 1) * g2.S + 1:(y2 + 1) * g2.S + 1 + len(row)] = array(g2.code, row)
        out.append(g2.from_arr(b) & g2.INNER)
    return out

def upsample(g2, g, xs):
    """nearest 2x upsample from small grid g2 to big grid g (cropped to g size)."""
    out = []
    for x in xs:
        a = g2.to_arr(x)
        b = array(g.code, [0] * g.n)
        for y in range(g.H):
            y2 = y // 2
            src = a[(y2 + 1) * g2.S + 1:(y2 + 1) * g2.S + 1 + g2.W]
            row = array(g.code, [0] * (2 * g2.W))
            row[0::2] = array(g.code, src); row[1::2] = array(g.code, src)
            b[(y + 1) * g.S + 1:(y + 1) * g.S + 1 + g.W] = row[:g.W]
        out.append(g.from_arr(b))
    return out

# ------------------------------------------------------------------ network
def rand_w(cin, cout, k1=False):
    Wp = [[[0] * 9 for _ in range(cin)] for _ in range(cout)]
    Wn = [[[0] * 9 for _ in range(cin)] for _ in range(cout)]
    for co in range(cout):
        for ci in range(cin):
            for k in ((4,) if k1 else range(9)):
                w = random.randint(-7, 7)
                (Wp if w > 0 else Wn)[co][ci][k] = abs(w)
    return Wp, Wn

def build(cin, c1, c2, c3):
    return dict(e1a=rand_w(cin, c1), e1b=rand_w(c1, c1),
                e2a=rand_w(c1, c2), e2b=rand_w(c2, c2),
                bna=rand_w(c2, c3), bnb=rand_w(c3, c3),
                d2a=rand_w(c3 + c2, c2), d2b=rand_w(c2, c2),
                d1a=rand_w(c2 + c1, c1), d1b=rand_w(c1, c1),
                out=rand_w(c1, 1, k1=True))

Q = 5  # rescale shift per layer (would come from training)

def forward(net, L1, L2, L3, x, timing=None):
    t = time.perf_counter(); T = {}
    def mark(name):
        nonlocal t
        if timing is not None:
            now = time.perf_counter(); T[name] = (now - t) * 1000; t = now
    e1 = conv(L1, conv(L1, x, *net['e1a'], Q), *net['e1b'], Q); mark('enc1 (full res)')
    p1 = maxpool(L1, L2, e1)
    e2 = conv(L2, conv(L2, p1, *net['e2a'], Q), *net['e2b'], Q); mark('enc2 (1/2 res)')
    p2 = maxpool(L2, L3, e2)
    bn = conv(L3, conv(L3, p2, *net['bna'], Q), *net['bnb'], Q); mark('bottleneck (1/4)')
    u2 = upsample(L3, L2, bn)
    d2 = conv(L2, conv(L2, u2 + e2, *net['d2a'], Q), *net['d2b'], Q); mark('dec2 (1/2 res)')
    u1 = upsample(L2, L1, d2)
    d1 = conv(L1, conv(L1, u1 + e1, *net['d1a'], Q), *net['d1b'], Q); mark('dec1 (full res)')
    o = conv(L1, d1, *net['out'], 0, relu=False, k1=True); mark('1x1 head')
    if timing is not None: timing.update(T)
    return o

# ------------------------------------------------------------------ plain reference
def r_conv(xs, W, H, Wp, Wn, q, relu=True, k1=False):
    out = []
    for co in range(len(Wp)):
        o = []
        for y in range(H):
            for x in range(W):
                s = 0
                for ci in range(len(xs)):
                    for k in ((4,) if k1 else range(9)):
                        yy, xx = y + k // 3 - 1, x + k % 3 - 1
                        if 0 <= yy < H and 0 <= xx < W:
                            s += (Wp[co][ci][k] - Wn[co][ci][k]) * xs[ci][yy * W + xx]
                if relu: s = max(0, s)
                s >>= q
                o.append(min(AMAX, s) if relu else s)
        out.append(o)
    return out

def r_pool(xs, W, H):
    W2, H2 = (W + 1) // 2, (H + 1) // 2
    g = lambda c, x, y: c[y * W + x] if x < W and y < H else 0
    return [[max(g(c, 2*x, 2*y), g(c, 2*x+1, 2*y), g(c, 2*x, 2*y+1), g(c, 2*x+1, 2*y+1))
             for y in range(H2) for x in range(W2)] for c in xs], W2, H2

def r_up(xs, W2, W, H):
    return [[c[(y // 2) * W2 + x // 2] for y in range(H) for x in range(W)] for c in xs]

def r_forward(net, x, W, H):
    e1 = r_conv(r_conv(x, W, H, *net['e1a'], Q), W, H, *net['e1b'], Q)
    p1, W2, H2 = r_pool(e1, W, H)
    e2 = r_conv(r_conv(p1, W2, H2, *net['e2a'], Q), W2, H2, *net['e2b'], Q)
    p2, W3, H3 = r_pool(e2, W2, H2)
    bn = r_conv(r_conv(p2, W3, H3, *net['bna'], Q), W3, H3, *net['bnb'], Q)
    d2 = r_conv(r_conv(r_up(bn, W3, W2, H2) + e2, W2, H2, *net['d2a'], Q), W2, H2, *net['d2b'], Q)
    d1 = r_conv(r_conv(r_up(d2, W2, W, H) + e1, W, H, *net['d1a'], Q), W, H, *net['d1b'], Q)
    return r_conv(d1, W, H, *net['out'], 0, relu=False, k1=True)

def levels(W, H):
    W2, H2 = (W + 1) // 2, (H + 1) // 2
    return G(W, H, 16), G(W2, H2, 32), G((W2 + 1) // 2, (H2 + 1) // 2, 32)

random.seed(3)
# correctness on a small odd-sized map
W, H = 11, 9
net = build(4, 4, 6, 8)
x = [[random.randint(0, AMAX) for _ in range(W * H)] for _ in range(4)]
L1, L2, L3 = levels(W, H)
got = L1.unpack(forward(net, L1, L2, L3, [L1.pack(c) for c in x])[0])
ref = r_forward(net, x, W, H)[0]
got = [v - L1.HALF for v in got]
print("U-Net matches reference:", got == ref)

# timing on a 30x30 map
for cin, c1, c2, c3 in ((6, 8, 16, 32), (6, 8, 16, 16), (6, 12, 24, 32), (6, 16, 32, 64)):
    net = build(cin, c1, c2, c3)
    L1, L2, L3 = levels(30, 30)
    xs = [L1.pack([random.randint(0, AMAX) for _ in range(900)]) for _ in range(cin)]
    forward(net, L1, L2, L3, xs)
    T = {}; n = 5; tot = time.perf_counter()
    for _ in range(n): forward(net, L1, L2, L3, xs, T)
    tot = (time.perf_counter() - tot) / n * 1000
    print(f"\nchannels {cin}->{c1}->{c2}->{c3}: total {tot:.1f} ms per forward pass")
    for k, v in T.items(): print(f"   {k:18s} {v:6.2f} ms")
