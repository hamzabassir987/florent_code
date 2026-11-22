exec(open('unet.py').read().split('# ------------------------------------------------------------------ network')[0])
import random, time

def finish(g, p, n, q, relu=True):
    r = (p + g.OFF - n) & g.FULL
    if relu:
        sign = (r >> (g.B - 1)) & g.ONES
        r &= sign * (g.HALF - 1)
    if q:
        r = (r >> q) & (g.ONES * ((1 << (g.B - 1 - q)) - 1))
    over = ge_mask(g, r, g.ONES * (AMAX + 1))
    r = (r & ~(over * ((1 << g.B) - 1))) | (over * AMAX)
    return r & g.INNER

def pw(g, xs, Wp, Wn, q):
    """1x1 conv: Wp/Wn[co][ci] scalars."""
    out = []
    for co in range(len(Wp)):
        p = n = 0
        a, b = Wp[co], Wn[co]
        for ci, x in enumerate(xs):
            if a[ci]: p += a[ci] * x
            if b[ci]: n += b[ci] * x
        out.append(finish(g, p, n, q))
    return out

def dw(g, xs, Wp, Wn, q, dil=1, res=None):
    """depthwise 3x3 (optionally dilated), optional residual add."""
    S, B = g.S, g.B
    out = []
    for c, x in enumerate(xs):
        p = n = 0
        a, b = Wp[c], Wn[c]
        for k in range(9):
            s = ((k // 3 - 1) * S + (k % 3 - 1)) * B * dil
            v = x >> s if s > 0 else x << -s
            if a[k]: p += a[k] * v
            if b[k]: n += b[k] * v
        r = finish(g, p, n, q)
        if res is not None:   # residual add, then clamp
            r = r + res[c]
            over = ge_mask(g, r, g.ONES * (AMAX + 1))
            r = (r & ~(over * ((1 << B) - 1))) | (over * AMAX)
        out.append(r)
    return out

def rw(shape):
    import itertools
    Wp, Wn = [], []
    def rec(dims):
        if len(dims) == 1:
            p, n = [], []
            for _ in range(dims[0]):
                w = random.randint(-7, 7)
                p.append(max(w, 0)); n.append(max(-w, 0))
            return p, n
        P, N = [], []
        for _ in range(dims[0]):
            p, n = rec(dims[1:]); P.append(p); N.append(n)
        return P, N
    return rec(shape)

def best(f, n=40):
    f(); b = 1e9
    for _ in range(n):
        t = time.perf_counter(); f(); b = min(b, time.perf_counter() - t)
    return b * 1000

random.seed(10)
W, C, L = 15, 32, 3
g = G(W, W, 16)   # 32 inputs * 7 * 15 * 9 = 30240 < 32768, fits 16-bit
xs = [g.pack([random.randint(0, 15) for _ in range(W * W)]) for _ in range(C)]

full = [rand_w(C, C) for _ in range(L)] if 'rand_w' in globals() else None
def rand3(cin, cout):
    Wp = [[[0]*9 for _ in range(cin)] for _ in range(cout)]; Wn = [[[0]*9 for _ in range(cin)] for _ in range(cout)]
    for co in range(cout):
        for ci in range(cin):
            for k in range(9):
                w = random.randint(-7, 7); (Wp if w > 0 else Wn)[co][ci][k] = abs(w)
    return Wp, Wn
full = [rand3(C, C) for _ in range(L)]
pws = [rw((C, C)) for _ in range(L)]
dws = [rw((C, 9)) for _ in range(L)]

def normal():
    h = xs
    for w in full: h = conv(g, h, *w, 5)
def sepnet(dils=(1, 1, 1), residual=False):
    h = xs
    for i in range(L):
        t = pw(g, h, *pws[i], 4)
        h = dw(g, t, *dws[i], 4, dils[i], h if residual else None)

print(f"15x15, {C}->{C}->{C}->{C} ({L} layers), best of 40:")
print(f"  normal 3x3 convs:                    {best(normal):6.2f} ms")
print(f"  1x1 + depthwise 3x3:                 {best(sepnet):6.2f} ms")
print(f"  1x1 + depthwise, dilations 1,2,4:    {best(lambda: sepnet((1,2,4))):6.2f} ms")
print(f"  1x1 + depthwise + residual:          {best(lambda: sepnet(residual=True)):6.2f} ms")
one_pw = best(lambda: pw(g, xs, *pws[0], 4)); one_dw = best(lambda: dw(g, xs, *dws[0], 4))
print(f"  per block: 1x1 {one_pw:.2f} ms + depthwise {one_dw:.2f} ms = {one_pw+one_dw:.2f} ms")

# classic ResNet: 3 residual blocks, each = 2 x (1x1 + depthwise), skip around the pair
pws6 = [rw((C, C)) for _ in range(6)]
dws6 = [rw((C, 9)) for _ in range(6)]
def resnet3():
    h = xs
    for b in range(3):
        t = dw(g, pw(g, h, *pws6[2*b], 4), *dws6[2*b], 4)
        h = dw(g, pw(g, t, *pws6[2*b+1], 4), *dws6[2*b+1], 4, 1, h)
print(f"  ResNet, 3 blocks x 2 layers (1x1+depthwise), 32ch: {best(resnet3):6.2f} ms")
