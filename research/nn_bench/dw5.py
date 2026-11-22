exec(open('crop.py').read().split('print("centre outputs')[0])

def dw5(g, xs, Wp, Wn, q, res=None):
    S, B = g.S, g.B
    out = []
    for c, x in enumerate(xs):
        p = n = 0
        a, b = Wp[c], Wn[c]
        for k in range(25):
            s = ((k // 5 - 2) * S + (k % 5 - 2)) * B
            v = x >> s if s > 0 else x << -s
            if a[k]: p += a[k] * v
            if b[k]: n += b[k] * v
        r = finish(g, p, n, q)
        if res is not None:
            r = r + res[c]
            over = ge_mask(g, r, g.ONES * (AMAX + 1))
            r = (r & ~(over * ((1 << B) - 1))) | (over * AMAX)
        out.append(r)
    return out

# 5x5 needs a 2-slot border: use grids with 1 extra tile of zero padding each side
def grid5(w):
    return grid(w + 2)
def pad(g, xs, w):        # place a w x w map inside a (w+2) grid, centred
    return crop(g, xs, w + 2) if g.W > w + 2 else (g, xs)

random.seed(12)
p5 = [rw((C, C)) for _ in range(4)]
d5 = [rw((C, 25)) for _ in range(4)]
g19 = grid(19)
# 17x17 data centred in a 19x19 grid (extra zero ring so 5x5 shifts stay in bounds)
base = [[random.randint(0, 15) for _ in range(289)] for _ in range(C)]
def embed(vals):
    out = [0] * 361
    for y in range(17):
        for x in range(17):
            out[(y + 1) * 19 + x + 1] = vals[y * 17 + x]
    return out
x19 = [g19.pack(embed(v)) for v in base]

def layer5(g, h, i, res=None):
    return dw5(g, pw(g, h, *p5[i], 4), *d5[i], 4, res)

def net5_full():
    g, h = g19, x19
    for b in range(2):
        t = layer5(g, h, 2 * b)
        h = layer5(g, t, 2 * b + 1, h)
    return g, h

def net5_crop():
    # layer i (0..3) output needed at radius 2*(3-i); input radius 2*(4-i); +1 zero ring
    g, h = g19, x19
    for b in range(2):
        i = 2 * b
        g, h = crop(g, h, 2 * (2 * (4 - i)) + 1 + 2)
        hg = g
        t = layer5(g, h, i)
        g, t = crop(g, t, 2 * (2 * (3 - i)) + 1 + 2)
        _, hs = crop(hg, h, g.W)
        h = layer5(g, t, i + 1, hs)
    return g, h

def centre(g, h):
    c = g.W // 2
    return [g.unpack(x)[c * g.W + c] for x in h]

print("5x5 cropped == full at centre:", centre(*net5_full()) == centre(*net5_crop()))
print(f"17x17, 2 ResNet blocks x 2 layers (1x1 + depthwise 5x5), 32ch, full:    {best(net5_full):.2f} ms")
print(f"17x17, 2 ResNet blocks x 2 layers (1x1 + depthwise 5x5), 32ch, cropped: {best(net5_crop):.2f} ms")
print(f"(reference) 4 ResNet blocks x 2 layers of 3x3, cropped:                  {best(cropped):.2f} ms")
