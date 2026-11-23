src = open('sep.py').read().replace("W, C, L = 15, 32, 3", "W, C, L = 17, 32, 3")
exec(src.split("print(f\"{W}x{W}")[0])
# Selected deployment architecture: four residual blocks, each consisting of
# a 1x1 pointwise projection followed by a depthwise 3x3 layer. The cropped
# implementation computes only the radius needed for the centre output.
GRIDS = {}
def grid(w):
    if w not in GRIDS: GRIDS[w] = G(w, w, 16)
    return GRIDS[w]

def crop(gs, xs, w):
    """crop packed channels from grid gs to the central w x w grid."""
    if gs.W == w: return gs, xs
    gd = grid(w); off = (gs.W - w) // 2
    out = []
    for x in xs:
        a = gs.to_arr(x); b = array(gd.code, [0] * gd.n)
        for y in range(w):
            s = (y + 1 + off) * gs.S + 1 + off
            b[(y + 1) * gd.S + 1:(y + 1) * gd.S + 1 + w] = a[s:s + w]
        out.append(gd.from_arr(b))
    return gd, out

random.seed(11)
pws8 = [rw((C, C)) for _ in range(8)]
dws8 = [rw((C, 9)) for _ in range(8)]
g17 = grid(17)
xs17 = [g17.pack([random.randint(0, 15) for _ in range(289)]) for _ in range(C)]

def layer(g, h, i, res=None):
    return dw(g, pw(g, h, *pws8[i], 4), *dws8[i], 4, 1, res)

def full():
    g, h = g17, xs17
    for b in range(4):
        t = layer(g, h, 2 * b)
        h = layer(g, t, 2 * b + 1, h)
    return g, h

def cropped():
    g, h = g17, xs17
    for b in range(4):
        i = 2 * b
        g, h = crop(g, h, 2 * (8 - i) + 1)          # layer i+1 needs radius 8-i
        hg = g
        t = layer(g, h, i)
        g, t = crop(g, t, 2 * (7 - i) + 1)
        _, hs = crop(hg, h, g.W)                   # skip cropped onto the same grid
        h = layer(g, t, i + 1, hs)
    return g, h

def centre(g, h):
    c = g.W // 2
    return [g.unpack(x)[c * g.W + c] for x in h]

print("centre outputs identical:", centre(*full()) == centre(*cropped()))
print(f"17x17, 4 ResNet blocks, 32ch, full grid every layer: {best(full):.2f} ms")
print(f"17x17, 4 ResNet blocks, 32ch, cropped to what's needed: {best(cropped):.2f} ms")
