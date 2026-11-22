"""3x3 conv layer as ONE big-int multiply per output channel (Kronecker substitution)."""
import time, random
from array import array

def setup(W, H, C, B):
    S = W + 2; n = S * (H + 2)
    L = n + 2 * S + 4                   # slots per channel block (room for the product tail)
    return dict(W=W, H=H, S=S, n=n, L=L, B=B, C=C)

def pack_channels(g, chans):
    """all input channels into one int: channel ci at block ci."""
    W, H, S, L, B = g['W'], g['H'], g['S'], g['L'], g['B']
    code = 'I' if B == 32 else 'Q'
    a = array('Q', [0] * (L * len(chans))) if B == 64 else array('I' if B == 32 else 'H', [0] * (L * len(chans)))
    for ci, v in enumerate(chans):
        for y in range(H):
            base = ci * L + (y + 1) * S + 1
            a[base:base + W] = array(a.typecode, v[y * W:(y + 1) * W])
    return int.from_bytes(a.tobytes(), 'little')

def kernels(g, Wt, bias):
    """Wt[co][ci][k] in -7..7 -> nonneg (w+bias); kernel block for channel ci at (C-1-ci), flipped."""
    S, L, B, C = g['S'], g['L'], g['B'], g['C']
    Ks = []
    for co in range(len(Wt)):
        K = 0
        for ci in range(C):
            for k in range(9):
                dy, dx = k // 3 - 1, k % 3 - 1
                pos = (C - 1 - ci) * L + (1 - dy) * S + (1 - dx)
                K |= (Wt[co][ci][k] + bias) << (pos * B)
        Ks.append(K)
    # box-sum kernel (all ones over every channel) to remove the bias afterwards
    ONESK = 0
    for ci in range(C):
        for k in range(9):
            dy, dx = k // 3 - 1, k % 3 - 1
            ONESK |= 1 << (((C - 1 - ci) * L + (1 - dy) * S + (1 - dx)) * B)
    return Ks, ONESK

def conv_kron(g, X, Ks, ONESK, bias, q, amax=15):
    S, L, B, C, n, W, H = g['S'], g['L'], g['B'], g['C'], g['n'], g['W'], g['H']
    start = ((C - 1) * L + S + 1) * B   # output slot i lives at block C-1, offset +S+1
    span = (1 << (n * B)) - 1
    box = ((X * ONESK) >> start) & span
    code = {16: 'H', 32: 'I', 64: 'Q'}[B]
    outs = []
    boxa = array(code, box.to_bytes(n * B // 8, 'little'))
    for K in Ks:
        y = ((X * K) >> start) & span
        outs.append((y, box))
    return outs

def finish_list(g, outs, bias, q, amax=15):
    """decode to python lists (only for checking): v = y - bias*box, relu, >>q, clamp."""
    S, B, n, W, H = g['S'], g['B'], g['n'], g['W'], g['H']
    code = {16: 'H', 32: 'I', 64: 'Q'}[B]
    res = []
    for y, box in outs:
        ya = array(code, y.to_bytes(n * B // 8, 'little')); ba = array(code, box.to_bytes(n * B // 8, 'little'))
        o = []
        for yy in range(H):
            for xx in range(W):
                i = (yy + 1) * S + xx + 1
                v = ya[i] - bias * ba[i]
                o.append(min(amax, max(0, v) >> q))
        res.append(o)
    return res

def naive(vals, W, H, Wt, q, amax=15):
    out = []
    for co in range(len(Wt)):
        o = []
        for y in range(H):
            for x in range(W):
                s = 0
                for ci in range(len(vals)):
                    for k in range(9):
                        yy, xx = y + k // 3 - 1, x + k % 3 - 1
                        if 0 <= yy < H and 0 <= xx < W:
                            s += Wt[co][ci][k] * vals[ci][yy * W + xx]
                o.append(min(amax, max(0, s) >> q))
        out.append(o)
    return out

random.seed(5)
# correctness
W, H, C, CO = 9, 7, 5, 4
g = setup(W, H, C, 32)
vals = [[random.randint(0, 15) for _ in range(W * H)] for _ in range(C)]
Wt = [[[random.randint(-7, 7) for _ in range(9)] for _ in range(C)] for _ in range(CO)]
X = pack_channels(g, vals); Ks, ONESK = kernels(g, Wt, 7)
print("kron conv matches reference:", finish_list(g, conv_kron(g, X, Ks, ONESK, 7, 5), 7, 5) == naive(vals, W, H, Wt, 5))

# timing: just the multiplies (the conv), same layer sizes as before
for W, c in ((30, 8), (30, 16), (15, 16), (15, 32)):
    B = 32
    g = setup(W, W, c, B)
    vals = [[random.randint(0, 15) for _ in range(W * W)] for _ in range(c)]
    Wt = [[[random.randint(-7, 7) for _ in range(9)] for _ in range(c)] for _ in range(c)]
    X = pack_channels(g, vals); Ks, ONESK = kernels(g, Wt, 7)
    conv_kron(g, X, Ks, ONESK, 7, 5)
    t = time.perf_counter()
    for _ in range(5): conv_kron(g, X, Ks, ONESK, 7, 5)
    dt = (time.perf_counter() - t) / 5 * 1000
    print(f"{W}x{W} {c:2d}->{c:2d}: {dt:6.2f} ms  ({W*W*9*c*c/dt/1e3:.0f} M MAC/s)")
