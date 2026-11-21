"""SWAR conv kernels (copied from benchmark)."""

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

