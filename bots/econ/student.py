"""Int4 student policy, deployed side: pure-Python SWAR inference that matches
rl/train/student.hpp bit for bit (rl/tests/student_exact.py checks it).

Each 17x17 channel is one big integer: 16-bit slots on a 19x19 grid with a
zero border, so one Python int operation updates every cell. Convolutions
accumulate positive and negative weight parts separately and finish with
ReLU / shift / clamp per slot. Integer logits; divide by 16 for softmax.
Only the standard library is used.
"""
import json
import sys
from array import array

PATCH, HALF = 17, 8
ACTOR_CH, SCALARS = 60, 50   # 60th channel: planned conveyor direction (copied as is, 1..8)
COSTS = 40
BITS = 16
# 4 residual blocks x 2 (1x1 -> depthwise 3x3) layers = 8 depthwise convs:
# receptive field 17x17, exactly the patch.
LAYERS = 8
RADIUS = 8
SLOT = (1 << BITS) - 1
HALF_SLOT = 1 << (BITS - 1)


class Geo:
    """Slot layout for a (2r+1)^2 window with a zero border.

    The heads read only the centre tile, so block b (2 depthwise layers) only
    needs its input within radius 8 - 2b: windows 17, 13, 9, 5. Edge cells of a
    window go wrong inside the block, but those are exactly the cells the next
    (smaller) window drops, so the centre output stays exact."""
    def __init__(self, r):
        self.r = r
        self.win = 2 * r + 1
        self.s = S = self.win + 2
        n = S * S
        self.ones = sum(1 << (i * BITS) for i in range(n))
        inner = 0
        for y in range(1, self.win + 1):
            inner |= ((1 << (self.win * BITS)) - 1) << ((y * S + 1) * BITS)
        self.inner = inner
        self.off = self.ones * HALF_SLOT
        self.full = (1 << (n * BITS)) - 1
        self.sixteen = self.ones * 16
        self.centre = (r + 1) * S + (r + 1)
        self.dw = [((k // 3 - 1) * S + (k % 3 - 1)) * BITS for k in range(9)]

    def crop(self, v, g):
        """Central window of geometry g from big int v (laid out in self)."""
        d, out, rowmask = self.r - g.r, 0, (1 << (g.win * BITS)) - 1
        for y in range(g.win):
            row = (v >> (((y + d + 1) * self.s + d + 1) * BITS)) & rowmask
            if row:
                out |= row << (((y + 1) * g.s + 1) * BITS)
        return out


GEOS = [Geo(RADIUS - 2 * b) for b in range(LAYERS // 2)]
G0 = GEOS[0]
WIN, S = G0.win, G0.s
# encoder channel layout (bots/econ/encoder.py)
C_STALE, C_VISIT, C_ATTR_STALE, C_HP, C_RAY = 29, 30, 31, 42, 48


def quantize(raw):
    """Raw encoder ints (spatial 59x17x17 channel-major, then 40 scalars) -> 0..15."""
    P2 = PATCH * PATCH
    q = [0] * len(raw)
    for c in range(ACTOR_CH):
        base = c * P2
        for i in range(P2):
            v = raw[base + i]
            if not v:
                continue
            if c in (C_STALE, C_VISIT, C_ATTR_STALE, C_ATTR_STALE + 1):
                o = v >> 4
            elif c in (C_HP, C_HP + 1):
                o = v * 15 // 100
            elif c >= C_RAY:
                o = (v + 1) >> 1 if (c - C_RAY) <= 1 else v
            else:
                o = 15
            q[base + i] = 0 if o < 0 else (15 if o > 15 else o)
    s0 = ACTOR_CH * P2
    for i in range(SCALARS):
        v = raw[s0 + i]
        o = 0
        if i <= 5 or 36 <= i <= 39:
            o = 15 if v else 0
        elif i == 6:
            o = v * 15 // 100
        elif i in (7, 8):
            o = v * 3
        elif i == 9:
            o = v // 67
        elif i in (10, 11):
            o = v // 100
        elif i == 12:
            o = v // 50
        elif i == 14:
            o = v // 4
        elif i == 15:
            o = v
        elif i == 13:
            o = (v - 100) // 25 if v >= 100 else -((100 - v) // 25)   # truncates toward 0, like C++
        elif i >= COSTS:
            o = v // 4
        q[s0 + i] = 0 if o < 0 else (15 if o > 15 else o)
    return q


def _pack(q, c):
    """Channel c (the full 17x17 patch) into a bordered big int (C-level copies)."""
    base = c * PATCH * PATCH
    seg = q[base:base + PATCH * PATCH]
    if not any(seg):
        return 0
    buf = [0] * (S * S)
    for y in range(PATCH):
        o = (y + 1) * S + 1
        buf[o:o + PATCH] = seg[y * PATCH:(y + 1) * PATCH]
    return int.from_bytes(array("H", buf).tobytes(), sys.byteorder)


def _finish(g, p, n, shift):
    """clamp(floor((p - n) / 2^shift), 0, 15) per slot, border cleared."""
    ones = g.ones
    r = (p + g.off - n) & g.full
    sign = (r >> (BITS - 1)) & ones                     # 1 where p - n >= 0
    r &= sign * (HALF_SLOT - 1)                          # ReLU, drop the offset
    if shift:
        r = (r >> shift) & (ones * ((1 << (BITS - 1 - shift)) - 1))
    over = (((r + g.off - g.sixteen) >> (BITS - 1)) & ones)  # 1 where r >= 16
    r = (r & ~(over * SLOT)) | (over * 15)
    return r & g.inner


def _bias(g, p, n, b):
    if b > 0:
        p += b * g.ones
    elif b < 0:
        n += (-b) * g.ones
    return p, n


class StudentPolicy:
    def __init__(self, path):
        data = json.loads(open(path).read())
        assert data["format"] == "florent-student-v1"
        self.logit_shift = data["logit_shift"]
        self.hidden = data["hidden"]
        self.pol = {}
        C = data["channels"]
        for pd in data["policies"]:
            p = pd["policy"]
            P = {"shifts": pd["shifts"], "n": pd["nlogits"]}
            sw = pd["stem_w"]                      # [C][ACTOR_CH][1][1]
            P["stem"] = [[sw[co * ACTOR_CH + ci] for ci in range(ACTOR_CH)] for co in range(C)]
            P["stem_b"] = pd["stem_b"]
            for b in range(LAYERS):
                pw = pd[f"pw_w{b}"]
                P[f"pw{b}"] = [[pw[co * C + ci] for ci in range(C)] for co in range(C)]
                P[f"pwb{b}"] = pd[f"pw_b{b}"]
                dw = pd[f"dw_w{b}"]
                P[f"dw{b}"] = [[dw[c * 9 + k] for k in range(9)] for c in range(C)]
                P[f"dwb{b}"] = pd[f"dw_b{b}"]
            F = C + SCALARS
            fw = pd["fc_w"]
            P["fc"] = [[fw[j * F + i] for i in range(F)] for j in range(self.hidden)]
            P["fc_b"] = pd["fc_b"]
            ow = pd["out_w"]
            P["out"] = [[ow[j * self.hidden + i] for i in range(self.hidden)] for j in range(P["n"])]
            P["out_b"] = pd["out_b"]
            P["fcT"] = [list(c) for c in zip(*P["fc"])]
            P["stem_g"] = self._groups(P["stem"])
            for b in range(LAYERS):
                P[f"pwg{b}"] = self._groups(P[f"pw{b}"])
            self.pol[p] = P

    # ---- SWAR trunk ----------------------------------------------------------
    @staticmethod
    def _pointwise(g, xs, W, B, shift):
        # W[co] = (pos, neg): for each sign, input indices grouped by |w| = 7..1.
        # sum_v v * S_v is accumulated as running sums: adds only, no multiplies.
        out = []
        for co, (pos, neg) in enumerate(W):
            pn = []
            for groups in (pos, neg):
                t = acc = 0
                for idx in groups:
                    for ci in idx:
                        t += xs[ci]
                    acc += t
                pn.append(acc)
            p, n = _bias(g, pn[0], pn[1], B[co])
            out.append(_finish(g, p, n, shift))
        return out

    @staticmethod
    def _depthwise(g, ts, hs, W, B, shift):
        out = []
        for c, x in enumerate(ts):
            p = n = 0
            for k, w in enumerate(W[c]):
                if not w:
                    continue
                s = g.dw[k]
                v = x >> s if s > 0 else x << -s
                if w > 0:
                    p += w * v
                else:
                    n -= w * v
            p, n = _bias(g, p, n, B[c])
            if hs is not None:
                p += hs[c] << shift                 # residual at accumulator scale
            out.append(_finish(g, p, n, shift))
        return out

    @staticmethod
    def _groups(W):
        """Dense rows -> (pos, neg) lists of 7 index groups for |w| = 7, 6, ..., 1."""
        res = []
        for row in W:
            pos = [[ci for ci, w in enumerate(row) if w == v] for v in range(7, 0, -1)]
            neg = [[ci for ci, w in enumerate(row) if w == -v] for v in range(7, 0, -1)]
            res.append((pos, neg))
        return res

    def features(self, policy, q):
        P = self.pol[policy]
        sh = P["shifts"]
        xs = [_pack(q, c) for c in range(ACTOR_CH)]
        g = G0
        h = self._pointwise(g, xs, P["stem_g"], P["stem_b"], sh[0])
        for blk in range(LAYERS // 2):
            if blk:
                ng = GEOS[blk]
                h = [g.crop(v, ng) for v in h]
                g = ng
            hb = h
            for b in (2 * blk, 2 * blk + 1):
                t = self._pointwise(g, h, P[f"pwg{b}"], P[f"pwb{b}"], sh[1 + 2 * b])
                h = self._depthwise(g, t, hb if b % 2 else None, P[f"dw{b}"], P[f"dwb{b}"], sh[2 + 2 * b])
        centre = [(v >> (g.centre * BITS)) & SLOT for v in h]
        s0 = ACTOR_CH * PATCH * PATCH
        f = centre + list(q[s0:s0 + SCALARS])
        acc = list(P["fc_b"])
        fcT = P["fcT"]
        for i, fi in enumerate(f):
            if fi:
                col = fcT[i]
                acc = [a + fi * w for a, w in zip(acc, col)]
        s9 = sh[-1]
        return [0 if v < 0 else (15 if v > 15 else v) for v in (a >> s9 for a in acc)]

    def logits_int(self, policy, q, indices=None):
        """Integer logits (all, or only `indices`)."""
        P = self.pol[policy]
        hid = self.features(policy, q)
        rows = range(P["n"]) if indices is None else indices
        out = []
        for j in rows:
            acc = P["out_b"][j]
            for i, w in enumerate(P["out"][j]):
                if w and hid[i]:
                    acc += w * hid[i]
            out.append(acc)
        return out

    def logits(self, policy, raw):
        """Float logits for the head decoder (integer / 16, exactly representable)."""
        d = float(1 << self.logit_shift)
        return [v / d for v in self.logits_int(policy, quantize(raw))]


def reference_logits(model, policy, q):
    """Plain-integer reference (slow, no SWAR) used to cross-check the packing."""
    P = model.pol[policy]
    sh = P["shifts"]
    C = len(P["stem"])
    P2 = PATCH * PATCH

    def act(v, s):
        v >>= s
        return 0 if v < 0 else (15 if v > 15 else v)

    x = [[q[c * P2 + i] for i in range(P2)] for c in range(ACTOR_CH)]
    h = [[act(P["stem_b"][co] + sum(P["stem"][co][ci] * x[ci][i] for ci in range(ACTOR_CH)), sh[0]) for i in range(P2)] for co in range(C)]
    hb = h
    for b in range(LAYERS):
        t = [[act(P[f"pwb{b}"][co] + sum(P[f"pw{b}"][co][ci] * h[ci][i] for ci in range(C)), sh[1 + 2 * b]) for i in range(P2)] for co in range(C)]
        nh = []
        for c in range(C):
            row = []
            for y in range(PATCH):
                for xx in range(PATCH):
                    acc = P[f"dwb{b}"][c]
                    for k in range(9):
                        yy, x2 = y + k // 3 - 1, xx + k % 3 - 1
                        if 0 <= yy < PATCH and 0 <= x2 < PATCH:
                            acc += P[f"dw{b}"][c][k] * t[c][yy * PATCH + x2]
                    if b % 2:
                        acc += hb[c][y * PATCH + xx] << sh[2 + 2 * b]
                    row.append(act(acc, sh[2 + 2 * b]))
            nh.append(row)
        h = nh
        if b % 2:
            hb = h
    centre = [h[c][HALF * PATCH + HALF] for c in range(C)]
    s0 = ACTOR_CH * P2
    f = centre + list(q[s0:s0 + SCALARS])
    hid = [act(P["fc_b"][j] + sum(w * f[i] for i, w in enumerate(P["fc"][j])), sh[-1]) for j in range(len(P["fc"]))]
    return [P["out_b"][j] + sum(w * hid[i] for i, w in enumerate(P["out"][j])) for j in range(P["n"])]
