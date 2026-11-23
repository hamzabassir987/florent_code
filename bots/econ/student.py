"""Int4 student policy, deployed side: pure-Python SWAR inference that matches
rl/train/student.hpp bit for bit (rl/tests/student_exact.py checks it).

Each 17x17 channel is one big integer: 16-bit slots on a 19x19 grid with a
zero border, so one Python int operation updates every cell. Convolutions
accumulate positive and negative weight parts separately and finish with
ReLU / shift / clamp per slot. Integer logits; divide by 16 for softmax.
Only the standard library is used.
"""
import json

PATCH, HALF = 17, 8
ACTOR_CH, SCALARS = 59, 40
BITS = 16
S = PATCH + 2
NSLOT = S * S
SLOT = (1 << BITS) - 1
HALF_SLOT = 1 << (BITS - 1)
ONES = sum(1 << (i * BITS) for i in range(NSLOT))
INNER = 0
for _y in range(1, PATCH + 1):
    INNER |= ((1 << (PATCH * BITS)) - 1) << ((_y * S + 1) * BITS)
OFF = ONES * HALF_SLOT
FULL = (1 << (NSLOT * BITS)) - 1
FIFTEEN = ONES * 15
SIXTEEN = ONES * 16
CENTRE_SLOT = (HALF + 1) * S + (HALF + 1)
# encoder channel layout (bots/econ/encoder.py)
C_STALE, C_VISIT, C_ATTR_STALE, C_HP, C_RAY = 29, 30, 31, 42, 48
DW_SHIFTS = [((k // 3 - 1) * S + (k % 3 - 1)) * BITS for k in range(9)]


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
        q[s0 + i] = 0 if o < 0 else (15 if o > 15 else o)
    return q


def _pack(q, c):
    acc, base = 0, c * PATCH * PATCH
    for y in range(PATCH):
        row = base + y * PATCH
        slot = (y + 1) * S + 1
        for x in range(PATCH):
            v = q[row + x]
            if v:
                acc |= v << ((slot + x) * BITS)
    return acc


def _finish(p, n, shift):
    """clamp(floor((p - n) / 2^shift), 0, 15) per slot, border cleared."""
    r = (p + OFF - n) & FULL
    sign = (r >> (BITS - 1)) & ONES                     # 1 where p - n >= 0
    r &= sign * (HALF_SLOT - 1)                          # ReLU, drop the offset
    if shift:
        r = (r >> shift) & (ONES * ((1 << (BITS - 1 - shift)) - 1))
    over = (((r + OFF - SIXTEEN) >> (BITS - 1)) & ONES)  # 1 where r >= 16
    r = (r & ~(over * SLOT)) | (over * 15)
    return r & INNER


def _bias(p, n, b):
    if b > 0:
        p += b * ONES
    elif b < 0:
        n += (-b) * ONES
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
            sw = pd["stem_w"]                      # [C][59][1][1]
            P["stem"] = [[sw[co * ACTOR_CH + ci] for ci in range(ACTOR_CH)] for co in range(C)]
            P["stem_b"] = pd["stem_b"]
            for b in range(4):
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
            self.pol[p] = P

    # ---- SWAR trunk ----------------------------------------------------------
    @staticmethod
    def _pointwise(xs, W, B, shift):
        out = []
        for co, row in enumerate(W):
            p = n = 0
            for ci, w in enumerate(row):
                if w > 0:
                    p += w * xs[ci]
                elif w < 0:
                    n -= w * xs[ci]
            p, n = _bias(p, n, B[co])
            out.append(_finish(p, n, shift))
        return out

    @staticmethod
    def _depthwise_res(ts, hs, W, B, shift):
        out = []
        for c, x in enumerate(ts):
            p = n = 0
            for k, w in enumerate(W[c]):
                if not w:
                    continue
                s = DW_SHIFTS[k]
                v = x >> s if s > 0 else x << -s
                if w > 0:
                    p += w * v
                else:
                    n -= w * v
            p, n = _bias(p, n, B[c])
            p += hs[c] << shift                     # residual at accumulator scale
            out.append(_finish(p, n, shift))
        return out

    def features(self, policy, q):
        P = self.pol[policy]
        sh = P["shifts"]
        xs = [_pack(q, c) for c in range(ACTOR_CH)]
        h = self._pointwise(xs, P["stem"], P["stem_b"], sh[0])
        for b in range(4):
            t = self._pointwise(h, P[f"pw{b}"], P[f"pwb{b}"], sh[1 + 2 * b])
            h = self._depthwise_res(t, h, P[f"dw{b}"], P[f"dwb{b}"], sh[2 + 2 * b])
        centre = [(v >> (CENTRE_SLOT * BITS)) & SLOT for v in h]
        s0 = ACTOR_CH * PATCH * PATCH
        f = centre + list(q[s0:s0 + SCALARS])
        hid = []
        for j, row in enumerate(P["fc"]):
            acc = P["fc_b"][j]
            for i, w in enumerate(row):
                if w and f[i]:
                    acc += w * f[i]
            v = acc >> sh[9]
            hid.append(0 if v < 0 else (15 if v > 15 else v))
        return hid

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
    for b in range(4):
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
                    acc += h[c][y * PATCH + xx] << sh[2 + 2 * b]
                    row.append(act(acc, sh[2 + 2 * b]))
            nh.append(row)
        h = nh
    centre = [h[c][HALF * PATCH + HALF] for c in range(C)]
    s0 = ACTOR_CH * P2
    f = centre + list(q[s0:s0 + SCALARS])
    hid = [act(P["fc_b"][j] + sum(w * f[i] for i, w in enumerate(P["fc"][j])), sh[9]) for j in range(len(P["fc"]))]
    return [P["out_b"][j] + sum(w * hid[i] for i, w in enumerate(P["out"][j])) for j in range(P["n"])]
