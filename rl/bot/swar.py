"""Pure-Python SWAR spatial policy runtime for Florent.

Activations are packed as 4-bit values in one big integer per channel, so a
whole 17x17 patch is a handful of machine-word-parallel operations instead of
289 Python-level scalar loops. Weights are small signed integers (-7..7),
stored as separate positive/negative accumulators so no per-slot borrow can
cross a channel boundary.

The network is the deployment architecture from ``docs/rl-layout.md``:

    patch 32ch --[1x1, depthwise 3x3, residual] x 4 blocks--> centre 32ch
    centre 32ch + 40 actor scalars --dense--> logits over candidate slots

Only the current unit's head is evaluated, and only over candidates that are
actually legal, which keeps the per-turn cost well inside the 10 ms budget.
"""

# Grid geometry. A 1-slot zero ring keeps every shifted neighbour in bounds.
PATCH = 17
SCALARS = 40
CHANNELS = 32
BITS = 16
S, N = PATCH + 2, (PATCH + 2) * (PATCH + 2)
SLOT_MASK = (1 << BITS) - 1
HALF = 1 << (BITS - 1)
AMAX = 15
_FULL = (1 << (N * BITS)) - 1


def _ones():
    v = 0
    for i in range(N):
        v |= 1 << (i * BITS)
    return v


ONES = _ones()
INNER = 0
for _y in range(1, PATCH + 1):
    _row = ((1 << (PATCH * BITS)) - 1) << ((_y * S + 1) * BITS)
    INNER |= _row
OFF = ONES * HALF
LOW = ONES * (HALF - 1)


def _shift(x, slots):
    """Shift a packed grid by a signed slot offset."""
    return x >> (slots * BITS) if slots > 0 else x << (-slots * BITS)


def _ge_mask(a, b):
    """Per-slot 1 where a >= b, for values below HALF."""
    return ((a + OFF - b) >> (BITS - 1)) & ONES


def pack(values):
    """Pack a channel-major list of 17x17 values into a bordered big int."""
    acc = 0
    for y in range(PATCH):
        base = (y + 1) * S + 1
        for x in range(PATCH):
            v = values[y * PATCH + x] & SLOT_MASK
            if v:
                acc |= v << ((base + x) * BITS)
    return acc


def unpack_centre(vals):
    """Read the centre tile of a packed grid's channel list."""
    c = PATCH // 2
    slot = (c + 1) * S + (c + 1)
    return [(v >> (slot * BITS)) & SLOT_MASK for v in vals]


def _finish(p, n, q):
    """(p - n) with offset arithmetic, ReLU, >>q, clamp to AMAX."""
    r = (p + OFF - n) & _FULL
    sign = (r >> (BITS - 1)) & ONES
    r &= sign * (HALF - 1)
    if q:
        r = (r >> q) & (ONES * ((1 << (BITS - 1 - q)) - 1))
    over = _ge_mask(r, ONES * (AMAX + 1))
    r = (r & ~(over * SLOT_MASK)) | (over * AMAX)
    return r & INNER


def pw_conv(xs, pos, neg, q):
    """1x1 convolution over packed channels.

    ``pos[co][ci]`` / ``neg[co][ci]`` are non-negative int magnitudes.
    """
    out = []
    for co in range(len(pos)):
        p = n = 0
        wp, wn = pos[co], neg[co]
        for ci, x in enumerate(xs):
            if wp[ci]:
                p += wp[ci] * x
            if wn[ci]:
                n += wn[ci] * x
        out.append(_finish(p, n, q))
    return out


def dw_conv(xs, pos, neg, q, kernel, residual=None):
    """Depthwise convolution (``kernel`` x ``kernel``), optional residual add."""
    reach = kernel // 2
    shifts = [((ky - reach) * S + (kx - reach)) * BITS
              for ky in range(kernel) for kx in range(kernel)]
    out = []
    for c, x in enumerate(xs):
        p = n = 0
        wp, wn = pos[c], neg[c]
        for k, s in enumerate(shifts):
            a, b = wp[k], wn[k]
            if not a and not b:
                continue
            v = x >> s if s > 0 else x << -s
            if a:
                p += a * v
            if b:
                n += b * v
        if residual is not None:
            p += residual[c]
        r = _finish(p, n, q)
        if residual is not None:
            over = _ge_mask(r, ONES * (AMAX + 1))
            r = (r & ~(over * SLOT_MASK)) | (over * AMAX)
        out.append(r)
    return out


def _split(w):
    """Split a signed kernel into non-negative positive/negative parts."""
    return max(w, 0), max(-w, 0)


def _pw_tables(w):
    """w[co][ci] -> (pos, neg)."""
    pos = [[max(v, 0) for v in row] for row in w]
    neg = [[max(-v, 0) for v in row] for row in w]
    return pos, neg


def _dw_tables(w):
    """w[ch][k] -> (pos, neg)."""
    pos = [[max(v, 0) for v in row] for row in w]
    neg = [[max(-v, 0) for v in row] for row in w]
    return pos, neg


class SpatialTrunk:
    """Stacked residual blocks over a packed 17x17 patch."""

    def __init__(self, blocks, channels=CHANNELS):
        self.blocks = blocks
        self.channels = channels

    def forward(self, xs):
        h = xs
        for blk in self.blocks:
            t = pw_conv(h, *blk["pw"], blk.get("q_pw", 4))
            t = dw_conv(t, *blk["dw"], blk.get("q_dw", 4), blk.get("kernel", 3), h)
            h = t
        return h


class Head:
    """Dense head evaluated only for candidate slots we actually need."""

    def __init__(self, features, hidden_out, hidden_in):
        self.w_out = hidden_out          # [n_logits][h]
        self.b_out = features.get("bias_out", [])
        self.w_hid = features["hidden_weights"]
        self.b_hid = features["hidden_bias"]

    def hidden(self, x):
        w, b = self.w_hid, self.b_hid
        out = [float(v) for v in b] if b else [0.0] * len(w)
        nz = [(i, float(v)) for i, v in enumerate(x) if v]
        for j, row in enumerate(w):
            z = out[j]
            for i, xv in nz:
                if i < len(row):
                    z += row[i] * xv
            out[j] = z
        return [v if v > 0.0 else 0.0 for v in out]

    def logits(self, hidden, indices):
        w, b = self.w_out, self.b_out
        out = []
        for i in indices:
            row = w[i]
            z = float(b[i]) if i < len(b) else 0.0
            for k, hv in enumerate(hidden):
                if hv and k < len(row):
                    z += float(row[k]) * hv
            out.append(z)
        return out
