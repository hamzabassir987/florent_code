"""Decoder for .replay26 files (protobuf, schema in the visualiser). No dependencies.

    python replay.py FILE [first_turn] [last_turn]   -> human-readable dump
"""
import sys

ENV = {0: "EMPTY", 1: "WALL", 2: "ORE_TI", 4: "ORE_VO", 5: "VOID"}
DIRS = ["C", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
RES = {0: None, 1: "Ti", 4: "Vo"}
KINDS = {10: "builder", 11: "conveyor", 12: "splitter", 15: "harvester", 18: "barrier",
         20: "core", 21: "gunner", 22: "sentinel", 24: "launcher", 25: "spike", 27: "mine"}


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            return r, i


def fields(b):
    i, n, out = 0, len(b), []
    while i < n:
        k, i = _varint(b, i)
        f, t = k >> 3, k & 7
        if t == 0:
            v, i = _varint(b, i)
        elif t == 2:
            l, i = _varint(b, i); v = b[i:i + l]; i += l
        elif t == 5:
            v = int.from_bytes(b[i:i + 4], "little"); i += 4
        elif t == 1:
            v = int.from_bytes(b[i:i + 8], "little"); i += 8
        else:
            raise ValueError(f"wire type {t}")
        out.append((f, v))
    return out


def sint(v):  # int32 negative values are encoded as 64-bit two's complement
    return v - (1 << 64) if v >= 1 << 63 else v


def pos(b):
    d = dict(fields(b))
    return (sint(d.get(1, 0)), sint(d.get(2, 0)))


def packed(b):
    out, i = [], 0
    while i < len(b):
        v, i = _varint(b, i); out.append(v)
    return out


def player(b):
    d = {}
    for f, v in fields(b):
        if f == 6:
            d.setdefault("comm", []).extend(packed(v) if isinstance(v, bytes) else [v])
        else:
            d[{1: "ti", 3: "res_collected", 4: "ti_collected", 7: "ammo", 8: "vo", 9: "vo_collected"}.get(f, f)] = sint(v)
    return d


def entity(b):
    e = {"id": 0, "team": 0, "pos": (0, 0), "hp": 0, "max_hp": 0}
    for f, v in fields(b):
        if f == 1: e["id"] = v
        elif f == 2: e["team"] = v
        elif f == 3: e["pos"] = pos(v)
        elif f == 4: e["hp"] = sint(v)
        elif f == 5: e["max_hp"] = sint(v)
        elif f in KINDS:
            e["kind"] = KINDS[f]
            sub = dict(fields(v))
            if f in (11, 12, 21, 22):
                e["dir"] = DIRS[sub.get(1, 0)]
            if f in (11, 12):
                e["stored"] = RES[sub.get(2, 0)]
            if f == 15:
                e["cooldown"] = sub.get(1, 0); e["res"] = RES[sub.get(2, 0)]
            if f == 10:
                e["acd"] = sub.get(1, 0); e["mcd"] = sub.get(2, 0)
            if f == 20:
                e["acd"] = sub.get(1, 0)
    return e


def update(b):
    (f, v), = fields(b)
    d = dict(fields(v)) if f not in (4,) else None
    if f == 1: return ("place", entity(fields(v)[0][1]))
    if f == 2: return ("move", d[1], pos(d[2]))
    if f == 3: return ("remove", d[1])
    if f == 4:
        moves = []
        for _, m in fields(v):
            md = dict(fields(m))
            moves.append((pos(md[1]), pos(md[2]), md.get(3)))
        return ("distribute", moves)
    if f == 5: return ("hp", d[1], sint(d.get(2, 0)))
    if f == 6:
        pl = dict(fields(d[1])) if 1 in d else {}
        ps = dict(fields(v))[1]
        a = b2 = {}
        for ff, vv in fields(ps):
            if ff == 1: a = player(vv)
            if ff == 2: b2 = player(vv)
        return ("players", a, b2)
    if f == 7: return ("acd", d[1], sint(d.get(2, 0)))
    if f == 8: return ("mcd", d[1], sint(d.get(2, 0)))
    if f == 9: return ("output", d.get(1, 0), d.get(3, 0), bool(d.get(4, 0)), d.get(2, b"").decode(errors="replace"))
    if f == 10: return ("line",)
    if f == 11: return ("dot",)
    if f == 12: return ("fire", pos(d[1]), pos(d[2]))
    if f == 13: return ("attack", d[1], pos(d[2]))
    if f == 14: return ("convert", d.get(1, 0), d.get(2, 0))
    if f == 15: return ("heal", d[1], pos(d[2]))
    if f == 16: return ("build", d[1], pos(d[2]))
    if f == 17: return ("mine", d.get(1), d.get(2), pos(d[3]))
    if f == 18: return ("fell", d[1], pos(d[2]), pos(d[3]))
    return ("unknown", f)


class Replay:
    def __init__(self, path):
        b = open(path, "rb").read()
        top = fields(b)
        self.turns = []
        for f, v in top:
            if f == 1:
                self._map(v)
            elif f == 3:
                self.turns.append([update(u) for _, u in fields(v)])
            elif f == 4:
                self.winner = v
            elif f == 6:
                self.win_condition = v.decode()

    def _map(self, b):
        self.cores = []
        rows = []
        self.pit = None
        for f, v in fields(b):
            if f == 1: self.W = v
            elif f == 2: self.H = v
            elif f == 3:
                r = fields(v)
                rows.append(packed(r[0][1]) if r else [])
            elif f == 4:
                d = dict(fields(v)); self.cores.append((d.get(1), d.get(2, 0), pos(d[3])))
            elif f == 8:
                d = dict(fields(v)); self.pit = (d.get(1, 0), d.get(2, 0), d.get(3, 0))
        self.grid = [r + [0] * (self.W - len(r)) for r in rows]


if __name__ == "__main__":
    r = Replay(sys.argv[1])
    a = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    z = int(sys.argv[3]) if len(sys.argv) > 3 else a + 3
    print(f"map {r.W}x{r.H} cores {r.cores} pit {r.pit} turns {len(r.turns)} "
          f"winner {getattr(r, 'winner', None)} {getattr(r, 'win_condition', '')}")
    for t in range(a, min(z, len(r.turns))):
        print(f"--- turn {t}")
        for u in r.turns[t]:
            if u[0] in ("output", "line", "dot"):
                continue
            print("  ", u)
