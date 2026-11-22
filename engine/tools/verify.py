"""Replay-driven verifier for the native engine.

For every turn of an official replay, rebuild the actions the units took, run
them through the engine (build/libfengine.so), and compare with what the replay
recorded: accepted actions, HP of every entity, deaths, resource moves and the
team totals. After each turn the engine is resynchronised to the replay, so one
mismatch does not cascade.

    python verify.py REPLAY [REPLAY ...] [-v]
"""
import collections
import ctypes
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from replay import Replay  # noqa: E402

LIB = os.path.join(os.path.dirname(__file__), "..", "build", "libfengine.so")
TYPE = {"builder": 0, "core": 1, "gunner": 2, "sentinel": 3, "launcher": 4, "spike": 5,
        "conveyor": 6, "splitter": 7, "harvester": 8, "barrier": 9, "mine": 10}
DIRS = ["C", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DXY = {(0, -1): 1, (1, 0): 3, (0, 1): 5, (-1, 0): 7}
ERR = ["OK", "DEAD", "WRONG_TYPE", "COOLDOWN", "ALREADY_ACTED", "RANGE", "BLOCKED", "FUNDS",
       "NOT_ORE", "BAD_DIR", "UNIT_CAP", "SPIKE_COLLAR", "NO_TARGET", "NO_AMMO", "TEAM",
       "GAME_OVER", "BAD_ARG", "ONCE_PER_TURN"]


def lib():
    L = ctypes.CDLL(LIB)
    L.fe_new.restype = ctypes.c_void_p
    for name in dir(L):
        pass
    for fn in ("fe_free", "fe_load_map", "fe_reset", "fe_width", "fe_height", "fe_round", "fe_winner",
               "fe_n_entities", "fe_find", "fe_entity", "fe_team", "fe_building_at", "fe_bot_at",
               "fe_move", "fe_build", "fe_attack", "fe_heal", "fe_destroy", "fe_self_destruct", "fe_spawn",
               "fe_convert_ammo", "fe_fire", "fe_rotate", "fe_launch", "fe_end_round", "fe_last_moves",
               "fe_can_move", "fe_can_build", "fe_can_attack", "fe_can_heal", "fe_can_destroy", "fe_can_spawn",
               "fe_can_convert", "fe_can_fire", "fe_can_rotate", "fe_can_launch",
               "fe_win_condition", "fe_vis_tiles", "fe_vis_ids", "fe_vis_tile", "fe_dbg_set_stored", "fe_dbg_set_hcd", "fe_dbg_set_sides", "fe_dbg_set_team", "fe_dbg_set_hp", "fe_dbg_kill", "fe_dbg_set_last_out"):
        getattr(L, fn).argtypes = None
    return L


def write_map(r, path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
    from genmap import encode
    core = {team: pos for _, team, pos in r.cores}
    pit = None
    if r.pit:
        pit = r.pit
    with open(path, "wb") as f:
        f.write(encode(r.W, r.H, r.grid, core[0], core[1], 0, pit))


class Verifier:
    def __init__(self, L, path, verbose=False, seed=None):
        self.L, self.path, self.v = L, path, verbose
        if seed is None:
            import re
            m = re.search(r"_s(\d+)\.replay26$", path)
            seed = int(m.group(1)) if m else None
        self.seed = seed
        self.stats = collections.Counter()
        self.examples = collections.defaultdict(list)

    def note(self, key, ex=None):
        self.stats[key] += 1
        if ex is not None and len(self.examples[key]) < 5:
            self.examples[key].append(ex)

    def ent(self, i):
        o = (ctypes.c_int * 13)()
        self.L.fe_entity(self.g, i, o)
        return list(o)

    def team(self, t):
        o = (ctypes.c_int * 8)()
        self.L.fe_team(self.g, t, o)
        return list(o)

    def run(self):
        L = self.L
        r = Replay(self.path)
        fd, mp = tempfile.mkstemp(suffix=".map26"); os.close(fd)
        write_map(r, mp)
        self.g = ctypes.c_void_p(L.fe_new())
        g = self.g
        assert L.fe_load_map(g, mp.encode()), "map load failed"
        if self.seed is not None:
            L.fe_reset(g, ctypes.c_ulonglong(self.seed))
        os.unlink(mp)
        idmap = {}                     # replay id -> engine index
        for cid, team, _ in r.cores:
            idmap[cid] = self.team(team)[7]
        kind = {cid: "core" for cid, _, _ in r.cores}
        teamof = {cid: team for cid, team, _ in r.cores}
        hp = {cid: 500 for cid, _, _ in r.cores}
        stored = {}                    # (x,y) -> res code, ground truth
        res_of = {}                    # resource stack id -> res code
        hlast = {}                     # harvester replay id -> round of last output
        at = {}                        # building pos -> replay id
        for cid, _, (x, y) in r.cores:
            for dx in (0, 1):
                for dy in (0, 1):
                    at[(x + dx, y + dy)] = cid
        pos = {cid: p for cid, _, p in r.cores}
        spiked = set()
        dead_check = []
        inl = {}; outl = {}
        stored_id = {}
        self.fresh = []

        for t, turn in enumerate(r.turns):
            ev = [u for u in turn if u[0] not in ("line", "dot") and (u[0] != "output" or (u[4] and ("Q " in u[4] or "V " in u[4])))]
            players = None
            dist = None
            i = 0
            while i < len(ev):
                u = ev[i]
                k = u[0]
                nxt = ev[i + 1] if i + 1 < len(ev) else None
                if k == "place":
                    e = u[1]
                    if e["id"] in idmap:  # rotation
                        rc = L.fe_rotate(g, idmap[e["id"]], DIRS.index(e["dir"]))
                        self.check(rc, "rotate", t, e)
                    elif e["kind"] == "builder":  # spawn
                        core = self.team(e["team"])[7]
                        rc = L.fe_spawn(g, core, *e["pos"])
                        self.check(rc, "spawn", t, e)
                        self.adopt(e, idmap, kind, teamof, hp, pos, at)
                    else:
                        self.note("place without build", (t, e))
                        self.adopt(e, idmap, kind, teamof, hp, pos, at, engine_new=False)
                elif k == "build":
                    bid, p = u[1], u[2]
                    if nxt and nxt[0] == "place":
                        e = nxt[1]
                        d = DIRS.index(e.get("dir", "C"))
                        rc = L.fe_build(g, idmap.get(bid, -1), TYPE[e["kind"]], p[0], p[1], d)
                        self.check(rc, "build " + e["kind"], t, (bid, e))
                        self.adopt(e, idmap, kind, teamof, hp, pos, at)
                        if e["kind"] == "harvester":
                            hlast[e["id"]] = None
                        i += 1
                elif k == "move":
                    bid, to = u[1], u[2]
                    frm = pos.get(bid)
                    launcher = None
                    if nxt and nxt[0] != "mcd":
                        for w in ev[i + 1:i + 6]:
                            if w[0] == "acd" and kind.get(w[1]) == "launcher":
                                launcher = w[1]; break
                    if launcher is not None:
                        rc = L.fe_launch(g, idmap[launcher], frm[0], frm[1], to[0], to[1])
                        self.check(rc, "launch", t, (bid, frm, to))
                    else:
                        d = DXY.get((to[0] - frm[0], to[1] - frm[1])) if frm else None
                        if d is None:
                            self.note("move not 1 step", (t, bid, frm, to))
                        else:
                            rc = L.fe_move(g, idmap.get(bid, -1), d)
                            self.check(rc, "move", t, (bid, frm, to))
                    pos[bid] = to
                elif k == "fell":
                    bid, frm, to = u[1], u[2], u[3]
                    launcher = None
                    for w in ev[i + 1:i + 8]:
                        if w[0] == "acd" and kind.get(w[1]) == "launcher":
                            launcher = w[1]; break
                    if launcher is None:
                        for w in ev[max(0, i - 6):i]:
                            if w[0] == "acd" and kind.get(w[1]) == "launcher":
                                launcher = w[1]
                    rc = L.fe_launch(g, idmap.get(launcher, -1), frm[0], frm[1], to[0], to[1])
                    self.check(rc, "launch (fell)", t, u)
                    if rc == 0 and self.ent(idmap[bid])[7]:
                        self.note("engine: thrown bot did not fall", (t, u))
                    pos[bid] = to
                elif k == "attack":
                    rc = L.fe_attack(g, idmap.get(u[1], -1), *u[2]); self.check(rc, "attack", t, u)
                elif k == "heal":
                    rc = L.fe_heal(g, idmap.get(u[1], -1), *u[2]); self.check(rc, "heal", t, u)
                elif k == "fire":
                    src = at.get(u[1])
                    if kind.get(src) == "spike":
                        # a spike logs one fire event per tile it hits (its own tile if none):
                        # only the first one of the round is the action
                        if (src, t) in spiked:
                            i += 1; continue
                        spiked.add((src, t))
                        x, y = u[1]
                        rc = L.fe_fire(g, idmap.get(src, -1), x + 1 if x + 1 < r.W else x - 1, y)
                    else:
                        rc = L.fe_fire(g, idmap.get(src, -1), *u[2])
                    self.check(rc, "fire " + str(kind.get(src)), t, u)
                elif k == "convert":
                    rc = L.fe_convert_ammo(g, self.team(u[1])[7], u[2]); self.check(rc, "convert", t, u)
                elif k == "hp":
                    if u[1] in hp:
                        hp[u[1]] += u[2]
                elif k == "remove":
                    rid = u[1]
                    ei = idmap.get(rid)
                    if ei is not None and self.ent(ei)[7]:
                        if hp.get(rid, 1) <= 0:
                            # the replay logs a hit's effects before the hit itself:
                            # check at the end of the turn that the engine killed it too
                            dead_check.append((rid, ei))
                        else:
                            L.fe_dbg_kill(g, ei)   # destroy / self-destruct
                    p = pos.get(rid)
                    if p and at.get(p) == rid:
                        at.pop(p); stored.pop(p, None)
                    hp.pop(rid, None)
                elif k == "output" and u[4] and ("Q " in u[4] or "V " in u[4]):
                    self.queries(u[1], u[4], idmap, t)
                elif k == "distribute":
                    dist = u[1]
                elif k == "players":
                    players = (u[1], u[2])
                i += 1
            for rid, ei in dead_check:
                if self.ent(ei)[7]:
                    self.note("engine missed a death", (t, kind.get(rid)))
                    L.fe_dbg_kill(g, ei)
                else:
                    self.stats["deaths ok"] += 1
            dead_check = []
            for p in self.fresh:
                inl.pop(p, None); outl.pop(p, None)
            self.fresh = []
            # ---- end of round in the engine
            L.fe_end_round(g)
            # compare resource moves
            n = L.fe_n_entities(g)
            buf = (ctypes.c_int * (6 * 4096))()
            m = L.fe_last_moves(g, buf, 4096)
            exact = self.seed is not None
            emoves = set((buf[6 * j], buf[6 * j + 1], buf[6 * j + 2], buf[6 * j + 3]) + ((buf[6 * j + 5] & 0xFFFFFFFF,) if exact else ()) for j in range(m))
            rmoves = set((a[0], a[1], b[0], b[1]) + ((rid,) if exact else ()) for a, b, rid in (dist or []))
            self.stats["moves"] += len(rmoves)
            self.stats["moves mismatched"] += len(rmoves ^ emoves)
            if rmoves ^ emoves:
                def lab(m):
                    a, b = (m[0], m[1]), (m[2], m[3])
                    ka = kind.get(at.get(a)); kb = kind.get(at.get(b))
                    da = self.ent(idmap[at[a]])[8] if a in at and at[a] in idmap else None
                    db = self.ent(idmap[at[b]])[8] if b in at and at[b] in idmap else None
                    return f"{a}{ka}{DIRS[da] if da is not None and ka in ('conveyor','splitter') else ''}->{b}{kb}{DIRS[db] if db is not None and kb in ('conveyor','splitter') else ''}"
                self.note("move diff", (t, "replay only:", [lab(m) for m in rmoves - emoves], "engine only:", [lab(m) for m in emoves - rmoves]))
            # ground truth stored, harvester timing, side history
            SIDE = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
            for a, b, rid in (dist or []):
                d = SIDE.get((b[0] - a[0], b[1] - a[1]))
                if d is not None:
                    outl.setdefault(a, [-1] * 4)[d] = t
                    inl.setdefault(b, [-1] * 4)[SIDE[(a[0] - b[0], a[1] - b[1])]] = t
            for a, b, rid in (dist or []):
                src = at.get(a)
                if src is not None and kind.get(src) == "harvester":
                    hlast[src] = t
                    res_of[rid] = 1 if self.ent(idmap[src])[12] == 1 else 4
                elif stored.get(a) is not None:
                    res_of.setdefault(rid, stored[a]); stored.pop(a)
            for a, b, rid in (dist or []):
                stored_id.pop(a, None)
            for a, b, rid in (dist or []):
                dst = at.get(b)
                if dst is not None and kind.get(dst) in ("conveyor", "splitter"):
                    stored[b] = res_of.get(rid, 1); stored_id[b] = rid
            # compare team totals, then resync
            if players:
                for tm in (0, 1):
                    e = self.team(tm)
                    pr = players[tm]
                    want = (pr.get("ti", 0), pr.get("vo", 0), pr.get("ammo", 0),
                            pr.get("ti_collected", 0), pr.get("vo_collected", 0))
                    got = tuple(e[:5])
                    if want != got:
                        self.note("team totals differ", (t, tm, "replay", want, "engine", got))
                    else:
                        self.stats["team totals ok"] += 1
                    L.fe_dbg_set_team(g, tm, want[0], want[1], want[2])
            # compare hp and alive, resync hp
            for rid, ei in idmap.items():
                if rid not in hp:
                    continue
                e = self.ent(ei)
                if not e[7]:
                    self.note("engine killed, replay alive", (t, kind.get(rid), hp[rid], rid, pos.get(rid)))
                    hp.pop(rid, None)
                    continue
                if e[5] != hp[rid]:
                    self.note("hp differs", (t, kind.get(rid), "replay", hp[rid], "engine", e[5]))
                    L.fe_dbg_set_hp(g, ei, hp[rid])
                else:
                    self.stats["hp ok"] += 1
            # resync conveyors and harvesters
            for p, rid in at.items():
                kd = kind.get(rid)
                if kd in ("conveyor", "splitter", "harvester"):
                    if kd != "harvester":
                        L.fe_dbg_set_stored(g, p[0], p[1], stored.get(p, 0), stored_id.get(p, 0))
                    elif rid in hlast:
                        last = hlast[rid]
                        L.fe_dbg_set_hcd(g, idmap[rid], 0 if last is None else max(0, 3 - (t - last)))
                    i4 = (ctypes.c_int * 4)(*inl.get(p, [-1] * 4)); o4 = (ctypes.c_int * 4)(*outl.get(p, [-1] * 4))
                    L.fe_dbg_set_sides(g, idmap[rid], i4, o4)
                elif kd == "harvester" and rid in hlast:
                    pass
        want = (getattr(r, "winner", None), getattr(r, "win_condition", None))
        got = (L.fe_winner(g), self.L_wincond(g))
        if want[0] is not None:
            if (want[0], want[1]) == got: self.stats["winner ok"] += 1
            else: self.note("WINNER differs", ("replay", want, "engine", got))
        L.fe_free(g)
        return self

    def L_wincond(self, g):
        buf = ctypes.create_string_buffer(64)
        self.L.fe_win_condition(g, buf, 64)
        return buf.value.decode()

    QDIR = {"CENTRE": 0, "NORTH": 1, "NORTHEAST": 2, "EAST": 3, "SOUTHEAST": 4, "SOUTH": 5,
            "SOUTHWEST": 6, "WEST": 7, "NORTHWEST": 8, "-": 0}
    QTYPE = {"BUILDER_BOT": 0, "CORE": 1, "GUNNER": 2, "SENTINEL": 3, "LAUNCHER": 4, "SPIKE": 5,
             "CONVEYOR": 6, "SPLITTER": 7, "HARVESTER": 8, "BARRIER": 9, "MINE": 10}

    def queries(self, uid, text, idmap, t):
        """Ask the engine every legality question the official engine answered."""
        L, g = self.L, self.g
        a = idmap.get(uid)
        if a is None:
            return
        D, T = self.QDIR, self.QTYPE
        for line in text.splitlines():
            if line.startswith("V "):
                self.vision_check(a, line, t)
                continue
            if not line.startswith("Q "):
                continue
            parts = line.split()
            name, args, want = parts[1], parts[2:-2], parts[-1] == "1"
            n = [int(x) for x in args if x.lstrip("-").isdigit()]
            if name == "move": rc = L.fe_can_move(g, a, D[args[0]])
            elif name == "build": rc = L.fe_can_build(g, a, T[args[0]], n[0], n[1], D[args[3]])
            elif name == "heal": rc = L.fe_can_heal(g, a, n[0], n[1])
            elif name == "destroy": rc = L.fe_can_destroy(g, a, n[0], n[1])
            elif name == "attack": rc = L.fe_can_attack(g, a, n[0], n[1])
            elif name == "spawn": rc = L.fe_can_spawn(g, a, n[0], n[1])
            elif name == "convert": rc = L.fe_can_convert(g, a, n[0])
            elif name == "fire": rc = L.fe_can_fire(g, a, n[0], n[1])
            elif name == "rotate": rc = L.fe_can_rotate(g, a, D[args[0]])
            elif name == "launch": rc = L.fe_can_launch(g, a, n[0], n[1], n[2], n[3])
            else: continue
            got = rc == 0
            key = name + (" " + args[0] if name == "build" else "")
            if got == want:
                self.stats["legality ok"] += 1
            else:
                e = self.ent(a)
                self.note(f"LEGALITY {key}: official {int(want)} engine {int(got)} ({ERR[rc] if 0 <= rc < len(ERR) else rc})",
                          (t, line, "unit at", tuple(e[3:5]), "acd", e[9], "mcd", e[10]))

    ENV_NAMES = {"EMPTY": 0, "WALL": 1, "ORE_TITANIUM": 2, "ORE_VOLTIUM": 4, "VOID": 5}

    def vision_check(self, a, line, t):
        L, g = self.L, self.g
        p = line.split()
        if p[1] == "tiles":
            r2 = -1 if p[2] == "d" else int(p[2])
            want = p[3].split(",") if len(p) > 3 else []
            buf = (ctypes.c_int * 4000)()
            n = L.fe_vis_tiles(g, a, r2, buf, 2000)
            got = [f"{buf[2 * k]}:{buf[2 * k + 1]}" for k in range(n)]
            self.cmp("vision tiles", got, want, t, line)
        elif p[1] in ("ents", "blds", "units"):
            r2 = -1 if p[2] == "d" else int(p[2])
            want = [int(x) for x in p[3].split(",")] if len(p) > 3 else []
            buf = (ctypes.c_int * 4000)()
            n = L.fe_vis_ids(g, a, r2, {"ents": 0, "blds": 1, "units": 2}[p[1]], buf, 4000)
            got = [buf[k] & 0xFFFFFFFF for k in range(n)]
            self.cmp("vision " + p[1], got, want, t, line)
        elif p[1] == "tile":
            x, y = int(p[2]), int(p[3])
            o = (ctypes.c_int * 5)()
            L.fe_vis_tile(g, a, x, y, o)
            got = [o[0], o[1], o[2] & 0xFFFFFFFF if o[2] != -1 else -1, o[3] & 0xFFFFFFFF if o[3] != -1 else -1, o[4]]
            want = [int(p[4]), int(p[5]), int(p[6]), int(p[7]), self.ENV_NAMES[p[8]]]
            for name, gv, wv in zip(("empty", "passable", "building id", "bot id", "env"), got, want):
                self.cmp("vision tile " + name, gv, wv, t, line)

    def cmp(self, key, got, want, t, line):
        if got == want:
            self.stats[key + " ok"] += 1
        elif isinstance(got, list) and sorted(got) == sorted(want):
            self.note(key + ": ORDER differs", (t, "official", want[:12], "engine", got[:12]))
        else:
            self.note(key + " DIFFERS", (t, line[:120], "engine", got if not isinstance(got, list) else got[:12]))

    def adopt(self, e, idmap, kind, teamof, hp, pos, at, engine_new=True):
        n = self.L.fe_n_entities(self.g)
        if engine_new:
            last = self.ent(n - 1)
            if last[3:5] == list(e["pos"]) and last[1] == TYPE[e["kind"]]:
                idmap[e["id"]] = n - 1
                if self.seed is not None:
                    if (last[0] & 0xFFFFFFFF) == e["id"]: self.stats["ids ok"] += 1
                    else: self.note("entity id differs", (e["kind"], "replay", e["id"], "engine", last[0] & 0xFFFFFFFF))
        kind[e["id"]] = e["kind"]; teamof[e["id"]] = e["team"]; hp[e["id"]] = e["hp"]; pos[e["id"]] = e["pos"]
        if e["kind"] != "builder":
            at[e["pos"]] = e["id"]
            self.fresh.append(e["pos"])

    def check(self, rc, what, t, ex):
        if rc == 0:
            self.stats["accepted " + what.split()[0]] += 1
        else:
            self.note(f"REJECTED {what}: {ERR[rc] if rc < len(ERR) else rc}", (t, ex))


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    L = lib()
    total = collections.Counter(); ex = collections.defaultdict(list)
    for f in files:
        v = Verifier(L, f).run()
        total.update(v.stats)
        for k, xs in v.examples.items():
            if len(ex[k]) < 4:
                ex[k].extend(xs[:2])
    for k in sorted(total):
        print(f"{total[k]:8d}  {k}")
    print()
    for k in sorted(ex):
        print(k)
        for x in ex[k][:4]:
            print("    ", str(x)[:220])


if __name__ == "__main__":
    main()
