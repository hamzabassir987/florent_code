"""Parity: C++ factorized heads (rl/sim/florent_heads.cpp) vs the production
Python heads (bots/econ/heads.py).

Both decode the same engine state with the same random logits (argmax). The
Python side runs through a stand-in controller backed by the native engine,
using the official fcode types, so it exercises the exact production code.
Compared per unit turn: has_choice, every participating head (id, mask,
choice) and the decoded actions; spikes: the deterministic action.

    python heads_parity.py MAP [games] [rounds] [seed]
Needs the official fcode package (for its types) and build/librlcapi.so.
"""
import ctypes
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bots", "econ"))
import heads as H  # noqa: E402
from fcode import Direction, EntityType, Position  # noqa: E402

LIB = os.environ.get("RLCAPI", "build/librlcapi.so")
TYPES = [getattr(EntityType, n) for n in H.TYPE_NAMES]
KIND = {"move": 1, "spawn": 2, "convert": 3, "build": 4, "attack": 5, "heal": 6,
        "destroy": 7, "fire": 8, "rotate": 9, "launch": 10}


def lib():
    L = ctypes.CDLL(LIB)
    L.fe_new.restype = ctypes.c_void_p
    return L


class EngineCt:
    """Minimal official-API controller over the native engine (one unit)."""

    def __init__(self, L, g, a):
        self.L, self.g, self.a = L, g, a
        self.e = self.ent(a)

    def ent(self, i):
        o = (ctypes.c_int * 13)()
        self.L.fe_entity(self.g, i, o)
        return list(o)

    def idx(self, eid):
        return self.L.fe_find(self.g, eid)

    def _dir(self, d):
        return H.DXY.index(d.delta())

    def get_position(self):
        return Position(self.e[3], self.e[4])

    def get_entity_type(self, eid=None):
        return TYPES[self.e[1] if eid is None else self.ent(self.idx(eid))[1]]

    def get_team(self, eid=None):
        return self.e[2] if eid is None else self.ent(self.idx(eid))[2]

    def get_direction(self, eid=None):
        d = self.e[8]
        return next(x for x in Direction if x.delta() == H.DXY[d])

    def get_map_width(self):
        return self.L.fe_width(self.g)

    def get_map_height(self):
        return self.L.fe_height(self.g)

    def get_global_voltium(self):
        o = (ctypes.c_int * 8)()
        self.L.fe_team(self.g, self.e[2], o)
        return o[1]

    def can_spawn(self, p): return self.L.fe_can_spawn(self.g, self.a, p.x, p.y) == 0
    def can_convert_ammo(self, n): return self.L.fe_can_convert(self.g, self.a, n) == 0
    def can_move(self, d): return self.L.fe_can_move(self.g, self.a, self._dir(d)) == 0
    def can_heal(self, p): return self.L.fe_can_heal(self.g, self.a, p.x, p.y) == 0
    def can_destroy(self, p): return self.L.fe_can_destroy(self.g, self.a, p.x, p.y) == 0
    def can_rotate(self, d): return self.L.fe_can_rotate(self.g, self.a, self._dir(d)) == 0

    def can_fire(self, p):
        if self.e[1] == H.BUILDER:
            return self.L.fe_can_attack(self.g, self.a, p.x, p.y) == 0
        return self.L.fe_can_fire(self.g, self.a, p.x, p.y) == 0

    def can_build(self, t, p, d):
        return self.L.fe_can_build(self.g, self.a, TYPES.index(t), p.x, p.y, 0 if d is None else self._dir(d)) == 0

    def can_launch(self, b, t):
        return self.L.fe_can_launch(self.g, self.a, b.x, b.y, t.x, t.y) == 0

    def get_tile_builder_bot_id(self, p):
        i = self.L.fe_bot_at(self.g, p.x, p.y)
        return None if i < 0 else self.ent(i)[0]

    def get_tile_env(self, p):
        from fcode import Environment
        o = (ctypes.c_int * 5)()
        self.L.fe_vis_tile(self.g, self.a, p.x, p.y, o)
        return {0: Environment.EMPTY, 1: Environment.WALL, 2: Environment.ORE_TITANIUM,
                4: Environment.ORE_VOLTIUM, 5: Environment.VOID}[o[4]]

    def get_tile_building_id(self, p):
        o = (ctypes.c_int * 5)()
        self.L.fe_vis_tile(self.g, self.a, p.x, p.y, o)
        return None if o[2] == -1 else o[2]


def cplan(plan4):
    return None if plan4 is None else (ctypes.c_int * 4)(*plan4)


def cpp_decide(L, g, a, logits, plan4=None, own_l=None):
    out = (ctypes.c_int * 8192)()
    arr = (ctypes.c_float * len(logits))(*logits)
    L.rl_decide(g, a, arr, out, cplan(plan4), -1 if own_l is None else own_l)
    k = 0
    n = out[k]; k += 1
    acts = []
    for _ in range(n):
        acts.append(tuple(out[k:k + 5])); k += 5
    n = out[k]; k += 1
    recs = []
    for _ in range(n):
        h, c, ml = out[k], out[k + 1], out[k + 2]; k += 3
        recs.append((h, list(out[k:k + ml]), c)); k += ml
    return acts, recs


def py_action_tuple(a):
    k = KIND[a[0]]
    args = list(a[1:]) + [0] * (4 - len(a[1:]))
    return (k,) + tuple(args)


def main():
    mp = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    L = lib()
    rng = random.Random(seed)
    stats = {"turns": 0, "records": 0, "mismatch": 0, "skips": 0, "spikes": 0, "actions": 0}
    by_head = [0] * 18
    for gi in range(games):
        g = ctypes.c_void_p(L.fe_new())
        assert L.fe_load_map(g, mp.encode())
        L.fe_reset(g, ctypes.c_ulonglong(seed * 100 + gi))
        spikes_seen = set()
        for r in range(rounds):
            if L.fe_winner(g) >= 0:
                break
            if gi % 2 == 1:   # rich teams so turrets have ammo and every head is reachable
                for tm in (0, 1):
                    L.fe_dbg_set_team(g, tm, 3000, 300, 400)
            order = (ctypes.c_int * 512)()
            n = L.fe_turn_order(g, order, 512)
            for a in list(order[:n]):
                ct = EngineCt(L, g, a)
                if not ct.e[7]:
                    continue
                typ = ct.e[1]
                api = H.Api(ct)
                if typ == H.SPIKE:
                    xy = (ctypes.c_int * 2)()
                    eid = ct.e[0]
                    first = eid not in spikes_seen
                    spikes_seen.add(eid)
                    c_fire = L.rl_spike(g, a, xy, int(first))
                    p_act = H.spike_action(api, first)
                    want = (xy[0], xy[1]) if c_fire else None
                    got = (p_act[1], p_act[2]) if p_act else None
                    stats["spikes"] += 1
                    if want != got:
                        stats["mismatch"] += 1
                        print("SPIKE mismatch", r, a, want, got)
                    if c_fire:
                        L.rl_apply(g, a, 8, xy[0], xy[1], 0, 0)
                    continue
                p = L.rl_policy_of_type(typ)
                if p < 0:
                    continue
                # conveyor planner mask: half the builder turns get a random plan
                # (each adjacent tile: no plan, or a random direction)
                plan4 = None
                if typ == 0 and rng.random() < 0.5:
                    plan4 = [rng.choice((0, 0, 1, 3, 5, 7, rng.randint(1, 8))) for _ in range(4)]
                    stats["planned"] = stats.get("planned", 0) + 1
                api.plan4 = plan4
                own_l = rng.choice((None, 0, 1, 2, 3, 4)) if typ == 0 else None   # launcher cap
                api.own_launchers = own_l
                c_has = L.rl_has_choice(g, a, cplan(plan4), -1 if own_l is None else own_l) == 1
                p_has = H.has_choice(api)
                if c_has != p_has:
                    stats["mismatch"] += 1
                    print("HAS_CHOICE mismatch", r, a, typ, c_has, p_has)
                if not c_has:
                    stats["skips"] += 1
                    continue
                stats["turns"] += 1
                logits = [rng.uniform(-3, 3) for _ in range(L.rl_policy_logits(p))]
                if p == H.P_BUILDER and plan4 is not None:   # tilt toward conveyors / splitters: the planner mask
                    logits[H.head_offset(p, H.HB_MODE) + 2] += 2.5
                    for t in (0, 1):
                        logits[H.head_offset(p, H.HB_BTYPE) + t] += 3.0
                elif p == H.P_BUILDER and gi % 2 == 1:   # tilt toward turrets/launchers for coverage
                    logits[H.head_offset(p, H.HB_MODE) + 2] += 2.5
                    for t in (3, 4, 5, 6):
                        logits[H.head_offset(p, H.HB_BTYPE) + t] += 3.0
                logits = [float(ctypes.c_float(v).value) for v in logits]
                c_acts, c_recs = cpp_decide(L, g, a, logits, plan4, own_l)
                p_acts, p_recs = H.decide(api, logits)
                p_recs2 = [(h, m, c) for h, m, c, _ in p_recs]
                p_acts2 = [py_action_tuple(x) for x in p_acts]
                stats["records"] += len(c_recs)
                for h, _, _ in c_recs:
                    by_head[h] += 1
                if c_recs != p_recs2 or c_acts != p_acts2:
                    stats["mismatch"] += 1
                    if stats["mismatch"] < 6:
                        print("DECIDE mismatch r", r, "type", typ, "\n  C++", c_acts, c_recs[:3], "\n  Py ", p_acts2, p_recs2[:3])
                for act in c_acts:
                    if L.rl_apply(g, a, *act) != 0:
                        print("apply failed", act)
                    else:
                        stats["actions"] += 1
            L.fe_end_round(g)
        L.fe_free(g)
    print(stats)
    print("records per head:", {i: n for i, n in enumerate(by_head) if n})
    print("HEADS PARITY", "OK" if stats["mismatch"] == 0 else "FAILED")


if __name__ == "__main__":
    main()
