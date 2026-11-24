"""Parity test: the C++ encoder (engine/src/obs.cpp) vs the Python encoder
(bots/econ/encoder.py).

Both sides are fed by the same game through a fog-respecting adapter, so this
also proves the Python encoder only ever reads what a controller may see.

    python3 obs_parity.py [--games N] [--rounds R] [--seed S] [--map PATH]

Checks, per game:
  * every channel and scalar of every unit's actor observation, every round
  * leakage: perturbing an entity outside a unit's vision must not change its
    observation
  * critic sanity: identical on repeat, changes when the board changes
  * timing of the Python encoder (the deployed path)
"""
import argparse
import ctypes
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LIB = os.path.join(HERE, "..", "build", "libfengine.so")
sys.path.insert(0, os.path.join(ROOT, "bots", "econ"))
import encoder as E  # noqa: E402

# engine Env -> encoder terrain code
TERRAIN = {0: E.T_EMPTY, 1: E.T_WALL, 2: E.T_ORE_TI, 4: E.T_ORE_VO, 5: E.T_VOID}
ERR = ["OK", "DEAD", "WRONG_TYPE", "COOLDOWN", "ALREADY_ACTED", "RANGE", "BLOCKED",
       "FUNDS", "NOT_ORE", "BAD_DIR", "UNIT_CAP", "SPIKE_COLLAR", "NO_TARGET",
       "NO_AMMO", "TEAM", "GAME_OVER", "BAD_ARG", "ONCE_PER_TURN"]

BUILDER, CORE, GUNNER, SENTINEL, LAUNCHER, SPIKE = 0, 1, 2, 3, 4, 5
CONVEYOR, SPLITTER, HARVESTER, BARRIER, MINE = 6, 7, 8, 9, 10
VISION2 = {BUILDER: 20, CORE: 36, GUNNER: 13, SENTINEL: 32, LAUNCHER: 26, SPIKE: 8}


def load_lib():
    L = ctypes.CDLL(LIB)
    L.fe_new.restype = ctypes.c_void_p
    L.fe_find.restype = ctypes.c_int
    L.fe_get_store.restype = ctypes.c_longlong
    L.fe_obs_new.restype = ctypes.c_void_p
    L.fe_obs_next_reports.restype = ctypes.c_int
    return L


def ent(L, g, i):
    o = (ctypes.c_int * 13)()
    L.fe_entity(g, i, o)
    return list(o)


def team(L, g, t):
    o = (ctypes.c_int * 8)()
    L.fe_team(g, t, o)
    return list(o)


def turn_order(L, g):
    buf = (ctypes.c_int * 4000)()
    n = L.fe_turn_order(g, buf, 2000)
    return [buf[i] for i in range(n)]


class EngineAdapter:
    """The controller-like view of one unit, backed by the engine's fog model."""

    def __init__(self, L, g, actor):
        self.L, self.g, self.actor = L, g, actor
        o = ent(L, g, actor)
        self.o = o
        self.aid, self.typ, self.tm = o[0], o[1], o[2]
        self.x, self.y = o[3], o[4]

    def map_size(self):
        return self.L.fe_width(self.g), self.L.fe_height(self.g)

    def team(self):
        return self.tm

    def actor_id(self):
        return self.aid

    def current_round(self):
        return self.L.fe_round(self.g)

    def visible_tiles(self):
        buf = (ctypes.c_int * 8000)()
        n = self.L.fe_vis_tiles(self.g, self.actor, -1, buf, 4000)
        return [(buf[2 * i], buf[2 * i + 1]) for i in range(n)]

    def _tile(self, x, y):
        o = (ctypes.c_int * 5)()
        self.L.fe_vis_tile(self.g, self.actor, x, y, o)
        return o

    def tile_terrain(self, x, y):
        return TERRAIN[self._tile(x, y)[4]]

    def building_id(self, x, y):
        b = self._tile(x, y)[2]
        return None if b < 0 else b

    def builder_id(self, x, y):
        u = self._tile(x, y)[3]
        return None if u < 0 else u

    def entity_type(self, eid):
        return ent(self.L, self.g, self.L.fe_find(self.g, eid))[1]

    def entity_team(self, eid):
        return ent(self.L, self.g, self.L.fe_find(self.g, eid))[2]

    def entity_direction(self, eid):
        return ent(self.L, self.g, self.L.fe_find(self.g, eid))[8]

    def entity_hp_percent(self, eid):
        o = ent(self.L, self.g, self.L.fe_find(self.g, eid))
        return (100 * o[5]) // o[6] if o[6] else 0

    def actor_pos(self):
        return self.x, self.y

    def actor_state(self):
        return self.typ, self.o[5], self.o[6], self.o[9], self.o[10]

    def team_resources(self, t):
        if t != self.tm:
            return 0, 0, 0, 0, 0
        o = team(self.L, self.g, t)
        return o[0], o[1], o[2], o[5], o[6]

    # engine BASE_COST (conveyor, splitter, harvester, barrier, gunner, sentinel,
    # launcher, spike, mine, builder bot) x team scale / 100
    _BASE = (3, 6, 20, 3, 20, 30, 20, 20, 6, 30)

    def costs(self):
        sc = team(self.L, self.g, self.tm)[5]
        return [b * sc // 100 for b in self._BASE]

    def store(self, i):
        return self.L.fe_get_store(self.g, self.tm, i)


# ---------------------------------------------------------------- game driver
def play_unit(L, g, a, typ, rng, W, H):
    """One random legal action (or none)."""
    o = ent(L, g, a)
    x, y = o[3], o[4]
    if typ == BUILDER:
        opts = []
        for d in (1, 3, 5, 7):
            if L.fe_can_move(g, a, d) == 0:
                opts.append(("move", d))
        for (bx, by) in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if not (0 <= bx < W and 0 <= by < H):
                continue
            for t in (CONVEYOR, SPLITTER, HARVESTER, GUNNER, BARRIER, SENTINEL, MINE):
                d = rng.choice((1, 3, 5, 7))
                if L.fe_can_build(g, a, t, bx, by, d) == 0:
                    opts.append(("build", t, bx, by, d))
            if L.fe_can_heal(g, a, bx, by) == 0:
                opts.append(("heal", bx, by))
            if L.fe_can_attack(g, a, bx, by) == 0:
                opts.append(("attack", bx, by))
            if L.fe_can_destroy(g, a, bx, by) == 0:
                opts.append(("destroy", bx, by))
        if not opts:
            return
        c = rng.choice(opts)
        if c[0] == "move":
            L.fe_move(g, a, c[1])
        elif c[0] == "build":
            L.fe_build(g, a, c[1], c[2], c[3], c[4])
        elif c[0] == "heal":
            L.fe_heal(g, a, c[1], c[2])
        elif c[0] == "attack":
            L.fe_attack(g, a, c[1], c[2])
        else:
            L.fe_destroy(g, a, c[1], c[2])
    elif typ == CORE:
        if rng.random() < 0.7:
            for _ in range(6):
                sx = x + rng.randint(-2, 2)
                sy = y + rng.randint(-2, 2)
                if L.fe_can_spawn(g, a, sx, sy) == 0:
                    return L.fe_spawn(g, a, sx, sy) == 0
        else:
            vo = team(L, g, o[2])[1]
            if vo > 0 and L.fe_can_convert(g, a, vo) == 0:
                L.fe_convert_ammo(g, a, vo)
    elif typ in (GUNNER, SENTINEL, SPIKE):
        if rng.random() < 0.15:
            d = rng.choice((1, 2, 3, 4, 5, 6, 7, 8))
            if L.fe_can_rotate(g, a, d) == 0:
                L.fe_rotate(g, a, d)
                return
        r = 5 if typ != SPIKE else 1
        for _ in range(8):
            tx = x + rng.randint(-r, r)
            ty = y + rng.randint(-r, r)
            if L.fe_can_fire(g, a, tx, ty) == 0:
                L.fe_fire(g, a, tx, ty)
                break


# ---------------------------------------------------------------- comparison
def compare(name, cpp_ch, cpp_sc, py_ch, py_sc, rnd, eid, typ, stats, show=3):
    bad = 0
    for i, (a, b) in enumerate(zip(cpp_ch, py_ch)):
        if a != b:
            bad += 1
            if stats["shown"] < show:
                c = i // (E.PATCH * E.PATCH)
                cell = i % (E.PATCH * E.PATCH)
                dx = cell % E.PATCH - E.HALF
                dy = cell // E.PATCH - E.HALF
                print(f"  MISMATCH {name} round={rnd} unit={eid} type={typ} "
                      f"ch={c} dx={dx} dy={dy}: cpp={a} py={b}")
                stats["shown"] += 1
    for i, (a, b) in enumerate(zip(cpp_sc, py_sc)):
        if a != b:
            bad += 1
            if stats["shown"] < show:
                print(f"  MISMATCH {name} round={rnd} unit={eid} scalar={i}: cpp={a} py={b}")
                stats["shown"] += 1
    return bad


def run_game(L, map_path, seed, rounds, rng, stats):
    g = ctypes.c_void_p(L.fe_new())
    assert L.fe_load_map(g, map_path.encode())
    L.fe_reset(g, ctypes.c_ulonglong(seed))
    W, H = L.fe_width(g), L.fe_height(g)
    cpp_mem, py_mem = {}, {}
    cbuf = (ctypes.c_int * E.ACTOR_SPATIAL)()
    sbuf = (ctypes.c_int * E.ACTOR_SCALARS)()
    try:
        for r in range(rounds):
            if L.fe_winner(g) >= 0:
                break
            for a in turn_order(L, g):
                o = ent(L, g, a)
                aid, typ, tm = o[0], o[1], o[2]
                if a not in cpp_mem:
                    cpp_mem[a] = ctypes.c_void_p(L.fe_obs_new(g, tm))
                    py_mem[a] = E.ObsMemory(W, H, tm, L.fe_pit_axis(g))
                ad = EngineAdapter(L, g, a)
                # team report bus: both merge the same store snapshot, see, choose a report
                snap = [L.fe_get_store(g, tm, i) for i in range(16)]
                rnd = L.fe_round(g)
                L.fe_obs_merge(cpp_mem[a], (ctypes.c_longlong * 16)(*snap), rnd)
                py_mem[a].merge_store(snap, rnd)
                L.fe_obs_update(cpp_mem[a], g, a)
                py_mem[a].update(ad)
                cs, cw = (ctypes.c_int * 8)(), (ctypes.c_longlong * 8)()
                nrep = L.fe_obs_next_reports(cpp_mem[a], ctypes.c_longlong(aid), rnd, cs, cw)
                c_out = [(cs[i], cw[i]) for i in range(nrep)]
                p_out = py_mem[a].next_reports(aid, rnd)
                if c_out != p_out:
                    stats["bad"] += 1
                    print("  REPORT mismatch r", r, "unit", aid, c_out, p_out)
                for slot, word in c_out:
                    L.fe_write_store(g, tm, slot, ctypes.c_longlong(word))
                    stats["reports"] = stats.get("reports", 0) + 1
                    if word & (1 << 30):
                        stats["spawn_bits"] = stats.get("spawn_bits", 0) + 1
                if py_mem[a].sym >= 0:
                    stats["sym_known"] = stats.get("sym_known", 0) + 1
                L.fe_obs_encode(cpp_mem[a], g, a, cbuf, sbuf)
                pch, psc = py_mem[a].encode(ad)
                bad = compare("actor", list(cbuf), list(sbuf), pch, psc, r, aid, typ, stats)
                stats["cells"] += len(cbuf) + len(sbuf)
                stats["units"] += 1
                if any(pch[E.C_PATH * E.PATCH * E.PATCH:(E.C_PATH + 1) * E.PATCH * E.PATCH]):
                    stats["path_encodings"] = stats.get("path_encodings", 0) + 1
                if bad:
                    stats["bad"] += bad
                if play_unit(L, g, a, typ, rng, W, H) is True:
                    L.fe_obs_set_spawned(cpp_mem[a])
                    py_mem[a].spawned_now = True
            L.fe_end_round(g)
    finally:
        for p in cpp_mem.values():
            L.fe_obs_free(p)
        L.fe_free(g)


def leakage_check(L, map_path, seed, rng, stats):
    """Perturb an entity outside a unit's vision; its observation must not move."""
    g = ctypes.c_void_p(L.fe_new())
    L.fe_load_map(g, map_path.encode())
    L.fe_reset(g, ctypes.c_ulonglong(seed))
    W, H = L.fe_width(g), L.fe_height(g)
    try:
        for _ in range(20):   # let some units exist
            for a in turn_order(L, g):
                play_unit(L, g, a, ent(L, g, a)[1], rng, W, H)
            L.fe_end_round(g)
        unit = None
        for a in turn_order(L, g):
            o = ent(L, g, a)
            unit = (a, o[1], o[3], o[4], o[2], o[0])
            break
        if unit is None:
            return
        a, typ, ux, uy, tm, uid = unit
        # the first enemy builder far outside this unit's vision
        victim = None
        r2 = VISION2.get(typ, 32)
        for i in range(L.fe_n_entities(g)):
            o = ent(L, g, i)
            if not o[7] or o[0] == uid:
                continue
            d = (o[3] - ux) ** 2 + (o[4] - uy) ** 2
            if d > r2 + 16:
                victim = (i, o[5])
                break
        if victim is None:
            return
        m = E.ObsMemory(W, H, tm, L.fe_pit_axis(g))
        ad = EngineAdapter(L, g, a)
        m.update(ad)
        before, _ = m.encode(ad)
        L.fe_dbg_set_hp(g, victim[0], 1 if victim[1] != 1 else 2)
        ad2 = EngineAdapter(L, g, a)
        m.update(ad2)
        after, _ = m.encode(ad2)
        L.fe_dbg_set_hp(g, victim[0], victim[1])
        stats["leak_checks"] += 1
        if before != after:
            stats["leaks"] += 1
            print("  LEAK: observation changed after perturbing an out-of-vision entity")
    finally:
        L.fe_free(g)


def visit_history_check(L, map_path, seed, stats):
    """Observed presence stamps once; repeated visibility does not refresh it."""
    g = ctypes.c_void_p(L.fe_new())
    assert L.fe_load_map(g, map_path.encode())
    L.fe_reset(g, ctypes.c_ulonglong(seed))
    W, H = L.fe_width(g), L.fe_height(g)
    cpp_mem = None
    py_mem = None
    cbuf = (ctypes.c_int * E.ACTOR_SPATIAL)()
    sbuf = (ctypes.c_int * E.ACTOR_SCALARS)()
    try:
        core = L.fe_find(g, 1)
        spawned = None
        for y in range(H):
            for x in range(W):
                if L.fe_can_spawn(g, core, x, y) == 0:
                    if L.fe_spawn(g, core, x, y) == 0:
                        spawned = (L.fe_n_entities(g) - 1, x, y)
                        break
            if spawned:
                break
        if not spawned:
            raise AssertionError("no legal spawn tile for visit test")
        actor, x0, y0 = spawned
        tm = ent(L, g, actor)[2]
        cpp_mem = ctypes.c_void_p(L.fe_obs_new(g, tm))
        py_mem = E.ObsMemory(W, H, tm, L.fe_pit_axis(g))

        def check():
            ad = EngineAdapter(L, g, actor)
            L.fe_obs_update(cpp_mem, g, actor)
            py_mem.update(ad)
            L.fe_obs_encode(cpp_mem, g, actor, cbuf, sbuf)
            pch, psc = py_mem.encode(ad)
            assert list(cbuf) == pch and list(sbuf) == psc, "visit-test encoder mismatch"
            return list(cbuf)

        cell0 = E.HALF * E.PATCH + E.HALF
        values = check()
        assert values[E.C_VISIT * E.PATCH * E.PATCH + cell0] == 0
        L.fe_end_round(g)
        values = check()
        assert values[E.C_VISIT * E.PATCH * E.PATCH + cell0] == 1, "stationary sighting refreshed visit time"

        moved = False
        for d, dx, dy in ((1, 0, -1), (3, 1, 0), (5, 0, 1), (7, -1, 0)):
            if L.fe_can_move(g, actor, d) == 0 and L.fe_move(g, actor, d) == 0:
                moved = (dx, dy)
                break
        if not moved:
            raise AssertionError("spawned builder had no legal move for visit test")
        values = check()
        assert values[E.C_VISIT * E.PATCH * E.PATCH + cell0] == 0, "newly observed tile was not stamped"
        ox, oy = -moved[0], -moved[1]
        old_cell = (E.HALF + oy) * E.PATCH + E.HALF + ox
        assert values[E.C_VISIT * E.PATCH * E.PATCH + old_cell] == 1, "historical visit was erased on visible empty"
        stats["visit_checks"] += 1
    finally:
        if cpp_mem:
            L.fe_obs_free(cpp_mem)
        L.fe_free(g)


def turret_core_test(L, stats):
    """A production-observable friendly sentinel covers a known enemy core."""
    import tempfile
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from genmap import encode as encode_map
    with tempfile.TemporaryDirectory(prefix="obs-turret-") as td:
        path = os.path.join(td, "map.map26")
        grid = [[0] * 8 for _ in range(8)]
        with open(path, "wb") as f:
            f.write(encode_map(8, 8, grid, (0, 0), (6, 6), 0))
        g = ctypes.c_void_p(L.fe_new())
        try:
            assert L.fe_load_map(g, path.encode())
            L.fe_reset(g, ctypes.c_ulonglong(551))
            core = L.fe_find(g, 1)
            assert L.fe_spawn(g, core, 2, 1) == 0
            builder = L.fe_n_entities(g) - 1
            L.fe_end_round(g)
            assert L.fe_move(g, builder, 5) == 0  # south to (2,2)
            L.fe_end_round(g)
            assert L.fe_move(g, builder, 3) == 0  # east to (3,2)
            L.fe_end_round(g)
            assert L.fe_build(g, builder, SENTINEL, 3, 3, 4) == 0  # faces SE
            turret = L.fe_n_entities(g) - 1

            cpp_mem = ctypes.c_void_p(L.fe_obs_new(g, 0))
            try:
                py_mem = E.ObsMemory(8, 8, 0)
                ad = EngineAdapter(L, g, turret)
                L.fe_obs_update(cpp_mem, g, turret)
                py_mem.update(ad)
                cbuf = (ctypes.c_int * E.ACTOR_SPATIAL)()
                sbuf = (ctypes.c_int * E.ACTOR_SCALARS)()
                L.fe_obs_encode(cpp_mem, g, turret, cbuf, sbuf)
                pch, psc = py_mem.encode(ad)
                assert list(cbuf) == pch and list(sbuf) == psc, "turret feature parity mismatch"
                # Enemy core footprint (6,6) is three tiles SE from turret at (3,3).
                cell = (E.HALF + 3) * E.PATCH + E.HALF + 3
                assert cbuf[E.C_THREAT * E.PATCH * E.PATCH + cell] == 1
                assert cbuf[(E.C_THREAT + 2) * E.PATCH * E.PATCH + cell] == 1
                assert sbuf[36] == 1 and sbuf[37] == 1
                stats["turret_core_checks"] += 1
            finally:
                L.fe_obs_free(cpp_mem)
        finally:
            L.fe_free(g)


def critic_check(L, map_path, seed, stats):
    g = ctypes.c_void_p(L.fe_new())
    L.fe_load_map(g, map_path.encode())
    L.fe_reset(g, ctypes.c_ulonglong(seed))
    ch = (ctypes.c_int * L.fe_critic_spatial())()
    sc = (ctypes.c_int * L.fe_critic_nscalars())()
    ch2 = (ctypes.c_int * L.fe_critic_spatial())()
    sc2 = (ctypes.c_int * L.fe_critic_nscalars())()
    try:
        L.fe_critic_encode(g, ch, sc)
        L.fe_critic_encode(g, ch2, sc2)
        stats["critic_checks"] += 1
        if list(ch) != list(ch2) or list(sc) != list(sc2):
            stats["critic_bad"] += 1
            print("  CRITIC: not deterministic")
        # change the board materially (kill an entity): the encoding must change
        target = None
        for i in range(L.fe_n_entities(g)):
            o = ent(L, g, i)
            if o[7] and o[1] not in (CORE,):
                target = i
                break
        if target is not None:
            L.fe_dbg_kill(g, target)
            L.fe_critic_encode(g, ch2, sc2)
            stats["critic_checks"] += 1
            if list(ch) == list(ch2) and list(sc) == list(sc2):
                stats["critic_bad"] += 1
                print("  CRITIC: unchanged after killing an entity")
    finally:
        L.fe_free(g)


def timing(L, map_path, seed, rng, n_units=200):
    g = ctypes.c_void_p(L.fe_new())
    L.fe_load_map(g, map_path.encode())
    L.fe_reset(g, ctypes.c_ulonglong(seed))
    W, H = L.fe_width(g), L.fe_height(g)
    samples = []
    try:
        for _ in range(30):
            for a in turn_order(L, g):
                play_unit(L, g, a, ent(L, g, a)[1], rng, W, H)
            L.fe_end_round(g)
        done = 0
        while done < n_units:
            for a in turn_order(L, g):
                ad = EngineAdapter(L, g, a)
                m = E.ObsMemory(W, H, ent(L, g, a)[2], L.fe_pit_axis(g))
                t0 = time.perf_counter()
                m.update(ad)
                m.encode(ad)
                samples.append(time.perf_counter() - t0)
                done += 1
                if done >= n_units:
                    break
            L.fe_end_round(g)
    finally:
        L.fe_free(g)
    samples.sort()
    return samples[len(samples) // 2] * 1000.0, samples[int(len(samples) * 0.95)] * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--map", default=os.path.join(ROOT, "maps", "gen00.map26"))
    args = ap.parse_args()

    L = load_lib()
    stats = {"units": 0, "cells": 0, "bad": 0, "shown": 0, "leaks": 0, "leak_checks": 0,
             "critic_bad": 0, "critic_checks": 0, "visit_checks": 0, "turret_core_checks": 0}
    rng = random.Random(args.seed)
    maps = [args.map] + [os.path.join(ROOT, "maps", m) for m in
                         ("gen03.map26", "pit01.map26", "s7_gen02.map26")]
    maps = [m for m in maps if os.path.exists(m)]
    t0 = time.time()
    for k in range(args.games):
        mp = maps[k % len(maps)]
        seed = 1000 + k
        before = stats["bad"]
        run_game(L, mp, seed, args.rounds, rng, stats)
        leakage_check(L, mp, seed, rng, stats)
        critic_check(L, mp, seed, stats)
        visit_history_check(L, mp, seed, stats)
        print(f"game {k} {os.path.basename(mp)} seed={seed}: "
              f"{'OK' if stats['bad'] == before else 'MISMATCH'}")
    turret_core_test(L, stats)
    med, p95 = timing(L, maps[0], 4242, rng)
    print(f"units compared: {stats['units']}, values: {stats['cells']}, "
          f"mismatches: {stats['bad']}")
    print(f"leak checks: {stats['leak_checks']}, leaks: {stats['leaks']}")
    print(f"critic checks: {stats['critic_checks']}, failures: {stats['critic_bad']}")
    print(f"visit-history checks: {stats['visit_checks']}")
    print(f"encodings with a planned conveyor path: {stats.get('path_encodings', 0)}")
    print(f"team reports posted: {stats.get('reports', 0)} (with the spawn bit: {stats.get('spawn_bits', 0)})")
    print(f"unit turns with the map symmetry resolved: {stats.get('sym_known', 0)} of {stats['units']}")
    print(f"turret/core reach checks: {stats['turret_core_checks']}")
    print(f"python encode: median {med:.3f} ms, p95 {p95:.3f} ms per unit")
    print(f"elapsed {time.time() - t0:.1f}s")
    ok = (stats["bad"] == 0 and stats["leaks"] == 0 and stats["critic_bad"] == 0
          and stats["turret_core_checks"] == 1)
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
