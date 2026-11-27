"""Production-safe fog-limited actor observation encoder.

Keep one ActorEncoder on the bot's persistent Player object. It keys independent
map memory by unit ID and reads only tiles/entities this controller can see.
Mirrors engine/src/obs.cpp; engine/tools/obs_parity.py compares every output.
No numpy is used.
"""
import math

PATCH, HALF, RAY_LEN = 17, 8, 32
T_UNKNOWN, T_EMPTY, T_WALL, T_ORE_TI, T_ORE_VO, T_VOID = range(6)
BUILDER, CORE, GUNNER, SENTINEL, LAUNCHER, SPIKE = range(6)
CONVEYOR, SPLITTER, HARVESTER, BARRIER, MINE = range(6, 11)
UNIT_TYPES = frozenset(range(6))
CH_TERRAIN, CH_OWN, CH_ENEMY, CH_OCC = 6, 11, 10, 2
CH_STALE, CH_VISIT, CH_ATTR_STALE = 1, 1, 2
CH_DIR, CH_HP, CH_THREAT, CH_RAY = 9, 2, 4, 11
CH_PATH = 1   # builders only: planned conveyor cell, value = conveyor direction (Dir 1..8), 0 none
CH_HIT = 1    # rounds since a unit / building on the tile was seen losing HP, cap HIT_CAP (= never)
HIT_CAP = 10
CH_MINE = 1   # known hidden enemy mine (detected by a builder's attack probe, or reported): 1
ACTOR_CH = CH_TERRAIN + CH_OWN + CH_ENEMY + CH_OCC + CH_STALE + CH_VISIT + CH_ATTR_STALE + CH_DIR + CH_HP + CH_THREAT + CH_RAY + CH_PATH + CH_HIT + CH_MINE
ACTOR_SPATIAL = ACTOR_CH * PATCH * PATCH
ACTOR_SCALARS = 50
ACTOR_COSTS = 40   # 40..49: current prices (conveyor, splitter, harvester, barrier, gunner,
                   # sentinel, launcher, spike, mine, builder bot)
C_TERRAIN = 0
C_OWN = C_TERRAIN + CH_TERRAIN
C_ENEMY = C_OWN + CH_OWN
C_OCC = C_ENEMY + CH_ENEMY
C_STALE = C_OCC + CH_OCC
C_VISIT = C_STALE + CH_STALE
C_ATTR_STALE = C_VISIT + CH_VISIT
C_DIR = C_ATTR_STALE + CH_ATTR_STALE
C_HP = C_DIR + CH_DIR
C_THREAT = C_HP + CH_HP
C_RAY = C_THREAT + CH_THREAT
C_PATH = C_RAY + CH_RAY
C_HIT = C_PATH + CH_PATH
C_MINE = C_HIT + CH_HIT
_NXY = ((0, -1), (1, 0), (0, 1), (-1, 0))   # N, E, S, W
_NDIR = (1, 3, 5, 7)                        # engine Dir codes of N, E, S, W


def ray_path(x0, y0, dx, dy, maxlen=RAY_LEN):
    out = []
    length = max(abs(dx), abs(dy))
    if not length:
        return out
    sx, sy = dx / length, dy / length
    px, py, lx, ly = float(x0), float(y0), x0, y0
    for _ in range(maxlen):
        px += sx
        py += sy
        ix, iy = int(math.floor(px + .5)), int(math.floor(py + .5))
        if (ix, iy) != (lx, ly):
            out.append((ix, iy))
            lx, ly = ix, iy
    return out


# ---- team report bus (identical to engine/src/obs.cpp; see its "report bus"
# comment and BusConfig / BusGroup in obs.hpp; checked by engine/tools/obs_parity.py)
G_MYB, G_EB, G_ORE, G_EATK, G_MINE = range(5)
BUS_BASE, BUS_SIZE = (0, 4, 6, 10, 14), (4, 2, 4, 4, 2)
SLOT_GROUP = [g for g in range(5) for _ in range(BUS_SIZE[g])]
W_MARK, W_SPAWN, W_ALARM, W_REMOVE = 1 << 31, 1 << 30, 1 << 27, 1 << 14
TAG_CORE = 63
EATK_TYPES = (CORE, GUNNER, SENTINEL, LAUNCHER, SPIKE)
EATK_MINE_CODE = 5   # G_EATK code 5: a known hidden enemy mine (ObsMemory.emine)
MINE_TYPES = (LAUNCHER, SPIKE, GUNNER, SENTINEL, MINE, HARVESTER)
REFRESH, REFRESH_FACTS, REFRESH_GATE = 4, 16, 4
SPAWN_COOLDOWN, QUIET_AFTER, QUIET_ROUNDS, ENEMY_BOT_TTL = 4, 40, 4, 10


def _content(w):
    return w & ~(W_MARK | W_SPAWN | (3 << 28))


def _cheb(a, b, W):
    return max(abs(a % W - b % W), abs(a // W - b // W))


LINE_CAP = 4   # conveyor planner: harvesters per line into the core (1 stack / round)


class BusItem:
    __slots__ = ("key", "word", "due", "sent", "fresh", "remove", "sent_word", "anchor")

    def __init__(self, key, word, due, remove):
        self.key, self.word, self.due, self.sent = key, word, due, -1
        self.fresh, self.remove, self.sent_word, self.anchor = True, remove, 0, -1


class ObsMemory:
    def __init__(self, W, H, team, pit_axis=-1):
        self.W, self.H, self.team = W, H, team
        n = W * H
        self.terrain = [T_UNKNOWN] * n
        self.bld_id = [-1] * n
        self.bld_type = [-1] * n
        self.bld_team = [-1] * n
        self.bld_dir = [-1] * n
        self.bld_hp = [-1] * n
        self.bot_id = [-1] * n
        self.bot_team = [-1] * n
        self.bot_hp = [-1] * n
        self.seen_round = [-1] * n
        self.stood = [0] * n
        self.visit_round = [-1] * n
        self.hit_round = [-1] * n
        self.emine = [0] * n   # known hidden enemy mines (as ObsMemory::emine in C++)
        self.attr_seen_round = [-1] * n
        self.bus_q = [[] for _ in range(5)]
        self.snap, self.snap_ore = set(), [0] * n
        self.rep_bot_tile, self.rep_bot_round = [-1] * 128, [-1] * 128
        self.born, self.last_spawn_react, self.snap_spawn = -1, -1000, False
        self.core_alarm_round, self.last_hp = -1, -1
        self.spawned_now = self.core_alarm_now = self.is_builder = False
        self.sym_alive, self.sym = 7, -1
        # a pit lies on the mirror axis: x pit -> mirror x (1), y pit -> mirror y (2)
        if pit_axis >= 0:
            self.sym_alive = 1 << (1 if pit_axis == 0 else 2)

    # ---- map symmetry (identical to ObsMemory::mirror / learn_terrain / resolve_symmetry)
    def mirror(self, t, s):
        W, H = self.W, self.H
        x, y = t % W, t // W
        return (y if s == 1 else H - 1 - y) * W + (x if s == 2 else W - 1 - x)

    def learn_terrain(self, t):
        terrain = self.terrain
        ter = terrain[t]
        if ter == T_UNKNOWN:
            return
        blocked = ter == T_WALL or ter == T_VOID
        for s in range(3):
            if not (self.sym_alive >> s) & 1:
                continue
            mt = self.mirror(t, s)
            if (terrain[mt] != T_UNKNOWN and terrain[mt] != ter) or (blocked and self.stood[mt]):
                self.sym_alive &= ~(1 << s) & 7
        if self.sym >= 0:
            mt = self.mirror(t, self.sym)
            if terrain[mt] == T_UNKNOWN:
                terrain[mt] = ter

    def learn_passable(self, t):
        """A reported unit / building stands on t: a symmetry mapping t onto a known wall or void is out."""
        self.stood[t] = 1
        for s in range(3):
            if (self.sym_alive >> s) & 1 and self.terrain[self.mirror(t, s)] in (T_WALL, T_VOID):
                self.sym_alive &= ~(1 << s) & 7

    def resolve_symmetry(self):
        n, team = self.W * self.H, self.team
        bid, bty, btm = self.bld_id, self.bld_type, self.bld_team
        own_core = [t for t in range(n) if bid[t] >= 0 and btm[t] == team and bty[t] == CORE]
        for s in range(3):
            if not (self.sym_alive >> s) & 1:
                continue
            for t in own_core:
                mt = self.mirror(t, s)
                if bid[mt] >= 0 and btm[mt] == team and bty[mt] == CORE:
                    self.sym_alive &= ~(1 << s) & 7
                    break
        # the cores mirror each other: a candidate is out when it maps a known
        # enemy core tile onto a tile this unit saw without our core, or one of
        # our core tiles onto a tile it saw without the enemy core
        if self.sym < 0:
            seen = self.seen_round
            for t in range(n):
                if bid[t] < 0 or bty[t] != CORE:
                    continue
                want = team if btm[t] != team else 1 - team
                for s in range(3):
                    if not (self.sym_alive >> s) & 1:
                        continue
                    mt = self.mirror(t, s)
                    if seen[mt] >= 0 and not (bid[mt] >= 0 and bty[mt] == CORE and btm[mt] == want):
                        self.sym_alive &= ~(1 << s) & 7
        # a known enemy core tile must mirror a tile of our 2x2 core: a known one,
        # or an unseen tile next to every known one (part of our core not seen yet)
        if own_core and self.sym < 0:
            W = self.W
            for t in range(n):
                if not (bid[t] >= 0 and btm[t] != team and bty[t] == CORE):
                    continue
                for s in range(3):
                    if not (self.sym_alive >> s) & 1:
                        continue
                    mt = self.mirror(t, s)
                    match = mt in own_core or (self.seen_round[mt] < 0 and all(
                        max(abs(o % W - mt % W), abs(o // W - mt // W)) <= 1 for o in own_core))
                    if not match:
                        self.sym_alive &= ~(1 << s) & 7
        if self.sym < 0 and self.sym_alive in (1, 2, 4):
            self.sym = {1: 0, 2: 1, 4: 2}[self.sym_alive]
            terrain = self.terrain
            for t in range(n):
                mt = self.mirror(t, self.sym)
                if terrain[t] != T_UNKNOWN and terrain[mt] == T_UNKNOWN:
                    terrain[mt] = terrain[t]
        if self.sym >= 0:
            enemy = 1 - team
            for t in own_core:
                mt = self.mirror(t, self.sym)
                if bid[mt] < 0 and self.seen_round[mt] < 0:
                    bid[mt], bty[mt], btm[mt] = enemy + 1, CORE, enemy
                    self.bld_dir[mt], self.bld_hp[mt] = -1, 100
                    if self.terrain[mt] == T_UNKNOWN:
                        self.terrain[mt] = T_EMPTY

    # ---- report bus (identical to engine/src/obs.cpp)
    def own_count(self, typ):
        """Own buildings of a type in memory (seen or reported)."""
        team, bid, bty, btm = self.team, self.bld_id, self.bld_type, self.bld_team
        return sum(1 for t in range(len(bid)) if bid[t] >= 0 and btm[t] == team and bty[t] == typ)

    def targets(self):
        """The masks' target knowledge (as targets_from_memory in C++): known enemy
        core / building tiles as (x, y), own gunners + sentinels."""
        W, team = self.W, self.team
        core, bld = set(), set()
        for t in range(W * self.H):
            if self.bld_id[t] < 0 or self.bld_team[t] == team:
                continue
            bld.add((t % W, t // W))
            if self.bld_type[t] == CORE:
                core.add((t % W, t // W))
        vo = sum(1 for t in range(W * self.H) if self.bld_id[t] >= 0 and self.bld_team[t] == team
                 and self.bld_type[t] == HARVESTER and self.terrain[t] == T_ORE_VO)
        return {"core": core, "bld": bld, "own_turrets": self.own_count(GUNNER) + self.own_count(SENTINEL),
                "own_vo_harvesters": vo}

    def extra_bits(self):
        return (self.sym + 1) << 28 if self.sym >= 0 else 0

    def _clear_bot(self, t):
        self.bot_id[t] = self.bot_team[t] = self.bot_hp[t] = -1

    def _place_reported_bot(self, tm, tag, t, rr):
        idx = tm * 64 + tag
        sid = (1 << 28) | idx
        old = self.rep_bot_tile[idx]
        if old >= 0 and old != t and self.bot_id[old] == sid:
            self._clear_bot(old)
        if self.bot_id[t] < 0 or self.bot_id[t] == sid:
            self.bot_id[t], self.bot_team[t], self.bot_hp[t] = sid, tm, -1
            self.visit_round[t] = rr   # terrain left as is: a builder can stand on ore
            self.learn_passable(t)
        self.rep_bot_tile[idx], self.rep_bot_round[idx] = t, rr

    def merge_store(self, store, rnd):
        W, H, n, team = self.W, self.H, self.W * self.H, self.team
        enemy = 1 - team
        self.snap = set()
        self.snap_ore = [0] * n
        self.snap_spawn = False
        for i, v in enumerate(store):
            if v < 0 or v > 0xFFFFFFFF or not (v & W_MARK):
                continue
            grp, t = SLOT_GROUP[i], v & 0x7FF
            if t >= n:
                continue
            ss = (v >> 28) & 3
            if ss and self.sym < 0 and (self.sym_alive >> (ss - 1)) & 1:
                self.sym_alive = 1 << (ss - 1)
            if v & W_SPAWN:
                self.snap_spawn = True
            self.snap.add((grp << 28) | _content(v))
            if grp == G_MYB or grp == G_EB:
                tag = (v >> 11) & 63
                rr = rnd - ((rnd - ((v >> 17) & 63)) & 63)
                if grp == G_MYB and tag == TAG_CORE:
                    if v & W_ALARM:
                        self.core_alarm_round = max(self.core_alarm_round, rr)
                    continue
                if rnd - rr > ENEMY_BOT_TTL:
                    continue
                self._place_reported_bot(team if grp == G_MYB else enemy, tag, t, rr)
            elif grp == G_ORE:
                kind, x, y = (v >> 11) & 1, t % W, t // W
                bits = (v >> 12) & 0x7FFF
                for k in range(16):
                    if k > 0 and not (bits >> (k - 1)) & 1:
                        continue
                    xx, yy = x + k % 4, y + k // 4
                    if xx >= W or yy >= H:
                        continue
                    tt = yy * W + xx
                    self.snap_ore[tt] |= 1 << kind
                    if self.terrain[tt] == T_UNKNOWN:
                        self.terrain[tt] = T_ORE_VO if kind else T_ORE_TI
                    self.learn_terrain(tt)
            else:
                eatk = grp == G_EATK
                code = (v >> 11) & 7
                if eatk and code == EATK_MINE_CODE:   # a reported hidden mine
                    self.emine[t] = 0 if v & W_REMOVE else 1
                    continue
                if code >= (5 if eatk else 6):
                    continue
                typ = (EATK_TYPES if eatk else MINE_TYPES)[code]
                owner = enemy if eatk else team
                if v & W_REMOVE:
                    if self.bld_id[t] >= 0 and self.bld_team[t] == owner and self.bld_type[t] == typ:
                        self.bld_id[t] = self.bld_type[t] = self.bld_team[t] = -1
                        self.bld_dir[t] = self.bld_hp[t] = -1
                elif self.bld_id[t] < 0:
                    self.bld_id[t] = owner + 1 if typ == CORE else ((1 << 30) if eatk else (1 << 29)) + t
                    self.bld_type[t], self.bld_team[t] = typ, owner
                    self.bld_dir[t], self.bld_hp[t] = -1, 100   # terrain left as is (a harvester stands on ore)
                    self.learn_passable(t)
        for idx in range(128):
            t = self.rep_bot_tile[idx]
            if t < 0 or rnd - self.rep_bot_round[idx] <= ENEMY_BOT_TTL:
                continue
            if self.bot_id[t] == ((1 << 28) | idx):
                self._clear_bot(t)
            self.rep_bot_tile[idx] = -1

    def _bus_after_update(self, ad, rnd, gone_eatk, gone_mine, enemy_bots, enemy_near, gone_emine=()):
        W, n, team = self.W, self.W * self.H, self.team
        enemy = 1 - team
        typ, hp = ad.actor_state()[:2]
        ax, ay = ad.actor_pos()
        uid = ad.actor_id()
        if typ == CORE:
            self.core_alarm_now = (self.last_hp >= 0 and hp < self.last_hp) or enemy_near
            self.last_hp = hp
            if self.core_alarm_now:
                self.core_alarm_round = rnd

        def want(grp, key, word, remove):
            for it in self.bus_q[grp]:
                if it.key == key:
                    it.word = word
                    return it
            it = BusItem(key, word, rnd, remove)
            self.bus_q[grp].append(it)
            return it

        r6 = (rnd & 63) << 17
        core_status = typ == CORE and (self.core_alarm_now or self.spawned_now)
        self.bus_q[G_MYB] = [it for it in self.bus_q[G_MYB]
                             if (it.key == 0 and self.is_builder) or (it.key == TAG_CORE and core_status)]
        if self.is_builder:
            t = ay * W + ax
            it = want(G_MYB, 0, t | ((uid % 63) << 11) | r6, False)
            if it.anchor >= 0 and _cheb(it.anchor, t, W) >= 2:
                it.due = min(it.due, rnd)
        if core_status:
            it = want(G_MYB, TAG_CORE, (ay * W + ax) | (TAG_CORE << 11) | r6 | (W_ALARM if self.core_alarm_now else 0), False)
            it.due = min(it.due, rnd)
        tags = set(tag for _, tag in enemy_bots)
        self.bus_q[G_EB] = [it for it in self.bus_q[G_EB] if it.key in tags]
        for t, tag in enemy_bots:
            it = want(G_EB, tag, t | (tag << 11) | r6, False)
            if it.anchor >= 0 and _cheb(it.anchor, t, W) >= 2:
                it.due = min(it.due, rnd)
        keep = [set() for _ in range(5)]
        terrain, bid, bty, btm = self.terrain, self.bld_id, self.bld_type, self.bld_team
        for t in range(n):
            ter = terrain[t]
            if ter == T_ORE_TI or ter == T_ORE_VO:
                key = t | ((1 << 11) if ter == T_ORE_VO else 0)
                want(G_ORE, key, key, False)
                keep[G_ORE].add(key)
            if self.emine[t]:
                key = t | (EATK_MINE_CODE << 11)
                want(G_EATK, key, key, False)
                keep[G_EATK].add(key)
            if bid[t] < 0:
                continue
            if btm[t] == enemy:
                if bty[t] in EATK_TYPES:
                    key = t | (EATK_TYPES.index(bty[t]) << 11)
                    want(G_EATK, key, key, False)
                    keep[G_EATK].add(key)
            elif btm[t] == team:
                if bty[t] in MINE_TYPES:
                    key = t | (MINE_TYPES.index(bty[t]) << 11)
                    want(G_MINE, key, key, False)
                    keep[G_MINE].add(key)
        for t, c in gone_eatk:
            key = t | (c << 11) | W_REMOVE
            want(G_EATK, key, key, True)
            keep[G_EATK].add(key)
        for t in gone_emine:
            key = t | (EATK_MINE_CODE << 11) | W_REMOVE
            want(G_EATK, key, key, True)
            keep[G_EATK].add(key)
        for t, c in gone_mine:
            key = t | (c << 11) | W_REMOVE
            want(G_MINE, key, key, True)
            keep[G_MINE].add(key)
        for grp in (G_ORE, G_EATK, G_MINE):
            self.bus_q[grp] = [it for it in self.bus_q[grp] if it.key in keep[grp] or it.remove]
        # delivery checks against this turn's snapshot
        for grp in range(5):
            q = []
            for it in self.bus_q[grp]:
                if grp == G_ORE:
                    delivered = bool((self.snap_ore[it.key & 0x7FF] >> ((it.key >> 11) & 1)) & 1)
                elif grp <= G_EB:
                    delivered = it.sent >= 0 and ((grp << 28) | it.sent_word) in self.snap
                else:
                    delivered = ((grp << 28) | it.word) in self.snap
                if delivered:
                    if it.remove:
                        continue
                    if it.sent >= 0 or it.due <= rnd:
                        it.due = rnd + (REFRESH if grp <= G_EB else REFRESH_FACTS)
                    it.fresh, it.sent, it.anchor = False, -1, it.word & 0x7FF
                elif it.sent >= 0 and it.sent < rnd:
                    it.due, it.sent = rnd, -1
                q.append(it)
            self.bus_q[grp] = q
        if self.snap_spawn and rnd - self.last_spawn_react >= SPAWN_COOLDOWN:
            self.last_spawn_react = rnd
            for q in self.bus_q:
                for it in q:
                    it.due = min(it.due, rnd)

    def next_reports(self, unit_id, rnd):
        """This turn's writes: [(slot, word)], at most one per slot group."""
        out = []
        if self.is_builder and self.born > QUIET_AFTER and rnd - self.born < QUIET_ROUNDS:
            return out
        W, H, n, team = self.W, self.H, self.W * self.H, self.team
        own_core = [t for t in range(n) if self.bld_id[t] >= 0 and self.bld_team[t] == team and self.bld_type[t] == CORE]
        k = unit_id + rnd
        spread = (unit_id * 2654435761 + rnd * 40503) & 0xFFFFFFFF
        for grp in range(5):
            refresh_ok = grp <= G_EB or k % REFRESH_GATE == 0
            best, bk = None, None
            for it in self.bus_q[grp]:
                if it.due > rnd or (not it.fresh and not refresh_ok):
                    continue
                prio = 0
                if grp == G_EATK and own_core:
                    prio = min(_cheb(c, it.key & 0x7FF, W) for c in own_core)
                if grp == G_MINE:   # our buildings by type code, launchers first
                    prio = (it.key >> 11) & 7
                key = (0 if it.fresh else 1, prio, it.key ^ spread)
                if best is None or key < bk:
                    best, bk = it, key
            if best is None:
                continue
            w = best.word
            if grp == G_ORE:
                t = w & 0x7FF
                x, y = t % W, t // W
                ore = T_ORE_VO if (w >> 11) & 1 else T_ORE_TI
                for c in range(1, 16):
                    xx, yy = x + c % 4, y + c // 4
                    if xx < W and yy < H and self.terrain[yy * W + xx] == ore:
                        w |= 1 << (11 + c)
            best.sent, best.sent_word = rnd, w
            full = W_MARK | w | self.extra_bits()
            if grp == G_MYB and best.key == TAG_CORE and self.spawned_now:
                full |= W_SPAWN
                self.spawned_now = False
            out.append((BUS_BASE[grp] + k % BUS_SIZE[grp], full))
        return out

    def update(self, ad):
        r, W = ad.current_round(), self.W
        team = self.team
        if self.born < 0:
            self.born = r
        self.is_builder = ad.actor_state()[0] == BUILDER
        gone_eatk, gone_mine, enemy_bots, enemy_near = [], [], [], False
        for x, y in ad.visible_tiles():
            t = y * W + x
            old_id, old_type, old_team = self.bld_id[t], self.bld_type[t], self.bld_team[t]
            old_bhp, old_bot, old_uhp = self.bld_hp[t], self.bot_id[t], self.bot_hp[t]
            self.terrain[t] = ad.tile_terrain(x, y)
            b = ad.building_id(x, y)
            if b is None:
                self.bld_id[t] = self.bld_type[t] = self.bld_team[t] = -1
                self.bld_dir[t] = self.bld_hp[t] = -1
            else:
                self.bld_id[t] = b
                self.bld_type[t] = ad.entity_type(b)
                self.bld_team[t] = ad.entity_team(b)
                self.bld_dir[t] = ad.entity_direction(b)
                self.bld_hp[t] = ad.entity_hp_percent(b)
                self.attr_seen_round[t] = r
            u = ad.builder_id(x, y)
            if u is None:
                self.bot_id[t] = self.bot_team[t] = self.bot_hp[t] = -1
            else:
                if self.bot_id[t] != u:
                    self.visit_round[t] = r
                self.bot_id[t] = u
                self.bot_team[t] = ad.entity_team(u)
                self.bot_hp[t] = ad.entity_hp_percent(u)
                self.attr_seen_round[t] = r
            # seen losing HP since the last look (same entity, lower HP percentage)
            if ((self.bld_id[t] >= 0 and self.bld_id[t] == old_id and old_bhp >= 0 and self.bld_hp[t] < old_bhp) or
                    (self.bot_id[t] >= 0 and self.bot_id[t] == old_bot and old_uhp >= 0 and self.bot_hp[t] < old_uhp)):
                self.hit_round[t] = r
            self.seen_round[t] = r
            self.learn_terrain(t)
            if self.bot_id[t] >= 0 and self.bot_team[t] != team and u is not None:
                enemy_bots.append((t, u % 63))
                enemy_near = True
            if self.bld_id[t] >= 0 and self.bld_team[t] != team:
                enemy_near = True
            if old_id >= 0:
                same = self.bld_id[t] >= 0 and self.bld_type[t] == old_type and self.bld_team[t] == old_team
                if not same:
                    if old_team == 1 - team and old_type in EATK_TYPES:
                        gone_eatk.append((t, EATK_TYPES.index(old_type)))
                    if old_team == team and old_type in MINE_TYPES:
                        gone_mine.append((t, MINE_TYPES.index(old_type)))
        # hidden enemy mines: the attack probe (as in C++, see emine in obs.hpp); a
        # visible building or our own builder on a tile means no hidden mine there
        gone_emine = []
        typ, _, _, acd, _ = ad.actor_state()
        if typ == BUILDER and acd == 0 and ad.team_resources(team)[0] >= 2:
            ax, ay = ad.actor_pos()
            for dx, dy in _NXY:
                x, y = ax + dx, ay + dy
                if not (0 <= x < W and 0 <= y < self.H):
                    continue
                t = y * W + x
                if ad.building_id(x, y) is not None:
                    continue
                if ad.can_attack(x, y):
                    self.emine[t] = 1
                elif self.emine[t]:
                    self.emine[t] = 0
                    gone_emine.append(t)
        for x, y in ad.visible_tiles():
            t = y * W + x
            if self.emine[t] and (self.bld_id[t] >= 0 or (self.bot_id[t] >= 0 and self.bot_team[t] == team)):
                self.emine[t] = 0
                gone_emine.append(t)
        self.resolve_symmetry()
        self._bus_after_update(ad, r, gone_eatk, gone_mine, enemy_bots, enemy_near, gone_emine)

    def plan_paths(self):
        """Conveyor planner, identical to ObsMemory::plan_paths (engine/src/obs.cpp):
        per tile the planned conveyor direction (Dir code), 0 = not on a path.
        A line (the path's last tile before the core) takes at most LINE_CAP
        own harvesters; once full, its tiles are closed and later sources are
        planned around it."""
        W, H, team = self.W, self.H, self.team
        N = W * H
        terrain, bid, bty, btm = self.terrain, self.bld_id, self.bld_type, self.bld_team
        cores = [t for t in range(N) if bid[t] >= 0 and btm[t] == team and bty[t] == CORE]
        blocked = [False] * N
        free = [terrain[t] == T_EMPTY and (bid[t] < 0 or (btm[t] == team and (bty[t] == CONVEYOR or bty[t] == SPLITTER)))
                for t in range(N)]

        def bfs():
            dist = [-1] * N
            q = list(cores)
            for t in q:
                dist[t] = 0
            h = 0
            while h < len(q):
                t = q[h]; h += 1
                x, y = t % W, t // W
                for dx, dy in _NXY:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        nt = ny * W + nx
                        if dist[nt] < 0 and free[nt] and not blocked[nt]:
                            dist[nt] = dist[t] + 1
                            q.append(nt)
            return dist

        dist = bfs()
        plan = [0] * N
        owner = [-1] * N
        load = [0] * N
        for pas in (0, 1):
            for s in range(N):
                if terrain[s] != T_ORE_TI and terrain[s] != T_ORE_VO:
                    continue
                harv = bid[s] >= 0 and btm[s] == team and bty[s] == HARVESTER
                if bid[s] >= 0 and not harv:
                    continue
                if harv != (pas == 0):
                    continue
                sx, sy = s % W, s // W
                best, bd = -1, 1 << 30
                for dx, dy in _NXY:
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        nt = ny * W + nx
                        if 0 <= dist[nt] < bd:
                            best, bd = nt, dist[nt]
                if best < 0 or bd == 0:
                    continue
                path = []
                c, joined = best, -1
                while dist[c] > 0:
                    if plan[c] != 0 and joined < 0:
                        joined = owner[c]
                    x, y = c % W, c // W
                    for k, (dx, dy) in enumerate(_NXY):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny < H and dist[ny * W + nx] == dist[c] - 1:
                            if plan[c] == 0:
                                plan[c] = _NDIR[k]
                                path.append(c)
                            if dist[c] == 1 and joined < 0:
                                joined = c
                            c = ny * W + nx
                            break
                for t in path:
                    owner[t] = joined
                if harv:
                    load[joined] += 1
                    if load[joined] == LINE_CAP:
                        for t in range(N):
                            if owner[t] == joined:
                                blocked[t] = True
                        dist = bfs()
        return plan

    def encode(self, ad):
        W, H, team = self.W, self.H, self.team
        rnd = ad.current_round()
        ax, ay = ad.actor_pos()
        typ, hp, maxhp, acd, mcd = ad.actor_state()
        cs = PATCH * PATCH
        out = [0] * ACTOR_SPATIAL
        ch = out
        terrain = self.terrain

        for dy in range(-HALF, HALF + 1):
            y = ay + dy
            if not 0 <= y < H:
                continue
            row = (dy + HALF) * PATCH
            for dx in range(-HALF, HALF + 1):
                x = ax + dx
                if not 0 <= x < W:
                    continue
                t, cell = y * W + x, row + dx + HALF
                ter = terrain[t]
                tc = CH_TERRAIN - 1 if ter == T_UNKNOWN else ter - 1
                ch[(C_TERRAIN + tc) * cs + cell] = 1
                b = self.bld_id[t]
                if b >= 0:
                    bt = self.bld_type[t]
                    if self.bld_team[t] == team:
                        ch[(C_OWN + bt) * cs + cell] = 1
                    elif bt != MINE:
                        ch[(C_ENEMY + bt) * cs + cell] = 1
                if self.bot_id[t] >= 0:
                    own = self.bot_team[t] == team
                    ch[((C_OWN if own else C_ENEMY) + BUILDER) * cs + cell] = 1
                    ch[(C_OCC + (0 if own else 1)) * cs + cell] = 1
                sr = self.seen_round[t]
                ch[C_STALE * cs + cell] = 255 if sr < 0 else min(255, rnd - sr)
                vr = self.visit_round[t]
                ch[C_VISIT * cs + cell] = 255 if vr < 0 else min(255, rnd - vr)
                hr = self.hit_round[t]
                ch[C_HIT * cs + cell] = HIT_CAP if hr < 0 else min(HIT_CAP, rnd - hr)
                ch[C_MINE * cs + cell] = self.emine[t]
                if b >= 0:
                    d = self.bld_dir[t]
                    if 0 <= d < CH_DIR:
                        ch[(C_DIR + d) * cs + cell] = 1
                    ch[C_HP * cs + cell] = self.bld_hp[t]
                    ar = self.attr_seen_round[t]
                    ch[C_ATTR_STALE * cs + cell] = 255 if ar < 0 else min(255, rnd - ar)
                if self.bot_id[t] >= 0:
                    ch[(C_HP + 1) * cs + cell] = self.bot_hp[t]
                    ar = self.attr_seen_round[t]
                    ch[(C_ATTR_STALE + 1) * cs + cell] = 255 if ar < 0 else min(255, rnd - ar)

        # Possible / confirmed reach of remembered friendly and enemy turrets.
        # These are not exact can_fire answers (ammo/cooldowns/hidden mines).
        own_reaches_enemy_core = own_core_confirmed = False
        enemy_reaches_own_core = enemy_core_confirmed = False
        seen_turrets = set()
        delta = (None, (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
        for t, tid in enumerate(self.bld_id):
            bt, d = self.bld_type[t], self.bld_dir[t]
            if tid < 0 or d <= 0 or d > 8:
                continue
            if bt not in (GUNNER, SENTINEL, SPIKE) or tid in seen_turrets:
                continue
            seen_turrets.add(tid)
            tx, ty = t % W, t // W
            vx, vy = delta[d]
            if bt == SPIKE:
                targets = [(tx + xx, ty + yy) for yy in (-1, 0, 1) for xx in (-1, 0, 1) if xx or yy]
            else:
                length = ((3 if d in (1, 3, 5, 7) else 2) if bt == GUNNER
                          else (5 if d in (1, 3, 5, 7) else 4))
                targets = [(tx + vx * k, ty + vy * k) for k in range(1, length + 1)]
            for x, y in targets:
                if not (0 <= x < W and 0 <= y < H):
                    continue
                friendly = self.bld_team[t] == team
                confirmed = self.attr_seen_round[t] == rnd and self.seen_round[t] == rnd
                blocked = False
                if bt in (GUNNER, SENTINEL):
                    steps = max(abs(x - tx), abs(y - ty))
                    sx = 0 if x == tx else (1 if x > tx else -1)
                    sy = 0 if y == ty else (1 if y > ty else -1)
                    for k in range(1, steps):
                        q = (ty + sy * k) * W + tx + sx * k
                        fresh = self.seen_round[q] == rnd
                        if bt == GUNNER and not fresh:
                            confirmed = False
                        if bt == GUNNER and (terrain[q] in (T_WALL, T_VOID) or
                                             (fresh and (self.bld_id[q] >= 0 or self.bot_id[q] >= 0))):
                            blocked = True
                            break
                if blocked:
                    continue
                target_t = y * W + x
                if self.bld_id[target_t] >= 0 and self.bld_type[target_t] == CORE:
                    target_friendly = self.bld_team[target_t] == team
                    core_confirmed = confirmed and self.seen_round[target_t] == rnd
                    if friendly and not target_friendly:
                        own_reaches_enemy_core = True
                        own_core_confirmed |= core_confirmed
                    elif not friendly and target_friendly:
                        enemy_reaches_own_core = True
                        enemy_core_confirmed |= core_confirmed
                dx, dy = x - ax, y - ay
                if abs(dx) <= HALF and abs(dy) <= HALF:
                    cell = (dy + HALF) * PATCH + dx + HALF
                    ch[(C_THREAT + (0 if friendly else 1)) * cs + cell] = 1
                    if confirmed:
                        ch[(C_THREAT + (2 if friendly else 3)) * cs + cell] = 1

        # Rays originate on boundary cells and begin one step outside each.
        for dy in range(-HALF, HALF + 1):
            for dx in range(-HALF, HALF + 1):
                if abs(dx) != HALF and abs(dy) != HALF:
                    continue
                v = [0] * CH_RAY
                bx, by = ax + dx, ay + dy
                if 0 <= bx < W and 0 <= by < H:
                    seen_b, seen_u, step = set(), set(), 0
                    for px, py in ray_path(bx, by, dx, dy):
                        if not (0 <= px < W and 0 <= py < H):
                            break
                        step += 1
                        t = py * W + px
                        ter = terrain[t]
                        if v[0] == 0 and ter in (T_WALL, T_VOID):
                            v[0] = step
                        if v[1] == 0 and ter == T_UNKNOWN:
                            v[1] = step
                        if ter == T_ORE_TI:
                            v[2] += 1
                        if ter == T_ORE_VO:
                            v[3] += 1
                        if ter == T_UNKNOWN:
                            v[10] += 1
                        bid = self.bld_id[t]
                        if bid >= 0 and bid not in seen_b:
                            seen_b.add(bid)
                            own = self.bld_team[t] == team
                            v[4 if own else 5] += 1
                            bt = self.bld_type[t]
                            if bt in UNIT_TYPES:
                                v[6 if own else 7] += 1
                            if bt == HARVESTER:
                                v[8 if own else 9] += 1
                        uid = self.bot_id[t]
                        if uid >= 0 and uid not in seen_u:
                            seen_u.add(uid)
                            v[6 if self.bot_team[t] == team else 7] += 1
                cell = (dy + HALF) * PATCH + dx + HALF
                for k, value in enumerate(v):
                    ch[(C_RAY + k) * cs + cell] = value

        scalars = [0] * ACTOR_SCALARS
        if typ < 6:
            scalars[typ] = 1
        scalars[6] = 100 * hp // maxhp if maxhp else 0
        scalars[7:10] = [acd, mcd, rnd]
        scalars[10:15] = ad.team_resources(team)
        scalars[15:20] = ad.team_resources(1 - team)
        # rounds since our core last raised the under-attack alarm (team bus), cap 15
        scalars[15] = 15 if self.core_alarm_round < 0 else min(15, rnd - self.core_alarm_round)
        for i in range(16):                      # as int32, like the engine (u32 reports set bit 31)
            v = ad.store(i) & 0xFFFFFFFF
            scalars[20 + i] = v - (1 << 32) if v >= (1 << 31) else v
        scalars[ACTOR_COSTS:ACTOR_COSTS + 10] = ad.costs()
        scalars[36:40] = [int(own_reaches_enemy_core), int(own_core_confirmed),
                          int(enemy_reaches_own_core), int(enemy_core_confirmed)]
        if typ == BUILDER:                       # planned conveyor paths (builders only)
            plan = self.plan_paths()
            base = C_PATH * cs
            for dy in range(-HALF, HALF + 1):
                y = ay + dy
                if not 0 <= y < H:
                    continue
                row = base + (dy + HALF) * PATCH + HALF
                for dx in range(-HALF, HALF + 1):
                    x = ax + dx
                    if 0 <= x < W:
                        v = plan[y * W + x]
                        if v:
                            ch[row + dx] = v
        return out, scalars


def _team_index(team):
    from fcode import Team
    if team == Team.A:
        return 0
    if team == Team.B:
        return 1
    return int(team)


_TYPE_INDEX = None


def init_type_index():
    global _TYPE_INDEX
    from fcode import EntityType as E
    _TYPE_INDEX = {
        E.BUILDER_BOT: BUILDER, E.CORE: CORE, E.GUNNER: GUNNER,
        E.SENTINEL: SENTINEL, E.LAUNCHER: LAUNCHER, E.SPIKE: SPIKE,
        E.CONVEYOR: CONVEYOR, E.SPLITTER: SPLITTER, E.HARVESTER: HARVESTER,
        E.BARRIER: BARRIER, E.MINE: MINE,
    }


class ControllerAdapter:
    def __init__(self, ct):
        if _TYPE_INDEX is None:
            init_type_index()
        self.ct = ct
        self.W, self.H = ct.get_map_width(), ct.get_map_height()
        self._team = _team_index(ct.get_team())
        from fcode import Environment as E
        self.env = {E.EMPTY: T_EMPTY, E.WALL: T_WALL, E.ORE_TITANIUM: T_ORE_TI,
                    E.ORE_VOLTIUM: T_ORE_VO, E.VOID: T_VOID}

    def team(self):
        return self._team

    def current_round(self):
        return self.ct.get_current_round()

    def visible_tiles(self):
        return ((p.x, p.y) for p in self.ct.get_nearby_tiles())

    def tile_terrain(self, x, y):
        from fcode import Position
        return self.env[self.ct.get_tile_env(Position(x, y))]

    def building_id(self, x, y):
        from fcode import Position
        return self.ct.get_tile_building_id(Position(x, y))

    def builder_id(self, x, y):
        from fcode import Position
        return self.ct.get_tile_builder_bot_id(Position(x, y))

    def entity_type(self, eid):
        return _TYPE_INDEX[self.ct.get_entity_type(eid)]

    def entity_team(self, eid):
        return _team_index(self.ct.get_team(eid))

    def entity_direction(self, eid):
        from fcode import Direction as D
        try:
            value = self.ct.get_direction(eid)
        except Exception:
            # C++ memory records non-facing entities as centre/unavailable.
            return 0
        dirs = (D.CENTRE, D.NORTH, D.NORTHEAST, D.EAST, D.SOUTHEAST,
                D.SOUTH, D.SOUTHWEST, D.WEST, D.NORTHWEST)
        for i, d in enumerate(dirs):
            if value == d:
                return i
        return -1

    def entity_hp_percent(self, eid):
        maximum = self.ct.get_max_hp(eid)
        return (100 * self.ct.get_hp(eid)) // maximum if maximum else 0

    def actor_id(self):
        return self.ct.get_id()

    def actor_pos(self):
        p = self.ct.get_position()
        return p.x, p.y

    def actor_state(self):
        ct = self.ct
        return (_TYPE_INDEX[ct.get_entity_type()], ct.get_hp(), ct.get_max_hp(),
                ct.get_action_cooldown(), ct.get_move_cooldown())

    def team_resources(self, team):
        if team != self._team:
            return 0, 0, 0, 0, 0
        ct = self.ct
        # The production controller exposes only our own global economy.
        return (ct.get_global_resources(), ct.get_global_voltium(), ct.get_global_ammo(),
                int(round(ct.get_scale_percent())), ct.get_unit_count())

    def costs(self):
        ct = self.ct
        return [ct.get_conveyor_cost(), ct.get_splitter_cost(), ct.get_harvester_cost(),
                ct.get_barrier_cost(), ct.get_gunner_cost(), ct.get_sentinel_cost(),
                ct.get_launcher_cost(), ct.get_spike_cost(), ct.get_mine_cost(),
                ct.get_builder_bot_cost()]

    def store(self, i):
        return self.ct.read_store(i)

    def can_attack(self, x, y):
        """Builder attack probe (official can_fire: needs a building, hidden enemy mines included)."""
        from fcode import Position
        try:
            return bool(self.ct.can_fire(Position(x, y)))
        except Exception:
            return False


class ActorEncoder:
    """Keep one instance on Player; map memory is independent per unit ID."""
    def __init__(self):
        self.memories = {}
        self.size = None

    def encode(self, ct):
        """observe(ct), then the actor observation."""
        memory, ad = self.observe(ct)
        return memory.encode(ad)

    def observe(self, ct):
        """Memory and team report bus only (every unit, every turn; spikes too)."""
        ad = ControllerAdapter(ct)
        key = (ad.W, ad.H, ad.team())
        if key != self.size:
            self.memories.clear()
            self.size = key
        uid = ct.get_id()
        memory = self.memories.get(uid)
        if memory is None:
            pit = ct.get_pit()
            memory = ObsMemory(ad.W, ad.H, ad.team(), -1 if pit is None else (0 if pit[0] == "x" else 1))
            self.memories[uid] = memory
        # team report bus: merge the store snapshot, see, post (one write per slot group)
        rnd = ct.get_current_round()
        memory.merge_store([ct.read_store(i) for i in range(16)], rnd)
        memory.update(ad)
        for slot, word in memory.next_reports(uid, rnd):
            ct.write_store(slot, word)
        return memory, ad

    def forget(self, entity_id):
        self.memories.pop(entity_id, None)
