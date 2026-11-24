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
ACTOR_CH = CH_TERRAIN + CH_OWN + CH_ENEMY + CH_OCC + CH_STALE + CH_VISIT + CH_ATTR_STALE + CH_DIR + CH_HP + CH_THREAT + CH_RAY
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


class ObsMemory:
    def __init__(self, W, H, team):
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
        self.visit_round = [-1] * n
        self.attr_seen_round = [-1] * n

    def update(self, ad):
        r, W = ad.current_round(), self.W
        for x, y in ad.visible_tiles():
            t = y * W + x
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
            self.seen_round[t] = r

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
        for i in range(16):
            scalars[20 + i] = ad.store(i)
        scalars[ACTOR_COSTS:ACTOR_COSTS + 10] = ad.costs()
        scalars[36:40] = [int(own_reaches_enemy_core), int(own_core_confirmed),
                          int(enemy_reaches_own_core), int(enemy_core_confirmed)]
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


class ActorEncoder:
    """Keep one instance on Player; map memory is independent per unit ID."""
    def __init__(self):
        self.memories = {}
        self.size = None

    def encode(self, ct):
        ad = ControllerAdapter(ct)
        key = (ad.W, ad.H, ad.team())
        if key != self.size:
            self.memories.clear()
            self.size = key
        uid = ct.get_id()
        memory = self.memories.get(uid)
        if memory is None:
            memory = ObsMemory(ad.W, ad.H, ad.team())
            self.memories[uid] = memory
        memory.update(ad)
        return memory.encode(ad)

    def forget(self, entity_id):
        self.memories.pop(entity_id, None)
