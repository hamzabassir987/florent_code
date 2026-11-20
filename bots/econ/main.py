"""Economy-first bot.

Builder bots remember the map they have seen, claim ore tiles, build a
harvester on each and lay a shortest conveyor chain (BFS) from the harvester
into the core or into an existing friendly conveyor. Voltium is converted to
ammo by the core; spikes and gunners are built around the core when enemies
show up there, and late in the game spare builders raid enemy conveyors.

Store slots:
  0  core position (packed)
  1  number of harvesters built (approx, counter)
  2  threat position near our core (packed, 0 = none)
  3  enemy core position if seen (packed, 0 = unknown)
  4  enemy conveyor seen (packed raid target, 0 = none)
  5-14  ore claims, one per builder slot (id % 10)
"""

from collections import deque

from fcode import Controller, Direction, EntityType, Environment, Position, Team

CARDINALS = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
DIRS8 = tuple(d for d in Direction if d != Direction.CENTRE)
CARD_DELTA = tuple(d.delta() for d in CARDINALS)

S_CORE = 0
S_HARV = 1
S_THREAT = 2
S_ECORE = 3
S_RAID = 4
S_CLAIM0 = 5
N_CLAIM = 10

# env codes in memory
U, EMPTY, WALL, TI, VO, VOID = 0, 1, 2, 3, 4, 5
ENV_CODE = {
    Environment.EMPTY: EMPTY,
    Environment.WALL: WALL,
    Environment.ORE_TITANIUM: TI,
    Environment.ORE_VOLTIUM: VO,
    Environment.VOID: VOID,
}

# building codes in memory (0 = none / unknown)
B_NONE, B_OWN_CONV, B_OWN_OTHER, B_ENEMY_CONV, B_ENEMY_OTHER, B_CORE, B_OWN_HARV = 0, 1, 2, 3, 4, 5, 6

WALKABLE_TYPES = (EntityType.CONVEYOR, EntityType.SPLITTER)
TURRETS = (EntityType.GUNNER, EntityType.SENTINEL, EntityType.LAUNCHER, EntityType.SPIKE)


def pack(x: int, y: int) -> int:
    return ((x + 1) << 8) | (y + 1)


def unpack(v: int):
    if v <= 0:
        return None
    return Position((v >> 8) - 1, (v & 0xFF) - 1)


class Player:
    def __init__(self):
        self.init = False
        # core
        self.spawned = 0
        # builder
        self.jobs = []  # list of (x, y, Direction) conveyors to build
        self.chain_h = None  # harvester the current jobs serve
        self.chain_fail = False
        self.replans = 0
        self.jfail = 0
        self.target = None  # ore tile we intend to harvest (x, y)
        self.mode = "econ"
        self.stuck = 0
        self.last = None
        self.explore = None
        self.fail = {}
        self.orphan_tries = {}
        self.raid_target = None

    # ------------------------------------------------------------------
    def _setup(self, ct: Controller):
        self.init = True
        self.W = ct.get_map_width()
        self.H = ct.get_map_height()
        self.me = ct.get_team()
        n = self.W * self.H
        self.env = bytearray(n)
        self.bld = bytearray(n)
        self.bdir = {}  # idx -> Direction for own conveyors
        self.seen_round = [-1] * n
        self.core = None
        self.ecore = None

    def run(self, ct: Controller) -> None:
        if not self.init:
            self._setup(ct)
        t = ct.get_entity_type()
        try:
            if t == EntityType.CORE:
                self.core_turn(ct)
            elif t == EntityType.BUILDER_BOT:
                self.builder_turn(ct)
            elif t == EntityType.GUNNER:
                self.gunner_turn(ct)
            elif t == EntityType.SENTINEL:
                self.sentinel_turn(ct)
            elif t == EntityType.SPIKE:
                self.spike_turn(ct)
        except Exception as e:  # never let a unit die from a bug
            import sys

            print("ERR", t, repr(e), file=sys.stderr)

    # ------------------------------------------------------------------
    # memory
    def scan(self, ct: Controller, full_env=True):
        W = self.W
        env = self.env
        bld = self.bld
        me = self.me
        for p in ct.get_nearby_tiles():
            i = p.y * W + p.x
            if env[i] == U:
                env[i] = ENV_CODE.get(ct.get_tile_env(p), EMPTY)
            if env[i] == WALL or env[i] == VOID:
                continue
            bid = ct.get_tile_building_id(p)
            if bid is None:
                bld[i] = B_NONE
                continue
            et = ct.get_entity_type(bid)
            own = ct.get_team(bid) == me
            if et == EntityType.CORE:
                if own:
                    bld[i] = B_CORE
                else:
                    bld[i] = B_ENEMY_OTHER
                    if self.ecore is None:
                        self.ecore = ct.get_position(bid)
            elif et in WALKABLE_TYPES:
                bld[i] = B_OWN_CONV if own else B_ENEMY_CONV
                if own:
                    self.bdir[i] = ct.get_direction(bid)
            elif own and et == EntityType.HARVESTER:
                bld[i] = B_OWN_HARV
            else:
                bld[i] = B_OWN_OTHER if own else B_ENEMY_OTHER

    def inb(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def walkable(self, i):
        e = self.env[i]
        if e == WALL or e == VOID:
            return False
        b = self.bld[i]
        return b == B_NONE or b == B_OWN_CONV or b == B_ENEMY_CONV

    def buildable(self, i):
        e = self.env[i]
        return (e == EMPTY or e == U) and self.bld[i] == B_NONE

    # ------------------------------------------------------------------
    # core
    def core_turn(self, ct: Controller):
        pos = ct.get_position()
        r = ct.get_current_round()
        if r == 0:
            ct.write_store(S_CORE, pack(pos.x, pos.y))
        # threat detection
        me = self.me
        threat = None
        best = 10**9
        cx, cy = pos.x, pos.y
        for uid in ct.get_nearby_entities():
            if ct.get_team(uid) == me:
                continue
            et = ct.get_entity_type(uid)
            if et == EntityType.CORE:
                ep = ct.get_position(uid)
                ct.write_store(S_ECORE, pack(ep.x, ep.y))
                continue
            if et == EntityType.CONVEYOR or et == EntityType.SPLITTER:
                continue
            p = ct.get_position(uid)
            d = (p.x - cx) ** 2 + (p.y - cy) ** 2
            if et in TURRETS or et == EntityType.BUILDER_BOT:
                if d < best:
                    best = d
                    threat = p
        ct.write_store(S_THREAT, pack(threat.x, threat.y) if threat else 0)

        # ammo: convert all voltium
        v = ct.get_global_voltium()
        if v > 0 and ct.get_global_ammo() < 400 and ct.can_convert_ammo(v):
            ct.convert_ammo(v)

        if ct.get_action_cooldown() != 0:
            return
        if ct.get_unit_count() >= 48:
            return
        ti = ct.get_global_resources()
        cost = ct.get_builder_bot_cost()
        area = self.W * self.H
        harv = ct.read_store(S_HARV)
        cap = min(14, 4 + area // 100)
        want = min(cap, 3 + harv // 2)
        if threat is not None and self.spawned < want + 2:
            want = self.spawned + 1
        if self.spawned >= want:
            return
        reserve = 0 if self.spawned < 2 else 20 + 10 * self.spawned
        if ti < cost + reserve:
            return
        # spawn toward map centre
        mid = Position(self.W // 2, self.H // 2)
        best_p = None
        best_d = 10**9
        for dx in range(-1, 3):
            for dy in range(-1, 3):
                p = Position(cx + dx, cy + dy)
                if 0 <= dx <= 1 and 0 <= dy <= 1:
                    continue
                if not self.inb(p.x, p.y):
                    continue
                if ct.can_spawn(p):
                    d = p.distance_squared(mid)
                    if d < best_d:
                        best_d = d
                        best_p = p
        if best_p is not None:
            ct.spawn_builder(best_p)
            self.spawned += 1

    # ------------------------------------------------------------------
    # builder
    def builder_turn(self, ct: Controller):
        if self.core is None:
            c = unpack(ct.read_store(S_CORE))
            if c is not None:
                self.core = c
                for dx in (0, 1):
                    for dy in (0, 1):
                        i = (c.y + dy) * self.W + c.x + dx
                        self.bld[i] = B_CORE
                        self.env[i] = EMPTY
        self.scan(ct)
        if self.ecore is None:
            e = unpack(ct.read_store(S_ECORE))
            if e is not None:
                self.ecore = e
        if self.ecore is not None:
            ct.write_store(S_ECORE, pack(self.ecore.x, self.ecore.y))
        pos = ct.get_position()
        if self.last == pos:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last = pos

        if self.core is None:
            return

        # heal adjacent damaged friendly buildings cheaply first if nothing else
        threat = unpack(ct.read_store(S_THREAT))
        if threat is not None and self.defend(ct, pos, threat):
            return

        if self.jobs:
            self.do_jobs(ct, pos)
            self.claim(ct)
            return

        # opportunistic: attack adjacent enemy building
        if self.attack_adjacent(ct, pos):
            return

        if self.target is not None:
            i = self.target[1] * self.W + self.target[0]
            if self.bld[i] != B_NONE or self.fail.get(self.target, 0) > 30:
                self.target = None
        if self.target is None and ct.get_current_round() % 10 == ct.get_id() % 10:
            o = self.find_orphan(pos)
            if o is not None:
                i = o[1] * self.W + o[0]
                self.orphan_tries[i] = self.orphan_tries.get(i, 0) + 1
                self.replans = 0
                self.plan_chain(*o)
                if self.jobs:
                    self.do_jobs(ct, pos)
                    return
        if self.target is None:
            self.target = self.pick_ore(ct, pos)
        self.claim(ct)
        if self.target is not None:
            self.go_harvest(ct, pos)
            return
        # nothing to harvest: raid or explore
        if ct.get_current_round() > 120 and self.raid(ct, pos):
            return
        self.do_explore(ct, pos)

    def claim(self, ct):
        slot = S_CLAIM0 + ct.get_id() % N_CLAIM
        if self.target is not None:
            ct.write_store(slot, pack(self.target[0], self.target[1]))
        else:
            ct.write_store(slot, 0)

    def claimed(self, ct):
        mine = S_CLAIM0 + ct.get_id() % N_CLAIM
        out = set()
        for s in range(S_CLAIM0, S_CLAIM0 + N_CLAIM):
            if s == mine:
                continue
            v = ct.read_store(s)
            if v:
                p = unpack(v)
                out.add((p.x, p.y))
        return out

    # BFS over walkable tiles from pos; returns dist dict-ish via generator
    def bfs_first_step(self, pos, goal_fn, max_nodes=1200, unknown_ok=True):
        """BFS from pos; goal_fn(i) -> bool on tile indices.  Returns (goal_i, first_dir) or None."""
        W, H = self.W, self.H
        start = pos.y * W + pos.x
        if goal_fn(start):
            return start, Direction.CENTRE
        prev = {start: -1}
        dq = deque([start])
        env = self.env
        n = 0
        while dq:
            i = dq.popleft()
            n += 1
            if n > max_nodes:
                break
            x, y = i % W, i // W
            for k in range(4):
                dx, dy = CARD_DELTA[k]
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    continue
                j = ny * W + nx
                if j in prev:
                    continue
                if env[j] == U:
                    if not unknown_ok:
                        continue
                elif not self.walkable(j):
                    continue
                prev[j] = i
                if goal_fn(j):
                    c = j
                    while prev[c] != start:
                        c = prev[c]
                    ddx, ddy = c % W - pos.x, c // W - pos.y
                    for kk in range(4):
                        if CARD_DELTA[kk] == (ddx, ddy):
                            return j, CARDINALS[kk]
                    return j, Direction.CENTRE
                dq.append(j)
        return None

    def step(self, ct, pos, d):
        if d == Direction.CENTRE or ct.get_move_cooldown() != 0:
            return False
        if ct.can_move(d):
            ct.move(d)
            return True
        # blocked (probably a bot); try sideways
        if self.stuck >= 2:
            for d2 in (d.rotate_left().rotate_left(), d.rotate_right().rotate_right()):
                if ct.can_move(d2):
                    ct.move(d2)
                    return True
        return False

    def move_adjacent_to(self, ct, pos, tx, ty):
        """Walk to a tile orthogonally adjacent to (tx, ty)."""
        W = self.W
        goals = set()
        for dx, dy in CARD_DELTA:
            nx, ny = tx + dx, ty + dy
            if self.inb(nx, ny):
                goals.add(ny * W + nx)
        res = self.bfs_first_step(pos, lambda i: i in goals)
        if res is None:
            return False
        return self.step(ct, pos, res[1])

    def pick_ore(self, ct, pos):
        W = self.W
        claimed = self.claimed(ct)
        env = self.env
        bld = self.bld
        harv = ct.read_store(S_HARV)
        best = None
        bestd = 10**9
        core = self.core
        for i in range(W * self.H):
            e = env[i]
            if (e == TI or e == VO) and bld[i] == B_NONE:
                x, y = i % W, i // W
                if (x, y) in claimed or self.fail.get((x, y), 0) > 30:
                    continue
                # distance from bot plus distance to core (conveyor cost)
                d = abs(x - pos.x) + abs(y - pos.y) + abs(x - core.x) + abs(y - core.y)
                if e == VO and harv >= 2:
                    d -= 6
                if d < bestd:
                    bestd = d
                    best = (x, y)
        return best

    def go_harvest(self, ct, pos):
        tx, ty = self.target
        tp = Position(tx, ty)
        if abs(tx - pos.x) + abs(ty - pos.y) == 1:
            if ct.get_action_cooldown() != 0:
                return
            hc = ct.get_harvester_cost()
            if ct.get_global_resources() < hc:
                return
            if ct.can_build_harvester(tp):
                self.plan_chain(tx, ty)
                need = hc + len(self.jobs) * ct.get_conveyor_cost()
                if len(self.jobs) > 30 or (not self.jobs and self.chain_fail):
                    self.fail[self.target] = 99
                    self.target = None
                    self.jobs = []
                    return
                if ct.get_global_resources() < need:
                    self.jobs = []
                    return
                ct.build_harvester(tp)
                ct.write_store(S_HARV, ct.read_store(S_HARV) + 1)
                self.bld[ty * self.W + tx] = B_OWN_HARV
                self.replans = 0
                self.target = None
                self.claim(ct)
                if self.jobs:
                    self.do_jobs(ct, pos)
            else:
                self.fail[self.target] = self.fail.get(self.target, 0) + 10
            return
        if not self.move_adjacent_to(ct, pos, tx, ty):
            self.fail[self.target] = self.fail.get(self.target, 0) + 1

    def good_conveyors(self):
        """Set of own conveyor tiles whose chain reaches our core."""
        W, H = self.W, self.H
        bld = self.bld
        state = {}  # i -> True/False
        for i0 in range(W * H):
            if bld[i0] != B_OWN_CONV or i0 in state:
                continue
            trail = []
            onpath = set()
            i = i0
            res = False
            while True:
                if i in state:
                    res = state[i]
                    break
                b = bld[i]
                if b == B_CORE:
                    res = True
                    break
                if b != B_OWN_CONV or i in onpath:
                    res = False
                    break
                d = self.bdir.get(i)
                if d is None:
                    res = False
                    break
                trail.append(i)
                onpath.add(i)
                dx, dy = d.delta()
                x, y = i % W + dx, i // W + dy
                if not (0 <= x < W and 0 <= y < H):
                    res = False
                    break
                i = y * W + x
            for t in trail:
                state[t] = res
        return {i for i, v in state.items() if v}

    def plan_chain(self, hx, hy):
        """Cheapest conveyor chain from a harvester at (hx, hy) into the network."""
        import heapq

        W, H = self.W, self.H
        h = hy * W + hx
        self.chain_h = (hx, hy)
        self.jfail = 0
        bld = self.bld
        env = self.env
        good = self.good_conveyors()

        def sink_ok(j, from_i):
            b = bld[j]
            if b == B_CORE:
                return True
            if j in good:
                dx, dy = self.bdir[j].delta()
                return (j % W + dx) + (j // W + dy) * W != from_i
            return False

        for dx, dy in CARD_DELTA:
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < W and 0 <= ny < H and sink_ok(ny * W + nx, h):
                self.jobs = []
                self.chain_fail = False
                return

        def cost(j):
            b = bld[j]
            e = env[j]
            if e == WALL or e == VOID:
                return None
            if b == B_NONE:
                if e == EMPTY:
                    return 2
                if e == U:
                    return 3
                return 9  # ore tile: wasteful but allowed
            if b == B_OWN_CONV and j not in good:
                return 2
            return None

        dist = {}
        prev = {}
        pq = []
        for dx, dy in CARD_DELTA:
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < W and 0 <= ny < H:
                j = ny * W + nx
                c = cost(j)
                if c is not None:
                    dist[j] = c
                    prev[j] = -1
                    heapq.heappush(pq, (c, j))
        found = None
        sink = None
        while pq:
            dcur, i = heapq.heappop(pq)
            if dcur > dist.get(i, 1 << 30):
                continue
            x, y = i % W, i // W
            for dx, dy in CARD_DELTA:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                j = ny * W + nx
                if sink_ok(j, i):
                    found = i
                    sink = j
                    break
            if found is not None:
                break
            for dx, dy in CARD_DELTA:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                j = ny * W + nx
                if j == h:
                    continue
                c = cost(j)
                if c is None:
                    continue
                nd = dcur + c
                if nd < dist.get(j, 1 << 30):
                    dist[j] = nd
                    prev[j] = i
                    heapq.heappush(pq, (nd, j))
        if found is None:
            self.jobs = []
            self.chain_fail = True
            return
        self.chain_fail = False
        path = [found]
        while prev[path[-1]] != -1:
            path.append(prev[path[-1]])
        path.reverse()
        path.append(sink)
        jobs = []
        for k in range(len(path) - 1):
            a, b = path[k], path[k + 1]
            d = Position(a % W, a // W).direction_to(Position(b % W, b // W))
            jobs.append((a % W, a // W, d))
        self.jobs = jobs

    def replan(self):
        self.jobs = []
        self.replans += 1
        if self.chain_h is not None and self.replans <= 4:
            self.plan_chain(*self.chain_h)

    def find_orphan(self, pos):
        """Own harvester in memory with no adjacent own conveyor/core."""
        W, H = self.W, self.H
        bld = self.bld
        good = self.good_conveyors()
        best = None
        bestd = 10**9
        for i in range(W * H):
            if bld[i] != B_OWN_HARV:
                continue
            x, y = i % W, i // W
            ok = False
            for dx, dy in CARD_DELTA:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    j = ny * W + nx
                    if bld[j] == B_CORE or j in good:
                        ok = True
                        break
            if ok or self.orphan_tries.get(i, 0) >= 2:
                continue
            d = abs(x - pos.x) + abs(y - pos.y)
            if d < bestd:
                bestd = d
                best = (x, y)
        return best

    def do_jobs(self, ct, pos):
        W = self.W
        # drop finished jobs
        while self.jobs:
            x, y, d = self.jobs[0]
            i = y * W + x
            b = self.bld[i]
            if b == B_OWN_CONV and self.bdir.get(i) == d:
                self.jobs.pop(0)
                continue
            if b != B_NONE and b != B_OWN_CONV:
                self.replan()
                return
            break
        if not self.jobs:
            return
        x, y, d = self.jobs[0]
        tp = Position(x, y)
        dist = abs(x - pos.x) + abs(y - pos.y)
        if dist == 1:
            if ct.get_action_cooldown() != 0:
                return
            if ct.get_global_resources() < ct.get_conveyor_cost():
                return
            i = y * W + x
            if self.bld[i] == B_OWN_CONV and ct.can_destroy(tp):
                ct.destroy(tp)
                self.bld[i] = B_NONE
            if ct.can_build_conveyor(tp, d):
                ct.build_conveyor(tp, d)
                i = y * W + x
                self.bld[i] = B_OWN_CONV
                self.bdir[i] = d
                self.jobs.pop(0)
                self.jfail = 0
            else:
                self.jfail += 1
                if self.jfail > 4:
                    self.replan()
            return
        if dist == 0:
            # step off, preferably onto the next job tile
            order = []
            if len(self.jobs) > 1:
                nx, ny, _ = self.jobs[1]
                for k, (dx, dy) in enumerate(CARD_DELTA):
                    if pos.x + dx == nx and pos.y + dy == ny:
                        order.append(CARDINALS[k])
            order.extend(CARDINALS)
            for dd in order:
                if ct.get_move_cooldown() == 0 and ct.can_move(dd):
                    ct.move(dd)
                    return
            return
        if self.move_adjacent_to(ct, pos, x, y):
            self.jfail = 0
        else:
            self.jfail += 1
            if self.jfail > 8:
                self.replan()

    # ------------------------------------------------------------------
    def attack_adjacent(self, ct, pos):
        if ct.get_action_cooldown() != 0:
            return False
        me = self.me
        for d in CARDINALS:
            p = pos.add(d)
            if not self.inb(p.x, p.y):
                continue
            bid = ct.get_tile_building_id(p)
            if bid is not None and ct.get_team(bid) != me and ct.get_entity_type(bid) != EntityType.CORE:
                if ct.get_global_resources() > 40 and ct.can_fire(p):
                    ct.fire(p)
                    return True
        return False

    def defend(self, ct, pos, threat):
        core = self.core
        # only respond if close to home
        if pos.distance_squared(core) > 64:
            return False
        me = self.me
        if ct.get_action_cooldown() == 0:
            # spike next to enemy units adjacent to us
            enemies = []
            for uid in ct.get_nearby_entities():
                if ct.get_team(uid) != me:
                    et = ct.get_entity_type(uid)
                    if et in TURRETS or et == EntityType.BUILDER_BOT:
                        enemies.append(ct.get_position(uid))
            if not enemies:
                return False
            # existing own spike covering?
            for ep in enemies:
                covered = False
                for bid in ct.get_nearby_buildings():
                    if (
                        ct.get_team(bid) == me
                        and ct.get_entity_type(bid) == EntityType.SPIKE
                        and ct.get_position(bid).distance_squared(ep) <= 2
                    ):
                        covered = True
                        break
                if covered:
                    continue
                if ct.get_global_resources() >= ct.get_spike_cost():
                    for d in CARDINALS:
                        s = pos.add(d)
                        if s.distance_squared(ep) <= 2 and self.inb(s.x, s.y) and ct.can_build_spike(s):
                            ct.build_spike(s)
                            return True
            # attack adjacent enemy buildings
            if self.attack_adjacent(ct, pos):
                return True
            # heal adjacent damaged friendly
            for d in CARDINALS:
                s = pos.add(d)
                if self.inb(s.x, s.y) and ct.can_heal(s):
                    bid = ct.get_tile_building_id(s)
                    if bid is not None and ct.get_hp(bid) < ct.get_max_hp(bid):
                        ct.heal(s)
                        return True
        # approach nearest enemy within core area
        if threat.distance_squared(core) <= 50 and pos.distance_squared(threat) > 2:
            return self.move_adjacent_to(ct, pos, threat.x, threat.y)
        return False

    def raid(self, ct, pos):
        """Go break enemy conveyors (cheap, cuts their delivery)."""
        W = self.W
        if ct.get_global_resources() < 60:
            return False
        # find nearest known enemy conveyor
        best = None
        bestd = 10**9
        for i in range(W * self.H):
            if self.bld[i] == B_ENEMY_CONV or self.bld[i] == B_ENEMY_OTHER:
                x, y = i % W, i // W
                d = abs(x - pos.x) + abs(y - pos.y)
                if self.bld[i] == B_ENEMY_OTHER:
                    d += 4
                if d < bestd:
                    bestd = d
                    best = (x, y)
        if best is None:
            if self.ecore is not None:
                if pos.distance_squared(self.ecore) > 20:
                    res = self.bfs_first_step(
                        pos, lambda i: (i % W - self.ecore.x) ** 2 + (i // W - self.ecore.y) ** 2 <= 20
                    )
                    if res:
                        return self.step(ct, pos, res[1])
            return False
        bx, by = best
        if abs(bx - pos.x) + abs(by - pos.y) == 1:
            p = Position(bx, by)
            if ct.get_action_cooldown() == 0 and ct.can_fire(p):
                ct.fire(p)
            return True
        return self.move_adjacent_to(ct, pos, bx, by)

    def do_explore(self, ct, pos):
        W, H = self.W, self.H
        env = self.env
        res = self.bfs_first_step(pos, lambda i: env[i] == U, max_nodes=2000)
        if res is not None:
            self.step(ct, pos, res[1])
            return
        # everything seen: go guess enemy core (rotational symmetry)
        if self.core is not None:
            g = Position(W - 1 - self.core.x, H - 1 - self.core.y)
            res = self.bfs_first_step(pos, lambda i: (i % W - g.x) ** 2 + (i // W - g.y) ** 2 <= 9)
            if res:
                self.step(ct, pos, res[1])

    # ------------------------------------------------------------------
    # turrets
    def gunner_turn(self, ct):
        t = ct.get_gunner_target()
        if t is not None:
            bid = ct.get_tile_building_id(t)
            uid = ct.get_tile_builder_bot_id(t)
            enemy = (bid is not None and ct.get_team(bid) != self.me) or (
                uid is not None and ct.get_team(uid) != self.me
            )
            if enemy and ct.can_fire(t):
                ct.fire(t)

    def sentinel_turn(self, ct):
        if ct.get_action_cooldown() != 0:
            return
        for p in ct.get_attackable_tiles():
            if not self.inb(p.x, p.y):
                continue
            uid = ct.get_tile_builder_bot_id(p)
            bid = ct.get_tile_building_id(p)
            if (uid is not None and ct.get_team(uid) != self.me) or (
                bid is not None and ct.get_team(bid) != self.me
            ):
                if ct.can_fire(p):
                    ct.fire(p)
                    return

    def spike_turn(self, ct):
        if ct.get_action_cooldown() != 0:
            return
        pos = ct.get_position()
        for d in DIRS8:
            p = pos.add(d)
            if not self.inb(p.x, p.y):
                continue
            uid = ct.get_tile_builder_bot_id(p)
            bid = ct.get_tile_building_id(p)
            if (uid is not None and ct.get_team(uid) != self.me) or (
                bid is not None and ct.get_team(bid) != self.me
            ):
                if ct.can_fire(p):
                    ct.fire(p)
                    return
