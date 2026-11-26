"""Factorized action heads, production side. Mirrors rl/sim/florent_heads.cpp
exactly (rl/tests/heads_parity.py checks masks, choices and actions).

Each acting type has a mode head plus sub-heads used only for the chosen mode;
each head is its own masked categorical. Only the standard library and the
official controller API are used.
"""
import math

# head ids and sizes (checkpoint ABI, same order as the C++ enum)
(HC_SPAWN, HC_CONVERT, HB_MODE, HB_MOVE, HB_BTYPE, HB_BTILE, HB_BDIR, HB_ATTACK, HB_HEAL,
 HB_DESTROY, HG_MODE, HG_FIRE, HG_ROTATE, HS_MODE, HS_FIRE, HL_MODE, HL_BOT, HL_DEST) = range(18)
HEAD_SIZE = [13, 5, 6, 4, 9, 4, 8, 4, 4, 4, 3, 5, 8, 2, 5, 2, 8, 120]
P_CORE, P_BUILDER, P_GUNNER, P_SENTINEL, P_LAUNCHER = range(5)
POLICY_HEADS = [
    [HC_SPAWN, HC_CONVERT],
    [HB_MODE, HB_MOVE, HB_BTYPE, HB_BTILE, HB_BDIR, HB_ATTACK, HB_HEAL, HB_DESTROY],
    [HG_MODE, HG_FIRE, HG_ROTATE],
    [HS_MODE, HS_FIRE],
    [HL_MODE, HL_BOT, HL_DEST],
]

# engine enums
C, N, NE, E, SE, S, SW, W, NW = range(9)
DXY = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
CARD4 = (N, E, S, W)
BUILDER, CORE, GUNNER, SENTINEL, LAUNCHER, SPIKE = range(6)
CONVEYOR, SPLITTER, HARVESTER, BARRIER, MINE = range(6, 11)
MAX_LAUNCHERS = 3   # launcher cap (as florent_heads.hpp)
BUILD_ORDER = (CONVEYOR, SPLITTER, HARVESTER, GUNNER, SENTINEL, LAUNCHER, SPIKE, BARRIER, MINE)
DIRECTIONAL = frozenset((CONVEYOR, SPLITTER, GUNNER, SENTINEL))
CONVERT_PCT = (0, 10, 25, 50, 100)
TYPE_NAMES = ["BUILDER_BOT", "CORE", "GUNNER", "SENTINEL", "LAUNCHER", "SPIKE",
              "CONVEYOR", "SPLITTER", "HARVESTER", "BARRIER", "MINE"]
POLICY_OF_TYPE = {CORE: P_CORE, BUILDER: P_BUILDER, GUNNER: P_GUNNER, SENTINEL: P_SENTINEL, LAUNCHER: P_LAUNCHER}


def head_offset(policy, head):
    off = 0
    for h in POLICY_HEADS[policy]:
        if h == head:
            return off
        off += HEAD_SIZE[h]
    return -1


def policy_logit_count(policy):
    return sum(HEAD_SIZE[h] for h in POLICY_HEADS[policy])


# ---- controller adapter (official fcode types) -------------------------------
class Api:
    def __init__(self, ct):
        from fcode import Direction, EntityType, Position
        self.ct, self.P = ct, Position
        self.D = {}
        for d in Direction:
            dd = d.delta()
            if dd in DXY:
                self.D[DXY.index(dd)] = d
        self.T = {i: getattr(EntityType, n) for i, n in enumerate(TYPE_NAMES)}
        self.Tinv = {v: k for k, v in self.T.items()}

        self.cache = {}
        # Conveyor planner mask (docs/rl-layout.md): planned conveyor direction
        # (Dir code, 0 none) of the 4 adjacent tiles N E S W, from the actor
        # observation's path channel (encoder C_PATH). None: no restriction.
        self.plan4 = None
        # launcher cap: own launchers known to the unit (encoder memory + team bus);
        # a launcher may be built only while fewer than MAX_LAUNCHERS. None: no cap.
        self.own_launchers = None
        # target knowledge (docs/masks.md, Targets in florent_heads.hpp): None, or
        # dict(core=set of (x, y), bld=set of (x, y), own_turrets=int). Sentinels
        # only facing a line reaching a known enemy core tile, gunners only facing
        # a known enemy building before any wall, convert only with own turrets.
        self.targets = None

    def once(self, key, fn):
        """Legal-option tables are computed once per turn and shared."""
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def call(self, name, *args):
        try:
            return bool(getattr(self.ct, name)(*args))
        except Exception:
            return False

    def inb(self, x, y):
        return 0 <= x < self.ct.get_map_width() and 0 <= y < self.ct.get_map_height()


def _type(api):
    return api.Tinv[api.ct.get_entity_type()]


# ---- legal options -------------------------------------------------------------
def core_opts(api):
    return api.once('core_opts', lambda: _core_opts(api))


def _core_opts(api):
    ct = api.ct
    p = ct.get_position()
    spawn, sx = [1], [None]
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            if 0 <= dx <= 1 and 0 <= dy <= 1:
                continue
            x, y = p.x + dx, p.y + dy
            sx.append((x, y))
            spawn.append(int(api.inb(x, y) and api.call("can_spawn", api.P(x, y))))
    vo = ct.get_global_voltium()
    conv, amount = [1], [0]
    for pct in CONVERT_PCT[1:]:
        a = (vo * pct + 99) // 100
        amount.append(a)
        conv.append(int(vo > 0 and a > 0 and api.call("can_convert_ammo", a)))
    # Voltium has no other use than ammo: always convert all of it (as in C++)
    conv = [0, 0, 0, 0, 1] if conv[4] else [1, 0, 0, 0, 0]
    return spawn, sx, conv, amount


def builder_opts(api):
    return api.once('builder_opts', lambda: _builder_opts(api))


def _builder_opts(api):
    p = api.ct.get_position()
    tiles = [(p.x + DXY[d][0], p.y + DXY[d][1]) for d in CARD4]
    move = [int(api.call("can_move", api.D[d])) for d in CARD4]
    attack, heal, destroy = [], [], []
    build = [[[0] * 8 for _ in range(4)] for _ in range(9)]
    for i, (x, y) in enumerate(tiles):
        inb = api.inb(x, y)
        q = api.P(x, y)
        attack.append(int(inb and _enemy_building_on(api, x, y) and api.call("can_fire", q)))
        heal.append(int(inb and api.call("can_heal", q)))
        destroy.append(int(inb and api.call("can_destroy", q)))
        if not inb:
            continue
        for t, typ in enumerate(BUILD_ORDER):
            if typ == LAUNCHER and api.own_launchers is not None and api.own_launchers >= MAX_LAUNCHERS:
                continue
            if typ in DIRECTIONAL:
                planned = api.plan4 is not None and typ in (CONVEYOR, SPLITTER)
                for d in range(8):
                    if planned and api.plan4[i] != d + 1:
                        continue
                    if api.targets is not None and typ in (GUNNER, SENTINEL) and not _aims(api, typ, x, y, d + 1):
                        continue
                    build[t][i][d] = int(api.call("can_build", api.T[typ], q, api.D[d + 1]))
            else:
                build[t][i][0] = int(api.call("can_build", api.T[typ], q, None))
    return move, attack, heal, destroy, build, tiles


def _aims(api, typ, x, y, d):
    """A sentinel / gunner built at (x, y) facing Dir d aims at a target (as aims in C++)."""
    from fcode import Environment
    tg = api.targets
    diag = d % 2 == 0
    n = (4 if diag else 5) if typ == SENTINEL else (2 if diag else 3)
    dx, dy = DXY[d]
    for k in range(1, n + 1):
        xx, yy = x + dx * k, y + dy * k
        if not api.inb(xx, yy):
            return False
        if typ == SENTINEL:
            if (xx, yy) in tg["core"]:
                return True
            continue
        if (xx, yy) in tg["bld"]:
            return True
        if _safe(lambda: api.ct.get_tile_env(api.P(xx, yy))) == Environment.WALL:
            return False
    return False


def line_opts(api, typ):
    return api.once('line_opts', lambda: _line_opts(api, typ))


def _enemy_building_on(api, x, y):
    """A visible enemy building on (x, y): what a builder attack damages (as enemy_building_on in C++)."""
    if not api.inb(x, y):
        return False
    ct = api.ct
    b = _safe(lambda: ct.get_tile_building_id(api.P(x, y)))
    return b is not None and ct.get_team(b) != ct.get_team()


def _enemy_hit(api, x, y):
    """What a turret shot on (x, y) hits (the builder bot, else the building) is a
    visible enemy (as enemy_hit in C++)."""
    if not api.inb(x, y):
        return False
    ct = api.ct
    q = api.P(x, y)
    u = _safe(lambda: ct.get_tile_builder_bot_id(q))
    hit = u if u is not None else _safe(lambda: ct.get_tile_building_id(q))
    return hit is not None and ct.get_team(hit) != ct.get_team()


def _line_opts(api, typ):
    """Fire only at a visible enemy; a gunner rotates only to a direction whose
    line (3 straight, 2 diagonal) first meets an enemy. As line_opts in C++."""
    ct = api.ct
    p = ct.get_position()
    dd = ct.get_direction().delta()
    fire, fxy = [], []
    for k in range(1, 6):
        x, y = p.x + dd[0] * k, p.y + dd[1] * k
        fxy.append((x, y))
        fire.append(int(api.inb(x, y) and _enemy_hit(api, x, y) and api.call("can_fire", api.P(x, y))))
    rot = [0] * 8
    if typ == GUNNER:
        from fcode import Environment
        for d in range(8):
            if not api.call("can_rotate", api.D[d + 1]):
                continue
            dx, dy = DXY[d + 1]
            for k in range(1, (3 if d % 2 == 0 else 2) + 1):
                x, y = p.x + dx * k, p.y + dy * k
                if not api.inb(x, y):
                    break
                q = api.P(x, y)
                if _safe(lambda: ct.get_tile_env(q)) == Environment.WALL:
                    break
                if _enemy_hit(api, x, y):
                    rot[d] = 1
                    break
                if _safe(lambda: ct.get_tile_builder_bot_id(q)) is not None or _safe(lambda: ct.get_tile_building_id(q)) is not None:
                    break
    return fire, fxy, rot


def launcher_opts(api):
    return api.once('launcher_opts', lambda: _launcher_opts(api))


def _launcher_opts(api):
    ct = api.ct
    p = ct.get_position()
    dests = [(p.x + dx, p.y + dy) for dy in range(-5, 6) for dx in range(-5, 6) if dx or dy]
    bots, ok, botm = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            bx, by = p.x + dx, p.y + dy
            bots.append((bx, by))
            row = [0] * 120
            has = api.inb(bx, by) and _safe(lambda: ct.get_tile_builder_bot_id(api.P(bx, by))) is not None
            if has:
                for k, (tx, ty) in enumerate(dests):
                    if api.inb(tx, ty) and api.call("can_launch", api.P(bx, by), api.P(tx, ty)):
                        row[k] = 1
            ok.append(row)
            botm.append(int(any(row)))
    # pit launcher: on the edge of a pit 1-2 wide it only throws our own builder
    # (the first one next to it) across the pit, to any legal tile there (as launcher_opts in C++)
    pit = _safe(lambda: ct.get_pit())
    if pit and 1 <= pit[2] <= 2:
        axis, at, w = pit
        c = p.x if axis == "x" else p.y
        west, east = c == at - 1, c == at + w
        if west or east:
            me = ct.get_team()
            taken = False
            for b, (bx, by) in enumerate(bots):
                u = _safe(lambda: ct.get_tile_builder_bot_id(api.P(bx, by))) if api.inb(bx, by) else None
                mine = not taken and u is not None and ct.get_team(u) == me
                row = []
                for k, (tx, ty) in enumerate(dests):
                    tc = tx if axis == "x" else ty
                    across = (tc >= at + w) if west else (tc < at)
                    row.append(int(mine and across and ok[b][k]))
                ok[b] = row
                botm[b] = int(any(row))
                if any(row):
                    taken = True
    return botm, ok, bots, dests


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def has_choice(api):
    t = _type(api)
    if t == CORE:
        spawn, _, conv, _ = core_opts(api)
        return any(spawn[1:]) or any(conv[1:])
    if t == BUILDER:
        move, attack, heal, destroy, build, _ = builder_opts(api)
        return any(move) or any(attack) or any(heal) or any(destroy) or any(any(any(r) for r in tt) for tt in build)
    if t in (GUNNER, SENTINEL):
        fire, _, rot = line_opts(api, t)
        return any(fire) or any(rot)
    if t == LAUNCHER:
        return any(launcher_opts(api)[0])
    return False


def spike_action(api, first_turn=False):
    """('fire', x, y) at the first neighbour with a visible enemy, else None.
    On its first turn a spike fires once anyway (a shot hits every enemy
    building around it, hidden mines included). As spike_action in C++."""
    ct = api.ct
    p = ct.get_position()
    me = ct.get_team()
    if first_turn:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not dx and not dy:
                    continue
                x, y = p.x + dx, p.y + dy
                if api.inb(x, y) and api.call("can_fire", api.P(x, y)):
                    return ("fire", x, y)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            x, y = p.x + dx, p.y + dy
            if not api.inb(x, y):
                continue
            q = api.P(x, y)
            u = _safe(lambda: ct.get_tile_builder_bot_id(q))
            b = _safe(lambda: ct.get_tile_building_id(q))
            enemy = (u is not None and ct.get_team(u) != me) or (b is not None and ct.get_team(b) != me)
            if enemy and api.call("can_fire", q):
                return ("fire", x, y)
    return None


# ---- masked softmax -------------------------------------------------------------
def masked_logp(l, m, c):
    mx = max(l[i] for i in range(len(m)) if m[i])
    z = sum(math.exp(l[i] - mx) for i in range(len(m)) if m[i])
    return l[c] - mx - math.log(z)


def argmax_choice(l, m):
    best = -1
    for i in range(len(m)):
        if m[i] and (best < 0 or l[i] > l[best]):
            best = i
    return best, (masked_logp(l, m, best) if best >= 0 else 0.0)


# ---- decode ----------------------------------------------------------------------
def decide(api, logits, choose=None):
    """Returns (actions, records). actions: list of tuples
    ('spawn',x,y) ('convert',a) ('move',dir) ('build',type,x,y,dir) ('attack',x,y)
    ('heal',x,y) ('destroy',x,y) ('fire',x,y) ('rotate',dir) ('launch',bx,by,tx,ty);
    records: list of (head, mask, choice, logp)."""
    choose = choose or (lambda head, l, m: argmax_choice(l, m))
    if callable(logits):
        logits = logits()
    t = _type(api)
    p = POLICY_OF_TYPE[t]
    actions, records = [], []

    def run(head, mask):
        off = head_offset(p, head)
        c, lp = choose(head, logits[off:off + HEAD_SIZE[head]], mask)
        records.append((head, list(mask), c, lp))
        return c

    if p == P_CORE:
        spawn, sxy, conv, amount = core_opts(api)
        s = run(HC_SPAWN, spawn)
        if s > 0:
            actions.append(("spawn",) + sxy[s])
        c = run(HC_CONVERT, conv)
        if c > 0:
            actions.append(("convert", amount[c]))
    elif p == P_BUILDER:
        move, attack, heal, destroy, build, tiles = builder_opts(api)
        # crossable pit (1-2 wide) and no own launcher: build one on the pit's edge (as in C++)
        pit = _safe(lambda: api.ct.get_pit())
        if api.own_launchers == 0 and pit and 1 <= pit[2] <= 2:
            lt = BUILD_ORDER.index(LAUNCHER)
            for i, (x, y) in enumerate(tiles):
                c = x if pit[0] == "x" else y
                if (c == pit[1] - 1 or c == pit[1] + pit[2]) and build[lt][i][0]:
                    return [("build", LAUNCHER, x, y, C)], []
        bt = [int(any(any(r) for r in build[t2])) for t2 in range(9)]
        mode = [1, int(any(move)), int(any(bt)), int(any(attack)), int(any(heal)), int(any(destroy))]
        m = run(HB_MODE, mode)
        if m == 1:
            actions.append(("move", CARD4[run(HB_MOVE, move)]))
        elif m == 2:
            t2 = run(HB_BTYPE, bt)
            i = run(HB_BTILE, [int(any(build[t2][k])) for k in range(4)])
            d = run(HB_BDIR, build[t2][i]) + 1 if BUILD_ORDER[t2] in DIRECTIONAL else C
            actions.append(("build", BUILD_ORDER[t2]) + tiles[i] + (d,))
        elif m == 3:
            actions.append(("attack",) + tiles[run(HB_ATTACK, attack)])
        elif m == 4:
            actions.append(("heal",) + tiles[run(HB_HEAL, heal)])
        elif m == 5:
            actions.append(("destroy",) + tiles[run(HB_DESTROY, destroy)])
    elif p == P_GUNNER:
        fire, fxy, rot = line_opts(api, t)
        m = run(HG_MODE, [1, int(any(fire)), int(any(rot))])
        if m == 1:
            actions.append(("fire",) + fxy[run(HG_FIRE, fire)])
        elif m == 2:
            actions.append(("rotate", run(HG_ROTATE, rot) + 1))
    elif p == P_SENTINEL:
        fire, fxy, _ = line_opts(api, t)
        if run(HS_MODE, [1, int(any(fire))]) == 1:
            actions.append(("fire",) + fxy[run(HS_FIRE, fire)])
    elif p == P_LAUNCHER:
        botm, ok, bots, dests = launcher_opts(api)
        if run(HL_MODE, [1, int(any(botm))]) == 1:
            i = run(HL_BOT, botm)
            k = run(HL_DEST, ok[i])
            actions.append(("launch",) + bots[i] + dests[k])
    return actions, records


def apply(api, action):
    """Execute one decoded action through the official controller."""
    ct, P, D = api.ct, api.P, api.D
    k = action[0]
    if k == "spawn":
        return ct.spawn_builder(P(action[1], action[2]))
    if k == "convert":
        return ct.convert_ammo(action[1])
    if k == "move":
        return ct.move(D[action[1]])
    if k == "build":
        typ, x, y, d = action[1:]
        return ct.build(api.T[typ], P(x, y), D[d] if typ in DIRECTIONAL else None)
    if k in ("attack", "fire"):
        return ct.fire(P(action[1], action[2]))
    if k == "heal":
        return ct.heal(P(action[1], action[2]))
    if k == "destroy":
        return ct.destroy(P(action[1], action[2]))
    if k == "rotate":
        return ct.rotate(D[action[1]])
    if k == "launch":
        return ct.launch(P(action[1], action[2]), P(action[3], action[4]))
    raise ValueError(k)


def plan4_from_obs(spatial):
    """The 4 adjacent tiles' planned conveyor directions (N E S W) from the
    actor observation's path channel; same cells as plan4_from_obs in C++."""
    from encoder import C_PATH, PATCH, HALF
    base, c = C_PATH * PATCH * PATCH, HALF
    return [spatial[base + (c - 1) * PATCH + c], spatial[base + c * PATCH + c + 1],
            spatial[base + (c + 1) * PATCH + c], spatial[base + c * PATCH + c - 1]]
