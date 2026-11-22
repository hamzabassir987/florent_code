"""Chaos probe for legality parity.

Each unit alternates: on even rounds it only ASKS the engine can_* questions on
many legal and illegal inputs and prints the answers; on odd rounds it takes a
random legal action (exercising every entity type). The verifier replays the
game in the native engine and compares every answer.
Output lines:  Q <query> <args...> = <0|1>
"""
import random

from fcode import Direction, EntityType, Position

DIRS = list(Direction)
CARD = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
BUILD = [EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.HARVESTER, EntityType.BARRIER,
         EntityType.GUNNER, EntityType.SENTINEL, EntityType.LAUNCHER, EntityType.SPIKE, EntityType.MINE]
NEEDS_DIR = (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.GUNNER, EntityType.SENTINEL)


class Player:
    def __init__(self):
        self.rng = None

    def q(self, name, args, val):
        print("Q", name, *args, "=", int(bool(val)))

    def safe(self, fn, *a):
        try:
            return fn(*a)
        except Exception:
            return "EXC"

    def run(self, ct):
        if self.rng is None:
            self.rng = random.Random(ct.get_id())
        R = self.rng
        r = ct.get_current_round()
        t = ct.get_entity_type()
        me = ct.get_position()
        W, H = ct.get_map_width(), ct.get_map_height()
        near = [Position(me.x + dx, me.y + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
        near = [p for p in near if 0 <= p.x < W and 0 <= p.y < H]
        if r % 2 == 0:
            self.ask(ct, R, t, me, near)
            return
        self.act(ct, R, t, me, near, r, W, H)
        self.ask(ct, R, t, me, near)   # questions after acting: cooldowns, one-action-per-round

    def ask(self, ct, R, t, me, near):
        if True:
            if t == EntityType.BUILDER_BOT:
                for d in DIRS:
                    self.q("move", (d.name,), self.safe(ct.can_move, d))
                for p in R.sample(near, min(8, len(near))):
                    et = R.choice(BUILD)
                    d = R.choice(DIRS)
                    v = self.safe(ct.can_build, et, p, d if et in NEEDS_DIR else None)
                    self.q("build", (et.name, p.x, p.y, d.name if et in NEEDS_DIR else "-"), v if v != "EXC" else 0)
                    self.q("heal", (p.x, p.y), self.safe(ct.can_heal, p) is True)
                    self.q("destroy", (p.x, p.y), self.safe(ct.can_destroy, p) is True)
                    self.q("attack", (p.x, p.y), self.safe(ct.can_fire, p) is True)
            elif t == EntityType.CORE:
                for p in near:
                    self.q("spawn", (p.x, p.y), self.safe(ct.can_spawn, p) is True)
                for a in (0, 1, 5, 10**6):
                    self.q("convert", (a,), self.safe(ct.can_convert_ammo, a) is True)
            elif t in (EntityType.GUNNER, EntityType.SENTINEL, EntityType.SPIKE):
                for p in near:
                    self.q("fire", (p.x, p.y), self.safe(ct.can_fire, p) is True)
                if t == EntityType.GUNNER:
                    for d in DIRS:
                        self.q("rotate", (d.name,), self.safe(ct.can_rotate, d) is True)
            elif t == EntityType.LAUNCHER:
                bots = [p for p in near if abs(p.x - me.x) <= 1 and abs(p.y - me.y) <= 1]
                for b in bots:
                    for tp in R.sample(near, min(4, len(near))):
                        self.q("launch", (b.x, b.y, tp.x, tp.y), self.safe(ct.can_launch, b, tp) is True)

    def act(self, ct, R, t, me, near, r, W, H):
        try:
            if t == EntityType.CORE:
                if ct.get_global_voltium() > 0 and R.random() < 0.5:
                    a = R.randint(1, ct.get_global_voltium())
                    if ct.can_convert_ammo(a):
                        ct.convert_ammo(a)
                ps = [p for p in near if ct.can_spawn(p)]
                if ps and R.random() < 0.5:
                    ct.spawn_builder(R.choice(ps))
            elif t == EntityType.BUILDER_BOT:
                x = R.random()
                adj = [me.add(d) for d in CARD]
                adj = [p for p in adj if 0 <= p.x < W and 0 <= p.y < H]
                if x < 0.4:
                    ds = [d for d in CARD if ct.can_move(d)]
                    if ds:
                        ct.move(R.choice(ds))
                elif x < 0.8:
                    opts = []
                    for p in adj:
                        for et in BUILD:
                            for d in (CARD if et in (EntityType.CONVEYOR, EntityType.SPLITTER) else DIRS[:8] if et in NEEDS_DIR else [None]):
                                if ct.can_build(et, p, d):
                                    opts.append((et, p, d))
                    if opts:
                        fav = [o for o in opts if o[0] in (EntityType.LAUNCHER, EntityType.MINE, EntityType.HARVESTER, EntityType.SPLITTER)]
                        et, p, d = R.choice(fav if fav and R.random() < 0.5 else opts)
                        ct.build(et, p, d)
                elif x < 0.88:
                    ps = [p for p in adj if ct.can_heal(p)]
                    if ps:
                        ct.heal(R.choice(ps))
                elif x < 0.95:
                    ps = [p for p in adj if ct.can_fire(p)]
                    if ps:
                        ct.fire(R.choice(ps))
                elif x < 0.99:
                    ps = [p for p in adj if ct.can_destroy(p)]
                    if ps:
                        ct.destroy(R.choice(ps))
                elif r > 300:
                    ct.self_destruct()
            elif t in (EntityType.GUNNER, EntityType.SENTINEL, EntityType.SPIKE):
                ps = [p for p in near if ct.can_fire(p)]
                if ps:
                    ct.fire(R.choice(ps))
                elif t == EntityType.GUNNER and R.random() < 0.1:
                    ds = [d for d in DIRS if ct.can_rotate(d)]
                    if ds:
                        ct.rotate(R.choice(ds))
            elif t == EntityType.LAUNCHER:
                bots = [p for p in near if abs(p.x - me.x) <= 1 and abs(p.y - me.y) <= 1]
                opts = [(b, tp) for b in bots for tp in near if ct.can_launch(b, tp)]
                if opts:
                    ct.launch(*R.choice(opts))
        except Exception as e:
            print("ERR", repr(e))
