"""Unit-cap probe: play the economy bot until round 400, then spend everything on
units (spawned builders, spikes, gunners, launchers) to reach the 50-unit cap,
logging the official answers to spawn/build questions (checked by the verifier)."""
import random
from fcode import Direction, EntityType, Position
from econ import Player as Econ

UNITS = [EntityType.SPIKE, EntityType.GUNNER, EntityType.LAUNCHER, EntityType.SENTINEL]


class Player:
    def __init__(self):
        self.econ = Econ()
        self.rng = random.Random(7)

    def run(self, ct):
        r = ct.get_current_round()
        if r < 400:
            self.econ.run(ct)
            return
        t = ct.get_entity_type()
        me = ct.get_position()
        W, H = ct.get_map_width(), ct.get_map_height()
        if t == EntityType.CORE:
            for dx in range(-1, 3):
                for dy in range(-1, 3):
                    p = Position(me.x + dx, me.y + dy)
                    if 0 <= p.x < W and 0 <= p.y < H and ct.can_spawn(p):
                        ct.spawn_builder(p); break
                else:
                    continue
                break
            for dx in range(-1, 3):
                for dy in range(-1, 3):
                    p = Position(me.x + dx, me.y + dy)
                    if 0 <= p.x < W and 0 <= p.y < H:
                        print("Q spawn", p.x, p.y, "=", int(ct.can_spawn(p)))
            print("I units", ct.get_unit_count())
            return
        if t != EntityType.BUILDER_BOT:
            return
        adj = [me.add(d) for d in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)]
        adj = [p for p in adj if 0 <= p.x < W and 0 <= p.y < H]
        for p in adj:
            for et in UNITS:
                d = Direction.NORTH if et in (EntityType.GUNNER, EntityType.SENTINEL) else None
                if ct.can_build(et, p, d):
                    ct.build(et, p, d)
                    break
            else:
                continue
            break
        else:
            for d in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
                if ct.can_move(d) and self.rng.random() < 0.5:
                    ct.move(d); break
        for p in adj:
            for et in UNITS + [EntityType.BARRIER, EntityType.CONVEYOR]:
                d = Direction.NORTH if et in (EntityType.GUNNER, EntityType.SENTINEL, EntityType.CONVEYOR) else None
                print("Q build", et.name, p.x, p.y, d.name if d else "-", "=", int(ct.can_build(et, p, d)))
