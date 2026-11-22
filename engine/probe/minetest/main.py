"""Mine probe. Map 16x12, core A at (2,5), core B at (12,5) (tiles 12-13, 5-6).
A: one builder walks to (10,4) and lays mines on (11,4) (in B's spawn ring) and (10,5).
B: core spawns onto the mined ring tile at round 30 and onto (11,5) at round 32; the
second builder asks build/move questions about the mine tile, then walks onto it.
Questions print as 'Q ...' (checked by the verifier); what B sees prints as 'I ...'."""
from fcode import Direction, EntityType, Position, Team

TYPES = [EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BARRIER, EntityType.MINE,
         EntityType.GUNNER, EntityType.SPIKE, EntityType.LAUNCHER, EntityType.HARVESTER]


class Player:
    def __init__(self):
        self.step = 0

    def run(self, ct):
        t = ct.get_entity_type()
        r = ct.get_current_round()
        if ct.get_team() == Team.A:
            if t == EntityType.CORE and r == 0:
                ct.spawn_builder(Position(4, 5))
            elif t == EntityType.BUILDER_BOT:
                me = ct.get_position()
                if self.step == 0:
                    goal = Position(10, 4)
                    if me != goal:
                        d = me.cardinal_direction_to(goal)
                        if ct.can_move(d): ct.move(d)
                        return
                    self.step = 1
                if self.step == 1 and ct.can_build_mine(Position(11, 4)):
                    ct.build_mine(Position(11, 4)); self.step = 2; return
                if self.step == 2 and ct.can_build_mine(Position(10, 5)):
                    ct.build_mine(Position(10, 5)); self.step = 3; return
                if self.step == 3:
                    for p in (Position(11, 4), Position(10, 5)):
                        for et in TYPES:
                            print("Q build", et.name, p.x, p.y, "-" if et not in (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.GUNNER) else "NORTH",
                                  "=", int(ct.can_build(et, p, Direction.NORTH if et in (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.GUNNER) else None)))
                    if ct.can_move(Direction.WEST):
                        ct.move(Direction.WEST); self.step = 4
            return
        # team B
        if t == EntityType.CORE:
            if r == 30 and ct.can_spawn(Position(11, 4)):
                ct.spawn_builder(Position(11, 4))
            if r == 32 and ct.can_spawn(Position(11, 5)):
                ct.spawn_builder(Position(11, 5))
            for x in range(10, 15):
                for y in range(3, 9):
                    p = Position(x, y)
                    try:
                        v = ct.can_spawn(p)
                    except Exception:
                        v = False
                    if r >= 20: print("Q spawn", x, y, "=", int(v))
            return
        if t == EntityType.BUILDER_BOT:
            me = ct.get_position()
            mp = Position(10, 5)
            if r >= 34 and me == Position(11, 5) and ct.can_move(Direction.WEST):
                ct.move(Direction.WEST)
                return   # stepped on the mine: dead
            for et in TYPES:
                nd = et in (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.GUNNER)
                v = ct.can_build(et, mp, Direction.NORTH if nd else None)
                print("Q build", et.name, mp.x, mp.y, "NORTH" if nd else "-", "=", int(v))
            for d in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
                print("Q move", d.name, "=", int(ct.can_move(d)))
            print("I", r, "empty", int(ct.is_tile_empty(mp)), "passable", int(ct.is_tile_passable(mp)),
                  "building", ct.get_tile_building_id(mp), "nearby_has_mine",
                  int(any(ct.get_entity_type(e) == EntityType.MINE for e in ct.get_nearby_entities())))
            if r >= 34 and me == Position(11, 5) and ct.can_move(Direction.WEST):
                ct.move(Direction.WEST)
