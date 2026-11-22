"""Pit-throw probe: builder 0 walks to the pit edge and builds a launcher; builder 1
stands beside it; the launcher logs can_launch for every target across the pit,
then throws builder 1 across. Team B idles. Map: vertical pit, cores on the left/right."""
from fcode import Direction, EntityType, Position, Team


class Player:
    def __init__(self):
        self.k = None

    def run(self, ct):
        if ct.get_team() != Team.A:
            return
        t = ct.get_entity_type()
        r = ct.get_current_round()
        pit = ct.get_pit()  # (axis, at, width)
        edge = pit[1] - 1   # last land column before the pit
        if t == EntityType.CORE:
            if r < 2:
                p = ct.get_position()
                for d in Direction:
                    q = p.add(d)
                    if ct.can_spawn(q):
                        ct.spawn_builder(q); return
                for d in Direction:
                    q = p.add(d).add(d)
                    if ct.can_spawn(q):
                        ct.spawn_builder(q); return
            return
        if t == EntityType.LAUNCHER:
            me = ct.get_position()
            W, H = ct.get_map_width(), ct.get_map_height()
            for dy in (-1, 0, 1):
                b = Position(me.x - 1 if False else me.x, me.y + dy)
            bots = [Position(me.x + dx, me.y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (dx or dy) and ct.get_tile_builder_bot_id(Position(me.x + dx, me.y + dy)) is not None]
            if r >= 40 and bots:
                far = Position(pit[1] + pit[2], me.y)
                if ct.can_launch(bots[0], far):
                    ct.launch(bots[0], far)
                bots = [p for p in bots if ct.get_tile_builder_bot_id(p) is not None]
            for b in bots:   # asked after acting: the verifier checks at the end of the turn
                for tx in range(0, W):
                    for ty in range(max(0, me.y - 5), min(H, me.y + 6)):
                        tp = Position(tx, ty)
                        try:
                            ok = ct.can_launch(b, tp)
                        except Exception:
                            ok = False
                        print("Q launch", b.x, b.y, tx, ty, "=", int(bool(ok)))
            return
        if t != EntityType.BUILDER_BOT:
            return
        if self.k is None:
            self.k = ct.read_store(0); ct.write_store(0, self.k + 1)
        me = ct.get_position()
        goal = Position(edge, 6 + self.k)       # builder 0 at (edge,6), builder 1 at (edge,7)
        if me != goal:
            d = me.cardinal_direction_to(goal)
            for dd in (d, Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST):
                if ct.can_move(dd):
                    ct.move(dd); return
            return
        if self.k == 0:
            spot = Position(edge - 1, 6)        # launcher west of builder 0, diagonal to builder 1
            if ct.can_build_launcher(spot):
                ct.build_launcher(spot)
