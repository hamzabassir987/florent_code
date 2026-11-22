"""Scenario probe bot: builds an exact layout from scenario.json, then idles.

scenario.json: {"A": {"builders": n, "tasks": [[type, x, y, dir], ...]}, "B": {...}}
Builder k does tasks k, k+n, k+2n, ... in order (walking next to each tile).
Every build attempt result is printed as  PROBE <round> <json>.
"""
import json
import os
from collections import deque

from fcode import Direction, EntityType, Position, Team

HERE = os.path.dirname(os.path.abspath(__file__))
SC = json.load(open(os.path.join(HERE, "scenario.json")))
CARD = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
DIRS = {d.name: d for d in Direction}
TYPES = {t.name: t for t in EntityType}


class Player:
    def __init__(self):
        self.k = None
        self.spawned = 0
        self.step = 0

    def run(self, ct):
        team = "A" if ct.get_team() == Team.A else "B"
        sc = SC.get(team, {"builders": 0, "tasks": []})
        t = ct.get_entity_type()
        if t == EntityType.CORE:
            if self.spawned < sc["builders"] and ct.get_action_cooldown() == 0:
                p = ct.get_position()
                for d in Direction:
                    q = p.add(d)
                    for q2 in (q, q.add(d)):
                        try:
                            if ct.can_spawn(q2):
                                ct.spawn_builder(q2); self.spawned += 1
                                ct.write_store(self.spawned, 0)
                                return
                        except Exception:
                            pass
            return
        if t != EntityType.BUILDER_BOT:
            return
        if self.k is None:
            # builder index = order of creation, via the store counter
            self.k = ct.read_store(15)
            ct.write_store(15, self.k + 1)
        mine = sc["tasks"][self.k::max(1, sc["builders"])]
        if self.step >= len(mine):
            return
        typ, x, y, d = mine[self.step]
        target = Position(x, y)
        pos = ct.get_position()
        if abs(pos.x - x) + abs(pos.y - y) == 1:
            if ct.get_action_cooldown() == 0 and ct.get_move_cooldown() == 0:
                et = TYPES[typ]
                args = (target, DIRS[d]) if d else (target,)
                ok = ct.can_build(et, target, DIRS[d] if d else None)
                if ok:
                    ct.build(et, target, DIRS[d] if d else None)
                    import sys; print("PROBE", ct.get_current_round(), json.dumps({"built": [typ, x, y, d]}), file=sys.stderr)
                    self.step += 1
            return
        # walk to a tile next to the target, avoiding tiles still to be built
        todo = set((tx, ty) for _, tx, ty, _ in mine[self.step:])
        goals = set()
        for dd in CARD:
            q = target.add(dd)
            goals.add((q.x, q.y))
        prev = {(pos.x, pos.y): None}
        dq = deque([(pos.x, pos.y)])
        found = None
        W, H = ct.get_map_width(), ct.get_map_height()
        while dq:
            cx, cy = dq.popleft()
            if (cx, cy) in goals and (cx, cy) not in todo:
                found = (cx, cy); break
            for dd in CARD:
                nx, ny = cx + dd.delta()[0], cy + dd.delta()[1]
                if not (0 <= nx < W and 0 <= ny < H) or (nx, ny) in prev:
                    continue
                q = Position(nx, ny)
                if ct.is_in_vision(q) and not ct.is_tile_passable(q):
                    continue
                prev[(nx, ny)] = (cx, cy)
                dq.append((nx, ny))
        if found is None or ct.get_move_cooldown() != 0:
            return
        c = found
        while prev[c] != (pos.x, pos.y):
            c = prev[c]
        d = pos.cardinal_direction_to(Position(*c))
        if ct.can_move(d):
            ct.move(d)
