"""Deterministic Florent candidate ABI shared by training and deployment.

The candidate list is **fixed**: indices never renumber when a choice is
illegal. Legality is reported separately as a mask, so a checkpoint's head
layout is stable across maps, rounds and engine states. This mirrors
``rl/sim/florent_actions.cpp`` exactly; the ordering below is part of the
checkpoint ABI and must be changed in both places together.

Only the standard library is used: no numpy, no imports outside fcode.
"""
from dataclasses import dataclass

# Engine direction index ABI (fengine.hpp enum Dir): C=0, N=1 .. NW=8.
C, N, NE, E, SE, S, SW, W, NW = range(9)
DIRS8 = (N, NE, E, SE, S, SW, W, NW)
CARDINALS = (N, E, S, W)
_DIR_DELTA = {
    N: (0, -1), NE: (1, -1), E: (1, 0), SE: (1, 1),
    S: (0, 1), SW: (-1, 1), W: (-1, 0), NW: (-1, -1),
}

# Engine building-type ABI, in the exact order the native enumerator emits:
# CONVEYOR=6 SPLITTER=7 HARVESTER=8 GUNNER=2 SENTINEL=3 LAUNCHER=4 SPIKE=5
# BARRIER=9 MINE=10 (fengine.hpp enum Type).
BUILD_TYPES = (6, 7, 8, 2, 3, 4, 5, 9, 10)
BUILD_NAMES = {
    6: "CONVEYOR", 7: "SPLITTER", 8: "HARVESTER", 2: "GUNNER",
    3: "SENTINEL", 4: "LAUNCHER", 5: "SPIKE", 9: "BARRIER", 10: "MINE",
}
# Only these take a facing Direction as the build `extra` argument.
DIRECTIONAL_BUILDS = frozenset((6, 7, 2, 3))
CONVERT_AMOUNTS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class Candidate:
    kind: str
    args: tuple = ()


def _offsets(radius):
    """Row-major (dx, dy) in [-r, r]^2, excluding the origin."""
    return [(dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx or dy]


OFFSETS1 = tuple(_offsets(1))
OFFSETS2 = tuple(_offsets(2))
OFFSETS5 = tuple(_offsets(5))


def _type_name(ct):
    value = ct.get_entity_type()
    return getattr(value, "value", str(value).split(".")[-1]).lower()


def enumerate_all(ct):
    """Full fixed candidate list for this unit, independent of legality."""
    typ = _type_name(ct)
    pos = ct.get_position()
    x, y = pos.x, pos.y
    out = [Candidate("wait")]
    if typ in ("builder", "builder_bot"):
        for d in CARDINALS:
            out.append(Candidate("move", (d,)))
        for bt in BUILD_TYPES:
            for dx, dy in OFFSETS1:
                for d in range(9):
                    out.append(Candidate("build", (bt, x + dx, y + dy, d)))
        for dx, dy in OFFSETS1:
            out.append(Candidate("attack", (x + dx, y + dy)))
            out.append(Candidate("heal", (x + dx, y + dy)))
            out.append(Candidate("destroy", (x + dx, y + dy)))
    elif typ == "core":
        for dx, dy in OFFSETS2:
            out.append(Candidate("spawn", (x + dx, y + dy)))
        for amount in CONVERT_AMOUNTS:
            out.append(Candidate("convert", (amount,)))
    elif typ in ("gunner", "sentinel", "spike"):
        for dx, dy in OFFSETS5:
            out.append(Candidate("fire", (x + dx, y + dy)))
        if typ == "gunner":
            for d in DIRS8:
                out.append(Candidate("rotate", (d,)))
    elif typ == "launcher":
        for bx, by in OFFSETS1:
            for dx, dy in OFFSETS5:
                out.append(Candidate("launch", (x + bx, y + by, x + dx, y + dy)))
    return out


def _pos(x, y):
    from fcode import Position
    return Position(int(x), int(y))


def _dir(d):
    from fcode import Direction
    delta = _DIR_DELTA.get(d)
    if delta is None:
        return Direction.CENTRE
    for cand in Direction:
        if cand.delta() == delta:
            return cand
    return Direction.CENTRE


def _entity_type(bt):
    from fcode import EntityType
    return getattr(EntityType, BUILD_NAMES[bt])


def _can(ct, name, *args):
    fn = getattr(ct, name, None)
    if fn is None:
        return False
    try:
        return bool(fn(*args))
    except Exception:
        return False


def legal_mask(ct, candidates):
    """Legality mask aligned 1:1 with ``candidates``. Never renumbers them."""
    mask = [True]  # wait is always legal
    for c in candidates[1:]:
        k, a = c.kind, c.args
        if k == "move":
            ok = _can(ct, "can_move", _dir(a[0]))
        elif k == "build":
            bt, tx, ty, d = a
            extra = _dir(d) if bt in DIRECTIONAL_BUILDS else None
            ok = _can(ct, "can_build", _entity_type(bt), _pos(tx, ty), extra)
        elif k == "attack":
            # Official ABI has no can_attack(): a builder bot damages the
            # building on an orthogonally adjacent tile via fire().
            ok = _can(ct, "can_fire", _pos(*a))
        elif k == "heal":
            ok = _can(ct, "can_heal", _pos(*a))
        elif k == "destroy":
            ok = _can(ct, "can_destroy", _pos(*a))
        elif k == "spawn":
            ok = _can(ct, "can_spawn", _pos(*a))
        elif k == "fire":
            ok = _can(ct, "can_fire", _pos(*a))
        elif k == "rotate":
            ok = _can(ct, "can_rotate", _dir(a[0]))
        elif k == "launch":
            bx, by, tx, ty = a
            ok = _can(ct, "can_launch", _pos(bx, by), _pos(tx, ty))
        elif k == "convert":
            ok = _can(ct, "can_convert_ammo", a[0])
        else:
            ok = False
        mask.append(ok)
    return mask


def candidates(ct):
    """Return ``(fixed_candidates, legal_mask)`` for one controller turn."""
    cands = enumerate_all(ct)
    return cands, legal_mask(ct, cands)


def apply(ct, candidate):
    """Execute exactly one candidate."""
    k, a = candidate.kind, candidate.args
    if k == "wait":
        return None
    if k == "move":
        return ct.move(_dir(a[0]))
    if k == "build":
        bt, tx, ty, d = a
        extra = _dir(d) if bt in DIRECTIONAL_BUILDS else None
        return ct.build(_entity_type(bt), _pos(tx, ty), extra)
    if k == "attack":
        return ct.fire(_pos(*a))
    if k == "heal":
        return ct.heal(_pos(*a))
    if k == "destroy":
        return ct.destroy(_pos(*a))
    if k == "spawn":
        return ct.spawn_builder(_pos(*a))
    if k == "fire":
        return ct.fire(_pos(*a))
    if k == "rotate":
        return ct.rotate(_dir(a[0]))
    if k == "launch":
        bx, by, tx, ty = a
        return ct.launch(_pos(bx, by), _pos(tx, ty))
    if k == "convert":
        return ct.convert_ammo(a[0])
    raise ValueError(k)
