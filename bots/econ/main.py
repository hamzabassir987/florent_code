"""Economy-first bot.

Builder bots remember the map they have seen, claim ore tiles, build a
harvester on each and lay a shortest conveyor chain (BFS) from the harvester
into the core or into an existing friendly conveyor. Voltium is converted to
ammo by the core; spikes and gunners are built around the core when enemies
show up there, and late in the game spare builders raid enemy conveyors.

Team memory
-----------
The runtime gives every unit its own Player object and a single 16-slot u32
team store whose writes become visible to the *next* round. Slots 3-12 are a
priority-banded report bus (see docs/team-memory.md):

  0      core position (packed)
  1      harvester count, written by the core from its merged view
  2      threat position near our core (packed, 0 = none)
  3-5    resource discoveries          (highest priority)
  6-8    turret / sentinel / spike / harvester add or remove
  9-12   builder sightings, refreshed every 3 rounds
  13,14  guard lease
  15     phase flags

Every unit queues its own uncommitted observations and broadcasts one report
per band per turn; a report survives in the slot until overwritten, so every
living unit sees it on its next turn. Losing a same-round slot collision is
detected next turn (the report is absent from the snapshot) and retried.
"""

from collections import deque
import os
import sys

from fcode import Controller, Direction, EntityType, Environment, Position, Team

# The sibling modules (encoder, heads, student, sw*) must import however
# the runner loads this file: put this file's own folder on the import path first.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
except NameError:                     # no __file__: rely on the runner's path
    _HERE = None
IMPORT_ERRORS = []
try:
    from encoder import ActorEncoder
except Exception as _e:
    ActorEncoder = None
    IMPORT_ERRORS.append("encoder: " + repr(_e))
try:
    import heads as HEADS
    from student import StudentPolicy
except Exception as _e:
    HEADS = None
    StudentPolicy = None
    IMPORT_ERRORS.append("heads/student: " + repr(_e))

# Learned turns start the network only while this much CPU of the turn is left
# unused (the SWAR forward takes ~4-5 ms of the 10 ms budget).
# The student only saw games of TRAINED_ROUNDS rounds: its round input is scaled so the
# official 1000-round game spans the same 0..TRAINED_ROUNDS range (ages, cooldowns and
# every other input stay in real rounds). FLORENT_ROUND_DIV overrides the divisor.
OFFICIAL_ROUNDS, TRAINED_ROUNDS = 1000, 400
ROUND_DIV = float(os.environ.get("FLORENT_ROUND_DIV", str(OFFICIAL_ROUNDS / TRAINED_ROUNDS)))   # 2.5
ROUND_SCALAR = 9                     # actor scalar index of the current round (encoder)
LEARNED_START_LIMIT_US = int(os.environ.get("FLORENT_START_LIMIT_US", "4500"))
_STATUS_SHOWN = False


def _embedded_weights():
    """The student.json data as plain Python literals, written by
    rl/tools/make_submission.py: sw_meta.py (META) and one module per policy
    (sw0.py ... sw7.py, POLICY)."""
    from sw_meta import META
    pols = []
    for k in range(META["n_policies"]):
        if k == 0: from sw0 import POLICY
        elif k == 1: from sw1 import POLICY
        elif k == 2: from sw2 import POLICY
        elif k == 3: from sw3 import POLICY
        elif k == 4: from sw4 import POLICY
        elif k == 5: from sw5 import POLICY
        elif k == 6: from sw6 import POLICY
        else: from sw7 import POLICY
        pols.append(POLICY)
    data = dict(META)
    del data["n_policies"]
    data["policies"] = pols
    return data


def _diag(ct, text):
    """Server diagnostics (replays keep no prints, but keep the team store):
    the core writes `text` into store slots 0-15, 3 ASCII bytes per word,
    word = chunk index (top byte) + 3 bytes of text. Decode with
    rl/tools/read_diag.py. Only the core writes, once per round."""
    try:
        if ct.get_entity_type() != EntityType.CORE:
            return
        b = text.encode("ascii", "replace")[:765]
        rnd = ct.get_current_round()
        n = (len(b) + 2) // 3
        for k in range(16):                      # rotate through the message, 16 chunks per round
            j = (rnd * 16 + k) % max(n, 1)
            c = b[3 * j:3 * j + 3].ljust(3, b" ")
            ct.write_store(k, (j & 0xFF) << 24 | c[0] << 16 | c[1] << 8 | c[2])
    except Exception:
        pass


if os.environ.get("FLORENT_POLICY"):
    # A cyclic-GC pause can push a turn past the 10 ms limit; reference
    # counting still frees everything, cycles just wait.
    import gc
    gc.disable()

CARDINALS = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
DIRS8 = tuple(d for d in Direction if d != Direction.CENTRE)
CARD_DELTA = tuple(d.delta() for d in CARDINALS)

S_CORE = 0
S_HARV = 1
S_THREAT = 2
S_PHASE = 15
S_GUARD = 13
S_GUARD_BEAT = 14

# report bands (priority order)
BAND_RES = (3, 4, 5)
BAND_STR = (6, 7, 8)
BAND_BLD = (9, 10, 11, 12)
REPORT_SLOTS = BAND_RES + BAND_STR + BAND_BLD

# report categories
R_TI, R_VO, R_GUNNER, R_SENTINEL, R_SPIKE, R_HARV, R_BUILDER = range(7)
A_ADD, A_REMOVE, A_REFRESH = 0, 1, 2
BUILDER_REFRESH_ROUNDS = 3
BUILDER_SIGHTING_TTL = 3
PENDING_CAP = 64

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

MAX_TRUNK = 3
WALKABLE_TYPES = (EntityType.CONVEYOR, EntityType.SPLITTER)
DANGER_TYPES = (EntityType.SPIKE, EntityType.GUNNER, EntityType.SENTINEL)
TURRETS = (EntityType.GUNNER, EntityType.SENTINEL, EntityType.LAUNCHER, EntityType.SPIKE)


def pack(x: int, y: int) -> int:
    return ((x + 1) << 8) | (y + 1)


def unpack(v: int):
    if v <= 0:
        return None
    return Position((v >> 8) - 1, (v & 0xFF) - 1)


# ---------------------------------------------------------------------------
# Team-store report packets
#
# One self-contained message per slot, packed into a u32 (0 = empty):
#
#   bit  0     valid flag (always 1 in a written packet)
#   bits 1-2   action: 0 = add/observe, 1 = remove, 2 = refresh (sighting)
#   bits 3-5   category: R_TI .. R_BUILDER
#   bit  6     team: 0 = own, 1 = enemy
#   bits 7-17  tile index (11 bits; every shipped map fits in 44x36)
#   bits 18-27 observation round (10 bits, wraps every 1024 rounds)
#
# Fields are independent of slot choice, so a receiver needs nothing else.
# ---------------------------------------------------------------------------
PACK_TILE_BITS = 11
PACK_ROUND_BITS = 10
PACK_TILE_MAX = (1 << PACK_TILE_BITS) - 1
PACK_ROUND_MASK = (1 << PACK_ROUND_BITS) - 1


def pack_report(action, cat, team, tile, round_no):
    return (1
            | (action & 3) << 1
            | (cat & 7) << 3
            | (team & 1) << 6
            | (tile & PACK_TILE_MAX) << 7
            | (round_no & PACK_ROUND_MASK) << 18)


def unpack_report(v):
    if v <= 0 or not (v & 1):
        return None
    return (
        (v >> 1) & 3,
        (v >> 3) & 7,
        (v >> 6) & 1,
        (v >> 7) & PACK_TILE_MAX,
        (v >> 18) & PACK_ROUND_MASK,
    )


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
        self.bad_sym = set()
        self.siege_cool = 0
        self.mines = set()
        self.pending_reports = []
        self.known_resources = {}
        self.known_structures = {}
        self.known_builders = {}
        self.last_report_round = -1
        self.actor_encoder = ActorEncoder() if ActorEncoder is not None else None
        # FLORENT_POLICY=<student.json>: the int8 student plays every unit;
        # otherwise the heuristic bot below plays.
        self.student = None
        policy_path = os.environ.get("FLORENT_POLICY")
        # local testing only: FLORENT_TEACHER=<model.pt> plays the float teacher through
        # FLORENT_TEACHER_LIB (rl/tools/teacher_capi.cpp, libtorch) instead of the student
        if HEADS is not None and os.environ.get("FLORENT_TEACHER"):
            try:
                from teacher_local import TeacherPolicy   # local tests only, not shipped
                self.student = TeacherPolicy(os.environ["FLORENT_TEACHER_LIB"], os.environ["FLORENT_TEACHER"])
                policy_path = None
            except Exception as e:
                print("FLORENT_TEACHER load failed:", repr(e), file=sys.stderr)
        if StudentPolicy is not None and HEADS is not None and policy_path:
            try:
                self.student = StudentPolicy(policy_path)
            except Exception as e:
                pass
                print("FLORENT_POLICY load failed:", repr(e), file=sys.stderr)
        # submission: the student's weights embedded as a module (rl/tools/make_submission.py);
        # used when no FLORENT_POLICY / FLORENT_TEACHER is set
        if self.student is None and StudentPolicy is not None and HEADS is not None and not policy_path \
                and not os.environ.get("FLORENT_TEACHER"):
            try:
                self.student = StudentPolicy(data=_embedded_weights())
            except ImportError as e:
                IMPORT_ERRORS.append("weights: " + repr(e))
            except Exception as e:
                IMPORT_ERRORS.append("embedded student: " + repr(e))
        # say once per process which brain plays (visible in the bot output / logs)
        global _STATUS_SHOWN
        if not _STATUS_SHOWN:
            _STATUS_SHOWN = True
            if self.student is not None:
                print("[florent] learned policy loaded:", type(self.student).__name__, file=sys.stderr)
            else:
                print("[florent] NO learned policy, the bot does nothing; import errors:", IMPORT_ERRORS or "none", file=sys.stderr)
        self.policy_disabled = False
        self.learned_turns = 0
        self.fallback_turns = 0

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
        self.danger = [0] * n  # tiles hit by known enemy turrets
        self.turrets = {}  # idx -> (entity id, danger tiles)
        self.seen_round = [-1] * n
        self.core = None
        self.ecore = None

    def run(self, ct: Controller) -> None:
        # no learned policy loaded: do nothing (no heuristic fallback), unless
        # FLORENT_HEURISTIC=1 asks for the heuristic (local tests)
        if self.student is None and os.environ.get("FLORENT_HEURISTIC") != "1":
            _diag(ct, "NOLOAD " + "; ".join(IMPORT_ERRORS))
            return
        if not self.init:
            self._setup(ct)
        # Learned inference (opt-in). Heuristic handlers remain the fallback for
        # an unavailable policy, an error, or a turn without enough CPU left.
        # Learned inference. A turn the student does not play (error, CPU too short)
        # is skipped: no heuristic fallback unless FLORENT_HEURISTIC=1.
        heuristic = os.environ.get("FLORENT_HEURISTIC") == "1"
        if self.student is not None and not self.policy_disabled:
            try:
                if self.learned_turn(ct):
                    return
            except Exception as e:
                print("LEARNED error, turn skipped:", repr(e), file=sys.stderr)
                import traceback as _tb
                _diag(ct, "ERR " + repr(e) + " @ " + " < ".join(
                    "%s:%d" % (f.name, f.lineno) for f in reversed(_tb.extract_tb(e.__traceback__))))
                if heuristic:
                    self.policy_disabled = True
        if not heuristic:
            return
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
            pass

            import traceback

            print("ERR", t, repr(e), file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def learned_turn(self, ct):
        """One turn of the int4 student (same heads and masks as training).
        Returns False to hand this turn to the heuristic (CPU too short)."""
        api = HEADS.Api(ct)
        typ = api.Tinv[ct.get_entity_type()]
        if typ == HEADS.SPIKE:                       # deterministic, as in training
            first = ct.get_id() not in self.actor_encoder.memories   # its first turn: fires once (clears mines)
            self.actor_encoder.observe(ct)           # its memory + the team report bus
            act = HEADS.spike_action(api, first)
            if act:
                HEADS.apply(api, act)
            return True
        p = HEADS.POLICY_OF_TYPE.get(typ)
        if p is None:
            return True
        spatial, scalars = self.actor_encoder.encode(ct)   # memory updates every turn
        api.plan4 = HEADS.plan4_from_obs(spatial)   # conveyor planner mask (path channel)
        mem = self.actor_encoder.memories[ct.get_id()]
        api.own_launchers = mem.own_count(HEADS.LAUNCHER)   # launcher cap
        api.targets = mem.targets()                         # turret direction / convert masks
        if not HEADS.has_choice(api):
            return True                              # only "wait" is legal: no forward
        if ct.get_cpu_time_elapsed() > LEARNED_START_LIMIT_US:
            self.fallback_turns += 1
            print("FALLBACK", ct.get_current_round(), ct.get_cpu_time_elapsed(), file=sys.stderr)
            return True    # wait: the heuristic bot's store layout clashes with the team bus
        scalars = list(scalars)
        scalars[ROUND_SCALAR] = int(scalars[ROUND_SCALAR] / ROUND_DIV)   # trained on TRAINED_ROUNDS-round games
        raw = spatial + scalars
        actions, _ = HEADS.decide(api, lambda: self.student.logits(p, raw))
        for a in actions:
            HEADS.apply(api, a)
            if a[0] == "spawn":                     # team bus: the core's next report carries the spawn bit
                self.actor_encoder.memories[ct.get_id()].spawned_now = True
        self.learned_turns += 1
        print("LEARNED", ct.get_current_round(), p, actions, ct.get_cpu_time_elapsed())
        return True

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
            if i in self.turrets and self.turrets[i][0] != bid:
                for t in self.turrets.pop(i)[1]:
                    self.danger[t] -= 1
            if bid is None:
                bld[i] = B_NONE
                continue
            et = ct.get_entity_type(bid)
            own = ct.get_team(bid) == me
            if not own and i not in self.turrets and et in DANGER_TYPES:
                tiles = []
                if et == EntityType.SPIKE:
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            x, y = p.x + dx, p.y + dy
                            if (dx or dy) and 0 <= x < W and 0 <= y < self.H:
                                tiles.append(y * W + x)
                else:
                    for q in ct.get_attackable_tiles_from(p, ct.get_direction(bid), et):
                        if 0 <= q.x < W and 0 <= q.y < self.H:
                            tiles.append(q.y * W + q.x)
                for t in tiles:
                    self.danger[t] += 1
                self.turrets[i] = (bid, tiles)
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
                # The enemy core is symmetry-derived; a direct sighting is
                # still kept as a high-confidence local fact, not broadcast in
                # the report bus.
                if not own:
                    self.ecore = ct.get_position(uid)
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

        phase = ct.read_store(S_PHASE) & 1
        if r >= 150 and ct.get_global_resources() >= 500:
            phase = 1
        ct.write_store(S_PHASE, phase | (2 if ct.get_hp() < ct.get_max_hp() else 0))

        if ct.get_action_cooldown() != 0:
            return
        if ct.get_unit_count() >= 48:
            return
        ti = ct.get_global_resources()
        cost = ct.get_builder_bot_cost()
        area = self.W * self.H
        harv = ct.read_store(S_HARV)
        cap = min(14, 4 + area // 100)
        want = min(cap, 4 + harv // 2)
        if phase:
            want = min(30, want + ti // 400)
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
        self._merge_reports(ct)
        self._scan_reportable(ct)
        self._publish_reports(ct)
        if self.ecore is None:
            self.ecore = self.enemy_core_guess()
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
        if ct.read_store(S_PHASE) & 1 and self.is_guard(ct):
            self.guard(ct, pos)
            return
        if ct.read_store(S_PHASE) & 2 and self.heal_core(ct, pos):
            return

        if self.jobs:
            self.do_jobs(ct, pos)
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
        if self.target is not None:
            self.go_harvest(ct, pos)
            return
        # nothing to harvest: siege, raid or explore
        r = ct.get_current_round()
        if ct.read_store(S_PHASE) & 1 and r >= self.siege_cool:
            if self.siege(ct, pos):
                return
            self.siege_cool = r + 25
        if ct.get_current_round() > 120 and self.raid(ct, pos):
            return
        self.do_explore(ct, pos)

    # ------------------------------------------------------------------
    # enemy core location
    def enemy_core_guess(self):
        if self.ecore is not None:
            return self.ecore
        W, H = self.W, self.H
        cx, cy = self.core.x, self.core.y
        cands = []
        for sym, (gx, gy) in enumerate(((W - 2 - cx, H - 2 - cy), (W - 2 - cx, cy), (cx, H - 2 - cy))):
            if (gx, gy) == (cx, cy):
                continue
            if sym in self.bad_sym:
                continue
            cands.append((sym, gx, gy))
        env = self.env
        for sym, gx, gy in cands:
            # rule out by having seen the spot without a core
            if env[gy * W + gx] != U and self.ecore is None:
                self.bad_sym.add(sym)
                continue
            # rule out by terrain mismatch (cheap sample)
            bad = False
            for i in range(0, W * H, 7):
                e = env[i]
                if e == U:
                    continue
                x, y = i % W, i // W
                if sym == 0:
                    mx, my = W - 1 - x, H - 1 - y
                elif sym == 1:
                    mx, my = W - 1 - x, y
                else:
                    mx, my = x, H - 1 - y
                m = env[my * W + mx]
                if m != U and m != e:
                    bad = True
                    break
            if bad:
                self.bad_sym.add(sym)
                continue
            return Position(gx, gy)
        return None

    def siege(self, ct, pos):
        ec = self.enemy_core_guess()
        if ec is None:
            return False
        W = self.W
        foot = [(ec.x + dx, ec.y + dy) for dx in (0, 1) for dy in (0, 1)]
        d2 = min((pos.x - fx) ** 2 + (pos.y - fy) ** 2 for fx, fy in foot)
        if d2 > 20 or self.ecore is None:
            res = self.bfs_first_step(
                pos, lambda i: min((i % W - fx) ** 2 + (i // W - fy) ** 2 for fx, fy in foot) <= 13
            )
            if res is None:
                return False
            self.step(ct, pos, res[1])
            return True
        ec = self.ecore
        foot = [(ec.x + dx, ec.y + dy) for dx in (0, 1) for dy in (0, 1)]
        me = self.me
        ti = ct.get_global_resources()
        acd = ct.get_action_cooldown() == 0
        # 1. heal our damaged siege turrets next to us
        if acd:
            for d in CARDINALS:
                q = pos.add(d)
                if not self.inb(q.x, q.y):
                    continue
                bid = ct.get_tile_building_id(q)
                if (
                    bid is not None
                    and ct.get_team(bid) == me
                    and ct.get_hp(bid) < ct.get_max_hp(bid)
                    and ct.can_heal(q)
                ):
                    ct.heal(q)
                    return True
        # 2. count our gunners around the core, find free ring tiles
        ring = []
        own_guns = 0
        for x in range(ec.x - 1, ec.x + 3):
            for y in range(ec.y - 1, ec.y + 3):
                if (x, y) in foot or not self.inb(x, y):
                    continue
                i = y * W + x
                b = self.bld[i]
                if b == B_OWN_OTHER:
                    own_guns += 1
                elif b == B_NONE and self.env[i] in (EMPTY, TI, VO):
                    ring.append((x, y))
        ammo = ct.get_global_ammo()
        want_guns = own_guns < 3 or (ammo >= 40 * own_guns)
        if ring and want_guns and ti >= ct.get_gunner_cost() + 10:
            ring.sort(key=lambda t: abs(t[0] - pos.x) + abs(t[1] - pos.y))
            for gx, gy in ring:
                if abs(gx - pos.x) + abs(gy - pos.y) == 1:
                    if not acd:
                        return True
                    gp = Position(gx, gy)
                    best = min(foot, key=lambda f: (f[0] - gx) ** 2 + (f[1] - gy) ** 2)
                    face = gp.direction_to(Position(best[0], best[1]))
                    if ct.can_build_gunner(gp, face):
                        ct.build_gunner(gp, face)
                        self.bld[gy * W + gx] = B_OWN_OTHER
                        return True
            gx, gy = ring[0]
            if self.move_adjacent_to(ct, pos, gx, gy):
                return True
        # 3. attack the core directly if adjacent and rich
        if acd and ti >= 100:
            for d in CARDINALS:
                q = pos.add(d)
                if (q.x, q.y) in foot and ct.can_fire(q):
                    ct.fire(q)
                    return True
            # attack any adjacent enemy building
            if self.attack_adjacent(ct, pos):
                return True
        # walk next to the core footprint
        adj = set()
        for fx, fy in foot:
            for dx, dy in CARD_DELTA:
                t = (fx + dx, fy + dy)
                if t not in foot and self.inb(*t):
                    adj.add(t[1] * W + t[0])
        if pos.y * W + pos.x in adj:
            return True
        res = self.bfs_first_step(pos, lambda i: i in adj)
        if res is not None:
            self.step(ct, pos, res[1])
            return True
        return False

    def _report_band(self, cat):
        if cat in (R_TI, R_VO):
            return BAND_RES
        if cat == R_BUILDER:
            return BAND_BLD
        return BAND_STR

    def _queue_report(self, action, cat, team, tile, round_no):
        if not (0 <= tile <= PACK_TILE_MAX):
            return
        fact = (action, cat, team, tile)
        if fact not in self.pending_reports:
            self.pending_reports.append(fact)
            if len(self.pending_reports) > PENDING_CAP:
                # Preserve earlier, higher-priority facts; newest builder
                # sightings are the only intentionally lossy class.
                self.pending_reports = self.pending_reports[:PENDING_CAP]

    def _merge_reports(self, ct):
        for slot in REPORT_SLOTS:
            msg = unpack_report(ct.read_store(slot))
            if msg is None:
                continue
            action, cat, team, tile, seen_round = msg
            if cat in (R_TI, R_VO):
                if action != A_REMOVE:
                    self.known_resources[tile] = TI if cat == R_TI else VO
            elif cat == R_BUILDER:
                if action == A_REMOVE:
                    self.known_builders.pop(tile, None)
                else:
                    self.known_builders[tile] = (team, seen_round)
            else:
                key = (tile, cat, team)
                if action == A_REMOVE:
                    self.known_structures.pop(key, None)
                else:
                    self.known_structures[key] = seen_round
        now = ct.get_current_round()
        for tile, (_, seen) in tuple(self.known_builders.items()):
            if ((now - seen) & PACK_ROUND_MASK) > BUILDER_SIGHTING_TTL:
                self.known_builders.pop(tile, None)

    def _publish_reports(self, ct):
        """Publish deterministic, retryable reports; writes land next round."""
        if not self.pending_reports:
            return
        now = ct.get_current_round()
        remaining = []
        for fact in self.pending_reports:
            action, cat, team, tile = fact
            band = self._report_band(cat)
            # A stable hash distributes writers without a shared append cursor.
            slot = band[(tile * 17 + cat * 7 + team) % len(band)]
            ct.write_store(slot, pack_report(action, cat, team, tile, now))
            remaining.append(fact)
        self.pending_reports = remaining[-PENDING_CAP:]
        self.last_report_round = now

    def _scan_reportable(self, ct):
        """Queue facts visible on this unit's current controller view."""
        now = ct.get_current_round()
        for p in ct.get_nearby_tiles():
            i = p.y * self.W + p.x
            env = ENV_CODE.get(ct.get_tile_env(p), EMPTY)
            if env == TI:
                self._queue_report(A_ADD, R_TI, 0, i, now)
            elif env == VO:
                self._queue_report(A_ADD, R_VO, 0, i, now)
            bid = ct.get_tile_building_id(p)
            if bid is not None:
                et = ct.get_entity_type(bid)
                if et in (EntityType.GUNNER, EntityType.SENTINEL, EntityType.SPIKE, EntityType.HARVESTER):
                    cat = {
                        EntityType.GUNNER: R_GUNNER, EntityType.SENTINEL: R_SENTINEL,
                        EntityType.SPIKE: R_SPIKE, EntityType.HARVESTER: R_HARV,
                    }[et]
                    self._queue_report(A_ADD, cat, int(ct.get_team(bid) != self.me), i, now)
            uid = ct.get_tile_builder_bot_id(p)
            if uid is not None and now % BUILDER_REFRESH_ROUNDS == (uid % BUILDER_REFRESH_ROUNDS):
                self._queue_report(A_REFRESH, R_BUILDER, int(ct.get_team(uid) != self.me), i, now)

    def claimed(self, ct):
        # Resource reports are globally known only after the next-round store
        # snapshot. Do not use this as an atomic reservation; deterministic
        # distance/ID tie-breaking below handles occasional duplicate picks.
        return {(tile % self.W, tile // self.W) for tile in self.known_resources}

    # BFS over walkable tiles from pos; returns dist dict-ish via generator
    def bfs_first_step(self, pos, goal_fn, max_nodes=1200, unknown_ok=True):
        """BFS from pos avoiding enemy turret zones if possible."""
        r = self._bfs(pos, goal_fn, max_nodes, unknown_ok, True)
        if r is None:
            r = self._bfs(pos, goal_fn, max_nodes, unknown_ok, False)
        return r

    def _bfs(self, pos, goal_fn, max_nodes, unknown_ok, avoid):
        W, H = self.W, self.H
        danger = self.danger
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
                if avoid and danger[j] and not goal_fn(j):
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
        known = self.known_resources
        best = None
        bestd = 10**9
        core = self.core
        for i in range(W * self.H):
            e = known.get(i, env[i])
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
                self.bld[ty * self.W + tx] = B_OWN_HARV
                self.replans = 0
                self.target = None
                if self.jobs:
                    self.do_jobs(ct, pos)
            else:
                self.fail[self.target] = self.fail.get(self.target, 0) + 10
            return
        if not self.move_adjacent_to(ct, pos, tx, ty):
            self.fail[self.target] = self.fail.get(self.target, 0) + 1

    def good_conveyors(self):
        """Map own conveyor tile -> trunk end tile, for chains that reach our core."""
        W, H = self.W, self.H
        bld = self.bld
        state = {}  # i -> end tile index or -1
        for i0 in range(W * H):
            if bld[i0] != B_OWN_CONV or i0 in state:
                continue
            trail = []
            onpath = set()
            i = i0
            res = -1
            last = -1
            while True:
                if i in state:
                    res = state[i]
                    break
                b = bld[i]
                if b == B_CORE:
                    res = last
                    break
                if b != B_OWN_CONV or i in onpath:
                    break
                d = self.bdir.get(i)
                if d is None:
                    break
                trail.append(i)
                onpath.add(i)
                last = i
                dx, dy = d.delta()
                x, y = i % W + dx, i // W + dy
                if not (0 <= x < W and 0 <= y < H):
                    break
                i = y * W + x
            for t in trail:
                state[t] = res
        return {i: v for i, v in state.items() if v >= 0}

    def trunk_loads(self, good):
        W, H = self.W, self.H
        bld = self.bld
        load = {}
        for i in range(W * H):
            if bld[i] != B_OWN_HARV:
                continue
            x, y = i % W, i // W
            for dx, dy in CARD_DELTA:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    j = ny * W + nx
                    if j in good:
                        t = good[j]
                        load[t] = load.get(t, 0) + 1
                        break
        return load

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
        load = self.trunk_loads(good)

        def sink_ok(j, from_i):
            b = bld[j]
            if b == B_CORE:
                return True
            if j in good and load.get(good[j], 0) < MAX_TRUNK:
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
    def attack_adjacent(self, ct, pos, min_ti=300):
        if ct.get_action_cooldown() != 0:
            return False
        me = self.me
        for d in CARDINALS:
            p = pos.add(d)
            if not self.inb(p.x, p.y):
                continue
            bid = ct.get_tile_building_id(p)
            if bid is not None and ct.get_team(bid) != me and ct.get_entity_type(bid) != EntityType.CORE:
                if ct.get_global_resources() > min_ti and ct.can_fire(p):
                    ct.fire(p)
                    return True
        return False

    def is_guard(self, ct):
        r = ct.get_current_round()
        gid = ct.read_store(S_GUARD)
        beat = ct.read_store(S_GUARD_BEAT)
        me = ct.get_id() & 0xFFFFFF
        if gid == me:
            ct.write_store(S_GUARD_BEAT, r)
            return True
        if gid == 0 or beat + 3 < r:
            if not self.jobs and self.target is None:
                ct.write_store(S_GUARD, me)
                ct.write_store(S_GUARD_BEAT, r)
        return False

    def guard(self, ct, pos):
        core = self.core
        W = self.W
        foot = {(core.x + dx, core.y + dy) for dx in (0, 1) for dy in (0, 1)}
        entry = set()
        for fx, fy in foot:
            for dx, dy in CARD_DELTA:
                t = (fx + dx, fy + dy)
                if t not in foot:
                    entry.add(t)
        if ct.read_store(S_PHASE) & 2 and self.heal_core(ct, pos):
            return
        if ct.get_action_cooldown() == 0 and self.attack_adjacent(ct, pos, 20):
            return
        ti = ct.get_global_resources()
        # lay mines on free tiles near the core, away from trunk entries
        if ti >= 60:
            cand = []
            for x, y in ((core.x - 1, core.y - 1), (core.x + 2, core.y - 1), (core.x - 1, core.y + 2), (core.x + 2, core.y + 2)):
                    if not self.inb(x, y):
                        continue
                    i = y * W + x
                    if self.bld[i] == B_NONE and self.env[i] == EMPTY and (x, y) not in self.mines:
                        cand.append((abs(x - pos.x) + abs(y - pos.y), x, y))
            if cand and len(self.mines) < 24:
                cand.sort()
                for _, x, y in cand:
                    if abs(x - pos.x) + abs(y - pos.y) == 1:
                        if ct.get_action_cooldown() == 0:
                            mp = Position(x, y)
                            if ct.can_build_mine(mp):
                                ct.build_mine(mp)
                                self.mines.add((x, y))
                                self.bld[y * W + x] = B_OWN_OTHER
                            else:
                                self.mines.add((x, y))  # unbuildable, skip
                        return
                _, x, y = cand[0]
                if self.move_adjacent_to(ct, pos, x, y):
                    return
        # idle: stay near the core
        if pos.distance_squared(core) > 8:
            res = self.bfs_first_step(pos, lambda i: (i % W - core.x) ** 2 + (i // W - core.y) ** 2 <= 8)
            if res:
                self.step(ct, pos, res[1])

    def heal_core(self, ct, pos):
        core = self.core
        if pos.distance_squared(core) > 50 or ct.get_global_resources() < 5:
            return False
        W = self.W
        adj = set()
        for dx in (0, 1):
            for dy in (0, 1):
                fx, fy = core.x + dx, core.y + dy
                for ddx, ddy in CARD_DELTA:
                    x, y = fx + ddx, fy + ddy
                    if self.inb(x, y) and self.bld[y * W + x] != B_CORE:
                        adj.add(y * W + x)
        if pos.y * W + pos.x in adj:
            if ct.get_action_cooldown() != 0:
                return True
            for d in CARDINALS:
                q = pos.add(d)
                if self.inb(q.x, q.y) and self.bld[q.y * W + q.x] == B_CORE and ct.can_heal(q):
                    ct.heal(q)
                    return True
            return False
        res = self.bfs_first_step(pos, lambda i: i in adj)
        if res is None:
            return False
        self.step(ct, pos, res[1])
        return True

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
            if self.attack_adjacent(ct, pos, 20):
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
        if ct.get_global_resources() < 300:
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
