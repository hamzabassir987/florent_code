# Observation encoding

The actor observation is the only feature set intended for deployment. Both [C++](../engine/src/obs.cpp) and [Python](../bots/econ/encoder.py) implementations use only information available through that unit's controller plus its own memory. The Python module sits alongside `bots/econ/main.py` and can be imported as a sibling module; any submission archive must include both files. The existing heuristic `main.py` does not call the encoder yet. The training-only critic sees the full board once per round.

Parity is checked by `python3 engine/tools/obs_parity.py`.

## Actor observation

A 17×17 patch centred on the acting unit, from its own fog memory, plus edge-ray summaries. Buffers are `int32`, channel-major:

`index = channel * 17*17 + (dy + 8) * 17 + (dx + 8)`

where `dx,dy ∈ [-8,8]`. Off-map cells are all-zero. The patch extends beyond every unit's vision, so remembered information and age are intentional parts of the input.

### Per-unit memory

Memory is per unit, not team-shared. Each unit stores map-sized arrays:

| Memory | Meaning |
|---|---|
| Terrain | Unknown, empty, wall, Ti ore, Vo ore, void. Written when visible, retained forever (immutable). |
| Building / builder ID, type, team | Last observed occupants. A visible empty tile clears its occupant; fog preserves it. Enemy mines remain invisible and are not recorded. |
| `seen_round` | Last round the tile was visible. |
| `visit_round` | Last round this unit **observed a builder bot on the tile**. This is intentionally not the hidden world's exact last step; a later visit while the tile is unseen cannot be known in production. A visible empty tile does not erase the historical timestamp. |
| Building direction and HP percent; builder HP percent | Last values the controller exposed while the entity was visible. Cleared when a visible tile no longer contains that entity. |
| Attribute observation round | Used to express how old the remembered direction / HP is. |

Update only over the controller's `get_nearby_tiles()` and its tile/entity queries. Production uses `get_direction(id)`, `get_hp(id)`, and `get_max_hp(id)`, which are already used by the repository's bots. Per-entity cooldown is deliberately omitted because no production query for another entity's cooldown has been established. The acting unit's own cooldowns remain in the scalar vector.

To use the production encoder, keep a single `ActorEncoder` on the persistent `Player` instance and call it from `run`; it keys its fog memory by `ct.get_id()` so separate units do not share observations:

```python
from encoder import ActorEncoder

class Player:
    def __init__(self):
        self.encoder = ActorEncoder()

    def run(self, ct):
        spatial, scalars = self.encoder.encode(ct)
        # feed spatial/scalars to the policy
```

`ActorEncoder` includes no training-only methods or omniscient state.

### Spatial channels (59)

| Channel(s) | Content |
|---|---|
| 0–5 | Terrain one-hot: empty, wall, ore-Ti, ore-Vo, void, unknown |
| 6–16 | Own entity one-hot, index = engine `Type` (builder…mine) |
| 17–26 | Enemy entity one-hot, mine omitted (enemy mines are invisible) |
| 27–28 | Own / enemy builder occupancy (independent of building occupancy) |
| 29 | Tile staleness: rounds since last visible, capped 255; 255 = never |
| 30 | Age of last observed builder presence, capped 255; 255 = never |
| 31–32 | Age of remembered building / builder attributes, capped 255 |
| 33–41 | Building direction one-hot: centre, N, NE, E, SE, S, SW, W, NW. Builder direction is not needed (only builders move). |
| 42–43 | Building / builder HP percent (0–100, remembered) |
| 44–45 | Own / enemy remembered turret geometrically reaches this tile (possible reach) |
| 46–47 | Own / enemy turret reach with current direction and relevant line information observed this round (confirmed geometry) |
| 48–58 | Eleven edge-ray channels below; zero off the boundary |

A building and builder can coexist on one tile, so their memories and HP ages are distinct. A visible empty tile clears occupant attributes; historical visit time remains.

### Turret reach channels

These are **geometric cues, not exact `can_fire` answers**: the deployed controller cannot see hidden ammo, cooldowns, or enemy mines. The encoder uses remembered own and enemy turret type and direction. For gunners, known static walls/void and currently visible intervening occupants block the line; hidden or stale mutable blockers are treated as uncertain, not as proof the line is blocked. Sentinels ignore blockers; spikes reach adjacent tiles. Channels 44–45 are possible own/enemy reach; 46–47 are confirmed geometry when the turret direction is fresh and (for gunners) the relevant line is observed. On a known core footprint tile, these channels tell whether remembered friendly or enemy turret geometry covers it. They do not guarantee an actual shot can fire now.

### Edge rays (channels 48–58)

For each of the 64 boundary cells (`|dx|==8` or `|dy|==8`), cast radially outward from that cell, at most 32 tiles, stopping at the map edge. A step uses `p += (dx/L,dy/L)`, `L=max(|dx|,|dy|)`, rounded half-up. Rays use only this unit's remembered map. Values are raw counts / distances, never normalized.

| Offset | Value |
|---|---|
| +0 | Distance to first remembered wall/void; 0 = none |
| +1 | Distance to first unknown tile; 0 = none |
| +2, +3 | Ti ore / Vo ore tile counts |
| +4, +5 | Own / enemy buildings (unique entity IDs) |
| +6, +7 | Own / enemy units (unique IDs) |
| +8, +9 | Own / enemy harvesters |
| +10 | Unknown tile count |

### Scalar vector (40)

`0–5` actor type one-hot (builder, core, gunner, sentinel, launcher, spike); `6` own HP percent; `7–8` own action / move cooldown; `9` round; `10–14` own Ti, Vo, ammo, unavailable scale (0), unit count; `15–19` opponent fields are **zero**, because the production controller does not expose them; `20–35` own store slots 0–15; `36` a remembered own turret geometrically reaches a known enemy-core tile; `37` same with fresh geometry; `38–39` remembered enemy turret reaches own core, possible / confirmed. These core flags indicate remembered geometric reach, not guaranteed shots. The actor vector intentionally does not read engine-only scale.

## Critic observation (training only)

Whole board at native resolution padded to 48×48. It is omniscient, computed once per round, and shared by all units; it does not need visit timestamps because exact mine state is present.

- 29 spatial channels: terrain (5), team-0 entities (11), team-1 entities (11), builder occupancy (2).
- 33 scalars: round; both teams' Ti/Vo/ammo/scale/units; live per-type entity counts for both teams.

Maps larger than 48 in either dimension would be clipped; existing maps fit. No critic feature is shipped in the deployed actor.

## Verification status

On generated and pit-map simulations, the parity harness compares all actor channels/scalars from C++ and Python, checks out-of-vision perturbations, and checks critic determinism/change. Latest extended run: 8 generated/pit-map games × 250 rounds, **25,809 unit encodings**, **0 mismatches**, **0 leaks**, 8 visit-history checks, 8 critic checks, and a deterministic sentinel-to-core coverage case passed; Python encoding median 0.51 ms, p95 0.76 ms/unit. The official `fcode run --tle 10` check still requires the fcode runtime, which is not installed in this environment.

## Hit channel (channel 60, `C_HIT`)

Per tile: rounds since this unit saw the building or builder on it lose HP
(same entity id, lower HP percentage than at its previous look), capped at
`HIT_CAP` = 10; 10 also means "never seen hit". From the unit's own memory,
identical in `bots/econ/encoder.py`. Lets the network infer recent fire, e.g.
an enemy spike that just shot (4-round cooldown). Old 60-channel checkpoints:
`widen_input OLD.pt NEW.pt` (rl/tools/widen_input.cpp) copies every weight and
gives the new channel zero weights, so the converted net decides as before.

## Mine channel (C_MINE, channel 61; ACTOR_CH 62)

Known hidden enemy mines, 1 on the tile. A builder that can still act (action
cooldown 0, team Ti >= 2) probes its 4 orthogonal neighbours at the start of its
turn: an attack is legal on a tile with no visible building only if a hidden
enemy mine is there (the official `can_fire` only needs a building; checked in
fcode). A failed probe there, a visible building or our own builder on the tile
clears it. Known mines are shared on the team bus in G_EATK with code 5 (removal
with W_REMOVE). Same in engine/src/obs.cpp and bots/econ/encoder.py
(engine/tools/obs_parity.py counts encodings showing a mine). The student
quantizes it as a one-hot (255). A 61-channel checkpoint is converted with
`widen_input OLD.pt NEW.pt` (zero weights for the new channel: same decisions);
a 61-channel student.json still runs (the bot drops the last channel).
