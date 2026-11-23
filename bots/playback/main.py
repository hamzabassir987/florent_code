"""Playback bot: re-issues the actions of a logged native-engine game
(rl/train/game_log.hpp) through the official controller, so `fcode run`
produces the official replay. Driven by rl/tools/make_replay.py; the log path
comes from FLORENT_PLAYBACK.

Each logged action is "round entity_id kind a b c d" (rl/sim/florent_actions.hpp
ActionKind). If the logged game was cut by a shortened round limit, the team
that lost the cut resigns at the start of the next round, so the official
result matches the training game.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "econ"))
import heads as HEADS  # noqa: E402

from fcode import Team  # noqa: E402

NAME = {1: "move", 2: "spawn", 3: "convert", 4: "build", 5: "attack", 6: "heal",
        7: "destroy", 8: "fire", 9: "rotate", 10: "launch"}
ARITY = {"move": 1, "spawn": 2, "convert": 1, "build": 4, "attack": 2, "heal": 2,
         "destroy": 2, "fire": 2, "rotate": 1, "launch": 4}
CUT_CONDITIONS = ("titanium_collected", "harvesters", "titanium_stored", "coinflip")


def _load(path):
    hdr, acts = {}, {}
    with open(path) as f:
        assert f.readline().strip() == "florent-game-v1"
        for line in f:
            p = line.split()
            if p[0] == "actions":
                break
            hdr[p[0]] = p[1:]
        for line in f:
            r, i, k, a, b, c, d = map(int, line.split())
            name = NAME[k]
            acts.setdefault((r, i), []).append((name,) + (a, b, c, d)[:ARITY[name]])
    return hdr, acts


HDR, ACTS = _load(os.environ["FLORENT_PLAYBACK"])
ROUNDS = int(HDR["rounds"][0])
WINNER = int(HDR["winner"][0])
COND = HDR["winner"][1] if len(HDR["winner"]) > 1 else ""
RESIGN = COND in CUT_CONDITIONS and ROUNDS < 1000   # cut early: loser resigns after the cut
LOSER = Team.B if WINNER == 0 else Team.A


class Player:
    def __init__(self):
        pass

    def run(self, ct):
        r = ct.get_current_round()
        if RESIGN and r >= ROUNDS:
            if ct.get_team() == LOSER:
                ct.resign("round limit of the training game")
            return
        todo = ACTS.get((r, ct.get_id()))
        if not todo:
            return
        api = HEADS.Api(ct)
        for a in todo:
            try:
                HEADS.apply(api, a)
            except Exception as e:  # reported by make_replay.py
                print("PBFAIL", r, ct.get_id(), a, repr(e))
