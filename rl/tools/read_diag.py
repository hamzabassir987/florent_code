"""Decode the bot's diagnostic text from a replay's team store (bots/econ/main.py _diag).

    python rl/tools/read_diag.py game.replay26
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "engine", "tools"))
from replay import Replay  # noqa: E402

r = Replay(sys.argv[1])
for tm, idx in ((0, 1), (1, 2)):
    chunks = {}
    for ups in r.turns:
        for u in ups:
            if u[0] == "players":
                for w in u[idx].get("comm", []):
                    w &= 0xFFFFFFFF
                    c = bytes([(w >> 16) & 255, (w >> 8) & 255, w & 255])
                    if all(32 <= x < 127 for x in c):
                        chunks.setdefault(w >> 24, c)
    if chunks:
        text = b"".join(chunks.get(j, b"???") for j in range(max(chunks) + 1)).decode()
        print("team", tm, ":", text)
