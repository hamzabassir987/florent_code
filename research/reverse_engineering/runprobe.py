import sys
sys.path.insert(0, '/tmp/claude-0/venv/lib/python3.13/site-packages')
from fcode.fcode_engine import run_game
import os
d = sys.argv[1]
bot = os.path.join(d, 'bot', 'main.py')
r = run_game(bot, bot, '/tmp/claude-0/venv/lib/python3.13/site-packages', os.path.join(d, 'map.map26'), os.path.join(d, 'gdb.replay26'), 1, 0, 0)
print(r['winner'], r['turns'])
os._exit(0)
