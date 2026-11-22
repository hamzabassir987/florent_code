#!/bin/bash
mkdir -p /tmp/claude-0/ladder
cd /home/user/florent_code/engine/tools
for f in /tmp/claude-0/reps/test/*.replay26; do
  b=$(basename $f .replay26)
  ls /tmp/claude-0/ladder/${b}_s*.replay26 >/dev/null 2>&1 && continue
  ids=$(python3 -c "
from replay import Replay
r=Replay('$f')
print(*[u[1]['id'] for t in r.turns for u in t if u[0]=='place'][:2])")
  s=$(/tmp/claude-0/seedsearch $ids 0 4294967296 | grep SEED | awk '{print $2}')
  echo "$b $s"
  [ -n "$s" ] && cp $f /tmp/claude-0/ladder/${b}_s$s.replay26
done
echo DONE
