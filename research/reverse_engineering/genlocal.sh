#!/bin/bash
# official-engine replays with known seeds
cd /home/user/florent_code
i=0
for pair in "econ starter" "v1 econ" "v2 v1" "starter v2" "econ econ" "v1 starter"; do
  set -- $pair
  for m in gen01 gen03 gen05 pit00 s7_gen02; do
    seed=$((i * 7 + 3)); i=$((i+1))
    /tmp/claude-0/venv/bin/fcode run $1 $2 maps/$m.map26 --seed $seed --mark 0 --replay /tmp/claude-0/localreps/${1}_${2}_${m}_s${seed}.replay26 >/dev/null 2>&1 &
    if (( i % 4 == 0 )); then wait; fi
  done
done
wait
