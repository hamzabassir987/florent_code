#!/bin/bash
cd /home/user/florent_code
i=0
for m in gen00 gen02 gen04 gen06 gen08 pit00 pit01 pit02 pit03 s7_pit00 s7_pit01 s7_gen05; do
  for pair in "chaos econ" "econ chaos"; do
    set -- $pair
    seed=$((1000 + i)); i=$((i+1))
    /tmp/claude-0/venv/bin/fcode run $1 $2 maps/$m.map26 --seed $seed --mark 0 --replay /tmp/claude-0/chaosreps/${1}_${2}_${m}_s${seed}.replay26 >/dev/null 2>&1 &
    if (( i % 4 == 0 )); then wait; fi
  done
done
wait
