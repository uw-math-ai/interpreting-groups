#!/bin/bash
# Queue v5: Spin(4) central-quotient pilot -- four conditions, one seed.
# Hypotheses recorded in core/spin4.py BEFORE running: spin4 -> (1/2,0)+(0,1/2)
# forced sum; so4 -> (1/2,1/2) vector rep; leftq -> (1,0)+(0,1/2) [5d, 2 blocks]
# vs (1,1/2) [6d, 1 block]; pso4 -> (1,0)+(0,1) [6d] vs (1,1) [9d].
set -u
cd "$(dirname "$0")/.."
mkdir -p data/spin4

CMDS=()
for v in spin4 so4 leftq pso4; do
  d="data/spin4/${v}_N2048_f0.01_s0"
  CMDS+=("mkdir -p $d && uv run python -m experiments.spin4_train --variant $v --N 2048 --frac 0.01 --seed 0 --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
done

printf '%s\n' "${CMDS[@]}" > data/spin4/queue_v5_cmds.txt
echo "queue v5: ${#CMDS[@]} runs, concurrency 2"
printf '%s\n' "${CMDS[@]}" | xargs -P 2 -I CMD bash -c CMD
echo "QUEUE_V5_DONE" > data/spin4/QUEUE_V5_DONE
