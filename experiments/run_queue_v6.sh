#!/bin/bash
# Queue v6: Spin(4) seed replication (seeds 1-4, four conditions).
set -u
cd "$(dirname "$0")/.."
CMDS=()
for s in 1 2 3 4; do
  for v in spin4 so4 leftq pso4; do
    d="data/spin4/${v}_N2048_f0.01_s${s}"
    CMDS+=("mkdir -p $d && uv run python -m experiments.spin4_train --variant $v --N 2048 --frac 0.01 --seed $s --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done
printf '%s\n' "${CMDS[@]}" > data/spin4/queue_v6_cmds.txt
echo "queue v6: ${#CMDS[@]} runs, concurrency 3"
printf '%s\n' "${CMDS[@]}" | xargs -P 3 -I CMD bash -c CMD
echo "QUEUE_V6_DONE" > data/spin4/QUEUE_V6_DONE
