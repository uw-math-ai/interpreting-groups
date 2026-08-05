#!/bin/bash
# Managed queue: K sweep -> training-fraction sweep (delayed-generalization
# search) -> seed replication. Concurrency-limited; launch detached via setsid.
set -u
cd "$(dirname "$0")/.."
mkdir -p data/su2

CMDS=()

# --- K sweep (effect of candidate count), f=0.01, seed 0, 60k steps ---------
for v in su2 so3; do
  for K in 128 256; do
    b=4096; [ "$K" = 256 ] && b=2048
    d="data/su2/${v}_N2048_f0.01_K${K}_s0"
    CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant $v --N 2048 --frac 0.01 --K $K --batch $b --seed 0 --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done

# --- training-fraction sweep at low f (delayed-generalization search) --------
# 300k steps, fine eval grid
for v in su2 so3; do
  for f in 0.001 0.0005 0.0002; do
    d="data/su2/${v}_N2048_f${f}_s0"
    CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant $v --N 2048 --frac $f --seed 0 --steps 300000 --eval_every 500 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done

# --- seed replication of the pilot cells (seeds 1-4, 100k steps) -------------
for s in 1 2 3 4; do
  for v in su2 so3; do
    for f in 0.002 0.01; do
      d="data/su2/${v}_N2048_f${f}_s${s}"
      CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant $v --N 2048 --frac $f --seed $s --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
    done
  done
done

printf '%s\n' "${CMDS[@]}" > data/su2/queue_cmds.txt
echo "queue: ${#CMDS[@]} runs, concurrency 5"
printf '%s\n' "${CMDS[@]}" | xargs -P 5 -I CMD bash -c CMD
echo "QUEUE_DONE" > data/su2/QUEUE_DONE
