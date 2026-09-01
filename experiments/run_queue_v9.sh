#!/bin/bash
# Queue v9: preregistered P1B regularization sweep (Stage 1) followed by
# capacity controls (Stage 2). See preregistration/p1b_wd_sweep_2026-09-01.md.
set -u
cd "$(dirname "$0")/.."
CMDS=()

# Stage 1: wd sweep, 3 new wd x {frame, block} x 5 seeds
for wd in 0.1 0.3 3.0; do
  for s in 0 1 2 3 4; do
    for enc in so3frame so3block; do
      d="data/su2/so3_${enc}_wd${wd}_f0.01_s${s}"
      CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant so3 --encoding $enc --N 2048 --frac 0.01 --seed $s --wd $wd --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
    done
  done
done

# Stage 2: capacity controls at wd 1.0, both arms
for s in 0 1 2 3 4; do
  for enc in so3frame so3block; do
    d="data/su2/so3_${enc}_cap1024x1_f0.01_s${s}"
    CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant so3 --encoding $enc --N 2048 --frac 0.01 --seed $s --wd 1.0 --mlp_hidden 1024 --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
    d="data/su2/so3_${enc}_cap512x2_f0.01_s${s}"
    CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant so3 --encoding $enc --N 2048 --frac 0.01 --seed $s --wd 1.0 --mlp_depth 2 --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done

echo "${#CMDS[@]} jobs queued"
CONC=4
running=0
for cmd in "${CMDS[@]}"; do
  bash -c "$cmd" &
  running=$((running+1))
  if [ "$running" -ge "$CONC" ]; then
    wait -n
    running=$((running-1))
  fi
done
wait
echo "queue v9 complete"
