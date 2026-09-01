#!/bin/bash
# Queue v10 (G5.3): P4 replication to five seeds per retained cell, launched
# only after the calibrated Haar estimator passed (tests/test_hk_haar.py).
set -u
cd "$(dirname "$0")/.."
CMDS=()
for s in 3 4; do
  for t in 0.3 0.1 0.05; do
    d="data/su2/su2hk_t${t}_N2048_f0.01_s${s}"
    CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant su2 --loss heatkernel --hk_t $t --spec_jmax 4 --N 2048 --frac 0.01 --seed $s --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done
for s in 1 2 3 4; do
  d="data/su2/su2bl_j1j2_f0.01_s${s}"
  CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant su2 --loss bandlim --bl_js 1.0 2.0 --bl_coefs 1.0 0.5 --spec_jmax 3 --N 2048 --frac 0.01 --seed $s --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  d="data/su2/su2bl_j1j2_wd0.1_f0.01_s${s}"
  CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant su2 --loss bandlim --bl_js 1.0 2.0 --bl_coefs 1.0 0.5 --spec_jmax 3 --wd 0.1 --N 2048 --frac 0.01 --seed $s --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
done
echo "${#CMDS[@]} jobs queued"
CONC=4
running=0
for cmd in "${CMDS[@]}"; do
  bash -c "$cmd" &
  running=$((running+1))
  if [ "$running" -ge "$CONC" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "queue v10 complete"
