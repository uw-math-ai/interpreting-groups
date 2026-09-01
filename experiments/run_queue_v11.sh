#!/bin/bash
# Queue v11 (G4): preregistered SO(5) independent-observation runs.
# See preregistration/so5_observations_2026-09-01.md (committed first).
set -u
cd "$(dirname "$0")/.."
CMDS=()
for s in 0 1 2 3 4; do
  d="data/spin5/so5rf_f0.03_s${s}"
  CMDS+=("mkdir -p $d && uv run python -m experiments.spin5_train --variant so5 --encoding so5rf --N 2048 --frac 0.03 --seed $s --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
done
for s in 0 1; do
  d="data/spin5/so5vec_f0.03_s${s}"
  CMDS+=("mkdir -p $d && uv run python -m experiments.spin5_train --variant so5 --encoding so5vec --N 2048 --frac 0.03 --seed $s --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
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
echo "queue v11 complete"
