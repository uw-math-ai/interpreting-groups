#!/bin/bash
# SU(2) vs SO(3) pilot: locate the grokking regime and get the first
# spin-spectrum comparison between the matched twins.
# 4 concurrent runs on one GPU (model is tiny / python-bound).
set -u
cd "$(dirname "$0")/.."

N=2048
STEPS=200000
SEED=0

for variant in su2 so3; do
  for frac in 0.002 0.01; do
    dir="data/su2/${variant}_N${N}_f${frac}_s${SEED}"
    mkdir -p "$dir"
    echo "launching $dir"
    nohup uv run python -m experiments.su2_train \
      --variant "$variant" --N "$N" --frac "$frac" --seed "$SEED" \
      --steps "$STEPS" --save_dir "$dir" \
      > "$dir/stdout.log" 2>&1 &
    sleep 2
  done
done
wait
echo "pilot complete"
