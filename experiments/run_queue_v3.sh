#!/bin/bash
# Queue v3: observation-map robustness (A3), heat-kernel bandwidth (D1),
# SU(3) vs PSU(3) pilot (B). Concurrency 3.
set -u
cd "$(dirname "$0")/.."
mkdir -p data/su2 data/su3

CMDS=()

# --- A3: observation-map robustness (60k steps) ------------------------------
for encv in randmlp funcval; do
  for v in su2 so3; do
    d="data/su2/${v}_N2048_f0.01_${encv}_s0"
    CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant $v --encoding $encv --N 2048 --frac 0.01 --seed 0 --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done

# --- D1: heat-kernel regression, temperature sweep (60k steps) ---------------
for t in 0.3 0.1 0.05; do
  d="data/su2/su2hk_t${t}_N2048_f0.01_s0"
  CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant su2 --loss heatkernel --hk_t $t --spec_jmax 4 --N 2048 --frac 0.01 --seed 0 --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
done

# --- B: SU(3) vs PSU(3) pilot (100k steps) -----------------------------------
for v in su3 psu3; do
  d="data/su3/${v}_N2048_f0.01_s0"
  CMDS+=("mkdir -p $d && uv run python -m experiments.su3_train --variant $v --N 2048 --frac 0.01 --seed 0 --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
done

printf '%s\n' "${CMDS[@]}" > data/su2/queue_v3_cmds.txt
echo "queue v3: ${#CMDS[@]} runs, concurrency 3"
printf '%s\n' "${CMDS[@]}" | xargs -P 3 -I CMD bash -c CMD
echo "QUEUE_V3_DONE" > data/su2/QUEUE_V3_DONE
