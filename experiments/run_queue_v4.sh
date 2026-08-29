#!/bin/bash
# Queue v4: SU(3)/PSU(3) seed replication (seeds 1-4) + D1 capacity probes
# (is the heat-kernel spectral cutoff capacity- or optimization-limited?).
set -u
cd "$(dirname "$0")/.."
mkdir -p data/su3

CMDS=()
for s in 1 2 3 4; do
  for v in su3 psu3; do
    d="data/su3/${v}_N2048_f0.01_s${s}"
    CMDS+=("mkdir -p $d && uv run python -m experiments.su3_train --variant $v --N 2048 --frac 0.01 --seed $s --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done

d="data/su2/su2hk_t0.05_wide_N2048_f0.01_s0"
CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant su2 --loss heatkernel --hk_t 0.05 --spec_jmax 4 --d_enc 256 --enc_hidden 512 --mlp_hidden 1024 --N 2048 --frac 0.01 --seed 0 --steps 120000 --save_dir $d --resume >> $d/stdout.log 2>&1")
d="data/su2/su2hk_t0.05_long_N2048_f0.01_s0"
CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant su2 --loss heatkernel --hk_t 0.05 --spec_jmax 4 --N 2048 --frac 0.01 --seed 0 --steps 120000 --save_dir $d --resume >> $d/stdout.log 2>&1")

printf '%s\n' "${CMDS[@]}" > data/su3/queue_v4_cmds.txt
echo "queue v4: ${#CMDS[@]} runs, concurrency 3"
printf '%s\n' "${CMDS[@]}" | xargs -P 3 -I CMD bash -c CMD
echo "QUEUE_V4_DONE" > data/su3/QUEUE_V4_DONE
