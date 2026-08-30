#!/bin/bash
# Queue v7: Spin(5)/SO(5) pilot (pre-registered: spinor 4 vs vector 5) and the
# heat-kernel wide-model retune (2x width failed at wd=1.0; test wd=0.1 at two
# learning rates).
set -u
cd "$(dirname "$0")/.."
mkdir -p data/spin5

CMDS=()
for v in spin5 so5; do
  d="data/spin5/${v}_N2048_f0.01_s0"
  CMDS+=("mkdir -p $d && uv run python -m experiments.spin5_train --variant $v --N 2048 --frac 0.01 --seed 0 --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
done
for lr in 1e-3 3e-4; do
  d="data/su2/su2hk_t0.05_wide_lr${lr}_wd0.1_s0"
  CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant su2 --loss heatkernel --hk_t 0.05 --spec_jmax 4 --d_enc 256 --enc_hidden 512 --mlp_hidden 1024 --lr $lr --wd 0.1 --N 2048 --frac 0.01 --seed 0 --steps 120000 --save_dir $d --resume >> $d/stdout.log 2>&1")
done

printf '%s\n' "${CMDS[@]}" > data/spin5/queue_v7_cmds.txt
echo "queue v7: ${#CMDS[@]} runs, concurrency 3"
printf '%s\n' "${CMDS[@]}" | xargs -P 3 -I CMD bash -c CMD
echo "QUEUE_V7_DONE" > data/spin5/QUEUE_V7_DONE
