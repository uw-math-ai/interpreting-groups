#!/bin/bash
# Queue v8 (v2 plan): P1B frame-vs-block matched training; P4 calibration +
# partial hk seed replication; P2 SO(5) conditioning stress test.
set -u
cd "$(dirname "$0")/.."
CMDS=()

# P1B: SO(3) explicit-feature runs, 5 seeds x {frame 6d, block 9d}, 60k
for s in 0 1 2 3 4; do
  for enc in so3frame so3block; do
    d="data/su2/so3_${enc}_f0.01_s${s}"
    CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant so3 --encoding $enc --N 2048 --frac 0.01 --seed $s --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done

# P4 calibration: band-limited target with known support {1, 2}
d="data/su2/su2bl_j1j2_f0.01_s0"
CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant su2 --loss bandlim --bl_js 1.0 2.0 --bl_coefs 1.0 0.5 --spec_jmax 3 --N 2048 --frac 0.01 --seed 0 --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")

# P4 partial hk seed replication (seeds 1-2 per temperature)
for s in 1 2; do
  for t in 0.3 0.1 0.05; do
    d="data/su2/su2hk_t${t}_N2048_f0.01_s${s}"
    CMDS+=("mkdir -p $d && uv run python -m experiments.su2_train --variant su2 --loss heatkernel --hk_t $t --spec_jmax 4 --N 2048 --frac 0.01 --seed $s --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done

# P2: Spin(5)/SO(5) seeds 1-4 + SO(5) conditioning variants
for s in 1 2 3 4; do
  for v in spin5 so5; do
    d="data/spin5/${v}_N2048_f0.01_s${s}"
    CMDS+=("mkdir -p $d && uv run python -m experiments.spin5_train --variant $v --N 2048 --frac 0.01 --seed $s --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
  done
done
d="data/spin5/so5_N2048_f0.01_K256_s0"
CMDS+=("mkdir -p $d && uv run python -m experiments.spin5_train --variant so5 --N 2048 --frac 0.01 --K 256 --batch 1024 --seed 0 --steps 60000 --save_dir $d --resume >> $d/stdout.log 2>&1")
d="data/spin5/so5_N2048_f0.03_s0"
CMDS+=("mkdir -p $d && uv run python -m experiments.spin5_train --variant so5 --N 2048 --frac 0.03 --seed 0 --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
d="data/spin5/so5_N2048_f0.01_wd0.1_s0"
CMDS+=("mkdir -p $d && uv run python -m experiments.spin5_train --variant so5 --N 2048 --frac 0.01 --wd 0.1 --seed 0 --steps 100000 --save_dir $d --resume >> $d/stdout.log 2>&1")
d="data/spin5/so5_N2048_f0.01_s0"
CMDS+=("uv run python -m experiments.spin5_train --variant so5 --N 2048 --frac 0.01 --seed 0 --steps 200000 --save_dir $d --resume >> $d/stdout.log 2>&1")

printf '%s\n' "${CMDS[@]}" > data/queue_v8_cmds.txt
echo "queue v8: ${#CMDS[@]} runs, concurrency 3"
printf '%s\n' "${CMDS[@]}" | xargs -P 3 -I CMD bash -c CMD
echo "QUEUE_V8_DONE" > data/QUEUE_V8_DONE
