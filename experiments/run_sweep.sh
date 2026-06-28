#!/usr/bin/env bash
# Detached aggressive grokking sweep for I_h = A5 x C2 (order 120) on the T4.
# Sequential (concurrency is net-negative on this launch-bound tiny model);
# uses the lean CUDA driver (fused AdamW + TF32 + compile, no per-epoch host sync).
#
# Configs run flagship-first: tf=0.6 is the best bet to actually generalize, so
# we get an interpretable grokked checkpoint ASAP, then widen coverage.
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-500000}
EVAL_EVERY=${EVAL_EVERY:-500}
CKPT_EVERY=${CKPT_EVERY:-5000}
SEED=${SEED:-1}
MASTER=logs/sweep.log
echo "=== sweep start $(date -u) epochs=$EPOCHS seed=$SEED ===" | tee -a "$MASTER"

for TF in 0.6 0.5 0.7 0.4; do
  SD="experiments/tf${TF}_s${SEED}"
  LOG="logs/train_tf${TF}_s${SEED}.log"
  echo ">>> [$(date -u +%H:%M:%S)] START tf=$TF seed=$SEED -> $SD" | tee -a "$MASTER"
  uv run python -m experiments.aws_train \
    --epochs "$EPOCHS" --eval_every "$EVAL_EVERY" --checkpoint_every "$CKPT_EVERY" \
    --train_frac "$TF" --eval_frac 0.2 --weight_decay 1.0 --lr 1e-3 \
    --seed "$SEED" --save_dir "$SD" > "$LOG" 2>&1
  RC=$?
  BEST=$(grep -oE "best_eval=[0-9.]+" "$LOG" | tail -1)
  echo ">>> [$(date -u +%H:%M:%S)] END   tf=$TF seed=$SEED rc=$RC $BEST" | tee -a "$MASTER"
done
echo "=== sweep done $(date -u) ===" | tee -a "$MASTER"
