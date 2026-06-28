#!/usr/bin/env bash
# Detached grokking sweep over order-~120 groups (and relatives), sequential.
# Flagship-first: 2I (600-cell) and S5 (A4 roots) are the headline targets.
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-300000}
EVAL_EVERY=${EVAL_EVERY:-500}
CKPT_EVERY=${CKPT_EVERY:-5000}
TF=${TF:-0.6}
SEED=${SEED:-1}
TASKS=${TASKS:-"2I S5 A5 2O 2T"}
MASTER=logs/groups.log
echo "=== group sweep start $(date -u) epochs=$EPOCHS tf=$TF seed=$SEED tasks=[$TASKS] ===" | tee -a "$MASTER"

for TASK in $TASKS; do
  SD="experiments/${TASK}_s${SEED}"
  LOG="logs/train_${TASK}.log"
  echo ">>> [$(date -u +%H:%M:%S)] START $TASK -> $SD" | tee -a "$MASTER"
  uv run python -m experiments.aws_train \
    --task "$TASK" \
    --epochs "$EPOCHS" --eval_every "$EVAL_EVERY" --checkpoint_every "$CKPT_EVERY" \
    --train_frac "$TF" --eval_frac 0.2 --weight_decay 1.0 --lr 1e-3 \
    --seed "$SEED" --save_dir "$SD" > "$LOG" 2>&1
  RC=$?
  BEST=$(grep -oE "best_eval=[0-9.]+" "$LOG" | tail -1)
  echo ">>> [$(date -u +%H:%M:%S)] END   $TASK rc=$RC $BEST" | tee -a "$MASTER"
done
echo "=== group sweep done $(date -u) ===" | tee -a "$MASTER"
