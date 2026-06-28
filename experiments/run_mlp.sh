#!/usr/bin/env bash
# MLP-ONLY grokking sweep (no attention) over all groups, parallel to the
# transformer runs. Low train split (0.3), eval = the rest of the group (0.7),
# generous post-grok purification (>=20k epochs via *_grok20k.pt), detached.
# Sequential (concurrency is net-negative on the T4).
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-800000}
TF=${TF:-0.3}
EF=$(awk "BEGIN{printf \"%.2f\", 1-$TF}")
EXTRA=${EXTRA:-20000}
SEED=${SEED:-1}
TASKS=${TASKS:-"icosahedral 2I A5 2O S5 2T"}
MASTER=logs/mlp_sweep.log

echo "=== MLP sweep start $(date -u) epochs=$EPOCHS tf=$TF ef=$EF extra_after_grok=$EXTRA ===" | tee -a "$MASTER"
for TASK in $TASKS; do
  SD="data/mlp/${TASK}_s${SEED}"
  LOG="logs/mlp_${TASK}.log"
  echo ">>> [$(date -u +%H:%M:%S)] START $TASK -> $SD" | tee -a "$MASTER"
  uv run python -m experiments.mlp_train \
    --task "$TASK" --epochs "$EPOCHS" \
    --eval_every 1000 --checkpoint_every 10000 \
    --train_frac "$TF" --eval_frac "$EF" \
    --grok_thresh 0.95 --extra_after_grok "$EXTRA" \
    --weight_decay 1.0 --lr 1e-3 --seed "$SEED" \
    --save_dir "$SD" --force > "$LOG" 2>&1
  echo ">>> [$(date -u +%H:%M:%S)] END   $TASK rc=$? $(grep -oE 'best_eval=[0-9.]+' "$LOG" | tail -1)" | tee -a "$MASTER"
done
echo "=== MLP sweep done $(date -u) ===" | tee -a "$MASTER"
