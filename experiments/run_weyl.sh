#!/usr/bin/env bash
# Grokking sweep over the Weyl/exotic groups (MLP-only: faster + groks at lower
# data, which matters for the big ones). Per-group train fraction (smaller groups
# need more data), generous crystallization (+30k), snapshots for GIFs. Sequential.
set -uo pipefail
cd "$(dirname "$0")/.."
SEED=${SEED:-1}
MASTER=logs/weyl_sweep.log

declare -A TF=( [WG2]=0.7 [WA3]=0.6 [WB3]=0.5 [PSL27]=0.5 [WD4]=0.5 [A6]=0.4 [WB4]=0.4 )
declare -A EP=( [WG2]=200000 [WA3]=200000 [WB3]=300000 [PSL27]=400000 [WD4]=400000 [A6]=500000 [WB4]=500000 )
ORDER=${ORDER:-"WG2 WA3 WB3 PSL27 WD4 A6 WB4"}

echo "=== weyl sweep start $(date -u) ===" | tee -a "$MASTER"
for TASK in $ORDER; do
  tf=${TF[$TASK]}; ep=${EP[$TASK]}; ef=$(awk "BEGIN{printf \"%.2f\",1-$tf}")
  SD="data/weyl/${TASK}_s${SEED}"; LOG="logs/weyl_${TASK}.log"
  echo ">>> [$(date -u +%H:%M:%S)] START $TASK (tf$tf ep$ep) -> $SD" | tee -a "$MASTER"
  uv run python -m experiments.mlp_train \
    --task "$TASK" --epochs "$ep" --eval_every 1000 --checkpoint_every 10000 \
    --snapshot_every 4000 --grok_thresh 0.95 --extra_after_grok 30000 \
    --train_frac "$tf" --eval_frac "$ef" --weight_decay 1.0 --lr 1e-3 \
    --seed "$SEED" --save_dir "$SD" --force > "$LOG" 2>&1
  echo ">>> [$(date -u +%H:%M:%S)] END   $TASK rc=$? $(grep -oE 'best_eval=[0-9.]+' "$LOG" | tail -1)" | tee -a "$MASTER"
done
echo "=== weyl sweep done $(date -u) ===" | tee -a "$MASTER"
