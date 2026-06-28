#!/usr/bin/env bash
# Aggressive LOW-train-split sweep: force each model to learn the full group
# representation (not just memorize) so the decoded polytope is sharp.
#   * train_frac kept as low as plausibly grokkable per group
#   * eval_frac = 1 - train_frac  (held-out set covers the rest of the whole group)
#   * very long schedule so the (delayed) grok happens AND the rep purifies
# Sequential (concurrency is net-negative on the T4). Adapt fractions upward only
# if a group stays flat at chance (see logs/lowsplit.log + per-task logs).
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-800000}
SEED=${SEED:-1}
MASTER=logs/lowsplit.log

# starting train fractions (low; smaller groups need higher fractions since they
# have fewer ABSOLUTE training examples to pin the representation). A5@0.5 was
# confirmed below-threshold (flat at chance through 220k), hence A5/2O bumped.
declare -A TF=( ["2I"]=0.45 ["icosahedral"]=0.45 ["A5"]=0.6 ["2O"]=0.6 ["S5"]=0.7 ["2T"]=0.85 )
ORDER=${ORDER:-"2I A5 2O S5 2T"}

echo "=== lowsplit sweep start $(date -u) epochs=$EPOCHS seed=$SEED ===" | tee -a "$MASTER"
for TASK in $ORDER; do
  tf=${TF[$TASK]}
  ef=$(awk "BEGIN{printf \"%.3f\", 1-$tf}")
  SD="data/transformer/low_${TASK}_s1"
  LOG="logs/low_${TASK}.log"
  echo ">>> [$(date -u +%H:%M:%S)] START $TASK tf=$tf ef=$ef -> $SD" | tee -a "$MASTER"
  uv run python -m experiments.aws_train \
    --task "$TASK" --epochs "$EPOCHS" \
    --eval_every 1000 --checkpoint_every 10000 \
    --train_frac "$tf" --eval_frac "$ef" \
    --weight_decay 1.0 --lr 1e-3 --seed "$SEED" \
    --save_dir "$SD" --force > "$LOG" 2>&1
  echo ">>> [$(date -u +%H:%M:%S)] END   $TASK rc=$? $(grep -oE 'best_eval=[0-9.]+' "$LOG" | tail -1)" | tee -a "$MASTER"
done
echo "=== lowsplit sweep done $(date -u) ===" | tee -a "$MASTER"
