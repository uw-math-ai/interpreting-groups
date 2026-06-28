#!/usr/bin/env bash
# Full mech-interp + polytope-recovery pipeline for one group task.
# Usage: bash experiments/analyze_group.sh <TASK> [SAVE_DIR] [DEVICE]
#   e.g. bash experiments/analyze_group.sh 2I data/transformer/2I_s1 cpu
set -uo pipefail
cd "$(dirname "$0")/.."
TASK=${1:?task required (2I|S5|A5|2O|2T|icosahedral)}
SD=${2:-experiments/${TASK}_s1}
DEV=${3:-cpu}
export MPLBACKEND=Agg
for S in training_curves attention_embedding pca_fve mlp_frequencies ov_principal_angles; do
  echo "=== $S ($TASK) ==="
  uv run python -m experiments.$S --task "$TASK" --seed 1 --save_dir "$SD" --device "$DEV" 2>&1 \
    | grep -vE "UserWarning|warnings.warn|softmax|split the reduction|important|^online|Not enough|^$|Deprecation" | tail -4
done
echo "=== group_rep_analysis ($TASK): isotypic + polytope recovery ==="
uv run python -m experiments.group_rep_analysis --task "$TASK" --seed 1 --save_dir "$SD" --device "$DEV" 2>&1 \
  | grep -vE "UserWarning|warnings.warn|softmax|split the reduction|important|^online|Not enough|^$|Deprecation"
echo "Done -> images/${TASK}_*_s1/"
