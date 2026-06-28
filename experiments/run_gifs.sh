#!/usr/bin/env bash
# Dedicated snapshot runs (grok + crystallize) for the convergence GIFs, one per
# group, using the architecture/config that groks AND crystallizes cleanly:
#   I_h/2I/2O/A5 -> transformer ;  S5 -> MLP (transformer degrades S5 with training)
# I_h already has its GIF; this does the rest. Sequential.
set -uo pipefail
cd "$(dirname "$0")/.."
MASTER=logs/gifs.log
echo "=== gif snapshot runs start $(date -u) ===" | tee -a "$MASTER"

run() {  # task arch tf epochs savedir log
  local task=$1 arch=$2 tf=$3 ep=$4 sd=$5 log=$6 ef m
  if [ "$arch" = mlp ]; then ef=$(awk "BEGIN{printf \"%.2f\",1-$tf}"); m=experiments.mlp_train
  else ef=0.2; m=experiments.aws_train; fi
  echo ">>> [$(date -u +%H:%M:%S)] START $task ($arch tf$tf ef$ef) -> $sd" | tee -a "$MASTER"
  uv run python -m "$m" --task "$task" --epochs "$ep" --eval_every 1000 --snapshot_every 2000 \
    --checkpoint_every 20000 --train_frac "$tf" --eval_frac "$ef" --weight_decay 1.0 --lr 1e-3 \
    --seed 1 --save_dir "$sd" --force > "$log" 2>&1
  echo ">>> [$(date -u +%H:%M:%S)] END   $task rc=$?" | tee -a "$MASTER"
}

run 2I tf  0.6 180000 data/transformer/snap_2I_s1     logs/snap_2I.log
run 2O tf  0.8 150000 data/transformer/snap_2O_s1     logs/snap_2O.log
run A5 tf  0.8 150000 data/transformer/snap_A5_s1     logs/snap_A5.log
run S5 mlp 0.3 180000 data/mlp/snap_S5_s1 logs/snap_S5.log
echo "=== gif snapshot runs done $(date -u) ===" | tee -a "$MASTER"
