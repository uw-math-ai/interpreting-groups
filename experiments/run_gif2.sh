#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
M=logs/gif2.log
echo "=== gif2 start $(date -u) ===" | tee -a "$M"
run(){ local task=$1 arch=$2 tf=$3 ep=$4 sd=$5 ef m
  if [ "$arch" = mlp ]; then ef=$(awk "BEGIN{printf \"%.2f\",1-$tf}"); m=experiments.mlp_train; else ef=0.2; m=experiments.aws_train; fi
  echo ">>> [$(date -u +%H:%M)] START $task -> $sd" | tee -a "$M"
  uv run python -m "$m" --task "$task" --epochs "$ep" --eval_every 100 --snapshot_every 500 \
    --snapshot_dense_until 2000 --checkpoint_every 20000 --train_frac "$tf" --eval_frac "$ef" \
    --weight_decay 1.0 --lr 1e-3 --seed 1 --save_dir "$sd" --force > "logs/gif2_${task}.log" 2>&1
  echo ">>> [$(date -u +%H:%M)] END $task rc=$?" | tee -a "$M"
}
run 2O  tf  0.8 70000  data/transformer/gif2_2O_s1
run WA3 mlp 0.6 90000  data/weyl/gif2_WA3_s1
run WD4 mlp 0.5 70000  data/weyl/gif2_WD4_s1
run PSL27 mlp 0.5 90000 data/weyl/gif2_PSL27_s1
run A5  tf  0.8 90000  data/transformer/gif2_A5_s1
run S5  mlp 0.3 100000 data/mlp/gif2_S5_s1
run 2I  tf  0.6 110000 data/transformer/gif2_2I_s1
run icosahedral tf 0.6 110000 data/transformer/gif2_ih_s1
run WB4 mlp 0.4 100000 data/weyl/gif2_WB4_s1
echo "=== gif2 done $(date -u) ===" | tee -a "$M"
