#!/usr/bin/env bash
# Aggressive grokking run for the icosahedral group I_h = A5 x C2 (order 120),
# tuned for a CUDA GPU (e.g. an AWS g5/p3/p4 instance).
#
# Why this is needed: I_h is non-abelian, so (unlike modular multiplication) the
# model needs positional embeddings AND a long grokking schedule. On MPS/CPU the
# transition is impractically slow; on a GPU with --compile this tiny model runs
# at thousands of epochs/sec, so 300k-500k epochs is only a few minutes.
#
# Setup on the GPU box:
#   uv sync                      # or: pip install torch numpy matplotlib sympy scikit-learn plotly pandas
#   bash experiments/run_icosahedral_aws.sh
#
# Outputs: experiments/icosahedral_P120_..._s{seed}.pt        (final weights)
#          experiments/icosahedral_P120_..._s{seed}_best.pt   (highest-eval weights = the grok)
# Pull the *_best.pt back locally and run experiments/analyze_icosahedral.sh.
set -euo pipefail

EPOCHS=${EPOCHS:-500000}        # grokking on order-120 non-abelian is slow; go big
EVAL_EVERY=${EVAL_EVERY:-500}
CKPT_EVERY=${CKPT_EVERY:-5000}  # also writes *_best.pt at each new eval high
WD=${WD:-1.0}
LR=${LR:-1e-3}

# A small sweep: a clean grok needs a real generalization gap (lower train_frac),
# but too little data may never generalize. Sweep the threshold and a few seeds.
for TF in 0.4 0.5 0.6; do
  for SEED in 1 2; do
    echo "=== train_frac=$TF seed=$SEED epochs=$EPOCHS ==="
    python -m core.trainer \
      --task icosahedral \
      --epochs "$EPOCHS" \
      --eval_every "$EVAL_EVERY" \
      --checkpoint_every "$CKPT_EVERY" \
      --train_frac "$TF" \
      --eval_frac 0.2 \
      --weight_decay "$WD" \
      --lr "$LR" \
      --seed "$SEED" \
      --compile \
      --save_dir "experiments/tf${TF}"
  done
done

echo "Done. Inspect eval curves; the _best.pt files hold the grokked weights."
