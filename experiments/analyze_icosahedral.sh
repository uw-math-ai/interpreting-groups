#!/usr/bin/env bash
# Run the full icosahedral analysis pipeline on a trained I_h checkpoint.
# Usage: bash experiments/analyze_icosahedral.sh [SEED] [SAVE_DIR]
# Figures are written to images/icosahedral_P120_d128_h4_mlp512_s{SEED}/.
set -euo pipefail
SEED=${1:-1}
SAVE_DIR=${2:-experiments}

echo "=== training curves (grokking) ==="
python -m experiments.training_curves --task icosahedral --seed "$SEED" --save_dir "$SAVE_DIR"
echo "=== attention + embedding PCA (conjugacy class & composition-series coset) ==="
python -m experiments.attention_embedding --task icosahedral --seed "$SEED" --save_dir "$SAVE_DIR"
echo "=== local PCA dimension per conjugacy class vs random ==="
python -m experiments.pca_fve --task icosahedral --seed "$SEED" --save_dir "$SAVE_DIR"
echo "=== MLP neuron activations by conjugacy class ==="
python -m experiments.mlp_frequencies --task icosahedral --seed "$SEED" --save_dir "$SAVE_DIR"
echo "=== OV principal-angle alignment ==="
python -m experiments.ov_principal_angles --task icosahedral --seed "$SEED" --save_dir "$SAVE_DIR"
echo "=== representation theory: isotypic spectra + icosahedron recovery ==="
python -m experiments.rep_analysis --seed "$SEED" --save_dir "$SAVE_DIR"
echo "Done. See images/icosahedral_P120_d128_h4_mlp512_s${SEED}/"
