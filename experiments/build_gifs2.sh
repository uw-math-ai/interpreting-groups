#!/usr/bin/env bash
# Rebuild every grokking GIF in gallery/ from the dense snapshots under data/.
# Requires the *_snapshots.npz + paired *.pt checkpoints (unzip data.zip into data/).
#   convergence GIFs -> gallery/grokking_gifs/   (make_grok_gif / psl27_gif)
#   4-polytope 3-D shadows -> gallery/shadows_3d/ (shadow3d, writes .png/.svg too)
set -u
cd "$(dirname "$0")/.."
DUR=120
GG=gallery/grokking_gifs
SH=gallery/shadows_3d
mkdir -p "$GG" "$SH"
log(){ echo ">>> [$(date -u +%H:%M)] $*"; }

mg(){  # task snap out label   (native-3D / 2-D Coxeter convergence GIF)
  log "make_grok_gif $1"
  MPLBACKEND=Agg uv run python -m experiments.make_grok_gif \
    --task "$1" --snap "$2" --out "$3" --label "$4" --duration $DUR 2>&1 | grep -E "saved|Error" | tail -1
}
sh3(){ # task snap out label   (4-polytope 3-D shadow GIF + static png/svg)
  log "shadow3d $1"
  MPLBACKEND=Agg uv run python -m experiments.shadow3d \
    --task "$1" --snap "$2" --out "$3" --label "$4" --duration $DUR --spin 0.5 2>&1 | grep -E "saved|static|Error" | tail -2
}

# --- native-3D / Coxeter-2D convergence GIFs ---
mg icosahedral data/transformer/gif2_ih_s1/icosahedral_P120_d128_h4_mlp512_s1_snapshots.npz "$GG/I_h_icosahedron.gif" "I_h"
mg A5  data/transformer/gif2_A5_s1/A5_P60_d128_h4_mlp512_s1_snapshots.npz   "$GG/A5_icosahedron.gif"  "A5"
mg 2O  data/transformer/gif2_2O_s1/2O_P48_d128_h4_mlp512_s1_snapshots.npz   "$GG/2O_24cell.gif"       "2O"
mg 2I  data/transformer/gif2_2I_s1/2I_P120_d128_h4_mlp512_s1_snapshots.npz  "$GG/2I_600cell.gif"      "2I"
mg S5  data/mlp/gif2_S5_s1/S5_P120_d128_h4_mlp512_s1_snapshots.npz          "$GG/S5_A4roots.gif"      "S5"
mg WA3 data/weyl/gif2_WA3_s1/WA3_P24_d128_h4_mlp512_s1_snapshots.npz        "$GG/WA3_cuboctahedron.gif" "W(A3)"
mg WD4 data/weyl/gif2_WD4_s1/WD4_P192_d128_h4_mlp512_s1_snapshots.npz       "$GG/WD4_24cell.gif"      "W(D4)"
mg WB4 data/weyl/gif2_WB4_s1/WB4_P384_d128_h4_mlp512_s1_snapshots.npz       "$GG/WB4_16cell.gif"      "W(B4)"

# --- PSL(2,7) Fano heptagon ---
log "psl27_gif"
MPLBACKEND=Agg uv run python -m experiments.psl27_gif \
  --snap data/weyl/gif2_PSL27_s1/PSL27_P168_d128_h4_mlp512_s1_snapshots.npz \
  --out "$GG/PSL27_fano_heptagon.gif" --duration $DUR 2>&1 | grep -E "saved|Error" | tail -1

# --- 4-polytope 3-D shadows ---
sh3 2O  data/transformer/gif2_2O_s1/2O_P48_d128_h4_mlp512_s1_snapshots.npz  "$SH/shadow3d_24cell_2O.gif"  "2O"
sh3 2I  data/transformer/gif2_2I_s1/2I_P120_d128_h4_mlp512_s1_snapshots.npz "$SH/shadow3d_600cell_2I.gif" "2I"
sh3 S5  data/mlp/gif2_S5_s1/S5_P120_d128_h4_mlp512_s1_snapshots.npz         "$SH/shadow3d_A4roots_S5.gif" "S5"
sh3 WD4 data/weyl/gif2_WD4_s1/WD4_P192_d128_h4_mlp512_s1_snapshots.npz      "$SH/shadow3d_24cell_WD4.gif" "W(D4)"
sh3 WB4 data/weyl/gif2_WB4_s1/WB4_P384_d128_h4_mlp512_s1_snapshots.npz      "$SH/shadow3d_16cell_WB4.gif" "W(B4)"

log "=== build_gifs2 done ==="
