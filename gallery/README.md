# Gallery

Curated visuals of grokked models recovering each group's geometry. Red = the
true polytope (vertices + edges); blue = the model's **held-out** decoded points
(group elements left out of the fit). Every GIF carries a bottom panel with the
train/eval-accuracy curves on a log-epoch axis and a vertical rule marking the
current frame's epoch (memorization → grokking → crystallized).

## `grokking_gifs/` — embedding → polytope, over training

| file | group | recovers |
|---|---|---|
| `I_h_icosahedron.gif` | I_h = A5×C2 | icosahedron (3-D) |
| `A5_icosahedron.gif` | A5 | icosahedron (3-D) |
| `2O_24cell.gif` | 2O (binary octahedral) | 24-cell + dual (4-D Coxeter) |
| `2I_600cell.gif` | 2I (binary icosahedral) | 600-cell (4-D Coxeter) |
| `S5_A4roots.gif` | S5 = W(A4) | A4 root system (4-D Coxeter) |
| `WA3_cuboctahedron.gif` | W(A3) = S4 | cuboctahedron (3-D) |
| `WD4_24cell.gif` | W(D4) | 24-cell (4-D Coxeter) |
| `WB4_16cell.gif` | W(B4) | 16-cell (4-D Coxeter) |
| `PSL27_fano_heptagon.gif` | PSL(2,7) | Fano plane / 6-simplex (heptagon + K7) |

## `shadows_3d/` — 4-polytopes as rotatable 3-D shadows

Orthographic projection ⟂ to the (1,1,1,1) diagonal turns each 4-cell into a
rotatable 3-D solid (true 4-D edges drawn in the shadow). `00_xcells_montage.png`
tiles all five. `.gif` (rotating + grokking), `.png`/`.svg` (static, sharpest frame).

- `shadow3d_600cell_2I.*` — 600-cell (120 verts; coloured by 4-D depth = nested icosahedral shells)
- `shadow3d_24cell_2O.*`, `shadow3d_24cell_WD4.*` — 24-cell
- `shadow3d_16cell_WB4.*` — 16-cell
- `shadow3d_A4roots_S5.*` — A4 roots

## `polytopes/` — static decoded figures

- `transformer/` — I_h, 2O, 2I, S5, A5 (R² in each filename) + `00_ALL_polytopes_montage.png`
- `mlp/` — same groups from the MLP-only models + `00_transformer_vs_MLP.png` (architecture-independence)
- `weyl/` — W(A3), W(D4), W(B4) + `00_weyl_polytopes_montage.png`

## `interactive_3d/` — rotatable plotly HTML

Open in a browser to drag-rotate the 3-D decodes (I_h, 2O, 2I, S5, A5).

Regenerate everything from `data/` with `bash experiments/build_gifs2.sh`
(GIFs) and `python -m experiments.polytope_figures` (static figures).
