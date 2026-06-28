# Polytope gallery

The symmetric polytope each grokked transformer recovers from its group's
multiplication table. In every picture:
- **red** = the true polytope (vertices + edges)
- **bold blue** = the decoded **centroid per vertex** (average over the group
  elements that map to that vertex)
- **faint blue** = the individual held-out elements (their spread around a vertex)
- **R²** = out-of-sample decode fit (1.0 = held-out elements land exactly on the
  true vertices). Tautology-proof: a non-encoding embedding fails it.

Why some look tighter than others (investigated): it is **not** a projection-angle
artifact — the decode error is isotropic (in-plane ≈ total), and visual scatter
≈ √(1−R²) (so R²=0.93 ⇒ ~26% positional spread, R²=0.99 ⇒ ~10%). I_h and 2O are
clean because they concentrate in a **single** irrep; 2I and A5 spread across
**two** irreps (2I: spinor + 3-dim; A5: two conjugate 3-dim), a structural cap no
projection or extra training removes. The centroid (bold blue) averages out the
per-element noise for the degenerate groups (S5, A5, I_h), which is why those bold
points sit much closer to the vertices than the faint cloud.

## Start here
- **`00_ALL_polytopes_montage.png`** — all five at a glance.

## Individual pictures (ordered by sharpness)
| file | group | polytope | R² |
|---|---|---|---|
| `1_I_h_icosahedron_R0.999` | I_h = A5×C2 | icosahedron (3-D) | 0.999 |
| `2_2O_24-cell_R0.988` | 2O = GL(2,3) | 24-cell (8-fold proj.) | 0.988 |
| `3_2I_600-cell_R0.932` | 2I = SL(2,5) | 600-cell (10-fold proj.) | 0.932 |
| `4_S5_A4-root-system_R0.928` | S5 = W(A4) | A4 roots (5-fold proj.) | 0.928 |
| `5_A5_icosahedron_R0.781` | A5 | icosahedron (3-D) | 0.781 |

`.png` for quick viewing, `.svg` for vector/zoom. **`interactive_3d/*.html`** —
open in a browser to rotate the 3-D decode overlays.

4-D polytopes (600-cell, 24-cell, A4 roots) are shown in their **Coxeter-plane**
projection (the symmetric "flower" view); I_h and A5 are native 3-D.

Reproduce: `experiments/polytope_figures.py` (rendering) and
`experiments/geometry_decode.py` (the held-out R²). Full write-up: `../RESULTS_groups.md`.
