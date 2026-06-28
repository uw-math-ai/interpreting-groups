# Weyl & exotic groups gallery

Seven additional groups run through the same pipeline (grok → isotypic → decode →
polytope). Reflection groups are built as their standard reflection representation,
so the geometric target is the **root system** (coords[g] = M_g·root). Trained
MLP-only at low train fractions with crystallization.

## Recovered polytopes (model concentrates in the geometric irrep)
`00_weyl_polytopes_montage.png` + individual figures:

| group | order | polytope | held-out R² | GIF |
|---|---|---|---|---|
| **W(A3)=S4** | 24 | **cuboctahedron** (A3 roots) | 0.96 | `grok_cuboctahedron_WA3.gif` |
| **W(D4)** | 192 | **24-cell** (D4 roots, triality; 6-fold Coxeter) | 0.91 | `grok_24cell_WD4.gif` |
| **W(B4)=W(C4)** | 384 | **16-cell** (B4 short roots) | 0.97→1.0 | `grok_16cell_WB4.gif` |
| **PSL(2,7)=GL(3,2)** | 168 | **6-simplex / Fano plane** (heptagon + all diagonals) | 0.99 | `grok_fano_heptagon_PSL27.gif` |

PSL(2,7) has no *root* polytope, but its dominant 6-dim irrep IS the 7-point Fano-plane
permutation rep (minus trivial), whose orbit is the 6-simplex — so the embedding decodes
to the 7 Fano points (R²=0.99) and renders as a heptagon with all 21 diagonals (K₇).

## Isotypic results for all seven (the universal, rigorous signal)

| group | order | grok? | dominant irrep (energy / enrichment) | polytope |
|---|---|---|---|---|
| W(A3)=S4 | 24 | ✓ | 3-dim standard 73% / 1.9× | cuboctahedron ✓ |
| W(B3)=O_h | 48 | ✓ | spread (3-dim 44% / 2.4×, not the vector rep) | ✗ no clean octahedron |
| W(D4) | 192 | ✓ | 4-dim vector 26% (+3-dims) / 3.1× | 24-cell ✓ |
| W(B4) | 384 | ✓ | **4-dim vector 45% / 10.7×** (strong) | 16-cell ✓ |
| W(G2) | 12 | ✓ | 2-dim reflection 42% / 1.3× | hexagon — n=12 too small for reliable decode |
| **PSL(2,7)=GL(3,2)** | 168 | ✓ | **6-dim 99.5% / 4.6×** (cleanest of all) | ✗ 6-dim, no Euclidean solid (Klein quartic) |
| A6 | 360 | ✗ | — (did not grok at tf=0.4: flat at eval 0 through 100k) | ✗ 5-dim min |

## Takeaways
- **3-D and 4-D root polytopes are recovered** (cuboctahedron, 24-cell, 16-cell)
  when the model concentrates in the vector/reflection rep — including the
  192- and 384-element 4-D Weyl groups.
- **Concentration is not guaranteed.** O_h (W(B3)) groks but spreads across its
  many 3-dim irreps (and not the vector one), so its octahedron isn't recovered —
  whereas its 4-D siblings W(D4)/W(B4) concentrate cleanly.
- **PSL(2,7) is the standout**: 99.5% of its embedding in the single 6-dim irrep —
  the cleanest concentration in the whole study — but 6-dimensional, so no
  Euclidean polytope (this is the Klein-quartic representation).
- **A6 did not grok** at tf=0.4 (large group, big irreps, 60% held out) — a
  group-size/complexity limit, consistent with the small-group failures (2T).

Reproduce: `core/weyl.py` (constructors), `experiments/run_weyl.sh` (sweep),
`experiments/mlp_analyze.py` (decode + render), `experiments/make_grok_gif.py`.
