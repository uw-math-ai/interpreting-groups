# MLP-only gallery (no attention)

A parallel control to the transformer: an MLP (shared embedding → concat operands
→ MLP → unembed, *no attention*) trained on the same group Cayley tables, at a
**low train split (tf=0.3, 70% held out)** with generous post-grok purification.

## Start here
- **`00_transformer_vs_MLP.png`** — the same polytope recovered by both
  architectures, side by side (transformer left, MLP right).

## Individual MLP figures
red (hollow) = true polytope vertices+edges; bold blue = decoded centroid per
vertex; faint blue = individual held-out elements. R² = out-of-sample decode.

| file | group | polytope | MLP R² | transformer R² |
|---|---|---|---|---|
| `1_I_h_icosahedron_MLP` | I_h | icosahedron | 0.995 | 0.999 |
| `2_2O_24-cell_MLP` | 2O | 24-cell | 0.991 | 0.988 |
| `3_S5_A4-roots_MLP` | S5 | A4 root system | **0.995** | 0.93 |
| `4_A5_icosahedron_MLP` | A5 | icosahedron | 0.762 | 0.78 |

## Findings (MLP vs transformer)

1. **Same geometry, architecture-independent.** The MLP recovers the *identical*
   polytopes as the transformer, with matching decode quality per group
   (I_h/2O clean; A5 capped ~0.76–0.78 by its two conjugate 3-dim irreps in
   *both* models). The recovered representation is a property of the **group**,
   not the architecture.
2. **The MLP groks from far less data.** All four cleanly-grokking groups did so
   at **tf=0.3**, where the transformer needed 0.45–0.6 — and **S5 grokked at
   tf=0.3 for the MLP though the transformer needed tf=0.8** (and the MLP's S5 is
   *cleaner*, 0.995 vs 0.93). Consistent with MLPs grokking algebraic tasks at
   low data fractions.
3. **Limits are also architecture-independent.** **2I** only *partially* grokked
   at tf=0.3 (eval ~0.77; its two-rep structure + 70% held-out is too hard → no
   clean 600-cell). **2T** (order 24, ~173 train examples at tf=0.3) stayed below
   threshold and did not grok — exactly as it failed for the transformer.

Reproduce: `experiments/run_mlp.sh` (TASKS/TF/EPOCHS/EXTRA env), model in
`core/mlp_model.py`, analysis `experiments/mlp_analyze.py`. Transformer gallery is
in `../polytope_gallery/`.
