# Grokking finite groups: transformers recover the lowest-dimensional faithful representation

We trained the repo's single-layer decoder-only transformer on the Cayley tables
of several finite groups of order ~120 (and relatives), watched each **grok**
(train acc 1.0 by ~1k epochs; eval at chance for tens of thousands of epochs,
then a sharp jump to 1.0), and did mechanistic interpretability on the grokked
embeddings via **representation theory**.

## Headline result

Every grokked model concentrates the bulk of its **embedding energy** into the
**lowest-dimensional faithful (geometric) irreducible representation** of the
group — i.e. it reconstructs the group's natural geometry. Energy fraction is
measured by isotypic projection onto each irrep; the random baseline for an irrep
of dimension `d` is `d²/|G|`, so "enrichment" = fraction / baseline.

| group | \|G\| | grok @tf | dominant geom. irrep | energy | enrichment | geometry |
|---|---|---|---|---|---|---|
| **I_h = A5×C2** | 120 | 0.6 | 3-dim | ~99%* | 13.2× | icosahedron |
| **2I = SL(2,5)** | 120 | 0.6 | **2-dim spinor** (+3-dim) | 42% (+48%) | **12.6×** (+6.3×) | **600-cell** (+icosahedron) |
| **A5** | 60 | 0.8 | 3-dim | 91.8% | 6.1× | icosahedron |
| **2O = GL(2,3)** | 48 | 0.8 | **2-dim spinor** | 60% | 7.2× | 24-cell |
| **S5 = W(A4)** | 120 | 0.8 | 4-dim standard | 86.9% | 6.5× | A4 root space |
| 2T = SL(2,3) | 24 | — | 2-dim spinor (partial) | 25% | **1.5×** | 24-cell (only partial grok) |

\* I_h embedding measured at ep165k (well post-grok); at the grok moment (~ep57k)
it was ~75%. Continued weight-decay training **purifies** the representation
(75% → 99%), and sharpens the geometry (icosahedron 3-PC variance 0.27 → 0.37).

The standout is **2I (binary icosahedral)**: it uses **both** the 2-dim spinor
(the genuinely faithful rep that distinguishes `q` from `−q` — its orbit is the
**600-cell**, highest enrichment 12.6×) **and** the 3-dim icosahedral rep it
shares with A5. The two together carry ~90% of the embedding.

## The decisive test: held-out elements decode to the true polytope

Energy enrichment (above) says the embedding *transforms as* the geometric irrep,
but it does **not** by itself mean a raw PCA shows the Platonic solid — the
embedding stores the rep's *matrix coefficients*, whose PCA is a blurry cloud, not
the *orbit* `ρ(g)·v₀` that is the polytope. So raw-PCA pictures are NOT convincing
on their own (an honest correction to an earlier overclaim).

The tautology-proof test: fit a ridge linear map `embedding → true polytope
coordinates` on a subset of group elements and predict the coordinates of
**held-out** elements (5-fold CV). A random or non-generalizing embedding fails
out-of-sample; only genuinely encoded geometry generalizes.

| group | polytope | held-out R² | shuffled control |
|---|---|---|---|
| 2O | 24-cell | **0.98** | −0.18 |
| S5 | A4 roots | **0.93** | −0.22 |
| 2I | 600-cell | **0.91** | −0.20 |
| A5 | icosahedron | **0.71** | −0.19 |
| 2T (no grok) | 24-cell | −1.38 | −0.41 |

Held-out group elements linearly decode to their true polytope coordinates with
R² 0.71–0.98 — rigorous, out-of-sample confirmation that the geometry is really
there.

**Pretty pictures** (`experiments/polytope_figures.py`): applying that decoder and
rendering in the projection that reveals the symmetry gives the actual objects —
`images/_summary/polytopes_montage.png` plus per-group `images/<task>_*_s1/polytope.{png,svg}`:
- A5 → **icosahedron** (native 3-D, 12 vertices / 30 edges / 20 faces).
- 2I → **600-cell** (10-fold Coxeter-plane projection — concentric-ring "flower").
- 2O → **24-cell** (8-fold Coxeter-plane projection).
- S5 → **A4 root system** (5-fold Coxeter-plane projection — two decagons).
Red = true polytope; blue = the model's held-out decoded embedding sitting on it.
(Raw PCA does NOT show these — the embedding holds matrix coefficients, not the
orbit; you must apply the learned decoder, which is what these figures do.)

Two honest subtleties this surfaced:
- **Galois-conjugate irrep.** A5 and 2I are defined over Q(√5) and have *two*
  conjugate 3-dim/spinor irreps; the model is free to pick either. Decoding to the
  canonical coords failed (R² ≈ −0.2) until we used the √5→−√5 relabeling, which
  jumps to 0.91 (2I) / 0.71 (A5). So the geometry was there all along — just in
  the conjugate copy.
- **Enrichment is the rep-theory signal; decode is the geometric signal.** For a
  faithful irrep a *Procrustes shape-match* is tautological (the orbit is the
  polytope by construction, random baseline ≈ 0.99) — which is why we use
  out-of-sample **decode R²**, not shape-match, as the geometric evidence.
- **Train fraction matters and scales with irrep size.** Order-120 binary/
  icosahedral groups grok at tf=0.6, but **S5 stays in pure memorization at
  tf=0.6** (eval flat at chance through 100k epochs) and only groks at tf≥0.8 —
  its larger 4/5/5/6-dim irreps raise the data threshold. The smallest group
  (**2T, order 24**) **does not cleanly grok** at all — best eval ~0.35 (d=128,
  tf=0.9), and a smaller model is worse (0.13). It is so over-parameterized
  relative to its 24-element table that it memorizes without ever needing a
  generalizing circuit. This is a **group-size threshold for grokking**.
- **Enrichment causally tracks generalization.** The partial-grok 2T (eval 0.35)
  shows only **weak** spinor enrichment (1.5×) with energy spread across irreps —
  vs 7.2× for fully-grokked 2O. So the representation-theoretic concentration is
  tied to grokking, not an artifact: full grok → 6–13×, partial → 1.5×, none → flat.

## Supporting circuits (per group, in `images/<task>_*_s1/`)

- `isotypic_embedding.svg` / `isotypic_unembed.svg` — energy per irrep vs baseline.
- `geometry_pca_conjugacy.html` — embedding projected onto the dominant irrep,
  colored by conjugacy class (the "see the polytope" scatter).
- `pca_fve.svg` — conjugacy-class subspaces are lower-dimensional than random.
- `ov_principal_angles_head0.svg` — the attention OV subspace aligns with the
  conjugacy-class embedding subspaces post-grok.
- `training_curves.pdf` — the grokking curve.

## Train-split & purification study (what actually makes the geometry sharp)

We tested the hypothesis "lower train split + more epochs → cleaner geometry."
Best out-of-sample decode R² per group (over high-split, low-split, grok-moment,
and purified checkpoints):

| group | best R² | from | what limits it |
|---|---|---|---|
| **I_h** | **0.999** | high-split tf=0.6, purified ep165k | single irrep, reaches eval=1.0 |
| **2O** | **0.988** | tf=0.8, purified ep45k | single spinor |
| **2I** | 0.932 | low-split tf=0.45 | splits spinor + 3-dim (two reps) |
| **S5** | 0.928 | **grok-moment** ep85k | single rep, but degrades with over-training |
| **A5** | 0.781 | tf=0.8, ep125k | mixes two conjugate 3-dim (no split fixes it) |

Findings (some counter to the initial hypothesis):
1. **Sharpness is set by irrep multiplicity, not the split.** Single-irrep groups
   (I_h, 2O) decode near-perfectly *at any reasonable split*; multi-/conjugate-irrep
   groups (2I spinor+3dim, A5 two conjugate 3-dim) are structurally capped.
2. **You want the lowest split that still reaches eval = 1.0 — not lower.** Below
   that the model never fully generalizes (I_h@0.45 plateaus at eval 0.982 → R²
   0.971 < 0.999 at tf=0.6). Too-low split *hurts*.
3. **Post-grok purification is group-dependent, not universally good.** It sharpens
   I_h (0.75→0.99 isotypic; R²→0.999), is neutral for 2O, but *degrades* S5
   (R² 0.93 at grok → 0.56 by ep300k — weight-decay drift off the clean standard
   rep). So "always train 10k+ more" helps some groups and hurts others; the
   trainer now saves an explicit `*_grok10k.pt` so the grok-vs-purified comparison
   is available per run.
4. Net: the prettiest pictures come from **single-irrep groups at the split that
   just reaches eval=1.0, at the right (group-specific) checkpoint** — exemplified
   by I_h's R²=0.999 icosahedron. Montage: `images/_summary/polytopes_montage.png`.

## Reproduce

```bash
uv sync
# train one group (lean CUDA driver; fused AdamW + TF32 + compile, no per-epoch sync)
uv run python -m experiments.aws_train --task 2I --epochs 300000 --train_frac 0.6 --seed 1 --save_dir experiments/2I_s1
# or the whole sweep:  TASKS="2I S5 A5 2O 2T" TF=0.8 bash experiments/run_groups.sh
# analyze a grokked checkpoint (all figures + isotypic + polytope recovery):
bash experiments/analyze_group.sh 2I experiments/2I_s1 cpu
# cross-group comparison figure + table:
uv run python -m experiments.summary_groups
```

## New code

- `core/reptheory.py` — generic numeric character table (Dixon class-algebra method),
  works for any Cayley table; validated against I_h, S5, and the binary groups.
- `core/quaternions.py` — 2I/2O/2T as unit-quaternion groups (elements = 600-cell/
  24-cell vertices); A5 as the rotation quotient 2I/{±1}.
- `core/groups.py::get_group` — registry (icosahedral, 2I, 2O, 2T, A5, S5) + S5
  builder; generalized `analysis_partition` / `group_partition` (conjugacy + coset).
- `experiments/group_rep_analysis.py` — generic, complex-safe isotypic analysis +
  honest geometric recovery (enrichment-led, with random-baseline match context).
- `experiments/aws_train.py`, `run_groups.sh`, `analyze_group.sh`, `summary_groups.py`.
