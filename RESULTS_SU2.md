# SU(2) vs SO(3): half-integer spin sectors emerge only for the double cover (pilot)

**Claim under test.** When a generic network learns the *composition law* of a compact
continuous group, does it internalize the group's harmonic (Peter–Weyl) structure — and
does it learn the group's **global topology**, not just its Lie algebra? SU(2) and SO(3)
share the Lie algebra 𝔰𝔲(2) ≅ 𝔰𝔬(3) but differ globally: SU(2) is the double cover, and
only SU(2) admits **half-integer spin** (spinor) representations. The signature to look
for is therefore basis-independent and binary: *half-integer spin sectors in the learned
representation exist iff the model was trained on the double cover.*

**Pilot verdict (4 runs, single seed): confirmed, cleanly.**

![pilot summary](gallery/su2/su2_pilot_summary.png)

## Setup

Everything is continuous — no tokens, no finite vocabulary anywhere.

- **Elements.** Haar-random unit quaternions q ∈ S³ ≅ SU(2).
- **Observation map (frozen).** Ψ(q) = exp(α⟨q, hᵢ⟩), i = 1…32 fixed Haar landmarks
  (α = 3, standardized). The **SO(3) control** uses |⟨q, hᵢ⟩| — the *only* difference
  between the two conditions is this one absolute value, which makes Ψ even
  (Ψ(q) = Ψ(−q)), i.e. the data quotients to SO(3). Same landmarks, same dimension,
  same everything else: a matched-pair design where the manipulated
  variable is exactly the double cover.
- **Task (InfoNCE candidate scoring).** Given Ψ(a), Ψ(b) and K = 64 candidates, pick
  the product: the positive is (ab)·exp(ξ), ξ ~ 𝒩(0, σ²I₃) (σ = 0.2, a heat-kernel-like
  soft target whose Bayes-optimal scorer is analytically log K_t ≈ Σⱼ(2j+1)e^{−tj(j+1)}χⱼ);
  63 frozen Haar negatives (≥ 0.5 rad from the positive class). Data = fixed pool of
  N = 2048 elements, training on a frozen fraction f of the N² ordered pairs.
- **Model (~0.3M params).** Encoder-factorized triple scorer:
  z_a = E(Ψ(a)), z_b = E(Ψ(b)); h = MLP([z_a, z_b]); score = ⟨h, E_c(Ψ(c))⟩.
  E's activations z(g) are the analysis object (the continuous analog of the embedding
  table). AdamW, lr 1e-3, weight decay 1.0, batch 4096, 200k steps.
- **Eval tiers.** T1 = held-out pairs of pool elements; T2 = fresh Haar pairs (elements
  never seen); T3 = exact product vs 63 negatives at fixed geodesic radius r.
- **Spectroscopy.** FVE of z(g) onto sampled, jointly-orthonormalized Wigner-D bases
  per spin j ∈ {0, ½, 1, 3/2, 2} over a fixed 2048-element Haar probe (cross-spin
  leakage floor ~r_j/N ≈ 0.2%); cosine of z(g) vs z(−g); holonomy of a 2π SO(3)
  rotation; ridge decode of z to the quaternion (odd, j=½) and to qqᵀ (even, j=1),
  both with held-out 80/20 splits.

Code: `core/su2.py` (exact half-integer Wigner-D via the homogeneous-polynomial
construction, self-tested for homomorphism/unitarity/characters/center),
`core/su2_model.py`, `experiments/su2_train.py`, `experiments/run_su2_pilot.sh`.

## Results (step 200k, N=2048, seed 0)

| run | train | T1 | T2 | FVE j=½ | j=1 | j=3/2 | j=2 | cos(z(g),z(−g)) | 2π-holonomy | R²→q | R²→qqᵀ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SU(2)** f=0.002 | 0.996 | 0.912 | 0.907 | **0.968** | 0.023 | 0.007 | 0.002 | −0.96 | **−0.63** | **0.998** | 0.987 |
| **SU(2)** f=0.01 | 0.978 | 0.926 | 0.924 | **0.797** | 0.182 | 0.017 | 0.002 | −0.66 | **−0.39** | **0.995** | 0.912 |
| **SO(3)** f=0.002 | 1.000 | 0.834 | 0.827 | 0.002 | **0.980** | 0.000 | 0.013 | +1.000 | **+1.000** | −0.006 | 0.995 |
| **SO(3)** f=0.01 | 0.999 | 0.885 | 0.884 | 0.002 | **0.909** | 0.001 | 0.084 | +1.000 | **+1.000** | −0.006 | 0.999 |

### 1. The double-cover signature

The SU(2) models put **80–97% of encoder variance in the spin-½ spinor sector**: the
representation is odd under the center (z(−g) ≈ −z(g)), transporting it around a 2π
SO(3) rotation flips it (negative holonomy), and the quaternion — the j=½ matrix
coefficients themselves — is linearly decodable at R² ≈ 0.998. The matched SO(3) controls
keep **every half-integer sector at the 0.2% measurement floor**, concentrate in j=1
(the lowest faithful SO(3) irrep, with spin 2 next), are exactly even, return
+1 holonomy, and cannot decode the quaternion (R² ≈ 0) while decoding the even qqᵀ at
R² ≈ 0.995+. One `abs()` in the observation map decides which representation theory
the network builds. This extends the repo's finite-group result — "grokked embeddings
concentrate in the lowest faithful irrep" (RESULTS.md: 2O/2I spinors) — to the
continuous group those binary groups sit inside.

### 2. Higher spins enter in order of Casimir eigenvalue

In both conditions, variance appears in additional isotypic blocks over training, in
order of Casimir eigenvalue: the SU(2) model adds spin-1 (2% → 18%) and then spin-3/2
(→ 1.7%) content on top of the dominant spin-½ block; the SO(3) control adds spin-2
(→ 8.4%) on top of spin-1, with half-integer blocks absent throughout. This is
consistent with the heat-kernel structure of the task, where target precision t sets a
spin cutoff through the e^{−tj(j+1)} envelope. The larger training fraction yields both
a broader spectrum and better fine-scale resolution (T3 at r=0.2: 0.60 vs 0.42),
consistent with a bandwidth interpretation: resolving the product at scale r requires
irreps up to j ~ 1/r.

### 3. Late-phase dynamics depend on data budget

At f=0.002 the SU(2) run kept improving late (T1 0.907 → 0.922 between 100k and 150k)
while its spectrum *re-purified* toward 97% spin-½ — weight-decay purification, the
continuous analog of the I_h purification effect in RESULTS.md. At f=0.01 the spectrum
instead slowly broadened at flat accuracy.

### 4. The memorization boundary is nearby but uncrossed

No delayed generalization ("grokking") occurred at these budgets — generalization was
essentially immediate, consistent with the prediction that smooth structured input
encodings suppress the grok delay (GrokTransfer). The SO(3) control at f=0.002 sits in a
persistent overfit state (train 1.000, T1 0.834) that 200k steps of weight decay never
flipped. The classic grokking regime, if it exists here, is at lower f or larger init
scale.

### 5. The quotient task is uniformly harder

At matched budget the SO(3) control generalizes 3–8 points worse than SU(2) (e.g. T1
0.834 vs 0.912 at f=0.002). Unexplained; candidate causes: information loss and
conditioning of the |·| encoding near its fold, or the intrinsic harder geometry of
RP³ vs S³.

## Honest caveats

- **Single seed per cell.** Everything above is n=1 per condition; the finite-group
  literature (Clock/Pizza, irrep arbitrariness) says the claim must ultimately be
  distributional. Seed sweep is the first follow-up.
- **The SO(3) control's missing half-integers are partially by construction**: an even
  input forces an even encoder. The non-trivial content is (i) the SU(2) side — the
  model *could* have solved the task with integer spins plus sign bookkeeping
  elsewhere, but instead builds a dominant, clean spinor sector; and (ii) the exact
  Peter–Weyl identity of what each condition builds (j=½ vs j=1, then j=3/2 vs j=2 —
  each group's lowest faithful irreps, not arbitrary features). The deeper causal
  version (ablate the j=½ subspace, break U vs −U discrimination) is phase 2.
- **Correlational, not yet mechanistic.** FVE concentration and decode R² say where
  the information lives, not what algorithm uses it (the Chughtai-vs-Stander lesson).
  The composition test (does a probe's D̂ satisfy D̂(g)D̂(h) ≈ D̂(gh) on held-out pairs,
  under patching) and the scorer-vs-log K_t character fit are the planned upgrades.
- T3 accuracy at r ≤ 0.05 is near chance for all runs — no model resolves products
  much below the training noise scale σ; "beating the information floor" remains open.

## Reproduce

```bash
uv run python -m core.su2                 # self-tests: Wigner-D, FVE, encodings
bash experiments/run_su2_pilot.sh         # 4 runs, ~8h total on a T4
# single run:
uv run python -m experiments.su2_train --variant su2 --N 2048 --frac 0.01 \
    --seed 0 --save_dir data/su2/su2_N2048_f0.01_s0        # --resume to continue
```

Logs (`log.jsonl`: per-1k-step accuracies; per-5k-step spectroscopy) and checkpoints
(with optimizer state) land in `data/su2/<run>/` (not in git).

## Next

1. Seed sweep (5–10×) of the four pilot cells → report distributions of all quantities.
2. Scorer character fit against the analytic Bayes target log K_t, and the
   held-out composition test — first causal-depth items.
3. N-sweep with the exact-closure 2I rung (N=120) and the σ (precision) sweep for the
   full spectral-curriculum curve.
4. Grokking hunt: lower f, inflated init norm; does delayed generalization exist for
   continuous group tasks at all?
