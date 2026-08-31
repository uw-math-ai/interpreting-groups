# Learning and Interpreting Continuous Group Operations: Compact Groups and their Central Quotients

**Status:** evolving working report for agent/collaborator handoff. Last updated
2026-08-31. Complete: SU(2)/SO(3) headline + causal double dissociation (5 seeds,
Sections 3–5.3); mechanistic loop closure (Section 5.4); observation-map
robustness, heat-kernel bandwidth in Casimir order with the capacity and
wide-retune verdicts (Sections 5.5, 5.8a); SU(3)/PSU(3) with the Z₃ causal
dissociation, seed-replicated (Section 5.5c); the representation-ring/fusion
result (Section 5.6, 5 seeds); the Spin(4) central-quotient program with the
minimal-carrier selection observation and block-resolved causal map (Section
5.7, 5 seeds); the Spin(5)/SO(5) pilot (Section 5.8, one seed — the SO(5)
mixture is an open stress test); and the carrier-cost search (null result,
Section 5.7/theory notes). Theory: `summary/theory_notes.md` and
`summary/selection_theorem.md`. The program now operates under the v2 research
plan (archived at `ideas/research_plan_2026-08-30_v2.md`), whose central
question is no longer whether representations emerge but **which end-to-end
mechanism — feature system plus composition law plus scorer — regularized
optimization selects, under what complexity cost**; Section 8 tracks the
v2 workstreams (P0 hygiene, P1 two-sided closure and frame-vs-block, P2 SO(5)
conditioning, P3 theory, P4 quantitative spectral bias, P5 literal
Clebsch–Gordan). `summary/results_bundle.json` is the compact tracked bundle
from which headline tables regenerate; `RESULTS_SU2.md` carries the
public-facing SU(2) summary.

---

## 1. Motivation and relation to prior work

This project extends the lab's mechanistic-interpretability program on algebraic operations
— modular multiplication over composite moduli (Chen et al., arXiv:2607.07066, `refs/`) and
finite-group composition (this repo, `RESULTS.md`) — from finite structures to a compact
continuous group. Prior work established that small networks trained on finite group
composition represent elements via matrix coefficients of a small number of irreducible
representations, and that grokked embeddings concentrate their variance in the
lowest-dimensional faithful irrep (Chughtai et al., arXiv:2302.03025; this repo). All of
this machinery presupposes a finite multiplication table and a learned embedding per
element. No published work, to our knowledge, trains a network on the composition law of a
continuous group and analyzes the learned representation; a literature review supporting
this novelty claim is maintained locally in `ideas/continuous_groups_review.md`.

For a compact group the natural generalization of the discrete Fourier/character analysis
is the Peter–Weyl theorem: L²(SU(2)) decomposes into matrix coefficients of the irreps
D^j (Wigner D-matrices) indexed by spin j ∈ {0, ½, 1, 3/2, …}, each contributing a
(2j+1)²-dimensional isotypic block. The quotient SO(3) = SU(2)/{±1} admits only the
integer-spin irreps. This yields a sharp, basis-independent diagnostic: **the presence of
half-integer isotypic blocks in a learned representation indicates that the model has
learned the double cover SU(2) rather than SO(3)**. Since su(2) ≅ so(3) as Lie algebras,
this is a statement about the global topology of the group, invisible to any
infinitesimal (Lie-algebra-level) analysis. Testing whether trained networks make this
distinction is the central question of the project.

## 2. Experimental setup

**Task.** Candidate-ranking formulation of group composition. Given encodings of
a, b ∈ SU(2) and of K = 64 candidates, the model must identify the product ab. The
positive candidate is (ab)·exp(ξ) with ξ ~ N(0, σ²I₃) in the Lie algebra (σ = 0.2), a
smoothed target comparable to a heat-kernel distribution on the group; the remaining 63
candidates are i.i.d. Haar samples, resampled until their geodesic distance to the
positive class exceeds 0.5 rad. All candidates are frozen at dataset construction. The
loss is cross-entropy over the K scores. Under this construction the Bayes-optimal
decision rule ranks candidates by geodesic distance to ab, and its accuracy — the noise
ceiling of the task — is 0.951 (Monte-Carlo estimate, Section 3).

**Element encoding.** No tokens are used. Each element is presented through a frozen
observation map Ψ(q) = exp(α⟨q, hᵢ⟩), i = 1,…,32, with fixed Haar landmarks hᵢ and
α = 3, standardized to zero mean and unit variance. The **SO(3) control** replaces
⟨q, hᵢ⟩ with |⟨q, hᵢ⟩|, making Ψ even (Ψ(q) = Ψ(−q)) so that the data determine elements
only up to sign — i.e., the same task on the quotient group. The two conditions share
landmarks, dimension, and all hyperparameters; the absolute value is the only difference.

**Model.** An encoder-factorized scorer (~0.3M parameters): z_a = E(Ψ(a)),
z_b = E(Ψ(b)) with a shared two-layer GELU encoder E: ℝ³² → ℝ²⁵⁶ → ℝ¹²⁸; an interaction
MLP h = MLP([z_a, z_b]) (256 → 512 → 128); and score s(a,b,c) = ⟨h, E_c(Ψ(c))⟩ with a
separate candidate encoder E_c. The encoder activations z(g) play the role of the
embedding matrix in the finite-group analyses. AdamW, lr 10⁻³, weight decay 1.0, batch
4096, 200k steps.

**Data.** A fixed pool of N = 2048 Haar elements; the training set is a frozen fraction
f of the N² = 4.19M ordered pairs. The pilot used f = 0.002 (8,389 pairs; ≈ 4 pairs per
element; one epoch ≈ 2 steps) and f = 0.01 (41,943 pairs; ≈ 20 pairs per element; one
epoch ≈ 10 steps), so 200k steps correspond to ≈ 10⁵ and ≈ 2×10⁴ epochs respectively.

**Evaluation tiers.** T1: held-out pairs of pool elements (the analogue of held-out
cells of a multiplication table). T2: pairs of fresh Haar elements never seen in
training (no finite-group analogue; tests generalization over the group manifold).
T3: the exact product versus 63 candidates placed at a fixed geodesic radius r from it
(resolution profile).

**Representation analysis.** For a fixed probe set of 2048 Haar elements we compute the
fraction of variance explained (FVE) of the centered activations z(g) by the sampled,
jointly orthonormalized spans of the Wigner matrix coefficients D^j_{mm'}, per spin
j ∈ {0, ½, 1, 3/2, 2}. Wigner D-matrices for half-integer j are computed exactly via the
action on homogeneous polynomials (`core/su2.py`, with unit tests for the homomorphism
property, unitarity, characters, and the center relation D^j(−q) = (−1)^{2j} D^j(q)).
Finite-sample leakage between spin subspaces bounds the FVE noise floor at ≈ 0.2%.
Supplementary diagnostics: the mean cosine between z(g) and z(−g); the sign of the
holonomy of z under a 2π rotation of SO(3) along one-parameter subgroups; and held-out
ridge regressions from z to the quaternion coordinates (an odd, spin-½ target) and to
the entries of qqᵀ (an even, spin-1 target).

## 3. Pilot results (one seed per cell)

| condition | train | T1 | T2 | FVE j=½ | j=1 | j=3/2 | j=2 | cos(z(g),z(−g)) | 2π holonomy | R²(q) | R²(qqᵀ) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SU(2), f=0.002 | 0.996 | 0.912 | 0.907 | **0.968** | 0.023 | 0.007 | 0.002 | −0.96 | negative | 0.998 | 0.987 |
| SU(2), f=0.01 | 0.978 | 0.926 | 0.924 | **0.797** | 0.182 | 0.017 | 0.002 | −0.66 | negative | 0.995 | 0.912 |
| SO(3), f=0.002 | 1.000 | 0.834 | 0.827 | 0.002 | **0.980** | 0.000 | 0.013 | +1.000 | +1 | −0.006 | 0.995 |
| SO(3), f=0.01 | 0.999 | 0.885 | 0.884 | 0.002 | **0.909** | 0.001 | 0.084 | +1.000 | +1 | −0.006 | 0.999 |

Figure: `gallery/su2/su2_pilot_summary.png`.

**(a) Half-integer isotypic blocks appear only in the SU(2) condition.** The SU(2)
models place 80–97% of encoder variance in the spin-½ block; the representation is odd
under the center (z(−q) ≈ −z(q)), reverses sign under a 2π rotation, and the quaternion
is linearly decodable from it (held-out R² = 0.998). In the SO(3) control every
half-integer block remains at the 0.2% measurement floor; variance concentrates in
spin 1 (with spin 2 next), the representation is exactly even, and the quaternion is not
decodable (R² ≈ 0) while qqᵀ is (R² ≥ 0.99). Each condition thus concentrates in the
lowest-dimensional faithful irrep of its own group — the continuous counterpart of the
finite-group result in `RESULTS.md`.

**(b) Higher spins enter in order of Casimir eigenvalue.** Over training, the SU(2)
f = 0.01 run adds spin-1 (2% → 18%) and then spin-3/2 (→ 1.7%) content on top of the
dominant spin-½ block; the SO(3) control adds spin-2 (→ 8.4%) on top of spin-1. The
larger training fraction produces both a broader spectrum and better fine-scale
resolution (T3 at r = 0.2: 0.60 vs 0.42), consistent with a bandwidth interpretation:
resolving the product at scale r requires irreps up to j ~ 1/r.

**(c) Generalization is immediate; no grokking at these budgets.** T1/T2 reach ≈ 0.94
(SU(2), f = 0.01) by the first evaluation at step 1000 and are flat or slightly
declining thereafter; there is no delayed transition. The SU(2) models operate at
96–97% of the Bayes ceiling (0.951) from early in training. We attribute the absence of
delayed generalization to the smooth structured input encoding (cf. GrokTransfer,
arXiv:2504.13292: embedding structure controls the presence of the delay). The one
regime with a persistent train/test gap is SO(3) at f = 0.002 (train 1.000, T1 0.834,
declining), which 200k steps of weight decay did not resolve.

**(d) The quotient task is uniformly harder.** At matched budget the SO(3) control
generalizes 3–8 points below SU(2) and further below the shared Bayes ceiling. Cause
not yet identified; candidates include information loss and poor conditioning of the
even encoding near its fold locus, and the difference between the intrinsic geometries
of S³ and RP³.

## 4. Causal and mechanistic evidence (added 2026-08-05; pilot checkpoints, one seed)

We supplement the spectra with interventions on the trained models
(`experiments/su2_causal.py`, results in `data/su2/<run>/causal_200000.json`). All
interventions act on the argument-encoder activations z_a, z_b; the candidate encoder
is left intact. We report three metrics on fresh Haar pairs: ranking accuracy against
63 excluded Haar negatives ("task"); accuracy against 63 negatives at geodesic radius
0.2 ("fine"); and center discrimination D_c = P[s(a,b,ab) > s(a,b,−ab)], i.e. whether
the model prefers the true product to its antipode (chance 0.5).

**Interventions** (f = 0.01 models; f = 0.002 results are qualitatively identical and
recorded in the JSON files):

| intervention | SU(2): task / fine / D_c | SO(3): task / fine / D_c |
|---|---|---|
| none | 1.000 / 0.596 / 1.000 | 1.000 / 0.612 / 0.500 |
| even projection: z(g) ← (z(g)+z(−g))/2 | **0.021 / 0.000 / 0.498** | 1.000 / 0.612 / 0.500 |
| odd projection: z(g) ← (z(g)−z(−g))/2 | 0.908 / 0.082 / 1.000 | 0.030 / 0.000 / 0.500 |
| function-ablate spin-½ | **0.024 / 0.000 / 0.506** | 1.000 / 0.603 / 0.500 |
| function-ablate spin-1 | 1.000 / 0.378 / 1.000 | **0.025 / 0.000 / 0.500** |
| ablate 4 random directions | 1.000 / 0.540 / 1.000 | 1.000 / 0.578 / 0.500 |

Here "function-ablate spin-j" subtracts from z the component linearly predictable from
the D^j matrix coefficients of the input, fit on an independent probe set.

The pattern is a double dissociation. For the SU(2) model, removing the spin-½
component — by parity projection or by function ablation — reduces the task to chance
(0.02 ≈ 1/64) and center discrimination to 0.5, while removing spin-1 leaves the task
and D_c intact and costs only fine resolution (0.596 → 0.378): the spin-½ block is
necessary for composition and for distinguishing ±g, and spin-1 contributes
resolution. For the SO(3) model the roles transpose: ablating spin-1 is fatal, while
ablating the (absent) spin-½ component is a no-op — as is the even projection, which
is an exact no-op by construction and serves as a consistency check of the
intervention machinery. Random rank-matched ablations leave all metrics essentially
unchanged.

Two methodological notes. (i) Subspace estimates of near-floor isotypic blocks are
noise-dominated, and ablating them spuriously damages the model (e.g. the estimated
"spin-½ subspace" of the SO(3) model overlaps its spin-1 directions); the
function-space ablation does not have this failure mode and is the instrument we
report. (ii) The odd projection also removes the mean and any j = 0 component, which
shifts the interaction MLP off-distribution; its damage upper-bounds but does not
isolate the contribution of even spin content (the per-spin function ablations do).

**Score–character fit.** Regressing the trained scorer s(a,b,c) on the characters
χ_j(ab c⁻¹) over triples with c at mixed distances from ab: for the SU(2) models,
χ_{1/2} alone explains 97.6% (f = 0.01) and 87.7% (f = 0.002) of the score variance,
with no measurable gain from adding any other character up to j = 3. For the SO(3)
models, χ_{1/2} explains 0.0% and χ_1 brings the fit to 84.9% / 76.1%. The learned
decision rule is, to good approximation, a single-character function of ab c⁻¹ — the
form of the GCR logit of Chughtai et al. — built on the lowest faithful irrep of the
respective group.

**Composition test.** For the SU(2) models, ridge probes decode the quaternion from
z (held-out R² = 0.997) and from the product locus h (R² = 0.988 for the product ab
on fresh held-out pairs; pair-shuffled control R² = −1.03). Composing the element
probes algebraically matches the truth — median cos(q̂(a)·q̂(b), ab) = 0.998 — and
matches the independent h-probe (0.995); the shuffled control is at 0. The internal
state h therefore carries the composed spin-½ representation, which a lookup-table
account of the scorer does not predict. For the SO(3) models the h-probe decodes the
product's spin-1 target (qqᵀ) at R² = 0.92 / 0.88 against a shuffled control of −0.9.

## 5. Second wave of experiments (launched 2026-08-05)

A 26-run queue (concurrency 4, single T4) covers three batches: the candidate-count
sweep (complete, below), a training-fraction sweep at low f (f ∈ {0.001, 0.0005,
0.0002}, 300k steps, fine evaluation grid — the delayed-generalization search), and
seed replication (seeds 1–4 for the four pilot cells, 100k steps).

### 5.1 Candidate-count sweep (complete; one seed, 60k steps)

K ∈ {64, 128, 256} at f = 0.01. The Bayes ceiling of the ranking task falls with K
(0.951 / 0.936 / 0.922, Monte-Carlo), since more negatives give more chances that one
falls closer to ab than the noisy positive. Caveat: the K = 256 runs used batch 2048
(memory), the others 4096.

| K | SU(2): T1 (× ceiling) / T3 r=0.2 / FVE j=½ / j=1 | SO(3): T1 (× ceiling) / T3 r=0.2 / FVE j=1 / j=2 |
|---|---|---|
| 64 | 0.921 (0.97) / 0.618 / 0.848 / 0.136 | 0.886 (0.93) / 0.609 / 0.927 / 0.068 |
| 128 | 0.902 (0.96) / 0.743 / 0.865 / 0.120 | 0.843 (0.90) / 0.561 / 0.916 / 0.078 |
| 256 | 0.884 (0.96) / 0.921 / **0.975** / 0.018 | 0.851 (0.92) / 0.793 / **0.949** / 0.046 |

Three observations. (i) Held-out accuracy tracks the falling ceiling at a roughly
constant fraction (~0.96 for SU(2)). (ii) Fine resolution improves markedly with K
(SU(2), r = 0.2: 0.62 → 0.74 → 0.92; r = 0.1: 0.14 → 0.19 → 0.36). (iii) Contrary to
the prediction recorded before the sweep, the spin spectrum **concentrates** rather
than broadens: at K = 256 the SU(2) encoder holds 97.5% of its variance in spin-½,
and the score–character fit rises to R² = 0.989 on χ_{1/2} alone (SO(3): 0.907 on
χ_1). The concentration is not an artifact of the batch-size difference: at matched
effective epochs the K = 64 model is already far broader (j=½ 0.893, j=1 0.095 at
step 25k).

This revises the interpretation of pilot observation (b). Because χ_{1/2}(ab c⁻¹) is
monotone in the geodesic distance from c to ab, ranking by the fundamental character
is Bayes-optimal at *any* resolution; fine resolution therefore does not require
higher-spin content, and the models with the purest spin-½ representations in fact
resolve best. The spin-1 content that accumulates at K = 64 coincides with slightly
*declining* fine resolution over training, consistent with drift rather than with a
resolution mechanism. The heat-kernel bandwidth argument applies to representing the
score as a density (log K_t), not to ranking; a direct test of bandwidth selection
should therefore use a regression or density objective (Task 6, revised).

The causal suite replicates on the K = 256 checkpoints: function-ablating spin-½
collapses the SU(2) model (task 1.000 → 0.027, D_c → 0.50) and is a no-op for the
SO(3) model (1.000 → 1.000); the composition test strengthens (median cosine 0.999,
h-probe agreement 0.998, shuffled controls at 0).

### 5.2 Training-fraction sweep (complete; one seed, 300k steps, evaluations every 500)

f ∈ {0.0002, 0.0005, 0.001} joined with the pilot values {0.002, 0.01}. T2 below is
the mean of the last ten evaluations; "gap" = train − T2; "steps to 90%" is the first
step at which T2 reaches 90% of its final value.

| f | pairs | SU(2): train / T2 / gap / steps to 90% | SO(3): train / T2 / gap / steps to 90% |
|---|---|---|---|
| 0.0002 | 839 | 1.000 / 0.726 / 0.27 / 500 | 1.000 / 0.636 / 0.36 / 500 |
| 0.0005 | 2,097 | 1.000 / 0.826 / 0.17 / 500 | 1.000 / 0.702 / 0.30 / 500 |
| 0.001 | 4,194 | 1.000 / 0.850 / 0.15 / 500 | 1.000 / 0.779 / 0.22 / 500 |
| 0.002 | 8,389 | 0.996 / 0.905 / 0.09 / 1,000 | 1.000 / 0.834 / 0.17 / 1,000 |
| 0.01 | 41,943 | 0.978 / 0.920 / 0.06 / 1,000 | 0.999 / 0.881 / 0.12 / 1,000 |

Three conclusions. **(i) There is no delayed-generalization transition at any
training fraction.** Every run reaches 90% of its final test accuracy within 500–1000
steps and is statistically flat for the remaining ~300k steps; lowering f increases
the size of the persistent train–test gap but does not produce a memorize-then-
generalize dynamic. In the vocabulary of the grokking literature: the time to
generalize is essentially independent of dataset size here, and only the *level* of
generalization depends on it. **(ii) Memorization and the group-structured
representation coexist rather than compete.** Even at f = 0.0002 — 839 pairs, train
accuracy 1.000, a 0.27 generalization gap — the SU(2) encoder still holds 97.6% of
its variance in the spin-½ block with quaternion decode R² = 0.993. The model fits
the training set on top of, not instead of, the group representation; 300k steps of
weight decay (1.0) neither removes the memorized component nor improves the test
accuracy. This differs qualitatively from the circuit-competition picture used to
explain grokking on finite algebraic tasks. **(iii) The quotient condition is more
data-hungry at every f**, with the deficit widening from 4 points (f = 0.01) to 9
points (f = 0.0002) of T2.

We attribute the absence of a delayed transition to the smooth, structured input
encoding: the encoder is a continuous function of the observation, so partial
generalization is available immediately, unlike free per-token embeddings. Whether a
delayed transition can be recovered by inflating the initialization scale or
roughening the encoding kernel remains open (Task 8).

### 5.3 Seed replication (complete; 5 seeds per cell, compared at step 100k)

Seeds 1–4 replicate the pilot in each of the four principal cells; aggregates
(mean ± sd; full table in `summary/seed_summary.json`):

| quantity | SU(2) f=0.002 | SU(2) f=0.01 | SO(3) f=0.002 | SO(3) f=0.01 |
|---|---|---|---|---|
| T2 | 0.901 ± 0.006 | 0.921 ± 0.003 | 0.842 ± 0.015 | 0.883 ± 0.003 |
| FVE j=½ | 0.964 ± 0.010 | 0.835 ± 0.010 | 0.0023 ± 0.0004 | 0.0023 ± 0.0004 |
| FVE j=1 | 0.027 ± 0.009 | 0.148 ± 0.010 | 0.981 ± 0.001 | 0.917 ± 0.008 |
| cos(z(g), z(−g)) | −0.95 ± 0.02 | −0.72 ± 0.02 | +1.000 | +1.000 |
| R²(q) | 0.998 | 0.995 | −0.016 | −0.012 |
| D_c (intact) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5000 | 0.5000 |
| task acc, spin-½ ablated | 0.022 ± 0.002 | 0.026 ± 0.002 | 0.997 ± 0.002 | 1.000 ± 0.000 |
| task acc, spin-1 ablated | 0.998 ± 0.001 | 1.000 ± 0.000 | 0.027 ± 0.002 | 0.027 ± 0.003 |
| score–character R² | 0.894 ± 0.024 (χ_{1/2}) | 0.974 ± 0.006 (χ_{1/2}) | 0.747 ± 0.088 (χ_1) | 0.876 ± 0.014 (χ_1) |

No seed deviates from the pattern in any cell: the spin spectra, the parity of the
representation, the decodability asymmetry, and the full causal double dissociation
reproduce with small variance. Unlike the finite-group setting — where seed-dependent
irrep selection is the norm (Chughtai et al.) — irrep selection here is deterministic:
every seed selects the lowest faithful irrep of its group. The plausible reason is
that SU(2) has a unique irrep per dimension (no Galois-conjugate or equivalent-irrep
multiplicity for the model to choose among), so the arbitrariness observed on finite
groups has no room to appear.

## 5.4 Mechanistic loop closure (added 2026-08-28; 5 seeds per cell)

Following the research plan of 2026-08-09 (archived in
`ideas/research_plan_2026-08-09.md`), three analyses close the remaining gap
between "the information lives in the spin-j block" and "the network computes with
a representation" (`experiments/su2_mechanism.py`; per-run JSONs
`mechanism_*.json`). All numbers are mean ± sd over seeds.

**(a) The latent space carries a genuine group action (plan A1).** For Haar-random
k we fit affine maps T_k with z(kg) ≈ T_k(z(g)) on a probe set and evaluate held
out. The activations concentrate on a low-dimensional subspace (99.9% of variance
in 9–14 PCA dimensions), and the dominant isotypic block has *exactly* the
Schur-corrected real coefficient-block dimension in every seed:
dim_R M_W = (dim_R W)²/dim_R End_G(W) = 16/4 = 4 for SU(2) spin-½
(quaternionic type) and 9/1 = 9 for the SO(3) vector (real type). On that block the fitted translation operators match the irrep
matrices up to a fixed change of basis, measured basis-invariantly:
tr W̃_k versus d_j·χ_j(k) has correlation 1.0000 ± 0.0000 (relative error
1.1 ± 0.1%), and the eigenvalue phases of W̃_k match those of d_j copies of
D^j(k) to 0.0014 ± 0.0001 rad (SU(2)) and 0.012–0.021 rad (SO(3)). Globally,
held-out equivariance R² is 0.90–0.98, the homomorphism defect
||T_k∘T_h − T_{kh}|| is 0.04–0.16 (relative), T_e recovers the identity to
numerical precision, and inverse defects are ~0.25. The latent space does not
merely correlate with representation coordinates; left translation acts on it by
the representation.

**(b) The product state is the composed representation, and it steers the model
(plan A2).** The interaction output h(a, b), analyzed as a function of the product
ab, has 89 ± 1% of its variance in the spin-½ block (SU(2); 73–75% in spin-1 for
SO(3)) with ~10–27% residual dependence beyond the product. Function-ablating the
D^j(ab)-predictable component of h collapses the task to chance for the model's
own irrep and leaves the other condition's models untouched. The strongest
result is the patching test: replacing h(a, b) by its *ridge reconstruction from
D^{1/2}(a′b′) alone* redirects the model to select a′b′ with probability
0.996–0.999 in every seed — as well as or better than patching the full
activation h(a′, b′) (0.92–0.98) — while the wrong-irrep reconstruction control
redirects at chance (0.016 ± 0.008). The eight realified spin-½ coefficients of
the product are causally sufficient to control the model's output.

**(c) Seeds differ by an intertwiner (plan A4).** Affine stitches between seeds
achieve held-out R² 0.96–0.99, and the stitch approximately intertwines the
fitted group actions (relative defect 0.07–0.15): seed universality is a
statement about a shared representation up to gauge.

## 5.5 Queue v3 results (2026-08-28/29; one seed each)

### (a) Observation-map robustness (plan A3) — confirmed

With the landmark encoding replaced by a frozen random two-layer tanh MLP of the
raw quaternion (quotient control = the center-orbit average of the same map), the
SU(2)/SO(3) result reproduces at 60k steps: SU(2) places 89.8% of encoder variance
in spin-½ (quaternion decode R² 0.993, cos(z(g), z(−g)) = −0.83); the symmetrized
control pins spin-½ at the 0.2% floor and concentrates in spin-1 (87.6%, plus
spin-2 at 9.8%). The function-value encoding f(g·xᵢ) (random Fourier f, fixed
probe points; 60k steps) reproduces the pattern as well, with somewhat more
integer-spin admixture on the SU(2) side: spin-½ 69.1% + spin-1 21.4%, with
quaternion decode R² 0.990 and cos(z(g), z(−g)) = −0.52; the symmetrized control
has spin-½ at the floor with spin-1 82.5% + spin-2 15.8%. Across three
qualitatively different observation maps, the selected irrep is therefore driven
by the task and group, not by the observation map.

### (b) Heat-kernel regression (plan D1) — bandwidth is forced, and allocated
bottom-up in Casimir order

MSE regression onto the standardized K_t(ab c⁻¹), t ∈ {0.3, 0.1, 0.05}, 60k
steps. Unlike ranking (Section 5.1), this target requires spectral content up to
the (2j+1)e^{−t j(j+1)} envelope. Results (encoder E / candidate encoder E_c FVE
per spin at 60k):

| t | test MSE | dominant sectors E | dominant sectors E_c | envelope support |
|---|---|---|---|---|
| 0.3 | 0.051 | ½: 0.29, 1: 0.50, 3/2: 0.20 | ½: 0.16, 1: 0.22, 3/2: 0.45, 2: 0.16 | j ≲ 2 |
| 0.1 | 0.324 | ½: 0.64, 3/2: 0.34 | 1: 0.42, 3/2: 0.53 | j ≲ 3.5 |
| 0.05 | 0.459 | 1: 0.67, 3/2: 0.22 | 3/2: 0.42, 2: 0.36, (5/2, 3 above floor) | j ≲ 5 |

Two conclusions. (i) The single-irrep concentration of the ranking task is a
property of the objective, not of the architecture: the same model develops a
genuinely multi-spin representation the moment the target requires one, and the
represented sectors extend with 1/t in order of Casimir eigenvalue. (ii) At fixed
capacity and training budget the model implements a *truncated* expansion: at
t = 0.1 and 0.05 it fits the lowest-Casimir sectors the envelope demands and
leaves the higher tail unfit, paying the corresponding MSE (0.32, 0.46 of target
variance). The allocation is bottom-up in Casimir order — direct evidence for the
low-Casimir spectral bias conjectured in the theory questions (plan G1). Whether
the cutoff is capacity- or optimization-limited (width, steps) is a follow-up.

### (c) SU(3) vs PSU(3) (plan B) — both pre-registered hypotheses confirmed

Machinery: `core/su3.py` (exact matrix coefficients for (p,q) ∈ {(0,0), (1,0),
(0,1), (1,1), (2,0), (0,2), (3,0), (0,3)}, self-tested for homomorphism,
unitarity, center characters, Schur orthogonality) and `experiments/su3_train.py`;
the PSU(3) control uses the center-orbit-averaged observation map. Real
activations cannot distinguish an irrep from its conjugate, so spectroscopy
tracks conjugate-pair sectors 3⊕3̄, 8, 6⊕6̄, 10⊕10̄.

**SU(3)** (100k steps): T1 0.989 / T2 0.991; **97.9% of encoder variance in the
fundamental pair 3⊕3̄** (adjoint 0.6%, 6⊕6̄ 0.7%, 10⊕10̄ 0.2%); charged-energy
fraction 0.993; mean cos(z(U), z(ωU)) = −0.489 against the exact triality-1
prediction of cos(2π/3) = −0.5; fundamental entries decode at R² 0.988.

**PSU(3)** (100k steps, complete): train 1.000, T1 0.779, T2 0.682 — the
quotient task is again the substantially harder one, now with a persistent
generalization gap at this budget (the SO(3)-versus-SU(2) hardness pattern,
amplified for the larger center and group). The representation is unambiguous
and stable from early training: **93.7% of encoder variance in the adjoint 8**
(candidate encoder: 96.1%) — the smallest faithful center-trivial irrep — with
the charged sectors at the floor (3⊕3̄ 0.6%, 6⊕6̄ 0.1%), 10⊕10̄ at 2.6%,
charged-energy fraction exactly 0, and the adjoint matrix decodable at R² 0.69.
Optimization thus selects the adjoint when the fundamental is forbidden, as
hypothesized — with the caveat that the model solves the quotient task only
partially at this budget, so the selection statement is about the representation
it builds, not yet about a fully solved task.

Together with SU(2)/SO(3), this is the second instance of the central-quotient
pattern: **the learned representation identifies the global form of the group —
the smallest faithful irrep of G, and the smallest faithful irrep of G/Z when
the observations are Z-invariant.**

**Causal evidence (plan B3/B4; `experiments/su3_causal.py`; now 5 seeds,
mean ± sd).** With D_ω = P[s(a,b,ab) beats both s(a,b,ω^k ab)] (tie-aware;
chance 1/3):

| intervention on z_a, z_b | SU(3): task / D_ω | PSU(3): task / D_ω |
|---|---|---|
| none | 0.998 ± 0.001 / 1.000 | 0.739 ± 0.019 / 0.333 |
| center projection (orbit average) | **0.016 ± 0.001 / 0.337 ± 0.008** | 0.739 / 0.333 (exact no-op) |
| charged projection (remove invariant part) | 0.998 / 1.000 | **0.015 ± 0.001 / 0.333** |
| function-ablate 3⊕3̄ | **0.018 ± 0.002** | 0.733 (no-op) |
| function-ablate adjoint 8 | 0.997 (no-op) | **0.014 ± 0.003** |

The Z₃ analogue of the SU(2) double dissociation, in both directions. The
composition test transfers: SU(3) fundamental probes compose algebraically
(median cosine of D̂(a)D̂(b) against D(ab): 0.977; shuffled −0.02); PSU(3)
adjoint probes compose at 0.70 (shuffled −0.02; the weaker probe R² ≈ 0.69
compounds in the product). Character fit (B4): for SU(3), the realified
fundamental character explains 70% of the score variance with no gain from any
higher sector — consistent with a single-character rule read through a
nonlinearity (Re χ_{(1,0)}(g) = (6 − ||I − g||²_F)/2 is exactly the chordal
distance statistic, so a monotone function of it is Bayes-reasonable); a
quantitative nonlinear fit is future work. For PSU(3) the cumulative fit is
confounded at the first step: the regression sample is concentrated near the
identity, where charged and invariant characters correlate (Schur orthogonality
holds only under Haar), so the (1,0) row absorbs variance that belongs to the
invariant sectors; a fit restricted to descending sectors is the clean version
(future work). One metric limitation: the "fine" evaluation at algebra radius
0.4 sits at both models' resolution floor (the training noise scale) and reads
≈ 0 everywhere; larger radii should be used next time.

**Seed replication (queue v4).** Five seeds per condition (the last PSU(3) seed
evaluated at its 65k interim; all others at 100k), with every seed conforming:

| quantity | SU(3) | PSU(3) |
|---|---|---|
| T2 | 0.994 ± 0.001 | 0.685 ± 0.020 |
| FVE 3⊕3̄ | **0.971 ± 0.009** | 0.0059 ± 0.0004 |
| FVE adjoint 8 | 0.009 ± 0.003 | **0.934 ± 0.007** |
| FVE 10⊕10̄ | 0.002 | 0.027 ± 0.003 |
| cos(z(U), z(ωU)) | −0.484 ± 0.005 (pred. −0.5) | +1.000 |
| charged-energy fraction | 0.989 ± 0.005 | 0.0000 ± 0.0000 |
| R²(fundamental) | 0.978 ± 0.006 | −0.021 |
| R²(adjoint) | 0.46 ± 0.09 | 0.699 ± 0.012 |

As in the SU(2)/SO(3) pair, irrep selection is deterministic across seeds.

**D1 capacity probes (t = 0.05, complete).** The 2×-length default-width run
improved test MSE from 0.459 (60k) to 0.383 (120k) while its candidate-encoder
spectrum extended to j = 2 at 60% with j = 5/2 and j = 3 above floor — the
spectral cutoff is at least partly optimization-limited, and the allocation
remains bottom-up in Casimir order throughout. The 2×-width model at unchanged
hyperparameters never escaped the mean predictor (MSE 0.997 through 120k):
its encoders formed low-spin content but the score head never coupled to it.
Widening therefore did not buy bandwidth at this temperature; it destabilized
optimization (a learning-rate/weight-decay retune for the wider model is the
obvious follow-up before drawing capacity conclusions).

## 5.6 Representation-ring / fusion analysis (plan E; added 2026-08-29)

Functions on G×G decompose into (j_a, j_b) sectors under the two-sided action
(basis: products D^{j_a}(a)_{mk} D^{j_b}(b)_{ln}); `experiments/su2_fusion.py`
computes the two-variable spectroscopy of the interaction MLP's hidden layer
and output h, plus a fusion-selectivity statistic: within the diagonal (j, j)
sector, the composed representation D^j(a)D^j(b) = D^j(ab) spans only the
matrix-product contraction subspace of the full realified (j, j) sector — 4 of
16 real dimensions for SU(2) (j = ½, quaternionic type) and 9 of 81 for SO(3)
(j = 1, real type); sector dimensions are the measured ranks of the sampled
realified product basis and depend on Schur type.

Results (f = 0.01, now 5 seeds per condition): for SU(2), h places
89.9 ± 1.4% of its variance in the (½, ½) sector, and **98.2 ± 0.3% of that
sector's energy lies in the 4-of-16-dimensional contraction subspace**; the
hidden layer
occupies exactly the channels tensor-generated from the encoder content —
pass-through (½, 0) and (0, ½) at ≈ 25% each, the bilinear (½, ½) at 22%, and
small (1, 0)/(0, 1) — with all other channels at or near the floor. For SO(3):
h has 77.5 ± 1.6% in (1, 1) with 93.9 ± 0.9% of it in the
9-of-81-dimensional contraction subspace. The interaction layer is therefore not an arbitrary bilinear coupling
of the represented sectors: it concentrates on the specific Clebsch–Gordan
contraction that implements the group product — the network's internal algebra
follows the representation ring.

## 5.7 Spin(4) and its central quotients: the selection principle (added 2026-08-29)

Spin(4) = SU(2) × SU(2) has center Z₂ × Z₂ and, because that center is not
cyclic, **no faithful irrep**: any model must carry a direct sum. Its three
central quotients realize different global forms with the same Lie algebra.
Machinery: `core/spin4.py` (pair-quaternion operations, (j_L, j_R) sector
spectroscopy, descent rules, orbit-averaged encodings; self-tested);
`experiments/spin4_train.py`. Competing minimal-carrier hypotheses were
recorded in the module docstring before running; for two conditions the
minimal-total-dimension choice and the minimal-block-count choice disagree,
making them discovery runs. Results (now 5 seeds per condition, 100k steps,
N = 2048, f = 0.01; the table shows the seed-0 spectra, with distributions
below):

| condition (group) | T2 | learned sectors (FVE) | competing hypothesis (FVE) |
|---|---|---|---|
| spin4 (Spin(4)) | 0.945 | **(½,0) 0.50 + (0,½) 0.48** | — (sum forced; no faithful irrep) |
| so4 (SO(4)) | 0.962 | **(½,½) 0.97** (vector rep) | — |
| leftq (SO(3)×SU(2)) | 0.968 | **(0,½) 0.73 + (1,0) 0.25** [5 dims] | (1,½) [6 dims]: 0.007 |
| pso4 (PSO(4)) | 0.969 | **(1,0) 0.50 + (0,1) 0.48** [6 dims] | (1,1) [9 dims]: 0.011 |

All center-orbit cosines match the representation-theoretic predictions
(e.g. leftq: L-flip invariant at +1.0, R-flip at −0.48 ≈ 0.25 − 0.73; pso4 all
+1.0). In the so4 condition the observations do distinguish (q_L, q_R) from
(−q_L, q_R), yet single-factor content is absent (R²(q_L) ≈ 0): nothing but
the task's own group is represented.

**Selection principle (answers plan G1's "which carrier?").** In both
discovery conditions optimization chose the smaller-dimensional reducible sum
over the single faithful irrep. Combined with the earlier conditions, all
eight tested group forms are consistent with one statement:

> *Trained on composition in a compact group G (observed through a generic
> Z-invariant map), the encoder concentrates on a **minimal-total-dimension
> faithful representation of G/Z**, reducible when necessary — and the
> single-irrep concentration seen for SU(2), SO(3), SU(3), PSU(3), and SO(4)
> is the special case where that minimal representation happens to be
> irreducible.*

(Case check: for SU(2), SO(3), SU(3), SO(4) the minimal faithful
representation is irreducible; for PSU(3), a simple group, every nontrivial
rep is faithful and the adjoint 8 is the smallest; for Spin(4), SO(3)×SU(2)
and PSO(4) the minimal faithful representation is a two-block sum, and that is
what is learned.)

**Block-resolved causal map (`experiments/spin4_causal.py`; one seed).** With
tie-aware center discriminations D_L, D_R, D_LR for the three center flips
(chance/tie 0.5), every intervention lands where the central characters place
it. spin4: removing the (−,+) character component — equivalently
function-ablating the (½,0) block — collapses D_L to 0.512 while D_R stays at
1.000, and mirror for (+,−)/(0,½); D_LR survives either single ablation (the
remaining block still sees the joint flip); the unused (½,½), (1,0), (0,1)
ablations are no-ops. so4: only the (½,½) removal matters (task 1.000 → 0.017,
both single-flip D's → 0.48); D_LR is an exact tie as the quotient requires.
leftq: the two blocks separate causally — ablating (0,½) kills D_R, D_LR and
most of the task (→ 0.555), while ablating (1,0) costs task accuracy
(1.000 → 0.862) with all center discriminations intact: one block carries the
SO(3)-factor geometry, the other the center charge. pso4: all D's at tie;
each learned block carries one factor (ablations → 0.807 / 0.806,
symmetric); everything else no-op. Each isotypic block is causally necessary
for exactly the task component and center discrimination that representation
theory assigns to its central character.

**Seed replication (5 seeds per condition).** Selection is deterministic here
as well: the learned carrier holds 0.973–0.984 (± 0.002) of encoder variance
in every condition, and the rejected larger faithful irreps sit at
0.0076 ± 0.0002 (leftq) and 0.0094 ± 0.0011 (pso4) in every seed. The
load-bearing causal cells replicate with near-zero variance: spin4, ablating
(½,0): D_L 0.507 ± 0.007 with D_R 1.000 ± 0.000; so4, ablating (½,½): task
0.017 ± 0.001; leftq, ablating (0,½): D_R 0.493 ± 0.006, and ablating the
factor carrier (1,0): task 0.853 ± 0.036 with center discriminations intact;
pso4, per-factor ablations: task 0.809 ± 0.016. T2 = 0.965–0.970 (± 0.003–0.021)
across conditions.

## 5.8 Spin(5) / SO(5) (added 2026-08-30; one seed)

The fifth group of the central-quotient program: Spin(5) = Sp(2) (2×2
quaternionic unitaries; machinery in `core/spin5.py`, self-tested), with the
carrier-cost search's pre-registered predictions of spinor 4 vs vector 5.

**spin5** (100k steps): T2 0.969; **97.1% of encoder variance in the spinor 4**
(vector 0.3%, adjoint 1.1%), cos(z(U), z(−U)) = −0.97, spinor coefficients
decode at R² 0.983 — as pre-registered.

**so5** (100k): T2 0.605 — the largest quotient-hardness gap yet — with the
spinor at the 0.5% floor and the carrier split **vector-5 at 68.7% + adjoint-10
at 22.1%**. The admixture is stable, not drift: the encoding's initialization
spectrum (5: 29%, 10: 56%) is inverted by step 15k and constant thereafter.
The minimal carrier is strongly preferred but not purified at this budget.
Under the v2 plan this condition is treated as a **stress test with a
precommitted decision rule** (Section 8, P2): if well-conditioned,
high-performing SO(5) models purify toward vector-5 the minimal-carrier
observation strengthens; if the vector+adjoint mixture survives replication
and conditioning-matched training, the observation must be revised — the
mixture is not to be explained away because vector-5 was expected. Both
faithful sectors being present is consistent with the faithful-character lemma
(any faithful carrier ranks).

### 5.8a Wide-model retune verdict (t = 0.05, complete)

At weight decay 0.1 the 2×-width model trains normally (the wd = 1.0 collapse
was a regularization pathology, not a capacity limit), and its candidate-
encoder spectrum extends beyond any default-width run — j = 5/2 and j = 3 at
3–7%, above the floor. But both learning rates land in a memorization regime:
train MSE 0.24 / 0.18 against test 0.56 / 0.63, versus 0.383 test for the
default-width model trained 120k steps. The three-way conclusion for the
heat-kernel task at this data budget: the spectral cutoff is not a hard
capacity limit (width extends the representable spectrum) and not purely an
optimization limit (longer default-width training keeps improving), but a
data/regularization trade-off — extra capacity is spent memorizing the frozen
training targets rather than sharpening the generalizing expansion. The
natural follow-up, if the bandwidth frontier matters, is more training data
(larger f) with the wide model.

## 5.9 P1A: two-sided latent action (added 2026-08-31; 5 seeds per condition)

The v2 plan's mechanism-discrimination test (`experiments/twosided.py`): fit
affine operators for left translation M^L_k (z(g) → z(kg)) and right
translation M^R_k (z(g) → z(gk)) and test the properties that distinguish a
full Peter–Weyl coefficient block — which is two-sided translation-stable, by
the G_L × G_R-irreducibility of V* ⊠ V — from smaller one-sided faithful
frames. Metrics: held-out left/right R², right-homomorphism defect
(M^R_{ab} vs M^R_b M^R_a), and the commutator ‖[M^L_a, M^R_b]‖ (exactly 0 for
a full block, since the regular actions commute). Feature systems: the trained
dominant block; the same architecture at random initialization; the raw
observation features; the explicit faithful frame (SO(3): two columns of the
rotation matrix, 6 real dims; SU(2): the quaternion, 4 dims); the true full
block (SO(3): 9 dims); and a random truncation of the true block at the
frame's dimension.

| system (SO(3) condition) | dim | left R² | right R² | hom-R defect | ‖[L,R]‖ |
|---|---|---|---|---|---|
| trained block | 9 | 0.989 ± 0.001 | **0.989 ± 0.001** | 0.012 | **0.012 ± 0.001** |
| true full block | 9 | 1.000 | 1.000 | 0.000 | 0.000 |
| faithful frame | 6 | **1.000** | **0.638** | **0.526** | 0.039 |
| random truncation | 6 | 0.674 | 0.675 | 0.411 | 0.448 |
| raw observations | 32 | 0.992 | 0.991 | 0.058 | 0.073 |
| random-init encoder | 16 | 0.892 ± 0.008 | 0.890 ± 0.008 | 0.179 | 0.160 ± 0.014 |

The prediction lands in full: the trained code supports accurate **commuting
left and right linear actions** (SU(2): L = R = 0.978 ± 0.003, commutator
0.010 ± 0.001), matching the true block and sharply separated from the
faithful frame — which passes the left test *exactly* (left translation is
linear on the two columns) and fails linear right closure exactly as the
missing-column analysis predicts (right R² 0.638 ≈ the 2/3 that the
uncorrelated third column implies; right-homomorphism defect 0.53) — and from
the random truncation, which fails everything. A Schur-type consequence makes
SU(2) its own control: the quaternionic block has real dimension
(dim_R W)²/dim_R H = 4, so the minimal frame *coincides* with the full block
and passes two-sided closure exactly — explaining, in v2 terms, why the SU(2)
mechanism could be both minimal and fully closed while for SO(3) (real type)
those properties part ways at 6 vs 9 dimensions. The learned representation
has precisely the structural property that full coefficient blocks possess and
cheaper one-sided frames lack. Whether that property is *why* full blocks are
selected is the P1B question (frame-vs-block matched training, running).

## 5.9a P1B: frame-versus-block matched training (added 2026-08-31; 5 seeds per arm)

The decisive v2 retraining experiment: identical architecture and task
(SO(3) condition, f = 0.01, 60k steps), with the observation replaced by exact
representation features — the 6-dimensional faithful two-column frame versus
the 9-dimensional full coefficient block (both even, standardized; encoder
still trainable, so the frame arm is free to reconstruct the missing column
nonlinearly — the question is what that costs).

| features | T2 | margin (T2) | total param norm | margin / norm |
|---|---|---|---|---|
| frame, 6 dims | 0.8914 ± 0.0023 | 6.49 ± 0.05 | 26.71 ± 0.10 | 0.243 ± 0.002 |
| block, 9 dims | 0.8891 ± 0.0027 | **7.29 ± 0.16** | **22.89 ± 0.07** | **0.319 ± 0.008** |

At statistically matched accuracy, the block mechanism attains a larger
margin at 17% lower total parameter norm — cheaper in every module (encoder
16.7 vs 19.6, interaction 9.8 vs 11.9, scorer 12.2 vs 13.7) — i.e. 31% more
margin per unit norm, with non-overlapping seed distributions. The plan's
first fork fires: **the full block is the cheaper end-to-end mechanism
despite more input coordinates**; the frame's saved storage is repaid with
interest by the composition cost its one-sided closure forces. Together with
P1A this gives the mechanistic explanation of full-block emergence: two-sided
linear closure → bilinear composition and linear scoring → lower regularized
cost at fixed margin → weight decay selects the block.

## 5.10 P4: score-level spectral recovery (added 2026-08-31; existing checkpoints)

`experiments/hk_analysis.py` regresses the learned score and the exact target
on a **sequentially orthogonalized** character design (Casimir order; raw
character columns are near-collinear under the sampling distribution, so
individual betas are unidentifiable — the identifiable object is the
incremental variance per sector, which is also the natural learning-order
statistic). "Recovery" = model's variance fraction in a sector / target's
fraction; 1.0 = the sector carries its required relative weight.

| model | recovery (j = ½, 1, 3/2, 2) | beyond |
|---|---|---|
| t = 0.3, 60k | 1.05, 1.05, 1.01, 0.07 | 0.000 |
| t = 0.1, 60k | 1.47, 1.39, 1.43, 0.00 | 0.000 |
| t = 0.05, 120k (long) | 1.58, 1.55, 1.46, **1.20** | 0.000 |

Three findings. (i) **A sharp truncation frontier**: recovered sectors sit at
ratio ≈ 1 (or above, compensating), sectors beyond the frontier at exactly 0 —
the score is a Casimir-truncated character expansion, now measured at the
score level rather than inferred from activations. (ii) **Sector-by-sector
filling in Casimir order**: the t = 0.05 long run's j = 2 recovery climbs
0.12 → 0.25 → 0.83 → 1.20 across 50k → 120k steps while the over-weighting of
j ≤ 3/2 relaxes (1.81 → 1.55) and j ≥ 5/2 stays at 0; likewise t = 0.3's
j = 3/2 fills from 0.13 to 1.00 after the lower sectors. (iii) The score–
character fit R² is 0.92–0.99 throughout: the learned score genuinely lives
in the character algebra.

**Band-limited calibration (target χ₁ + ½χ₂, known support {1, 2}; 60k, wd 1.0).**
Support recovery is exact where it should be: every out-of-support sector
(half-integers and j ≥ 5/2) sits at a 0.0000 floor, and — a bonus
quotient-selection check — because the target is a central-even class
function, the model on the SU(2)-faithful encoding builds an integer-spin
representation only (encoder FVE: spin-1 at 98.7%, spin-½ at 1%): the
*target's* central character, not the encoding's, determines the parity
content. The instructive deviation: the orthogonal χ₂ remainder carries 12.6%
of target variance, and the trained model **prunes it** — partial spin-2
recovery at 25–50k (0.33–0.44) is removed by 60k, the final test MSE (0.131)
matching the abandoned variance almost exactly, with encoder spin-2 content
at 0.07%. The truncation frontier is therefore not a hard bandwidth but a
**cost/benefit boundary set by regularization**: under weight decay 1.0 the
25-real-dimensional spin-2 block costs more norm than its 12.6% variance
contribution is worth. The falsifiable corollary was run and confirmed: at
weight decay 0.1 on the same target, test MSE drops 0.131 → 0.004 and the
recovery profile is essentially exact — (1.00, 1.01, 0.94, 0.95) across the
four orthogonalized sectors with every out-of-support floor at 0.0000 and the
score–character fit at R² 0.997. Calibration verdict: the analysis pipeline
recovers support and amplitudes exactly when the model retains them, and the
truncation observed at strong regularization is a real property of the
learned solution, not of the instrument. Remaining for P4: hk seed
replication (in queue).

## 5.11 P2 verdict: the SO(5) mixture survives — the selection observation is revised (2026-08-31; queue v8 complete)

The full stress test under the precommitted decision rule. Spin(5) replicates
perfectly (5 seeds: spinor-4 at 0.964–0.973, T2 0.969–0.989). For SO(5), the
vector+adjoint mixture is **stable in every condition tried**:

| condition | T2 | FVE vector-5 | FVE adjoint-10 |
|---|---|---|---|
| f = 0.01, seeds 0–4 | 0.55–0.64 | 0.680–0.695 | 0.211–0.233 |
| 200k extension (s0) | 0.620 | 0.688 | 0.214 |
| K = 256 | 0.469 | 0.664 | 0.224 |
| wd = 0.1 | 0.518 | 0.660 | 0.240 |
| f = 0.03 | **0.731** | 0.707 | **0.173** |

Per the rule — "do not explain away the mixture because vector-5 was
expected" — the minimal-carrier statement is **revised**: it holds in nine of
ten tested group forms and fails as stated for SO(5), where the learned
carrier is a reproducible vector ⊕ adjoint mixture at every achievable
budget. The one systematic trend is that the adjoint fraction falls as the
task gets solved (0.23 at T2 0.55 → 0.17 at T2 0.73 under more data, the only
manipulation that improved generalization; K and weight decay changed neither
performance nor mixture in the helpful direction). Two hypotheses remain
live, and the current data cannot separate them: (a) asymptotic purification
tracking task completion — the mixture is a transient of an unsolved task;
(b) a genuinely mixed minimum-cost end-to-end mechanism for SO(5) — which
the v2 cost framing makes respectable, since P1B showed the selected object
is the cheapest complete mechanism, not the smallest carrier. Deciding
between them requires a condition that actually solves SO(5) (larger f, or a
better-conditioned quotient encoding); the conditioning-matching diagnostics
(Gram/Fisher spectra, candidate-hardness matching) remain to be run and are
the other open P2 item.

## 5.12 P5 first pass: multi-spin fusion structure (2026-08-31; pilot)

On the multi-spin heat-kernel model (t = 0.3), the two-variable fusion
analysis extends the Section 5.6 result: the product state h concentrates on
the **diagonal** (j, j) sectors of the spins the score requires —
(3/2, 3/2): 0.33, (1, 1): 0.31, (½, ½): 0.16, off-diagonal contamination
≤ 0.03 per channel — and within (½, ½) the matrix-product contraction again
carries 98.1%. The hidden layer occupies the tensor-generating channels
((0,1), (1,0), (1,1), (0,3/2), (3/2,0), (3/2,3/2)). The literal
Clebsch–Gordan test with controlled bilinearity (quadratic-activation
interaction, where the channel decomposition is exact) is training
(`su2hk_t0.3_quad`); the per-channel allowed/forbidden-J verdicts and channel
ablations belong to that run's analysis.

## 6. Interpretation and limitations

- **Replication status.** The SU(2)/SO(3), SU(3)/PSU(3), Spin(4)-family and fusion
  results hold in 5/5 seeds per condition, including the causal cells. Single-seed:
  the K sweep, the training-fraction sweep, Spin(5)/SO(5), and the heat-kernel
  temperature sweep (seed replication of the latter two is scheduled under P2/P4).
- **Evenness of the quotient controls is by construction.** A Z-invariant input
  forces a Z-invariant representation, so the absence of forbidden sectors is the
  exact null constraint (descent rule), not a discovery. The discoveries are the
  selection among allowed sectors, its determinism across seeds, and the causal
  block-to-character assignment.
- **"Minimal faithful carrier" is a selection observation, not a mechanism law.**
  The v2 plan's correction stands at the head of the program: a mechanism need not
  store a full coefficient block at all — smaller faithful frames (SO(3): two
  columns, 6 real dims; SU(3): two columns, 12) separate the group with fewer
  coordinates but lose linear closure under right translation. What the experiments
  have established is that trained models build full Schur-corrected coefficient
  blocks (measured block dimensions exactly 4/9; two-sided closure now under direct
  test in Section 8, P1A) — why optimization prefers them over cheaper frames is
  the open cost-identification question (P1B).
- **Quotient hardness is confounded until conditioning is matched.** The uniform
  cover-easier-than-quotient pattern (SU(2)>SO(3), SU(3)>PSU(3), Spin(5)>SO(5))
  may reflect observation geometry, candidate geometry, or optimization rather
  than representation selection; P2 measures and matches Gram spectra, candidate
  hardness, and nearest-neighbor baselines before drawing conclusions.
- **Resolution is bounded by the training noise.** No ranking model resolves below
  the target noise scale σ; the heat-kernel line (P4) is the controlled setting for
  spectral-resolution questions.

## 7. Historical task list (v1, closed)

All thirteen v1 tasks are either complete (1–5, 6 via the heat-kernel line, 8,
9) or superseded/deprioritized by the v2 plan (7 pool-size sweep; 10 topology;
11 quadratic variant — subsumed by P5's controlled-bilinearity requirement;
12 transformer variant; 13 theory — continued as P3). See git history for the
v1 list.

## 8. v2 workstreams (active; plan at `ideas/research_plan_2026-08-30_v2.md`)

Standing policy unchanged: ≥5 seeds for reported results; pilots labeled.

- **P0 — hygiene/auditability.** Tracked results bundle
  (`summary/results_bundle.json`, regenerated by
  `experiments/export_bundle.py`: per-run config, seed, checkpoint hash, final
  metrics, causal/mechanism/fusion JSONs); pytest suite (`tests/`) wrapping the
  representation-theory self-tests; ridge probes now select λ on a validation
  split and report on a disjoint test split; Schur-type dimensionality
  terminology corrected throughout (real block dim = (dim_R W)²/dim_R End_G(W)).
- **P1A — two-sided latent action (existing checkpoints).**
  `experiments/twosided.py`: fit left AND right translation operators; test
  left/right homomorphisms and the commutation [T^L_a, T^R_b] ≈ 0; baselines:
  raw observations, random-init encoders, explicit faithful frames, and
  matched-dimension random truncations of the true block. Interpretive role:
  mechanism discrimination — does the learned code have exactly the two-sided
  linear closure that distinguishes full coefficient blocks from smaller
  one-sided faithful frames?
- **P1B — matched frame-vs-block training.** SO(3) with fixed 6-dim two-column
  frame features versus fixed 9-dim full-matrix features, all else matched;
  measure end-to-end cost (loss, margin, module norms, convergence, seed
  stability). Forks and falsifiers per the plan; no canonical cost predefined.
- **P2 — SO(5) conditioning stress test.** ≥5 seeds both conditions; f/K/budget
  variations; conditioning diagnostics (Gram spectra, distance distributions,
  NN baselines); precommitted decision rule for the vector+adjoint mixture.
- **P3 — theory.** Two-sided block lemma (complex + real Schur form) added to
  `summary/selection_theorem.md`; tr dV(X) statement corrected (purely
  imaginary in general; zero for the semisimple groups used here); the
  necessity lemma now carries explicit sampling/continuity assumptions
  (continuous candidate distributions can make exact kernel ambiguities
  measure-zero).
- **P4 — quantitative spectral bias.** Score-level coefficient recovery
  â_j(t)/(d_j e^{−t j(j+1)}) on the heat-kernel models
  (`experiments/hk_analysis.py`), learning-order curves, j_cut ~ t^{−1/2};
  calibration first on a band-limited target with exactly known support;
  seed replication.
- **P5 — literal Clebsch–Gordan.** Off-diagonal fusion channels
  (j₁ ⊗ j₂ decompositions with allowed/forbidden-J verdicts and causal
  channel ablations) on the multi-spin heat-kernel models, where the encoder
  genuinely carries several spins.
- **P6 — new groups only when they discriminate predeclared cost theories.**

**Big picture (v2).** Established: networks trained on compact-group
composition develop causal representation-theoretic mechanisms sensitive to
the global form of the group, concentrating on small faithful carriers allowed
by the quotient, computing by the representation's own composition law.
Open: why optimization selects the particular feature system it does once the
cost of implementing the group law is counted together with the cost of
storing the features — with two-sided linear closure of full Peter–Weyl blocks
as the leading candidate explanation, and quotient conditioning and spectral
bias as first-class companion questions.

## Appendix: repository pointers

- Group machinery (each with `uv run python -m core.<mod>` self-tests):
  `core/su2.py` (Wigner D^j incl. half-integer, encodings, heat kernel),
  `core/su3.py`, `core/spin4.py`, `core/spin5.py`; `core/su2_model.py` (scorer).
- Training drivers: `experiments/{su2,su3,spin4,spin5}_train.py`
  (`--resume`; checkpoints include optimizer state); queue scripts
  `experiments/run_queue_v*.sh` (launch detached with `setsid`).
- Analysis: `experiments/su2_causal.py`, `su3_causal.py`, `spin4_causal.py`
  (interventions), `su2_mechanism.py` (A1/A2/A4), `twosided.py` (P1A),
  `su2_fusion.py` (representation ring), `hk_analysis.py` (P4),
  `carrier_search.py` (cost-model search), `su2_seed_summary.py`,
  `export_bundle.py` (P0 bundle).
- Theory: `summary/theory_notes.md`, `summary/selection_theorem.md`.
- Tracked results: `summary/results_bundle.json`, `summary/seed_summary.json`.
- `data/<group>/<run>/` — logs/checkpoints (not in git); `tests/` — pytest suite.
- `RESULTS_SU2.md` — public-facing SU(2) summary; `RESULTS.md` — finite groups.
- `ideas/` (not in git) — literature review, design decisions, archived plans
  (`research_plan_2026-08-09.md`, `research_plan_2026-08-30_v2.md`,
  `understanding_notes_2026-08-30.md`).
