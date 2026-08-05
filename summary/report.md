# Learning and Interpreting Continuous Group Operations: SU(2) and its Quotient SO(3)

**Status:** evolving working report for agent/collaborator handoff. Last updated 2026-08-05
(after the four-run pilot, the first causal analyses, and the candidate-count sweep). Single-seed results throughout — treat every number below as a
pilot observation pending the seed sweep (Task 1).

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

## 6. Interpretation and limitations

- **One seed per condition.** The finite-group literature documents seed-dependent
  algorithm and irrep selection; every claim above requires the ≥5-seed replication
  (Task 1) before it is more than a pilot observation.
- **Evenness of the SO(3) control is by construction.** An even input forces an even
  learned representation, so the absence of half-integer blocks in the control is
  necessary, not discovered. The informative content is (i) on the SU(2) side — the
  model is not forced to build a dominant, clean spinor block, yet does — and (ii) the
  identity of what each condition builds: the lowest faithful irrep of the respective
  group, followed by the next irreps in Casimir order. The designed causal version of
  the double-cover claim (ablate the spin-½ block; measure loss of ±q discrimination)
  is Task 2.
- **The spectra alone are correlational; Section 4 addresses this in part.** FVE
  concentration locates information but does not identify the algorithm (Stander et
  al., arXiv:2312.06581, showed coset circuits reproduce isotypic signatures; Wu et
  al., arXiv:2410.07476, showed causal patching can also under-determine the
  mechanism). The Section 4 results — necessity of the spin-½ block, sufficiency of a
  single character to explain the scores, and the composition test with shuffled
  controls — go substantially beyond spectra, but per-argument equivariance (Task 5)
  and replication across seeds remain outstanding before mechanism-level claims are
  made with confidence.
- **Resolution is bounded by the training noise.** No model ranks accurately at
  r ≤ 0.05 (near chance); whether any training regime yields resolution below the
  target noise scale σ is open (Task 6).

## 7. Task list (evolving)

Standing policy: **every reported experiment uses ≥5 seeds**; pilot (1-seed) results
must be labeled as such. Tasks are ordered roughly by priority; none is committed to a
particular sweep — they are a menu for future runs.

1. **Seed replication of the pilot.** *In progress (queue item 3).* Seeds 1–4 ×
   {SU(2), SO(3)} × f ∈ {0.002, 0.01}; report distributions of FVE spectra, T1/T2,
   and decode R². Establishes the pilot findings as results.
2. **Causal double-cover test.** *Pilot complete (Section 4); repeat across seeds and
   extend to h.* Remaining: ablations applied at the product locus h;
   restricted/excluded-loss curves per isotypic block during training (the progress
   measures of Nanda et al., arXiv:2301.05217, transposed to spins).
3. **Logit–character fit.** *Pilot complete (Section 4).* Remaining: compare fitted
   coefficients quantitatively against the heat-kernel envelope (2j+1)e^{−tj(j+1)},
   and repeat at larger K and smaller σ where more characters should be needed.
4. **Composition test.** *Pilot complete (Section 4).* Remaining: the patched
   version (does the h-probe transfer under activation patching of one argument?),
   and the SO(3) case with a proper rotation-matrix composition step.
5. **Per-argument equivariance.** For fixed k, test whether a single linear map M_k on
   z-space satisfies "z(ka) at the a-slot ⇒ output shifts from ab to kab" across all a,
   and whether M_k M_{k'} ≈ M_{kk'} (a group action on the latent space; cf. Wu et al.).
6. **Resolution/bandwidth sweep (revised after Section 5.1).** The K sweep showed
   that ranking resolution does not require higher spins (the fundamental character
   is monotone in distance), so bandwidth selection must be tested with an objective
   whose optimum genuinely contains higher spins — e.g. regressing the heat-kernel
   density value K_t(ab c⁻¹) itself, or classification against *structured* (not
   Haar) negative distributions whose Bayes scorer is non-monotone. Sweep t and
   measure which characters the fitted score uses.
7. **Pool-size sweep with exact finite subgroups.** N ∈ {120 (= 2I, exactly closed),
   512, 2048, 8192} at fixed pair count; connects the finite-group regime of
   `RESULTS.md` to the continuous regime, and tests whether the critical pair count is
   set by spectral content rather than pool size.
8. **Delayed-generalization (grokking) search.** *In progress (queue item 2:
   f ∈ {0.001, 0.0005, 0.0002}).* Remaining: inflated initialization scale
   (Omnigrok, arXiv:2210.01117) and rougher encoding kernels if the f-sweep alone
   does not produce a delayed transition.
9. **Encoding robustness.** Replicate the pilot with an independent observation map
   (e.g., the group acting on a generic function on S³, with an even function as the
   SO(3) control) to confirm the results are not artifacts of the landmark encoding.
   Related: explain finding (d) (why the quotient task is harder).
10. **Topology of the learned representation.** Persistent homology of {z(g)} against
    the Betti profile of S³ (β = 1,0,0,1) with permutation nulls and geodesic
    distances; the projective-coordinates diagnostic (presence of a ℤ/2 class in H¹)
    separates S³ from RP³ = SO(3).
11. **Quadratic-activation variant.** With x ↦ x² activations the interaction layer
    can implement irrep tensor products exactly (Clebsch–Gordan); an analytic
    weight construction exists in principle (cf. Gromov, arXiv:2301.02679). Compare
    the trained solution against the construction.
12. **Transformer variant.** Re-run the pilot with the repo's 1-layer transformer
    (`core/model.py`) consuming Ψ-projected inputs, for architectural continuity with
    `RESULTS.md` and arXiv:2607.07066.
13. **Theory.** Extend the max-margin analysis of Morwani et al. (arXiv:2311.07568)
    and/or the Fourier-recovery theorem of Marchetti et al. (arXiv:2312.08550) from
    finite groups to compact groups (Haar-integral versions); the pilot's spectra are
    the empirical target.

**Big picture.** The through-line of the repo: networks trained on algebraic operations
represent elements via irreducible representations and compute via them — established
for monoids (arXiv:2607.07066) and finite groups (`RESULTS.md`), now extended to a
compact Lie group, where the representation theory is infinite and the model must
truncate it. The distinctive contributions available here, in order of ambition:
(1) the double cover — a global-topology property, invisible at the Lie-algebra level,
read out of a learned representation; (2) bandwidth selection — which irreps a network
uses as a function of task precision, with an analytic (heat-kernel) prediction;
(3) recovery of the group manifold S³ itself from the trained network. All three should
ultimately rest on causal evidence (Tasks 2–5), not spectra alone.

## Appendix: repository pointers

- `core/su2.py` — quaternion/SU(2) utilities, exact Wigner D^j, FVE machinery,
  encodings (self-tests: `uv run python -m core.su2`).
- `core/su2_model.py` — the encoder-factorized scorer.
- `experiments/su2_train.py` — training driver (all evaluation tiers + spectroscopy
  logged to `log.jsonl`; `--resume` supported; checkpoints include optimizer state).
- `experiments/run_su2_pilot.sh` — the four-run pilot (launch detached with `setsid`).
- `data/su2/<run>/` — logs and checkpoints (not in git).
- `RESULTS_SU2.md` — public-facing pilot write-up; `RESULTS.md` — finite-group results.
- `ideas/` (not in git) — literature review (`continuous_groups_review.md`), design
  decisions (`grilling-08-03.md`), earlier brainstorm (`gpt_notes.txt`).
