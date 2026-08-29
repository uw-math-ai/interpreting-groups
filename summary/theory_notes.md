# Theory notes: selection principles for learned group representations

Working notes toward plan items G1–G5, kept separate from the results report
(`summary/report.md`). Empirical inputs: the eight group-form conditions
(SU(2), SO(3), SU(3), PSU(3), Spin(4), SO(4), SO(3)×SU(2), PSO(4)), the
heat-kernel bandwidth experiments, and the fusion analysis. Last updated
2026-08-29.

## 1. The observed selection principle

Across all eight conditions, the argument encoder concentrates its variance on
a **faithful representation of the task group G/Z of minimal total dimension**,
reducible when no faithful irrep exists or when a reducible sum is smaller than
the smallest faithful irrep. The two Spin(4)-quotient discovery conditions are
the load-bearing cases: SO(3)×SU(2) selects (1,0)⊕(0,½) (5 dims) over the
faithful irrep (1,½) (6 dims); PSO(4) selects (1,0)⊕(0,1) (6) over (1,1) (9).

## 2. Why faithfulness is necessary and almost sufficient

*Necessary:* the ranking task separates candidates in G/Z; a non-faithful
carrier cannot distinguish elements that differ by its kernel, and the causal
suites confirm each block is necessary for exactly the discriminations its
central character sees.

*Almost sufficient (the faithful-character lemma):* for any faithful
finite-dimensional representation V of a compact group, Re χ_V(g) attains its
maximum χ_V(e) only at g = e, and g ↦ χ_V(e) − Re χ_V(g) is a continuous
nonnegative class function vanishing only at the identity — a
distance-surrogate. A scorer of the form s(a,b,c) = f(Re χ_V(ab c⁻¹)) with f
monotone therefore ranks correctly whenever the candidate distribution is
class-symmetric and the noise density is monotone in the surrogate. This
matches the observed scorers: χ_{1/2} explains 97–99% of the SU(2) score
variance; Re χ_{(1,0)} (which equals (6 − ||I − g||²_F)/2, the chordal
statistic) explains 70% of the SU(3) score linearly with no gain from other
sectors. So *any* faithful carrier suffices for ranking, and optimization is
free to select on cost.

## 3. The cost model, and what current data cannot decide

Under weight decay, the parameter cost of carrying an isotypic block scales
with its size. Two natural cost functionals:

- **Σᵢ dᵢ** (activation width used by the block: realified coefficient count
  per element vector);
- **Σᵢ dᵢ²** (matrix-coefficient count: rows of the effective embedding).

Both are minimized by the same carrier in every condition run so far
(5 < 6 and 13 < 36 for SO(3)×SU(2); 6 < 9 and 18 < 81 for PSO(4)), so the
experiments to date **cannot distinguish the two cost models**. A
discriminating experiment needs a group form whose minimal-Σd and minimal-Σd²
faithful carriers differ, i.e. faithful options {a ⊕ b} vs {c} with
a + b > c but a² + b² < c² (e.g. dimensions 5 ⊕ 5 vs 8). Candidate
constructions: products of higher-rank groups and their diagonal-center
quotients, where fundamental-pair sums compete with a single mixed-weight
irrep; enumerating (G, Z) pairs with this property is the next theory-driven
design task. Until then the principle should be stated as "minimal-dimension"
with the cost functional unresolved.

## 4. Two regimes, one bias

The ranking objective needs only the distance surrogate, so the minimal
faithful carrier is terminal: larger K (denser negatives) *concentrates* the
spectrum further (Section 5.1 of the report). The regression objective
(heat-kernel values) needs genuine bandwidth, and there the model adds sectors
**bottom-up in Casimir order**, truncating when optimization stalls rather
than spreading. A single description covers both: optimization fills spectral
capacity in order of (roughly) Casimir eigenvalue, and stops as soon as the
objective permits — immediately after the minimal faithful carrier for
ranking; at the optimization/capacity frontier for regression. This is the
compact-group form of the low-frequency spectral bias documented for
finite-group and modular tasks.

## 5. Observability (G5), stated exactly

The orbit-averaged observation map realizes precisely the quotient: averaging
Ψ over a central subgroup Z makes the observations Z-invariant and nothing
more (verified numerically: invariance under the quotiented subgroup to
1e-4; discrimination of non-quotiented center elements bounded away from 0).
Every function of the observations is then a function on G/Z, so isotypic
content in non-descending sectors is forbidden — the measured floors
(≤ 0.2–0.6%) are the empirical confirmation. The non-trivial phenomenon is
never the *absence* of forbidden sectors (that is the null constraint of plan
G3) but the *selection among allowed ones* — and, in the so4 condition, the
converse finding: observations that do distinguish single-factor signs still
produce no single-factor content, because the task's positive class does not
require it. Task symmetry, not observation symmetry, sets the carrier.

## 6. Open lemmas (ordered by expected difficulty)

1. **Faithful-character ranking lemma** (formalize Section 2; elementary).
2. **Carrier cost lower bound:** in the factorized architecture with weight
   decay, realizing score ≈ f(Re χ_V(ab c⁻¹)) at fixed margin requires
   parameter norm bounded below by an increasing function of the carrier size;
   identify whether the binding term scales with Σd or Σd². (This decides
   Section 3 theoretically; the discriminating experiment decides it
   empirically.)
3. **Selection theorem:** among all scorers achieving a given InfoNCE loss
   level, the minimum-norm one uses the minimal-cost faithful carrier.
   Ingredients: Peter–Weyl, Schur orthogonality, positive-definite class
   functions, and the rank/nuclear-norm structure of the bilinear score.
4. **Casimir-order filling for regression** (Section 4; plausibly a
   compact-group transposition of existing spectral-bias arguments for
   rotation-invariant kernels: the NTK of the encoder architecture on
   G-invariant data diagonalizes in the Peter–Weyl basis with eigenvalues
   decreasing in Casimir).
