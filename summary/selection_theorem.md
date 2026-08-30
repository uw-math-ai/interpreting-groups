# Toward a selection theorem for learned group representations (plan G2)

Companion to `summary/theory_notes.md`; states and proves what is provable now,
and isolates the remaining conjecture. Setting: G compact with Haar measure μ;
Z ⊆ Z(G) the quotiented central subgroup; the composition-ranking task of the
experiments (positive (ab)·exp(ξ), Haar negatives excluded from a ball around
the positive class); the factorized architecture s(a,b,c) = ⟨h(z(a), z(b)),
z_c(c)⟩ with weight decay.

## Lemma 1 (faithful-character ranking)

*Let V = ⊕ᵢ Vᵢ be a faithful finite-dimensional unitary representation of the
compact group H, with character χ_V. Then φ(g) := Re χ_V(g) satisfies
φ(g) ≤ φ(e) for all g, with equality iff g = e; φ is a continuous class
function; and φ(e) − φ(exp X) = ½ Q_V(X) + O(‖X‖⁴), where Q_V(X) =
Σᵢ ‖dVᵢ(X)‖²_F is a positive-definite quadratic form on the Lie algebra.*

Proof. Each Vᵢ(g) is unitary, so its eigenvalues lie on the unit circle and
Re tr Vᵢ(g) ≤ dᵢ, with equality iff every eigenvalue equals 1, i.e. iff
Vᵢ(g) = I. Summing, φ(g) ≤ φ(e) with equality iff g ∈ ∩ᵢ ker Vᵢ, which is
{e} by faithfulness. Continuity and conjugation-invariance are immediate. The
expansion follows from Vᵢ(exp X) = exp dVᵢ(X) and tr exp A = dᵢ + tr A +
½ tr A² + …, with tr dVᵢ(X) = 0 (compactness) and tr dVᵢ(X)² =
−‖dVᵢ(X)‖²_F (anti-Hermitian). Positive definiteness of Q_V: dVᵢ(X) = 0 for
all i implies X generates a one-parameter subgroup in ∩ ker Vᵢ = {e}, so
X = 0. ∎

Consequence: g ↦ φ(e) − φ(g) is a squared-distance surrogate (locally
equivalent to the bi-invariant metric), so a scorer of the form
f(Re χ_V(ab c⁻¹)) with f monotone ranks the exact product above every
excluded negative, for **any** faithful V. This is the observed scorer family
(score–character fits of 0.97–0.99 for SU(2), 0.70 linear for SU(3)), and it
frees optimization to choose V on cost alone.

## Lemma 2 (faithfulness is necessary)

*If every activation of the model factors through the matrix coefficients of a
representation V with nontrivial kernel N ⊆ H, then the InfoNCE loss is
bounded away from its optimum.*

Proof sketch. For n ∈ N, all V-coefficients of c and cn agree, so
s(a, b, cn) = s(a, b, c) identically. Take n with d(e, n) larger than the
exclusion radius (exists whenever N ⊄ the exclusion ball, which holds for
every nontrivial closed N once the radius is below the diameter of N): then
ab·n is an admissible negative receiving the same score as the positive
region around ab, forcing a tie on a set of positive measure. ∎

The causal suites verify the contrapositive block by block: ablating a block
erases exactly the discriminations (center flips) its central character sees.

## Observation 3 (which realization, hence which cost)

The score requires χ_V(ab c⁻¹) = Σᵢ ⟨Vᵢ(ab), Vᵢ(c)⟩_HS: a Hilbert–Schmidt
pairing of **matrix blocks**, not of single orbit vectors. A vector
realization z(g) = V(g)v₀ (cost Σ dᵢ) does not suffice to form the pairing;
the full coefficient block (realified dimension rᵢ = dᵢ² for real and
quaternionic type, 2dᵢ² for complex type) does. The A1 measurements confirm
the models carry the matrix realization: the active block dimension is
exactly 4 = d² for SU(2) (spin-½, quaternionic) and 9 = d² for SO(3)
(spin-1, real) in every seed. The canonical carrier cost is therefore

    r(V) = Σᵢ rᵢ  (realified matrix-coefficient count),

and the empirical selection principle reads: **the learned carrier is
argmin r(V) over faithful representations V of the task group.** The
carrier-cost search (`experiments/carrier_search.py`) shows this argmin
coincides with the minimal-embedding-dimension argmin on all catalogued
group forms, so the data cannot contradict the coarser statement either;
but r(V) is the functional the architecture motivates.

## Conjecture 4 (selection theorem)

*In the factorized architecture with weight decay, among parameter settings
achieving population InfoNCE loss within ε of the optimum, the minimum-norm
setting has encoder activations whose variance concentrates (as ε → 0) on
the matrix coefficients of an argmin-r(V) faithful representation of the task
group, and the score converges to a monotone function of Re χ_V(ab c⁻¹).*

What is missing for a proof: ruling out non-representation-theoretic
ε-optimal solutions of smaller norm (the compact-group analogue of the
coset-circuit alternatives documented for finite groups), and a lower bound
tying the norm of any solution to the effective dimension of the functions it
computes on G (a reproducing-kernel argument on L²(G) is the natural route:
any score component orthogonal to the carrier's isotypic subspaces
contributes variance that either helps ranking — forcing it into some
faithful carrier's span — or is pure cost). The empirical inputs constrain
the conjecture tightly: selection is deterministic across seeds in all twelve
conditions, and the rejected carriers sit at the measurement floor.

## Remark 5 (regression regime)

For the heat-kernel objective the optimum is a fixed element of L²(G) with
Peter–Weyl coefficients (2j+1)e^{−t j(j+1)}. If the architecture's neural
tangent kernel on Haar-distributed inputs is bi-invariant, it diagonalizes in
the Peter–Weyl basis with eigenvalues λ_ρ decreasing in Casimir (the
compact-group transposition of standard dot-product-kernel spectral
asymptotics), and gradient descent fits sectors at rates ∝ λ_ρ — the
bottom-up Casimir-order filling observed at every temperature, with the
observed truncation as the finite-time consequence. Making the bi-invariance
step rigorous for the landmark encoding (which is only approximately
invariant in distribution) is the remaining gap.
