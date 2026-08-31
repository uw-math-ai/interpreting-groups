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
½ tr A² + …: A = dVᵢ(X) is anti-Hermitian, so tr A is **purely imaginary**
(zero when G is semisimple, as for every group used here, but not for general
compact G — e.g. U(1)) and does not contribute to the real part, while
Re tr A² = −‖A‖²_F. Equivalently Re tr exp(tA) = Σ_k cos(t λ_k) is even
in t. Positive definiteness of Q_V: dVᵢ(X) = 0 for
all i implies X generates a one-parameter subgroup in ∩ ker Vᵢ = {e}, so
X = 0. ∎

Consequence: g ↦ φ(e) − φ(g) is a squared-distance surrogate (locally
equivalent to the bi-invariant metric), so a scorer of the form
f(Re χ_V(ab c⁻¹)) with f monotone ranks the exact product above every
excluded negative, for **any** faithful V. This is the observed scorer family
(score–character fits of 0.97–0.99 for SU(2), 0.70 linear for SU(3)), and it
frees optimization to choose V on cost alone.

## Lemma 2 (faithfulness is necessary — with sampling assumptions)

*Assume: (i) the score factors as s(a,b,c) = S(φ(a), φ(b), φ(c)) with no
bypass path, where φ factors through the matrix coefficients of a
representation V with nontrivial closed kernel N ⊆ H; (ii) the learned score
is continuous in c; (iii) negatives are sampled from a distribution assigning
positive probability to every open subset of the admissible region (Haar
restricted to the excluded-ball complement does); (iv) the exclusion radius is
smaller than min_{n ∈ N, n ≠ e} d(e, n). Then the population ranking loss is
bounded away from the optimum.*

Proof sketch. Pick n ∈ N \ {e}; by (iv), ab·n lies in the admissible negative
region. By (i), s(a, b, ab·n) = s(a, b, ab) ≥ the score of the noisy positive
minus a continuity gap; by (ii) there is a δ-ball around ab·n on which the
score stays within any ε of that value; by (iii) that ball receives positive
probability, so with positive probability a sampled negative ties or beats the
positive. ∎

The assumptions are not decoration: with a *finite* kernel and continuous
candidate sampling, the exact ambiguous points have measure zero, and without
score continuity (or a positive-measure ambiguous set) non-separation alone
implies no positive error bound — the v2 plan's caveat, adopted here. The
causal suites verify the contrapositive block by block: ablating a block
erases exactly the discriminations its central character sees.

The causal suites verify the contrapositive block by block: ablating a block
erases exactly the discriminations (center flips) its central character sees.

## Lemma 3 (two-sided Peter–Weyl block; complex and real forms)

*Complex form. For an irreducible unitary ρ: G → U(V), the coefficient space
M_ρ = span{g ↦ ⟨u, ρ(g)v⟩} is isomorphic to V* ⊠ V as a G_L × G_R
representation; the external tensor product of irreducibles is irreducible for
G × G, so any nonzero complex subspace of M_ρ invariant under BOTH left and
right translations equals M_ρ, and dim_C M_ρ = d_ρ².*

*Real form. For an irreducible real representation W with commutant
D = End_G(W) ∈ {R, C, H} (real Schur lemma), the real coefficient block has
dimension dim_R M_W = (dim_R W)² / dim_R D, and it is the minimal nonzero
two-sided translation-stable real subspace of its sector.*

Instances (matching every measured block dimension): SO(3) vector — 9/1 = 9;
SU(2) spin-½ — 16/4 = 4 (quaternionic); SU(3) fundamental pair — 36/2 = 18
(complex type). This lemma gives d² its precise role: **not** the minimum
coordinate count of every faithful encoding — smaller one-sided frames exist
(SO(3): two columns, 6 dims; SU(3): two columns, 12) — but the dimension of
the full two-sided-stable block. The P1A experiment realizes the
lemma/experiment pair: trained codes support commuting left/right linear
actions at the block dimension (commutator 0.010–0.012 across all seeds),
faithful frames pass left exactly and fail linear right closure (right R²
0.638, defect 0.53 for SO(3)), and matched random truncations of the true
block fail both.

## Observation 4 (which realization; the cost question is now end-to-end)

The observed scorer family s ≈ f(Re χ_V(ab c⁻¹)) uses Hilbert–Schmidt pairings
of matrix blocks; a one-sided orbit/frame realization cannot form the pairing
linearly and needs a nonlinear reconstruction of the missing columns for right
translation (SO(3): g e₃ = (g e₁) × (g e₂)). The v2 correction adopted here:
this makes the carrier question an **end-to-end cost problem** — storage
(frames are smaller) against composition/scoring complexity (blocks are
two-sided linear with bilinear composition) — and no canonical cost functional
should be fixed before the matched frame-vs-block training experiment (P1B)
measures both mechanisms' total regularized cost. The carrier-cost search
(`experiments/carrier_search.py`) shows that among *block* realizations the
linear and quadratic storage costs never disagree on catalogued group forms;
P1B extends the comparison to non-block feature systems.

## Conjecture 5 (selection theorem)

*In the factorized architecture with weight decay, among parameter settings
achieving population InfoNCE loss within ε of the optimum, the minimum-norm
setting realizes a separating equivariant feature system of minimal
end-to-end regularized cost (storage + composition + scoring), and — the
empirical content of P1A/P1B — that minimum is attained by full
Schur-corrected coefficient blocks of a minimal faithful carrier, whose
two-sided linear closure makes composition bilinear and scoring a
Hilbert–Schmidt pairing.*

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

## Remark 6 (regression regime)

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
