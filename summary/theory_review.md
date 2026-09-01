# Independent theory review: representation-theoretic statements

**Reviewer:** independent verification agent (machine review).
**Date:** 2026-09-01.
**Scope:** the mathematical statements in `summary/selection_theorem.md` and the
corresponding lemmas, background paragraphs, and proofs in `paper/main.tex`
(main-text Lemmas at lines 221–229 and 305–315; appendix "Background" lines
589–766; appendix "Proofs" lines 768–831). This review was produced with no
access to the authors' derivations beyond the reviewed text; every claim was
re-derived from first principles and checked against the standard treatments
(Bröcker–tom Dieck, *Representations of Compact Lie Groups*, Ch. II–III;
Fulton–Harris, *Representation Theory*; Sepanski, *Compact Lie Groups*;
Folland, *A Course in Abstract Harmonic Analysis*). The file's own
justifications were deliberately not trusted; where a proof is judged valid it
is because the reviewer's independent derivation coincides with it.

**Verdict summary**

| # | Item | Verdict |
|---|------|---------|
| 1 | Two-sided Peter–Weyl block, complex form, action conventions | CORRECT |
| 2 | Real form: dimension formula, W\*⊗_D W, minimality | CORRECT-WITH-CAVEAT (statements true; the written minimality proof is invalid for D = ℂ, ℍ and must be restated) |
| 3 | Faithful-character facts (bound, expansion, imaginary trace) | CORRECT |
| 4 | Kernel-collision Proposition 2 | CORRECT as scoped in the summary; the companion paper still carries the retracted stronger lemma, which its proof does not establish |
| 5 | Descent rule and the faithful-irrep/center Fact | CORRECT |

---

## Item 1. Two-sided Peter–Weyl block (complex form) — CORRECT

**Claim reviewed.** For an irreducible unitary ρ: G → U(V) of a compact group,
M_ρ = span{g ↦ ⟨u, ρ(g)v⟩} ⊆ L²(G) is isomorphic to V\* ⊠ V as a G_L × G_R
representation under left and right translation, is G × G-irreducible, and
dim_ℂ M_ρ = d_ρ² (`selection_theorem.md` Lemma 3, complex form;
`paper/main.tex` lem:twosided and its proof at lines 817–825).

**Derivation.** Fix the bra–ket convention ⟨·,·⟩ conjugate-linear in the first
slot (the convention the paper uses implicitly, since it writes coefficients as
⟨u, ρ(g)v⟩ with u the bra), and the standard translation actions

  (L_k f)(g) = f(k⁻¹g),  (R_k f)(g) = f(gk).

Both are left actions of G (the inverse in L is what makes L_{k₁}L_{k₂} =
L_{k₁k₂}; R needs no inverse), and they commute, giving an action of G × G by
((k₁,k₂)·f)(g) = f(k₁⁻¹ g k₂). Writing f_{u,v}(g) = ⟨u, ρ(g)v⟩ and using
unitarity ρ(k⁻¹) = ρ(k)\*:

  (L_k f_{u,v})(g) = ⟨u, ρ(k⁻¹)ρ(g)v⟩ = ⟨ρ(k)u, ρ(g)v⟩ = f_{ρ(k)u, v}(g),
  (R_k f_{u,v})(g) = ⟨u, ρ(g)ρ(k)v⟩ = f_{u, ρ(k)v}(g).

So left translation moves the u-slot and right translation the v-slot. The
u-slot is conjugate-linear in u; under the conjugate-linear identification
u ↦ u\* := ⟨u,·⟩ of V with V\*, the action u ↦ ρ(k)u transports to
(ρ(k)u)\* = u\* ∘ ρ(k)⁻¹ = ρ\*(k)u\*, the contragredient action on V\*. Hence
the map V\* ⊗ V → L²(G), u\* ⊗ v ↦ f_{u,v}, is complex-linear in both slots
and intertwines L ⊗ R with ρ\* ⊠ ρ. **Left acts on V\*, right acts on V —
exactly the assignment written in both files. The convention is confirmed.**

A convention-independent cross-check in matrix entries ρ_{ij}(g) = ⟨e_i,
ρ(g)e_j⟩: L_k ρ_{ij} = Σ_m ρ(k⁻¹)_{im} ρ_{mj} mixes the row index within a
fixed column, and the coordinates of a fixed-column span transform by
ρ(k⁻¹)ᵀ = ρ̄(k) ≅ ρ\*(k); R_k ρ_{ij} = Σ_m ρ_{im} ρ(k)_{mj} mixes columns,
and a fixed-row span transforms by ρ(k). This also confirms the experimental
facts the paper leans on: a fixed set of columns of ρ(·) spans a
left-invariant subspace but is mixed by right translation, which is why the
two-column SO(3) frame passes the left test exactly and fails linear right
closure.

**Injectivity and dimension.** Every element of the image is g ↦ tr(ρ(g)A)
for A ∈ End(V) ≅ V ⊗ V\*; by Schur orthogonality ⟨ρ_{ij}, ρ_{kl}⟩ =
δ_{ik}δ_{jl}/d_ρ, the d_ρ² entry functions are orthogonal and nonzero, hence
linearly independent, so tr(ρ(·)A) ≡ 0 forces A = 0. The map V\* ⊗ V → M_ρ is
therefore a linear isomorphism and dim_ℂ M_ρ = d_ρ². **Correct.**

**Irreducibility.** For compact G × G with product Haar measure,
∫|χ_{V\*⊠V}(g,h)|² dg dh = (∫|χ_{V\*}|²)(∫|χ_V|²) = 1, so V\* ⊠ V is
irreducible; this is the standard fact that the outer tensor product of
complex irreducibles is irreducible (valid over ℂ; see Item 2 for why this
step does *not* transfer to ℝ). A subspace stable under both one-sided
translations is G × G-invariant, so any nonzero such subspace equals M_ρ.
**Correct.**

**Caveats (non-verdict-affecting).** (i) Neither file writes the formulas for
L_k and R_k where the lemma is stated; the assignment "left ↔ V\*, right ↔ V"
is tied to the pair of conventions above (with the opposite sesquilinearity
convention, or with L_k f = f(kg), the factors swap), so one sentence fixing
(L_k f)(g) = f(k⁻¹g), (R_k f)(g) = f(gk) should be added. (ii) The
experiments fit the operator family z(g) ↦ z(kg), which as a k-indexed family
is k ↦ L_{k⁻¹}, an anti-homomorphism; the *set* of operators coincides with
{L_k}, so every invariance and closure statement transfers verbatim. No error.

## Item 2. Real form — CORRECT-WITH-CAVEAT

**Claims reviewed.** For an irreducible real representation π: G → O(W) with
commutant D = End_G(W) ∈ {ℝ, ℂ, ℍ} (real Schur lemma): (a) the real
matrix-coefficient span M_W is identified with W\* ⊗_D W; (b) dim_ℝ M_W =
(dim_ℝ W)²/dim_ℝ D; (c) M_W is the minimal nonzero subspace of its isotypic
component stable under both translation actions. (`selection_theorem.md`
Lemma 3, real form; `paper/main.tex` lines 650–674 and the proof at lines
825–831.)

**(b) Dimension formula — verified independently, two ways.**

*Via the double commutant.* Every real coefficient ⟨u, π(g)v⟩ equals
tr(π(g)A) for the rank-one A = v⟨u,·⟩, and these A span End_ℝ(W), so M_W =
{tr(π(·)A) : A ∈ End_ℝ(W)}. The kernel of A ↦ tr(π(·)A) is the trace-form
orthocomplement of E := span_ℝ π(G). Since W is a simple module over E and
D = End_E(W), the Jacobson density theorem (finite dimension) gives E =
End_D(W); with m := dim_D W (so dim_ℝ W = m·dim_ℝ D), End_D(W) is a full
m × m matrix algebra over the division ring D^op, of real dimension
m² dim_ℝ D. The trace form is nondegenerate on E (E is closed under transpose
because π(g)ᵀ = π(g⁻¹), and tr(AAᵀ) > 0 for A ≠ 0), so

  dim_ℝ M_W = dim_ℝ E = m² dim_ℝ D = (dim_ℝ W)² / dim_ℝ D.

*Via complexification, per type.* Real type (complex irrep V of dim d with
invariant real structure): W is the real form, dim_ℝ W = d, D = ℝ, block
d²/1 = d². Complex type (V ≇ V̄ of dim d): W = V_ℝ, dim_ℝ W = 2d, D = ℂ; the
real-valued functions in M_V ⊕ M_{V̄} are {f + f̄ : f ∈ M_V}, real dimension
2d² = (2d)²/2. Quaternionic type (V of dim d with invariant quaternionic
structure): W = V_ℝ, dim_ℝ W = 2d, D = ℍ; M_V is conjugation-stable and its
real points have real dimension d² = (2d)²/4. Both routes agree, and agree
with the realification table as stated in `paper/main.tex` lines 659–671
(real d → d², complex d → 2d², quaternionic d → d²). **Formula correct.**

*The three instances.* SO(3) vector: d = 3 real type, 3² = 9. ✓
SU(3) fundamental pair **3** ⊕ **3̄**: d = 3 complex type, W = ℝ⁶, 36/2 = 18. ✓
SU(2) spin-½: d = 2 quaternionic, W = ℝ⁴, 16/4 = 4. ✓ The SU(2) case was also
checked concretely: π(g) is left multiplication by a unit quaternion q on
ℍ ≅ ℝ⁴, span_ℝ π(G) is the algebra of left multiplications ≅ ℍ (real dim 4 =
16/4), and M_W is spanned by the four coordinate functions q₀, q₁, q₂, q₃.
**All three instances correct.**

**(a) The identification W\* ⊗_D W — correct once sidedness is specified.**
For noncommutative D the symbol ⊗_D requires a right module on the left: take
W a left D-module (D acting as operators) and make W\* a right D-module by
(φ·T)(w) = φ(Tw). Then the balanced product is defined, the coefficient map
φ ⊗ v ↦ φ(π(·)v) respects the balancing relation because f_{φ·T, v} =
φ(Tπ(·)v) = φ(π(·)Tv) = f_{φ, Tv} (T commutes with π(G)), it is G × G-
equivariant (π(k) commutes with D), surjective by definition, and injective by
the dimension count above: D^m ⊗_D D^m ≅ D^{m²} has real dimension
m² dim_ℝ D = dim_ℝ M_W. The paper's bare "W\* ⊗_D W" should state this
right-module structure, but the identification itself is correct.

**(c) Minimality — the statement is TRUE, the written proof is NOT valid.**
The paper's proof (lines 829–830) reads: "Minimality follows from
irreducibility of the two-sided action on this space, by the same
outer-tensor argument applied over D." Two independent objections:

1. The outer-tensor principle invoked in the complex case — "an outer tensor
   product of irreducibles is irreducible" — is *false* over ℝ. The general
   criterion is that V ⊠ W is irreducible iff End(V) ⊗_k End(W) is a division
   algebra; here both commutants are D, and ℂ ⊗_ℝ ℂ ≅ ℂ ⊕ ℂ and ℍ ⊗_ℝ ℍ ≅
   M₄(ℝ) are not division algebras. Concrete counterexample: for the 2-dim
   rotation representation W of SO(2) at frequency n (commutant ℂ), W ⊗_ℝ W is
   a 4-dim representation of SO(2) × SO(2) that splits into the two 2-dim
   pieces spanned by cos n(θ∓φ), sin n(θ∓φ); the actual coefficient block
   span{cos nθ, sin nθ} has dimension 2 = 2²/2, not 4. So "the same argument"
   does not run.
2. Over D = ℍ the phrase is not even well-formed: an outer tensor product
   "over D" of two left D-modules does not exist for noncommutative D, and
   W\* ⊗_D W (the balanced product that does exist) carries no D-module
   structure on which a Schur-type argument could be run "over D".

*Correct proof (the fix).* Transport the two-sided action to E = End_D(W)
through the trace identification A ↦ tr(Aπ(·)) of (b). One computes
L_k f_A = f_{Aπ(k)⁻¹} and R_k f_A = f_{π(k)A}, so M_W is isomorphic as a
two-sided translation module to E acting on itself by left and right
multiplication by π(G). Since span_ℝ π(G) = E (Jacobson density, as above), a
subspace of E invariant under both translations is exactly a two-sided ideal
of E; and E ≅ M_m(D^op) is a simple algebra (matrix algebra over a division
ring), so it has no proper nonzero two-sided ideal. Hence M_W is G × G-
irreducible, and in particular minimal. This is the double-commutant /
Wedderburn argument, uniform in D. (An alternative correct route: complexify.
Real type: (M_W)_ℂ ≅ V\* ⊠ V irreducible ⇒ M_W irreducible. Quaternionic:
same. Complex type: (M_W)_ℂ ≅ (V\* ⊠ V) ⊕ (V̄\* ⊠ V̄), two non-isomorphic
conjugate irreducibles, and a proper invariant real subspace would complexify
to a conjugation-stable proper invariant subspace, impossible.)

One further check: the "isotypic component" language is consistent — in
L²(G, ℝ) under one-sided translation the W-isotypic component *is* M_W
(multiplicity dim_D W, the standard real regular-representation count), so
"minimal nonzero two-sided-stable subspace of its isotypic component" is
true and in fact "unique nonzero" holds.

**Required correction (C1).** In `paper/main.tex`, proof of lem:twosided,
replace the final sentence with the ideal-theoretic argument, e.g.:
"Minimality follows because M_W is irreducible under G × G: by the Jacobson
density theorem span_ℝ π(G) = End_D(W), the trace pairing identifies M_W with
End_D(W) carrying the two-sided multiplication action, a subspace invariant
under both translations is then a two-sided ideal, and a matrix algebra over
a division ring is simple." Optionally state the right D-module structure on
W\* used in W\* ⊗_D W. The same one-line fix applies wherever
`selection_theorem.md` Lemma 3's real form is proved rather than cited.

## Item 3. Faithful-character facts — CORRECT

**(i) Re χ_V(g) ≤ χ_V(e) with equality iff g = e for faithful V.** Each
V_i(g) is unitary, hence normal with unimodular eigenvalues e^{iθ_k};
Re tr V_i(g) = Σ cos θ_k ≤ d_i with equality iff all eigenvalues are 1, which
for a diagonalizable matrix means V_i(g) = I. Summing over summands, equality
in Re χ_V(g) ≤ χ_V(e) holds iff g ∈ ∩_i ker V_i = ker V = {e} by
faithfulness. The proofs in both files are exactly this and are **correct**
(the diagonalizability step, implicit in "every eigenvalue equals 1 ⇒
identity", is supplied by unitarity).

**(ii) Local expansion.** V(exp X) = exp dV(X) with A = dV(X) anti-Hermitian,
eigenvalues iλ_k, λ_k ∈ ℝ. Then Re χ_V(exp X) = Σ_k cos λ_k(X) and

  χ_V(e) − Re χ_V(exp X) = Σ_k (1 − cos λ_k) = ½ Σ_k λ_k² − Σ_k O(λ_k⁴)
   = ½ ‖dV(X)‖_F² + O(‖X‖⁴),

using Σλ_k² = tr(A\*A) = ‖A‖_F² and 0 ≤ λ²/2 − (1 − cos λ) ≤ λ⁴/24, which
gives a remainder bounded by ‖dV(X)‖_F⁴/24 ≤ C‖X‖⁴ — the O(‖X‖⁴) is uniform,
slightly stronger than the pointwise evenness argument, and confirms the
stated form. The odd orders vanish because t ↦ Σ cos(tλ_k) is even, as the
summary notes. **Correct.**

**(iii) The trace term.** tr A for anti-Hermitian A is purely imaginary
(eigenvalues iλ_k), so it contributes nothing to the real part — this is all
either proof uses, and it is valid for *every* compact group; neither proof
smuggles in semisimplicity. The summary's parenthetical aside is also
accurate as stated: X ↦ tr dV(X) is a Lie-algebra homomorphism 𝔤 → iℝ into an
abelian algebra, so it kills [𝔤,𝔤] and vanishes identically when 𝔤 is
semisimple (𝔤 = [𝔤,𝔤]), which covers every group used in the paper, while for
U(1) with V = e^{iθ} one has tr dV(X) = iX ≠ 0, so the vanishing is genuinely
not general. The aside is clearly marked as not needed by the proof.
**Correct, and correctly scoped.**

**(iv) Positive definiteness.** Q_V(X) = Σ_i ‖dV_i(X)‖_F² is a quadratic form
(dV linear in X); Q_V(X) = 0 ⇒ dV(X) = 0 ⇒ exp(tX) ∈ ker V = {e} for all t ⇒
X = 0 since exp is a local diffeomorphism at 0. Note the kernel of Q_V is in
general the Lie algebra of ker V, so faithfulness of the *sum* V (not of each
V_i) is the right hypothesis, and it is the one used. **Correct.**

## Item 4. Kernel-collision Proposition 2 — CORRECT as scoped (with a demanded consistency fix in the paper)

**Statement and proof.** If s(a,b,c) = S(φ(a), φ(b), φ(c)) with φ factoring
through the matrix coefficients of V and N = ker V ≠ {e} (closed, and
automatically normal), then every coefficient satisfies ⟨u, V(gn)v⟩ =
⟨u, V(g)V(n)v⟩ = ⟨u, V(g)v⟩ for n ∈ N, so φ is constant on N-cosets, ab·n
lies in the coset (ab)N, and s(a,b,ab·n) = s(a,b,ab). The one-line proof is
complete and needs no continuity, sampling, or metric hypotheses.
**Correct.**

**Scoping.** Within `selection_theorem.md` the proposition claims exactly
non-identifiability within a kernel coset plus exact score ties (which the
tie-splitting measurement convention converts to chance-value
discriminations), and the following paragraph explicitly *disclaims* the
uniform population-loss gap, itemizing what a quantitative version would
need (modulus of continuity, positive density near kernel translates,
quantified comparison with the noisy positive). No stronger claim leaks into
the summary document. **Correct as scoped.**

**However — required correction (C2).** The companion paper has not been
brought in line with this revision. `paper/main.tex` lines 787–810 still
state lem:necessity ("the population ranking loss is bounded away from its
optimum"), referenced in the main text at lines 233–236, and the reviewer's
independent reading confirms the summary's own critique of it:

1. The hypothesis "exclusion radius < min_{n ∈ N, n ≠ e} d(e,n)" is
   unsatisfiable whenever N is infinite (a positive-dimensional closed
   subgroup accumulates at e; the infimum is 0 and the minimum does not
   exist), making the lemma vacuous precisely in the positive-dimensional-
   kernel cases.
2. The proof's step "with positive probability a sampled negative ties or
   exceeds the positive" is not established: exact ties at ab·n occur with
   probability zero under a continuous negative distribution, the argument
   only shows negatives come within an arbitrarily small δ plus the
   (unquantified) modulus over the positive displacement, and whether a
   near-tie forces excess loss depends on the loss functional, which the
   lemma never fixes (a 0–1 ranking loss is insensitive to near-ties; a
   softmax/InfoNCE loss is not).
3. "Bounded away from its optimum" reads as a uniform gap, which pointwise
   continuity of a single scorer cannot deliver over a scorer class.

The fix is the one the summary itself adopts: replace lem:necessity by the
exact-collision proposition (endpoint B) and downgrade the main-text sentence
at lines 233–236 accordingly, or else restrict to finite N, fix a
tie-sensitive loss, and add the quantified noisy-positive comparison as
explicit hypotheses. As it stands the paper asserts a lemma its proof does
not establish, and the two documents contradict each other.

## Item 5. Descent rule and the center Fact — CORRECT

**Descent rule.** By Schur's lemma a central z acts in an irrep ρ as the
scalar ω_ρ(z) ∈ U(1). Pullback along G → G/Z is a bijection between irreps
of G/Z and irreps of G with Z ⊆ ker ρ, and Z ⊆ ker ρ iff ω_ρ|_Z = 1. For the
spectral-support claim: right translation by z ∈ Z acts on the block M_ρ as
the scalar ω_ρ(z) (R_z f_{u,v} = f_{u, ρ(z)v} = ω_ρ(z) f_{u,v}), so a
Z-invariant function has zero component in every block with ω_ρ|_Z
nontrivial. Both statements as written at `paper/main.tex` lines 197–200 and
692–701 are **correct**.

**The Fact (lines 705–718).** If ρ is a faithful irrep, Schur gives
ρ(z) = ω_ρ(z)I on Z(G), faithfulness makes z ↦ ω_ρ(z) injective, its image
is a closed (compactness) subgroup of U(1), so Z(G) embeds as a closed
subgroup of U(1). Closed subgroups of U(1) are finite cyclic or U(1) itself;
ℤ₂ × ℤ₂ is finite and not cyclic; and Z(Spin(4)) = Z(SU(2) × SU(2)) =
{±1} × {±1} ≅ ℤ₂ × ℤ₂. Hence Spin(4) has no faithful complex irrep, and any
faithful representation must be a direct sum — exactly what the carrier
table needs. The proof as printed is complete and **correct**. Scope note:
"irrep" here means complex irrep, consistent with the paper's declared
shorthand; for completeness the reviewer verified that the real-irreducible
analogue also holds (for a real irrep, an injective Z(G) → D^× is forced,
and the unit groups ℝ^×, ℂ^×, ℍ^× each contain exactly one element of order
2, namely −1, so ℤ₂ × ℤ₂ embeds in none of them), so the conclusion is not
an artifact of working over ℂ — though the paper nowhere claims this and
need not.

**Incidental checks (in passing, all correct).** The central-character data
used alongside the descent rule were re-derived: SU(2) D^j has central
character (−1)^{2j}; SU(3) (p,q) has central character ω^{p−q} at ωI (checked
on 3, 3̄, 6, 8, 10; descent to PSU(3) iff p ≡ q mod 3); Spin(4) irreps
(j_L, j_R) carry ((−1)^{2j_L}, (−1)^{2j_R}) with the three quotients
retaining the stated sublattices; Spin(5) ≅ Sp(2) with quaternionic 4, real
5 and 10, and SO(5) retaining 5 and 10. Schur orthogonality as displayed at
lines 620–626 is the standard normalization.

---

## Consolidated list of required corrections

- **C1 (Item 2, must fix).** `paper/main.tex`, proof of lem:twosided, final
  sentence: the minimality/irreducibility of the real block cannot be proved
  "by the same outer-tensor argument applied over D" — the outer-tensor
  principle fails over ℝ (ℂ ⊗_ℝ ℂ and ℍ ⊗_ℝ ℍ are not division algebras;
  explicit SO(2) counterexample above) and is not defined over noncommutative
  D. Replace with the double-commutant/simple-algebra argument (or the
  complexification case analysis) given in Item 2. The lemma's *statements*
  all stand.
- **C2 (Item 4, must fix).** `paper/main.tex` lem:necessity (lines 787–810)
  and the main-text sentence referencing it (lines 233–236) assert a
  quantitative loss gap that the given proof does not establish and whose
  radius hypothesis is vacuous for infinite kernels; align the paper with
  the revised Proposition 2 of `summary/selection_theorem.md` (exact
  collisions only, quantitative gap stated as open) or add the missing
  hypotheses and a real proof.
- **Minor (Item 1).** State the translation conventions (L_k f)(g) =
  f(k⁻¹g), (R_k f)(g) = f(gk) next to lem:twosided; the V\*/V factor
  assignment is convention-dependent and currently implicit. Optionally note
  that the fitted family z(g) ↦ z(kg) is k ↦ L_{k⁻¹}, with the same operator
  set. Minor (Item 2): specify the right D-module structure on W\* in
  W\* ⊗_D W.

Everything else reviewed — the complex two-sided lemma and its convention,
the dimension formula in all three types and its three instances (9, 18, 4),
the character bound, expansion, imaginary-trace and positive-definiteness
claims, the collision proposition as scoped in the summary, the descent rule,
and the center Fact with its Spin(4) consequence — is confirmed correct as
written.

*Signed: independent verification agent (machine review), 2026-09-01.*
