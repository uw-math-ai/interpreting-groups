# Preregistration: controlled polynomial composition on SO(3) (handoff 4.4)

Date: 2026-09-01. Committed before any fitting or training in this
experiment is run. Companion to preregistration/p1b_wd_sweep_2026-09-01.md.
Motivation: G3 removed the causal missing-column reconstruction claim, so
the frame-versus-block cost account needs its structural leg made explicit
in a model class where algebraic degree is visible by construction.

## Model class

Exact standardized SO(3) features (exact Haar standardization, zero mean by
symmetry, scale sqrt(3) per rotation-matrix entry), no generic MLP:

    arm A  bilinear-full    general bilinear map R^9 x R^9 -> R^9
                            (tensor 9*9*9), degree 2 jointly, degree 1 per
                            argument; can represent matrix multiplication
                            exactly.
    arm B  bilinear-frame   general bilinear map R^6 x R^6 -> R^6, the best
                            degree-2 frame interaction. Exact frame
                            composition is impossible in this class: the
                            product frame needs the third column
                            w = u x v of the first argument, which is
                            quadratic in frame(a) alone, so generic frame
                            composition has degree 3.
    arm C  degree3-frame    quadratic layer frame(a) -> w (symmetric
                            quadratic features, 21 per output), then a
                            general bilinear map on ([u, v, w], frame(b))
                            -> R^6; can represent frame composition
                            exactly.

Shared scorer: score(a,b,c) = temperature * cosine(interaction output,
candidate features of c), candidate features 9D for arm A and 6D for
arms B/C.

## Hypotheses

    H1 (representational): least-squares fits of arms A and C reproduce the
        true product features to relative MSE < 1e-6 on held-out Haar
        pairs; arm B has an irreducible error floor (predicted near 1/3 of
        target variance: the E[b_z^2] * E[|u x v|^2] component of each
        product column is uncorrelated with every bilinear feature under
        Haar).

    H2 (trained ranking): on the actual ranking distribution, at canonical
        scale, final T2 margin ordering A >= C > B.

## Protocol

Step 1 exactness: closed-form least squares on 20000 Haar pairs, evaluated
on 20000 held-out pairs (arm C fitted in two closed-form stages, quadratic
layer to w then bilinear with the fitted w).

Step 2 training: make_pair_set ranking data exactly as the trainers build
it (variant so3, sigma 0.2, exclusion radius 0.5, K = 64, pool N = 2048,
f = 0.01), cross-entropy over K scores, AdamW lr 1e-3 wd 1.0, 20k steps,
batch 1024, seeds 0..4, CPU. Margin and per-tensor Frobenius norms logged
every 500 steps. T2 = 2048 fresh Haar pairs.

## Endpoints and scale normalization

Primary: final T2 margin per arm at canonical scale, paired across the
five seeds. Secondary: T2 accuracy, steps to 90% of final accuracy,
temperature, raw tensor norms (labeled parameterization-dependent).

Declared scale normalization: the scorer is exactly invariant to the
global scale of each arm's final bilinear tensor (cosine normalization),
so the canonical form rescales every learned tensor to unit Frobenius
norm and folds all scale into the temperature. Canonical margins equal
raw margins by this exact invariance; the canonicalization is what makes
cross-arm norm comparisons meaningful. Raw parameter norms are reported
but carry no intrinsic meaning across the differently parameterized arms.

## Failure criteria

A diverging seed (final train loss above 2.0 or NaN) is reported, not
discarded. If more than one seed of an arm diverges the arm's trained
comparison is declared unusable; H1 is unaffected by training outcomes.

## Claim boundary

The verdict is scoped to this explicit model class. No universal lower
bound over architectures is claimed.
