# Preregistration: SO(5) independent quotient observations

Date: 2026-09-01. Committed before the corresponding queue file exists.
Handoff reference: Evidence-first final handoff, Section 6.3.

## Question

Is the replicated SO(5) vector-plus-adjoint mixture a property of the
antipodally averaged Spin(5) observation (a conditioning artifact), or a
property of the SO(5) task itself?

## New observations (implemented and unit-tested before this file)

    so5vec   the 5x5 vector-representation matrix, standardized under Haar.
             Diagnostic upper control: exactly center-invariant and exposes
             vector coefficients directly, so it cannot by itself support a
             claim of spontaneous vector selection.

    so5rf    a frozen random feature map exp(alpha <V(g), W_i>_F), 48
             unit-norm Gaussian 5x5 projections, alpha chosen so the
             pre-activation standard deviation matches the cover landmark
             encoding empirically (recorded on the instance). The less
             biased selection test.

Both are exactly center-invariant, separate distinct quotient elements, and
have full-rank (10) local Jacobians (tests/test_so5_observations.py).

## Runs

    so5rf   seeds 0..4, f = 0.03, 100k steps      (the selection test)
    so5vec  seeds 0..1, f = 0.03, 100k steps      (upper control)

f = 0.03 because it is the only manipulation that has improved SO(5)
generalization (T2 0.60 to 0.731). All other settings identical to the
existing so5 runs (N 2048, sigma 0.15, exclusion 0.7, K 64, batch 2048,
lr 1e-3, wd 1.0). Run names data/spin5/so5rf_f0.03_s{s} and
data/spin5/so5vec_f0.03_s{s}.

## Performance bar

A condition "solves" the task if final T2 >= 0.85 (between the current
averaged-map 0.60-0.73 and the cover's 0.98).

## Precommitted verdict rule (handoff Section 6.3)

    1. so5rf reaches the bar AND the vector fraction rises with the
       adjoint fraction falling below 0.10 (purification):
         the mixture is linked to observation/optimization conditioning.

    2. so5rf reaches the bar AND the vector/adjoint mixture remains
       (adjoint fraction >= 0.15, stable across seeds):
         strong evidence for a genuinely mixed selected mechanism.

    3. Only so5vec reaches the bar:
         the task is solvable, but spontaneous selection remains
         unresolved.

    4. No new map reaches the bar:
         SO(5) is retained only as an unresolved negative stress test.

The mixture is not to be explained away because vector-5 was expected.
Fractions are encoder FVE in sectors 5 and 10 at the final checkpoint,
averaged over the seeds that reach the bar (rule 1/2) or reported per seed
(rule 3/4).
