"""G5.2: mandatory synthetic calibration for the Haar coefficient estimator.

Every test runs at realistic sample sizes on synthetic score functions with
known class-function structure. The final test guards the failure mode of
the old QR incremental-variance profile, which is not a coefficient
estimator.
"""
import numpy as np
import pytest

from core.su2 import expmap, haar, qconj, qmul, wigner_character, wigner_D
from experiments.hk_haar import estimate, factorize, reconstruction_r2

JS = [0.0, 0.5, 1.0, 1.5, 2.0]
RNG = np.random.default_rng(7)


def rel_of(a, b, c):
    return qmul(qmul(a, b), qconj(c))


def class_score(coefs):
    """Synthetic score depending on the triple only through r = a b c^-1."""
    def fn(a, b, c):
        r = rel_of(a, b, c)
        return sum(w * wigner_character(j, r) for j, w in coefs.items())
    return fn


def test_group_convention_triple_returns_requested_r():
    r = haar(512, RNG)
    a, b, c = factorize(r, np.random.default_rng(0))
    assert np.allclose(rel_of(a, b, c), r, atol=1e-10)


def test_character_orthonormality_under_haar():
    q = haar(200_000, np.random.default_rng(1))
    for j in (0.5, 1.0, 1.5, 2.0):
        chi = wigner_character(j, q)
        assert abs(np.mean(chi * chi) - 1.0) < 0.05, j
        for k in (0.5, 1.0, 1.5, 2.0):
            if k > j:
                assert abs(np.mean(chi * wigner_character(k, q))) < 0.05


def _run(coefs, n_f=4, seed=0):
    return estimate(class_score(coefs), JS, n_r=6000, n_f=n_f, seed=seed)


def test_bandlimited_target_recovers_coefficients():
    est = _run({1.0: 1.0, 2.0: 0.5})["coef"]
    assert abs(est[1.0]["a_hat"] - 1.0) < 0.1
    assert abs(est[2.0]["a_hat"] - 0.5) < 0.1
    for j in (0.5, 1.5):                     # calibrated interval covers zero
        lo, hi = est[j]["ci95"]
        assert lo < 0.0 < hi, (j, est[j])


def test_negative_sign_is_recovered():
    est = _run({0.5: 1.0, 1.5: -0.25})["coef"]
    assert abs(est[0.5]["a_hat"] - 1.0) < 0.1
    assert est[1.5]["a_hat"] < -0.15
    assert est[1.5]["ci95"][1] < 0.0


def test_global_scale_propagates():
    base = _run({1.0: 1.0, 2.0: 0.5})["coef"]
    scaled = _run({1.0: 3.7, 2.0: 3.7 * 0.5})["coef"]
    for j in (1.0, 2.0):
        assert abs(scaled[j]["a_hat"] / base[j]["a_hat"] - 3.7) < 0.15


def test_nuisance_component_is_detected_and_removed():
    def fn(a, b, c):
        r = rel_of(a, b, c)
        return wigner_character(1.0, r) + 0.6 * a[..., 1]
    est = estimate(fn, JS, n_r=6000, n_f=8, seed=3)
    assert abs(est["coef"][1.0]["a_hat"] - 1.0) < 0.12
    assert est["nuisance_fraction"] > 0.05    # detected
    lo, hi = est["coef"][0.5]["ci95"]
    assert lo < 0.0 < hi


def test_non_class_function_yields_incomplete_reconstruction():
    def fn(a, b, c):
        r = rel_of(a, b, c)
        return wigner_D(1.0, r)[..., 0, 1].real.astype(float)
    est = estimate(fn, JS, n_r=6000, n_f=2, seed=4)
    for j in JS:
        assert abs(est["coef"][j]["a_hat"]) < 0.1, j
    r2 = reconstruction_r2(fn, est["coef"], JS, n_r=3000, n_f=2, seed=5)
    assert r2 < 0.2                           # residual energy is nonzero


def test_old_qr_profile_fails_on_known_bandlimited_target():
    """Regression guard: the QR incremental-variance profile reports
    substantial half-integer content for the exact chi_1 + 0.5 chi_2 target
    under the task sampling measure. That is the calibration failure that
    blocked P4; this test must keep failing loudly if anyone reintroduces
    the QR profile as a coefficient or support estimator."""
    from experiments.hk_analysis import coef_fit, sample_design
    cfg = {"loss": "bandlim", "bl_js": [1.0, 2.0], "bl_coefs": [1.0, 0.5]}
    rng = np.random.default_rng(11)
    _, _, _, X, js, y = sample_design(cfg, 6000, rng, j_max=3.0)
    frac, _ = coef_fit(X, y)
    half = sum(f for j, f in zip(js, frac) if j % 1.0 != 0.0)
    assert half > 0.05, ("QR profile unexpectedly clean; if the estimator "
                         "was fixed, retire this guard deliberately")
