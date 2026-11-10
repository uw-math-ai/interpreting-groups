"""Permanent tests for the polynomial composition experiment (handoff 4.4)."""
import numpy as np

from core.su2 import haar, qmul
from experiments.poly_compose import (exactness, feat6, feat9, quad_features,
                                      rot, third_column_std)

RNG = np.random.default_rng(3)


def test_rotation_matrix_matches_quaternion_action():
    q = haar(256, RNG)
    R = rot(q)
    # orthogonality and homomorphism R(pq) = R(p) R(q)
    eye = np.einsum("nij,nkj->nik", R, R)
    assert np.allclose(eye, np.eye(3), atol=1e-10)
    p = haar(256, RNG)
    assert np.allclose(rot(qmul(p, q)), np.einsum("nij,njk->nik",
                                                  rot(p), rot(q)), atol=1e-10)


def test_third_column_is_cross_product_of_frame():
    q = haar(256, RNG)
    f = feat6(q)
    u = f[:, 0::2] / np.sqrt(3)
    v = f[:, 1::2] / np.sqrt(3)
    w = np.cross(u, v)
    assert np.allclose(third_column_std(q), np.sqrt(3) * w, atol=1e-10)


def test_exactness_gap_between_degrees():
    """Arms A and C are representationally exact; the best bilinear frame
    interaction has an irreducible floor more than 10x above A's error."""
    ex = exactness(n=4000, seed=0)
    assert ex["A"]["rel_mse_holdout"] < 1e-6
    assert ex["C"]["rel_mse_holdout"] < 1e-6
    assert ex["C"]["stage1_rel_mse"] < 1e-6
    assert ex["B"]["rel_mse_holdout"] > 10 * max(ex["A"]["rel_mse_holdout"],
                                                 1e-12)
    assert ex["B"]["rel_mse_holdout"] > 0.05


def test_quad_features_span_cross_product():
    q = haar(1024, RNG)
    f = feat6(q)
    X = quad_features(f)
    Y = third_column_std(q)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = X @ W - Y
    assert (resid ** 2).sum() / (Y ** 2).sum() < 1e-10
