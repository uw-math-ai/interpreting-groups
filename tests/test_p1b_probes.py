"""G3: correctness of the rotation-matrix construction used by the probes."""
import numpy as np

from core.su2 import haar, qconj, qmul
from experiments.p1b_probes import rot


def test_rotation_matrix_matches_quaternion_conjugation():
    """R(q) v must equal the vector part of q (0, v) q^{-1} for basis and
    random vectors."""
    rng = np.random.default_rng(3)
    q = haar(256, rng)
    R = rot(q)
    for v in np.eye(3):
        pure = np.zeros((256, 4))
        pure[:, 1:] = v
        rotated = qmul(qmul(q, pure), qconj(q))[:, 1:]
        assert np.allclose(R @ v, rotated, atol=1e-10)
    v = rng.standard_normal(3)
    pure = np.zeros((256, 4))
    pure[:, 1:] = v
    rotated = qmul(qmul(q, pure), qconj(q))[:, 1:]
    assert np.allclose(R @ v, rotated, atol=1e-10)


def test_rotation_matrix_is_orthogonal_det_one():
    q = haar(64, np.random.default_rng(4))
    R = rot(q)
    eye = np.einsum("nij,nkj->nik", R, R)
    assert np.allclose(eye, np.eye(3), atol=1e-10)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-10)
