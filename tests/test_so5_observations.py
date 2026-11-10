"""G4.2c: unit tests for the intrinsic SO(5) observations and the Sp(2)
geodesic distance. No data/ dependency."""
import numpy as np
import pytest

from core.spin5 import (So5RandomFeatureEncoding, So5VectorEncoding, _B10,
                        embed_c, geodesic_sp2, haar_sp2, qmat_dag, qmat_mul,
                        rep_vector, sp2_expmap)

RNG = np.random.default_rng(5)
ENCODINGS = [("so5vec", So5VectorEncoding(seed=0)),
             ("so5rf", So5RandomFeatureEncoding(m=16, seed=0))]


def test_geodesic_matches_eigendecomposition():
    U, V = haar_sp2(128, RNG), haar_sp2(128, RNG)
    W = qmat_mul(qmat_dag(U), V)
    th = np.abs(np.angle(np.linalg.eigvals(embed_c(W))))
    d_eig = np.sqrt((th ** 2).sum(1) / 2)     # phases come in +- pairs
    assert np.abs(geodesic_sp2(U, V) - d_eig).max() < 1e-9
    # quotient distance is symmetric under either argument's sign flip
    dq = geodesic_sp2(U, V, quotient=True)
    assert np.abs(geodesic_sp2(-U, V, quotient=True) - dq).max() < 1e-9


@pytest.mark.parametrize("name,enc", ENCODINGS)
def test_exact_center_invariance(name, enc):
    U = haar_sp2(256, np.random.default_rng(1))
    assert np.abs(enc(U) - enc(-U)).max() < 1e-6, name


@pytest.mark.parametrize("name,enc", ENCODINGS)
def test_separation_of_distinct_quotient_elements(name, enc):
    U = haar_sp2(512, np.random.default_rng(2))
    d_int = geodesic_sp2(U[:, None], U[None, :256], quotient=True)
    X = enc(U)
    d_obs = np.linalg.norm(X[:, None] - X[None, :256], axis=-1)
    mask = (d_int > 0.1) & ~np.eye(512, 256, dtype=bool)
    assert d_obs[mask].min() > 1e-3, name


@pytest.mark.parametrize("name,enc", ENCODINGS)
def test_local_jacobian_full_rank(name, enc):
    """Finite-difference Jacobian along the 10 orthonormal sp(2) generators
    has full rank 10 at every tested base point."""
    U = haar_sp2(8, np.random.default_rng(3))
    eps = 1e-4
    for i in range(8):
        base = U[i:i + 1]
        cols = []
        for g in range(10):
            step = sp2_expmap(eps * _B10[None, g])
            plus = enc(qmat_mul(base, step))
            minus = enc(qmat_mul(base, qmat_dag(step)))
            cols.append((plus - minus)[0] / (2 * eps))
        J = np.stack(cols, axis=1)                    # [m, 10]
        sv = np.linalg.svd(J, compute_uv=False)
        assert sv[9] > 1e-3 * sv[0], (name, i, sv)


def test_rf_alpha_matching_is_recorded():
    enc = So5RandomFeatureEncoding(m=16, seed=0)
    assert enc.pre_std_cover > 0 and enc.pre_std_rf > 0
    expected = 2.0 * enc.pre_std_cover / enc.pre_std_rf
    assert abs(enc.alpha - expected) < 1e-12
