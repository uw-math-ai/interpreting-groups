"""Spin(5) = Sp(2) (2x2 quaternionic unitaries) and SO(5) = Sp(2)/{+-1}.

Elements are quaternion matrices U [n, 2, 2, 4] with U^dag U = I (quaternionic
conjugate-transpose). Center = {+-I}. Tracked irreps and pre-registered
predictions (from experiments.carrier_search, recorded before running):

    4  spinor (quaternionic type; the defining rep)   -- Spin(5) prediction
    5  vector (real; action on traceless quaternion-hermitian 2x2)
                                                       -- SO(5) prediction
    10 adjoint (real; action on anti-hermitian 2x2)
    14 = Sym^2(5) minus trace (real)

Casimir order: 4 < 5 < 10 < 14.
"""
import math

import numpy as np

from core.su2 import haar as _qhaar
from core.su2 import qconj, qmul


# --------------------------- quaternion matrix ops ----------------------------

def qmat_mul(A, B):
    """[..., 2, 2, 4] x [..., 2, 2, 4] quaternion matrix product."""
    C = np.zeros(np.broadcast_shapes(A.shape, B.shape), dtype=A.dtype)
    for i in range(2):
        for k in range(2):
            acc = 0
            for j in range(2):
                acc = acc + qmul(A[..., i, j, :], B[..., j, k, :])
            C[..., i, k, :] = acc
    return C


def qmat_dag(U):
    """Quaternionic conjugate transpose."""
    return qconj(np.swapaxes(U, -3, -2))


def haar_sp2(n, rng):
    """Haar on Sp(2): quaternion Ginibre + modified Gram-Schmidt (columns as
    right-quaternion-module vectors)."""
    G = rng.standard_normal((n, 2, 2, 4))
    c0, c1 = G[:, :, 0, :], G[:, :, 1, :]           # columns: [n, 2, 4]

    def qdot(u, v):                                  # sum_i conj(u_i) v_i
        return qmul(qconj(u[:, 0]), v[:, 0]) + qmul(qconj(u[:, 1]), v[:, 1])

    def normalize(v):
        nrm = np.sqrt((v ** 2).sum(axis=(1, 2), keepdims=True))
        return v / nrm

    u0 = normalize(c0)
    c1 = c1 - qmul(u0, qdot(u0, c1)[:, None, :])
    u1 = normalize(c1)
    return np.stack([u0, u1], axis=2)                # [n, 2(row), 2(col), 4]


def embed_c(U):
    """Quaternion matrix -> 4x4 complex (entrywise q -> [[a, b], [-b*, a*]],
    a = w + ix, b = y + iz; an algebra homomorphism)."""
    w, x, y, z = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
    a, b = w + 1j * x, y + 1j * z
    n = U.shape[0]
    M = np.zeros((n, 4, 4), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            M[:, 2 * i, 2 * j] = a[:, i, j]
            M[:, 2 * i, 2 * j + 1] = b[:, i, j]
            M[:, 2 * i + 1, 2 * j] = -np.conj(b[:, i, j])
            M[:, 2 * i + 1, 2 * j + 1] = np.conj(a[:, i, j])
    return M


def unembed_c(M):
    n = M.shape[0]
    U = np.zeros((n, 2, 2, 4))
    for i in range(2):
        for j in range(2):
            a, b = M[:, 2 * i, 2 * j], M[:, 2 * i, 2 * j + 1]
            U[:, i, j, 0], U[:, i, j, 1] = a.real, a.imag
            U[:, i, j, 2], U[:, i, j, 3] = b.real, b.imag
    return U


def sp2_expmap(xi_herm):
    """exp of an anti-hermitian quaternion matrix given as [n, 2, 2, 4]
    (X^dag = -X), via the complex embedding."""
    import torch
    return unembed_c(torch.matrix_exp(torch.tensor(embed_c(xi_herm))).numpy())


def random_algebra(n, scale, rng):
    """Random anti-hermitian quaternion 2x2 (the sp(2) algebra, 10-dim)."""
    X = np.zeros((n, 2, 2, 4))
    X[:, 0, 0, 1:] = rng.standard_normal((n, 3)) * scale     # imaginary diag
    X[:, 1, 1, 1:] = rng.standard_normal((n, 3)) * scale
    off = rng.standard_normal((n, 4)) * scale                # free off-diag
    X[:, 0, 1, :] = off
    X[:, 1, 0, :] = -qconj(off)
    return X


def chordal(U, V, quotient=False):
    d1 = np.sqrt(((U - V) ** 2).sum(axis=(-3, -2, -1)))
    if not quotient:
        return d1
    d2 = np.sqrt(((U + V) ** 2).sum(axis=(-3, -2, -1)))
    return np.minimum(d1, d2)


# ------------------------------ representations -------------------------------

def _herm_basis():
    """Orthonormal basis (Re tr H^dag H' = delta) of traceless quaternion-
    hermitian 2x2 -- the vector 5."""
    B = []
    E = np.zeros((2, 2, 4))
    E[0, 0, 0], E[1, 1, 0] = 1 / math.sqrt(2), -1 / math.sqrt(2)
    B.append(E)
    for u in range(4):                               # off-diag 1, i, j, k
        E = np.zeros((2, 2, 4))
        E[0, 1, u] = 1 / math.sqrt(2)
        E[1, 0, 0 if u == 0 else u] = (1 if u == 0 else -1) / math.sqrt(2)
        B.append(E)
    return np.stack(B)                               # [5, 2, 2, 4]


def _antiherm_basis():
    """Orthonormal basis of anti-hermitian quaternion 2x2 -- the adjoint 10."""
    B = []
    for d in range(2):                               # imaginary diagonals
        for u in range(1, 4):
            E = np.zeros((2, 2, 4))
            E[d, d, u] = 1.0
            B.append(E)
    for u in range(4):                               # off-diag u, -conj(u)
        E = np.zeros((2, 2, 4))
        E[0, 1, u] = 1 / math.sqrt(2)
        E[1, 0, u] = (-1 if u == 0 else 1) / math.sqrt(2)
        B.append(E)
    return np.stack(B)                               # [10, 2, 2, 4]

_B5, _B10 = _herm_basis(), _antiherm_basis()


def _conj_action(U, basis):
    """R(U)[a, b] = Re tr_H( basis_a^dag  U basis_b U^dag ). For quaternion
    matrices, Re tr_H(E^dag B) equals the flat componentwise dot of E and B
    (the dagger's conjugation cancels the quaternion product's sign
    structure), so no explicit dagger is needed here."""
    n = U.shape[0]
    Ud = qmat_dag(U)
    cols = []
    for b in range(basis.shape[0]):
        X = qmat_mul(qmat_mul(U, np.broadcast_to(basis[b], (n, 2, 2, 4))), Ud)
        cols.append(X)
    X = np.stack(cols, axis=1)                       # [n, dB, 2, 2, 4]
    return np.einsum("aijq,nbijq->nab", basis, X)


def rep_vector(U):
    return _conj_action(U, _B5)                      # [n, 5, 5] real


def rep_adjoint(U):
    return _conj_action(U, _B10)                     # [n, 10, 10] real


def rep_spinor(U):
    return embed_c(U)                                # [n, 4, 4] complex


def rep_14(U):
    """Sym^2 of the vector 5 minus the trace part (traceless symmetric 5x5)."""
    R = rep_vector(U)
    n = R.shape[0]
    idx = [(a, b) for a in range(5) for b in range(a, 5)]
    # orthonormal basis of symmetric 5x5, then project out the identity
    out = np.einsum("nab,ncd->nacbd", R, R).reshape(n, 25, 25)
    return out                                       # full Sym handled in FVE


def sector_basis5(Us, which=("4", "5", "10", "14")):
    """Sampled orthonormal bases per sector, joint QR in Casimir order."""
    N = Us.shape[0]
    cols_by = {}
    if "4" in which:
        D = rep_spinor(Us).reshape(N, -1)
        cols_by["4"] = np.concatenate([D.real, D.imag], axis=1)
    if "5" in which:
        cols_by["5"] = rep_vector(Us).reshape(N, -1)
    if "10" in which:
        cols_by["10"] = rep_adjoint(Us).reshape(N, -1)
    if "14" in which:
        R = rep_vector(Us)
        T = np.einsum("nab,ncd->nacbd", R, R).reshape(N, 25, 25)
        cols_by["14"] = T.reshape(N, -1)             # contains 14 (+ lower;
        # joint QR against earlier sectors removes the overlap)
    bases, prev = {}, np.zeros((N, 0))
    for key in ["4", "5", "10", "14"]:
        if key not in cols_by:
            continue
        cols = cols_by[key]
        cols = cols - cols.mean(0, keepdims=True)
        if prev.shape[1]:
            cols = cols - prev @ (prev.T @ cols)
        keep = np.linalg.norm(cols, axis=0) > 1e-6 * math.sqrt(N)
        cols = cols[:, keep]
        if cols.shape[1] == 0:
            continue
        Qm, R_ = np.linalg.qr(cols)
        rank = np.abs(np.diag(R_)) > 1e-9 * np.abs(R_).max()
        bases[key] = Qm[:, rank]
        prev = np.concatenate([prev, bases[key]], axis=1)
    return bases


# --------------------------------- encoding -----------------------------------

class Sp2LandmarkEncoding:
    """Psi(U)_i = exp(alpha Re tr_H(H_i^dag U)); variant "so5" averages over
    the center orbit {U, -U}."""

    def __init__(self, m=48, alpha=2.0, variant="spin5", seed=0):
        assert variant in ("spin5", "so5")
        self.m, self.alpha, self.variant = m, alpha, variant
        rng = np.random.default_rng(seed + 80)
        H = rng.standard_normal((m, 2, 2, 4))
        self.H = H / np.sqrt((H ** 2).sum(axis=(1, 2, 3), keepdims=True))
        probe = haar_sp2(65536, np.random.default_rng(seed + 81))
        raw = self._sym(probe)
        self.mean, self.std = raw.mean(0), raw.std(0) + 1e-8

    def _raw(self, U):
        x = np.einsum("mijq,nijq->nm", self.H, U)    # Re tr_H(H^dag U)
        return np.exp(self.alpha * x)

    def _sym(self, U):
        if self.variant == "so5":
            return (self._raw(U) + self._raw(-U)) / 2
        return self._raw(U)

    def __call__(self, U):
        return ((self._sym(U) - self.mean) / self.std).astype(np.float32)

    def torch_fn(self, device):
        import torch
        H = torch.tensor(self.H, dtype=torch.float32, device=device)
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.std, dtype=torch.float32, device=device)

        def raw(U):
            x = torch.einsum("mijq,...ijq->...m", H, U)
            return torch.exp(self.alpha * x)

        def fn(U):
            x = (raw(U) + raw(-U)) / 2 if self.variant == "so5" else raw(U)
            return (x - mean) / std
        return fn



def geodesic_sp2(U, V, quotient=False):
    """Bi-invariant geodesic distance on Sp(2), elementwise over the batch.

    Eigenphases of W = U^dag V come in conjugate pairs {e^{+-i th1},
    e^{+-i th2}}; d = sqrt(th1^2 + th2^2). Solved from the trace identities
    cos th1 + cos th2 = tr_C(W)/2 and cos^2 th1 + cos^2 th2 =
    (tr_C(W^2)/2 + 2)/2 (no eigendecomposition needed). With
    quotient=True returns min(d(U, V), d(U, -V)), the SO(5) = Sp(2)/{+-1}
    quotient distance.
    """
    def _d(U, V):
        W = qmat_mul(qmat_dag(U), V)
        W2 = qmat_mul(W, W)
        s1 = W[..., 0, 0, 0] + W[..., 1, 1, 0]        # Re tr_H(W) = tr_C/2
        s2 = W2[..., 0, 0, 0] + W2[..., 1, 1, 0]      # Re tr_H(W^2)
        sumsq = (s2 + 2.0) / 2.0                      # cos^2 th1 + cos^2 th2
        prod = (s1 ** 2 - sumsq) / 2.0                # cos th1 * cos th2
        disc = np.sqrt(np.maximum(s1 ** 2 / 4.0 - prod, 0.0))
        c1 = np.clip(s1 / 2.0 + disc, -1.0, 1.0)
        c2 = np.clip(s1 / 2.0 - disc, -1.0, 1.0)
        return np.sqrt(np.arccos(c1) ** 2 + np.arccos(c2) ** 2)

    if not quotient:
        return _d(U, V)
    return np.minimum(_d(U, V), _d(U, -V))


def _torch_embed_c(U):
    """Torch version of embed_c. U: real tensor [..., 2, 2, 4] ->
    complex [..., 4, 4]."""
    import torch
    w, x, y, z = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
    a, b = torch.complex(w, x), torch.complex(y, z)
    rows = []
    for i in range(2):
        r0 = torch.stack([a[..., i, 0], b[..., i, 0],
                          a[..., i, 1], b[..., i, 1]], dim=-1)
        r1 = torch.stack([-torch.conj(b[..., i, 0]), torch.conj(a[..., i, 0]),
                          -torch.conj(b[..., i, 1]),
                          torch.conj(a[..., i, 1])], dim=-1)
        rows += [r0, r1]
    return torch.stack(rows, dim=-2)


def _torch_rep_vector(U, device):
    """Torch vector rep via the complex embedding:
    V(U)[a, b] = (1/2) Re tr_C(embed(E_a)^dag embed(U) embed(E_b)
    embed(U)^dag)."""
    import torch
    E = torch.tensor(embed_c(_B5.astype(np.float64)),
                     dtype=torch.complex64, device=device)   # [5, 4, 4]
    Ed = E.conj().transpose(-1, -2)
    M = _torch_embed_c(U).to(torch.complex64)
    Md = M.conj().transpose(-1, -2)
    out = torch.einsum("aij,njk,bkl,nli->nab", Ed, M, E, Md)
    return 0.5 * out.real


class So5VectorEncoding:
    """DIAGNOSTIC UPPER CONTROL: the raw 5x5 SO(5) vector-representation
    matrix, flattened and standardized under Haar. Exactly center-invariant
    by construction and it exposes the vector coefficients directly, so a
    model trained on it cannot be cited as evidence of spontaneous vector
    selection. Use So5RandomFeatureEncoding for the selection test."""

    def __init__(self, seed=0):
        self.m = 25
        probe = haar_sp2(65536, np.random.default_rng(seed + 91))
        raw = rep_vector(probe).reshape(-1, 25)
        self.mean, self.std = raw.mean(0), raw.std(0) + 1e-8

    def __call__(self, U):
        raw = rep_vector(U).reshape(U.shape[0], 25)
        return ((raw - self.mean) / self.std).astype(np.float32)

    def torch_fn(self, device):
        import torch
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.std, dtype=torch.float32, device=device)

        def fn(U):
            shp = U.shape[:-3]
            V = _torch_rep_vector(U.reshape(-1, 2, 2, 4), U.device)
            V = V.reshape(shp + (25,))
            return (V - mean) / std
        return fn


class So5RandomFeatureEncoding:
    """Generic quotient observation (the less biased SO(5) selection test):
    Psi(g)_i = exp(alpha <V(g), W_i>_F) with frozen unit-norm Gaussian 5x5
    matrices W_i, exactly center-invariant because V(-g) = V(g). alpha is
    chosen empirically so the pre-activation standard deviation matches the
    Sp(2) landmark encoding's (alpha_cover * std_cover / std_rf); the
    measured values are recorded on the instance."""

    def __init__(self, m=48, alpha_cover=2.0, seed=0):
        self.m = m
        rng = np.random.default_rng(seed + 95)
        W = rng.standard_normal((m, 5, 5))
        self.W = W / np.sqrt((W ** 2).sum(axis=(1, 2), keepdims=True))
        # empirical pre-activation matching against the cover landmarks
        probe = haar_sp2(65536, np.random.default_rng(seed + 96))
        H = rng.standard_normal((m, 2, 2, 4))
        H /= np.sqrt((H ** 2).sum(axis=(1, 2, 3), keepdims=True))
        pre_cover = np.einsum("mijq,nijq->nm", H, probe)
        Vp = rep_vector(probe)
        pre_rf = np.einsum("mab,nab->nm", self.W, Vp)
        self.pre_std_cover = float(pre_cover.std())
        self.pre_std_rf = float(pre_rf.std())
        self.alpha = alpha_cover * self.pre_std_cover / self.pre_std_rf
        raw = np.exp(self.alpha * pre_rf)
        self.mean, self.std = raw.mean(0), raw.std(0) + 1e-8

    def _raw(self, U):
        pre = np.einsum("mab,nab->nm", self.W, rep_vector(U))
        return np.exp(self.alpha * pre)

    def __call__(self, U):
        return ((self._raw(U) - self.mean) / self.std).astype(np.float32)

    def torch_fn(self, device):
        import torch
        W = torch.tensor(self.W, dtype=torch.float32, device=device)
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.std, dtype=torch.float32, device=device)

        def fn(U):
            shp = U.shape[:-3]
            V = _torch_rep_vector(U.reshape(-1, 2, 2, 4), U.device)
            V = V.reshape(shp + (5, 5))
            pre = torch.einsum("mab,...ab->...m", W, V)
            return (torch.exp(self.alpha * pre) - mean) / std
        return fn


def get_spin5_encoding(name, variant, m=48, alpha=2.0, seed=0):
    """Factory. "landmark" is the historical Sp2LandmarkEncoding (its own
    variant handling); "so5vec"/"so5rf" are intrinsic quotient observations
    and require variant == "so5"."""
    if name == "landmark":
        return Sp2LandmarkEncoding(m=m, alpha=alpha, variant=variant,
                                   seed=seed)
    assert variant == "so5", f"{name} is an intrinsic SO(5) observation"
    if name == "so5vec":
        return So5VectorEncoding(seed=seed)
    if name == "so5rf":
        return So5RandomFeatureEncoding(m=m, alpha_cover=alpha, seed=seed)
    raise ValueError(name)


# --------------------------------- self test ----------------------------------

def _self_test():
    rng = np.random.default_rng(0)
    U, V = haar_sp2(64, rng), haar_sp2(64, rng)
    eye = np.zeros((2, 2, 4))
    eye[0, 0, 0] = eye[1, 1, 0] = 1
    assert np.abs(qmat_mul(qmat_dag(U), U) - eye).max() < 1e-10
    UV = qmat_mul(U, V)
    # embedding is a homomorphism; spinor rep = embedding
    assert np.abs(embed_c(UV) - embed_c(U) @ embed_c(V)).max() < 1e-10
    assert np.abs(unembed_c(embed_c(U)) - U).max() < 1e-12
    for rep, d in [(rep_vector, 5), (rep_adjoint, 10)]:
        RU, RV, RUV = rep(U), rep(V), rep(UV)
        assert np.abs(RU @ RV - RUV).max() < 1e-9, d
        assert np.abs(RU @ np.swapaxes(RU, 1, 2) - np.eye(d)).max() < 1e-9, d
        assert np.abs(rep(-U) - RU).max() < 1e-9, d          # kernel {+-1}
    assert np.abs(rep_spinor(-U) + rep_spinor(U)).max() < 1e-12
    # expmap lands in Sp(2)
    E = sp2_expmap(random_algebra(16, 0.2, rng))
    assert np.abs(qmat_mul(qmat_dag(E), E) - eye).max() < 1e-8
    # FVE recovery: planted vector-rep signal lands in "5"
    P = haar_sp2(3072, rng)
    bases = sector_basis5(P)
    Z = rep_vector(P).reshape(3072, -1) @ rng.standard_normal((25, 16))
    Zc = Z - Z.mean(0)
    tot = (Zc ** 2).sum()
    fr = {k: ((Q.T @ Zc) ** 2).sum() / tot for k, Q in bases.items()}
    assert fr["5"] > 0.99 and fr["4"] < 0.01, fr
    Z2 = rep_spinor(P).reshape(3072, -1).real @ rng.standard_normal((16, 16))
    Z2c = Z2 - Z2.mean(0)
    fr2 = {k: ((Q.T @ Z2c) ** 2).sum() / (Z2c ** 2).sum()
           for k, Q in bases.items()}
    assert fr2["4"] > 0.99, fr2
    # encodings
    import torch
    for variant in ("spin5", "so5"):
        e = Sp2LandmarkEncoding(m=8, variant=variant)
        if variant == "so5":
            assert np.abs(e(U) - e(-U)).max() < 1e-4
        else:
            assert np.linalg.norm(e(U) - e(-U), axis=1).min() > 0.5
        tv = e.torch_fn("cpu")(torch.tensor(U, dtype=torch.float32)).numpy()
        assert np.abs(tv - e(U)).max() < 1e-3
    print("core.spin5 self-test OK")


if __name__ == "__main__":
    _self_test()
