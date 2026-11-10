"""SU(3) representation theory and encodings for the central-quotient program.

Elements are 3x3 special-unitary matrices (numpy complex128, batched [n, 3, 3]).
The center is Z_3 = {omega^k I}, omega = exp(2 pi i / 3); an irrep of highest
weight (p, q) carries central character omega^{p-q mod 3} ("triality") and
descends to PSU(3) = SU(3)/Z_3 iff p == q (mod 3).

Implemented irreps (matrix coefficients as explicit functions of U):
    (0,0) trivial 1;  (1,0) fundamental 3;   (0,1) anti-fundamental 3bar;
    (1,1) adjoint 8;  (2,0) Sym^2 = 6;       (0,2) 6bar;
    (3,0) Sym^3 = 10; (0,3) 10bar.
Symmetric powers use the monomial basis x^alpha / sqrt(alpha!) (the same
construction as the half-integer Wigner-D code in core.su2, in three variables);
the adjoint uses the orthonormal Gell-Mann frame E_a = lambda_a / sqrt(2).
Mixed tensors beyond (1,1), e.g. (2,1) = 15, are not implemented in v1.

Casimir normalization: C2(p,q) = (p^2 + q^2 + p q)/3 + p + q, so
C2(1,0) = 4/3, C2(1,1) = 3, C2(2,0) = 10/3, C2(3,0) = 6.
"""
import itertools
import math

import numpy as np

OMEGA = np.exp(2j * np.pi / 3)

# --------------------------- Gell-Mann matrices -------------------------------

GELL_MANN = np.array([
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(3), 0], [0, 0, -2 / np.sqrt(3)]],
], dtype=np.complex128)


# ------------------------------ group operations ------------------------------

def haar_su3(n, rng):
    """n Haar-distributed SU(3) matrices, [n, 3, 3] complex."""
    g = (rng.standard_normal((n, 3, 3)) + 1j * rng.standard_normal((n, 3, 3))
         ) / np.sqrt(2)
    q, r = np.linalg.qr(g)
    d = np.einsum("nii->ni", r)
    q = q * (d / np.abs(d))[:, None, :]           # Haar on U(3)
    det = np.linalg.det(q)
    return q * np.exp(-1j * np.angle(det) / 3)[:, None, None]


def mm(U, V):
    return np.einsum("...ij,...jk->...ik", U, V)


def inv(U):
    return np.conj(np.swapaxes(U, -1, -2))


def su3_expmap(xi):
    """xi [n, 8] real -> exp(i sum_a xi_a lambda_a / 2), [n, 3, 3]."""
    import torch
    X = np.einsum("na,aij->nij", xi, GELL_MANN) / 2.0
    return torch.matrix_exp(torch.tensor(1j * X)).numpy()


def chordal(U, V, quotient=False):
    """Frobenius distance ||U - V||_F; for the PSU(3) quotient, the minimum
    over the center orbit {omega^k V}."""
    if not quotient:
        return np.linalg.norm((U - V).reshape(*U.shape[:-2], 9), axis=-1)
    ds = [np.linalg.norm((U - (OMEGA ** k) * V).reshape(*U.shape[:-2], 9),
                         axis=-1) for k in range(3)]
    return np.min(np.stack(ds), axis=0)


# ------------------------------ irreducible reps ------------------------------

def _sym_power(U, k):
    """Sym^k of the defining rep: basis x^alpha / sqrt(alpha!), |alpha| = k."""
    exps = [a for a in itertools.product(range(k + 1), repeat=3)
            if sum(a) == k]
    dim = len(exps)
    n = U.shape[0]
    cols = {e: i for i, e in enumerate(exps)}
    D = np.zeros((n, dim, dim), dtype=np.complex128)
    for a_idx, alpha in enumerate(exps):
        # polynomial prod_j (x . U[:, :, j])^{alpha_j} as monomial coefficients
        poly = {(0, 0, 0): np.ones(n, dtype=np.complex128)}
        for j in range(3):
            for _ in range(alpha[j]):
                new = {}
                for e, cf in poly.items():
                    for i in range(3):
                        e2 = tuple(e[t] + (1 if t == i else 0) for t in range(3))
                        new[e2] = new.get(e2, 0) + cf * U[:, i, j]
                poly = new
        norm_a = math.sqrt(math.prod(math.factorial(t) for t in alpha))
        for beta, cf in poly.items():
            norm_b = math.sqrt(math.prod(math.factorial(t) for t in beta))
            D[:, cols[beta], a_idx] = cf * (norm_b / norm_a)
    return D


_E8 = GELL_MANN / np.sqrt(2)          # orthonormal: tr(E_a E_b) = delta_ab


def _adjoint(U):
    UE = np.einsum("nij,ajk->naik", U, _E8)
    UEU = np.einsum("naik,nkl->nail", UE, inv(U))
    return np.einsum("bli,nail->nba", _E8, UEU)   # tr(E_b . U E_a U^dag)


def su3_rep(U, pq):
    """Matrix coefficients of the (p, q) irrep at U [n, 3, 3]."""
    p, q = pq
    U = np.asarray(U)
    if pq == (0, 0):
        return np.ones((U.shape[0], 1, 1), dtype=np.complex128)
    if q == 0:
        return _sym_power(U, p)
    if p == 0:
        return np.conj(_sym_power(U, q))
    if pq == (1, 1):
        return _adjoint(U)
    raise NotImplementedError(f"irrep {pq} not implemented")


def su3_character(U, pq):
    return np.einsum("nii->n", su3_rep(U, pq))


def su3_dim(pq):
    p, q = pq
    return (p + 1) * (q + 1) * (p + q + 2) // 2


def triality(pq):
    return (pq[0] - pq[1]) % 3


def casimir(pq):
    p, q = pq
    return (p * p + q * q + p * q) / 3 + p + q


SECTORS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (3, 0), (0, 3)]

# Real-valued activations cannot distinguish a complex irrep from its
# conjugate: the realified matrix coefficients of (p, q) and (q, p) span the
# same real function space. Spectroscopy therefore tracks conjugate-PAIR
# sectors, labeled by the representative with p >= q:
#   (1,0) = 3 + 3bar (charged), (1,1) = 8 (trivial charge, real type),
#   (2,0) = 6 + 6bar (charged), (3,0) = 10 + 10bar (trivial charge).
REAL_SECTORS = [(0, 0), (1, 0), (1, 1), (2, 0), (3, 0)]


# ------------------------------- spectroscopy ---------------------------------

def sector_basis(Us, sectors=tuple(REAL_SECTORS)):
    """Jointly-orthonormalized sampled bases of realified matrix coefficients,
    processed in order of Casimir (ties: as listed). Same construction as
    core.su2.spin_basis. Keys are conjugate-pair representatives (p >= q)."""
    N = Us.shape[0]
    bases, prev = {}, np.zeros((N, 0))
    for pq in sorted(sectors, key=lambda s: (casimir(s), s)):
        D = su3_rep(Us, pq).reshape(N, -1)
        cols = np.concatenate([D.real, D.imag], axis=1)
        cols = cols - cols.mean(axis=0, keepdims=True)
        if prev.shape[1]:
            cols = cols - prev @ (prev.T @ cols)
        keep = np.linalg.norm(cols, axis=0) > 1e-6 * math.sqrt(N)
        cols = cols[:, keep]
        if cols.shape[1] == 0:
            continue
        Qm, R = np.linalg.qr(cols)
        rank = np.abs(np.diag(R)) > 1e-9 * np.abs(R).max()
        bases[pq] = Qm[:, rank]
        prev = np.concatenate([prev, bases[pq]], axis=1)
    return bases


def sector_fve(Z, bases):
    Zc = Z - Z.mean(axis=0, keepdims=True)
    total = (Zc ** 2).sum()
    return {pq: float(((Q.T @ Zc) ** 2).sum() / total)
            for pq, Q in bases.items()}


# -------------------------------- encodings -----------------------------------

class SU3LandmarkEncoding:
    """Psi(U) = exp(alpha [Re, Im] tr(H_i^dag U)), i = 1..m (2m features).

    variant "su3": as written (center-sensitive: tr(H^dag omega U) rotates by
    omega). variant "psu3": the center-orbit average (1/3) sum_k Psi_raw(w^k U)
    -- exactly Z_3-invariant, same landmarks, same dimension.
    """

    def __init__(self, m=32, alpha=1.5, variant="su3", seed=0):
        assert variant in ("su3", "psu3")
        self.m, self.alpha, self.variant = m, alpha, variant
        rng = np.random.default_rng(seed + 40)
        H = (rng.standard_normal((m, 3, 3))
             + 1j * rng.standard_normal((m, 3, 3)))
        self.H = H / np.linalg.norm(H.reshape(m, 9), axis=1)[:, None, None]
        probe = haar_su3(65536, np.random.default_rng(seed + 41))
        raw = self._sym(probe)
        self.mean, self.std = raw.mean(0), raw.std(0) + 1e-8

    def _raw(self, U):
        t = np.einsum("mji,nji->nm", np.conj(self.H), U)
        return np.exp(self.alpha
                      * np.concatenate([t.real, t.imag], axis=1))

    def _sym(self, U):
        if self.variant == "psu3":
            return sum(self._raw((OMEGA ** k) * U) for k in range(3)) / 3
        return self._raw(U)

    def __call__(self, U):
        return ((self._sym(U) - self.mean) / self.std).astype(np.float32)

    def torch_fn(self, device):
        import torch
        Hr = torch.tensor(self.H.real, dtype=torch.float32, device=device)
        Hi = torch.tensor(self.H.imag, dtype=torch.float32, device=device)
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.std, dtype=torch.float32, device=device)
        w_re = [1.0, -0.5, -0.5]
        w_im = [0.0, math.sqrt(3) / 2, -math.sqrt(3) / 2]

        def raw(Ur, Ui):
            tr = torch.einsum("mji,nji->nm", Hr, Ur) \
                + torch.einsum("mji,nji->nm", Hi, Ui)
            ti = torch.einsum("mji,nji->nm", Hr, Ui) \
                - torch.einsum("mji,nji->nm", Hi, Ur)
            return torch.exp(self.alpha * torch.cat([tr, ti], dim=1))

        def fn(U):
            # U packed real: [n, 2, 3, 3] with channels (Re, Im)
            Ur, Ui = U[:, 0], U[:, 1]
            if self.variant == "psu3":
                acc = 0
                for c, s in zip(w_re, w_im):    # omega^k U = (c + i s) U
                    acc = acc + raw(c * Ur - s * Ui, c * Ui + s * Ur)
                x = acc / 3
            else:
                x = raw(Ur, Ui)
            return (x - mean) / std
        return fn


def pack_real(U):
    """[n,3,3] complex -> [n,2,3,3] float32 (Re, Im channels) for torch."""
    return np.stack([U.real, U.imag], axis=1).astype(np.float32)


# --------------------------------- self test ----------------------------------

def _self_test():
    rng = np.random.default_rng(0)
    U, V = haar_su3(48, rng), haar_su3(48, rng)
    assert np.abs(np.linalg.det(U) - 1).max() < 1e-10
    assert np.abs(mm(U, inv(U)) - np.eye(3)).max() < 1e-10

    for pq in SECTORS:
        d = su3_dim(pq)
        RU, RV, RUV = su3_rep(U, pq), su3_rep(V, pq), su3_rep(mm(U, V), pq)
        assert RU.shape[1:] == (d, d), (pq, RU.shape)
        hom = np.abs(np.einsum("nij,njk->nik", RU, RV) - RUV).max()
        uni = np.abs(np.einsum("nij,nkj->nik", RU, np.conj(RU))
                     - np.eye(d)).max()
        assert hom < 1e-9 and uni < 1e-9, (pq, hom, uni)
        # center: R(omega U) = omega^{p-q} R(U)
        cen = np.abs(su3_rep(OMEGA * U, pq)
                     - OMEGA ** triality(pq) * RU).max()
        assert cen < 1e-9, (pq, cen)

    # character orthonormality under Haar (Monte-Carlo)
    W = haar_su3(60000, rng)
    chi = {pq: su3_character(W, pq) for pq in SECTORS}
    for a in SECTORS:
        for b in SECTORS:
            ip = np.mean(chi[a] * np.conj(chi[b]))
            want = 1.0 if a == b else 0.0
            assert abs(ip - want) < 0.15, (a, b, ip)

    # FVE machinery recovers planted sectors (incl. the conjugate-pair rule:
    # a 3bar-coefficient signal lands in the (1,0) pair sector)
    P = haar_su3(4096, rng)
    bases = sector_basis(P)
    Z = su3_rep(P, (1, 0)).reshape(4096, -1).real @ rng.standard_normal((9, 16))
    fve = sector_fve(Z, bases)
    assert fve[(1, 0)] > 0.99 and fve[(1, 1)] < 0.01, fve
    Zb = su3_rep(P, (0, 1)).reshape(4096, -1).imag @ rng.standard_normal((9, 16))
    fveb = sector_fve(Zb, bases)
    assert fveb[(1, 0)] > 0.99, fveb
    Za = su3_rep(P, (1, 1)).reshape(4096, -1).real @ rng.standard_normal((64, 16))
    fvea = sector_fve(Za, bases)
    assert fvea[(1, 1)] > 0.99 and fvea[(1, 0)] < 0.01, fvea

    # encodings: psu3 exactly center-invariant, su3 not; torch mirror agrees
    import torch
    for variant in ("su3", "psu3"):
        e = SU3LandmarkEncoding(m=8, variant=variant)
        x, xw = e(U), e(OMEGA * U)
        if variant == "psu3":
            assert np.abs(x - xw).max() < 1e-4
        else:
            assert np.linalg.norm(x - xw, axis=1).min() > 0.5
        tv = e.torch_fn("cpu")(torch.tensor(pack_real(U))).numpy()
        assert np.abs(tv - x).max() < 1e-3, variant

    # expmap lands in SU(3) and is near-isometric at small xi
    xi = rng.standard_normal((16, 8)) * 0.1
    E = su3_expmap(xi)
    assert np.abs(np.linalg.det(E) - 1).max() < 1e-8
    assert np.abs(mm(E, inv(E)) - np.eye(3)).max() < 1e-8
    print("core.su3 self-test OK")


if __name__ == "__main__":
    _self_test()
