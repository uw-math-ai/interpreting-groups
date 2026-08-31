"""SU(2) as unit quaternions: sampling, group law, Wigner D^j, landmark encodings.

Conventions
-----------
A quaternion q = (w, x, y, z), |q| = 1, corresponds to the SU(2) matrix

    g(q) = [[a, b], [-conj(b), conj(a)]],   a = w + i x,  b = y + i z,

so that quaternion multiplication (core.quaternions.qmul convention) matches
matrix multiplication g(q1 q2) = g(q1) g(q2). The center is {±1}; q and -q map
to the same SO(3) rotation.

Wigner D^j (j = 0, 1/2, 1, 3/2, ...) is built from the action of g on
homogeneous polynomials of degree n = 2j in (u, v): with the orthonormal
monomial basis e_k = u^{n-k} v^k / sqrt((n-k)! k!), the matrix elements come
from expanding (a u - conj(b) v)^{n-k} (b u + conj(a) v)^k. This is exact for
half-integer j and needs no external libraries.
"""
import math

import numpy as np


# ----------------------------- quaternion algebra -----------------------------

def qmul(p, q):
    """Batched quaternion product. p, q: [..., 4] -> [..., 4]."""
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ], axis=-1)


def qconj(q):
    out = q.copy()
    out[..., 1:] *= -1
    return out


def haar(n, rng):
    """n Haar-uniform unit quaternions (normalized 4-D Gaussians), [n, 4]."""
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def expmap(xi):
    """Tangent vector xi in R^3 (rotation-angle units) -> unit quaternion.

    |xi| is the geodesic angle on SU(2) (half the SO(3) rotation angle sits in
    the quaternion half-angle; here exp(xi) = (cos|xi|, sin|xi| * xi/|xi|)).
    """
    theta = np.linalg.norm(xi, axis=-1, keepdims=True)
    small = theta < 1e-12
    direction = np.where(small, np.zeros_like(xi), xi / np.where(small, 1.0, theta))
    return np.concatenate([np.cos(theta), np.sin(theta) * direction], axis=-1)


def noisy_product(a, b, sigma, rng):
    """(a*b) * exp(xi), xi ~ N(0, sigma^2 I_3): heat-kernel-like positive."""
    xi = rng.standard_normal((a.shape[0], 3)) * sigma
    return qmul(qmul(a, b), expmap(xi))


def geodesic_negatives(prod, radius, k, rng):
    """[n, k, 4] negatives at fixed geodesic radius around each product."""
    n = prod.shape[0]
    u = rng.standard_normal((n, k, 3))
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return qmul(prod[:, None, :], expmap(radius * u))


# ------------------------------- Wigner D^j -----------------------------------

def su2_matrix(q):
    """Unit quaternion [..., 4] -> complex 2x2 SU(2) matrix [..., 2, 2]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    a = w + 1j * x
    b = y + 1j * z
    out = np.empty(q.shape[:-1] + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = a
    out[..., 0, 1] = b
    out[..., 1, 0] = -np.conj(b)
    out[..., 1, 1] = np.conj(a)
    return out


def wigner_D(j, q):
    """Wigner D^j for a batch of unit quaternions. Returns [N, 2j+1, 2j+1] complex.

    Row/column index k = 0..2j labels the monomial basis e_k above.
    """
    n = int(round(2 * j))
    g = su2_matrix(np.atleast_2d(q))
    a, b = g[:, 0, 0], g[:, 0, 1]
    c, d = g[:, 1, 0], g[:, 1, 1]
    N = g.shape[0]
    dim = n + 1
    D = np.empty((N, dim, dim), dtype=np.complex128)
    # binomial coefficient tables for (a u + c v)^(n-k) and (b u + d v)^k
    for k in range(dim):
        # poly1[i] = C(n-k, i) a^(n-k-i) c^i  (coefficient of u^{n-k-i} v^i)
        p = n - k
        i1 = np.arange(p + 1)
        poly1 = np.array([math.comb(p, i) for i in i1])[None, :] * \
            a[:, None] ** (p - i1)[None, :] * c[:, None] ** i1[None, :]
        i2 = np.arange(k + 1)
        poly2 = np.array([math.comb(k, i) for i in i2])[None, :] * \
            b[:, None] ** (k - i2)[None, :] * d[:, None] ** i2[None, :]
        # product polynomial: coefficient of u^{n-m} v^m in pi(g) e'_k
        conv = np.zeros((N, dim), dtype=np.complex128)
        for ii in range(p + 1):
            conv[:, ii: ii + k + 1] += poly1[:, ii: ii + 1] * poly2
        # orthonormalization: e_k = monomial / sqrt((n-k)! k!)
        norm_k = math.sqrt(math.factorial(p) * math.factorial(k))
        for m in range(dim):
            norm_m = math.sqrt(math.factorial(n - m) * math.factorial(m))
            D[:, m, k] = conv[:, m] * (norm_m / norm_k)
    return D


def wigner_character(j, q):
    """chi_j evaluated at quaternions q: sin((2j+1)theta)/sin(theta), theta = arccos(w)."""
    w = np.clip(np.atleast_2d(q)[..., 0] if q.ndim > 1 else q[..., 0], -1.0, 1.0)
    theta = np.arccos(w)
    n = int(round(2 * j))
    out = np.empty_like(theta)
    reg = np.abs(np.sin(theta)) > 1e-8
    out[reg] = np.sin((n + 1) * theta[reg]) / np.sin(theta[reg])
    # limits at theta -> 0, pi: (n+1) * cos(theta)^n
    out[~reg] = (n + 1) * np.sign(np.cos(theta[~reg])) ** n
    return out


def spin_basis(qs, j_list=(0.0, 0.5, 1.0, 1.5, 2.0)):
    """Orthonormal sampled Peter-Weyl bases for FVE spectroscopy.

    Args:
        qs: [N, 4] fixed probe sample (Haar).
    Returns:
        dict j -> Q [N, r] with orthonormal columns spanning the realified
        sampled span of {Re, Im D^j_{mk}}. FVE of activations Z (columns
        mean-centered) onto spin j is ||Q_j^T Z||_F^2 / ||Z||_F^2.
    """
    N = qs.shape[0]
    bases = {}
    prev = np.zeros((N, 0))
    for j in sorted(j_list):
        D = wigner_D(j, qs).reshape(N, -1)
        cols = np.concatenate([D.real, D.imag], axis=1)
        cols = cols - cols.mean(axis=0, keepdims=True)  # match centered activations
        # project out lower spins so the sampled subspaces are exactly orthogonal
        # (they are orthogonal only in Haar expectation; leakage would be ~r_j/N)
        if prev.shape[1]:
            cols = cols - prev @ (prev.T @ cols)
        keep = np.linalg.norm(cols, axis=0) > 1e-6 * math.sqrt(N)
        cols = cols[:, keep]
        if cols.shape[1] == 0:
            continue
        Qm, R = np.linalg.qr(cols)
        rank = np.abs(np.diag(R)) > 1e-9 * np.abs(R).max()
        bases[j] = Qm[:, rank]
        prev = np.concatenate([prev, bases[j]], axis=1)
    return bases


def spin_fve(Z, bases):
    """Fraction of (centered) variance of Z [N, d] in each spin-j subspace."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    total = (Zc ** 2).sum()
    return {j: float(((Q.T @ Zc) ** 2).sum() / total) for j, Q in bases.items()}


# ----------------------------- landmark encoding ------------------------------

class LandmarkEncoding:
    """Frozen observation map Psi(q) = k(<q, h_i>), i = 1..m.

    variant "su2": signed inner product (center-sensitive, faithful for SU(2)).
    variant "so3": absolute inner product (center-blind: Psi(q) == Psi(-q)).
    Features are standardized with frozen Haar statistics.
    """

    def __init__(self, m=32, alpha=3.0, variant="su2", seed=0):
        assert variant in ("su2", "so3")
        self.m, self.alpha, self.variant = m, alpha, variant
        rng = np.random.default_rng(seed)
        self.landmarks = haar(m, rng)                     # frozen, shared across variants
        probe = haar(65536, np.random.default_rng(seed + 1))
        raw = self._raw(probe)
        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0) + 1e-8

    def _raw(self, q):
        x = q @ self.landmarks.T                          # [..., m]
        if self.variant == "so3":
            x = np.abs(x)
        return np.exp(self.alpha * x)

    def __call__(self, q):
        return ((self._raw(q) - self.mean) / self.std).astype(np.float32)

    def torch_fn(self, device):
        import torch
        L = torch.tensor(self.landmarks.T, dtype=torch.float32, device=device)
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.std, dtype=torch.float32, device=device)

        def fn(q):
            x = q @ L
            if self.variant == "so3":
                x = x.abs()
            return (torch.exp(self.alpha * x) - mean) / std
        return fn


class RandMLPEncoding:
    """Frozen random two-layer tanh MLP on the raw quaternion (plan item A3).

    variant "su2": Psi(q) = tanh(W2 tanh(W1 q + b1) + b2).
    variant "so3": the center-symmetrized map (Psi(q) + Psi(-q)) / 2 -- the
    generic quotient construction (average over the center orbit).
    """

    def __init__(self, m=32, hidden=64, variant="su2", seed=0, alpha=None):
        assert variant in ("su2", "so3")
        self.m, self.variant = m, variant
        rng = np.random.default_rng(seed + 10)
        self.W1 = rng.standard_normal((4, hidden)) * 1.5
        self.b1 = rng.standard_normal(hidden) * 0.5
        self.W2 = rng.standard_normal((hidden, m)) * (1.5 / np.sqrt(hidden))
        self.b2 = rng.standard_normal(m) * 0.5
        probe = haar(65536, np.random.default_rng(seed + 11))
        raw = self._sym(probe)
        self.mean, self.std = raw.mean(0), raw.std(0) + 1e-8

    def _raw(self, q):
        return np.tanh(np.tanh(q @ self.W1 + self.b1) @ self.W2 + self.b2)

    def _sym(self, q):
        if self.variant == "so3":
            return (self._raw(q) + self._raw(-q)) / 2
        return self._raw(q)

    def __call__(self, q):
        return ((self._sym(q) - self.mean) / self.std).astype(np.float32)

    def torch_fn(self, device):
        import torch
        P = {k: torch.tensor(np.asarray(v), dtype=torch.float32, device=device)
             for k, v in [("W1", self.W1), ("b1", self.b1), ("W2", self.W2),
                          ("b2", self.b2), ("mean", self.mean), ("std", self.std)]}

        def raw(q):
            return torch.tanh(torch.tanh(q @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"])

        def fn(q):
            x = (raw(q) + raw(-q)) / 2 if self.variant == "so3" else raw(q)
            return (x - P["mean"]) / P["std"]
        return fn


class FuncValEncoding:
    """Psi(g)_i = f(g * x_i): values of a fixed generic function f on S^3 at
    translates of fixed probe points (plan item A3 -- 'the group is observed
    only through its action on an object'). f = random Fourier features.
    variant "so3": center-symmetrized."""

    def __init__(self, m=32, n_freq=64, bandwidth=2.0, variant="su2", seed=0,
                 alpha=None):
        assert variant in ("su2", "so3")
        self.m, self.variant = m, variant
        rng = np.random.default_rng(seed + 20)
        self.X = haar(m, rng)                              # probe points on S^3
        self.W = rng.standard_normal((4, n_freq)) * bandwidth
        self.phase = rng.uniform(0, 2 * np.pi, n_freq)
        self.c = rng.standard_normal(n_freq) / np.sqrt(n_freq)
        probe = haar(65536, np.random.default_rng(seed + 21))
        raw = self._sym(probe)
        self.mean, self.std = raw.mean(0), raw.std(0) + 1e-8

    def _raw(self, q):
        y = qmul(q[..., None, :], self.X)                  # [..., m, 4]
        return np.cos(y @ self.W + self.phase) @ self.c    # [..., m]

    def _sym(self, q):
        if self.variant == "so3":
            return (self._raw(q) + self._raw(-q)) / 2
        return self._raw(q)

    def __call__(self, q):
        return ((self._sym(q) - self.mean) / self.std).astype(np.float32)

    def torch_fn(self, device):
        import torch
        P = {k: torch.tensor(np.asarray(v), dtype=torch.float32, device=device)
             for k, v in [("X", self.X), ("W", self.W), ("ph", self.phase),
                          ("c", self.c), ("mean", self.mean), ("std", self.std)]}

        def tqmul(p, q):
            pw, px, py, pz = p.unbind(-1)
            qw, qx, qy, qz = q.unbind(-1)
            return torch.stack([
                pw * qw - px * qx - py * qy - pz * qz,
                pw * qx + px * qw + py * qz - pz * qy,
                pw * qy - px * qz + py * qw + pz * qx,
                pw * qz + px * qy - py * qx + pz * qw], dim=-1)

        def raw(q):
            y = tqmul(q[..., None, :], P["X"])
            return torch.cos(y @ P["W"] + P["ph"]) @ P["c"]

        def fn(q):
            # The encoding is frozen (no parameters, constant w.r.t. the model),
            # so compute without autograd and in chunks: the [.., m, n_freq]
            # intermediate would otherwise dominate GPU memory.
            with torch.no_grad():
                flat = q.reshape(-1, 4)
                outs = []
                for i in range(0, flat.shape[0], 16384):
                    ch = flat[i:i + 16384]
                    x = ((raw(ch) + raw(-ch)) / 2 if self.variant == "so3"
                         else raw(ch))
                    outs.append((x - P["mean"]) / P["std"])
                return torch.cat(outs).reshape(*q.shape[:-1], self.m)
        return fn


class ExplicitFeatureEncoding:
    """P1B: exact representation features as the observation (no scramble).

    name "so3frame": two columns of R(q), 6 real dims -- a faithful frame,
    left-linear but not right-linear-closed. name "so3block": all 9 entries of
    R(q) -- the full real coefficient block. Both are even in q (SO(3)
    features). Standardized with Haar statistics.
    """

    def __init__(self, name, seed=0):
        assert name in ("so3frame", "so3block")
        self.name = name
        self.m = 6 if name == "so3frame" else 9
        probe = haar(65536, np.random.default_rng(seed + 90))
        raw = self._raw(probe)
        self.mean, self.std = raw.mean(0), raw.std(0) + 1e-8

    @staticmethod
    def _rot(q):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                      2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                      2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                      1 - 2 * (x * x + y * y)], -1)], -2)

    def _raw(self, q):
        R = self._rot(q)
        if self.name == "so3frame":
            return R[..., :, :2].reshape(*q.shape[:-1], 6)
        return R.reshape(*q.shape[:-1], 9)

    def __call__(self, q):
        return ((self._raw(q) - self.mean) / self.std).astype(np.float32)

    def torch_fn(self, device):
        import torch
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.std, dtype=torch.float32, device=device)

        def fn(q):
            w, x, y, z = q.unbind(-1)
            R = torch.stack([
                torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                             2 * (x * z + y * w)], -1),
                torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                             2 * (y * z - x * w)], -1),
                torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                             1 - 2 * (x * x + y * y)], -1)], -2)
            if self.name == "so3frame":
                raw = R[..., :, :2].reshape(*q.shape[:-1], 6)
            else:
                raw = R.reshape(*q.shape[:-1], 9)
            return (raw - mean) / std
        return fn


def get_encoding(name, variant, m=32, alpha=3.0, seed=0):
    if name == "landmark":
        return LandmarkEncoding(m=m, alpha=alpha, variant=variant, seed=seed)
    if name == "randmlp":
        return RandMLPEncoding(m=m, variant=variant, seed=seed)
    if name == "funcval":
        return FuncValEncoding(m=m, variant=variant, seed=seed)
    if name in ("so3frame", "so3block"):
        assert variant == "so3", "explicit SO(3) features are even in q"
        return ExplicitFeatureEncoding(name, seed=seed)
    raise ValueError(name)


def heat_kernel(q, t, log=False, tol=1e-7):
    """Heat kernel K_t on SU(2) at group elements q (relative to identity):
    K_t(g) = sum_j (2j+1) e^{-t j(j+1)} chi_j(g), j = 0, 1/2, 1, ...

    Truncated when the weight (2j+1)e^{-t j(j+1)} falls below tol; values are
    positive up to O(tol) truncation error (the tail near theta = pi is
    exponentially small and the alternating series can dip slightly below 0)."""
    out = np.zeros(np.atleast_2d(q).shape[0] if q.ndim > 1 else 1)
    j = 0.0
    while True:
        w = (2 * j + 1) * np.exp(-t * j * (j + 1))
        if w < tol and j > 0:
            break
        out = out + w * wigner_character(j, q)
        j += 0.5
    return np.log(np.maximum(out, 1e-30)) if log else out


# --------------------------------- self test ----------------------------------

def _self_test():
    rng = np.random.default_rng(0)
    q1, q2 = haar(64, rng), haar(64, rng)
    for j in (0.5, 1.0, 1.5, 2.0):
        D1, D2 = wigner_D(j, q1), wigner_D(j, q2)
        D12 = wigner_D(j, qmul(q1, q2))
        hom = np.abs(np.einsum("nij,njk->nik", D1, D2) - D12).max()
        eye = np.eye(int(round(2 * j)) + 1)
        uni = np.abs(np.einsum("nij,nkj->nik", D1, D1.conj()) - eye).max()
        chi = np.abs(np.trace(D1, axis1=1, axis2=2).real - wigner_character(j, q1)).max()
        assert hom < 1e-10 and uni < 1e-10 and chi < 1e-8, (j, hom, uni, chi)
        # center: D^j(-q) = (-1)^{2j} D^j(q)
        sign = (-1) ** int(round(2 * j))
        cen = np.abs(wigner_D(j, -q1) - sign * D1).max()
        assert cen < 1e-10, (j, cen)
    # FVE machinery: activations built from D^1 entries land fully in j=1
    probe = haar(2048, rng)
    bases = spin_basis(probe)
    Z = wigner_D(1.0, probe).reshape(2048, -1).real @ rng.standard_normal((9, 16))
    fve = spin_fve(Z, bases)
    # cross-spin leakage floor is ~r_j/N (finite Monte-Carlo sample), here ~2e-3
    assert fve[1.0] > 0.995 and fve[0.5] < 5e-3 and fve[2.0] < 5e-3, fve
    # encodings: so3 variant is even, su2 variant is not (and is well-conditioned at +-q)
    qs = haar(512, rng)
    for name in ("landmark", "randmlp", "funcval"):
        e_su2 = get_encoding(name, "su2")
        e_so3 = get_encoding(name, "so3")
        assert np.abs(e_so3(qs) - e_so3(-qs)).max() < 1e-5, name
        gap = np.linalg.norm(e_su2(qs) - e_su2(-qs), axis=1)
        assert gap.min() > 0.5, (name, gap.min())
        # torch mirror agrees with numpy
        import torch
        tv = e_su2.torch_fn("cpu")(torch.tensor(qs, dtype=torch.float32)).numpy()
        assert np.abs(tv - e_su2(qs)).max() < 1e-4, name
    # heat kernel: positive, peaked at identity, class function (depends on w)
    K = heat_kernel(qs, t=0.1)
    assert K.min() > -1e-4          # positive up to truncation error
    assert heat_kernel(np.array([[1.0, 0, 0, 0]]), t=0.1)[0] > K.max()
    print("core.su2 self-test OK")


if __name__ == "__main__":
    _self_test()
