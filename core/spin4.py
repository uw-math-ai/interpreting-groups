"""Spin(4) = SU(2) x SU(2) and its central quotients (plan F + C).

Elements are pairs of unit quaternions g = (qL, qR), stored [n, 2, 4]. The
center is Z2 x Z2 = {(+-1, +-1)}; irreps are (jL, jR) with dimension
(2jL+1)(2jR+1) and central character ((-1)^{2jL}, (-1)^{2jR}). Quotients by
central subgroups:

    variant "spin4":  Z = 1                       (all sectors observable)
    variant "so4":    Z = {(1,1), (-1,-1)}        (jL + jR integer descends)
    variant "leftq":  Z = {(1,1), (-1,1)}         (jL integer descends;
                                                   the group SO(3) x SU(2))
    variant "pso4":   Z = Z2 x Z2                 (jL and jR both integer)

Since Z2 x Z2 is not cyclic, Spin(4) has NO faithful irrep: any model of the
"spin4" condition must carry at least two isotypic blocks. Minimal faithful
choices, recorded as competing hypotheses BEFORE running (dims realified):
    spin4:  (1/2, 0) + (0, 1/2)          [4 + 4 dims, two blocks; forced sum]
    so4:    (1/2, 1/2)                   [the 4-dim vector rep, faithful]
    leftq:  (1, 0) + (0, 1/2)  [5 dims, two blocks]   vs   (1, 1/2) [6, one]
    pso4:   (1, 0) + (0, 1)    [6 dims, two blocks]   vs   (1, 1)   [9, one]
The leftq/pso4 selections trade total dimension against block count; which one
optimization prefers is the discovery question.
"""
import itertools
import math

import numpy as np

from core.su2 import haar, qmul, wigner_D


def haar_spin4(n, rng):
    return np.stack([haar(n, rng), haar(n, rng)], axis=1)


def mul(g, h):
    return np.stack([qmul(g[..., 0, :], h[..., 0, :]),
                     qmul(g[..., 1, :], h[..., 1, :])], axis=-2)


CENTER = {  # sign pairs (sL, sR) of the quotiented central subgroup
    "spin4": [(1, 1)],
    "so4": [(1, 1), (-1, -1)],
    "leftq": [(1, 1), (-1, 1)],
    "pso4": [(1, 1), (-1, -1), (-1, 1), (1, -1)],
}


def class_distance(c, p, variant):
    """Geodesic product-metric distance from c to the center orbit of p."""
    ds = []
    for sL, sR in CENTER[variant]:
        dotL = np.clip(np.sum(c[..., 0, :] * p[..., 0, :] * sL, -1), -1, 1)
        dotR = np.clip(np.sum(c[..., 1, :] * p[..., 1, :] * sR, -1), -1, 1)
        ds.append(np.sqrt(np.arccos(dotL) ** 2 + np.arccos(dotR) ** 2))
    return np.min(np.stack(ds), axis=0)


def descends(sector, variant):
    """Central character of (jL, jR) at (sL, sR) is sL^{2jL} sR^{2jR}."""
    jL, jR = sector
    return all(sL ** round(2 * jL) * sR ** round(2 * jR) == 1
               for sL, sR in CENTER[variant])


SECTORS4 = [(a / 2, b / 2) for a, b in itertools.product(range(4), repeat=2)]


def sector_basis4(gs, sectors=tuple(SECTORS4[1:])):
    """Sampled orthonormal bases of the (jL, jR) sectors of functions on
    SU(2) x SU(2), joint QR in order of total Casimir."""
    N = len(gs)
    qL, qR = gs[:, 0], gs[:, 1]
    DL = {j: wigner_D(j, qL).reshape(N, -1) for j in {s[0] for s in sectors}}
    DR = {j: wigner_D(j, qR).reshape(N, -1) for j in {s[1] for s in sectors}}
    bases, prev = {}, np.zeros((N, 0))
    key = lambda s: (s[0] * (s[0] + 1) + s[1] * (s[1] + 1), s)
    for s in sorted(sectors, key=key):
        A = DL[s[0]] if s[0] > 0 else np.ones((N, 1), dtype=complex)
        B = DR[s[1]] if s[1] > 0 else np.ones((N, 1), dtype=complex)
        cols = np.einsum("ni,nj->nij", A, B).reshape(N, -1)
        cols = np.concatenate([cols.real, cols.imag], axis=1)
        cols = cols - cols.mean(0, keepdims=True)
        if prev.shape[1]:
            cols = cols - prev @ (prev.T @ cols)
        keep = np.linalg.norm(cols, axis=0) > 1e-6 * math.sqrt(N)
        cols = cols[:, keep]
        if cols.shape[1] == 0:
            continue
        Qm, R = np.linalg.qr(cols)
        rank = np.abs(np.diag(R)) > 1e-9 * np.abs(R).max()
        bases[s] = Qm[:, rank]
        prev = np.concatenate([prev, bases[s]], axis=1)
    return bases


class Spin4LandmarkEncoding:
    """Psi(g) = exp(alpha(<qL, hL_i> + <qR, hR_i>)), i = 1..m -- a generic
    non-separable observation; quotient variants average over the center
    orbit of the quotiented subgroup."""

    def __init__(self, m=48, alpha=2.0, variant="spin4", seed=0):
        assert variant in CENTER
        self.m, self.alpha, self.variant = m, alpha, variant
        rng = np.random.default_rng(seed + 60)
        self.HL, self.HR = haar(m, rng), haar(m, rng)
        probe = haar_spin4(65536, np.random.default_rng(seed + 61))
        raw = self._sym(probe)
        self.mean, self.std = raw.mean(0), raw.std(0) + 1e-8

    def _raw(self, g):
        x = g[..., 0, :] @ self.HL.T + g[..., 1, :] @ self.HR.T
        return np.exp(self.alpha * x)

    def _sym(self, g):
        acc = 0
        for sL, sR in CENTER[self.variant]:
            gs = g.copy()
            gs[..., 0, :] *= sL
            gs[..., 1, :] *= sR
            acc = acc + self._raw(gs)
        return acc / len(CENTER[self.variant])

    def __call__(self, g):
        return ((self._sym(g) - self.mean) / self.std).astype(np.float32)

    def torch_fn(self, device):
        import torch
        HL = torch.tensor(self.HL.T, dtype=torch.float32, device=device)
        HR = torch.tensor(self.HR.T, dtype=torch.float32, device=device)
        mean = torch.tensor(self.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.std, dtype=torch.float32, device=device)
        signs = CENTER[self.variant]

        def fn(g):
            acc = 0
            for sL, sR in signs:
                x = (sL * g[..., 0, :]) @ HL + (sR * g[..., 1, :]) @ HR
                acc = acc + torch.exp(self.alpha * x)
            return (acc / len(signs) - mean) / std
        return fn


def _self_test():
    rng = np.random.default_rng(0)
    g = haar_spin4(256, rng)
    # descent rules
    assert descends((0.5, 0.5), "so4") and not descends((0.5, 0), "so4")
    assert descends((1, 0.5), "leftq") and not descends((0.5, 0.5), "leftq")
    assert descends((1, 1), "pso4") and not descends((1, 0.5), "pso4")
    # sector FVE recovers planted content
    bases = sector_basis4(g, sectors=((0.5, 0), (0, 0.5), (0.5, 0.5), (1, 0)))
    Z = np.concatenate([wigner_D(0.5, g[:, 0]).reshape(256, -1).real,
                        wigner_D(0.5, g[:, 1]).reshape(256, -1).imag], axis=1) \
        @ rng.standard_normal((8, 16))
    Zc = Z - Z.mean(0)
    tot = (Zc ** 2).sum()
    fr = {s: ((Q.T @ Zc) ** 2).sum() / tot for s, Q in bases.items()}
    assert fr[(0.5, 0)] + fr[(0, 0.5)] > 0.99, fr
    # encodings: invariance exactly on the quotiented subgroup, not beyond
    import torch
    for variant, dead, alive in [("so4", (-1, -1), (-1, 1)),
                                 ("leftq", (-1, 1), (1, -1)),
                                 ("pso4", (1, -1), None)]:
        e = Spin4LandmarkEncoding(m=8, variant=variant)
        gd = g.copy(); gd[..., 0, :] *= dead[0]; gd[..., 1, :] *= dead[1]
        assert np.abs(e(g) - e(gd)).max() < 1e-4, variant
        if alive:
            ga = g.copy(); ga[..., 0, :] *= alive[0]; ga[..., 1, :] *= alive[1]
            assert np.linalg.norm(e(g) - e(ga), axis=1).min() > 0.3, variant
        tv = e.torch_fn("cpu")(torch.tensor(g, dtype=torch.float32)).numpy()
        assert np.abs(tv - e(g)).max() < 1e-3, variant
    print("core.spin4 self-test OK")


if __name__ == "__main__":
    _self_test()
