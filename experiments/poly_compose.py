"""Handoff 4.4: controlled polynomial composition experiment on SO(3).

Why degree 3. Write R(a) = [u, v, u x v] with u, v the two frame columns.
For any column b_c of R(b), the product column is

    R(ab) e_c = R(a) b_c = b_cx u + b_cy v + b_cz (u x v).

The first two terms are bilinear in (frame(a), frame(b)). The third needs
u x v, which is quadratic in frame(a) alone, so the full map is degree 2
in frame(a) and degree 1 in frame(b): degree 3 jointly. A bilinear map of
(frame(a), frame(b)) can never produce the b_cz (u x v) component: under
Haar the third column of R(a) is uncorrelated with the first two
(E[R_ij R_kl] = delta_ik delta_jl / 3), so the best bilinear approximation
drops that component entirely, an irreducible error floor of about
E[b_cz^2] * E[|u x v|^2] / E[|R(ab) e_c|^2] = (1/3) / 1 per column.

Arms (exact standardized features, scale sqrt(3), zero mean by symmetry):
    A bilinear-full   9x9 -> 9 tensor, exact for matrix multiplication;
    B bilinear-frame  6x6 -> 6 tensor, best degree-2 frame interaction;
    C degree3-frame   symmetric-quadratic layer frame(a) -> w, then
                      bilinear ([u, v, w], frame(b)) -> 6, exact.

Shared scorer: temperature * cosine(interaction output, candidate
features). The scorer is exactly invariant to the global scale of each
final bilinear tensor, which is the declared canonicalization (unit
Frobenius tensors, scale folded into the temperature).

Step 1 fits each arm by closed-form least squares to the true product
features (arm C in two closed-form stages) and reports held-out relative
MSE. Step 2 trains all arms on the actual ranking distribution.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.envelope import make_envelope, write_analysis
from core.su2 import haar, qmul
from experiments.su2_train import make_pair_set

SQRT3 = np.sqrt(3.0)
TRI = list(zip(*[(i, j) for i in range(6) for j in range(i, 6)]))  # 21 pairs

DEGREES = {"A": {"joint": 2, "in_a": 1, "in_b": 1},
           "B": {"joint": 2, "in_a": 1, "in_b": 1},
           "C": {"joint": 3, "in_a": 2, "in_b": 1}}
MULT_COUNTS = {
    "A": {"parameters": 729, "naive_bilinear": 81 + 729,
          "structured_matmul_realization": 27},
    "B": {"parameters": 216, "naive_bilinear": 36 + 216},
    "C": {"parameters": 63 + 324,
          "naive": 21 + 63 + 54 + 324}}


def rot(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)], -1)], -2)


def feat9(q):
    """Exact standardized block features: sqrt(3) * vec R(q)."""
    return (SQRT3 * rot(q).reshape(*q.shape[:-1], 9)).astype(np.float64)


def feat6(q):
    """Exact standardized frame features: sqrt(3) * vec R(q)[:, :2]."""
    return (SQRT3 * rot(q)[..., :, :2].reshape(*q.shape[:-1], 6)
            ).astype(np.float64)


def third_column_std(q):
    """Standardized third column sqrt(3) * R(q)[:, 2]."""
    return (SQRT3 * rot(q)[..., :, 2]).astype(np.float64)


def quad_features(f6):
    """Symmetric quadratic features s_i s_j, i <= j (21 per sample)."""
    return f6[..., TRI[0]] * f6[..., TRI[1]]


def _lstsq(X, Y):
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def _rel_mse(X, W, Y):
    R = X @ W - Y
    return float((R ** 2).sum() / (Y ** 2).sum())


def exactness(n=20000, seed=0):
    """Step 1: closed-form least squares, held-out relative MSE per arm."""
    rng = np.random.default_rng(seed)
    out = {}
    a, b = haar(2 * n, rng), haar(2 * n, rng)
    ab = qmul(a, b)
    tr, te = slice(0, n), slice(n, 2 * n)

    def bil(fa, fb):
        return np.einsum("ni,nj->nij", fa, fb).reshape(fa.shape[0], -1)

    # arm A
    X, Y = bil(feat9(a), feat9(b)), feat9(ab)
    W = _lstsq(X[tr], Y[tr])
    out["A"] = {"rel_mse_holdout": _rel_mse(X[te], W, Y[te])}

    # arm B
    X, Y = bil(feat6(a), feat6(b)), feat6(ab)
    W = _lstsq(X[tr], Y[tr])
    out["B"] = {"rel_mse_holdout": _rel_mse(X[te], W, Y[te])}

    # arm C: stage 1 quadratic layer to w, stage 2 bilinear with fitted w
    fa6 = feat6(a)
    Q = _lstsq(quad_features(fa6[tr]), third_column_std(a)[tr])
    w_all = quad_features(fa6) @ Q
    stage1 = _rel_mse(quad_features(fa6[te]), Q, third_column_std(a)[te])
    X = bil(np.concatenate([fa6, w_all], -1), feat6(b))
    W = _lstsq(X[tr], Y[tr])
    out["C"] = {"rel_mse_holdout": _rel_mse(X[te], W, Y[te]),
                "stage1_rel_mse": stage1}
    out["n_train"] = out["n_holdout"] = n
    out["fit_method"] = "closed-form numpy lstsq (arm C in two stages)"
    return out


class Arm(torch.nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind = kind
        if kind == "A":
            self.T = torch.nn.Parameter(torch.randn(9, 9, 9, generator=g)
                                        / 9.0)
        elif kind == "B":
            self.T = torch.nn.Parameter(torch.randn(6, 6, 6, generator=g)
                                        / 6.0)
        else:
            self.Q = torch.nn.Parameter(torch.randn(3, 21, generator=g)
                                        / 21.0)
            self.T = torch.nn.Parameter(torch.randn(6, 9, 6, generator=g)
                                        / 7.0)
        self.log_temp = torch.nn.Parameter(torch.tensor(np.log(10.0),
                                                        dtype=torch.float32))

    def interaction(self, fa, fb):
        if self.kind == "C":
            quad = fa[:, TRI[0]] * fa[:, TRI[1]]
            w = quad @ self.Q.T
            fa = torch.cat([fa, w], -1)
        return torch.einsum("kij,bi,bj->bk", self.T, fa, fb)

    def forward(self, fa, fb, fc):
        h = F.normalize(self.interaction(fa, fb), dim=-1)
        z = F.normalize(fc, dim=-1)
        return self.log_temp.exp() * torch.einsum("bd,bkd->bk", h, z)


def features_torch(kind, q):
    f = feat9 if kind == "A" else feat6
    return torch.tensor(f(q), dtype=torch.float32)


def train_arm(kind, seed, steps=20000, batch=1024, log_every=500):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pool = haar(2048, rng)
    M = int(round(0.01 * 2048 * 2048))
    idx = rng.choice(2048 * 2048, size=M, replace=False)
    qa, qb = pool[idx // 2048], pool[idx % 2048]
    qa, qb, cands = make_pair_set(qa, qb, 64, 0.2, 0.5, "so3", rng)
    t2a, t2b, t2c = make_pair_set(haar(2048, rng), haar(2048, rng),
                                  64, 0.2, 0.5, "so3", rng)
    fa = features_torch(kind, qa)
    fb = features_torch(kind, qb)
    fc = features_torch(kind, cands)
    ta, tb, tc = (features_torch(kind, x) for x in (t2a, t2b, t2c))

    model = Arm(kind, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    traj = []
    perm, ptr = torch.randperm(M), 0
    for step in range(1, steps + 1):
        if ptr + batch > M:
            perm, ptr = torch.randperm(M), 0
        sel = perm[ptr:ptr + batch]
        ptr += batch
        lg = model(fa[sel], fb[sel], fc[sel])
        loss = F.cross_entropy(lg, torch.zeros(batch, dtype=torch.long))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                lg2 = model(ta, tb, tc)
                acc = float((lg2.argmax(1) == 0).float().mean())
                margin = float((lg2[:, 0]
                                - lg2[:, 1:].max(1).values).mean())
                rec = {"step": step, "loss": float(loss),
                       "acc_T2": acc, "margin_T2": margin,
                       "norm_T": float(model.T.norm()),
                       "temperature": float(model.log_temp.exp())}
                if kind == "C":
                    rec["norm_Q"] = float(model.Q.norm())
                traj.append(rec)
    return traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--exact-n", type=int, default=20000)
    p.add_argument("--arms", default="ABC")
    p.add_argument("--skip-exact", action="store_true")
    p.add_argument("--out", default="summary/poly_compose.json")
    args = p.parse_args()

    t0 = time.time()
    ex = None
    if not args.skip_exact:
        ex = exactness(n=args.exact_n)
        print("exactness (held-out rel MSE):",
              {k: v for k, v in ex.items() if k in "ABC"}, flush=True)

    arms = {}
    for kind, seed in itertools.product(args.arms, args.seeds):
        traj = train_arm(kind, seed, steps=args.steps)
        arms.setdefault(kind, {})[str(seed)] = traj
        f = traj[-1]
        print(f"arm {kind} seed {seed}: acc {f['acc_T2']:.4f} "
              f"margin {f['margin_T2']:.4f} temp {f['temperature']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    finals = {k: {s: t[-1] for s, t in v.items()} for k, v in arms.items()}
    mean = {k: {m: float(np.mean([f[m] for f in v.values()]))
                for m in ("acc_T2", "margin_T2", "temperature")}
            for k, v in finals.items()}
    verdict = None
    if ex is not None and set("ABC") <= set(arms):
        b_floor = ex["B"]["rel_mse_holdout"]
        verdict = (
            "in this explicit model class, the frame saves coordinates but "
            "exact closed composition requires degree 3, and the best "
            "degree-2 frame interaction has irreducible composition error "
            f"{b_floor:.4f} (relative MSE); trained at matched scale the "
            f"exact-degree arms attain margins "
            f"A={mean['A']['margin_T2']:.4f} / "
            f"C={mean['C']['margin_T2']:.4f} "
            f"vs B's {mean['B']['margin_T2']:.4f}")
    results = {
        "exactness": ex,
        "degrees": DEGREES,
        "multiplication_counts": MULT_COUNTS,
        "scale_normalization": (
            "the scorer is exactly invariant to the global scale of each "
            "final bilinear tensor (cosine normalization); canonical form "
            "rescales tensors to unit Frobenius and folds scale into the "
            "temperature, so canonical margins equal raw margins; raw "
            "norms are parameterization-dependent and reported as such"),
        "per_seed_final": finals,
        "arm_means": mean,
        "trajectories": arms,
        "verdict": verdict,
    }
    env = make_envelope("poly-compose", 1, analysis_file=__file__,
                        analysis_args=vars(args), analysis_seed=0)
    write_analysis(args.out, results, env)
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
