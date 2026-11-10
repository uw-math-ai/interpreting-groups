"""P1A: two-sided latent action test (mechanism discrimination).

For a feature system phi: G -> R^d, fit affine operators
    M^L_k : phi(g) -> phi(kg)      and      M^R_k : phi(g) -> phi(gk)
by ridge on a probe split and evaluate held out. Conventions (matrices acting
on feature vectors): M^L_{ab} ~= M^L_a M^L_b (homomorphism) and
M^R_{ab} ~= M^R_b M^R_a (anti-homomorphism); left and right regular actions
commute, so [M^L_a, M^R_b] ~= 0 exactly for a full coefficient block.

Feature systems compared (plan P1A):
    trained      dominant learned block of a trained encoder (spin-block frame)
    random_init  same architecture, untrained, PCA frame
    raw_obs      the observation features Psi(g)
    frame        explicit faithful frame -- SO(3): two columns of R(q), 6 real
                 dims (left-linear, right-nonlinear); SU(2): the quaternion q,
                 4 dims -- which by quaternionic Schur type IS the full real
                 block, so it is predicted to PASS two-sided closure
    true_block   full real coefficient block -- SO(3): all 9 entries of R(q);
                 SU(2): q (coincides with frame)
    rand_trunc   random projection of the true block to the frame's dimension
                 (SO(3): 6 of 9) -- predicted to fail BOTH closures

Interpretive role per the v2 plan: spectroscopy already locates the learned
code in a Peter-Weyl sector; this test asks whether it has the exact
structural property (two-sided linear closure) that distinguishes full
coefficient blocks from smaller one-sided faithful frames.

Usage:
    uv run python -m experiments.twosided --run_dir data/su2/su2_N2048_f0.01_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.su2 import haar, qconj, qmul
from core.su2_model import TripleScorer
from experiments.su2_causal import load_model
from experiments.su2_mechanism import PCAFrame, embed, spin_block_frame


def rot_matrix(q):
    """[n,4] -> [n,3,3] rotation matrices (vectorized)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)], -1)], -2)


def ridge_affine(X, Y, lam=1e-2):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, my - mx @ W


def r2(pred, true):
    return float(1 - ((true - pred) ** 2).sum()
                 / ((true - true.mean(0)) ** 2).sum())


def feature_systems(model, enc, cfg, device, g_ref, rng):
    variant = cfg["variant"]
    out = {}

    frame_tr = None
    if model is not None:
        Zr = embed(model, enc, g_ref, device)
        B = spin_block_frame(Zr, g_ref, 0.5 if variant == "su2" else 1.0)
        mu = Zr.mean(0)
        out["trained"] = lambda q: (embed(model, enc, q, device) - mu) @ B

        rmodel = TripleScorer(m=cfg["m"]).to(device).eval()
        torch.manual_seed(cfg["seed"] + 999)
        for p in rmodel.parameters():
            torch.nn.init.normal_(p, std=0.2)
        Zr2 = embed(rmodel, enc, g_ref, device)
        fr = PCAFrame(Zr2, cap=16)
        out["random_init"] = lambda q: fr(embed(rmodel, enc, q, device))

    out["raw_obs"] = lambda q: enc(q).astype(np.float64)

    if variant == "su2":
        out["frame"] = lambda q: q.copy()          # = full real block (H type)
        out["true_block"] = lambda q: q.copy()
        P = np.linalg.qr(rng.standard_normal((4, 3)))[0]
        out["rand_trunc"] = lambda q: q @ P
    else:
        out["frame"] = lambda q: rot_matrix(q)[:, :, :2].reshape(len(q), 6)
        out["true_block"] = lambda q: rot_matrix(q).reshape(len(q), 9)
        P = np.linalg.qr(rng.standard_normal((9, 6)))[0]
        out["rand_trunc"] = lambda q: rot_matrix(q).reshape(len(q), 9) @ P
    return out


def analyze(phi, g_tr, g_te, ks, rng):
    F_tr, F_te = phi(g_tr), phi(g_te)
    Ls, Rs = {}, {}
    lr2, rr2 = [], []
    for i, k in enumerate(ks):
        WL, bL = ridge_affine(F_tr, phi(qmul(k[None], g_tr)))
        WR, bR = ridge_affine(F_tr, phi(qmul(g_tr, k[None])))
        Ls[i], Rs[i] = (WL, bL), (WR, bR)
        lr2.append(r2(F_te @ WL + bL, phi(qmul(k[None], g_te))))
        rr2.append(r2(F_te @ WR + bR, phi(qmul(g_te, k[None]))))

    Fc = F_te - F_te.mean(0)
    nrm = np.linalg.norm(Fc)

    hom_l, hom_r, comm = [], [], []
    for _ in range(12):
        i, j = rng.integers(0, len(ks), 2)
        a, b = ks[i], ks[j]
        WLa, bLa = Ls[i]
        WLb, bLb = Ls[j]
        WLab, bLab = ridge_affine(F_tr, phi(qmul(qmul(a, b)[None], g_tr)))
        lhs = (F_te @ WLb + bLb) @ WLa + bLa
        hom_l.append(float(np.linalg.norm(lhs - (F_te @ WLab + bLab)) / nrm))
        WRa, bRa = Rs[i]
        WRb, bRb = Rs[j]
        WRab, bRab = ridge_affine(F_tr, phi(qmul(g_tr, qmul(a, b)[None])))
        rhs = (F_te @ WRa + bRa) @ WRb + bRb       # M^R_ab ~ M^R_b M^R_a
        hom_r.append(float(np.linalg.norm(rhs - (F_te @ WRab + bRab)) / nrm))
        one = (F_te @ WLa + bLa) @ WRb + bRb
        two = (F_te @ WRb + bRb) @ WLa + bLa
        comm.append(float(np.linalg.norm(one - two) / nrm))

    return {"dim": F_tr.shape[1],
            "left_r2": float(np.mean(lr2)), "right_r2": float(np.mean(rr2)),
            "hom_left": float(np.mean(hom_l)),
            "hom_right": float(np.mean(hom_r)),
            "commutator": float(np.mean(comm))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n", type=int, default=6144)
    p.add_argument("--n_k", type=int, default=12)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, enc, cfg, step = load_model(args.run_dir, device=args.device)
    print(f"== {args.run_dir} ({cfg['variant']}, step {step})")
    rng = np.random.default_rng(29)
    g = haar(args.n, rng)
    g_tr, g_te = g[:5120], g[5120:]
    g_ref = haar(4096, rng)
    ks = haar(args.n_k, rng)

    systems = feature_systems(model, enc, cfg, args.device, g_ref, rng)
    res = {"variant": cfg["variant"], "step": step, "systems": {}}
    hdr = (f"{'system':<12} {'dim':>4} {'L_r2':>7} {'R_r2':>7} "
           f"{'homL':>7} {'homR':>7} {'[L,R]':>7}")
    print(hdr)
    for name, phi in systems.items():
        m = analyze(phi, g_tr, g_te, ks, rng)
        res["systems"][name] = m
        print(f"{name:<12} {m['dim']:>4} {m['left_r2']:>7.4f} "
              f"{m['right_r2']:>7.4f} {m['hom_left']:>7.4f} "
              f"{m['hom_right']:>7.4f} {m['commutator']:>7.4f}")

    out = Path(args.run_dir) / f"twosided_{step}.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
