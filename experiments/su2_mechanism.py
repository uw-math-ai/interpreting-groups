"""Mechanistic loop-closure tests for SU(2)/SO(3) composition models.

Three analyses, runnable per checkpoint (plan items A1/A2/A4):

1. --test action (A1): latent group-action test. For fixed k, fit an affine map
   T_k(z) = W_k z + b_k with z(kg) ≈ T_k(z(g)) on a probe set (ridge). Report,
   on held-out g: equivariance R^2; homomorphism defect ||T_k(T_h(z)) − T_{kh}(z)||;
   identity defect ||W_e − I||; inverse defect ||T_k(T_{k^-1}(z)) − z||; and the
   comparison of the top eigenvalue phases of W_k with the representation-theoretic
   prediction (left translation acts on the spin-j matrix-coefficient block by
   d_j copies of D^j(k): eigenphases ±θ_k for spin-1/2, {0, ±2θ_k} for spin-1,
   where θ_k = arccos(w_k) is the quaternion half-angle).

2. --test product (A2): product-state analysis. Spin spectroscopy of h(a,b) as a
   function of the product ab (FVE per spin + residual = dependence beyond ab);
   inference-time function-ablation of the D^{1/2}(ab)-predictable component of h;
   and two patching tests: full patch h(a,b) <- h(a',b') (redirection ceiling) and
   fit patch h <- ridge reconstruction from D^{1/2}(a'b') (is the spin-1/2
   parametrization of h causally sufficient to steer the output?), with a D^1
   reconstruction control.

3. --test stitch (A4): cross-seed intertwiners. Affine stitch z_s'(g) ≈ A z_s(g),
   held-out R^2; intertwining defect ||A T^s_k − T^{s'}_k A|| for shared k
   (a representation-theoretic statement of seed universality).

Usage:
    uv run python -m experiments.su2_mechanism --run_dir data/su2/su2_N2048_f0.01_s0
    uv run python -m experiments.su2_mechanism --cell su2_N2048_f0.01 --test stitch
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.su2 import expmap, haar, qconj, qmul, spin_basis, wigner_D
from experiments.su2_causal import load_model

DATA = Path(__file__).resolve().parent.parent / "data/su2"


# ------------------------------ shared helpers --------------------------------

def embed(model, enc, q, device, batch=16384):
    out = []
    with torch.no_grad():
        for i in range(0, len(q), batch):
            out.append(model.embed(
                torch.tensor(enc(q[i:i + batch]), device=device)).cpu().numpy())
    return np.concatenate(out)


def ridge_affine(X, Y, lam=1e-2):
    """Y ≈ X W + b. Returns (W, b)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, my - mx @ W


def r2(pred, true):
    return float(1 - ((true - pred) ** 2).sum()
                 / ((true - true.mean(0)) ** 2).sum())


# ------------------------------- A1: group action -----------------------------

class PCAFrame:
    """Orthonormal r-dim frame capturing var_frac of the (centered) activations.
    All A1/A4 fits happen in these coordinates: without the reduction, near-dead
    activation directions make the ridge fits ill-conditioned."""

    def __init__(self, Z, var_frac=0.999, cap=64):
        self.mu = Z.mean(0)
        U, S, Vt = np.linalg.svd(Z - self.mu, full_matrices=False)
        r = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), var_frac)) + 1
        self.V = Vt[:min(r, cap)].T

    def __call__(self, Z):
        return (Z - self.mu) @ self.V


def fit_T(model, enc, k, g_tr, Y_tr, frame, device):
    """Affine T_k in PCA coords: frame(z(kg)) ≈ frame(z(g)) W + b."""
    Zk = frame(embed(model, enc, qmul(k[None, :], g_tr), device))
    return ridge_affine(Y_tr, Zk)


def spin_block_frame(Z, probe_q, j):
    """Orthonormal activation-space frame of the sampled spin-j block."""
    Q = spin_basis(probe_q, j_list=(j,))[j]
    C = Q.T @ (Z - Z.mean(0))
    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), 0.99)) + 1
    return Vt[:k].T


def test_action(model, enc, cfg, device, n=6144, n_k=16, seed=3):
    rng = np.random.default_rng(seed)
    g = haar(n, rng)
    g_tr, g_te = g[:5120], g[5120:]
    Zf_tr = embed(model, enc, g_tr, device)
    Zf_te = embed(model, enc, g_te, device)
    frame = PCAFrame(Zf_tr)
    Z_tr, Z_te = frame(Zf_tr), frame(Zf_te)

    ks = haar(n_k, rng)
    out = {"equi_r2": []}
    Ts = []
    for k in ks:
        W, b = fit_T(model, enc, k, g_tr, Z_tr, frame, device)
        Ts.append((W, b))
        Zk_te = frame(embed(model, enc, qmul(k[None, :], g_te), device))
        out["equi_r2"].append(r2(Z_te @ W + b, Zk_te))

    # homomorphism defect on held-out z: T_k(T_h(z)) vs T_{kh}(z)
    hom = []
    for _ in range(12):
        i, j = rng.integers(0, n_k, 2)
        k, h = ks[i], ks[j]
        Wk, bk = Ts[i]
        Wh, bh = Ts[j]
        Wkh, bkh = fit_T(model, enc, qmul(k, h), g_tr, Z_tr, frame, device)
        lhs = (Z_te @ Wh + bh) @ Wk + bk
        rhs = Z_te @ Wkh + bkh
        hom.append(float(np.linalg.norm(lhs - rhs)
                         / np.linalg.norm(rhs - rhs.mean(0))))
    out["hom_defect"] = hom

    # identity and inverse
    We, be = fit_T(model, enc, np.array([1.0, 0, 0, 0]), g_tr, Z_tr, frame,
                   device)
    out["identity_defect"] = float(
        np.linalg.norm(We - np.eye(len(We))) / np.sqrt(len(We)))
    inv = []
    for i in range(4):
        k = ks[i]
        Wk, bk = Ts[i]
        Wki, bki = fit_T(model, enc, qconj(k), g_tr, Z_tr, frame, device)
        back = (Z_te @ Wki + bki) @ Wk + bk
        inv.append(float(np.linalg.norm(back - Z_te)
                         / np.linalg.norm(Z_te - Z_te.mean(0))))
    out["inv_defect"] = inv
    equi_min = float(np.min(out["equi_r2"]))
    out = {k: (float(np.mean(v)) if isinstance(v, list) else v)
           for k, v in out.items()}
    out["equi_r2_min"] = equi_min
    out["pca_dims"] = int(Z_tr.shape[1])

    # dominant-block comparison against the irrep matrices, basis-invariantly:
    # left translation acts on the spin-j matrix-coefficient block by d_j copies
    # of D^j(k), so tr W̃_k should equal d_j Re chi_j(k), and the eigenphases
    # should match those of D^j(k) with multiplicity d_j.
    jdom = 0.5 if cfg["variant"] == "su2" else 1.0
    dj = int(round(2 * jdom)) + 1
    B = spin_block_frame(Zf_tr, g_tr, jdom)
    Zb_tr = (Zf_tr - Zf_tr.mean(0)) @ B
    traces, chis, phase_err = [], [], []
    for k, more in zip(haar(64, rng), range(64)):
        Zbk = (embed(model, enc, qmul(k[None, :], g_tr), device)
               - Zf_tr.mean(0)) @ B
        Wb, _ = ridge_affine(Zb_tr, Zbk)
        theta = float(np.arccos(np.clip(k[0], -1, 1)))
        traces.append(float(np.trace(Wb)))
        if cfg["variant"] == "su2":
            chis.append(2 * 2 * np.cos(theta))          # d_j * chi_{1/2}
            want = [theta, theta, -theta, -theta]
        else:
            chis.append(3 * (1 + 2 * np.cos(2 * theta)))  # d_j * chi_1
            want = [0.0, 0, 0, 2 * theta, 2 * theta, 2 * theta,
                    -2 * theta, -2 * theta, -2 * theta]
        want = np.sort(np.angle(np.exp(1j * np.array(want))))  # wrap to (-pi, pi]
        ev = np.linalg.eigvals(Wb)
        got = np.sort(np.angle(ev[np.argsort(-np.abs(ev))[:len(want)]]))
        phase_err.append(float(np.abs(got - want).mean()))
    tr_arr, chi_arr = np.array(traces), np.array(chis)
    out["block_dims"] = int(B.shape[1])
    out["trace_vs_char_corr"] = float(np.corrcoef(tr_arr, chi_arr)[0, 1])
    out["trace_vs_char_relerr"] = float(
        np.linalg.norm(tr_arr - chi_arr) / np.linalg.norm(chi_arr))
    out["block_eigphase_mae"] = float(np.mean(phase_err))
    return out


# ------------------------------ A2: product state -----------------------------

def product_state(model, enc, qa, qb, device, batch=16384):
    out = []
    with torch.no_grad():
        for i in range(0, len(qa), batch):
            out.append(model.product_state(
                torch.tensor(enc(qa[i:i + batch]), device=device),
                torch.tensor(enc(qb[i:i + batch]), device=device)).cpu().numpy())
    return np.concatenate(out)


def score_from_h(model, enc, H, cands, device):
    """Scores for candidate array [n, K, 4] given externally supplied h."""
    with torch.no_grad():
        zc = model.enc_c(torch.tensor(
            enc(cands.reshape(-1, 4)).reshape(cands.shape[0], cands.shape[1], -1),
            device=device))
        return torch.einsum("bd,bkd->bk",
                            torch.tensor(H, dtype=torch.float32, device=device),
                            zc).cpu().numpy()


def dhalf_features(q):
    D = wigner_D(0.5, q).reshape(len(q), -1)
    return np.concatenate([D.real, D.imag], axis=1)


def done_features(q):
    D = wigner_D(1.0, q).reshape(len(q), -1)
    return np.concatenate([D.real, D.imag], axis=1)


def test_product(model, enc, cfg, device, n=8192, seed=5):
    rng = np.random.default_rng(seed)
    qa, qb = haar(n, rng), haar(n, rng)
    prod = qmul(qa, qb)
    H = product_state(model, enc, qa, qb, device)

    # spectroscopy of h in the product variable
    bases = spin_basis(prod, j_list=(0.0, 0.5, 1.0, 1.5, 2.0))
    Hc = H - H.mean(0)
    tot = (Hc ** 2).sum()
    fve = {f"h_fve_{j}": float(((Q.T @ Hc) ** 2).sum() / tot)
           for j, Q in bases.items()}
    fve["h_residual"] = float(1.0 - sum(fve.values()))

    # candidate sets
    negs = haar(n * 62, rng).reshape(n, 62, 4)
    task_c = np.concatenate([prod[:, None], negs], 1).astype(np.float32)
    center_c = np.concatenate([prod[:, None], -prod[:, None]], 1).astype(np.float32)

    def metrics(Hmod):
        s = score_from_h(model, enc, Hmod, task_c, device)
        acc = float((s.argmax(1) == 0).mean())
        sc = score_from_h(model, enc, Hmod, center_c, device)
        dc = float(((sc[:, 0] > sc[:, 1]) + 0.5 * (sc[:, 0] == sc[:, 1])).mean())
        return acc, dc

    res = dict(fve)
    res["task_intact"], res["Dc_intact"] = metrics(H)

    # function-ablate the D^{1/2}(ab)-predictable component of h
    F = dhalf_features(prod)
    Wf, bf = ridge_affine(F, H)
    res["task_h_no_half"], res["Dc_h_no_half"] = metrics(H - (F @ Wf + bf) + H.mean(0))
    res["h_fit_r2_half"] = r2(F @ Wf + bf, H)
    F1 = done_features(prod)
    W1, b1 = ridge_affine(F1, H)
    res["h_fit_r2_one"] = r2(F1 @ W1 + b1, H)

    # patching: (a,b) trial, replace h by h(a',b'); candidates [ab, a'b', 62 Haar]
    qa2, qb2 = haar(n, rng), haar(n, rng)
    prod2 = qmul(qa2, qb2)
    H2 = product_state(model, enc, qa2, qb2, device)
    redirect_c = np.concatenate([prod[:, None], prod2[:, None], negs], 1
                                ).astype(np.float32)
    s = score_from_h(model, enc, H, redirect_c, device)
    res["redirect_unpatched_picks_ab"] = float((s.argmax(1) == 0).mean())
    s = score_from_h(model, enc, H2, redirect_c, device)
    res["redirect_fullpatch_picks_a2b2"] = float((s.argmax(1) == 1).mean())
    # fit patch: h <- reconstruction from D^{1/2}(a'b') (and D^1 control)
    s = score_from_h(model, enc, dhalf_features(prod2) @ Wf + bf, redirect_c, device)
    res["redirect_fitpatch_half"] = float((s.argmax(1) == 1).mean())
    s = score_from_h(model, enc, done_features(prod2) @ W1 + b1, redirect_c, device)
    res["redirect_fitpatch_one"] = float((s.argmax(1) == 1).mean())
    return res


# ------------------------------- A4: intertwiners -----------------------------

def test_stitch(cell, device, n=6144, n_k=4, seed=9):
    runs = sorted(DATA.glob(f"{cell}_s*"))
    runs = [r for r in runs if (r / "config.json").exists()]
    rng = np.random.default_rng(seed)
    g = haar(n, rng)
    g_tr, g_te = g[:5120], g[5120:]
    ks = haar(n_k, rng)

    loaded = []
    for rdir in runs:
        model, enc, cfg, _ = load_model(rdir, device=device)
        frame = PCAFrame(embed(model, enc, g_tr, device))
        Z_tr = frame(embed(model, enc, g_tr, device))
        Z_te = frame(embed(model, enc, g_te, device))
        Tk = [fit_T(model, enc, k, g_tr, Z_tr, frame, device) for k in ks]
        loaded.append((rdir.name, Z_tr, Z_te, Tk))

    out = []
    ref = loaded[0]
    for name, Z_tr, Z_te, Tk in loaded[1:]:
        A, c = ridge_affine(ref[1], Z_tr)
        stitch_r2 = r2(ref[2] @ A + c, Z_te)
        # intertwining defect: A T^0_k(z) vs T^s_k(A z) on held-out
        defect = []
        for (W0, b0), (Ws, bs) in zip(ref[3], Tk):
            lhs = (ref[2] @ W0 + b0) @ A + c
            rhs = (ref[2] @ A + c) @ Ws + bs
            defect.append(float(np.linalg.norm(lhs - rhs)
                                / np.linalg.norm(rhs - rhs.mean(0))))
        out.append({"pair": f"{ref[0]}->{name}", "stitch_r2": stitch_r2,
                    "intertwine_defect": float(np.mean(defect))})
    return out


# ----------------------------------- driver -----------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default=None)
    p.add_argument("--cell", default=None,
                   help="cell prefix for --test stitch, e.g. su2_N2048_f0.01")
    p.add_argument("--test", choices=["action", "product", "stitch", "all"],
                   default="all")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    results = {}
    if args.test in ("action", "product", "all"):
        assert args.run_dir, "--run_dir required for action/product tests"
        model, enc, cfg, step = load_model(args.run_dir, device=args.device)
        print(f"== {args.run_dir} ({cfg['variant']}, step {step})")
        if args.test in ("action", "all"):
            results["action"] = test_action(model, enc, cfg, args.device)
            print("A1 latent group action:",
                  json.dumps({k: round(v, 4) for k, v in results["action"].items()}))
        if args.test in ("product", "all"):
            results["product"] = test_product(model, enc, cfg, args.device)
            print("A2 product state:",
                  json.dumps({k: round(v, 4) for k, v in results["product"].items()}))
        out = Path(args.run_dir) / f"mechanism_{step}.json"
        out.write_text(json.dumps(results, indent=2))
        print("saved", out)
    if args.test in ("stitch",):
        assert args.cell, "--cell required for stitch"
        res = test_stitch(args.cell, args.device)
        for r in res:
            print(f"A4 {r['pair']}: stitch R2 {r['stitch_r2']:.4f}  "
                  f"intertwine defect {r['intertwine_defect']:.4f}")
        (DATA / f"stitch_{args.cell}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
