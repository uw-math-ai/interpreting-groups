"""G5: signed Peter--Weyl coefficient estimation under normalized Haar.

Estimand. For the relative element r = a b c^{-1}, define the
nuisance-averaged score

    s_bar(r) = E[s(a, b, c) | a b c^{-1} = r]

under the factorization distribution: a, b independent Haar, and
c = r^{-1} (a b) constructed algebraically so that a b c^{-1} = r exactly
(verified by a permanent convention test). r itself is sampled from
normalized Haar, independent of the task candidate distribution.

Characters estimate the class-function component of s_bar. Signed
coefficients are

    a_hat_j = <s_bar, chi_j>_Haar / <chi_j, chi_j>_Haar,

with the denominator estimated on the same batch as a conditioning check
(it equals 1 exactly under normalized Haar). Estimation and reconstruction
use independent Monte Carlo batches. Uncertainty comes from disjoint
r-folds. The old QR incremental-variance profile in hk_analysis.py is NOT
a coefficient estimator (order- and measure-dependent, unsigned); a
regression test guards that failure mode.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.envelope import make_envelope, write_analysis
from core.su2 import (get_encoding, haar, heat_kernel, qconj, qmul,
                      wigner_character)
from core.su2_model import TripleScorer

J_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def factorize(r, rng):
    """Sample (a, b) Haar and construct c with a b c^{-1} = r exactly."""
    n = r.shape[0]
    a, b = haar(n, rng), haar(n, rng)
    prod = qmul(a, b)
    c = qmul(qconj(r), prod)
    return a, b, c


def estimate(score_fn, js, n_r=8192, n_f=16, n_folds=10, seed=0):
    """Signed coefficient estimates with fold-based uncertainty.

    score_fn(a, b, c) -> [n] raw scores. Returns a dict with per-j signed
    coefficients, fold statistics, gram diagonal, and nuisance variance.
    """
    rng = np.random.default_rng(seed)
    r = haar(n_r, rng)
    S = np.stack([score_fn(*factorize(r, rng)) for _ in range(n_f)])
    s_bar = S.mean(0)
    nuis_var = float(S.var(0, ddof=1).mean()) if n_f > 1 else None
    total_var = float(S.var())

    chi = {j: wigner_character(j, r) for j in js}
    folds = np.array_split(np.arange(n_r), n_folds)
    coef, gram = {}, {}
    for j in js:
        per_fold = np.array([np.mean(s_bar[f] * chi[j][f]) for f in folds])
        gram_fold = np.array([np.mean(chi[j][f] ** 2) for f in folds])
        est = per_fold / np.maximum(gram_fold, 1e-12)
        m, sd = float(est.mean()), float(est.std(ddof=1))
        half = 2.262 * sd / np.sqrt(n_folds)          # t(9) 97.5%
        coef[j] = {"a_hat": m, "fold_sd": sd,
                   "ci95": [m - half, m + half]}
        gram[j] = float(gram_fold.mean())
    return {"coef": coef, "gram_diag": gram, "n_r": n_r, "n_f": n_f,
            "n_folds": n_folds,
            "nuisance_variance": nuis_var,
            "nuisance_fraction": (None if nuis_var is None or total_var == 0
                                  else nuis_var / total_var),
            "mc_noise_floor_on_sbar": (None if nuis_var is None
                                       else nuis_var / n_f),
            "_s_bar_var": float(s_bar.var())}


def reconstruction_r2(score_fn, coef, js, n_r=4096, n_f=16, seed=1):
    """Held-out Haar R^2 of the character reconstruction against s_bar."""
    rng = np.random.default_rng(seed)
    r = haar(n_r, rng)
    S = np.stack([score_fn(*factorize(r, rng)) for _ in range(n_f)])
    s_bar = S.mean(0)
    recon = sum(coef[j]["a_hat"] * wigner_character(j, r) for j in js)
    resid = s_bar - recon
    return float(1.0 - resid.var() / max(s_bar.var(), 1e-12))


def task_geometry_r2(score_fn, coef, js, n=4096, seed=2):
    """R^2 of the character reconstruction against raw scores on the task
    candidate geometry (half near-product, half Haar candidates)."""
    from experiments.hk_analysis import sample_design
    rng = np.random.default_rng(seed)
    qa, qb = haar(n, rng), haar(n, rng)
    prod = qmul(qa, qb)
    rad = rng.uniform(0.0, np.pi / 2, size=(n, 1))
    u = rng.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    from core.su2 import expmap
    near = qmul(prod, expmap(rad * u))
    c = np.where(rng.random(n)[:, None] < 0.5, near, haar(n, rng))
    rel = qmul(prod, qconj(c))
    s = score_fn(qa, qb, c)
    recon = sum(coef[j]["a_hat"] * wigner_character(j, rel) for j in js)
    resid = s - recon
    return float(1.0 - resid.var() / max(s.var(), 1e-12))


def target_coefficients(cfg, js):
    """Exact class-function coefficients of the (unstandardized) target."""
    if cfg.get("loss") == "bandlim":
        t = {float(j): float(c) for j, c in zip(cfg["bl_js"],
                                                cfg["bl_coefs"])}
        return {j: t.get(j, 0.0) for j in js}
    if cfg.get("loss") == "heatkernel":
        t = cfg["hk_t"]
        return {j: (2 * j + 1) * float(np.exp(-t * j * (j + 1)))
                for j in js}
    return None


def model_score_fn(run_dir, ckpt_name, device):
    cfg = json.load(open(Path(run_dir) / "config.json"))
    state = torch.load(Path(run_dir) / ckpt_name, map_location=device,
                       weights_only=False)
    # older checkpoints embed a config with fewer keys; config.json fills gaps
    eff = {**cfg, **state.get("config", {})}
    enc = get_encoding(eff["encoding"], eff["variant"], m=eff.get(
        "requested_overrides", {}).get("m", eff["m"]), alpha=eff["alpha"],
        seed=eff["enc_seed"])
    model = TripleScorer(m=eff["m"], d_enc=eff["d_enc"],
                         enc_hidden=eff["enc_hidden"],
                         mlp_hidden=eff["mlp_hidden"],
                         activation=eff["activation"],
                         mlp_depth=eff.get("mlp_depth", 1)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    psi = enc.torch_fn(device)

    @torch.no_grad()
    def fn(a, b, c):
        ta = psi(torch.tensor(a, dtype=torch.float32, device=device))
        tb = psi(torch.tensor(b, dtype=torch.float32, device=device))
        tc = psi(torch.tensor(c, dtype=torch.float32, device=device))
        h = model.product_state(ta, tb)
        zc = model.enc_c(tc)
        return (h * zc).sum(-1).cpu().numpy()

    return fn, eff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default=None,
                   help="checkpoint filename; default = highest step")
    p.add_argument("--jmax", type=float, default=3.0)
    p.add_argument("--n-r", type=int, default=8192)
    p.add_argument("--n-f", type=int, default=16)
    p.add_argument("--analysis-seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run = Path(args.run)
    ckpt = args.ckpt or sorted(run.glob("ckpt_*.pt"),
                               key=lambda q: int(q.stem.split("_")[1]))[-1].name
    js = [j for j in J_GRID if j <= args.jmax]
    fn, eff = model_score_fn(run, ckpt, args.device)

    est = estimate(fn, js, n_r=args.n_r, n_f=args.n_f,
                   seed=args.analysis_seed)
    r2_haar = reconstruction_r2(fn, est["coef"], js,
                                seed=args.analysis_seed + 1)
    r2_task = task_geometry_r2(fn, est["coef"], js,
                               seed=args.analysis_seed + 2)
    tgt = target_coefficients(eff, js)
    ratios = None
    if tgt is not None:
        per_j = {j: est["coef"][j]["a_hat"] / tgt[j]
                 for j in js if j != 0.0 and abs(tgt[j]) > 1e-6}
        scale = float(np.median(list(per_j.values()))) if per_j else None
        ratios = {
            "per_j_ratio": per_j,
            "median_scale": scale,
            "normalized_profile": ({j: r / scale for j, r in per_j.items()}
                                   if scale else None),
            "out_of_support_abs": {
                j: {"abs": abs(est["coef"][j]["a_hat"]),
                    "ci95": est["coef"][j]["ci95"]}
                for j in js if j != 0.0 and abs(tgt[j]) <= 1e-6}}

    results = {
        "run": str(run), "checkpoint": ckpt, "j_grid": js,
        "estimate": {str(j): est["coef"][j] for j in js},
        "gram_diag": {str(j): est["gram_diag"][j] for j in js},
        "nuisance_variance": est["nuisance_variance"],
        "nuisance_fraction": est["nuisance_fraction"],
        "mc_noise_floor_on_sbar": est["mc_noise_floor_on_sbar"],
        "reconstruction_r2_haar": r2_haar,
        "reconstruction_r2_task_geometry": r2_task,
        "target_coefficients": ({str(j): tgt[j] for j in js} if tgt else None),
        "target_comparison": ({str(k) if not isinstance(k, str) else k:
                               v for k, v in ratios.items()} if ratios
                              else None),
        "notes": ("signed Haar projection; ratio reported only where the "
                  "target coefficient is nonzero; j=0 excluded from ratios "
                  "because target standardization shifts the constant"),
    }
    out = args.out or run / f"hk_haar_{ckpt.replace('.pt','')}.json"
    env = make_envelope("hk-haar-coefficients", 1, analysis_file=__file__,
                        analysis_args=vars(args),
                        ckpt_path=run / ckpt,
                        run_config_path=run / "config.json",
                        training_seed=eff.get("seed"),
                        analysis_seed=args.analysis_seed)
    write_analysis(out, results, env)
    print(f"[{run.name}] wrote {out}")
    for j in js:
        c = est["coef"][j]
        t = f" target={tgt[j]:+.4f}" if tgt else ""
        print(f"  j={j:3}: a_hat={c['a_hat']:+.4f} "
              f"ci95=[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]{t}")
    print(f"  R2(haar)={r2_haar:.4f} R2(task)={r2_task:.4f} "
          f"nuisance_frac={est['nuisance_fraction']}")


if __name__ == "__main__":
    main()
