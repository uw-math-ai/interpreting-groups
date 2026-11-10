"""P4: score-level Peter-Weyl coefficient recovery for regression models.

For a heat-kernel (or band-limited) model, sample triples (a, b, c) with the
training candidate geometry, evaluate the learned score s(a, b, c), and
regress BOTH s and the exact standardized target on the same character design
{chi_j(ab c^-1)}, j = 0..j_max. Reporting the per-j ratio
beta_hat_j(model) / beta_hat_j(target) with a shared design removes
collinearity and standardization from the comparison. For band-limited
targets, coefficients on out-of-support j are the forbidden-channel floor.
Runs over every available checkpoint to give the learning-order curve.

Usage:
    uv run python -m experiments.hk_analysis --run_dir data/su2/su2hk_t0.05_long_N2048_f0.01_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.su2 import (expmap, haar, heat_kernel, qconj, qmul,
                      wigner_character)
from core.su2_model import TripleScorer
from experiments.su2_causal import load_model


def sample_design(cfg, n, rng, j_max=4.0):
    qa, qb = haar(n, rng), haar(n, rng)
    prod = qmul(qa, qb)
    r = rng.uniform(0.0, np.pi / 2, size=(n, 1))
    u = rng.standard_normal((n, 3))
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    near = qmul(prod, expmap(r * u))
    c = np.where(rng.random(n)[:, None] < 0.5, near, haar(n, rng))
    rel = qmul(prod, qconj(c))
    js = [j / 2 for j in range(1, int(2 * j_max) + 1)]
    X = np.stack([wigner_character(j, rel) for j in js], axis=1)
    # exact standardized target
    if cfg.get("loss") == "bandlim":
        y = sum(cc * wigner_character(j, rel)
                for j, cc in zip(cfg["bl_js"], cfg["bl_coefs"]))
    else:
        y = heat_kernel(rel, cfg["hk_t"])
    return qa, qb, c, X, js, y


def coef_fit(X, y):
    """Sequentially orthogonalized character fit (Casimir order).

    The raw character design is near-collinear under the sampling
    distribution (high-j characters agree near the identity), so individual
    least-squares betas are unidentifiable. QR-orthogonalize the centered
    columns in Casimir order and report, per sector, the INCREMENTAL fraction
    of variance explained by adding that sector after all lower ones -- the
    identifiable spectral profile, and the natural learning-order statistic.
    """
    Xc = X - X.mean(0)
    Q, _ = np.linalg.qr(Xc)
    yc = y - y.mean()
    c = Q.T @ yc
    frac = c ** 2 / (yc ** 2).sum()
    return frac, float(frac.sum())


@torch.no_grad()
def model_scores(model, enc, qa, qb, c, device, batch=8192):
    out = []
    for i in range(0, len(qa), batch):
        za = model.embed(torch.tensor(enc(qa[i:i + batch]), device=device))
        zb = model.embed(torch.tensor(enc(qb[i:i + batch]), device=device))
        h = model.interact(torch.cat([za, zb], dim=-1))
        zc = model.enc_c(torch.tensor(enc(c[i:i + batch]), device=device))
        out.append((h * zc).sum(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    run = Path(args.run_dir)
    cfg = json.loads((run / "config.json").read_text())
    rng = np.random.default_rng(41)
    qa, qb, c, X, js, y = sample_design(cfg, args.n, rng)
    beta_t, r2_t = coef_fit(X, y)

    results = {"js": js, "target_frac": beta_t.tolist(),
               "target_fit_r2": r2_t, "loss": cfg.get("loss"),
               "hk_t": cfg.get("hk_t"), "ckpts": {}}
    ckpts = sorted(run.glob("ckpt_*.pt"),
                   key=lambda q: int(q.stem.split("_")[1]))
    for ck in ckpts:
        model, enc, cfg2, step = load_model(run, ck.name, args.device)
        s = model_scores(model, enc, qa, qb, c, args.device)
        beta_m, r2_m = coef_fit(X, s)
        # recovery per sector: fraction of model-score variance in the
        # orthogonalized sector divided by the target's fraction; 1.0 = the
        # sector carries its required relative weight. Out-of-support sectors
        # (target frac < 1e-4) are reported as raw model fractions (floors).
        rec, floors = {}, {}
        for j, fm, ft in zip(js, beta_m, beta_t):
            if ft > 1e-4:
                rec[str(j)] = float(fm / ft)
            else:
                floors[str(j)] = float(fm)
        results["ckpts"][step] = {
            "model_frac": beta_m.tolist(), "model_fit_r2": r2_m,
            "recovery": rec, "out_of_support_frac": floors}
        print(f"step {step:>7}: score-char R2 {r2_m:.3f}  recovery "
              + " ".join(f"j{j}:{round(r, 3)}" for j, r in rec.items())
              + ("  floors " + " ".join(f"j{j}:{v:.4f}"
                 for j, v in floors.items()) if floors else ""))
    out = run / "hk_coefficients.json"
    out.write_text(json.dumps(results, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
