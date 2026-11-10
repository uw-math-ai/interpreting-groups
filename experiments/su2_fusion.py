"""Representation-ring / fusion analysis of the interaction layer (plan E).

Functions on G x G decompose into (j_a, j_b) sectors under the two-sided
action, with basis the products D^{j_a}(a)_{mk} D^{j_b}(b)_{ln}. Three
questions, per checkpoint:

1. Which (j_a, j_b) channels does the interaction MLP's hidden layer use?
   (The GELU nonlinearity may generate channels beyond the encoder content
   j_a, j_b in {content of z}; tensor products of the input sectors are the
   *allowed* set; anything else should sit at the measurement floor.)

2. Which channels does the interaction output h use? A function of the product
   ab alone lies on the diagonal j_a = j_b; the model's dominant D^j(ab)
   component lies specifically in (j, j).

3. Fusion selectivity: within the (j, j) sector — dimension 2(2j+1)^4 realified
   — the composed representation D^j(a) D^j(b) = D^j(ab) spans only the
   (2j+1)^2 matrix-product contraction subspace. What fraction of h's (j, j)
   content lies in that contraction subspace? A value near 1 means the network
   implements the Clebsch-Gordan-contracted channel, not an arbitrary bilinear
   coupling.

Usage:
    uv run python -m experiments.su2_fusion --run_dir data/su2/su2_N2048_f0.01_s0
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from core.su2 import haar, qmul, wigner_D
from experiments.su2_causal import load_model

SECTORS2 = [(ja / 2, jb / 2) for ja, jb in
            itertools.product(range(0, 4), repeat=2)][1:]   # drop (0,0)


def casimir2(s):
    return s[0] * (s[0] + 1) + s[1] * (s[1] + 1)


def realify(Dcols):
    return np.concatenate([Dcols.real, Dcols.imag], axis=1)


def pair_sector_bases(qa, qb, sectors=tuple(SECTORS2)):
    """Jointly-orthonormalized sampled bases of the (j_a, j_b) sectors of
    functions on G x G, processed in order of total Casimir."""
    N = len(qa)
    Da = {j: wigner_D(j, qa).reshape(N, -1) for j in {s[0] for s in sectors}}
    Db = {j: wigner_D(j, qb).reshape(N, -1) for j in {s[1] for s in sectors}}
    bases, prev = {}, np.zeros((N, 0))
    for s in sorted(sectors, key=lambda s: (casimir2(s), s)):
        A = Da[s[0]] if s[0] > 0 else np.ones((N, 1), dtype=complex)
        B = Db[s[1]] if s[1] > 0 else np.ones((N, 1), dtype=complex)
        cols = realify(np.einsum("ni,nj->nij", A, B).reshape(N, -1))
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


def fve_by_sector(Z, bases):
    Zc = Z - Z.mean(0)
    tot = (Zc ** 2).sum()
    return {s: float(((Q.T @ Zc) ** 2).sum() / tot) for s, Q in bases.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, enc, cfg, step = load_model(args.run_dir, device=args.device)
    jdom = 0.5 if cfg["variant"] == "su2" else 1.0
    print(f"== {args.run_dir} ({cfg['variant']}, step {step})")

    rng = np.random.default_rng(17)
    qa, qb = haar(args.n, rng), haar(args.n, rng)
    prod = qmul(qa, qb)

    with torch.no_grad():
        za = model.enc(torch.tensor(enc(qa), device=args.device))
        zb = model.enc(torch.tensor(enc(qb), device=args.device))
        cat = torch.cat([za, zb], dim=-1)
        hidden = model.interact[1](model.interact[0](cat)).cpu().numpy()
        h = model.interact(cat).cpu().numpy()

    bases = pair_sector_bases(qa, qb)
    res = {"variant": cfg["variant"], "step": step,
           "sector_dims": {str(s): int(Q.shape[1]) for s, Q in bases.items()},
           "fve_hidden": {str(s): v for s, v in
                          fve_by_sector(hidden, bases).items()},
           "fve_h": {str(s): v for s, v in fve_by_sector(h, bases).items()}}

    # fusion selectivity within the diagonal (jdom, jdom) sector
    Q = bases[(jdom, jdom)]
    hc = h - h.mean(0)
    sector_energy = ((Q.T @ hc) ** 2).sum()
    Dab = realify(wigner_D(jdom, prod).reshape(args.n, -1))
    Dab = Dab - Dab.mean(0)
    Qc, R = np.linalg.qr(Dab)
    Qc = Qc[:, np.abs(np.diag(R)) > 1e-9 * np.abs(R).max()]  # realified rank

    # the contraction subspace inside the sampled sector frame
    comp_energy = ((Qc.T @ hc) ** 2).sum()
    res["diag_sector_energy_frac"] = float(sector_energy / (hc ** 2).sum())
    res["contraction_frac_of_diag"] = float(comp_energy / sector_energy)
    res["contraction_dim"] = int(Qc.shape[1])
    res["diag_sector_dim"] = int(Q.shape[1])

    print("hidden-layer sectors (top 6):",
          sorted(res["fve_hidden"].items(), key=lambda kv: -kv[1])[:6])
    print("h sectors (top 6):",
          sorted(res["fve_h"].items(), key=lambda kv: -kv[1])[:6])
    print(f"h: diagonal ({jdom},{jdom}) sector carries "
          f"{res['diag_sector_energy_frac']:.3f} of variance; "
          f"contraction subspace ({res['contraction_dim']} of "
          f"{res['diag_sector_dim']} dims) carries "
          f"{res['contraction_frac_of_diag']:.3f} of that sector")

    out = Path(args.run_dir) / f"fusion_{step}.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
