"""Causal analyses of Spin(4)-family models: block-resolved center tests.

The center Z2 x Z2 acts on the argument encoders; z decomposes into four
character components indexed by parities (eL, eR) under the flips
(qL -> -qL) and (qR -> -qR). The carrier blocks have characters:
    (1/2, 0): (-,+)    (0, 1/2): (+,-)    (1/2, 1/2): (-,-)
    (1, 0), (0, 1), (1, 1): (+,+).
Interventions: remove one character component (basis-free, computed by orbit
signing); function-ablate individual (jL, jR) blocks; random control.
Metrics: task accuracy (product vs 63 excluded Haar negatives) and tie-aware
center discriminations D_L, D_R, D_LR = P[s(a,b,ab) > s(a,b, flipped ab)]
(chance 1/2; an exact tie for flips the variant's encoding quotients).

Sharp predictions (recorded before running): ablating a block collapses
exactly the D's whose flip that block's character sees, e.g. for spin4,
removing the (-,+) component kills D_L (and D_LR) but leaves D_R.

Usage:
    uv run python -m experiments.spin4_causal --run_dir data/spin4/spin4_N2048_f0.01_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.spin4 import (CENTER, Spin4LandmarkEncoding, class_distance,
                        haar_spin4, mul)
from core.su2 import wigner_D
from core.su2_model import TripleScorer

BLOCKS = [(0.5, 0), (0, 0.5), (0.5, 0.5), (1, 0), (0, 1), (1, 0.5), (0.5, 1)]


def load_model(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpts = sorted(run_dir.glob("ckpt_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    state = torch.load(ckpts[-1], map_location=device, weights_only=False)
    model = TripleScorer(m=cfg["m"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    enc = Spin4LandmarkEncoding(m=cfg["m"], alpha=cfg["alpha"],
                                variant=cfg["variant"], seed=cfg["enc_seed"])
    return model, enc, cfg, state["step"]


def embed_np(model, enc, g, device, batch=16384):
    out = []
    with torch.no_grad():
        for i in range(0, len(g), batch):
            out.append(model.embed(torch.tensor(
                enc(g[i:i + batch]), device=device)).cpu().numpy())
    return np.concatenate(out)


def flipped(g, sL, sR):
    gs = g.copy()
    gs[..., 0, :] *= sL
    gs[..., 1, :] *= sR
    return gs


def block_features(g, block):
    jL, jR = block
    n = len(g)
    A = wigner_D(jL, g[:, 0]).reshape(n, -1) if jL > 0 else np.ones((n, 1))
    B = wigner_D(jR, g[:, 1]).reshape(n, -1) if jR > 0 else np.ones((n, 1))
    C = np.einsum("ni,nj->nij", A, B).reshape(n, -1)
    return np.concatenate([C.real, C.imag], axis=1)


def make_interventions(model, enc, device, probe_n=4096, seed=123):
    rng = np.random.default_rng(seed)
    probe = haar_spin4(probe_n, rng)
    Z = embed_np(model, enc, probe, device)
    mu, Zc = Z.mean(0), Z - Z.mean(0)
    d = Z.shape[1]

    def identity_fn(g):
        return embed_np(model, enc, g, device)

    def remove_char(eL, eR):
        """Subtract the (eL, eR) character component of z."""
        def fn(g):
            comp = 0
            for sL, sR in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                w = (eL ** ((1 - sL) // 2)) * (eR ** ((1 - sR) // 2))
                comp = comp + w * embed_np(model, enc, flipped(g, sL, sR),
                                           device)
            return identity_fn(g) - comp / 4
        return fn

    def fablate(block):
        F = block_features(probe, block)
        muF, Fc = F.mean(0), F - F.mean(0)
        W = np.linalg.solve(Fc.T @ Fc + 1e-6 * np.eye(F.shape[1]), Fc.T @ Zc)

        def fn(g):
            return identity_fn(g) - (block_features(g, block) - muF) @ W
        return fn

    V = np.linalg.qr(rng.standard_normal((d, 8)))[0]

    def random_ablate(g):
        z = identity_fn(g)
        return z - (z - mu) @ V @ V.T

    ivs = [("none", identity_fn),
           ("remove_char_-+", remove_char(-1, 1)),
           ("remove_char_+-", remove_char(1, -1)),
           ("remove_char_--", remove_char(-1, -1))]
    ivs += [(f"fablate_{b[0]}_{b[1]}", fablate(b)) for b in BLOCKS[:5]]
    ivs.append(("ablate_random_k8", random_ablate))
    return ivs


@torch.no_grad()
def score_with(model, enc, fn, ga, gb, gc, device, batch=2048):
    out = []
    n, K = gc.shape[:2]
    for i in range(0, n, batch):
        za = torch.tensor(fn(ga[i:i + batch]), dtype=torch.float32,
                          device=device)
        zb = torch.tensor(fn(gb[i:i + batch]), dtype=torch.float32,
                          device=device)
        h = model.interact(torch.cat([za, zb], dim=-1))
        c = gc[i:i + batch].reshape(-1, 2, 4)
        zc = model.enc_c(torch.tensor(enc(c), device=device)
                         ).reshape(-1, K, h.shape[-1])
        out.append(torch.einsum("bd,bkd->bk", h, zc).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, enc, cfg, step = load_model(args.run_dir, args.device)
    print(f"== {args.run_dir} ({cfg['variant']}, step {step})")
    rng = np.random.default_rng(7)
    ga, gb = haar_spin4(args.n, rng), haar_spin4(args.n, rng)
    prod = mul(ga, gb)

    neg = haar_spin4(args.n * 63, rng).reshape(args.n, 63, 2, 4)
    for _ in range(50):
        bad = class_distance(neg, prod[:, None], cfg["variant"]) \
            < cfg["excl_radius"]
        if not bad.any():
            break
        neg[bad] = haar_spin4(int(bad.sum()), rng)
    task_c = np.concatenate([prod[:, None], neg], axis=1).astype(np.float32)

    flips = {"L": (-1, 1), "R": (1, -1), "LR": (-1, -1)}
    center_c = {t: np.stack([prod, flipped(prod, *sig)], axis=1
                            ).astype(np.float32) for t, sig in flips.items()}

    res = {"variant": cfg["variant"], "step": step, "interventions": {}}
    hdr = f"{'intervention':<18} {'task':>6}" + "".join(
        f" {'D_' + t:>6}" for t in flips)
    print(hdr)
    for name, fn in make_interventions(model, enc, args.device):
        row = {"task_acc": float((score_with(model, enc, fn, ga, gb, task_c,
                                             args.device).argmax(1) == 0
                                  ).mean())}
        for t in flips:
            sc = score_with(model, enc, fn, ga, gb, center_c[t], args.device)
            winners = sc >= sc.max(1, keepdims=True) - 1e-6
            row[f"D_{t}"] = float((winners[:, 0] / winners.sum(1)).mean())
        res["interventions"][name] = row
        print(f"{name:<18} {row['task_acc']:>6.3f}" + "".join(
            f" {row['D_' + t]:>6.3f}" for t in flips))

    out = Path(args.run_dir) / f"causal_{step}.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
