"""Train the encoder-factorized triple scorer on SU(2) (or SO(3)-quotient) composition.

Task (InfoNCE / candidate scoring):
    fixed pool of N Haar elements; training set = fraction `frac` of the N^2
    ordered pairs, frozen. Per pair: positive candidate = (a*b)*exp(xi),
    xi ~ N(0, sigma^2 I3); K-1 Haar negatives, frozen, rejection-resampled to be
    >= excl_radius from the positive class ({ab} for su2, {+-ab} for so3).
    Loss: cross-entropy over the K scores (positive at index 0; the scorer is
    permutation-equivariant across candidates so this leaks nothing).

Eval tiers (all frozen at setup):
    train_sub -- 8192 training pairs (memorization curve)
    T1 -- held-out pairs of *pool* elements (classic grokking eval)
    T2 -- fresh Haar pairs, elements never seen (continuum generalization)
    T3 -- fresh pairs, positive = exact product vs negatives at fixed geodesic
          radius r (precision profile; beating the training noise floor)

Spectroscopy snapshots (every spec_every steps): spin-FVE of the argument
encoder activations per j in {0,.5,1,1.5,2}, +-g cosine diagnostics, ridge
decode R^2 to the j=1/2 target (quaternion q) and the j=1 target (qq^T).

Usage:
    uv run python -m experiments.su2_train --variant su2 --N 2048 --frac 0.01 \
        --save_dir data/su2/su2_N2048_f0.01_s0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.su2 import (LandmarkEncoding, expmap, haar, qmul, spin_basis,
                      spin_fve)
from core.su2_model import TripleScorer


# ------------------------------ data construction -----------------------------

def class_distance(c, p, variant):
    """Geodesic distance from candidate c to the positive class of product p."""
    dot = np.sum(c * p, axis=-1)
    if variant == "so3":
        dot = np.abs(dot)
    return np.arccos(np.clip(dot, -1.0, 1.0))


def make_candidates(prod, K, sigma, excl_radius, variant, rng):
    """[n, K, 4]: index 0 = noisy positive, rest = Haar negatives (resampled
    until >= excl_radius from the positive class)."""
    n = prod.shape[0]
    xi = rng.standard_normal((n, 3)) * sigma
    pos = qmul(prod, expmap(xi))
    neg = haar(n * (K - 1), rng).reshape(n, K - 1, 4)
    for _ in range(50):
        bad = class_distance(neg, prod[:, None, :], variant) < excl_radius
        if not bad.any():
            break
        neg[bad] = haar(int(bad.sum()), rng)
    return np.concatenate([pos[:, None, :], neg], axis=1).astype(np.float32)


def make_pair_set(qa, qb, K, sigma, excl_radius, variant, rng):
    prod = qmul(qa, qb)
    cands = make_candidates(prod, K, sigma, excl_radius, variant, rng)
    return (qa.astype(np.float32), qb.astype(np.float32), cands)


def build_data(args, rng):
    pool = haar(args.N, rng)
    n_pairs = args.N * args.N
    M = int(round(args.frac * n_pairs))
    n_eval = args.n_eval
    idx = rng.choice(n_pairs, size=M + n_eval, replace=False)
    tr, t1 = idx[:M], idx[M:]
    train = make_pair_set(pool[tr // args.N], pool[tr % args.N], args.K,
                          args.sigma, args.excl_radius, args.variant, rng)
    T1 = make_pair_set(pool[t1 // args.N], pool[t1 % args.N], args.K,
                       args.sigma, args.excl_radius, args.variant, rng)
    T2 = make_pair_set(haar(n_eval, rng), haar(n_eval, rng), args.K,
                       args.sigma, args.excl_radius, args.variant, rng)
    # T3: exact positive, negatives at fixed radius r (no noise on positive)
    T3 = {}
    qa3, qb3 = haar(n_eval, rng), haar(n_eval, rng)
    prod3 = qmul(qa3, qb3)
    for r in args.t3_radii:
        u = rng.standard_normal((n_eval, args.K - 1, 3))
        u = u / np.linalg.norm(u, axis=-1, keepdims=True)
        neg = qmul(prod3[:, None, :], expmap(r * u))
        cands = np.concatenate([prod3[:, None, :], neg], axis=1)
        T3[r] = (qa3.astype(np.float32), qb3.astype(np.float32),
                 cands.astype(np.float32))
    return pool, train, T1, T2, T3


# ------------------------------ torch-side encoding ---------------------------

class TorchEncoding:
    """GPU mirror of core.su2.LandmarkEncoding (verified identical at setup)."""

    def __init__(self, enc: LandmarkEncoding, device):
        self.alpha = enc.alpha
        self.variant = enc.variant
        self.L = torch.tensor(enc.landmarks.T, dtype=torch.float32, device=device)
        self.mean = torch.tensor(enc.mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(enc.std, dtype=torch.float32, device=device)

    def __call__(self, q):
        x = q @ self.L
        if self.variant == "so3":
            x = x.abs()
        return (torch.exp(self.alpha * x) - self.mean) / self.std


# --------------------------------- evaluation ---------------------------------

@torch.no_grad()
def ranking_accuracy(model, psi, pair_set, device, batch=None):
    if batch is None:
        # cap candidate encodes per chunk (~1GB peak); K is pair_set's dim 1
        batch = max(512, 262144 // pair_set[2].shape[1])
    qa, qb, cands = pair_set
    n, correct = qa.shape[0], 0
    for i in range(0, n, batch):
        a = torch.tensor(qa[i:i + batch], device=device)
        b = torch.tensor(qb[i:i + batch], device=device)
        c = torch.tensor(cands[i:i + batch], device=device)
        logits = model(psi(a), psi(b), psi(c))
        correct += (logits.argmax(dim=1) == 0).sum().item()
    return correct / n


def ridge_r2(Z, Y, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Held-out R^2 of ridge Z -> Y with an 80/20 split and lambda grid."""
    n = Z.shape[0]
    n_tr = int(0.8 * n)
    Ztr, Zte = Z[:n_tr], Z[n_tr:]
    Ytr, Yte = Y[:n_tr], Y[n_tr:]
    mu_z, mu_y = Ztr.mean(0), Ytr.mean(0)
    Ztr, Zte, Ytr_c = Ztr - mu_z, Zte - mu_z, Ytr - mu_y
    G = Ztr.T @ Ztr
    best = -np.inf
    for lam in lams:
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Ztr.T @ Ytr_c)
        pred = Zte @ W + mu_y
        ss_res = ((Yte - pred) ** 2).sum()
        ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
        best = max(best, 1.0 - ss_res / ss_tot)
    return float(best)


@torch.no_grad()
def spectroscopy(model, psi, probe_q, bases, device):
    q = torch.tensor(probe_q, dtype=torch.float32, device=device)
    z = model.embed(psi(q)).cpu().numpy()
    z_neg = model.embed(psi(-q)).cpu().numpy()
    out = {f"fve_{j}": v for j, v in spin_fve(z, bases).items()}
    zc = z - z.mean(0)
    znc = z_neg - z_neg.mean(0)
    cos = (zc * znc).sum(1) / (np.linalg.norm(zc, axis=1)
                               * np.linalg.norm(znc, axis=1) + 1e-12)
    out["pm_cos_mean"] = float(cos.mean())
    out["r2_q"] = ridge_r2(z, probe_q)                       # j=1/2 target (odd)
    qq = np.einsum("ni,nj->nij", probe_q, probe_q).reshape(len(probe_q), 16)
    out["r2_qq"] = ridge_r2(z, qq)                           # j=1 target (even)
    return out


# ----------------------------------- driver -----------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["su2", "so3"], required=True)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--excl_radius", type=float, default=0.5)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--enc_seed", type=int, default=0,
                   help="landmark seed, shared across variants (matched control)")
    p.add_argument("--activation", choices=["gelu", "quadratic"], default="gelu")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_eval", type=int, default=8192)
    p.add_argument("--t3_radii", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--spec_every", type=int, default=5000)
    p.add_argument("--ckpt_every", type=int, default=25_000)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--resume", action="store_true",
                   help="resume from the latest ckpt_*.pt in save_dir")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    (save / "config.json").write_text(json.dumps(vars(args), indent=2))
    log_f = open(save / "log.jsonl", "a")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    enc = LandmarkEncoding(m=args.m, alpha=args.alpha, variant=args.variant,
                           seed=args.enc_seed)
    psi = TorchEncoding(enc, device)

    print(f"[{args.save_dir}] building data ...", flush=True)
    pool, train, T1, T2, T3 = build_data(args, rng)
    M = train[0].shape[0]
    print(f"[{args.save_dir}] {M} training pairs", flush=True)

    # frozen training subset for the memorization curve
    sub = slice(0, min(args.n_eval, M))
    train_sub = (train[0][sub], train[1][sub], train[2][sub])

    # spectroscopy probe (fresh Haar, fixed) + orthonormal spin bases
    probe_q = haar(2048, np.random.default_rng(args.seed + 777))
    bases = spin_basis(probe_q)

    # training tensors on GPU
    qa = torch.tensor(train[0], device=device)
    qb = torch.tensor(train[1], device=device)
    qc = torch.tensor(train[2], device=device)

    model = TripleScorer(m=args.m, activation=args.activation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)

    start_step = 1
    if args.resume:
        ckpts = sorted(save.glob("ckpt_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            state = torch.load(ckpts[-1], map_location=device,
                               weights_only=False)
            model.load_state_dict(state["model"])
            if "opt" in state:
                opt.load_state_dict(state["opt"])
            start_step = state["step"] + 1
            print(f"[{args.save_dir}] resumed from {ckpts[-1].name} "
                  f"(step {state['step']})", flush=True)

    t0 = time.time()
    perm = torch.randperm(M, device=device)
    ptr = 0
    for step in range(start_step, args.steps + 1):
        if ptr + args.batch > M:
            perm = torch.randperm(M, device=device)
            ptr = 0
        idx = perm[ptr: ptr + args.batch]
        ptr += args.batch

        logits = model(psi(qa[idx]), psi(qb[idx]), psi(qc[idx]))
        loss = F.cross_entropy(
            logits, torch.zeros(len(idx), dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            rec = {"step": step, "loss": float(loss.item()),
                   "acc_train": ranking_accuracy(model, psi, train_sub, device),
                   "acc_T1": ranking_accuracy(model, psi, T1, device),
                   "acc_T2": ranking_accuracy(model, psi, T2, device)}
            for r, ps in T3.items():
                rec[f"acc_T3_r{r}"] = ranking_accuracy(model, psi, ps, device)
            if step % args.spec_every == 0 or step == 1:
                rec.update(spectroscopy(model, psi, probe_q, bases, device))
            rec["wall_s"] = round(time.time() - t0, 1)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"step {step:>7d}  loss {rec['loss']:.4f}  "
                  f"train {rec['acc_train']:.3f}  T1 {rec['acc_T1']:.3f}  "
                  f"T2 {rec['acc_T2']:.3f}", flush=True)
            model.train()

        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "config": vars(args), "step": step},
                       save / f"ckpt_{step}.pt")

    log_f.close()
    print(f"[{args.save_dir}] done in {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
