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

from core.su2 import (LandmarkEncoding, expmap, get_encoding, haar,
                      heat_kernel, qconj, qmul, spin_basis, spin_fve)
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
    if args.loss in ("heatkernel", "bandlim"):
        target = None
        if args.loss == "bandlim":
            from core.su2 import wigner_character
            js, cs = args.bl_js, args.bl_coefs
            target = lambda rel: sum(
                c * wigner_character(j, rel) for j, c in zip(js, cs))
        train = make_hk_set(pool[tr // args.N], pool[tr % args.N], args.K,
                            args.hk_t, rng, target)
        T1 = make_hk_set(pool[t1 // args.N], pool[t1 % args.N], args.K,
                         args.hk_t, rng, target)
        T2 = make_hk_set(haar(n_eval, rng), haar(n_eval, rng), args.K,
                         args.hk_t, rng, target)
        # standardize targets with train-set statistics
        mu, sd = train[3].mean(), train[3].std() + 1e-8
        train = train[:3] + (((train[3] - mu) / sd).astype(np.float32),)
        T1 = T1[:3] + (((T1[3] - mu) / sd).astype(np.float32),)
        T2 = T2[:3] + (((T2[3] - mu) / sd).astype(np.float32),)
    else:
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


# --------------------------- heat-kernel regression data ----------------------

def make_hk_set(qa, qb, K, t, rng, target=None):
    """Candidates covering the kernel support (half near the product at uniform
    radius in (0, pi/2), half Haar) with targets y = target(ab c^-1)
    (default: the heat kernel K_t)."""
    n = qa.shape[0]
    prod = qmul(qa, qb)
    r = rng.uniform(0.0, np.pi / 2, size=(n, K, 1))
    u = rng.standard_normal((n, K, 3))
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    near = qmul(prod[:, None, :], expmap(r * u))
    far = haar(n * K, rng).reshape(n, K, 4)
    pick = rng.random((n, K, 1)) < 0.5
    c = np.where(pick, near, far)
    rel = qmul(prod[:, None, :], qconj(c)).reshape(-1, 4)
    y = (heat_kernel(rel, t) if target is None else target(rel)).reshape(n, K)
    return (qa.astype(np.float32), qb.astype(np.float32),
            c.astype(np.float32), y.astype(np.float32))


# --------------------------------- evaluation ---------------------------------

@torch.no_grad()
def ranking_accuracy(model, psi, pair_set, device, batch=None):
    if batch is None:
        # cap candidate encodes per chunk (~1GB peak); K is pair_set's dim 1
        batch = max(512, 262144 // pair_set[2].shape[1])
    qa, qb, cands = pair_set[:3]
    n, correct = qa.shape[0], 0
    for i in range(0, n, batch):
        a = torch.tensor(qa[i:i + batch], device=device)
        b = torch.tensor(qb[i:i + batch], device=device)
        c = torch.tensor(cands[i:i + batch], device=device)
        logits = model(psi(a), psi(b), psi(c))
        correct += (logits.argmax(dim=1) == 0).sum().item()
    return correct / n


@torch.no_grad()
def regression_mse(model, psi, hk_set, device, batch=None):
    qa, qb, cands, y = hk_set
    if batch is None:
        batch = max(512, 262144 // cands.shape[1])
    n, se = qa.shape[0], 0.0
    for i in range(0, n, batch):
        a = torch.tensor(qa[i:i + batch], device=device)
        b = torch.tensor(qb[i:i + batch], device=device)
        c = torch.tensor(cands[i:i + batch], device=device)
        pred = model(psi(a), psi(b), psi(c))
        se += ((pred - torch.tensor(y[i:i + batch], device=device)) ** 2
               ).sum().item()
    return se / (n * cands.shape[1])


def ridge_r2(Z, Y, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Held-out R^2 of ridge Z -> Y with a 60/20/20 train/validation/test
    split: lambda is selected on the validation split and R^2 reported on the
    disjoint test split (plan P0 — no hyperparameter selection on the
    reporting split)."""
    n = Z.shape[0]
    n_tr, n_va = int(0.6 * n), int(0.8 * n)
    Ztr, Zva, Zte = Z[:n_tr], Z[n_tr:n_va], Z[n_va:]
    Ytr, Yva, Yte = Y[:n_tr], Y[n_tr:n_va], Y[n_va:]
    mu_z, mu_y = Ztr.mean(0), Ytr.mean(0)
    Ztr, Zva, Zte = Ztr - mu_z, Zva - mu_z, Zte - mu_z
    Ytr_c = Ytr - mu_y
    G = Ztr.T @ Ztr
    XtY = Ztr.T @ Ytr_c

    def r2_of(W, Zs, Ys):
        pred = Zs @ W + mu_y
        return 1.0 - ((Ys - pred) ** 2).sum() / ((Ys - Ys.mean(0)) ** 2).sum()

    best_lam, best_val = lams[0], -np.inf
    Ws = {}
    for lam in lams:
        Ws[lam] = np.linalg.solve(G + lam * np.eye(G.shape[0]), XtY)
        v = r2_of(Ws[lam], Zva, Yva)
        if v > best_val:
            best_lam, best_val = lam, v
    return float(r2_of(Ws[best_lam], Zte, Yte))


@torch.no_grad()
def spectroscopy(model, psi, probe_q, bases, device):
    q = torch.tensor(probe_q, dtype=torch.float32, device=device)
    z = model.embed(psi(q)).cpu().numpy()
    z_neg = model.embed(psi(-q)).cpu().numpy()
    out = {f"fve_{j}": v for j, v in spin_fve(z, bases).items()}
    zc = model.enc_c(psi(q)).cpu().numpy()
    out.update({f"fve_c_{j}": v for j, v in spin_fve(zc, bases).items()})
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
    p.add_argument("--encoding",
                   choices=["landmark", "randmlp", "funcval", "so3frame",
                            "so3block"],
                   default="landmark")
    p.add_argument("--loss", choices=["infonce", "heatkernel", "bandlim"],
                   default="infonce")
    p.add_argument("--hk_t", type=float, default=0.05,
                   help="heat-kernel temperature for --loss heatkernel")
    p.add_argument("--bl_js", type=float, nargs="+", default=[1.0, 2.0],
                   help="band-limited target support (--loss bandlim)")
    p.add_argument("--bl_coefs", type=float, nargs="+", default=[1.0, 0.5])
    p.add_argument("--spec_jmax", type=float, default=2.0,
                   help="largest spin tracked by the FVE spectroscopy")
    p.add_argument("--activation", choices=["gelu", "quadratic"], default="gelu")
    p.add_argument("--d_enc", type=int, default=128)
    p.add_argument("--enc_hidden", type=int, default=256)
    p.add_argument("--mlp_hidden", type=int, default=512)
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

    if args.loss == "heatkernel":
        assert args.variant == "su2", "heat-kernel target is defined on SU(2)"
    if args.loss == "bandlim":
        assert args.variant == "su2"

    enc = get_encoding(args.encoding, args.variant, m=args.m, alpha=args.alpha,
                       seed=args.enc_seed)
    args.m = enc.m                      # explicit encodings fix their own width
    psi = enc.torch_fn(device)

    print(f"[{args.save_dir}] building data ...", flush=True)
    pool, train, T1, T2, T3 = build_data(args, rng)
    M = train[0].shape[0]
    print(f"[{args.save_dir}] {M} training pairs", flush=True)

    # frozen training subset for the memorization curve
    sub = slice(0, min(args.n_eval, M))
    train_sub = tuple(x[sub] for x in train)

    # spectroscopy probe (fresh Haar, fixed) + orthonormal spin bases
    probe_q = haar(2048, np.random.default_rng(args.seed + 777))
    bases = spin_basis(probe_q, j_list=tuple(
        np.arange(0, args.spec_jmax + 0.25, 0.5)))

    # training tensors on GPU
    qa = torch.tensor(train[0], device=device)
    qb = torch.tensor(train[1], device=device)
    qc = torch.tensor(train[2], device=device)
    qy = (torch.tensor(train[3], device=device)
          if args.loss in ("heatkernel", "bandlim") else None)

    model = TripleScorer(m=args.m, d_enc=args.d_enc, enc_hidden=args.enc_hidden,
                         mlp_hidden=args.mlp_hidden,
                         activation=args.activation).to(device)
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
        if args.loss in ("heatkernel", "bandlim"):
            loss = F.mse_loss(logits, qy[idx])
        else:
            loss = F.cross_entropy(
                logits, torch.zeros(len(idx), dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            rec = {"step": step, "loss": float(loss.item())}
            if args.loss in ("heatkernel", "bandlim"):
                rec["mse_train"] = regression_mse(model, psi, train_sub, device)
                rec["mse_T1"] = regression_mse(model, psi, T1, device)
                rec["mse_T2"] = regression_mse(model, psi, T2, device)
                head = (f"mse train {rec['mse_train']:.4f}  "
                        f"T1 {rec['mse_T1']:.4f}  T2 {rec['mse_T2']:.4f}")
            else:
                rec["acc_train"] = ranking_accuracy(model, psi, train_sub, device)
                rec["acc_T1"] = ranking_accuracy(model, psi, T1, device)
                rec["acc_T2"] = ranking_accuracy(model, psi, T2, device)
                head = (f"train {rec['acc_train']:.3f}  T1 {rec['acc_T1']:.3f}  "
                        f"T2 {rec['acc_T2']:.3f}")
            for r, ps in T3.items():
                rec[f"acc_T3_r{r}"] = ranking_accuracy(model, psi, ps, device)
            for mod, name in [(model.enc, "enc"), (model.interact, "int"),
                              (model.enc_c, "encc")]:
                rec[f"norm_{name}"] = float(torch.sqrt(sum(
                    (p ** 2).sum() for p in mod.parameters())).item())
            if args.loss == "infonce":
                with torch.no_grad():
                    a = torch.tensor(T2[0][:2048], device=device)
                    b = torch.tensor(T2[1][:2048], device=device)
                    c = torch.tensor(T2[2][:2048], device=device)
                    lg = model(psi(a), psi(b), psi(c))
                    rec["margin_T2"] = float(
                        (lg[:, 0] - lg[:, 1:].max(1).values).mean().item())
            if step % args.spec_every == 0 or step == 1:
                rec.update(spectroscopy(model, psi, probe_q, bases, device))
            rec["wall_s"] = round(time.time() - t0, 1)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"step {step:>7d}  loss {rec['loss']:.4f}  {head}", flush=True)
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
