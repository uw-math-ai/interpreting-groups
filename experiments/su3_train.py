"""Train the triple scorer on SU(3) (or PSU(3)-quotient) composition.

Mirrors experiments.su2_train (InfoNCE candidate ranking, pool-and-fraction
data, frozen candidates) with SU(3) machinery from core.su3. The PSU(3)
control uses the center-orbit-averaged observation map (exactly Z_3-invariant,
same landmarks and dimension); candidate exclusion and the positive class are
quotient-aware.

Diagnostics logged during training:
  - sector FVE of the argument encoder (and candidate encoder) over the
    conjugate-pair sectors (0,0), (1,0)=3+3bar, (1,1)=8, (2,0)=6+6bar,
    (3,0)=10+10bar;
  - center diagnostics: cos(z(U), z(omega U)); charged-energy fraction
    ||z - avg_k z(omega^k U)||^2 / ||z - mean||^2; center discrimination
    D_omega = P[s(a,b,ab) > max_k s(a,b, omega^k ab)];
  - held-out ridge decodes to the fundamental matrix entries (charged target)
    and to the adjoint matrix (center-trivial target).

Usage:
    uv run python -m experiments.su3_train --variant su3 --N 2048 --frac 0.01 \
        --save_dir data/su3/su3_N2048_f0.01_s0
"""
import argparse

from core.provenance import write_config, check_resume_compat
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.su3 import (OMEGA, SU3LandmarkEncoding, chordal, haar_su3, inv, mm,
                      pack_real, sector_basis, sector_fve, su3_expmap, su3_rep)
from core.su2_model import TripleScorer
from experiments.su2_train import ridge_r2


# ------------------------------ data construction -----------------------------

def make_candidates(prod, K, sigma, excl_radius, variant, rng):
    n = prod.shape[0]
    xi = rng.standard_normal((n, 8)) * sigma
    pos = mm(prod, su3_expmap(xi))
    neg = haar_su3(n * (K - 1), rng).reshape(n, K - 1, 3, 3)
    for _ in range(50):
        bad = chordal(neg, prod[:, None], quotient=(variant == "psu3")) \
            < excl_radius
        if not bad.any():
            break
        neg[bad] = haar_su3(int(bad.sum()), rng)
    return np.concatenate([pos[:, None], neg], axis=1)


def make_pair_set(qa, qb, K, sigma, excl_radius, variant, rng):
    cands = make_candidates(mm(qa, qb), K, sigma, excl_radius, variant, rng)
    return (pack_real(qa), pack_real(qb),
            pack_real(cands.reshape(-1, 3, 3)).reshape(
                qa.shape[0], K, 2, 3, 3))


def build_data(args, rng):
    pool = haar_su3(args.N, rng)
    M = int(round(args.frac * args.N ** 2))
    idx = rng.choice(args.N ** 2, size=M + args.n_eval, replace=False)
    tr, t1 = idx[:M], idx[M:]
    mk = lambda qa, qb: make_pair_set(qa, qb, args.K, args.sigma,
                                      args.excl_radius, args.variant, rng)
    train = mk(pool[tr // args.N], pool[tr % args.N])
    T1 = mk(pool[t1 // args.N], pool[t1 % args.N])
    T2 = mk(haar_su3(args.n_eval, rng), haar_su3(args.n_eval, rng))
    # T3: exact product vs negatives at fixed algebra radius r
    T3 = {}
    qa3, qb3 = haar_su3(args.n_eval, rng), haar_su3(args.n_eval, rng)
    prod3 = mm(qa3, qb3)
    for r in args.t3_radii:
        u = rng.standard_normal((args.n_eval, args.K - 1, 8))
        u = r * u / np.linalg.norm(u, axis=-1, keepdims=True)
        neg = mm(prod3[:, None], su3_expmap(u.reshape(-1, 8)).reshape(
            args.n_eval, args.K - 1, 3, 3))
        cands = np.concatenate([prod3[:, None], neg], axis=1)
        T3[r] = (pack_real(qa3), pack_real(qb3),
                 pack_real(cands.reshape(-1, 3, 3)).reshape(
                     args.n_eval, args.K, 2, 3, 3))
    return pool, train, T1, T2, T3


# --------------------------------- evaluation ---------------------------------

@torch.no_grad()
def ranking_accuracy(model, psi, pair_set, device, batch=None):
    qa, qb, cands = pair_set[:3]
    if batch is None:
        batch = max(256, 131072 // cands.shape[1])
    n, correct = qa.shape[0], 0
    for i in range(0, n, batch):
        a = torch.tensor(qa[i:i + batch], device=device)
        b = torch.tensor(qb[i:i + batch], device=device)
        c = torch.tensor(cands[i:i + batch], device=device)
        B, K = c.shape[:2]
        logits = torch.einsum(
            "bd,bkd->bk",
            model.interact(torch.cat([model.enc(psi(a)), model.enc(psi(b))],
                                     dim=-1)),
            model.enc_c(psi(c.reshape(B * K, *c.shape[2:])).reshape(B, K, -1)))
        correct += (logits.argmax(dim=1) == 0).sum().item()
    return correct / n


@torch.no_grad()
def spectroscopy(model, psi, probe_U, bases, device):
    P = torch.tensor(pack_real(probe_U), device=device)
    z = model.embed(psi(P)).cpu().numpy()
    out = {f"fve_{pq[0]}{pq[1]}": v for pq, v in sector_fve(z, bases).items()}
    zc_enc = model.enc_c(psi(P)).cpu().numpy()
    out.update({f"fve_c_{pq[0]}{pq[1]}": v
                for pq, v in sector_fve(zc_enc, bases).items()})
    # center diagnostics
    zs = [model.embed(psi(torch.tensor(pack_real((OMEGA ** k) * probe_U),
                                       device=device))).cpu().numpy()
          for k in range(3)]
    z0c = zs[0] - zs[0].mean(0)
    z1c = zs[1] - zs[1].mean(0)
    cos = (z0c * z1c).sum(1) / (np.linalg.norm(z0c, axis=1)
                                * np.linalg.norm(z1c, axis=1) + 1e-12)
    out["omega_cos_mean"] = float(cos.mean())
    avg = (zs[0] + zs[1] + zs[2]) / 3
    out["charged_frac"] = float(((zs[0] - avg) ** 2).sum()
                                / ((zs[0] - zs[0].mean(0)) ** 2).sum())
    # decodes: fundamental entries (charged) vs adjoint entries (trivial charge)
    fund = np.concatenate([probe_U.reshape(-1, 9).real,
                           probe_U.reshape(-1, 9).imag], axis=1)
    out["r2_fund"] = ridge_r2(z, fund)
    adj = su3_rep(probe_U, (1, 1)).real.reshape(len(probe_U), 64)
    out["r2_adj"] = ridge_r2(z, adj)
    return out


# ----------------------------------- driver -----------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["su3", "psu3"], required=True)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--excl_radius", type=float, default=0.7,
                   help="chordal exclusion radius for negatives")
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--m", type=int, default=32,
                   help="landmarks; feature dim is 2m")
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--enc_seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--t3_radii", type=float, nargs="+",
                   default=[0.1, 0.2, 0.4, 0.6, 1.0])
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--spec_every", type=int, default=5000)
    p.add_argument("--ckpt_every", type=int, default=25_000)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    write_config(save, args)   # no derived fields in this trainer (G0.2)
    log_f = open(save / "log.jsonl", "a")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    enc = SU3LandmarkEncoding(m=args.m, alpha=args.alpha, variant=args.variant,
                              seed=args.enc_seed)
    psi = enc.torch_fn(device)

    print(f"[{args.save_dir}] building data ...", flush=True)
    pool, train, T1, T2, T3 = build_data(args, rng)
    M = train[0].shape[0]
    print(f"[{args.save_dir}] {M} training pairs", flush=True)
    sub = slice(0, min(args.n_eval, M))
    train_sub = tuple(x[sub] for x in train)

    probe_U = haar_su3(3072, np.random.default_rng(args.seed + 777))
    bases = sector_basis(probe_U)

    qa = torch.tensor(train[0], device=device)
    qb = torch.tensor(train[1], device=device)
    qc = torch.tensor(train[2], device=device)

    model = TripleScorer(m=2 * args.m).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)

    start_step = 1
    if args.resume:
        ckpts = sorted(save.glob("ckpt_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            state = torch.load(ckpts[-1], map_location=device,
                               weights_only=False)
            check_resume_compat(state, args)
            model.load_state_dict(state["model"])
            if "opt" in state:
                opt.load_state_dict(state["opt"])
            start_step = state["step"] + 1
            print(f"[{args.save_dir}] resumed from {ckpts[-1].name}",
                  flush=True)

    t0 = time.time()
    perm = torch.randperm(M, device=device)
    ptr = 0
    for step in range(start_step, args.steps + 1):
        if ptr + args.batch > M:
            perm = torch.randperm(M, device=device)
            ptr = 0
        idx = perm[ptr: ptr + args.batch]
        ptr += args.batch

        c = qc[idx]
        B, K = c.shape[:2]
        logits = torch.einsum(
            "bd,bkd->bk",
            model.interact(torch.cat([model.enc(psi(qa[idx])),
                                      model.enc(psi(qb[idx]))], dim=-1)),
            model.enc_c(psi(c.reshape(B * K, *c.shape[2:])).reshape(B, K, -1)))
        loss = F.cross_entropy(
            logits, torch.zeros(B, dtype=torch.long, device=device))
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
                rec.update(spectroscopy(model, psi, probe_U, bases, device))
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
