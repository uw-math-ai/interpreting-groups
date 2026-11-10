"""Train the triple scorer on Spin(5) = Sp(2) or SO(5) composition.

Same InfoNCE construction as the other drivers; the SO(5) control uses the
center-orbit-averaged observation map and quotient-aware exclusion.
Pre-registered predictions (experiments.carrier_search): spin5 -> spinor 4;
so5 -> vector 5. Spectroscopy logs FVE over sectors {4, 5, 10, 14},
cos(z(U), z(-U)), and held-out decodes to the spinor (odd) and vector (even)
matrix coefficients.

Usage:
    uv run python -m experiments.spin5_train --variant spin5 --N 2048 \
        --frac 0.01 --save_dir data/spin5/spin5_N2048_f0.01_s0
"""
import argparse
import hashlib
from core.so5_conditioning import FrozenCoordinateEncoding

from core.provenance import write_config, check_resume_compat
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.spin5 import (Sp2LandmarkEncoding, chordal, get_spin5_encoding,
                        haar_sp2, qmat_mul, random_algebra, rep_spinor,
                        rep_vector, sector_basis5, sp2_expmap)
from core.su2_model import TripleScorer
from experiments.su2_train import ridge_r2


def make_pair_set(ga, gb, K, sigma, excl, variant, rng):
    n = ga.shape[0]
    prod = qmat_mul(ga, gb)
    pos = qmat_mul(prod, sp2_expmap(random_algebra(n, sigma, rng)))
    neg = haar_sp2(n * (K - 1), rng).reshape(n, K - 1, 2, 2, 4)
    for _ in range(50):
        bad = chordal(neg, prod[:, None], quotient=(variant == "so5")) < excl
        if not bad.any():
            break
        neg[bad] = haar_sp2(int(bad.sum()), rng)
    cands = np.concatenate([pos[:, None], neg], axis=1)
    return (ga.astype(np.float32), gb.astype(np.float32),
            cands.astype(np.float32))


def build_data(args, rng):
    pool = haar_sp2(args.N, rng)
    M = int(round(args.frac * args.N ** 2))
    idx = rng.choice(args.N ** 2, size=M + args.n_eval, replace=False)
    tr, t1 = idx[:M], idx[M:]
    mk = lambda a, b: make_pair_set(a, b, args.K, args.sigma,
                                    args.excl_radius, args.variant, rng)
    train = mk(pool[tr // args.N], pool[tr % args.N])
    T1 = mk(pool[t1 // args.N], pool[t1 % args.N])
    T2 = mk(haar_sp2(args.n_eval, rng), haar_sp2(args.n_eval, rng))
    return train, T1, T2


@torch.no_grad()
def ranking_accuracy(model, psi, pair_set, device, batch=None):
    ga, gb, cands = pair_set
    if batch is None:
        batch = max(256, 131072 // cands.shape[1])
    n, correct = ga.shape[0], 0
    for i in range(0, n, batch):
        a = torch.tensor(ga[i:i + batch], device=device)
        b = torch.tensor(gb[i:i + batch], device=device)
        c = torch.tensor(cands[i:i + batch], device=device)
        logits = model(psi(a), psi(b), psi(c))
        correct += (logits.argmax(1) == 0).sum().item()
    return correct / n


@torch.no_grad()
def spectroscopy(model, psi, probe, bases, device):
    P = torch.tensor(probe, dtype=torch.float32, device=device)
    z = model.embed(psi(P)).cpu().numpy()
    zc = z - z.mean(0)
    tot = (zc ** 2).sum()
    out = {f"fve_{k}": float(((Q.T @ zc) ** 2).sum() / tot)
           for k, Q in bases.items()}
    z2 = model.embed(psi(torch.tensor(-probe, dtype=torch.float32,
                                      device=device))).cpu().numpy()
    z2c = z2 - z2.mean(0)
    out["pm_cos"] = float(np.mean(
        (zc * z2c).sum(1) / (np.linalg.norm(zc, axis=1)
                             * np.linalg.norm(z2c, axis=1) + 1e-12)))
    D = rep_spinor(probe).reshape(len(probe), -1)
    out["r2_spinor"] = ridge_r2(z, np.concatenate([D.real, D.imag], axis=1))
    out["r2_vector"] = ridge_r2(z, rep_vector(probe).reshape(len(probe), -1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["spin5", "so5"], required=True)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--excl_radius", type=float, default=0.7)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--m", type=int, default=48)
    p.add_argument("--encoding", default="landmark",
                   choices=["landmark", "so5vec", "so5rf"],
                   help="so5vec/so5rf are intrinsic SO(5) observations "
                        "(require --variant so5); so5vec is the diagnostic "
                        "upper control, so5rf the generic selection test")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--enc_seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_eval", type=int, default=4096)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--spec_every", type=int, default=5000)
    p.add_argument("--ckpt_every", type=int, default=25_000)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument('--conditioning',choices=['identity','rotate','whiten'],default='identity')
    p.add_argument('--conditioning_seed',type=int,default=2001)
    p.add_argument('--conditioning_n',type=int,default=65536)
    args = p.parse_args()
    if args.variant != 'so5' or args.encoding != 'landmark':
        p.error('This experiment controls only the averaged SO(5) landmark map')
    if args.resume:
        p.error('The declared fixed-budget experiment starts fresh; resume is not used')

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # resolve derived fields BEFORE serializing the config (G0.2)
    enc = FrozenCoordinateEncoding(mode=args.conditioning,m=args.m,alpha=args.alpha,
                                   enc_seed=args.enc_seed,probe_seed=args.conditioning_seed,
                                   n_probe=args.conditioning_n)
    transform_file=save/'observation_transform.npz'
    np.savez_compressed(transform_file,matrix=enc.W)
    args.conditioning_diagnostics=enc.diagnostics
    args.conditioning_transform_sha256=hashlib.sha256(transform_file.read_bytes()).hexdigest()
    args.conditioning_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    requested = {}
    if enc.m != args.m:                  # intrinsic encodings fix their width
        requested["m"] = args.m
        args.m = enc.m
    write_config(save, args, requested)
    log_f = open(save / "log.jsonl", "a")
    psi = enc.torch_fn(device)

    print(f"[{args.save_dir}] building data ...", flush=True)
    train, T1, T2 = build_data(args, rng)
    M = train[0].shape[0]
    print(f"[{args.save_dir}] {M} training pairs", flush=True)
    train_sub = tuple(x[: args.n_eval] for x in train)

    probe = haar_sp2(3072, np.random.default_rng(args.seed + 777))
    bases = sector_basis5(probe)

    qa = torch.tensor(train[0], device=device)
    qb = torch.tensor(train[1], device=device)
    qc = torch.tensor(train[2], device=device)

    model = TripleScorer(m=args.m).to(device)
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
            if step % args.spec_every == 0 or step == 1:
                rec.update(spectroscopy(model, psi, probe, bases, device))
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
