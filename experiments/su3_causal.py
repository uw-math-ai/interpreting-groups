"""Causal analyses of trained SU(3)/PSU(3) composition models (plan B3/B4).

1. Center-charge and sector interventions on the argument encoders:
     - center_projection: z <- (1/3) sum_k z(omega^k U)  (removes all Z_3-charged
       content; an exact no-op for the PSU(3) control)
     - charged_projection: z <- z - center_projection    (removes invariant content)
     - fablate_fund: subtract the component of z linearly predictable from the
       realified (1,0) matrix coefficients (the 3+3bar pair sector)
     - fablate_adj: likewise for the adjoint (1,1)
     - rank-matched random control.
   Metrics: ranking accuracy vs 63 excluded Haar negatives ("task"); accuracy vs
   63 negatives at algebra radius 0.4 ("fine"); center discrimination
   D_omega = P[s(a,b,ab) > max_k s(a,b, omega^k ab)] (chance 1/3).

2. Score-character fit (plan B4): regress s(a,b,c) on Re/Im chi_(p,q)(ab c^-1)
   cumulatively in Casimir order.

3. Composition test: decode the fundamental U (SU(3)) or the adjoint matrix
   (PSU(3)) from z and from h; check the algebra composes on held-out pairs.

Usage:
    uv run python -m experiments.su3_causal --run_dir data/su3/su3_N2048_f0.01_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.su2_model import TripleScorer
from core.su3 import (OMEGA, SU3LandmarkEncoding, chordal, haar_su3, inv, mm,
                      pack_real, su3_character, su3_expmap, su3_rep)

SECT_FIT = [(0, 0), (1, 0), (1, 1), (2, 0), (3, 0)]   # Casimir order (pairs)


def load_model(run_dir, ckpt_name=None, device="cpu"):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    if ckpt_name is None:
        ckpts = sorted(run_dir.glob("ckpt_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        ckpt_name = ckpts[-1].name
    state = torch.load(run_dir / ckpt_name, map_location=device,
                       weights_only=False)
    model = TripleScorer(m=2 * cfg["m"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    enc = SU3LandmarkEncoding(m=cfg["m"], alpha=cfg["alpha"],
                              variant=cfg["variant"], seed=cfg["enc_seed"])
    return model, enc, cfg, state["step"]


def embed_np(model, enc, U, device, batch=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(U), batch):
            out.append(model.embed(torch.tensor(
                enc(U[i:i + batch]), device=device)).cpu().numpy())
    return np.concatenate(out)


def realified(U, pq):
    D = su3_rep(U, pq).reshape(len(U), -1)
    return np.concatenate([D.real, D.imag], axis=1)


def make_interventions(model, enc, cfg, device, probe_n=4096, seed=123):
    rng = np.random.default_rng(seed)
    probe = haar_su3(probe_n, rng)
    Z = embed_np(model, enc, probe, device)
    mu, Zc = Z.mean(0), Z - Z.mean(0)
    d = Z.shape[1]

    def identity_fn(U):
        return embed_np(model, enc, U, device)

    def center_avg(U):
        return sum(embed_np(model, enc, (OMEGA ** k) * U, device)
                   for k in range(3)) / 3

    def charged(U):
        return identity_fn(U) - center_avg(U) + mu

    def function_ablate(pq):
        F = realified(probe, pq)
        muF, Fc = F.mean(0), F - F.mean(0)
        W = np.linalg.solve(Fc.T @ Fc + 1e-6 * np.eye(F.shape[1]), Fc.T @ Zc)

        def fn(U):
            z = embed_np(model, enc, U, device)
            return z - (realified(U, pq) - muF) @ W
        return fn

    V = np.linalg.qr(rng.standard_normal((d, 18)))[0]

    def random_ablate(U):
        z = embed_np(model, enc, U, device)
        return z - (z - mu) @ V @ V.T

    return [("none", identity_fn), ("center_projection", center_avg),
            ("charged_projection", charged),
            ("fablate_fund", function_ablate((1, 0))),
            ("fablate_adj", function_ablate((1, 1))),
            ("ablate_random_k18", random_ablate)]


@torch.no_grad()
def score_with(model, enc, fn, Ua, Ub, Uc, device, batch=2048):
    out = []
    n, K = Uc.shape[:2]
    for i in range(0, n, batch):
        za = torch.tensor(fn(Ua[i:i + batch]), dtype=torch.float32,
                          device=device)
        zb = torch.tensor(fn(Ub[i:i + batch]), dtype=torch.float32,
                          device=device)
        h = model.interact(torch.cat([za, zb], dim=-1))
        c = Uc[i:i + batch].reshape(-1, 3, 3)
        zc = model.enc_c(torch.tensor(enc(c), device=device)
                         ).reshape(-1, K, model.enc_c[-1].out_features)
        out.append(torch.einsum("bd,bkd->bk", h, zc).cpu().numpy())
    return np.concatenate(out)


def run_interventions(model, enc, cfg, device, n=4096, seed=7):
    rng = np.random.default_rng(seed)
    Ua, Ub = haar_su3(n, rng), haar_su3(n, rng)
    prod = mm(Ua, Ub)
    quot = cfg["variant"] == "psu3"

    neg = haar_su3(n * 63, rng).reshape(n, 63, 3, 3)
    for _ in range(50):
        bad = chordal(neg, prod[:, None], quotient=quot) < cfg["excl_radius"]
        if not bad.any():
            break
        neg[bad] = haar_su3(int(bad.sum()), rng)
    task_c = np.concatenate([prod[:, None], neg], axis=1)

    u = rng.standard_normal((n, 63, 8))
    u = 0.4 * u / np.linalg.norm(u, axis=-1, keepdims=True)
    fine_c = np.concatenate(
        [prod[:, None],
         mm(prod[:, None], su3_expmap(u.reshape(-1, 8)).reshape(n, 63, 3, 3))],
        axis=1)

    center_c = np.stack([prod, OMEGA * prod, OMEGA ** 2 * prod], axis=1)

    results = {}
    for name, fn in make_interventions(model, enc, cfg, device):
        acc = float((score_with(model, enc, fn, Ua, Ub, task_c, device)
                     .argmax(1) == 0).mean())
        fine = float((score_with(model, enc, fn, Ua, Ub, fine_c, device)
                      .argmax(1) == 0).mean())
        sc = score_with(model, enc, fn, Ua, Ub, center_c, device)
        # tie-aware: exactly equal scores across the center orbit (as for any
        # center-invariant model) must read as chance = 1/3, not as a win
        winners = sc >= sc.max(1, keepdims=True) - 1e-6
        dom = float((winners[:, 0] / winners.sum(1)).mean())
        results[name] = {"task_acc": acc, "fine_acc_r0.4": fine,
                         "D_omega": dom}
    return results


def run_character_fit(model, enc, cfg, device, n=8192, seed=11):
    rng = np.random.default_rng(seed)
    Ua, Ub = haar_su3(n, rng), haar_su3(n, rng)
    prod = mm(Ua, Ub)
    r = rng.uniform(0.0, 1.5, size=n)
    u = rng.standard_normal((n, 8))
    u = r[:, None] * u / np.linalg.norm(u, axis=-1, keepdims=True)
    c_near = mm(prod, su3_expmap(u))
    pick = rng.random(n) < 0.5
    c = np.where(pick[:, None, None], c_near, haar_su3(n, rng))

    s = score_with(model, enc,
                   lambda U: embed_np(model, enc, U, device),
                   Ua, Ub, c[:, None], device)[:, 0]
    rel = mm(prod, inv(c))
    y = s - s.mean()
    out = {}
    cols = []
    for pq in SECT_FIT:
        chi = su3_character(rel, pq)
        cols += [chi.real, chi.imag]
        X = np.stack(cols, axis=1)
        X = X[:, X.std(0) > 1e-9]
        Xc = X - X.mean(0)
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        out[f"r2_upto_{pq[0]}{pq[1]}"] = float(
            1 - ((y - Xc @ beta) ** 2).sum() / (y ** 2).sum())
    return out


def run_composition(model, enc, cfg, device, n_train=8192, n_test=2048,
                    seed=13):
    rng = np.random.default_rng(seed)
    pq = (1, 0) if cfg["variant"] == "su3" else (1, 1)

    def target(U):
        return realified(U, pq)

    U_el = haar_su3(n_train, rng)
    Z = embed_np(model, enc, U_el, device)
    mz, my = Z.mean(0), target(U_el).mean(0)
    Wz = np.linalg.solve((Z - mz).T @ (Z - mz) + 1.0 * np.eye(Z.shape[1]),
                         (Z - mz).T @ (target(U_el) - my))

    Ua, Ub = haar_su3(n_train, rng), haar_su3(n_train, rng)
    with torch.no_grad():
        H = np.concatenate([model.interact(torch.cat(
            [model.embed(torch.tensor(enc(Ua[i:i + 4096]), device=device)),
             model.embed(torch.tensor(enc(Ub[i:i + 4096]), device=device))],
            dim=-1)).cpu().numpy() for i in range(0, n_train, 4096)])
    Yh = target(mm(Ua, Ub))
    mh, mhy = H.mean(0), Yh.mean(0)
    Wh = np.linalg.solve((H - mh).T @ (H - mh) + 1.0 * np.eye(H.shape[1]),
                         (H - mh).T @ (Yh - mhy))

    Ta, Tb = haar_su3(n_test, rng), haar_su3(n_test, rng)
    prod = mm(Ta, Tb)
    with torch.no_grad():
        Ht = np.concatenate([model.interact(torch.cat(
            [model.embed(torch.tensor(enc(Ta[i:i + 4096]), device=device)),
             model.embed(torch.tensor(enc(Tb[i:i + 4096]), device=device))],
            dim=-1)).cpu().numpy() for i in range(0, n_test, 4096)])
    pred_h = (Ht - mh) @ Wh + mhy

    def r2(pred, true):
        return float(1 - ((true - pred) ** 2).sum()
                     / ((true - true.mean(0)) ** 2).sum())

    out = {"r2_h_to_product": r2(pred_h, target(prod))}
    perm = rng.permutation(n_test)
    out["r2_h_to_product_shuffled"] = r2(pred_h, target(mm(Ta[perm], Tb)))

    # algebraic composition of element probes (matrix product in the rep)
    d = int(np.sqrt(target(prod).shape[1] // 2))
    def unpack(Y):
        half = Y.shape[1] // 2
        return (Y[:, :half] + 1j * Y[:, half:]).reshape(-1, d, d)
    Da = unpack((embed_np(model, enc, Ta, device) - mz) @ Wz + my)
    Db = unpack((embed_np(model, enc, Tb, device) - mz) @ Wz + my)
    comp = np.einsum("nij,njk->nik", Da, Db).reshape(n_test, -1)
    true = su3_rep(prod, pq).reshape(n_test, -1)
    cos = np.real(np.sum(comp * np.conj(true), axis=1)) / (
        np.linalg.norm(comp, axis=1) * np.linalg.norm(true, axis=1) + 1e-12)
    out["compose_vs_truth_cos_med"] = float(np.median(cos))
    comp_sh = np.einsum("nij,njk->nik", Da[perm], Db).reshape(n_test, -1)
    cos_sh = np.real(np.sum(comp_sh * np.conj(true), axis=1)) / (
        np.linalg.norm(comp_sh, axis=1) * np.linalg.norm(true, axis=1) + 1e-12)
    out["compose_vs_truth_cos_med_shuffled"] = float(np.median(cos_sh))
    out["element_probe_target"] = f"{pq}"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, enc, cfg, step = load_model(args.run_dir, args.ckpt, args.device)
    print(f"== {args.run_dir} ({cfg['variant']}, step {step})")
    res = {"variant": cfg["variant"], "step": step}

    res["interventions"] = run_interventions(model, enc, cfg, args.device)
    print(f"{'intervention':<22} {'task':>6} {'r=0.4':>6} {'D_om':>6}")
    for name, r in res["interventions"].items():
        print(f"{name:<22} {r['task_acc']:>6.3f} {r['fine_acc_r0.4']:>6.3f} "
              f"{r['D_omega']:>6.3f}")

    res["character_fit"] = run_character_fit(model, enc, cfg, args.device)
    print("char fit:", {k: round(v, 4) for k, v in res["character_fit"].items()})

    res["composition"] = run_composition(model, enc, cfg, args.device)
    print("composition:", {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in res["composition"].items()})

    out = Path(args.run_dir) / f"causal_{step}.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
