"""Causal and mechanistic analyses of trained SU(2)/SO(3) composition models.

Given a checkpoint from experiments.su2_train, runs:

1. Center-discrimination and ablations.
   Metric D_c = P[s(a,b,ab) > s(a,b,-ab)] over fresh Haar pairs: does the model
   prefer the true product to its antipode? Interventions applied to the
   *argument* encoder outputs z_a, z_b (candidate encoder untouched):
     - even-projection  z(g) <- (z(g) + z(-g))/2   (removes all odd/half-integer
       content as a function on the group; basis-free)
     - odd-projection   z(g) <- (z(g) - z(-g))/2   (removes all even content)
     - spin-j subspace ablation: remove the activation-space directions carrying
       the sampled spin-j component (SVD of Q_j^T Z), for j = 1/2 and j = 1
     - rank-matched controls: top-k PCA directions; random k-dim subspace.
   Each intervention reports: ranking accuracy on fresh pairs (task), fine
   accuracy at r = 0.2 (T3-style), and D_c.

2. Score-character fit.
   Regress the scorer s(a,b,c) on the characters {chi_j(ab c^-1)}, j <= 3, over
   triples with c at mixed distances from ab. Reports cumulative R^2 per j and
   the fitted coefficients (to compare against the heat-kernel envelope
   (2j+1) e^{-t j(j+1)} implied by the training noise).

3. Composition test.
   Ridge probes q_hat(a) from z_a and q_hat(ab) from h(a,b) (SU(2) variant;
   rotation matrices R_hat for the SO(3) variant). On held-out pairs of fresh
   elements, tests whether qmul(q_hat(a), q_hat(b)) matches q_hat_h(ab) --
   i.e. whether the product locus h carries the *composed* representation --
   against a pair-shuffled control.

Usage:
    uv run python -m experiments.su2_causal --run_dir data/su2/su2_N2048_f0.01_s0 \
        [--ckpt ckpt_200000.pt] [--n 8192] [--out causal.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.su2 import (LandmarkEncoding, expmap, get_encoding, haar, qmul,
                      spin_basis, wigner_character)
from core.su2_model import TripleScorer


# ----------------------------- loading and encoding ---------------------------

def load_model(run_dir, ckpt_name=None, device="cpu"):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    if ckpt_name is None:
        ckpts = sorted(run_dir.glob("ckpt_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        ckpt_name = ckpts[-1].name
    state = torch.load(run_dir / ckpt_name, map_location=device,
                       weights_only=False)
    model = TripleScorer(m=cfg["m"], d_enc=cfg.get("d_enc", 128),
                         enc_hidden=cfg.get("enc_hidden", 256),
                         mlp_hidden=cfg.get("mlp_hidden", 512),
                         activation=cfg.get("activation", "gelu"))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    enc = get_encoding(cfg.get("encoding", "landmark"), cfg["variant"],
                       m=cfg["m"], alpha=cfg["alpha"], seed=cfg["enc_seed"])
    return model, enc, cfg, state["step"]


def embed_np(model, enc, q, device):
    with torch.no_grad():
        return model.embed(
            torch.tensor(enc(q), device=device)).cpu().numpy()


# ------------------------- intervention machinery -----------------------------

class Intervention:
    """A map applied to argument-encoder activations z(g) (may use g itself)."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def __call__(self, model, enc, q, device):
        return self.fn(model, enc, q, device)


def make_interventions(model, enc, cfg, device, probe_n=4096, seed=123):
    """Build the intervention suite. Subspaces are fit on an independent probe."""
    rng = np.random.default_rng(seed)
    probe = haar(probe_n, rng)
    Z = embed_np(model, enc, probe, device)
    mu = Z.mean(axis=0)
    Zc = Z - mu
    d = Z.shape[1]
    bases = spin_basis(probe, j_list=(0.5, 1.0))

    def subspace_of(j, var_frac=0.99):
        C = bases[j].T @ Zc                        # [r_j, d]
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        k = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), var_frac)) + 1
        return Vt[:k].T                            # [d, k]

    V_half = subspace_of(0.5)
    V_one = subspace_of(1.0)
    k = V_half.shape[1] if cfg['variant'] == 'su2' else V_one.shape[1]
    # rank-matched controls
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    V_pca = Vt[:k].T
    V_rand = np.linalg.qr(rng.standard_normal((d, k)))[0]

    def project_out(V):
        Vt_t = torch.tensor(V, dtype=torch.float32, device=device)
        mu_t = torch.tensor(mu, dtype=torch.float32, device=device)

        def fn(model, enc, q, device):
            with torch.no_grad():
                z = model.embed(torch.tensor(enc(q), device=device))
                zc = z - mu_t
                return z - zc @ Vt_t @ Vt_t.T
        return fn

    def parity(sign):
        def fn(model, enc, q, device):
            with torch.no_grad():
                z1 = model.embed(torch.tensor(enc(q), device=device))
                z2 = model.embed(torch.tensor(enc(-q), device=device))
                return (z1 + sign * z2) / 2
        return fn

    def identity_fn(model, enc, q, device):
        with torch.no_grad():
            return model.embed(torch.tensor(enc(q), device=device))

    def function_ablate(j):
        """Subtract the component of z predictable from the D^j matrix
        coefficients of the input -- well-defined even when the block is at the
        measurement floor (unlike the subspace estimate)."""
        from core.su2 import wigner_D

        def feats(q):
            D = wigner_D(j, q).reshape(len(q), -1)
            return np.concatenate([D.real, D.imag], axis=1)
        F = feats(probe)
        muF = F.mean(0)
        Fc = F - muF
        W = np.linalg.solve(Fc.T @ Fc + 1e-6 * np.eye(Fc.shape[1]), Fc.T @ Zc)

        def fn(model, enc, q, device):
            with torch.no_grad():
                z = model.embed(torch.tensor(enc(q), device=device))
            pred = (feats(q) - muF) @ W
            return z - torch.tensor(pred, dtype=torch.float32, device=device)
        return fn

    return [
        Intervention("none", identity_fn),
        Intervention("even_projection", parity(+1)),
        Intervention("odd_projection", parity(-1)),
        Intervention("fablate_spin_half", function_ablate(0.5)),
        Intervention("fablate_spin_one", function_ablate(1.0)),
        Intervention(f"ablate_spin_half_k{V_half.shape[1]}", project_out(V_half)),
        Intervention(f"ablate_spin_one_k{V_one.shape[1]}", project_out(V_one)),
        Intervention(f"ablate_top_pca_k{k}", project_out(V_pca)),
        Intervention(f"ablate_random_k{k}", project_out(V_rand)),
    ]


@torch.no_grad()
def score_with(model, enc, iv, qa, qb, qc, device, batch=4096):
    """Scores s(a,b,c) with intervention iv on the argument encoders.

    qa, qb: [n,4]; qc: [n,K,4] -> [n,K]
    """
    out = []
    for i in range(0, qa.shape[0], batch):
        za = iv(model, enc, qa[i:i + batch], device)
        zb = iv(model, enc, qb[i:i + batch], device)
        h = model.interact(torch.cat([za, zb], dim=-1))
        c = qc[i:i + batch]
        zc = model.enc_c(torch.tensor(
            enc(c.reshape(-1, 4)).reshape(c.shape[0], c.shape[1], -1),
            device=device))
        out.append(torch.einsum("bd,bkd->bk", h, zc).cpu().numpy())
    return np.concatenate(out)


# ------------------------------- experiment 1 ---------------------------------

def run_interventions(model, enc, cfg, device, n=8192, seed=7):
    rng = np.random.default_rng(seed)
    qa, qb = haar(n, rng), haar(n, rng)
    prod = qmul(qa, qb)

    # task set: true product + 63 Haar negatives outside the exclusion radius
    neg = haar(n * 63, rng).reshape(n, 63, 4)
    for _ in range(50):
        dot = np.abs(np.sum(neg * prod[:, None, :], axis=-1)) \
            if cfg["variant"] == "so3" else np.sum(neg * prod[:, None, :], axis=-1)
        bad = np.arccos(np.clip(dot, -1, 1)) < cfg["excl_radius"]
        if not bad.any():
            break
        neg[bad] = haar(int(bad.sum()), rng)
    task_c = np.concatenate([prod[:, None, :], neg], axis=1).astype(np.float32)

    # fine set: product vs 63 negatives at r = 0.2
    u = rng.standard_normal((n, 63, 3))
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    fine_c = np.concatenate(
        [prod[:, None, :], qmul(prod[:, None, :], expmap(0.2 * u))],
        axis=1).astype(np.float32)

    # center set: product vs its antipode
    center_c = np.concatenate([prod[:, None, :], -prod[:, None, :]],
                              axis=1).astype(np.float32)

    results = {}
    for iv in make_interventions(model, enc, cfg, device):
        acc = float((score_with(model, enc, iv, qa, qb, task_c, device)
                     .argmax(1) == 0).mean())
        fine = float((score_with(model, enc, iv, qa, qb, fine_c, device)
                      .argmax(1) == 0).mean())
        sc = score_with(model, enc, iv, qa, qb, center_c, device)
        disc = float(((sc[:, 0] > sc[:, 1]) + 0.5 * (sc[:, 0] == sc[:, 1])).mean())
        results[iv.name] = {"task_acc": acc, "fine_acc_r0.2": fine,
                            "center_disc": disc}
    return results


# ------------------------------- experiment 2 ---------------------------------

def run_character_fit(model, enc, cfg, device, n=16384, j_max=3.0, seed=11):
    rng = np.random.default_rng(seed)
    qa, qb = haar(n, rng), haar(n, rng)
    prod = qmul(qa, qb)
    # candidates at mixed distances: half Haar, half at uniform radius in (0, pi)
    r = rng.uniform(0.0, np.pi, size=n)
    u = rng.standard_normal((n, 3))
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    c_near = qmul(prod, expmap(r[:, None] * u))
    c = np.where(rng.random(n)[:, None] < 0.5, c_near, haar(n, rng))

    iv = make_interventions(model, enc, cfg, device)[0]     # identity
    s = score_with(model, enc, iv, qa, qb, c[:, None, :].astype(np.float32),
                   device)[:, 0]

    rel = qmul(prod, qconj_np(c))                            # ab c^-1
    js = [j / 2 for j in range(0, int(2 * j_max) + 1)]
    X = np.stack([wigner_character(j, rel) for j in js], axis=1)
    out = {"js": js}
    y = s - s.mean()
    for upto in range(1, len(js) + 1):
        Xu = X[:, :upto] - X[:, :upto].mean(0)
        beta, *_ = np.linalg.lstsq(Xu, y, rcond=None)
        r2 = 1 - ((y - Xu @ beta) ** 2).sum() / (y ** 2).sum()
        out[f"r2_upto_j{js[upto - 1]}"] = float(r2)
        if upto == len(js):
            out["coeffs"] = {str(j): float(b) for j, b in zip(js, beta)}
    return out


def qconj_np(q):
    out = q.copy()
    out[..., 1:] *= -1
    return out


# ------------------------------- experiment 3 ---------------------------------

def _ridge_fit(Ztr, Ytr, lam=1.0):
    mu_z, mu_y = Ztr.mean(0), Ytr.mean(0)
    Zc, Yc = Ztr - mu_z, Ytr - mu_y
    W = np.linalg.solve(Zc.T @ Zc + lam * np.eye(Zc.shape[1]), Zc.T @ Yc)
    return W, mu_z, mu_y


def _ridge_apply(Z, W, mu_z, mu_y):
    return (Z - mu_z) @ W + mu_y


def run_composition_test(model, enc, cfg, device, n_train=16384, n_test=4096,
                         seed=13):
    """Probes: element target from z; product target from h. Tests whether the
    algebra composes: target(a) * target(b) == target_h(ab) on held-out pairs."""
    rng = np.random.default_rng(seed)
    su2 = cfg["variant"] == "su2"

    def target(q):
        if su2:
            return q                                          # spin-1/2
        return np.einsum("ni,nj->nij", q, q).reshape(len(q), 16)  # spin-1 (qq^T)

    # element probe on z
    q_el = haar(n_train, rng)
    Z_el = embed_np(model, enc, q_el, device)
    Wz, mz, my = _ridge_fit(Z_el, target(q_el))

    # product probe on h
    qa, qb = haar(n_train, rng), haar(n_train, rng)
    with torch.no_grad():
        H = model.product_state(
            torch.tensor(enc(qa), device=device),
            torch.tensor(enc(qb), device=device)).cpu().numpy()
    Wh, mh, mhy = _ridge_fit(H, target(qmul(qa, qb)))

    # held-out pairs, fresh elements
    ta_q, tb_q = haar(n_test, rng), haar(n_test, rng)
    prod = qmul(ta_q, tb_q)
    with torch.no_grad():
        Ht = model.product_state(
            torch.tensor(enc(ta_q), device=device),
            torch.tensor(enc(tb_q), device=device)).cpu().numpy()
    pred_h = _ridge_apply(Ht, Wh, mh, mhy)

    def r2(pred, true):
        return float(1 - ((true - pred) ** 2).sum()
                     / ((true - true.mean(0)) ** 2).sum())

    out = {"r2_h_to_product": r2(pred_h, target(prod))}
    # shuffled-pair control for the h probe
    perm = rng.permutation(n_test)
    out["r2_h_to_product_shuffled"] = r2(pred_h, target(qmul(ta_q[perm], tb_q)))

    if su2:
        pred_a = _ridge_apply(embed_np(model, enc, ta_q, device), Wz, mz, my)
        pred_b = _ridge_apply(embed_np(model, enc, tb_q, device), Wz, mz, my)
        pred_a /= np.linalg.norm(pred_a, axis=1, keepdims=True)
        pred_b /= np.linalg.norm(pred_b, axis=1, keepdims=True)
        composed = qmul(pred_a, pred_b)
        out["r2_element_probe"] = r2(pred_a, ta_q)
        # does the composed element probe match (i) truth and (ii) the h probe?
        cos_true = np.abs(np.sum(composed * prod, axis=1))
        out["compose_vs_truth_cos_med"] = float(np.median(
            np.sum(composed * prod, axis=1)))
        ph = pred_h / np.linalg.norm(pred_h, axis=1, keepdims=True)
        out["compose_vs_hprobe_cos_med"] = float(np.median(
            np.sum(composed * ph, axis=1)))
        out["compose_vs_truth_cos_med_shuffled"] = float(np.median(
            np.sum(qmul(pred_a[perm], pred_b) * prod, axis=1)))
    return out


# ----------------------------------- driver -----------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--out", default=None,
                   help="output JSON filename inside run_dir")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, enc, cfg, step = load_model(args.run_dir, args.ckpt, args.device)
    print(f"== {args.run_dir} (variant={cfg['variant']}, step={step})")

    res = {"run_dir": args.run_dir, "variant": cfg["variant"], "step": step}
    res["interventions"] = run_interventions(model, enc, cfg, args.device,
                                             n=args.n)
    print(f"{'intervention':<28} {'task':>6} {'r=0.2':>6} {'D_c':>6}")
    for name, r in res["interventions"].items():
        print(f"{name:<28} {r['task_acc']:>6.3f} {r['fine_acc_r0.2']:>6.3f} "
              f"{r['center_disc']:>6.3f}")

    res["character_fit"] = run_character_fit(model, enc, cfg, args.device)
    cf = res["character_fit"]
    print("character fit R^2:",
          {k: round(v, 4) for k, v in cf.items() if k.startswith("r2")})

    res["composition"] = run_composition_test(model, enc, cfg, args.device)
    print("composition:", {k: round(v, 4) for k, v in res["composition"].items()})

    out = Path(args.run_dir) / (args.out or f"causal_{step}.json")
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
