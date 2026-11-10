"""G3: localize the frame-side composition mechanism (P1B wd=1.0 models).

Layer-wise probes and causal interventions on the exact-feature SO(3)
models. For each arm (so3frame 6D, so3block 9D) and seed, ridge probes at
four sites decode the frame columns, the missing third column, and the
product matrix, with validation-split lambda selection, disjoint test
reporting, shuffled-label and random-initialization controls, and a
degree-2 polynomial-kernel probe for the missing column. The causal step
builds a cross-fitted unique-column-3 subspace at the earliest linearly
decodable site, ablates it, patches it from counterfactual inputs, and
compares against equal-rank random and transposed-subspace controls and
the block arm.

Verdict rules (recorded in the aggregate output): a positive causal
reconstruction claim requires col-3 linear decode >= 0.9 at some site AND
selective ablation damage (task drop >= 10x the random-subspace control)
AND counterfactual patch redirection substantially above chance, in all
five seeds. Degree-2-only success records "information present, not
linearly localized". Ablation/patch failure records "causal
reconstruction claim removed".

Usage:
    uv run python -m experiments.p1b_probes --device cpu \
        --out summary/p1b_probes_wd1.0.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.envelope import make_envelope, write_analysis
from core.su2 import ExplicitFeatureEncoding, get_encoding, haar, qmul
from core.su2_model import TripleScorer
from experiments.su2_train import class_distance, make_pair_set

SEEDS = (0, 1, 2, 3, 4)
ARMS = {"so3frame": "data/su2/so3_so3frame_f0.01_s{s}",
        "so3block": "data/su2/so3_so3block_f0.01_s{s}"}
SINGLE_SITES = ("enc_hidden", "z")           # activations of one element
PAIR_SITES = ("int_hidden", "h")             # activations of a pair
N_TRAIN, N_VAL, N_TEST = 8192, 2048, 2048
LAMBDAS = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
K2_LAMBDAS = [1e-6, 1e-4, 1e-2, 1.0, 10.0]
COL3_CRITERION = 0.9


def rot(q):
    """Rotation matrix of a unit quaternion, R[..., i, j] (row i, col j)."""
    return ExplicitFeatureEncoding._rot(q)


def load_model(run_dir, device):
    cfg = json.load(open(Path(run_dir) / "config.json"))
    state = torch.load(Path(run_dir) / "ckpt_60000.pt", map_location=device,
                       weights_only=False)
    eff = {**cfg, **state.get("config", {})}
    enc = get_encoding(eff["encoding"], eff["variant"],
                       m=eff.get("requested_overrides", {}).get("m", eff["m"]),
                       alpha=eff["alpha"], seed=eff["enc_seed"])
    model = TripleScorer(m=eff["m"], d_enc=eff.get("d_enc", 128),
                         enc_hidden=eff.get("enc_hidden", 256),
                         mlp_hidden=eff.get("mlp_hidden", 512),
                         activation=eff.get("activation", "gelu"),
                         mlp_depth=eff.get("mlp_depth", 1)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, enc.torch_fn(device), eff


def random_model(m, device, seed=1234):
    torch.manual_seed(seed)
    model = TripleScorer(m=m).to(device)
    model.eval()
    return model


@torch.no_grad()
def site_acts(model, psi, q, qb=None, batch=4096):
    """Activations at all sites. q (and qb for pair sites) are quaternions."""
    outs = {s: [] for s in (SINGLE_SITES + PAIR_SITES if qb is not None
                            else SINGLE_SITES)}
    for i in range(0, len(q), batch):
        xa = psi(torch.tensor(q[i:i + batch], dtype=torch.float32))
        eh = model.enc[1](model.enc[0](xa))
        z = model.enc[2](eh)
        outs["enc_hidden"].append(eh)
        outs["z"].append(z)
        if qb is not None:
            xb = psi(torch.tensor(qb[i:i + batch], dtype=torch.float32))
            zb = model.enc(xb)
            ih = model.interact[1](model.interact[0](
                torch.cat([z, zb], dim=-1)))
            outs["int_hidden"].append(ih)
            outs["h"].append(model.interact[2](ih))
    return {k: torch.cat(v).double().numpy() for k, v in outs.items()}


def ridge_probe(X, Y, rng=None, shuffle=False):
    """Multi-target ridge with val-selected lambda; returns (test R2 mean,
    lambda, W, means). Splits are fixed slices of the input order."""
    if shuffle:
        perm = rng.permutation(len(Y))
        Y = Y[perm]
    Xtr, Xv, Xte = (X[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_VAL],
                    X[N_TRAIN + N_VAL:])
    Ytr, Yv, Yte = (Y[:N_TRAIN], Y[N_TRAIN:N_TRAIN + N_VAL],
                    Y[N_TRAIN + N_VAL:])
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - xm, Ytr - ym
    G = Xc.T @ Xc / len(Xc)
    scale = np.trace(G) / G.shape[0]
    b = Xc.T @ Yc / len(Xc)
    best = (-np.inf, None, None)
    for lam in LAMBDAS:
        W = np.linalg.solve(G + lam * scale * np.eye(G.shape[0]), b)
        pv = (Xv - xm) @ W + ym
        r2 = r2_mean(Yv, pv)
        if r2 > best[0]:
            best = (r2, lam, W)
    _, lam, W = best
    pte = (Xte - xm) @ W + ym
    return r2_mean(Yte, pte), lam, W, (xm, ym)


def r2_mean(Y, P):
    ss = ((Y - P) ** 2).sum(0)
    tv = ((Y - Y.mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ss / np.maximum(tv, 1e-12)))


def kernel2_probe(X, Y):
    """Degree-2 polynomial kernel ridge for col3 (Gram kept tractable)."""
    n_tr, n_v, n_te = 4096, 1024, 1024
    Xs = X / np.sqrt(X.shape[1])
    Xtr, Xv, Xte = Xs[:n_tr], Xs[n_tr:n_tr + n_v], Xs[n_tr + n_v:n_tr + n_v + n_te]
    Ytr, Yv, Yte = Y[:n_tr], Y[n_tr:n_tr + n_v], Y[n_tr + n_v:n_tr + n_v + n_te]
    ym = Ytr.mean(0)
    K = (1.0 + Xtr @ Xtr.T) ** 2
    Kv = (1.0 + Xv @ Xtr.T) ** 2
    Kte = (1.0 + Xte @ Xtr.T) ** 2
    scale = np.trace(K) / n_tr
    best = (-np.inf, None)
    for lam in K2_LAMBDAS:
        A = np.linalg.solve(K + lam * scale * np.eye(n_tr), Ytr - ym)
        r2 = r2_mean(Yv, Kv @ A + ym)
        if r2 > best[0]:
            best = (r2, A)
    _, A = best
    return r2_mean(Yte, Kte @ A + ym)


def probe_suite(model, psi, rng, tag):
    n = N_TRAIN + N_VAL + N_TEST
    g = haar(n, rng)
    ga, gb = haar(n, rng), haar(n, rng)
    acts = site_acts(model, psi, g)
    pacts = site_acts(model, psi, ga, gb)
    R = rot(g)
    cols12 = R[..., :, :2].reshape(n, 6)
    col3 = R[..., :, 2]
    Rab = rot(qmul(ga, gb)).reshape(n, 9)
    out = {}
    for s in SINGLE_SITES:
        X = acts[s]
        r12, _, _, _ = ridge_probe(X, cols12)
        r3, lam3, W3, _ = ridge_probe(X, col3)
        r3_shuf, _, _, _ = ridge_probe(X, col3, rng=rng, shuffle=True)
        r3_k2 = kernel2_probe(X, col3)
        out[s] = {"cols12_linear": r12, "col3_linear": r3,
                  "col3_linear_lambda": lam3, "col3_shuffled": r3_shuf,
                  "col3_degree2": r3_k2}
    for s in PAIR_SITES:
        rp, _, _, _ = ridge_probe(pacts[s], Rab)
        rp_shuf, _, _, _ = ridge_probe(pacts[s], Rab, rng=rng, shuffle=True)
        out[s] = {"Rab_linear": rp, "Rab_shuffled": rp_shuf}
    return out, acts, (g, cols12, col3)


def fit_W(X, Y, lam):
    xm, ym = X.mean(0), Y.mean(0)
    Xc = X - xm
    G = Xc.T @ Xc / len(Xc)
    scale = np.trace(G) / G.shape[0]
    return np.linalg.solve(G + lam * scale * np.eye(G.shape[0]),
                           Xc.T @ (Y - ym) / len(Xc)), xm


def orth(B, tol=1e-8):
    Q, Rm = np.linalg.qr(B)
    keep = np.abs(np.diag(Rm)) > tol * max(1.0, np.abs(np.diag(Rm)).max())
    return Q[:, keep]


def unique_subspace(X, Y_main, Y_other, lam):
    """Orthonormal basis for the Y_main-predictive directions of X after
    projecting off the Y_other-predictive directions (cross-fit half A)."""
    Wm, xm = fit_W(X, Y_main, lam)
    Wo, _ = fit_W(X, Y_other, lam)
    Qo = orth(Wo)
    U = orth(Wm - Qo @ (Qo.T @ Wm))
    return U, xm


class SiteIntervention:
    """Forward hook subtracting/replacing the centered projection onto U."""

    def __init__(self, U, mu, mode="ablate"):
        self.U = torch.tensor(U, dtype=torch.float32)
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.mode = mode
        self.donors = []                     # queue for patch mode

    def __call__(self, module, inputs, output):
        c = output - self.mu
        proj = (c @ self.U) @ self.U.T
        if self.mode == "ablate":
            return output - proj
        donor = self.donors.pop(0)
        dproj = ((donor - self.mu) @ self.U) @ self.U.T
        return output - proj + dproj


SITE_MODULE = {"enc_hidden": lambda m: m.enc[1], "z": lambda m: m.enc[2],
               "int_hidden": lambda m: m.interact[1],
               "h": lambda m: m.interact[2]}


@torch.no_grad()
def rank_eval(model, psi, qa, qb, cands, batch=512):
    accs, margins = [], []
    for i in range(0, len(qa), batch):
        xa = psi(torch.tensor(qa[i:i + batch], dtype=torch.float32))
        xb = psi(torch.tensor(qb[i:i + batch], dtype=torch.float32))
        xc = psi(torch.tensor(cands[i:i + batch], dtype=torch.float32))
        h = model.product_state(xa, xb)
        s = torch.einsum("bd,bkd->bk", h, model.enc_c(xc))
        accs.append((s.argmax(-1) == 0).float())
        margins.append(s[:, 0] - s[:, 1:].max(-1).values)
    return (float(torch.cat(accs).mean()), float(torch.cat(margins).mean()))


@torch.no_grad()
def product_decode_r2(model, psi, rng, n=6144, n_te=2048):
    ga, gb = haar(n + n_te, rng), haar(n + n_te, rng)
    acts = site_acts(model, psi, ga, gb)
    H = acts["h"]
    Rab = rot(qmul(ga, gb)).reshape(n + n_te, 9)
    W, xm = fit_W(H[:n], Rab[:n], 1e-2)
    ym = Rab[:n].mean(0)
    return r2_mean(Rab[n:], (H[n:] - xm) @ W + ym)


def analyze_run(arm, seed, device, analysis_seed):
    rng = np.random.default_rng(analysis_seed + 1000 * seed)
    run = ARMS[arm].format(s=seed)
    model, psi, eff = load_model(run, device)
    probes, acts, (g, cols12, col3) = probe_suite(model, psi, rng, arm)

    # earliest linearly decodable site for col3
    site = None
    for s in SINGLE_SITES:
        if probes[s]["col3_linear"] >= COL3_CRITERION:
            site = s
            break
    criterion_met = site is not None
    if site is None:
        site = max(SINGLE_SITES, key=lambda s: probes[s]["col3_linear"])

    lam = probes[site]["col3_linear_lambda"]
    Xa = acts[site][:4096]                     # cross-fit half A
    U3, mu = unique_subspace(Xa, col3[:4096], cols12[:4096], lam)
    U12, _ = unique_subspace(Xa, cols12[:4096], col3[:4096], lam)
    U12 = U12[:, :U3.shape[1]] if U12.shape[1] >= U3.shape[1] else U12
    rng_ctrl = np.random.default_rng(analysis_seed + 77)
    Urand = orth(rng_ctrl.standard_normal((Xa.shape[1], U3.shape[1])))

    # task eval (fresh)
    ev_rng = np.random.default_rng(analysis_seed + 5)
    qa, qb, cands = make_pair_set(haar(2048, ev_rng), haar(2048, ev_rng),
                                  64, 0.2, 0.5, "so3", ev_rng)
    module = SITE_MODULE[site](model)

    res = {"run": run, "site": site, "criterion_met": criterion_met,
           "subspace_rank": int(U3.shape[1]), "probes": probes}

    base_acc, base_margin = rank_eval(model, psi, qa, qb, cands)
    base_decode = product_decode_r2(model, psi,
                                    np.random.default_rng(analysis_seed + 9))
    res["baseline"] = {"acc": base_acc, "margin": base_margin,
                       "product_decode_r2": base_decode}
    for name, U in (("ablate_col3_unique", U3),
                    ("ablate_random_equal_rank", Urand),
                    ("ablate_cols12_unique", U12)):
        hook = SiteIntervention(U, mu, "ablate")
        hh = module.register_forward_hook(hook)
        acc, margin = rank_eval(model, psi, qa, qb, cands)
        dec = product_decode_r2(model, psi,
                                np.random.default_rng(analysis_seed + 9))
        hh.remove()
        res[name] = {"acc": acc, "margin": margin, "product_decode_r2": dec}

    # counterfactual patch of the U3 component
    p_rng = np.random.default_rng(analysis_seed + 21)
    n_p = 1024
    a, b = haar(n_p, p_rng), haar(n_p, p_rng)
    a2, b2 = haar(n_p, p_rng), haar(n_p, p_rng)
    ab, a2b2 = qmul(a, b), qmul(a2, b2)
    neg = haar(n_p * 62, p_rng).reshape(n_p, 62, 4)
    for _ in range(50):
        bad = ((class_distance(neg, ab[:, None, :], "so3") < 0.5)
               | (class_distance(neg, a2b2[:, None, :], "so3") < 0.5))
        if not bad.any():
            break
        neg[bad] = haar(int(bad.sum()), p_rng)
    cands_p = np.concatenate([a2b2[:, None, :], ab[:, None, :], neg],
                             axis=1).astype(np.float32)

    def patch_rates(donor_first, donor_second):
        hook = SiteIntervention(U3, mu, "patch")
        if site in SINGLE_SITES:
            d1 = site_acts(model, psi, donor_first)[site]
            d2 = site_acts(model, psi, donor_second)[site]
        else:
            d = site_acts(model, psi, donor_first, donor_second)[site]
        hh = SITE_MODULE[site](model).register_forward_hook(hook)
        redirect = stay = 0.0
        bs = 512
        with torch.no_grad():
            for i in range(0, n_p, bs):
                if site in SINGLE_SITES:
                    hook.donors = [
                        torch.tensor(d1[i:i + bs], dtype=torch.float32),
                        torch.tensor(d2[i:i + bs], dtype=torch.float32)]
                else:
                    hook.donors = [
                        torch.tensor(d[i:i + bs], dtype=torch.float32)]
                xa = psi(torch.tensor(a[i:i + bs], dtype=torch.float32))
                xb = psi(torch.tensor(b[i:i + bs], dtype=torch.float32))
                xc = psi(torch.tensor(cands_p[i:i + bs],
                                      dtype=torch.float32))
                h = model.product_state(xa, xb)
                s = torch.einsum("bd,bkd->bk", h, model.enc_c(xc))
                am = s.argmax(-1)
                redirect += float((am == 0).sum())
                stay += float((am == 1).sum())
        hh.remove()
        return redirect / n_p, stay / n_p

    # unpatched reference on the same candidate sets
    ref_red, ref_stay = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, n_p, 512):
            xa = psi(torch.tensor(a[i:i + 512], dtype=torch.float32))
            xb = psi(torch.tensor(b[i:i + 512], dtype=torch.float32))
            xc = psi(torch.tensor(cands_p[i:i + 512], dtype=torch.float32))
            h = model.product_state(xa, xb)
            s = torch.einsum("bd,bkd->bk", h, model.enc_c(xc))
            am = s.argmax(-1)
            ref_red += float((am == 0).sum())
            ref_stay += float((am == 1).sum())
    red, stay = patch_rates(a2, b2)
    res["patch"] = {"redirect_rate": red, "stay_rate": stay,
                    "unpatched_redirect": ref_red / n_p,
                    "unpatched_stay": ref_stay / n_p,
                    "chance": 1.0 / 64}
    return res, eff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--analysis-seed", type=int, default=0)
    p.add_argument("--out", default="summary/p1b_probes_wd1.0.json")
    args = p.parse_args()
    torch.set_num_threads(8)

    agg = {"arms": {}, "controls": {}}
    for arm in ARMS:
        per_seed = {}
        for s in SEEDS:
            res, eff = analyze_run(arm, s, args.device, args.analysis_seed)
            per_seed[str(s)] = res
            run = Path(res["run"])
            env = make_envelope(
                "p1b-probes", 1, analysis_file=__file__,
                analysis_args={**vars(args), "arm": arm, "seed": s},
                ckpt_path=run / "ckpt_60000.pt",
                run_config_path=run / "config.json",
                training_seed=s, analysis_seed=args.analysis_seed)
            write_analysis(run / "p1b_probes.json", res, env)
            print(f"[{arm} s{s}] site={res['site']} "
                  f"col3(enc_hidden)={res['probes']['enc_hidden']['col3_linear']:.3f} "
                  f"col3(z)={res['probes']['z']['col3_linear']:.3f} "
                  f"abl={res['ablate_col3_unique']['acc']:.3f} "
                  f"rand={res['ablate_random_equal_rank']['acc']:.3f} "
                  f"patch={res['patch']['redirect_rate']:.3f}")
        agg["arms"][arm] = per_seed

        # random-init probe control, one per arm
        m = 6 if arm == "so3frame" else 9
        rng = np.random.default_rng(args.analysis_seed + 4321)
        rmodel = random_model(m, "cpu")
        enc = get_encoding(arm, "so3", seed=0)
        rprobes, _, _ = probe_suite(rmodel, enc.torch_fn("cpu"), rng, arm)
        agg["controls"][arm + "_random_init"] = rprobes

    # aggregates and verdict
    def agg_stat(arm, path):
        vals = []
        for s in SEEDS:
            v = agg["arms"][arm][str(s)]
            for k in path:
                v = v[k]
            vals.append(v)
        return {"per_seed": vals, "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1))}

    summary = {}
    for arm in ARMS:
        summary[arm] = {
            "col3_linear_enc_hidden": agg_stat(arm, ("probes", "enc_hidden",
                                                     "col3_linear")),
            "col3_linear_z": agg_stat(arm, ("probes", "z", "col3_linear")),
            "col3_degree2_enc_hidden": agg_stat(arm, ("probes", "enc_hidden",
                                                      "col3_degree2")),
            "col3_degree2_z": agg_stat(arm, ("probes", "z", "col3_degree2")),
            "Rab_h": agg_stat(arm, ("probes", "h", "Rab_linear")),
            "baseline_acc": agg_stat(arm, ("baseline", "acc")),
            "ablate_col3_acc": agg_stat(arm, ("ablate_col3_unique", "acc")),
            "ablate_rand_acc": agg_stat(arm, ("ablate_random_equal_rank",
                                              "acc")),
            "ablate_cols12_acc": agg_stat(arm, ("ablate_cols12_unique",
                                                "acc")),
            "ablate_col3_decode": agg_stat(arm, ("ablate_col3_unique",
                                                 "product_decode_r2")),
            "patch_redirect": agg_stat(arm, ("patch", "redirect_rate")),
            "patch_stay": agg_stat(arm, ("patch", "stay_rate")),
        }

    f = summary["so3frame"]
    crit_all = all(agg["arms"]["so3frame"][str(s)]["criterion_met"]
                   for s in SEEDS)
    drops = [f["baseline_acc"]["per_seed"][i]
             - f["ablate_col3_acc"]["per_seed"][i] for i in range(5)]
    rdrops = [f["baseline_acc"]["per_seed"][i]
              - f["ablate_rand_acc"]["per_seed"][i] for i in range(5)]
    selective = all(d >= 10 * max(r, 1e-4) for d, r in zip(drops, rdrops))
    patched = all(v > 5 * (1 / 64) for v in f["patch_redirect"]["per_seed"])
    if crit_all and selective and patched:
        verdict = "causal reconstruction localized"
    elif (not crit_all) and min(f["col3_degree2_enc_hidden"]["per_seed"]
                                + f["col3_degree2_z"]["per_seed"]) > 0.9:
        verdict = "information present, not linearly localized"
    elif crit_all and not (selective and patched):
        verdict = "causal reconstruction claim removed"
    else:
        verdict = "col3 not reliably decodable; claim removed"
    summary["verdict"] = {
        "text": verdict, "criterion_met_all_seeds": crit_all,
        "selective_ablation_all_seeds": selective,
        "patch_redirect_all_seeds": patched,
        "rules": ("positive iff col3 linear >= 0.9 at a site AND task drop "
                  ">= 10x random control AND patch redirect > 5x chance, "
                  "all in 5 seeds")}

    results = {"per_seed": agg["arms"], "random_init_controls":
               agg["controls"], "summary": summary}
    env = make_envelope("p1b-probes-aggregate", 1, analysis_file=__file__,
                        analysis_args=vars(args),
                        analysis_seed=args.analysis_seed)
    write_analysis(args.out, results, env)
    print(json.dumps(summary, indent=1, default=float)[:2000])


if __name__ == "__main__":
    main()
