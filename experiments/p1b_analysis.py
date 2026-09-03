"""G1: reanalysis of P1B frame-versus-block runs from logged trajectories.

Reads log.jsonl for each arm/seed, computes the declared norm functionals,
and reports paired comparisons at equal step, equal wall time, interpolated
equal T2 accuracy, and a common norm budget via trajectory Pareto
frontiers (matched achieved norm where the arms overlap, strict
dominance where they never do),
with exact paired sign-flip permutation tests and paired bootstrap
intervals. See preregistration/p1b_wd_sweep_2026-09-01.md for endpoint
declarations.

Held-out (T1/T2) loss was not logged by the trainer, only accuracy and
margin; equal-loss matching therefore uses train loss and is labeled as
such. This limitation is recorded in the output.

Usage:
    uv run python -m experiments.p1b_analysis \
        --cell wd1.0 --frame-glob 'data/su2/so3_so3frame_f0.01_s{s}' \
        --block-glob 'data/su2/so3_so3block_f0.01_s{s}' \
        --out summary/p1b_reanalysis_wd1.0.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from core.envelope import make_envelope, write_analysis

SEEDS = (0, 1, 2, 3, 4)
FIELDS = ("step", "loss", "acc_train", "acc_T1", "acc_T2", "margin_T2",
          "norm_enc", "norm_int", "norm_encc", "wall_s")


def load_traj(run_dir):
    recs = [json.loads(l) for l in open(Path(run_dir) / "log.jsonl")]
    recs = [r for r in recs if "margin_T2" in r]
    t = {f: np.array([r[f] for r in recs], dtype=float) for f in FIELDS}
    t["norm_total"] = np.sqrt(t["norm_enc"]**2 + t["norm_int"]**2
                              + t["norm_encc"]**2)
    t["norm_product"] = t["norm_enc"] * t["norm_int"] * t["norm_encc"]
    t["margin_per_norm"] = t["margin_T2"] / t["norm_total"]
    return t


def at_step(t, step):
    i = int(np.argmin(np.abs(t["step"] - step)))
    return {f: float(t[f][i]) for f in t}


def interp_at(t, xkey, x, ykeys):
    """Linear interpolation of ykeys at xkey == x (xkey must be monotone)."""
    xs = t[xkey]
    return {y: float(np.interp(x, xs, t[y])) for y in ykeys}


def first_step_reaching(t, key, level, smooth=3):
    """First (interpolated) step at which a trailing-mean of `key` >= level."""
    v = np.convolve(t[key], np.ones(smooth) / smooth, mode="same")
    idx = np.nonzero(v >= level)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(t["step"][0])
    x0, x1, y0, y1 = t["step"][i - 1], t["step"][i], v[i - 1], v[i]
    return float(x0 + (x1 - x0) * (level - y0) / max(y1 - y0, 1e-12))


def frontier(t, min_step=0):
    """Best margin achievable at total norm <= n.

    The frontier is the running maximum of margin over points sorted by
    norm, so early low-margin transients can never displace later points
    and no burn-in cutoff is needed; min_step is retained only for the
    sensitivity check reported alongside the primary analysis.
    """
    m = t["step"] >= min_step
    norms, margins = t["norm_total"][m], t["margin_T2"][m]
    order = np.argsort(norms)
    n_sorted, mg_sorted = norms[order], np.maximum.accumulate(margins[order])
    return n_sorted, mg_sorted


def frontier_margin_at(n_sorted, mg_sorted, n):
    if n < n_sorted[0]:
        return None
    return float(np.interp(n, n_sorted, mg_sorted))


def exact_paired_permutation(diffs):
    """Exact one- and two-sided sign-flip p for mean(diffs) with n<=20."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    obs = d.mean()
    means = [np.mean(d * np.array(signs)) for signs in
             itertools.product((1, -1), repeat=n)]
    means = np.array(means)
    p_one = float(np.mean(means >= obs - 1e-12))
    p_two = float(np.mean(np.abs(means) >= abs(obs) - 1e-12))
    return {"mean": float(obs), "p_one_sided": p_one, "p_two_sided": p_two,
            "n_assignments": 2 ** n}


def paired_bootstrap(diffs, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return {"ci95": [float(np.percentile(bs, 2.5)),
                     float(np.percentile(bs, 97.5))]}


def analyze_pair(frame_t, block_t, acc_floor=0.85):
    out = {}
    # equal step (final)
    f60, b60 = at_step(frame_t, 60000), at_step(block_t, 60000)
    out["final"] = {"frame": f60, "block": b60}
    out["floor_ok"] = bool(f60["acc_T2"] >= acc_floor
                           and b60["acc_T2"] >= acc_floor)
    # equal wall time
    tmax = min(frame_t["wall_s"][-1], block_t["wall_s"][-1])
    keys = ["acc_T2", "margin_T2", "norm_total", "norm_product", "loss"]
    out["equal_wall_time"] = {
        "wall_s": float(tmax),
        "frame": interp_at(frame_t, "wall_s", tmax, keys),
        "block": interp_at(block_t, "wall_s", tmax, keys)}
    # interpolated equal T2 accuracy
    level = min(f60["acc_T2"], b60["acc_T2"])
    rows = {}
    for name, t in (("frame", frame_t), ("block", block_t)):
        s = first_step_reaching(t, "acc_T2", level)
        rows[name] = (None if s is None else
                      dict(interp_at(t, "step", s,
                                     ["margin_T2", "norm_total", "loss"]),
                           step=s))
    out["equal_T2_accuracy"] = {"level": float(level), **rows}
    # margin at a common norm budget, via trajectory frontiers. Two
    # regimes, labeled distinctly: where the arms' achieved-norm ranges
    # overlap, the comparison averages the frontier difference over the
    # shared range (matched achieved norm). Where the block arm's entire
    # norm range lies strictly below the frame arm's, the arms never
    # achieve a common norm, and the honest statement is strict
    # norm-budget (Pareto) dominance: at the smallest budget the frame
    # ever achieves, the block frontier is already complete, so the
    # reported number is the margin difference at that budget.
    nf, mf = frontier(frame_t)
    nb, mb = frontier(block_t)
    lo = max(nf[0], nb[0])
    hi = min(nf[-1], nb[-1])
    if hi > lo:
        grid = np.linspace(lo, hi, 21)
        dm = [frontier_margin_at(nb, mb, n) - frontier_margin_at(nf, mf, n)
              for n in grid]
        out["norm_budget"] = {
            "regime": "matched_achieved_norm",
            "norm_overlap": [float(lo), float(hi)],
            "margin_diff_block_minus_frame": float(np.mean(dm)),
            "margin_diff_at_quantiles": {
                f"{q:.2f}": float(np.interp(np.quantile(grid, q), grid, dm))
                for q in (0.0, 0.25, 0.5, 0.75, 1.0)}}
    elif nb[-1] <= nf[0]:
        out["norm_budget"] = {
            "regime": "strict_dominance",
            "norm_overlap": None,
            "frame_min_norm": float(nf[0]),
            "block_max_norm": float(nb[-1]),
            "note": ("the block arm's entire achieved-norm range lies "
                     "below the frame arm's; margin difference evaluated "
                     "at the frame's minimum achieved norm budget, where "
                     "the block frontier is complete"),
            "margin_diff_block_minus_frame":
                float(mb[-1] - frontier_margin_at(nf, mf, nf[0]))}
    else:
        out["norm_budget"] = {
            "regime": "frame_dominates",
            "norm_overlap": None,
            "margin_diff_block_minus_frame":
                float(frontier_margin_at(nb, mb, nb[0]) - mf[-1])}
    # sensitivity of the frontier comparison to a 20k burn-in cutoff
    nf2, mf2 = frontier(frame_t, min_step=20000)
    nb2, mb2 = frontier(block_t, min_step=20000)
    lo2, hi2 = max(nf2[0], nb2[0]), min(nf2[-1], nb2[-1])
    if hi2 > lo2:
        g2 = np.linspace(lo2, hi2, 21)
        d2 = float(np.mean([frontier_margin_at(nb2, mb2, n)
                            - frontier_margin_at(nf2, mf2, n) for n in g2]))
    else:
        d2 = float(mb2[-1] - frontier_margin_at(nf2, mf2, nf2[0]))
    out["norm_budget"]["cutoff20k_sensitivity_diff"] = d2
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True)
    p.add_argument("--frame-glob", required=True,
                   help="pattern with {s} for the seed")
    p.add_argument("--block-glob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--traj-out", default=None)
    p.add_argument("--acc-floor", type=float, default=0.85)
    p.add_argument("--analysis-seed", type=int, default=0)
    args = p.parse_args()

    per_seed, trajs = {}, {}
    for s in SEEDS:
        fdir, bdir = args.frame_glob.format(s=s), args.block_glob.format(s=s)
        if not (Path(fdir) / "log.jsonl").exists():
            per_seed[str(s)] = {"missing": fdir}
            continue
        ft, bt = load_traj(fdir), load_traj(bdir)
        per_seed[str(s)] = analyze_pair(ft, bt, args.acc_floor)
        trajs[str(s)] = {"frame": {k: v.tolist() for k, v in ft.items()},
                         "block": {k: v.tolist() for k, v in bt.items()}}

    complete = [v for v in per_seed.values() if "missing" not in v]
    stats = {}
    endpoints = {
        "margin_at_norm_budget":
            lambda v: v["norm_budget"]["margin_diff_block_minus_frame"],
        "final_margin":
            lambda v: v["final"]["block"]["margin_T2"]
            - v["final"]["frame"]["margin_T2"],
        "final_total_norm":
            lambda v: v["final"]["block"]["norm_total"]
            - v["final"]["frame"]["norm_total"],
        "final_margin_per_norm":
            lambda v: v["final"]["block"]["margin_per_norm"]
            - v["final"]["frame"]["margin_per_norm"],
        "final_T2_accuracy":
            lambda v: v["final"]["block"]["acc_T2"]
            - v["final"]["frame"]["acc_T2"],
        "final_train_loss":
            lambda v: v["final"]["block"]["loss"]
            - v["final"]["frame"]["loss"],
    }
    for name, fn in endpoints.items():
        diffs = [fn(v) for v in complete]
        stats[name] = {"paired_diffs_block_minus_frame": diffs,
                       **exact_paired_permutation(diffs),
                       **paired_bootstrap(diffs, seed=args.analysis_seed)}

    results = {
        "cell": args.cell,
        "seeds_analyzed": [s for s in per_seed if "missing" not in
                           per_seed[s]],
        "acc_floor": args.acc_floor,
        "norm_functionals": {
            "total": "sqrt(sum of squared module Frobenius norms)",
            "product": "product of module Frobenius norms",
            "caveat": ("neither functional is invariant to rescaling "
                       "symmetries of the non-homogeneous network; values "
                       "are measured functionals of the trained "
                       "parameters, not intrinsic complexity")},
        "held_out_loss_note": ("T1/T2 loss was not logged; equal-loss "
                               "matching uses train loss"),
        "per_seed": per_seed,
        "paired_statistics": stats,
    }
    env = make_envelope(
        "p1b-reanalysis", 1, analysis_file=__file__,
        analysis_args=vars(args), analysis_seed=args.analysis_seed)
    write_analysis(args.out, results, env)
    if args.traj_out:
        Path(args.traj_out).write_text(json.dumps(
            {"envelope": env, "trajectories": trajs}))
    for name, st in stats.items():
        print(f"{name:28s} diffs={['%.4f' % d for d in st['paired_diffs_block_minus_frame']]}"
              f" mean={st['mean']:+.4f} p1={st['p_one_sided']:.4f}"
              f" ci95={st['ci95']}")


if __name__ == "__main__":
    main()
