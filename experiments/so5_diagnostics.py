"""G4.1/G4.2e: observation-geometry and candidate-geometry diagnostics for
the Spin(5)/SO(5) program.

For each observation map (faithful landmark, antipodally averaged landmark,
intrinsic vector-matrix control, intrinsic random-feature map) this
computes, on matched Haar samples:

  a. centered observation Gram spectrum, two effective ranks (exp of
     spectral entropy, and the 95%-variance count), and the condition
     number under a stated floor (1e-10 of the top eigenvalue), with
     bootstrap-over-elements uncertainty and replication over encoding
     seeds;
  b. local Jacobian singular spectra along the 10 orthonormal sp(2)
     generators (central finite differences, validated against a second
     step size);
  c. distortion between intrinsic geodesic distance (quotient-aware for
     center-invariant maps) and observation distance;
  d. nearest-neighbor retrieval agreement and positive/negative distance
     distributions under the actual training noise and exclusion radius;
  e. ranking ceilings on trainer-identical task instances for two rules,
     named exactly: the geodesic distance rule (monotone in the positive
     likelihood for small isotropic algebra noise, which is the training
     regime sigma = 0.15) and the observation-distance rule given the true
     product (isolates encoding distortion).

Output: summary/so5_diagnostics.json, envelope schema so5-diagnostics v1.
Run on CPU. Usage: uv run python -m experiments.so5_diagnostics
"""
import argparse
import json

import numpy as np

from core.envelope import make_envelope, write_analysis
from core.spin5 import (_B10, Sp2LandmarkEncoding, So5RandomFeatureEncoding,
                        So5VectorEncoding, chordal, geodesic_sp2, haar_sp2,
                        qmat_dag, qmat_mul, random_algebra, sp2_expmap)
from experiments.spin5_train import make_pair_set

SIGMA, EXCL, K = 0.15, 0.7, 64          # actual training values (config.json)
EIG_FLOOR = 1e-10


def encoding(kind, seed):
    if kind == "spin5_landmark":
        return Sp2LandmarkEncoding(m=48, alpha=2.0, variant="spin5",
                                   seed=seed)
    if kind == "so5_landmark":
        return Sp2LandmarkEncoding(m=48, alpha=2.0, variant="so5", seed=seed)
    if kind == "so5_vector_control":
        return So5VectorEncoding(seed=seed)
    if kind == "so5_random_feature":
        return So5RandomFeatureEncoding(m=48, alpha_cover=2.0, seed=seed)
    raise ValueError(kind)


def is_quotient(kind):
    return kind != "spin5_landmark"


def gram_stats(X, rng, n_boot=200):
    """Effective ranks and condition number of the centered covariance."""
    def one(Xs):
        C = np.cov(Xs.T)
        lam = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
        p = lam / lam.sum()
        ent = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
        n95 = int(np.searchsorted(np.cumsum(p), 0.95) + 1)
        cond = float(lam[0] / max(lam[-1], EIG_FLOOR * lam[0]))
        return ent, n95, cond

    ent, n95, cond = one(X)
    boots = np.array([one(X[rng.integers(0, len(X), len(X))])
                      for _ in range(n_boot)])
    return {"effective_rank_entropy": ent,
            "effective_rank_entropy_ci95":
                [float(np.percentile(boots[:, 0], 2.5)),
                 float(np.percentile(boots[:, 0], 97.5))],
            "n95": n95,
            "n95_ci95": [float(np.percentile(boots[:, 1], 2.5)),
                         float(np.percentile(boots[:, 1], 97.5))],
            "condition_number": cond,
            "condition_number_ci95":
                [float(np.percentile(boots[:, 2], 2.5)),
                 float(np.percentile(boots[:, 2], 97.5))],
            "eig_floor": EIG_FLOOR, "n_boot": n_boot}


def jacobian_stats(enc, rng, n_base=64, eps=1e-4, eps2=1e-3):
    """Local metric spectra along orthonormal sp(2) generators."""
    U = haar_sp2(n_base, rng)
    svs, disc = [], []
    for i in range(n_base):
        base = U[i:i + 1]
        cols, cols2 = [], []
        for g in range(10):
            for e, out in ((eps, cols), (eps2, cols2)):
                step = sp2_expmap(e * _B10[None, g])
                d = (enc(qmat_mul(base, step))
                     - enc(qmat_mul(base, qmat_dag(step))))[0] / (2 * e)
                out.append(d)
        J = np.stack(cols, axis=1)
        J2 = np.stack(cols2, axis=1)
        disc.append(float(np.linalg.norm(J - J2) /
                          max(np.linalg.norm(J), 1e-12)))
        svs.append(np.linalg.svd(J, compute_uv=False))
    svs = np.array(svs)                                # [n_base, 10]
    conds = svs[:, 0] / np.maximum(svs[:, -1], 1e-12)
    return {"sv_mean": svs.mean(0).tolist(),
            "sv_min_q05_q50_q95": [float(np.quantile(svs[:, -1], q))
                                   for q in (0.05, 0.5, 0.95)],
            "local_condition_q05_q50_q95": [float(np.quantile(conds, q))
                                            for q in (0.05, 0.5, 0.95)],
            "fd_step_discrepancy_median": float(np.median(disc)),
            "eps": eps, "eps_validation": eps2, "n_base": n_base}


def distortion_stats(enc, quotient, rng, n=4096):
    U, V = haar_sp2(n, rng), haar_sp2(n, rng)
    d_int = geodesic_sp2(U, V, quotient=quotient)
    d_obs = np.linalg.norm(enc(U) - enc(V), axis=1)
    # rank correlation + spread of the obs/int ratio over mid-range pairs
    r = np.corrcoef(np.argsort(np.argsort(d_int)),
                    np.argsort(np.argsort(d_obs)))[0, 1]
    mid = (d_int > np.quantile(d_int, 0.1)) & (d_int < np.quantile(d_int,
                                                                   0.9))
    ratio = d_obs[mid] / d_int[mid]
    return {"spearman_int_vs_obs": float(r),
            "obs_over_int_ratio_q05_q50_q95":
                [float(np.quantile(ratio, q)) for q in (0.05, 0.5, 0.95)],
            "ratio_dispersion": float(np.quantile(ratio, 0.95)
                                      / max(np.quantile(ratio, 0.05),
                                            1e-12))}


def nn_stats(enc, quotient, rng, n_probe=1024, n_gallery=2048):
    P, G = haar_sp2(n_probe, rng), haar_sp2(n_gallery, rng)
    Xp, Xg = enc(P), enc(G)
    d_obs = ((Xp[:, None] - Xg[None]) ** 2).sum(-1)
    nn_obs = d_obs.argmin(1)
    nn_int = np.empty(n_probe, dtype=int)
    for i in range(n_probe):
        nn_int[i] = geodesic_sp2(np.broadcast_to(P[i], G.shape), G,
                                 quotient=quotient).argmin()
    agree = float((nn_obs == nn_int).mean())

    # positive vs admissible-negative distance distributions (intrinsic and
    # observation space) under the training noise and exclusion radius
    n = 2048
    A = haar_sp2(n, rng)
    pos = qmat_mul(A, sp2_expmap(random_algebra(n, SIGMA, rng)))
    neg = haar_sp2(n * (K - 1), rng).reshape(n, K - 1, 2, 2, 4)
    for _ in range(50):
        bad = chordal(neg, A[:, None], quotient=quotient) < EXCL
        if not bad.any():
            break
        neg[bad] = haar_sp2(int(bad.sum()), rng)
    d_pos = geodesic_sp2(A, pos, quotient=quotient)
    d_neg_min = np.stack(
        [geodesic_sp2(np.broadcast_to(A[i], neg[i].shape), neg[i],
                      quotient=quotient).min() for i in range(n)])
    o_pos = np.linalg.norm(enc(A) - enc(pos), axis=1)
    XA = enc(A)
    o_neg_min = np.stack(
        [np.linalg.norm(XA[i] - enc(neg[i]), axis=1).min()
         for i in range(512)])
    hist = lambda x, lo, hi: np.histogram(x, bins=50, range=(lo, hi),
                                          density=True)[0]
    lo, hi = 0.0, float(max(d_pos.max(), d_neg_min.max()))
    ov_int = float(np.minimum(hist(d_pos, lo, hi),
                              hist(d_neg_min, lo, hi)).sum()
                   * (hi - lo) / 50)
    lo2, hi2 = 0.0, float(max(o_pos.max(), o_neg_min.max()))
    ov_obs = float(np.minimum(hist(o_pos, lo2, hi2),
                              hist(o_neg_min[:512], lo2, hi2)).sum()
                   * (hi2 - lo2) / 50)
    return {"nn_top1_agreement": agree,
            "d_pos_q05_q50_q95": [float(np.quantile(d_pos, q))
                                  for q in (0.05, 0.5, 0.95)],
            "d_negmin_q05_q50_q95": [float(np.quantile(d_neg_min, q))
                                     for q in (0.05, 0.5, 0.95)],
            "overlap_intrinsic": ov_int,
            "overlap_observation": ov_obs,
            "sigma": SIGMA, "excl_radius": EXCL, "K": K}


def ceiling_stats(enc, variant, rng, n=2048):
    """Ranking accuracy of the two named rules on trainer-identical
    instances. The geodesic distance rule is monotone in the positive
    likelihood for small isotropic algebra noise (the training regime),
    which is its justification; it is named a distance rule, not an
    oracle."""
    quotient = variant == "so5"
    ga, gb = haar_sp2(n, rng), haar_sp2(n, rng)
    ga_, gb_, cands = make_pair_set(ga, gb, K, SIGMA, EXCL, variant, rng)
    prod = qmat_mul(ga, gb)
    correct_geo = correct_obs = 0
    Xp = enc(prod)
    for i in range(n):
        d = geodesic_sp2(np.broadcast_to(prod[i], cands[i].shape),
                         cands[i].astype(np.float64), quotient=quotient)
        correct_geo += int(d.argmin() == 0)
        do = np.linalg.norm(Xp[i] - enc(cands[i].astype(np.float64)),
                            axis=1)
        correct_obs += int(do.argmin() == 0)
    return {"geodesic_distance_rule_acc": correct_geo / n,
            "observation_distance_rule_acc": correct_obs / n,
            "n_instances": n,
            "rule_note": ("geodesic rule is monotone in the positive "
                          "likelihood for small isotropic algebra noise; "
                          "observation rule receives the true product and "
                          "isolates encoding distortion")}


KINDS = ["spin5_landmark", "so5_landmark", "so5_vector_control",
         "so5_random_feature"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--enc-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--analysis-seed", type=int, default=0)
    p.add_argument("--out", default="summary/so5_diagnostics.json")
    args = p.parse_args()

    results = {"training_values": {"sigma": SIGMA, "excl_radius": EXCL,
                                   "K": K},
               "effective_rank_definitions":
                   "exp of spectral entropy; and min k with 95% variance",
               "encodings": {}}
    for kind in KINDS:
        variant = "spin5" if kind == "spin5_landmark" else "so5"
        per_seed = []
        for es in args.enc_seeds:
            rng = np.random.default_rng(args.analysis_seed * 100 + es)
            enc = encoding(kind, es)
            U = haar_sp2(args.n, rng)
            entry = {"enc_seed": es,
                     "gram": gram_stats(enc(U), rng),
                     "jacobian": jacobian_stats(enc, rng),
                     "distortion": distortion_stats(enc, is_quotient(kind),
                                                    rng),
                     "nn": nn_stats(enc, is_quotient(kind), rng),
                     "ceilings": ceiling_stats(enc, variant, rng)}
            if kind == "so5_random_feature":
                entry["alpha_matching"] = {
                    "alpha": enc.alpha,
                    "pre_std_cover": enc.pre_std_cover,
                    "pre_std_rf": enc.pre_std_rf}
            per_seed.append(entry)
            print(f"[{kind} s{es}] erank={entry['gram']['effective_rank_entropy']:.1f}"
                  f" cond={entry['gram']['condition_number']:.2e}"
                  f" nn={entry['nn']['nn_top1_agreement']:.3f}"
                  f" geo={entry['ceilings']['geodesic_distance_rule_acc']:.3f}"
                  f" obs={entry['ceilings']['observation_distance_rule_acc']:.3f}",
                  flush=True)
        agg = {}
        for path in [("gram", "effective_rank_entropy"), ("gram", "n95"),
                     ("gram", "condition_number"),
                     ("distortion", "spearman_int_vs_obs"),
                     ("distortion", "ratio_dispersion"),
                     ("nn", "nn_top1_agreement"),
                     ("nn", "overlap_intrinsic"),
                     ("nn", "overlap_observation"),
                     ("ceilings", "geodesic_distance_rule_acc"),
                     ("ceilings", "observation_distance_rule_acc"),
                     ("jacobian", "fd_step_discrepancy_median")]:
            vals = [e[path[0]][path[1]] for e in per_seed]
            agg["/".join(path)] = {"mean": float(np.mean(vals)),
                                   "sd": float(np.std(vals, ddof=1))
                                   if len(vals) > 1 else 0.0}
        results["encodings"][kind] = {"per_seed": per_seed,
                                      "aggregate": agg}

    env = make_envelope("so5-diagnostics", 1, analysis_file=__file__,
                        analysis_args=vars(args),
                        analysis_seed=args.analysis_seed)
    write_analysis(args.out, results, env)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
