"""Rigorous geometry recovery: can a linear readout of the embedding predict the
TRUE polytope coordinates of HELD-OUT group elements?

This is the tautology-proof test the raw-PCA pictures couldn't give. We 5-fold
cross-validate a ridge map  embedding -> canonical polytope coordinates; a high
held-out R^2 means the embedding genuinely encodes the geometry (a random or
non-generalizing embedding fails out-of-sample). For groups defined over Q(sqrt5)
(2I, A5) the model may learn either of the two Galois-conjugate irreps, so we try
both the canonical coords and their sqrt5 -> -sqrt5 relabeling and report the best.

Outputs, per group, to images/<task>_*_s1/:
    geometry_decode.svg  -- out-of-fold predicted coords overlaid on the true
                            polytope (with its nearest-neighbour edges)
    geometry_decode.html -- interactive 3D version
and prints the held-out R^2 vs a shuffled-correspondence control.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch

from core.checkpoint import load_model, checkpoint_path, image_dir
from core.groups import get_group
from core import quaternions as q

PHI = (1 + np.sqrt(5)) / 2


def galois_perm(V):
    """sqrt5 -> -sqrt5 relabeling of icosian vertices (swaps phi/2 <-> (phi-1)/2)."""
    def fc(c):
        a = abs(c)
        if abs(a - PHI / 2) < 1e-3:
            return -np.sign(c) * (PHI - 1) / 2
        if abs(a - (PHI - 1) / 2) < 1e-3:
            return -np.sign(c) * PHI / 2
        return c
    key = {tuple(np.round(v, 5)): i for i, v in enumerate(V)}
    perm = np.zeros(len(V), int)
    for i, v in enumerate(V):
        cv = np.array([fc(c) for c in v]); k = tuple(np.round(cv, 5))
        perm[i] = key.get(k, int(np.abs(V - cv).sum(1).argmin()))
    return perm


def cv_predict(emb, Y, alpha=1.0, seed=0):
    """Out-of-fold predictions + mean held-out R^2."""
    Yc = Y - Y.mean(0)
    kf = KFold(5, shuffle=True, random_state=seed)
    pred = np.zeros_like(Yc); r2 = []
    for tr, te in kf.split(emb):
        m = Ridge(alpha=alpha).fit(emb[tr], Yc[tr])
        pred[te] = m.predict(emb[te])
        r2.append(1 - ((Yc[te]-pred[te])**2).sum() / ((Yc[te]-Yc[tr].mean(0))**2).sum())
    return pred, float(np.mean(r2)), Yc


def true_coords(task, G):
    """Canonical polytope coords + optional Galois-conjugate variant for the model's irrep."""
    if task == "icosahedral":
        return q.ih_icosahedron_coords()      # {'canonical','galois'} 120x3 icosahedron
    geo = G["geometry"]
    base = geo.get("coords4d")
    if base is None:
        base = geo.get("coords3d")
    if base is None:
        base = geo.get("coordsND")
    base = np.asarray(base, float)
    variants = {"canonical": base}
    if task in ("2I", "2O", "2T"):
        variants["galois"] = base[galois_perm(q.verts_2I() if task == "2I" else
                                              (q.verts_2O() if task == "2O" else q.verts_2T()))]
    if task == "A5":
        # rebuild canonical A5 reps to get the galois permutation on them
        Vf = q.verts_2I()
        def canon(v):
            v = np.round(v, 6)
            for c in v:
                if abs(c) > 1e-9:
                    if c < 0: v = -v
                    break
            return v
        reps, key = [], {}
        for v in Vf:
            cv = canon(v); k = tuple(np.round(cv, 6))
            if k not in key: key[k] = len(reps); reps.append(cv)
        reps = np.array(reps)
        def fc(c):
            a = abs(c)
            if abs(a-PHI/2) < 1e-3: return -np.sign(c)*(PHI-1)/2
            if abs(a-(PHI-1)/2) < 1e-3: return -np.sign(c)*PHI/2
            return c
        keyr = {tuple(np.round(v, 5)): i for i, v in enumerate(reps)}
        perm = np.array([keyr.get(tuple(np.round(canon(np.array([fc(c) for c in v])), 5)),
                                  int(np.abs(reps-canon(np.array([fc(c) for c in v]))).sum(1).argmin()))
                         for v in reps])
        variants["galois"] = base[perm]
    return variants


def plot_overlay(pred, Y, task, geom, img, r2):
    """Out-of-fold predictions overlaid on the true polytope (PCA to <=3D)."""
    from sklearn.decomposition import PCA
    D = min(3, Y.shape[1])
    p = PCA(n_components=D)
    T = p.fit_transform(Y)           # true polytope
    P = p.transform(pred - Y.mean(0) + Y.mean(0))  # predictions in same basis
    P = p.transform(pred)
    # nearest-neighbour edges of the true polytope
    from scipy.spatial.distance import pdist, squareform
    Dm = squareform(pdist(T)); np.fill_diagonal(Dm, np.inf)
    dmin = Dm.min(); edges = np.argwhere((Dm < dmin*1.08) & (np.arange(len(T))[:,None] < np.arange(len(T))))

    fig = plt.figure(figsize=(7.5, 7))
    if D >= 3:
        ax = fig.add_subplot(111, projection="3d")
        for i, j in edges:
            ax.plot(*zip(T[i], T[j]), color="0.7", lw=0.5)
        ax.scatter(T[:,0], T[:,1], T[:,2], s=80, facecolors="none", edgecolors="#d62728",
                   linewidths=1.3, label=f"true {geom}")
        ax.scatter(P[:,0], P[:,1], P[:,2], s=14, c="#1f77b4", label="held-out prediction")
    else:
        ax = fig.add_subplot(111)
        for i, j in edges:
            ax.plot(*zip(T[i], T[j]), color="0.7", lw=0.5)
        ax.scatter(T[:,0], T[:,1], s=80, facecolors="none", edgecolors="#d62728", linewidths=1.3, label=f"true {geom}")
        ax.scatter(P[:,0], P[:,1], s=14, c="#1f77b4", label="held-out prediction")
        ax.set_aspect("equal")
    ax.set_title(f"{task}: held-out elements decode to {geom}\n(5-fold CV R²={r2:.3f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(os.path.join(img, "geometry_decode.svg"), bbox_inches="tight")
    plt.close(fig)

    try:
        import plotly.graph_objects as go
        z_t = T[:,2] if D >= 3 else np.zeros(len(T))
        z_p = P[:,2] if D >= 3 else np.zeros(len(P))
        fig = go.Figure()
        for i, j in edges:
            fig.add_trace(go.Scatter3d(x=[T[i,0],T[j,0]], y=[T[i,1],T[j,1]], z=[z_t[i],z_t[j]],
                          mode="lines", line=dict(color="lightgray", width=2), showlegend=False))
        fig.add_trace(go.Scatter3d(x=T[:,0], y=T[:,1], z=z_t, mode="markers",
                      marker=dict(size=6, color="red", opacity=0.5), name=f"true {geom}"))
        fig.add_trace(go.Scatter3d(x=P[:,0], y=P[:,1], z=z_p, mode="markers",
                      marker=dict(size=3, color="blue"), name="held-out prediction"))
        fig.update_layout(title=f"{task}: held-out decode to {geom} (R²={r2:.3f})")
        fig.write_html(os.path.join(img, "geometry_decode.html"))
    except Exception as e:
        print(f"  (plotly 3D skipped: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--mlp_dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    G = get_group(args.task); n = G["n"]
    emb = load_model(args.ckpt, torch.device("cpu"))[0].embed.weight[:n].detach().numpy().astype(float)
    variants = true_coords(args.task, G)
    geom = G["geometry"]["polytope"]

    best = None
    for name, Y in variants.items():
        pred, r2, Yc = cv_predict(emb, Y)
        rng = np.random.default_rng(0)
        shuf = np.mean([cv_predict(emb, Y[rng.permutation(n)])[1] for _ in range(3)])
        print(f"  [{args.task}] {name:>9} coords: held-out R²={r2:+.3f}  (shuffled {shuf:+.3f})")
        if best is None or r2 > best[1]:
            best = (name, r2, pred, Yc)

    name, r2, pred, Yc = best
    img = image_dir(n, args.embed_dim, 4, args.mlp_dim, args.seed, task=args.task)
    plot_overlay(pred, Yc, args.task, geom, img, r2)
    print(f"  -> {args.task}: best={name} R²={r2:.3f}; figure in {img}/geometry_decode.*")


if __name__ == "__main__":
    main()
