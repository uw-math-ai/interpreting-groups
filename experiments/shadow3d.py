"""3-D shadows of the 4-polytopes (the 'x-cells') each grokked group encodes.

The 4-D Weyl/binary groups (W(D4)->24-cell, W(B4)->16-cell, W(A4)=S5->A4 roots,
2I->600-cell, 2O->24-cell) are usually shown in their flat 2-D Coxeter projection.
This renders them instead as a rotatable 3-D solid: decode embedding -> true 4-D
coords, then project orthographically perpendicular to the (1,1,1,1) diagonal
(the natural high-symmetry shadow). True 4-D edges are drawn in the shadow.

  --snap S.npz  : embedding time series -> rotating + grokking GIF (with phase slider)
  static        : also writes a single-frame PNG/SVG from the final (grokked) embedding
"""
import argparse
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

from core.groups import get_group
from experiments.polytope_figures import _shadow_geometry, render_4d_as_3d, proj_4d_to_3d, _best_decode
from experiments.geometry_decode import true_coords
from experiments.make_grok_gif import add_acc_panel, load_acc, cur_acc, phase_bounds, cur_phase
from sklearn.linear_model import Ridge


def fig_to_img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80); buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True, help="output .gif path (static PNG written alongside)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--spin", type=float, default=0.5, help="degrees of view rotation per frame")
    a = ap.parse_args()
    label = a.label or a.task
    G = get_group(a.task)
    poly = G["geometry"]["polytope"]
    data = np.load(a.snap)
    embeds, epochs, evals = data["embeds"].astype(float), data["epochs"], data["evals"]

    # Geometry can be SHARPEST right after grok and then drift under weight decay
    # (group-dependent over-training degradation, e.g. W(D4)). Pick the post-grok
    # frame with the best held-out decode as the reference for both the static
    # figure and the fixed GIF decoder, so we show the crispest crystallization.
    variants = true_coords(a.task, G)
    from experiments.polytope_figures import cv_decode
    grokked = [t for t in range(len(embeds)) if evals[t] >= 0.9] or [len(embeds) - 1]
    ref = max(grokked, key=lambda t: max(cv_decode(embeds[t], np.asarray(Y, float))[1]
                                         for Y in variants.values()))
    print(f"reference frame = {ref} (epoch {int(epochs[ref]):,}, eval {evals[ref]:.2f})")

    # static figure from the sharpest grokked embedding
    png = a.out.replace(".gif", ".png")
    r2 = render_4d_as_3d(a.task, embeds[ref], label=label, out=png)
    render_4d_as_3d(a.task, embeds[ref], label=label, out=png.replace(".png", ".svg"))
    print(f"static -> {png}  (held-out R²={r2:.3f})")

    # fixed decoder (fit at the reference frame) applied to every snapshot
    best = None
    for nm, Y in variants.items():
        Y = np.asarray(Y, float); Yc = Y - Y.mean(0)
        m = Ridge(1.0).fit(embeds[ref], Yc); s = m.score(embeds[ref], Yc)
        if best is None or s > best[0]:
            best = (s, nm, Y, m)
    _, nm, Y, M = best
    ymean = Y.mean(0)
    W3 = proj_4d_to_3d()
    Vu, inv = np.unique(np.round(Y, 4), axis=0, return_inverse=True)
    per = len(Y) // len(Vu)
    from scipy.spatial.distance import pdist, squareform
    from itertools import combinations
    Dm = squareform(pdist(Vu)); np.fill_diagonal(Dm, 9); dmin = Dm.min()
    edges = [(i, j) for i, j in combinations(range(len(Vu)), 2) if Dm[i, j] < dmin * 1.04]
    Tu = Vu @ W3
    depth = Vu @ (np.ones(4) / 2.0)
    dense = len(Vu) > 60
    import matplotlib.cm as cm
    cols = cm.viridis((depth - depth.min()) / (np.ptp(depth) + 1e-9)) if dense else None
    lim = 1.1 * np.abs(np.vstack([Vu, M.predict(embeds[ref]) + ymean]) @ W3).max()
    bounds = phase_bounds(epochs, evals)
    acc = load_acc(a.snap)

    frames = []
    for t in range(len(embeds)):
        P = (M.predict(embeds[t]) + ymean) @ W3
        cent = np.array([P[inv == k].mean(0) for k in range(len(Vu))])
        fig = plt.figure(figsize=(6.6, 7.1))
        ax = fig.add_axes([0.02, 0.17, 0.96, 0.74], projection="3d")
        if dense:
            ax.add_collection3d(Line3DCollection([Tu[list(e)] for e in edges],
                                                 colors="#888888", linewidths=0.4, alpha=0.10))
            ax.scatter(*Tu.T, s=45, facecolors="none", edgecolors=cols, linewidth=1.4, depthshade=False)
            ax.scatter(*cent.T, s=12, c=cols, depthshade=False, edgecolor="k", linewidth=0.2)
        else:
            ax.add_collection3d(Line3DCollection([Tu[list(e)] for e in edges],
                                                 colors="#d62728", linewidths=1.0, alpha=0.5))
            ax.scatter(*Tu.T, s=90, facecolors="none", edgecolors="#d62728", linewidth=1.6, depthshade=False)
            ax.scatter(*P.T, s=7, c="#1f77b4", alpha=0.16, depthshade=False)
            ax.scatter(*cent.T, s=38, c="#1f77b4", depthshade=False, edgecolor="k", linewidth=0.3)
        for axl in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            axl(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=18, azim=30 + a.spin * t)
        grok = "  ✓" if evals[t] >= 0.95 else ""
        ta, _ = cur_acc(acc, float(epochs[t]))
        ax.set_title(f"{label}: embedding → {poly} (3-D shadow)\nepoch {int(epochs[t]):,}   "
                     f"train {ta:.2f}  eval {evals[t]:.2f}   [{cur_phase(epochs, t, bounds)}]{grok}", fontsize=12)
        add_acc_panel(fig, acc, float(epochs[t]), bounds)
        frames.append(fig_to_img(fig)); plt.close(fig)

    durs = [a.duration] * len(frames); durs[-1] = 1500
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durs, loop=0)
    print(f"saved {a.out}  ({len(frames)} frames, decode R²={r2:.3f}, dim4->3 shadow)")


if __name__ == "__main__":
    main()
