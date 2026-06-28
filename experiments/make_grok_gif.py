"""Animate the embedding converging onto the polytope vertices over training,
with an epoch slider + marked grokking phases at the bottom of every frame.

Fixed final-epoch decoder applied to every snapshot (blob -> grok -> crystallized).
3-D polytopes animate natively; 4-D ones in their Coxeter-plane projection. The
slider shows three phases inferred from the eval curve:
  memorization (eval ~chance) -> grokking (eval climbing) -> crystallized (eval~1).
"""
import argparse
import io
import os
from itertools import combinations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from PIL import Image

from core.groups import get_group
from core import quaternions as q
from experiments.geometry_decode import true_coords
from experiments.polytope_figures import symmetry_matrix, coxeter_basis, a5_icosahedron_coords


def fig_to_img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80); buf.seek(0)   # fixed size (no tight)
    return Image.open(buf).convert("RGB")


def phase_bounds(epochs, evals):
    epochs = np.asarray(epochs, float); evals = np.asarray(evals, float)
    Emax = float(epochs[-1])
    # smooth eval to ignore early transient noise when locating phase boundaries
    w = max(1, len(evals) // 40)
    sm = np.convolve(evals, np.ones(w)/w, mode="same") if w > 1 else evals
    comp = float(epochs[np.argmax(sm >= 0.90)]) if (sm >= 0.90).any() else Emax
    onset = float(epochs[np.argmax(sm >= 0.15)]) if (sm >= 0.15).any() else comp
    onset = min(onset, comp)
    return Emax, onset, comp


def add_phase_slider(fig, epochs, evals, t, bounds):
    Emax, onset, comp = bounds
    cur = float(epochs[t])
    ax = fig.add_axes([0.12, 0.075, 0.76, 0.05])
    ax.set_xlim(0, Emax); ax.set_ylim(0, 1)
    ax.axvspan(0, onset, color="#cfd8dc")                 # memorization
    ax.axvspan(onset, comp, color="#ffb74d")             # grokking
    ax.axvspan(comp, Emax, color="#a5d6a7")              # crystallized
    for x0, x1, lab in [(0, onset, "memorization"), (onset, comp, "grokking"), (comp, Emax, "crystallized")]:
        if x1 - x0 > 0.07 * Emax:
            ax.text((x0 + x1) / 2, 0.5, lab, ha="center", va="center", fontsize=8, color="#333")
    ax.axvline(cur, color="k", lw=2.2)
    ax.plot([cur], [1.0], marker="v", color="k", ms=9, clip_on=False)
    ax.set_yticks([])
    ticks = sorted({0.0, onset, comp, Emax})
    ax.set_xticks(ticks); ax.set_xticklabels([f"{x/1000:.0f}k" for x in ticks], fontsize=8)
    ax.tick_params(length=2)
    for s in ax.spines.values():
        s.set_visible(False)


def cur_phase(epochs, t, bounds):
    _, onset, comp = bounds
    e = float(epochs[t])
    return "memorization" if e < onset else ("grokking" if e < comp else "crystallized")


def load_acc(snap):
    """Per-epoch train accuracy + eval-point eval accuracy from the checkpoint that
    pairs with a *_snapshots.npz (the snapshot npz only stores eval at snapshot epochs)."""
    ck = snap.replace("_snapshots.npz", ".pt")
    if not os.path.exists(ck):
        for alt in ("_best.pt", "_grok20k.pt", "_grok10k.pt"):
            c = snap.replace("_snapshots.npz", alt)
            if os.path.exists(c):
                ck = c; break
    d = torch.load(ck, map_location="cpu", weights_only=False)
    tr = np.asarray(d.get("train_accs", []), float)
    tr_ep = np.arange(1, len(tr) + 1, dtype=float)
    if len(tr) > 1500:                                   # log-spaced thinning for a light curve
        idx = np.unique(np.round(np.geomspace(1, len(tr), 1500)).astype(int) - 1)
        tr_ep, tr = tr_ep[idx], tr[idx]
    ev = np.asarray(d.get("eval_accs", []), float)
    ev_ep = np.asarray(d.get("eval_epochs", []), float)
    return tr_ep, tr, ev_ep, ev


def cur_acc(acc, cur):
    tr_ep, tr, ev_ep, ev = acc
    ta = float(np.interp(cur, tr_ep, tr)) if len(tr) else float("nan")
    ea = float(np.interp(cur, ev_ep, ev)) if len(ev) else float("nan")
    return ta, ea


def add_acc_panel(fig, acc, cur, bounds, rect=(0.115, 0.065, 0.80, 0.085)):
    """Bottom panel: train- and eval-accuracy curves vs epoch (log x, so the fast
    memorization phase is resolved), faint phase bands behind them, and a vertical
    rule = the current epoch (the 'slider')."""
    tr_ep, tr, ev_ep, ev = acc
    Emax, onset, comp = bounds
    ax = fig.add_axes(list(rect))
    onset = max(onset, 1.0); comp = max(comp, onset + 1)
    ax.axvspan(1, onset, color="#eceff1", lw=0)                 # memorization
    ax.axvspan(onset, comp, color="#fff3e0", lw=0)             # grokking
    ax.axvspan(comp, Emax, color="#e8f5e9", lw=0)             # crystallized
    if len(tr):
        ax.plot(np.clip(tr_ep, 1, None), tr, color="#455a64", lw=1.3, label="train acc")
    if len(ev):
        ax.plot(np.clip(ev_ep, 1, None), ev, color="#1565c0", lw=1.6, label="eval acc")
    ax.axvline(max(cur, 1), color="k", lw=1.8)
    ax.set_xscale("log"); ax.set_xlim(1, Emax); ax.set_ylim(-0.04, 1.10)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", ".5", "1"], fontsize=6)
    ax.set_xlabel("epoch (log scale)", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6, length=2)
    ax.legend(loc="upper left", fontsize=6, ncol=2, frameon=False,
              handlelength=1.1, columnspacing=1.0, borderaxespad=0.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--duration", type=int, default=130)
    a = ap.parse_args()
    label = a.label or a.task
    data = np.load(a.snap)
    embeds, epochs, evals = data["embeds"].astype(float), data["epochs"], data["evals"]
    G = get_group(a.task)

    if a.task == "icosahedral":
        variants = q.ih_icosahedron_coords()
    elif a.task == "A5":
        variants = a5_icosahedron_coords()
    else:
        variants = true_coords(a.task, G)
    final = embeds[-1]
    best = None
    for nm, Y in variants.items():
        Y = np.asarray(Y, float); Yc = Y - Y.mean(0)
        m = Ridge(1.0).fit(final, Yc); r2 = m.score(final, Yc)
        if best is None or r2 > best[0]:
            best = (r2, nm, Y, m)
    r2f, nm, Y, M = best
    D = Y.shape[1]
    ymean = Y.mean(0)
    Vu, inv = np.unique(np.round(Y, 3), axis=0, return_inverse=True)
    poly = "icosahedron" if a.task in ("icosahedral", "A5") else G["geometry"]["polytope"]
    bounds = phase_bounds(epochs, evals)
    acc = load_acc(a.snap)

    Dm = squareform(pdist(Vu)); np.fill_diagonal(Dm, 9); dmin = Dm.min()
    edges = [(i, j) for i, j in combinations(range(len(Vu)), 2) if Dm[i, j] < dmin * (1.05 if D == 3 else 1.04)]
    use3d = (D == 3)
    if not use3d:
        W, fold = coxeter_basis(symmetry_matrix(a.task)); Tu = Vu @ W
    else:
        faces = [list(c) for c in combinations(range(len(Vu)), 3)
                 if Dm[c[0], c[1]] < dmin*1.05 and Dm[c[0], c[2]] < dmin*1.05 and Dm[c[1], c[2]] < dmin*1.05]
        lim = 1.15 * np.abs(np.vstack([Vu, M.predict(final) + ymean])).max()

    frames = []
    for t in range(len(embeds)):
        P = M.predict(embeds[t]) + ymean
        cent = np.array([P[inv == k].mean(0) for k in range(len(Vu))])
        fig = plt.figure(figsize=(6.6, 7.1))
        grok = "  ✓" if evals[t] >= 0.95 else ""
        ta, ea = cur_acc(acc, float(epochs[t]))
        ttl = (f"{label}: embedding → {poly}\nepoch {int(epochs[t]):,}   "
               f"train {ta:.2f}  eval {evals[t]:.2f}   [{cur_phase(epochs, t, bounds)}]{grok}")
        if use3d:
            ax = fig.add_axes([0.02, 0.17, 0.96, 0.74], projection="3d")
            ax.add_collection3d(Poly3DCollection([Vu[f] for f in faces], alpha=0.07, facecolor="#1f77b4"))
            ax.add_collection3d(Line3DCollection([Vu[list(e)] for e in edges], colors="#d62728", linewidths=1.6))
            ax.scatter(*Vu.T, s=130, facecolors="none", edgecolors="#d62728", linewidth=2.0, depthshade=False)
            ax.scatter(*P.T, s=8, c="#1f77b4", alpha=0.25, depthshade=False)
            ax.scatter(*cent.T, s=45, c="#1f77b4", depthshade=False, edgecolor="k", linewidth=0.3)
            for axl in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
                axl(-lim, lim)
            ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=30 + 0.15*t)
        else:
            Pp = P @ W; Cp = cent @ W
            ax = fig.add_axes([0.06, 0.17, 0.88, 0.74])
            for i, j in edges:
                ax.plot([Tu[i, 0], Tu[j, 0]], [Tu[i, 1], Tu[j, 1]], color="#d62728", lw=0.8, alpha=0.5)
            ax.scatter(Tu[:, 0], Tu[:, 1], s=80, facecolors="none", edgecolors="#d62728", linewidths=1.3, zorder=3)
            ax.scatter(Pp[:, 0], Pp[:, 1], s=7, c="#1f77b4", alpha=0.25, zorder=2)
            ax.scatter(Cp[:, 0], Cp[:, 1], s=32, c="#1f77b4", edgecolors="k", linewidths=0.3, zorder=4)
            lim = 1.2 * np.abs(Tu).max()
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(ttl, fontsize=12)
        add_acc_panel(fig, acc, float(epochs[t]), bounds)
        frames.append(fig_to_img(fig)); plt.close(fig)

    durs = [a.duration] * len(frames); durs[-1] = 1500
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durs, loop=0)
    print(f"saved {a.out}  ({len(frames)} frames, final decode R²={r2f:.3f}, variant={nm}, dim={D})")


if __name__ == "__main__":
    main()
