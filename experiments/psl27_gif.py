"""Convergence GIF for PSL(2,7)=GL(3,2): the embedding -> the Fano plane's
7-point structure = the 6-simplex, projected to a heptagon (with all diagonals,
the simplex shadow). PSL(2,7) has no Euclidean polytope, but its dominant 6-dim
irrep is exactly the 7-point Fano permutation rep minus trivial, so a linear
readout maps the embedding onto the 7 Fano points (held-out R^2~0.99).
"""
import io, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import Ridge
from PIL import Image
from core.checkpoint import load_model
from experiments.make_grok_gif import add_acc_panel, load_acc, cur_acc, phase_bounds, cur_phase


def gl32_elements():
    els = []
    for bits in range(512):
        M = np.array([[(bits >> (3*r+c)) & 1 for c in range(3)] for r in range(3)], dtype=np.int64)
        d = (M[0,0]*(M[1,1]*M[2,2]^M[1,2]*M[2,1]) ^ M[0,1]*(M[1,0]*M[2,2]^M[1,2]*M[2,0])
             ^ M[0,2]*(M[1,0]*M[2,1]^M[1,1]*M[2,0])) & 1
        if d == 1: els.append(M)
    key = {M.tobytes(): i for i, M in enumerate(els)}
    i0 = key[np.eye(3, dtype=np.int64).tobytes()]
    els[0], els[i0] = els[i0], els[0]
    return els


def fano_setup():
    els = gl32_elements()
    pts = [np.array([(v >> k) & 1 for k in range(3)], dtype=np.int64) for v in range(1, 8)]
    pk = {p.tobytes(): i for i, p in enumerate(pts)}
    fano = np.array([pk[((M @ pts[0]) % 2).tobytes()] for M in els])     # els[i]: p0 -> point
    C = np.eye(7)[fano] - 1/7                                            # 6-simplex coords
    # heptagon projection: order the 7 points by an order-7 element's 7-cycle
    I = np.eye(3, dtype=np.int64); c = None
    for M in els:
        X = M.copy(); o = 1
        while not np.array_equal(X % 2, I):
            X = (X @ M) % 2; o += 1
        if o == 7:
            c = M; break
    order = [0]
    for _ in range(6):
        order.append(pk[((c @ pts[order[-1]]) % 2).tobytes()])
    pos = np.zeros(7, int)
    for j, p in enumerate(order):
        pos[p] = j
    ang = 2 * np.pi * pos / 7
    W = np.stack([np.cos(ang), np.sin(ang)], 1)                          # 7 x 2 heptagon
    return C, W


def fig_to_img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80); buf.seek(0)   # fixed size for GIF
    return Image.open(buf).convert("RGB")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default="data/weyl/PSL27_s1/PSL27_P168_d128_h4_mlp512_s1_snapshots.npz")
    ap.add_argument("--out", default="gallery/grokking_gifs/PSL27_fano_heptagon.gif")
    ap.add_argument("--duration", type=int, default=130)
    a = ap.parse_args()
    snap, out = a.snap, a.out
    data = np.load(snap)
    embeds, epochs, evals = data["embeds"].astype(float), data["epochs"], data["evals"]
    C, W = fano_setup()
    Hv = (np.eye(7) - 1/7) @ W                                           # 7 heptagon vertices in 2D
    M = Ridge(1.0).fit(embeds[-1], C)                                    # fixed final decoder
    edges = list(combinations(range(7), 2))                             # K7 = simplex shadow
    bounds = phase_bounds(epochs, evals)
    acc = load_acc(snap)
    frames = []
    for t in range(len(embeds)):
        P = (M.predict(embeds[t])) @ W
        lab = np.argmin(((P[:, None, :] - Hv[None, :, :])**2).sum(2), axis=1)
        cent = np.array([P[lab == k].mean(0) if (lab == k).any() else Hv[k] for k in range(7)])
        fig = plt.figure(figsize=(6.6, 7.1))
        ax = fig.add_axes([0.06, 0.17, 0.88, 0.74])
        for i, j in edges:
            ax.plot([Hv[i,0], Hv[j,0]], [Hv[i,1], Hv[j,1]], color="#d62728", lw=0.8, alpha=0.45)
        ax.scatter(Hv[:,0], Hv[:,1], s=170, facecolors="none", edgecolors="#d62728", linewidths=2.0, zorder=3)
        ax.scatter(P[:,0], P[:,1], s=8, c="#1f77b4", alpha=0.25, zorder=2)
        ax.scatter(cent[:,0], cent[:,1], s=55, c="#1f77b4", edgecolors="k", linewidths=0.3, zorder=4)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        grok = "  ✓" if evals[t] >= 0.95 else ""
        ta, _ = cur_acc(acc, float(epochs[t]))
        ax.set_title(f"PSL(2,7) → Fano plane / 6-simplex (heptagon)\nepoch {int(epochs[t]):,}   "
                     f"train {ta:.2f}  eval {evals[t]:.2f}   [{cur_phase(epochs, t, bounds)}]{grok}", fontsize=12)
        add_acc_panel(fig, acc, float(epochs[t]), bounds)
        frames.append(fig_to_img(fig)); plt.close(fig)
    durs = [a.duration]*len(frames); durs[-1] = 1500
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0)
    frames[-1].save(out.replace(".gif", ".png"))
    print(f"saved {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
