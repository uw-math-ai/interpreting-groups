"""Geometry and topology of the learned embeddings.

Produces two figures from trained checkpoints, with no training required.

figure_geometry (main text):
  (a, b) The encoder image of the one-parameter subgroup exp(t X) for a
  rotation angle t in [0, 4pi], projected onto the top two principal
  components of the trajectory. The spinor-carrying SU(2) model traces a
  loop that closes only at 4pi, with t and t + 2pi on opposite sides
  (the double cover), while the SO(3) model closes at 2pi and retraces.
  (c) The encoder image of the maximal torus of SU(3), a 3D principal
  projection colored by one torus angle.

figure_geometry_app (appendix):
  (d) Antipodal chords: segments joining z(g) and z(-g) in the top two
  principal components of a Haar sample, long for SU(2), degenerate for
  SO(3).
  (e) Embedding distance against geodesic distance for Haar pairs, one
  panel per model, showing the metric shape of the learned code.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.su2 import expmap, get_encoding, haar, qmul, qconj
from core.su2_model import TripleScorer
from core.su3 import SU3LandmarkEncoding, haar_su3

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#1a1a19", "#5f5e56", "#e4e3db"
plt.rcParams.update({
    "font.size": 9, "text.color": INK, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "pdf.fonttype": 42,
})


def load_su2_encoder(run):
    cfg = json.load(open(ROOT / run / "config.json"))
    ck = sorted((ROOT / run).glob("ckpt_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]))[-1]
    st = torch.load(ck, map_location="cpu", weights_only=False)
    eff = {**cfg, **st.get("config", {})}
    enc = get_encoding(eff.get("encoding", "landmark"), eff["variant"], m=eff.get(
        "requested_overrides", {}).get("m", eff["m"]),
        alpha=eff["alpha"], seed=eff["enc_seed"])
    model = TripleScorer(m=eff["m"], d_enc=eff.get("d_enc", 128),
                         enc_hidden=eff.get("enc_hidden", 256),
                         mlp_hidden=eff.get("mlp_hidden", 512),
                         activation=eff.get("activation", "gelu"))
    model.load_state_dict(st["model"])
    model.eval()
    psi = enc.torch_fn("cpu")

    @torch.no_grad()
    def z(q):
        return model.embed(psi(torch.tensor(q, dtype=torch.float32))).numpy()
    return z


def load_su3_encoder(run):
    cfg = json.load(open(ROOT / run / "config.json"))
    ck = sorted((ROOT / run).glob("ckpt_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]))[-1]
    st = torch.load(ck, map_location="cpu", weights_only=False)
    eff = {**cfg, **st.get("config", {})}
    enc = SU3LandmarkEncoding(m=eff["m"], alpha=eff["alpha"],
                              variant=eff["variant"], seed=eff["enc_seed"])
    model = TripleScorer(m=2 * eff["m"])
    model.load_state_dict(st["model"])
    model.eval()

    @torch.no_grad()
    def z(U):
        return model.embed(torch.tensor(enc(U))).numpy()
    return z


def pca(X, k):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, S


def su2_path(n=800, axis=(0.36, 0.48, 0.8)):
    """Quaternions along exp(t X): rotation angle t in [0, 4pi]."""
    t = np.linspace(0, 4 * np.pi, n)
    u = np.array(axis) / np.linalg.norm(axis)
    return t, expmap((t[:, None] / 2.0) * u[None, :])


def panel_loop(ax, z_fn, title, color_map, coincide=False):
    t, q = su2_path()
    Z = z_fn(q)
    P, _ = pca(Z, 2)
    half = len(t) // 2
    sc = ax.scatter(P[:half, 0], P[:half, 1], c=t[:half], cmap=color_map,
                    s=6, lw=0, vmin=0, vmax=4 * np.pi)
    if coincide:
        # second traversal drawn dashed on top to show it retraces
        ax.plot(P[half:, 0], P[half:, 1], "--", color=INK, lw=0.9,
                dashes=(4, 3), alpha=0.8, zorder=4,
                label="second traversal")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        labels = [(0.0, r"$t = 0$ and $t = 2\pi$", (8, 4))]
    else:
        ax.scatter(P[half:, 0], P[half:, 1], c=t[half:], cmap=color_map,
                   s=6, lw=0, vmin=0, vmax=4 * np.pi)
        labels = [(0.0, r"$t=0$", (8, 6)), (2 * np.pi, r"$t=2\pi$", (8, 6))]
    for tv, lab, off in labels:
        i = int(np.argmin(np.abs(t - tv)))
        ax.plot(P[i, 0], P[i, 1], "o", ms=7, mfc="white", mec=INK, mew=1.2,
                zorder=5)
        ax.annotate(lab, (P[i, 0], P[i, 1]), textcoords="offset points",
                    xytext=off, fontsize=8.5, color=INK)
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_xlabel("PC 1")
    ax.set_aspect("equal", adjustable="datalim")
    return sc


def su3_torus(n=70):
    th1, th2 = np.meshgrid(np.linspace(0, 2 * np.pi, n, endpoint=False),
                           np.linspace(0, 2 * np.pi, n, endpoint=False))
    th1, th2 = th1.ravel(), th2.ravel()
    U = np.zeros((th1.size, 3, 3), dtype=complex)
    U[:, 0, 0] = np.exp(1j * th1)
    U[:, 1, 1] = np.exp(1j * th2)
    U[:, 2, 2] = np.exp(-1j * (th1 + th2))
    return th1, th2, U


def main():
    z2 = load_su2_encoder("data/su2/su2_N2048_f0.01_s0")
    z3 = load_su2_encoder("data/su2/so3_N2048_f0.01_s0")

    fig = plt.figure(figsize=(9.2, 2.75), facecolor="white")
    axA = fig.add_subplot(1, 3, 1)
    axB = fig.add_subplot(1, 3, 2)
    axC = fig.add_subplot(1, 3, 3, projection="3d")

    scA = panel_loop(axA, z2, "(a) SU(2) model, path $\\exp(tX)$",
                     "twilight")
    axA.set_ylabel("PC 2")
    panel_loop(axB, z3, "(b) SO(3) model, same path", "twilight",
               coincide=True)
    fig.subplots_adjust(wspace=0.32)
    cb = fig.colorbar(scA, ax=[axA, axB], fraction=0.025, pad=0.02,
                      ticks=[0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    cb.ax.set_yticklabels(["$0$", "$\\pi$", "$2\\pi$", "$3\\pi$", "$4\\pi$"])
    cb.set_label("rotation angle $t$", fontsize=8)

    zc = load_su3_encoder("data/su3/su3_N2048_f0.01_s0")
    th1, th2, U = su3_torus()
    Z = zc(U)
    P, _ = pca(Z, 3)
    axC.scatter(P[:, 0], P[:, 1], P[:, 2], c=th1, cmap="twilight", s=2,
                lw=0, alpha=0.85)
    axC.set_title("(c) SU(3) model, maximal torus", loc="left", fontsize=9.5)
    axC.set_xticklabels([]); axC.set_yticklabels([]); axC.set_zticklabels([])
    axC.grid(False)
    axC.view_init(elev=28, azim=35)

    for ax in (axA, axB):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(ROOT / "paper/figure_geometry.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "paper/figure_geometry.png", dpi=220,
                bbox_inches="tight")
    print("saved paper/figure_geometry.{pdf,png}")

    # ---------------- appendix figure ----------------
    fig2, axs = plt.subplots(1, 4, figsize=(9.6, 2.3), facecolor="white")
    rng = np.random.default_rng(0)
    q = haar(1500, rng)
    for ax, zf, name, col in [(axs[0], z2, "SU(2)", BLUE),
                              (axs[1], z3, "SO(3)", ORANGE)]:
        Z = zf(q)
        P, _ = pca(Z, 2)
        ax.scatter(P[:, 0], P[:, 1], s=3, color=GRID, lw=0)
        Zm = zf(-q[:40])
        Pc = (Z[:40] - Z.mean(0))
        Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][:2]
        A = (Z[:40] - Z.mean(0)) @ Vt.T
        B = (Zm - Z.mean(0)) @ Vt.T
        for i in range(40):
            ax.plot([A[i, 0], B[i, 0]], [A[i, 1], B[i, 1]], "-", color=col,
                    lw=0.8, alpha=0.8)
        ax.plot(A[:, 0], A[:, 1], "o", ms=2.5, color=INK)
        ax.plot(B[:, 0], B[:, 1], "o", ms=2.5, mfc="white", mec=INK, mew=0.6)
        ax.set_title(f"({'de'[ax is axs[1]]}) {name}: $z(g)$ to $z(-g)$",
                     loc="left", fontsize=9)
        ax.set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")

    qa, qb = haar(1200, rng), haar(1200, rng)
    inner = np.sum(qa * qb, axis=1)
    dcover = 2 * np.arccos(np.clip(inner, -1, 1))          # distance on SU(2)
    ang = 2 * np.arccos(np.clip(np.abs(inner), -1, 1))     # distance on SO(3)
    for ax, zf, name, col, dist in [
            (axs[2], z2, "SU(2)", BLUE, dcover),
            (axs[3], z3, "SO(3)", ORANGE, ang)]:
        dz = np.linalg.norm(zf(qa) - zf(qb), axis=1)
        ax.scatter(dist, dz, s=2, color=col, alpha=0.4, lw=0)
        ax.set_xlabel("geodesic distance")
        ax.set_title(f"({'fg'[ax is axs[3]]}) {name}: metric shape",
                     loc="left", fontsize=9)
    axs[2].set_ylabel(r"$\|z(g_1) - z(g_2)\|$")
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig2.tight_layout(pad=0.4)
    fig2.savefig(ROOT / "paper/figure_geometry_app.pdf")
    fig2.savefig(ROOT / "paper/figure_geometry_app.png", dpi=220)
    print("saved paper/figure_geometry_app.{pdf,png}")


if __name__ == "__main__":
    main()


# ====================== appendix zoo (more geometry) ======================
from core.spin4 import Spin4LandmarkEncoding
from core.spin5 import get_spin5_encoding
from core.su3 import OMEGA


def load_pair_encoder(run, enc_cls, model_m_key="m", **enc_kw):
    """Generic loader for spin4/spin5 scorers (element format left to psi)."""
    cfg = json.load(open(ROOT / run / "config.json"))
    ck = sorted((ROOT / run).glob("ckpt_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]))[-1]
    st = torch.load(ck, map_location="cpu", weights_only=False)
    eff = {**cfg, **st.get("config", {})}
    return st, eff


def load_spin4_encoder(run):
    st, eff = load_pair_encoder(run, None)
    enc = Spin4LandmarkEncoding(m=eff.get("requested_overrides", {}).get(
        "m", eff["m"]), alpha=eff["alpha"], variant=eff["variant"],
        seed=eff["enc_seed"])
    model = TripleScorer(m=eff["m"])
    model.load_state_dict(st["model"])
    model.eval()

    @torch.no_grad()
    def z(g):
        return model.embed(torch.tensor(enc(g))).numpy()
    return z


def load_spin5_encoder(run):
    st, eff = load_pair_encoder(run, None)
    enc = get_spin5_encoding(eff.get("encoding", "landmark"),
                             eff["variant"],
                             m=eff.get("requested_overrides", {}).get(
                                 "m", eff["m"]),
                             alpha=eff["alpha"], seed=eff["enc_seed"])
    model = TripleScorer(m=eff["m"])
    model.load_state_dict(st["model"])
    model.eval()

    @torch.no_grad()
    def z(U):
        return model.embed(torch.tensor(enc(U))).numpy()
    return z


def spin4_torus(n=64, axL=(0.2, 0.5, 0.84), axR=(0.7, 0.1, 0.7)):
    tL, tR = np.meshgrid(np.linspace(0, 4 * np.pi, n, endpoint=False),
                         np.linspace(0, 4 * np.pi, n, endpoint=False))
    tL, tR = tL.ravel(), tR.ravel()
    uL = np.array(axL) / np.linalg.norm(axL)
    uR = np.array(axR) / np.linalg.norm(axR)
    g = np.stack([expmap((tL[:, None] / 2) * uL[None, :]),
                  expmap((tR[:, None] / 2) * uR[None, :])], axis=1)
    return tL, tR, g


def sp2_center_path(n=800, axis=(0.36, 0.48, 0.8)):
    """U(t) = diag(q(t), q(t)) passes through the center -I at t = 2pi."""
    t = np.linspace(0, 4 * np.pi, n)
    u = np.array(axis) / np.linalg.norm(axis)
    q = expmap((t[:, None] / 2.0) * u[None, :])
    U = np.zeros((n, 2, 2, 4))
    U[:, 0, 0, :] = q
    U[:, 1, 1, :] = q
    return t, U


def panel_loop_generic(ax, Z, t, title, coincide, labels_at):
    P, _ = pca(Z, 2)
    half = len(t) // 2
    sc = ax.scatter(P[:half, 0], P[:half, 1], c=t[:half], cmap="twilight",
                    s=6, lw=0, vmin=0, vmax=4 * np.pi)
    if coincide:
        ax.plot(P[half:, 0], P[half:, 1], "--", color=INK, lw=0.9,
                dashes=(4, 3), alpha=0.8, zorder=4, label="second traversal")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    else:
        ax.scatter(P[half:, 0], P[half:, 1], c=t[half:], cmap="twilight",
                   s=6, lw=0, vmin=0, vmax=4 * np.pi)
    for tv, lab, off in labels_at:
        i = int(np.argmin(np.abs(t - tv)))
        ax.plot(P[i, 0], P[i, 1], "o", ms=6.5, mfc="white", mec=INK,
                mew=1.1, zorder=5)
        ax.annotate(lab, (P[i, 0], P[i, 1]), textcoords="offset points",
                    xytext=off, fontsize=8, color=INK)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("PC 1")
    ax.set_aspect("equal", adjustable="datalim")
    return sc


def appendix_zoo():
    # -------- maximal tori across the Spin(4) family --------
    fig = plt.figure(figsize=(9.6, 2.55), facecolor="white")
    tL, tR, g = spin4_torus()
    names = [("spin4", "$\\mathrm{Spin}(4)$"), ("so4", "$\\mathrm{SO}(4)$"),
             ("leftq", "$\\mathrm{SO}(3){\\times}\\mathrm{SU}(2)$"),
             ("pso4", "$\\mathrm{PSO}(4)$")]
    for k, (v, lab) in enumerate(names):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        zf = load_spin4_encoder(f"data/spin4/{v}_N2048_f0.01_s0")
        Z = zf(g)
        P, _ = pca(Z, 3)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.mod(tL, 2 * np.pi),
                   cmap="twilight", s=1.5,
                   lw=0, alpha=0.8)
        ax.set_title(f"({'abcd'[k]}) {lab} model", loc="left", fontsize=9)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_zticklabels([]); ax.grid(False)
        ax.view_init(elev=26, azim=38)
    fig.savefig(ROOT / "paper/figure_tori.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "paper/figure_tori.png", dpi=200,
                bbox_inches="tight")
    print("saved paper/figure_tori.{pdf,png}")

    # -------- Spin(5) double cover + SU(3) central triangles --------
    fig2 = plt.figure(figsize=(9.6, 2.5), facecolor="white")
    axA = fig2.add_subplot(1, 4, 1)
    axB = fig2.add_subplot(1, 4, 2)
    t, U = sp2_center_path()
    z5 = load_spin5_encoder("data/spin5/spin5_N2048_f0.01_s0")
    zr = load_spin5_encoder("data/spin5/so5rf_f0.03_s0")
    panel_loop_generic(axA, z5(U), t,
                       "(a) $\\mathrm{Spin}(5)$ model, $\\exp(tX)$",
                       False, [(0.0, "$t=0$", (7, 5)),
                               (2 * np.pi, "$t=2\\pi$", (7, 5))])
    axA.set_ylabel("PC 2")
    panel_loop_generic(axB, zr(U), t,
                       "(b) $\\mathrm{SO}(5)$ model, same path",
                       True, [(0.0, "$t = 0$ and $t = 2\\pi$", (7, 5))])

    rng = np.random.default_rng(3)
    from core.su3 import haar_su3
    for ax, run, lab, coincide in [
            (fig2.add_subplot(1, 4, 3), "data/su3/su3_N2048_f0.01_s0",
             "(c) $\\mathrm{SU}(3)$: center acts by $120^\\circ$", False),
            (fig2.add_subplot(1, 4, 4), "data/su3/psu3_N2048_f0.01_s0",
             "(d) $\\mathrm{PSU}(3)$: center orbits collapse", True)]:
        zf = load_su3_encoder(run)
        Uh = haar_su3(1200, rng)
        Z = zf(Uh)
        Zc = Z - Z.mean(0)
        if not coincide:
            # spectral coordinate of the fitted center action: fit R with
            # z(omega U) ~ R z(U) and pair activations with the LEFT
            # eigenvector l for the eigenvalue nearest exp(2 pi i/3), so
            # w = <l, z> satisfies w(omega U) = e^{2 pi i/3} w(U) up to
            # fit error, and every center orbit rotates 120 degrees about
            # the origin by construction
            Zw = zf(OMEGA * Uh) - Z.mean(0)
            R = np.linalg.lstsq(Zc, Zw, rcond=None)[0].T
            ev, L = np.linalg.eig(R.T)
            i = int(np.argmin(np.abs(ev - np.exp(2j * np.pi / 3))))
            l = L[:, i]
            scale = np.abs(Zc @ l).std() + 1e-9
            l = l / scale
            proj = lambda M: np.stack([(M @ l).real, (M @ l).imag], axis=1)
        else:
            B2 = np.linalg.svd(Zc, full_matrices=False)[2][:2]
            proj = lambda M: M @ B2.T
        ax.scatter(*(proj(Zc).T), s=2.5, color=GRID, lw=0)
        n_orb = 6 if not coincide else 24
        sel = Uh[:n_orb]
        pts = [proj(zf((OMEGA ** k) * sel) - Z.mean(0))
               for k in range(3)]
        cols = [BLUE, ORANGE, AQUA, YELLOW, "#9467bd", INK]
        for i in range(n_orb):
            tri = np.array([pts[0][i], pts[1][i], pts[2][i], pts[0][i]])
            c = cols[i % len(cols)]
            if coincide:
                ax.plot(pts[0][i, 0], pts[0][i, 1], "o", ms=3.2, mfc=BLUE,
                        mec=ORANGE, mew=0.8)
            else:
                ax.plot(tri[:, 0], tri[:, 1], "-o", color=c, lw=1.4, ms=4,
                        alpha=0.9)
        ax.set_title(lab, loc="left", fontsize=9)
        ax.set_xlabel("PC 1" if coincide else "spectral coordinate")
        if not coincide:
            ax.set_aspect("equal", adjustable="datalim")
    for ax in fig2.axes:
        if hasattr(ax, "spines"):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig2.tight_layout(pad=0.4)
    fig2.savefig(ROOT / "paper/figure_covers.pdf")
    fig2.savefig(ROOT / "paper/figure_covers.png", dpi=200)
    print("saved paper/figure_covers.{pdf,png}")


if __name__ == "__main__" and True:
    appendix_zoo()
