"""Pretty pictures: render the symmetric polytope each grokked model encodes.

Raw PCA of the embedding does NOT show the polytope (it stores the rep's matrix
coefficients, not the orbit). But the embedding linearly maps to true polytope
coordinates (held-out R^2 0.78-0.98, see geometry_decode.py), so we apply that
learned map and render the result where the symmetry is visible:

  * A5  -> icosahedron      : native 3-D, vertices/edges/faces (seed = a 5-fold axis)
  * 2I  -> 600-cell         : 10-fold Coxeter-plane projection
  * 2O  -> 24-cell(+dual)   :  8-fold Coxeter-plane projection
  * S5  -> A4 root system   :  5-fold Coxeter-plane projection

Red = true polytope (vertices + edges); blue = the model's held-out decoded
points. For A5/2I the model uses the Galois-conjugate (sqrt5->-sqrt5) irrep, so we
decode to whichever conjugate variant scores higher.
"""
import argparse
import os
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch

from core.checkpoint import load_model, image_dir
from core.groups import get_group
from core import quaternions as q
from experiments.geometry_decode import true_coords

PHI = (1 + np.sqrt(5)) / 2


def Lmat(quat):
    w, x, y, z = quat
    return np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])


def order(M, mx=24):
    I = np.eye(M.shape[0]); P = M.copy()
    for k in range(1, mx + 1):
        if np.allclose(P, I, atol=1e-6): return k
        P = P @ M
    return -1


def coxeter_basis(R):
    """Orthonormal basis of the eigenplane with the smallest positive rotation angle."""
    w, V = np.linalg.eig(R); ang = np.angle(w)
    cand = [i for i in range(len(w)) if ang[i] > 1e-3]
    i = cand[int(np.argmin(ang[np.array(cand)]))]
    u = np.real(V[:, i]); v = np.imag(V[:, i])
    return np.stack([u / np.linalg.norm(u), v / np.linalg.norm(v)], 1), int(round(2*np.pi/ang[i]))


def cv_decode(emb, Y):
    Yc = Y - Y.mean(0); kf = KFold(5, shuffle=True, random_state=0)
    pred = np.zeros_like(Yc); r = []
    for tr, te in kf.split(emb):
        m = Ridge(1.0).fit(emb[tr], Yc[tr]); pred[te] = m.predict(emb[te])
        r.append(1 - ((Yc[te]-pred[te])**2).sum() / ((Yc[te]-Yc[tr].mean(0))**2).sum())
    return pred + Y.mean(0), float(np.mean(r))


def a5_icosahedron_coords():
    """12-vertex icosahedron preserved by the quaternion A5 (orbit of a 5-fold axis),
    plus the Galois-conjugate relabeling."""
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
    reps = np.array(reps); Rm = [q.rotation_matrix(r) for r in reps]
    seed = None
    for M in Rm:
        if order(M, 6) == 5:
            w, V = np.linalg.eig(M); seed = np.real(V[:, int(np.argmin(np.abs(w-1)))])
            seed /= np.linalg.norm(seed); break
    coords = np.array([M @ seed for M in Rm])
    def fc(c):
        a = abs(c)
        if abs(a-PHI/2) < 1e-3: return -np.sign(c)*(PHI-1)/2
        if abs(a-(PHI-1)/2) < 1e-3: return -np.sign(c)*PHI/2
        return c
    keyr = {tuple(np.round(v, 5)): i for i, v in enumerate(reps)}
    gp = np.array([keyr.get(tuple(np.round(canon(np.array([fc(c) for c in v])), 5)),
                   int(np.abs(reps-canon(np.array([fc(c) for c in v]))).sum(1).argmin())) for v in reps])
    return {"canonical": coords, "galois": coords[gp]}


def render_3d_icosahedron(emb, img, variants=None, label="A5"):
    if variants is None:
        variants = a5_icosahedron_coords()
    best = max(((nm,) + cv_decode(emb, Y) for nm, Y in variants.items()), key=lambda t: t[2])
    nm, pred, r2 = best
    Y = variants[nm]
    Vu, inv = np.unique(np.round(Y, 3), axis=0, return_inverse=True)
    cent = np.array([pred[inv == k].mean(0) for k in range(len(Vu))])  # per-vertex decoded centroid
    per_elem = len(Y) // len(Vu)
    Dm = squareform(pdist(Vu)); np.fill_diagonal(Dm, 9); dmin = Dm.min()
    edges = [(i, j) for i, j in combinations(range(12), 2) if Dm[i, j] < dmin*1.05]
    faces = [[i, j, k] for i, j, k in combinations(range(12), 3)
             if Dm[i, j] < dmin*1.05 and Dm[i, k] < dmin*1.05 and Dm[j, k] < dmin*1.05]
    fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection([Vu[f] for f in faces], alpha=0.12, facecolor="#1f77b4"))
    ax.add_collection3d(Line3DCollection([Vu[list(e)] for e in edges], colors="#d62728", linewidths=2))
    # hollow red vertices so the (often coincident) blue decode shows through
    ax.scatter(*Vu.T, s=200, facecolors="none", edgecolors="#d62728", linewidth=2.2, depthshade=False)
    if per_elem > 1:
        ax.scatter(*pred.T, s=10, c="#1f77b4", alpha=0.25, depthshade=False)   # individuals (faint)
        ax.scatter(*cent.T, s=60, c="#1f77b4", depthshade=False, edgecolor="k", linewidth=0.3)  # centroids
    else:
        ax.scatter(*pred.T, s=45, c="#1f77b4", depthshade=False, edgecolor="k", linewidth=0.3)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=18, azim=32)
    ax.set_title(f"{label}: embedding decodes to the ICOSAHEDRON (held-out R²={r2:.2f})", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(img, f"polytope.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return r2


def symmetry_matrix(task):
    # reflection groups carry their own DxD matrices -> Coxeter element = highest order
    G = get_group(task)
    mats = G.get("_mats")
    if mats is not None:
        best = (0, None)
        for M in mats:
            o = order(M, 40)
            if o > best[0]: best = (o, M)
        return best[1]
    if task in ("2I", "2O", "2T"):
        V = {"2I": q.verts_2I, "2O": q.verts_2O, "2T": q.verts_2T}[task]()
        best = (0, None)
        for qq in V:
            o = order(Lmat(qq))
            if o > best[0]: best = (o, Lmat(qq))
        return best[1]
    d = 5; M = np.eye(d) - np.ones((d, d)) / d; B = np.linalg.qr(M)[0][:, :d-1]
    cyc = [1, 2, 3, 4, 0]; P = np.zeros((d, d))
    for x in range(d): P[cyc[x], x] = 1.0
    return B.T @ P @ B


def render_4d(task, emb, img, label=None):
    label = label or task
    G = get_group(task)
    variants = true_coords(task, G)
    best = max(((nm,) + cv_decode(emb, Y) for nm, Y in variants.items()), key=lambda t: t[2])
    nm, pred, r2 = best
    Y = variants[nm]
    W, fold = coxeter_basis(symmetry_matrix(task))
    Vu, inv = np.unique(np.round(Y, 4), axis=0, return_inverse=True)
    cent = np.array([pred[inv == k].mean(0) for k in range(len(Vu))])   # per-vertex decoded centroid
    per_elem = len(Y) // len(Vu)
    Tu = Vu @ W; P = pred @ W; C = cent @ W
    Dm = squareform(pdist(Vu)); np.fill_diagonal(Dm, 9); dmin = Dm.min()
    edges = [(i, j) for i, j in combinations(range(len(Vu)), 2) if Dm[i, j] < dmin*1.04]
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, j in edges:
        ax.plot([Tu[i, 0], Tu[j, 0]], [Tu[i, 1], Tu[j, 1]], color="#d62728", lw=0.8, alpha=0.5)
    ax.scatter(Tu[:, 0], Tu[:, 1], s=90, facecolors="none", edgecolors="#d62728", linewidths=1.4, zorder=3)
    if per_elem > 1:
        ax.scatter(P[:, 0], P[:, 1], s=8, c="#1f77b4", alpha=0.22, zorder=2)        # individuals (faint)
        ax.scatter(C[:, 0], C[:, 1], s=34, c="#1f77b4", edgecolors="k", linewidths=0.3, zorder=4)  # centroids
    else:
        ax.scatter(P[:, 0], P[:, 1], s=14, c="#1f77b4", alpha=0.85, zorder=2)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(f"{label}: embedding decodes to the {G['geometry']['polytope']}\n"
                 f"{fold}-fold Coxeter-plane projection (held-out R²={r2:.2f})", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(img, f"polytope.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return r2


def _best_decode(task, emb):
    G = get_group(task)
    variants = true_coords(task, G)
    best = max(((nm,) + cv_decode(emb, np.asarray(Y, float)) for nm, Y in variants.items()),
              key=lambda t: t[2])
    nm, pred, r2 = best
    return G, np.asarray(variants[nm], float), pred, r2


def proj_4d_to_3d():
    """Orthonormal 4x3 basis of the hyperplane perpendicular to (1,1,1,1).

    This is the natural high-symmetry 3-D *shadow* of these 4-polytopes: it drops
    the diagonal axis and projects orthographically into the perpendicular 3-space
    (24-cell -> cuboctahedral shell, 600-cell -> nested icosahedral shells, ...).
    """
    Q, _ = np.linalg.qr(np.column_stack([np.ones(4), np.eye(4)[:, :3]]))
    return Q[:, 1:4]


def _shadow_geometry(task, emb):
    """Decode -> 4-D true coords, project to 3-D, return
    (G, Vu3, edges, pred3, cent3, per, r2, depth) where depth = signed component
    along the dropped (1,1,1,1) axis (the 'shell' index for vertex-first views)."""
    G, Y, pred, r2 = _best_decode(task, emb)
    W3 = proj_4d_to_3d()
    Vu, inv = np.unique(np.round(Y, 4), axis=0, return_inverse=True)
    cent = np.array([pred[inv == k].mean(0) for k in range(len(Vu))])
    per = len(Y) // len(Vu)
    Dm = squareform(pdist(Vu)); np.fill_diagonal(Dm, 9); dmin = Dm.min()
    edges = [(i, j) for i, j in combinations(range(len(Vu)), 2) if Dm[i, j] < dmin * 1.04]
    depth = Vu @ (np.ones(4) / 2.0)
    return G, Vu @ W3, edges, pred @ W3, cent @ W3, per, r2, depth


def render_4d_as_3d(task, emb, img=None, label=None, ax=None, view=(18, 30), out=None):
    """3-D shadow (orthographic projection perpendicular to (1,1,1,1)) of a 4-polytope.

    Complements the 2-D Coxeter projection in render_4d: shows the 4-cell as a solid
    you can rotate, with the true 4-D edges drawn in the 3-D shadow. Dense polytopes
    (e.g. the 600-cell, 120 verts) are drawn as depth-coloured concentric shells with
    faint edges so the nested-icosahedra structure is legible.
    """
    label = label or task
    G, Tu, edges, Pp, C, per, r2, depth = _shadow_geometry(task, emb)
    dense = len(Tu) > 60
    own = ax is None
    if own:
        fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection="3d")
    if dense:
        import matplotlib.cm as cm
        norm = (depth - depth.min()) / (np.ptp(depth) + 1e-9)
        cols = cm.viridis(norm)
        ax.add_collection3d(Line3DCollection([Tu[list(e)] for e in edges],
                                             colors="#888888", linewidths=0.4, alpha=0.12))
        ax.scatter(*Tu.T, s=55, facecolors="none", edgecolors=cols, linewidth=1.6, depthshade=False)
        ax.scatter(*C.T, s=14, c=cols, depthshade=False, edgecolor="k", linewidth=0.2)  # decoded, shell-coloured
    else:
        ax.add_collection3d(Line3DCollection([Tu[list(e)] for e in edges],
                                             colors="#d62728", linewidths=1.1, alpha=0.55))
        ax.scatter(*Tu.T, s=110, facecolors="none", edgecolors="#d62728", linewidth=1.8, depthshade=False)
        if per > 1:
            ax.scatter(*Pp.T, s=7, c="#1f77b4", alpha=0.16, depthshade=False)
            ax.scatter(*C.T, s=42, c="#1f77b4", depthshade=False, edgecolor="k", linewidth=0.3)
        else:
            ax.scatter(*Pp.T, s=34, c="#1f77b4", depthshade=False, edgecolor="k", linewidth=0.3)
    lim = 1.1 * np.abs(Tu).max()
    for axl in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        axl(-lim, lim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=view[0], azim=view[1])
    if own:
        extra = "  (vertices coloured by 4-D depth = concentric shells)" if dense else ""
        ax.set_title(f"{label}: {G['geometry']['polytope']} — 3-D shadow of the 4-polytope\n"
                     f"(orthographic projection; held-out R²={r2:.2f}){extra}", fontsize=11)
        outs = [out] if out else [os.path.join(img, f"polytope_3d.{e}") for e in ("png", "svg")]
        for p in outs:
            fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return r2


def render_3d_generic(task, emb, img, label=None):
    label = label or task
    G, Y, pred, r2 = _best_decode(task, emb)
    Vu, inv = np.unique(np.round(Y, 3), axis=0, return_inverse=True)
    cent = np.array([pred[inv == k].mean(0) for k in range(len(Vu))]); per = len(Y) // len(Vu)
    from scipy.spatial import ConvexHull
    faces, eset = [], set()
    try:
        for s in ConvexHull(Vu).simplices:
            faces.append(Vu[s])
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
                eset.add(tuple(sorted((int(a), int(b)))))
    except Exception:
        pass
    fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection="3d")
    if faces: ax.add_collection3d(Poly3DCollection(faces, alpha=0.10, facecolor="#1f77b4"))
    if eset: ax.add_collection3d(Line3DCollection([Vu[list(e)] for e in eset], colors="#d62728", linewidths=1.4))
    ax.scatter(*Vu.T, s=170, facecolors="none", edgecolors="#d62728", linewidth=2.0, depthshade=False)
    if per > 1:
        ax.scatter(*pred.T, s=9, c="#1f77b4", alpha=0.22, depthshade=False)
        ax.scatter(*cent.T, s=55, c="#1f77b4", depthshade=False, edgecolor="k", linewidth=0.3)
    else:
        ax.scatter(*pred.T, s=45, c="#1f77b4", depthshade=False, edgecolor="k", linewidth=0.3)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=20, azim=30)
    ax.set_title(f"{label}: embedding decodes to the {G['geometry']['polytope']} (held-out R²={r2:.2f})", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(img, f"polytope.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig); return r2


def render_2d(task, emb, img, label=None):
    label = label or task
    G, Y, pred, r2 = _best_decode(task, emb)
    Vu, inv = np.unique(np.round(Y, 3), axis=0, return_inverse=True)
    cent = np.array([pred[inv == k].mean(0) for k in range(len(Vu))]); per = len(Y) // len(Vu)
    Dm = squareform(pdist(Vu)); np.fill_diagonal(Dm, 9); dmin = Dm.min()
    edges = [(i, j) for i, j in combinations(range(len(Vu)), 2) if Dm[i, j] < dmin*1.08]
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, j in edges:
        ax.plot([Vu[i, 0], Vu[j, 0]], [Vu[i, 1], Vu[j, 1]], color="#d62728", lw=1.0, alpha=0.6)
    ax.scatter(Vu[:, 0], Vu[:, 1], s=150, facecolors="none", edgecolors="#d62728", linewidths=1.6, zorder=3)
    if per > 1:
        ax.scatter(pred[:, 0], pred[:, 1], s=10, c="#1f77b4", alpha=0.3, zorder=2)
        ax.scatter(cent[:, 0], cent[:, 1], s=45, c="#1f77b4", edgecolors="k", linewidths=0.3, zorder=4)
    else:
        ax.scatter(pred[:, 0], pred[:, 1], s=40, c="#1f77b4", zorder=2)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(f"{label}: embedding decodes to the {G['geometry']['polytope']} (held-out R²={r2:.2f})", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(img, f"polytope.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig); return r2


def _geom_dim(task):
    geo = get_group(task)["geometry"]
    if geo.get("coords4d") is not None: return 4
    if geo.get("coords3d") is not None: return 3
    if geo.get("coordsND") is not None: return int(np.asarray(geo["coordsND"]).shape[1])
    return None


def render(task, ckpt, embed_dim=128, mlp_dim=512, seed=1):
    n = get_group(task)["n"]
    emb = load_model(ckpt, torch.device("cpu"))[0].embed.weight[:n].detach().numpy().astype(float)
    img = image_dir(n, embed_dim, 4, mlp_dim, seed, task=task)
    if task == "A5":
        r2 = render_3d_icosahedron(emb, img)
    elif task == "icosahedral":
        r2 = render_3d_icosahedron(emb, img, variants=q.ih_icosahedron_coords(), label="I_h")
    else:
        D = _geom_dim(task)
        if D == 2: r2 = render_2d(task, emb, img)
        elif D == 3: r2 = render_3d_generic(task, emb, img)
        else:
            r2 = render_4d(task, emb, img)                 # 2-D Coxeter projection
            render_4d_as_3d(task, emb, img)                # + 3-D shadow (polytope_3d.png)
    print(f"  {task}: held-out R²={r2:.3f} -> {img}/polytope.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.parse_args()
    a = ap.parse_args()
    render(a.task, a.ckpt)
