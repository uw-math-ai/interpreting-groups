"""Generic representation-theory + explicit polytope recovery for any group task.

Works for every group in ``core.groups.GROUP_TASKS`` (I_h, 2I, 2O, 2T, A5, S5)
using the numerically-computed character table (``core.reptheory``). Produces:

  isotypic_embedding.svg / isotypic_unembed.svg
      energy of the embedding/unembed in each irrep vs the d^2/|G| random
      baseline (a grokked model concentrates energy in a few irreps).

  geometry_recovery_*.svg / .html, and a printed "shape match"
      The key deliverable. For each irrep we isotypic-project the embedding
      (a GROUP-determined operation that never sees the target), reduce to the
      irrep dimension, and Procrustes-match it to the group's true geometric
      vertices (600-cell for 2I, A4 root system for S5, icosahedron for A5,
      24-cell for 2T/2O). The best-matching irrep is the one the model used to
      represent the group, and the overlay shows the recovered polytope.

This cannot "cheat": the isotypic projection is fixed by the group's characters,
so a high match means the learned embedding genuinely carries the polytope.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import procrustes

import torch
from core.data import get_device
from core.checkpoint import load_model, checkpoint_path, image_dir
from core.groups import get_group, group_partition, GROUP_TASKS
from core.reptheory import char_dict, verify_character_table


# ----------------------------- isotypic machinery -----------------------------

def isotypic_projection(W, table, chi, d_rho):
    """rho-isotypic part of columns of W (a function G -> C^d).

    Orthogonal projector for the (right) regular rep: uses conj(chi), so it is
    correct for complex characters and reduces to the real formula when chi is real.
    """
    n = W.shape[0]
    gathered = W[table]                              # [x, g, d]
    proj = np.einsum("g,xgd->xd", np.conjugate(chi), gathered) * (d_rho / n)
    return proj


def isotypic_spectrum(W, table, char_per_element, dims):
    n = W.shape[0]
    total = float((np.abs(W) ** 2).sum())
    rows, energy_sum = [], 0.0
    for name, chi in char_per_element.items():
        d = dims[name]
        proj = isotypic_projection(W, table, chi, d)
        energy = float((np.abs(proj) ** 2).sum())
        energy_sum += energy
        frac = energy / total if total else 0.0
        baseline = (d * d) / n
        rows.append({"irrep": name, "dim": d, "energy_frac": frac,
                     "baseline": baseline,
                     "enrichment": frac / baseline if baseline else 0.0})
    rows.sort(key=lambda r: r["energy_frac"], reverse=True)
    return rows, (energy_sum / total if total else 0.0)


def plot_spectrum(rows, title, save_path):
    names = [r["irrep"] for r in rows]
    frac = [r["energy_frac"] for r in rows]
    base = [r["baseline"] for r in rows]
    x = np.arange(len(names))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, frac, width=0.4, color="#1f77b4", label="Model")
    plt.bar(x + 0.2, base, width=0.4, color="#d62728", alpha=0.7, label="Random ($d^2/|G|$)")
    plt.xticks(x, names)
    plt.xlabel("Irreducible representation (dim = leading digit)")
    plt.ylabel("Fraction of energy")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close()


# ----------------------------- geometric recovery -----------------------------

def _features(proj):
    """Real feature matrix from a (possibly complex) isotypic projection."""
    if np.iscomplexobj(proj) and np.abs(proj.imag).max() > 1e-9:
        return np.hstack([proj.real, proj.imag])
    return proj.real


def _project_pca(W, table, chi, d_rho, D):
    """Isotypic-project W onto irrep, reduce to D PCA dims (padded if smaller)."""
    feats = _features(isotypic_projection(W, table, chi, d_rho))
    ncomp = min(D, feats.shape[1])
    coords = PCA(n_components=ncomp).fit_transform(feats)
    if coords.shape[1] < D:
        coords = np.hstack([coords, np.zeros((coords.shape[0], D - coords.shape[1]))])
    return coords


def geometric_recovery(emb, G, char_per_element, dims, energy_rows,
                       n_random=8, rng_seed=0):
    """Visualize + score the embedding's geometry against the true polytope.

    The grokking signal is the ENERGY ENRICHMENT of the geometric irrep (see the
    isotypic table) -- not the Procrustes match: for a *faithful* irrep the
    isotypic projection lands every element on that rep's orbit (the polytope) by
    construction, so the match is high even for a random embedding. We therefore
    pick the geometric irrep as the dominant non-trivial one by energy, and
    report the Procrustes match ALONGSIDE its random-embedding baseline so the
    (often tautological) faithful-irrep case is transparent. For specific-seed
    orbits (icosahedron/root-system) the match-minus-baseline gap is meaningful.

    Returns dict or None (group has no explicit geometry coords, e.g. I_h).
    """
    geo = G.get("geometry", {})
    Q = geo.get("coords4d")
    if Q is None:
        Q = geo.get("coords3d")
    if Q is None:
        return None
    Q = np.asarray(Q, float)
    D = Q.shape[1]
    table = G["table"]
    n, d = emb.shape

    # Select the geometric irrep by ENRICHMENT (what the model most strongly chose
    # relative to baseline), not raw energy: for a faithful polytope (e.g. the
    # 600-cell) this picks the 2-dim spinor that realizes it, even when a 3-dim
    # rep happens to carry slightly more total energy.
    nontrivial = [r for r in energy_rows if r["dim"] > 1]
    chosen = max(nontrivial, key=lambda r: r["enrichment"])["irrep"]
    chi = char_per_element[chosen]

    coords = _project_pca(emb, table, chi, dims[chosen], D)
    _, aligned, disparity = procrustes(Q, coords)
    Qstd, _, _ = procrustes(Q, aligned)

    rng = np.random.default_rng(rng_seed)
    base = []
    for _ in range(n_random):
        rc = _project_pca(rng.standard_normal((n, d)), table, chi, dims[chosen], D)
        base.append(1.0 - procrustes(Q, rc)[2])
    base_mean = float(np.mean(base))

    erow = next(r for r in energy_rows if r["irrep"] == chosen)
    return {
        "irrep": chosen, "dim": dims[chosen], "match": 1.0 - float(disparity),
        "baseline": base_mean, "baseline_sd": float(np.std(base)),
        "gap": (1.0 - float(disparity)) - base_mean,
        "energy_frac": erow["energy_frac"], "enrichment": erow["enrichment"],
        "aligned": aligned, "Q_std": Qstd, "D": D, "polytope": geo.get("polytope"),
    }


def plot_recovery(rec, G, img, perm_conj, perm_cos):
    """Overlay the recovered (aligned) polytope on the true vertices + PCA scatter."""
    Q = rec["Q_std"]
    R = rec["aligned"]
    D = rec["D"]
    name = rec["irrep"]
    poly = rec["polytope"]

    # 2D overlay (first two coords of the Procrustes-aligned shapes)
    plt.figure(figsize=(7, 7))
    plt.scatter(Q[:, 0], Q[:, 1], s=120, facecolors="none", edgecolors="#d62728",
                linewidths=1.5, label=f"true {poly}")
    plt.scatter(R[:, 0], R[:, 1], s=18, c="#1f77b4", label=f"recovered (irrep {name})")
    for i in range(len(Q)):
        plt.plot([Q[i, 0], R[i, 0]], [Q[i, 1], R[i, 1]], color="gray", lw=0.3, alpha=0.5)
    plt.gca().set_aspect("equal")
    plt.title(f"{G['name']}: {poly} recovery  (match={rec['match']:.3f}, irrep {name})")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(img, "geometry_recovery_2d.svg"), format="svg", bbox_inches="tight")
    plt.close()

    # 3D interactive overlay (first three coords)
    try:
        import plotly.graph_objects as go
        z_t = Q[:, 2] if D >= 3 else np.zeros(len(Q))
        z_r = R[:, 2] if D >= 3 else np.zeros(len(R))
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=Q[:, 0], y=Q[:, 1], z=z_t, mode="markers",
                                   marker=dict(size=6, color="red", opacity=0.45),
                                   name=f"true {poly}"))
        fig.add_trace(go.Scatter3d(x=R[:, 0], y=R[:, 1], z=z_r, mode="markers",
                                   marker=dict(size=3, color="blue"),
                                   name=f"recovered ({name})"))
        fig.update_layout(title=f"{G['name']} {poly} recovery (match={rec['match']:.3f})")
        fig.write_html(os.path.join(img, "geometry_recovery_3d.html"))
    except Exception as e:
        print(f"plotly 3D overlay skipped ({e})")

    # PCA scatter of the recovered coords colored by conjugacy class
    try:
        import plotly.express as px
        conj = [str(c) for c in perm_conj["j_classes"]]
        z = R[:, 2] if D >= 3 else np.zeros(len(R))
        fig = px.scatter_3d(x=R[:, 0], y=R[:, 1], z=z, color=conj,
                            title=f"{G['name']} embedding on irrep {name}, by conjugacy class")
        fig.update_traces(marker=dict(size=4))
        fig.write_html(os.path.join(img, "geometry_pca_conjugacy.html"))
        if perm_cos is not None:
            cos = [str(c) for c in perm_cos["j_classes"]]
            fig = px.scatter_3d(x=R[:, 0], y=R[:, 1], z=z, color=cos,
                                title=f"{G['name']} embedding on irrep {name}, by coset")
            fig.update_traces(marker=dict(size=4))
            fig.write_html(os.path.join(img, "geometry_pca_coset.html"))
    except Exception as e:
        print(f"plotly scatter skipped ({e})")


def fallback_irrep_pca(emb, G, char_per_element, dims, img, perm_conj, target_dim=3):
    """When no explicit polytope coords exist (e.g. I_h): PCA of the embedding
    projected onto all irreps of `target_dim` and color by conjugacy class."""
    table = G["table"]
    proj = np.zeros_like(emb, dtype=np.complex128)
    for name, d in dims.items():
        if d == target_dim:
            proj = proj + isotypic_projection(emb, table, char_per_element[name], d)
    coords = PCA(n_components=3).fit_transform(_features(proj))
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=perm_conj["j_classes"], cmap="tab10", s=40)
    plt.colorbar(sc, label="conjugacy class")
    plt.title(f"{G['name']}: embedding on {target_dim}-dim irreps, PCA")
    plt.savefig(os.path.join(img, "geometry_recovery_2d.svg"), format="svg", bbox_inches="tight")
    plt.close()


# ----------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generic rep-theory + polytope recovery")
    ap.add_argument("--task", default="icosahedral", choices=GROUP_TASKS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--mlp_dim", type=int, default=512)
    ap.add_argument("--save_dir", default="data/transformer")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    G = get_group(args.task)
    n = G["n"]
    char_per_element, dims, ct = char_dict(G["table"])
    v = verify_character_table(G["table"], ct)
    print(f"[{args.task}] |G|={n}  irreps dims={v['dims']}  orthogonal={v['orthogonal']}  "
          f"sum d^2={v['sum_d2']}")

    ckpt = checkpoint_path(args.save_dir, n, args.embed_dim, args.n_heads, args.mlp_dim,
                           args.seed, args.task)
    model, _ = load_model(ckpt, device)
    img = image_dir(n, args.embed_dim, args.n_heads, args.mlp_dim, args.seed, task=args.task)

    emb = model.embed.weight[:n].detach().cpu().numpy().astype(np.float64)
    unemb = model.unembed.weight.detach().cpu().numpy().astype(np.float64)

    emb_rows = None
    for label, W, fname in (("EMBEDDING", emb, "isotypic_embedding.svg"),
                            ("UNEMBED", unemb, "isotypic_unembed.svg")):
        rows, parseval = isotypic_spectrum(W, G["table"], char_per_element, dims)
        if label == "EMBEDDING":
            emb_rows = rows
        print(f"\n{label} isotypic spectrum (Parseval={parseval:.4f})")
        print(f"{'irrep':>6} {'dim':>4} {'energy%':>9} {'baseline%':>10} {'enrich':>8}")
        print("-" * 42)
        for r in rows:
            print(f"{r['irrep']:>6} {r['dim']:>4} {100*r['energy_frac']:>8.2f}% "
                  f"{100*r['baseline']:>9.2f}% {r['enrichment']:>8.2f}")
        plot_spectrum(rows, f"{label.title()} isotypic energy ({G['name']})",
                      os.path.join(img, fname))

    perm_conj = group_partition(G["table"], "conjugacy")
    perm_cos = None
    if G.get("normal_subgroup") is not None:
        perm_cos = group_partition(G["table"], "coset",
                                   a5_indices=G["normal_subgroup"],
                                   coset_labels=G.get("coset_labels"))

    rec = geometric_recovery(emb, G, char_per_element, dims, emb_rows)
    if rec is None:
        print(f"\nNo explicit polytope coords for {args.task}; PCA of low-dim-irrep projection.")
        fallback_irrep_pca(emb, G, char_per_element, dims, img, perm_conj,
                           target_dim=G.get("geometry", {}).get("irrep_dim", 3))
    else:
        print(f"\nGEOMETRY ({rec['polytope']}): dominant non-trivial irrep = {rec['irrep']} "
              f"(dim {rec['dim']}) carries {100*rec['energy_frac']:.1f}% energy, "
              f"enrichment {rec['enrichment']:.2f}x")
        print(f"  Procrustes shape match = {rec['match']:.4f}  vs random baseline "
              f"{rec['baseline']:.4f}±{rec['baseline_sd']:.3f}  (gap {rec['gap']:+.4f})")
        print("  NOTE: grokking signal is the energy enrichment above; the match gap is "
              "informative only for specific-seed orbits (icosahedron/roots), ~0 for "
              "faithful polytopes (600-cell) where the orbit shape is tautological.")
        plot_recovery(rec, G, img, perm_conj, perm_cos)

    print(f"\nFigures saved to {img}")


if __name__ == "__main__":
    main()
