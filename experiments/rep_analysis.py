"""Representation-theory analysis for the icosahedral group I_h = A5 x C2.

The non-abelian analog of the cyclic FFT / character-fitting scripts. Everything
is computed from the Cayley table plus the (real) character table of I_h, using
the projection onto isotypic components of the regular representation:

    (E_rho f)(x) = (d_rho / |G|) * sum_g  chi_rho(g) * f(x . g)

For each embedding/unembed dimension i, f_i(g) = W[g, i] is a function on G.
E_rho extracts its component in the rho-isotypic subspace. By Parseval the energy
splits exactly across irreps; a grokked model concentrates energy in a few
irreps (cf. Chughtai-Liu-Nanda, *A Toy Model of Universality*). The 3-dim irrep
is the icosahedral rotation representation, so a PCA of the embedding projected
onto it recovers icosahedral geometry.

Outputs (to the task image dir):
    isotypic_embedding.svg / isotypic_unembed.svg  — energy per irrep vs baseline
    icosahedron_pca_3d.html, icosahedron_pca_2d.svg — 3-dim-irrep projection
    prints an enrichment table and orthogonality / Parseval sanity checks.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from core.data import get_device
from core.checkpoint import load_model, checkpoint_path, image_dir
from core.groups import icosahedral_group, conjugacy_classes, group_partition


PHI_P = (1.0 + np.sqrt(5.0)) / 2.0     # golden ratio
PHI_M = (1.0 - np.sqrt(5.0)) / 2.0

# A5 character table keyed by class type. Order of columns: 1A, 2A, 3A, 5A, 5B.
A5_IRREPS = {
    "1":  (1, {"1A": 1, "2A": 1, "3A": 1, "5A": 1, "5B": 1}),
    "3a": (3, {"1A": 3, "2A": -1, "3A": 0, "5A": PHI_P, "5B": PHI_M}),
    "3b": (3, {"1A": 3, "2A": -1, "3A": 0, "5A": PHI_M, "5B": PHI_P}),
    "4":  (4, {"1A": 4, "2A": 0, "3A": 1, "5A": -1, "5B": -1}),
    "5":  (5, {"1A": 5, "2A": 1, "3A": -1, "5A": 0, "5B": 0}),
}


def element_order(g, table, identity):
    """Multiplicative order of element index g under the Cayley table."""
    x, k = g, 1
    while x != identity:
        x = int(table[x, g])
        k += 1
    return k


def a5_class_types(G):
    """Map each A5 element index (0..59) to its class type label (1A/2A/3A/5A/5B).

    5A vs 5B (the two order-5 classes) are distinguished by smallest min-index.
    """
    table = G["table"]
    a5 = G["a5_indices"]
    sub = table[np.ix_(a5, a5)]              # 60x60 A5 sub-table (closed)
    ident = 0
    classes = conjugacy_classes(sub)         # A5 conjugacy classes
    order5 = [c for c in classes if element_order(c[0], sub, ident) == 5]
    order5_sorted = sorted(order5, key=lambda c: c[0])
    label5 = {id(order5_sorted[0]): "5A", id(order5_sorted[1]): "5B"}

    types = {}
    for c in classes:
        o = element_order(c[0], sub, ident)
        if o == 1:
            t = "1A"
        elif o == 2:
            t = "2A"
        elif o == 3:
            t = "3A"
        else:
            t = label5[id(c)]
        for g in c:
            types[g] = t
    return types


def ih_character_table(G):
    """Build the I_h = A5 x C2 character table over the 10 conjugacy classes.

    Returns:
        classes: list of conjugacy-class index-lists (from group_partition order)
        char_per_element: dict irrep_name -> np.array length n of chi on each element
        dims: dict irrep_name -> dimension
    """
    table = G["table"]
    n = G["n"]
    n5 = n // 2
    a5_types = a5_class_types(G)

    char_per_element = {}
    dims = {}
    for a5_name, (d, vals) in A5_IRREPS.items():
        for parity_name, parity_sign in (("g", +1), ("u", -1)):
            name = f"{a5_name}{parity_name}"
            chi = np.empty(n, dtype=np.float64)
            for x in range(n):
                a5_part = x % n5
                sign_bit = x // n5            # 0 rotation, 1 rotoreflection
                base = vals[a5_types[a5_part]]
                eps = 1.0 if (parity_sign == 1 or sign_bit == 0) else -1.0
                chi[x] = base * eps
            char_per_element[name] = chi
            dims[name] = d

    classes = conjugacy_classes(table)
    return classes, char_per_element, dims


def check_orthogonality(char_per_element, dims, n):
    """Row orthogonality: <chi_p, chi_q> = n * delta_pq (real characters)."""
    names = list(char_per_element)
    ok = True
    for p in names:
        for q in names:
            ip = float(np.dot(char_per_element[p], char_per_element[q]))
            expect = n if p == q else 0.0
            if abs(ip - expect) > 1e-6:
                ok = False
    total_dim_sq = sum(d * d for d in dims.values())
    return ok, total_dim_sq


def isotypic_projection(W, table, chi, d_rho):
    """Project the columns of W (a function G -> R^d) onto the rho-isotypic part.

    (E_rho W)[x, i] = (d_rho / n) * sum_g chi[g] * W[table[x, g], i]
    """
    n = W.shape[0]
    gathered = W[table]                       # [x, g, d]
    proj = np.einsum("g,xgd->xd", chi, gathered) * (d_rho / n)
    return proj


def isotypic_spectrum(W, table, char_per_element, dims):
    """Energy of W in each irrep, with the analytic random baseline.

    Baseline: a random function on G has expected isotypic energy proportional to
    d_rho^2 (the multiplicity-weighted dimension in the regular rep), so the
    baseline energy fraction is d_rho^2 / |G|. Enrichment = actual / baseline.
    """
    n = W.shape[0]
    total = float((W ** 2).sum())
    rows = []
    energy_sum = 0.0
    for name, chi in char_per_element.items():
        d = dims[name]
        proj = isotypic_projection(W, table, chi, d)
        energy = float((proj ** 2).sum())
        energy_sum += energy
        frac = energy / total if total > 0 else 0.0
        baseline = (d * d) / n
        rows.append({
            "irrep": name, "dim": d, "energy_frac": frac,
            "baseline": baseline, "enrichment": frac / baseline if baseline else 0.0,
        })
    rows.sort(key=lambda r: r["energy_frac"], reverse=True)
    parseval = energy_sum / total if total > 0 else 0.0   # should be ~1.0
    return rows, parseval


def plot_spectrum(rows, title, save_path=None):
    names = [r["irrep"] for r in rows]
    frac = [r["energy_frac"] for r in rows]
    base = [r["baseline"] for r in rows]
    x = np.arange(len(names))

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, frac, width=0.4, color="#1f77b4", label="Model")
    plt.bar(x + 0.2, base, width=0.4, color="#d62728", alpha=0.7, label="Random baseline ($d^2/|G|$)")
    plt.xticks(x, names, rotation=0)
    plt.xlabel("Irreducible representation of $I_h$ (dim in subscript via label)")
    plt.ylabel("Fraction of embedding energy")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")


def plot_icosahedron(model, G, char_per_element, dims, img):
    """PCA of the embedding projected onto the 3-dim irreps -> icosahedral orbit."""
    table = G["table"]
    n = G["n"]
    emb = model.embed.weight[:n].detach().cpu().numpy()

    # Combine both 3-dim irreps (the conjugate pair 3a, 3b span the rotation rep).
    proj = np.zeros_like(emb)
    for name in ("3ag", "3bg", "3au", "3bu"):
        proj = proj + isotypic_projection(emb, table, char_per_element[name], dims[name])

    pca = PCA(n_components=3)
    coords = pca.fit_transform(proj)
    evr = pca.explained_variance_ratio_

    perm_conj = group_partition(table, "conjugacy", G["a5_indices"])
    perm_cos = group_partition(table, "coset", G["a5_indices"])
    conj_labels = [str(c) for c in perm_conj["j_classes"]]
    coset_labels = ["rotation" if s == 0 else "rotoreflection"
                    for s in perm_cos["j_classes"]]

    # 2D static
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1],
                     c=perm_conj["j_classes"], cmap="tab10", s=40, alpha=0.9)
    plt.colorbar(sc, label="conjugacy class")
    plt.title(f"Embedding projected onto 3-dim irrep, PCA (FVE3={evr.sum():.2f})")
    plt.xlabel(f"PC1 ({evr[0]:.2f})")
    plt.ylabel(f"PC2 ({evr[1]:.2f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(img, "icosahedron_pca_2d.svg"), format="svg", bbox_inches="tight")

    # 3D interactive
    try:
        import plotly.express as px
        for tag, labels in (("conjugacy", conj_labels), ("coset", coset_labels)):
            fig3d = px.scatter_3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                color=labels,
                title=f"3-dim-irrep projection of I_h embedding ({tag})",
                labels={"color": tag},
            )
            fig3d.update_traces(marker=dict(size=4))
            fig3d.write_html(os.path.join(img, f"icosahedron_pca_3d_{tag}.html"))
    except Exception as e:
        print(f"plotly unavailable, skipping 3D icosahedron plot ({e})")

    return evr


def main():
    parser = argparse.ArgumentParser(description="Representation-theory analysis for I_h = A5 x C2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="data/transformer")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import torch
    device = torch.device(args.device) if args.device else get_device()

    G = icosahedral_group()
    n = G["n"]
    _, char_per_element, dims = ih_character_table(G)

    ok, total_dim_sq = check_orthogonality(char_per_element, dims, n)
    print(f"Character table: {len(dims)} irreps, dims {sorted(dims.values())}, "
          f"sum d^2 = {total_dim_sq} (expect {n}); orthogonal: {ok}")

    ckpt = checkpoint_path(args.save_dir, n, args.embed_dim, args.n_heads, args.mlp_dim, args.seed, "icosahedral")
    model, _ = load_model(ckpt, device)
    img = image_dir(n, args.embed_dim, args.n_heads, args.mlp_dim, args.seed, task="icosahedral")

    emb = model.embed.weight[:n].detach().cpu().numpy()
    unemb = model.unembed.weight.detach().cpu().numpy()   # [n, d]

    for label, W, fname in (("EMBEDDING", emb, "isotypic_embedding.svg"),
                            ("UNEMBED", unemb, "isotypic_unembed.svg")):
        rows, parseval = isotypic_spectrum(W, G["table"], char_per_element, dims)
        print(f"\n{label} isotypic spectrum (Parseval={parseval:.4f}, expect ~1.0)")
        print(f"{'irrep':>6} {'dim':>4} {'energy%':>9} {'baseline%':>10} {'enrich':>8}")
        print("-" * 42)
        for r in rows:
            print(f"{r['irrep']:>6} {r['dim']:>4} {100*r['energy_frac']:>8.2f}% "
                  f"{100*r['baseline']:>9.2f}% {r['enrichment']:>8.2f}")
        plot_spectrum(rows, f"{label.title()} isotypic energy ($I_h = A_5\\times C_2$)",
                      os.path.join(img, fname))

    evr = plot_icosahedron(model, G, char_per_element, dims, img)
    print(f"\n3-dim-irrep projection PCA: variance in top 3 PCs = {evr.sum():.3f} {tuple(np.round(evr,3))}")
    print(f"\nFigures saved to {img}")


if __name__ == "__main__":
    main()
