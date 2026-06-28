"""Attention and embedding analysis for trained models."""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from core.data import generate_mod_mult_dataset, generate_group_dataset, get_device
from core.checkpoint import load_model, checkpoint_path, image_dir
from core.algebra import jclass_permutation, algebraic_jclass_order
from core.groups import analysis_partition


def plot_attention(model, P, device, save_path=None):
    """Plot raw attention heatmaps from '=' -> 'a' token, per head."""
    inputs, _ = generate_mod_mult_dataset(P)

    model.eval()
    with torch.no_grad():
        h = model.embed(inputs.to(device))
        _, attn_weights = model.attn(
            h, h, h,
            attn_mask=model.causal_mask,
            need_weights=True,
            average_attn_weights=False,
        )

    n_heads = model.n_heads
    QUERY_POS, KEY_POS = 2, 0

    fig, axes = plt.subplots(1, n_heads, figsize=(6 * n_heads, 5))
    if n_heads == 1:
        axes = [axes]

    for head_idx in range(n_heads):
        attn_to_a = attn_weights[:, head_idx, QUERY_POS, KEY_POS].view(P, P).cpu().numpy()
        ax = axes[head_idx]
        im = ax.imshow(attn_to_a, origin="lower", cmap="Blues", aspect="auto", vmin=0)
        ax.set(title=f"Head {head_idx}", xlabel="Input b", ylabel="Input a")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Attention: '=' -> 'a' (raw order)", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    # plt.show()  # uncomment for interactive display


def plot_attention_jclass(model, P, device, perm, save_path=None):
    """Plot structure-permuted attention heatmaps (J-class or conjugacy class)."""
    inputs, _ = generate_mod_mult_dataset(P)  # inputs [a,b,P] layout is task-agnostic

    model.eval()
    with torch.no_grad():
        h = model.embed(inputs.to(device))
        _, attn_weights = model.attn(
            h, h, h,
            attn_mask=model.causal_mask,
            need_weights=True,
            average_attn_weights=False,
        )

    n_heads = model.n_heads
    QUERY_POS, KEY_POS = 2, 0
    idx = perm["sorted_indices"]
    boundaries = perm["boundaries"]

    fig, axes = plt.subplots(1, n_heads, figsize=(7 * n_heads, 6))
    if n_heads == 1:
        axes = [axes]

    for head_idx in range(n_heads):
        attn_to_a = attn_weights[:, head_idx, QUERY_POS, KEY_POS].view(P, P).cpu().numpy()
        attn_rearranged = attn_to_a[np.ix_(idx, idx)]

        ax = axes[head_idx]
        im = ax.imshow(attn_rearranged, origin="lower", cmap="Blues", aspect="auto", vmin=0)

        for b in boundaries:
            ax.axhline(b - 0.5, color="red", linewidth=3, linestyle="--", alpha=0.7)
            ax.axvline(b - 0.5, color="red", linewidth=3, linestyle="--", alpha=0.7)

        ax.set_xticks(perm["tick_positions"])
        ax.set_xticklabels(perm["tick_labels"], rotation=45)
        ax.set_yticks(perm["tick_positions"])
        ax.set_yticklabels(perm["tick_labels"])
        ax.set(title=f"Head {head_idx}", xlabel="Input b (J-class)", ylabel="Input a (J-class)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Attention permuted by structural class (P={P})", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    # plt.show()  # uncomment for interactive display


def plot_embedding(model, P, perm, save_path=None):
    """Plot raw and structure-sorted embedding heatmaps."""
    emb = model.embed.weight.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    ax1.imshow(emb, aspect="auto")
    ax1.set(title="Embedding (raw order)", xlabel="Dimension", ylabel="Token Index")

    emb_sorted = emb[perm["sorted_indices"]]
    im = ax2.imshow(emb_sorted, aspect="auto", cmap="viridis")
    for b in perm["boundaries"]:
        ax2.axhline(b - 0.5, color="red", linewidth=2, linestyle="--", alpha=0.8)
    ax2.set_yticks(perm["tick_positions"])
    ax2.set_yticklabels(perm["tick_labels"])
    ax2.set(title=f"Embedding sorted by structural class (P={P})", xlabel="Dimension", ylabel="Class")
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    # plt.show()  # uncomment for interactive display


def plot_pca(model, P, colorings, save_path_2d=None, save_path_3d=None, img_dir=None):
    """Plot PCA projections of token embeddings (excluding the '=' token).

    Args:
        colorings: list of (name, label_array) where label_array has length P and
            assigns each token a categorical label. The first coloring drives the
            2D plot; every coloring gets its own interactive 3D html (when plotly
            is available). For modmult pass a single ("Token index", arange(P)).
        img_dir: directory for the per-coloring 3D html files (falls back to
            save_path_3d for the first coloring).
    """
    emb = model.embed.weight[:P].detach().cpu().numpy()

    pca = PCA(n_components=3)
    proj = pca.fit_transform(emb)
    evr = pca.explained_variance_ratio_

    # 2D PCA colored by the first (primary) coloring.
    primary_name, primary_labels = colorings[0]
    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=np.asarray(primary_labels),
                          cmap="hsv", alpha=0.8)
    plt.colorbar(scatter, label=primary_name)
    plt.title(f"2D PCA of Embeddings (P={P})  PC1+PC2 FVE={evr[0]+evr[1]:.2f}")
    plt.xlabel(f"PC1 ({evr[0]:.2f})")
    plt.ylabel(f"PC2 ({evr[1]:.2f})")
    plt.grid(True, alpha=0.3)
    if save_path_2d:
        plt.savefig(save_path_2d, format="svg")
    # plt.show()  # uncomment for interactive display

    # One interactive 3D scatter per coloring.
    try:
        import plotly.express as px
        for name, labels in colorings:
            labels = [str(v) for v in labels]
            fig3d = px.scatter_3d(
                x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
                color=labels,
                title=f"3D PCA of Embeddings by {name} (P={P})",
                labels={"color": name},
            )
            fig3d.update_traces(marker=dict(size=3))
            if img_dir is not None:
                slug = name.lower().replace(" ", "_")
                fig3d.write_html(os.path.join(img_dir, f"pca_3d_{slug}.html"))
            elif save_path_3d:
                fig3d.write_html(save_path_3d)
    except Exception as e:
        print(f"plotly/pandas not available, skipping 3D plot ({e})")


def main():
    parser = argparse.ArgumentParser(description="Attention & embedding analysis")
    parser.add_argument("--task", type=str, default="modmult",
                        choices=["modmult", "icosahedral", "2I", "2O", "2T", "A5", "S5"])
    parser.add_argument("--P", type=int, default=None, help="Modulus (modmult); group order is inferred for groups")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="data/transformer")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()

    # Structural partition (J-class for modmult, conjugacy class for groups).
    perm, G = analysis_partition(args.task, args.P, mode="conjugacy")
    P = G["n"] if G is not None else args.P

    ckpt = checkpoint_path(args.save_dir, P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed, args.task)
    model, _ = load_model(ckpt, device)
    img = image_dir(P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed, task=args.task)

    # PCA colorings: groups get conjugacy class (+ coset when an index-2 normal
    # subgroup exists); modmult keeps the raw token-index coloring.
    if args.task == "modmult":
        colorings = [("Token index", np.arange(P))]
    else:
        colorings = [("conjugacy class", perm["j_classes"])]
        perm_coset, _ = analysis_partition(args.task, P, mode="coset")
        if perm_coset is not None:
            colorings.append(("coset", perm_coset["j_classes"]))

    print(f"Generating plots for task={args.task}, P={P}, seed={args.seed}...")
    plot_attention(model, P, device, os.path.join(img, "attention_raw.svg"))
    plot_attention_jclass(model, P, device, perm, os.path.join(img, "attention_jclass.svg"))
    plot_embedding(model, P, perm, os.path.join(img, "embedding_jclass.svg"))
    plot_pca(model, P, colorings, os.path.join(img, "pca_2d.svg"), img_dir=img)

    print(f"Plots saved to {img}")


if __name__ == "__main__":
    main()
