"""PCA components for 95% FVE: J-class vs random baseline (Figure 2)."""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from core.data import get_device
from core.checkpoint import load_model, checkpoint_path, image_dir
from core.algebra import algebraic_jclass_order
from core.groups import analysis_partition


def num_components_for_fve(X, threshold=0.95):
    """Return number of PCA components needed to reach threshold FVE."""
    if len(X) <= 1:
        return 0
    pca = PCA()
    pca.fit(X)
    cum_fve = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cum_fve >= threshold) + 1)


def compute_pca_fve(model, P, threshold=0.95, n_random=50, seed=0, task="modmult"):
    """For each structural class and same-size random subsets, compute PCA components for threshold FVE.

    Classes are J-classes (modmult) or conjugacy classes (groups).

    Returns:
        list of dicts with keys: j, size, actual_dim, random_mean, random_std
    """
    perm, _ = analysis_partition(task, P, mode="conjugacy")
    emb = model.embed.weight[:P].detach().cpu().numpy()
    emb_sorted = emb[perm["sorted_indices"]]

    rng = np.random.default_rng(seed=seed)
    results = []
    start = 0

    for j, count in zip(perm["unique_j"], perm["class_counts"]):
        j = int(j)
        count = int(count)
        block = emb_sorted[start:start + count]
        start += count

        actual_dim = num_components_for_fve(block, threshold)

        random_dims = []
        for _ in range(n_random):
            idx = rng.choice(len(emb), size=count, replace=False)
            random_dims.append(num_components_for_fve(emb[idx], threshold))
        random_dims = np.array(random_dims)

        results.append({
            "j": j,
            "size": count,
            "actual_dim": actual_dim,
            "random_mean": random_dims.mean(),
            "random_std": random_dims.std(),
        })

    return results


def plot_pca_fve(results, P, save_path=None):
    """Plot PCA components needed for 95% FVE: J-class vs random baseline."""
    j_labels = [rf"$J_{{{r['j']}}}$" for r in results]
    actual = [r["actual_dim"] for r in results]
    rand_mean = [r["random_mean"] for r in results]
    rand_std = [r["random_std"] for r in results]

    x = np.arange(len(j_labels))

    plt.figure(figsize=(9, 5))

    plt.plot(x, actual, marker="o", markersize=6, linewidth=2.5,
             color="#1f77b4", label=r"Actual $\mathcal{J}$-class")
    plt.plot(x, rand_mean, marker="s", markersize=5.5, linewidth=2.5,
             linestyle="--", color="#d62728", label="Random subset baseline")

    rand_mean_arr = np.array(rand_mean)
    rand_std_arr = np.array(rand_std)
    plt.fill_between(x, rand_mean_arr - rand_std_arr, rand_mean_arr + rand_std_arr,
                     color="#d62728", alpha=0.15, label=r"Random baseline $\pm$1 std")

    plt.xticks(x, j_labels)
    plt.xlabel(r"$\mathcal{J}$-class")
    plt.ylabel("Components needed for 95% FVE")
    plt.title(r"Local PCA Dimension: $\mathcal{J}$-classes vs. Random Subsets")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")

    print("\nPCA FVE Report")
    print(f"{'J-class':>10} {'Size':>6} {'Actual':>8} {'Random mean':>12} {'Random std':>12}")
    print("-" * 52)
    for r in results:
        print(f"  J_{r['j']:<6} {r['size']:>6} {r['actual_dim']:>8} "
              f"{r['random_mean']:>12.2f} {r['random_std']:>12.2f}")


def main():
    parser = argparse.ArgumentParser(description="PCA FVE: structural class vs random baseline (Figure 2)")
    parser.add_argument("--task", type=str, default="modmult",
                        choices=["modmult", "icosahedral", "2I", "2O", "2T", "A5", "S5"])
    parser.add_argument("--P", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--n_random", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="data/transformer")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import torch
    device = torch.device(args.device) if args.device else get_device()

    from core.groups import get_group
    P = args.P if args.task == "modmult" else get_group(args.task)["n"]

    ckpt = checkpoint_path(args.save_dir, P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed, args.task)
    model, _ = load_model(ckpt, device)

    print(f"Computing PCA FVE for task={args.task}, P={P}, seed={args.seed}...")
    results = compute_pca_fve(model, P, args.threshold, args.n_random, task=args.task)

    img = image_dir(P, args.embed_dim, args.n_heads, args.mlp_dim, args.seed, task=args.task)
    plot_pca_fve(results, P, os.path.join(img, "pca_fve.svg"))

    print(f"Plot saved to {img}")


if __name__ == "__main__":
    main()
