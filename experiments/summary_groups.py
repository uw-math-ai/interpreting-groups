"""Cross-group synthesis: how grokked transformers concentrate embedding energy
into the lowest-dimensional faithful/geometric irrep of each finite group.

Loads each grokked checkpoint, computes the embedding isotypic spectrum via the
generic character table, and emits a comparison bar chart + a summary table.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from core.checkpoint import load_model
from core.groups import get_group
from core.reptheory import char_dict
from experiments.group_rep_analysis import isotypic_spectrum

# task -> (checkpoint path, human geometry label)
RUNS = [
    ("icosahedral", "data/transformer/clean_tf0.6_s1/icosahedral_P120_d128_h4_mlp512_s1.pt", "icosahedron (3-dim)"),
    ("2I", "data/transformer/grok_2I_s1/2I_P120_d128_h4_mlp512_s1.pt", "600-cell (2-dim spinor)"),
    ("S5", "data/transformer/grok_S5_s1/S5_P120_d128_h4_mlp512_s1.pt", "A4 roots (4-dim standard)"),
    ("A5", "data/transformer/grok_A5_s1/A5_P60_d128_h4_mlp512_s1.pt", "icosahedron (3-dim)"),
    ("2O", "data/transformer/grok_2O_s1/2O_P48_d128_h4_mlp512_s1.pt", "24-cell (2-dim spinor)"),
    ("2T", "data/transformer/grok_2T_s1/2T_P24_d128_h4_mlp512_s1.pt", "24-cell (2-dim spinor)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="images/_summary")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cpu")

    summary = []
    for task, ckpt, geom in RUNS:
        if not os.path.exists(ckpt):
            print(f"[skip] {task}: no checkpoint ({ckpt})")
            continue
        G = get_group(task)
        cpe, dims, ct = char_dict(G["table"])
        model, meta = load_model(ckpt, dev)
        emb = model.embed.weight[:G["n"]].detach().numpy().astype(np.float64)
        rows, parseval = isotypic_spectrum(emb, G["table"], cpe, dims)
        nt = [r for r in rows if r["dim"] > 1]
        top = max(nt, key=lambda r: r["enrichment"])
        eval_acc = meta.get("eval_accs", [float("nan")])[-1] if meta.get("eval_accs") else float("nan")
        summary.append({"task": task, "n": G["n"], "geom": geom,
                        "irrep_dim": top["dim"], "energy": top["energy_frac"],
                        "enrich": top["enrichment"], "eval": eval_acc, "rows": rows})

    # table
    print(f"\n{'group':>11} {'|G|':>4} {'eval':>5} {'geom irrep':>10} "
          f"{'energy%':>8} {'enrich':>7}   geometry")
    print("-" * 78)
    for s in summary:
        print(f"{s['task']:>11} {s['n']:>4} {s['eval']:>5.2f} "
              f"{s['irrep_dim']:>9}d {100*s['energy']:>7.1f}% {s['enrich']:>6.1f}x   {s['geom']}")

    # comparison bar chart: dominant geometric-irrep energy vs random baseline
    labels = [s["task"] for s in summary]
    energy = [100 * s["energy"] for s in summary]
    base = [100 * (s["irrep_dim"] ** 2) / s["n"] for s in summary]
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5.5))
    plt.bar(x - 0.2, energy, 0.4, color="#1f77b4", label="grokked model")
    plt.bar(x + 0.2, base, 0.4, color="#d62728", alpha=0.7, label="random baseline $d^2/|G|$")
    for i, s in enumerate(summary):
        plt.text(i - 0.2, energy[i] + 1, f"{s['enrich']:.1f}x", ha="center", fontsize=9)
    plt.xticks(x, [f"{s['task']}\n({s['geom'].split('(')[0].strip()})" for s in summary], fontsize=8)
    plt.ylabel("% embedding energy in dominant geometric irrep")
    plt.title("Grokked transformers concentrate into the lowest-dim faithful irrep")
    plt.legend(frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    out = os.path.join(args.out, "energy_concentration_by_group.svg")
    plt.savefig(out, format="svg", bbox_inches="tight")
    print(f"\nSaved comparison figure -> {out}")


if __name__ == "__main__":
    main()
