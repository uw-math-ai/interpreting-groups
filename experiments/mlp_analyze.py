"""Decode + render the polytope for an MLP-only checkpoint into a separate folder
(images_mlp/) so it never clobbers the transformer figures. Reuses the same
renderers; prints the held-out decode R².
"""
import argparse
import os

import numpy as np
import torch

from core.checkpoint import load_model
from core.groups import get_group
from core import quaternions as q
from experiments.polytope_figures import (render_3d_icosahedron, render_4d,
                                           render_3d_generic, render_2d, _geom_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outroot", default="images_mlp")
    a = ap.parse_args()
    n = get_group(a.task)["n"]
    emb = load_model(a.ckpt, torch.device("cpu"))[0].embed.weight[:n].detach().numpy().astype(float)
    out = os.path.join(a.outroot, f"{a.task}_P{n}_d128_h4_mlp512_s1")
    os.makedirs(out, exist_ok=True)
    lab = f"{a.task} (MLP)"
    if a.task == "A5":
        r2 = render_3d_icosahedron(emb, out, label=lab)
    elif a.task == "icosahedral":
        r2 = render_3d_icosahedron(emb, out, variants=q.ih_icosahedron_coords(), label=lab)
    else:
        D = _geom_dim(a.task)
        if D == 2: r2 = render_2d(a.task, emb, out, label=lab)
        elif D == 3: r2 = render_3d_generic(a.task, emb, out, label=lab)
        else: r2 = render_4d(a.task, emb, out, label=lab)
    print(f"  MLP {a.task}: held-out R²={r2:.3f} -> {out}/polytope.png")


if __name__ == "__main__":
    main()
