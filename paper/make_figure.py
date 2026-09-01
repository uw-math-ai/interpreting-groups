"""Figure 1 for the MATH-AI submission, built from tracked results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#1a1a19", "#5f5e56", "#e4e3db"

plt.rcParams.update({
    "font.size": 9, "text.color": INK, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "pdf.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 2.05), facecolor="white")

# --- A: SU(2) vs SO(3) spin spectra, mean +- sd over 5 seeds ---------------
js = [0.0, 0.5, 1.0, 1.5, 2.0]
jlab = ["0", "1/2", "1", "3/2", "2"]


def seed_spectra(variant):
    vals = []
    for s in range(5):
        p = ROOT / f"data/su2/{variant}_N2048_f0.002_s{s}/log.jsonl"
        recs = [json.loads(l) for l in open(p) if "fve_0.5" in l]
        r = [x for x in recs if x["step"] <= 100000][-1]
        vals.append([r.get(f"fve_{j}", 0.0) for j in js])
    v = np.array(vals)
    return v.mean(0), v.std(0)


su_m, su_s = seed_spectra("su2")
so_m, so_s = seed_spectra("so3")
x = np.arange(len(js))
w = 0.38
axA.bar(x - w / 2, su_m, w, yerr=su_s, capsize=2, color=BLUE,
        label=r"SU(2) condition", error_kw=dict(ecolor=INK, lw=0.8),
        edgecolor="white", linewidth=0.5)
axA.bar(x + w / 2, so_m, w, yerr=so_s, capsize=2, color=ORANGE,
        label=r"SO(3) condition", error_kw=dict(ecolor=INK, lw=0.8),
        edgecolor="white", linewidth=0.5)
axA.set_xticks(x, jlab)
axA.set_xlabel(r"spin $j$")
axA.set_ylabel("fraction of encoder variance")
axA.set_ylim(0, 1.05)
axA.legend(frameon=False, loc="upper right", fontsize=8)
axA.set_title("(a) Isotypic spectra, 5 seeds", loc="left", fontsize=9.5)

# --- B: score-level spectral recovery vs j ---------------------------------
series = [
    ("su2hk_t0.3_N2048_f0.01_s0", r"$K_t$, $t=0.3$", BLUE, "-"),
    ("su2hk_t0.1_N2048_f0.01_s0", r"$K_t$, $t=0.1$", AQUA, "-"),
    ("su2hk_t0.05_long_N2048_f0.01_s0", r"$K_t$, $t=0.05$", YELLOW, "-"),
    ("su2bl_j1j2_f0.01_s0", r"$\chi_1+0.5\,\chi_2$, wd $1.0$", ORANGE, "--"),
    ("su2bl_j1j2_wd0.1_f0.01_s0", r"$\chi_1+0.5\,\chi_2$, wd $0.1$",
     ORANGE, "-"),
]
jgrid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for run, lab, color, ls in series:
    d = json.loads((ROOT / f"data/su2/{run}/hk_coefficients.json").read_text())
    last = d["ckpts"][max(d["ckpts"], key=int)]
    ys = []
    for j in jgrid:
        k = str(j)
        if k in last["recovery"]:
            ys.append(last["recovery"][k])
        else:
            ys.append(last["out_of_support_frac"].get(k, 0.0))
    axB.plot(jgrid, ys, ls, marker="o", markersize=3.5, color=color,
             linewidth=1.6, label=lab)
axB.axhline(1.0, color=INK2, linewidth=0.7, linestyle=":")
axB.set_xlabel(r"spin $j$")
axB.set_ylabel("recovered / required weight")
axB.set_ylim(-0.08, 1.95)
axB.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
axB.set_title("(b) Score-level spectral recovery", loc="left", fontsize=9.5)

for ax in (axA, axB):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig(ROOT / "paper/figure1.pdf")
fig.savefig(ROOT / "paper/figure1.png", dpi=220)
print("saved paper/figure1.{pdf,png}")
