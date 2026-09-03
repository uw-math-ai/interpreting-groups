"""Figures for the workshop submission, built from tracked results."""
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
axA.set_title("(a) Isotypic decomposition, 5 seeds", loc="left", fontsize=9.5)

# --- B: calibrated signed Haar coefficients, normalized per seed --------
# Per-seed normalized profiles a_hat_j / (scale * target_j) for the two
# band-limited cells (5 seeds each), plus the hk t=0.3 seed mean. Values
# near 1 mean the subspace is carried at its target coefficient; the
# wd 1.0 spread at j=2 is the seed-dependent suppression reported in the
# text. Out-of-support coefficients are shown as absolute values near 0.
def load_profiles(glob):
    profs = []
    for s0 in range(5):
        p = ROOT / f"data/su2/{glob.format(s=s0)}/hk_haar_ckpt_60000.json"
        d = json.loads(p.read_text())["results"]
        tc = d["target_comparison"]["normalized_profile"]
        profs.append({float(k): v for k, v in tc.items()})
    return profs

cells = [
    ("su2bl_j1j2_wd0.1_f0.01_s{s}", "band-limited, wd 0.1", AQUA),
    ("su2bl_j1j2_f0.01_s{s}", "band-limited, wd 1.0", ORANGE),
    ("su2hk_t0.3_N2048_f0.01_s{s}", r"$K_t$, $t=0.3$", BLUE),
]
rng = np.random.default_rng(0)
for glob, lab, color in cells:
    profs = load_profiles(glob)
    js_cell = sorted(profs[0])
    first = True
    for prof in profs:
        xs = np.array(js_cell) + rng.uniform(-0.03, 0.03, len(js_cell))
        axB.plot(xs, [prof[j] for j in js_cell], "o", color=color,
                 markersize=4, alpha=0.65, label=lab if first else None)
        first = False
axB.axhline(1.0, color=INK2, linewidth=0.7, linestyle=":")
axB.axhline(0.0, color=INK2, linewidth=0.5, linestyle="-", alpha=0.4)
axB.set_xlabel(r"spin $j$")
axB.set_ylabel("coefficient / target")
axB.set_ylim(-0.1, 1.35)
axB.legend(frameon=False, fontsize=7.5, loc="lower left")
axB.set_title("(b) Signed Haar coefficients, 5 seeds", loc="left",
              fontsize=9.5)

for ax in (axA, axB):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig(ROOT / "paper/figure1.pdf")
fig.savefig(ROOT / "paper/figure1.png", dpi=220)
print("saved paper/figure1.{pdf,png}")


# ---------------- appendix figure: dynamics and per-group spectra -------------
def appendix_figure():
    fig = plt.figure(figsize=(9.0, 5.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.34,
                          left=0.07, right=0.98, top=0.93, bottom=0.09)

    spin_colors = {0.5: BLUE, 1.0: ORANGE, 1.5: AQUA, 2.0: YELLOW}
    spin_names = {0.5: "j=1/2", 1.0: "j=1", 1.5: "j=3/2", 2.0: "j=2"}

    # (a)(b) spin content during training
    for col, (variant, tag) in enumerate([("su2", "SU(2)"), ("so3", "SO(3)")]):
        ax = fig.add_subplot(gs[0, col])
        recs = [json.loads(l) for l in
                open(ROOT / f"data/su2/{variant}_N2048_f0.01_s0/log.jsonl")
                if "fve_0.5" in l]
        steps = [r["step"] for r in recs]
        for j in [0.5, 1.0, 1.5, 2.0]:
            ax.plot(steps, [max(r[f"fve_{j}"], 2e-4) for r in recs], "-",
                    color=spin_colors[j], linewidth=1.5, label=spin_names[j])
        ax.set_yscale("log")
        ax.set_ylim(1.5e-4, 2)
        ax.set_xlabel("training step")
        if col == 0:
            ax.set_ylabel("variance fraction (log)")
            ax.legend(frameon=False, fontsize=7, loc="center right")
        ax.set_title(f"({chr(97+col)}) {tag} spin content", loc="left",
                     fontsize=9.5)

    # (c) SU(3)/PSU(3) sector spectra (5-seed means)
    ax = fig.add_subplot(gs[0, 2])
    sect = ["fve_10", "fve_11", "fve_20", "fve_30"]
    lab = [r"$3\oplus\bar 3$", r"$8$", r"$6\oplus\bar 6$", r"$10\oplus\bar{10}$"]
    means = {}
    for v in ("su3", "psu3"):
        vals = []
        for s0 in range(5):
            recs = [json.loads(l) for l in
                    open(ROOT / f"data/su3/{v}_N2048_f0.01_s{s0}/log.jsonl")
                    if "fve_10" in l]
            vals.append([recs[-1][k] for k in sect])
        means[v] = np.array(vals).mean(0)
    x = np.arange(4)
    ax.bar(x - 0.19, means["su3"], 0.38, color=BLUE, label="SU(3)")
    ax.bar(x + 0.19, means["psu3"], 0.38, color=ORANGE, label="PSU(3)")
    ax.set_xticks(x, lab, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(c) SU(3) vs PSU(3) components", loc="left", fontsize=9.5)

    # (d) Spin(4) conditions: carrier vs rejected alternative
    ax = fig.add_subplot(gs[1, 0:2])
    conds = ["spin4", "so4", "leftq", "pso4"]
    cond_lab = ["Spin(4)", "SO(4)", r"SO(3)$\times$SU(2)", "PSO(4)"]
    KEY = {"spin4": ["fve_0.5_0", "fve_0_0.5"], "so4": ["fve_0.5_0.5"],
           "leftq": ["fve_0_0.5", "fve_1_0"], "pso4": ["fve_1_0", "fve_0_1"]}
    ALT = {"spin4": None, "so4": None, "leftq": "fve_1_0.5", "pso4": "fve_1_1"}
    car, alt = [], []
    for c in conds:
        vals, avals = [], []
        for s0 in range(5):
            recs = [json.loads(l) for l in
                    open(ROOT / f"data/spin4/{c}_N2048_f0.01_s{s0}/log.jsonl")
                    if "fve_0.5_0" in l]
            r = recs[-1]
            vals.append(sum(r[k] for k in KEY[c]))
            avals.append(r[ALT[c]] if ALT[c] else 0.0)
        car.append(np.mean(vals))
        alt.append(np.mean(avals))
    x = np.arange(4)
    ax.bar(x - 0.19, car, 0.38, color=BLUE, label="learned representation")
    ax.bar(x + 0.19, alt, 0.38, color=ORANGE,
           label="rejected faithful irreducible")
    ax.set_xticks(x, cond_lab, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(d) Spin(4) family: learned vs rejected",
                 loc="left", fontsize=9.5)

    # (e) Spin(5)/SO(5)
    ax = fig.add_subplot(gs[1, 2])
    sect5 = ["fve_4", "fve_5", "fve_10", "fve_14"]
    lab5 = [r"$\mathbf{4}$", r"$\mathbf{5}$", r"$\mathbf{10}$", r"$\mathbf{14}$"]
    m5 = {}
    for v in ("spin5", "so5"):
        vals = []
        for s0 in range(5):
            recs = [json.loads(l) for l in
                    open(ROOT / f"data/spin5/{v}_N2048_f0.01_s{s0}/log.jsonl")
                    if "fve_4" in l]
            vals.append([recs[-1][k] for k in sect5])
        m5[v] = np.array(vals).mean(0)
    x = np.arange(4)
    ax.bar(x - 0.19, m5["spin5"], 0.38, color=BLUE, label="Spin(5)")
    ax.bar(x + 0.19, m5["so5"], 0.38, color=ORANGE, label="SO(5)")
    ax.set_xticks(x, lab5, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(e) Spin(5) vs SO(5) components", loc="left", fontsize=9.5)

    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(ROOT / "paper/figureA1.pdf")
    fig.savefig(ROOT / "paper/figureA1.png", dpi=200)
    print("saved paper/figureA1.{pdf,png}")


appendix_figure()
