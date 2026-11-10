"""Camera-ready figures from raw logs and uniform-checkpoint reruns.

The transferred evidence is read only. All outputs go to camera_ready.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / 'analysis/selected_checkpoints'
OUTPUT = ROOT / 'figures'
OUTPUT.mkdir(exist_ok=True)
VER = ROOT / 'analysis'
STATS = json.loads((VER / 'verified_summary.json').read_text())
DIAGNOSTICS = json.loads((VER / 'verified_diagnostics_100k.json').read_text())
BLUE, ORANGE, AQUA, YELLOW = "#2C6E7F", "#C77D58", "#7A9E7E", "#C4A04A"
INK, INK2, GRID = "#24313A", "#62707B", "#E5E9EC"

plt.rcParams.update({
    "font.size": 9, "text.color": INK, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.grid.axis": "y", "axes.titlepad": 9,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})

figA_, axA = plt.subplots(figsize=(3.6, 2.05), facecolor="white")
figB_, axB = plt.subplots(figsize=(4.2, 2.85), facecolor="white")

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
axA.set_title("Isotypic decomposition, 5 seeds", loc="left", fontsize=9.5)

# --- B: calibrated signed Haar coefficients, normalized per seed --------
# Restore the actual training-target standard deviation before comparing
# signed coefficients with the target. No per-seed median rescaling.
def load_profiles(glob):
    profs = []
    calibrated = {r['run']:r for r in json.loads(
        (VER / 'spectral_target_calibration.json').read_text())['records']}
    for s0 in range(5):
        name = glob.format(s=s0)
        tc = calibrated[name]['coefficient_recovery']
        profs.append({float(k): v['ratio'] for k, v in tc.items()})
    return profs

cells = [
    ("su2bl_j1j2_wd0.1_f0.01_s{s}", "band-limited, wd 0.1", AQUA),
    ("su2bl_j1j2_f0.01_s{s}", "band-limited, wd 1.0", ORANGE),
    ("su2hk_t0.3_N2048_f0.01_s{s}", r"$K_t$, $t=0.3$", BLUE),
]
rng = np.random.default_rng(0)
profile_values = []
for (glob, lab, color), marker in zip(cells, ["s", "^", "o"]):
    profs = load_profiles(glob)
    js_cell = sorted(profs[0])
    first = True
    for prof in profs:
        profile_values.extend(prof.values())
        xs = np.array(js_cell) + rng.uniform(-0.03, 0.03, len(js_cell))
        axB.plot(xs, [prof[j] for j in js_cell], marker, color=color,
                 markersize=4, alpha=0.65, label=lab if first else None)
        first = False
axB.axhline(1.0, color=INK2, linewidth=0.7, linestyle=":")
axB.axhline(0.0, color=INK2, linewidth=0.5, linestyle="-", alpha=0.4)
axB.set_xlabel(r"spin $j$")
axB.set_ylabel("coefficient / target")
assert len(profile_values) == 50 and np.all(np.isfinite(profile_values))
profile_pad = 0.07 * (max(profile_values) - min(profile_values))
axB.set_ylim(min(profile_values) - profile_pad, max(profile_values) + profile_pad)
axB.legend(frameon=False, fontsize=7.3, loc="lower left",
           bbox_to_anchor=(0, 1.03), borderaxespad=0, labelspacing=0.25)
figB_.suptitle("Signed Haar coefficients, 5 seeds", x=0.15, ha="left", y=0.99, fontsize=9.5)

for ax in (axA, axB):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

figA_.tight_layout(pad=0.4)
figA_.savefig(OUTPUT / "figure1.pdf")
figA_.savefig(OUTPUT / "figure1.png", dpi=220)
figB_.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.69)
figB_.savefig(OUTPUT / "figure_spectral.pdf")
figB_.savefig(OUTPUT / "figure_spectral.png", dpi=220)
print("saved paper/figure1 and paper/figure_spectral")


# --- figure 2: interventions (a) and activation patching (b) ---------------
fig2, (axC, axD) = plt.subplots(1, 2, figsize=(7.2, 2.2), facecolor="white",
                                gridspec_kw={"width_ratios": [1.55, 1]})
labs = ["none", "remove\nspin 1/2", "remove\nspin 1", "center\naverage",
        "random\nsubspace"]
iv_keys = ['none', 'fablate_spin_half', 'fablate_spin_one',
           'even_projection', 'ablate_random_selected_rank']
def intervention_values(variant, field):
    ds = DIAGNOSTICS['diagnostics'][variant]['interventions']
    return [ds[k]['task_acc'][field] for k in iv_keys]
su2, su2e = intervention_values('su2', 'mean'), intervention_values('su2', 'sd')
so3, so3e = intervention_values('so3', 'mean'), intervention_values('so3', 'sd')
x = np.arange(len(labs))
w = 0.38
axC.bar(x - w / 2, su2, w, yerr=su2e, color=BLUE, label="SU(2) model",
        error_kw=dict(ecolor=INK, lw=0.8), edgecolor="white", linewidth=0.5)
axC.bar(x + w / 2, so3, w, yerr=so3e, color=ORANGE, label="SO(3) model",
        error_kw=dict(ecolor=INK, lw=0.8), edgecolor="white", linewidth=0.5)
axC.axhline(1 / 64, color=INK2, linewidth=0.8, linestyle=":")
axC.text(0.98, 0.045, "chance", transform=axC.transAxes, fontsize=7, color=INK2, ha="right", bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
axC.set_xticks(x, labs, fontsize=7.5)
axC.set_ylabel("ranking accuracy")
axC.set_ylim(0, 1.1)
axC.legend(frameon=False, fontsize=8, loc="lower left", ncol=2,
           bbox_to_anchor=(0, 1.01), borderaxespad=0, handlelength=1.2, columnspacing=1.1)
axC.set_title("(a) Encoder interventions", loc="left", fontsize=9.5, pad=27)

plabs = ["patch spinor\ncoefficients", "patch full\nactivation",
         "wrong\nsubspace"]
pk = ['redirect_fitpatch_half', 'redirect_fullpatch_picks_a2b2',
      'redirect_fitpatch_one']
pd = STATS['mechanisms_100k']['su2']['product']
pvals = [pd[k]['mean'] for k in pk]
perr = [pd[k]['sd'] for k in pk]
axD.bar(np.arange(3), pvals, 0.55, yerr=perr, color=[AQUA, BLUE, INK2],
        error_kw=dict(ecolor=INK, lw=0.8), edgecolor="white", linewidth=0.5)
axD.axhline(1 / 64, color=INK2, linewidth=0.8, linestyle=":")
axD.set_xticks(np.arange(3), plabs, fontsize=7.5)
axD.set_ylabel("counterfactual selected", labelpad=4)
axD.set_ylim(0, 1.1)
axD.set_title("(b) Product-state patching", loc="left", fontsize=9.5, pad=27)
for ax in (axC, axD):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig2.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.70, wspace=0.43)
fig2.savefig(OUTPUT / "figure2.pdf")
fig2.savefig(OUTPUT / "figure2.png", dpi=220)
print("saved paper/figure2.{pdf,png}")


# ---------------- appendix figure: dynamics and per-group spectra -------------
def appendix_figure():
    fig = plt.figure(figsize=(9.0, 5.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.78, wspace=0.40,
                          left=0.075, right=0.98, top=0.87, bottom=0.18)

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
        ax.set_xticks([0, 100000, 200000], ["0", "100k", "200k"])
        if col == 0:
            ax.set_ylabel("variance fraction (log)")
            fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7.5,
                       loc="upper left", bbox_to_anchor=(0.07, 0.995), ncol=4,
                       handlelength=1.8, columnspacing=1.0)
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
    ax.set_title("(c) SU(3) / PSU(3)", loc="left", fontsize=9.5)

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
    ax.legend(frameon=False, fontsize=7.3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=1.0)
    ax.set_title("(d) Spin(4) family",
                 loc="left", fontsize=9.5)

    # (e) Spin(5)/SO(5)
    ax = fig.add_subplot(gs[1, 2])
    sect5 = ["fve_4", "fve_5", "fve_10", "fve_14"]
    lab5 = [r"$\mathbf{4}$", r"$\mathbf{5}$", r"$\mathbf{10}$", r"$\mathbf{14}$"]
    m5 = {}
    runs5 = {"spin5": "data/spin5/spin5_N2048_f0.01_s{s}/log.jsonl",
             "so5": "data/spin5/so5_N2048_f0.01_s{s}/log.jsonl",
             "so5rf": "data/spin5/so5rf_f0.03_s{s}/log.jsonl"}
    for v, pat in runs5.items():
        vals = []
        for s0 in range(5):
            recs = [json.loads(l) for l in open(ROOT / pat.format(s=s0))
                    if "fve_4" in l]
            vals.append([recs[-1][k] for k in sect5])
        m5[v] = np.array(vals).mean(0)
    x = np.arange(4)
    ax.bar(x - 0.25, m5["spin5"], 0.24, color=BLUE, label="Spin(5)")
    ax.bar(x, m5["so5"], 0.24, color=ORANGE, label=r"SO(5), averaged $\Psi$")
    ax.bar(x + 0.25, m5["so5rf"], 0.24, color=AQUA,
           label=r"SO(5), intrinsic $\Psi$")
    ax.set_xticks(x, lab5, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), labelspacing=0.2)
    ax.set_title("(e) Spin(5) / SO(5)", loc="left", fontsize=9.5)

    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(OUTPUT / "figureA1.pdf")
    fig.savefig(OUTPUT / "figureA1.png", dpi=200)
    print("saved paper/figureA1.{pdf,png}")


appendix_figure()


# ---------------- per-seed figure: spectra and interventions ----------------
def seeds_figure():
    fig = plt.figure(figsize=(9.6, 4.6), facecolor="white")
    gs = fig.add_gridspec(3, 5, hspace=0.62, wspace=0.3,
                          left=0.085, right=0.99, top=0.93, bottom=0.17,
                          height_ratios=[1, 1, 1.15])
    js = [0.0, 0.5, 1.0, 1.5, 2.0]
    jlab = ["0", "½", "1", "3/2", "2"]
    for row, (variant, color, tag) in enumerate(
            [("su2", BLUE, "SU(2)"), ("so3", ORANGE, "SO(3)")]):
        for s in range(5):
            ax = fig.add_subplot(gs[row, s])
            recs = [json.loads(l) for l in open(
                ROOT / f"data/su2/{variant}_N2048_f0.01_s{s}/log.jsonl")
                if "fve_0.5" in l]
            r = [x for x in recs if x["step"] <= 100000][-1]
            ax.bar(np.arange(5), [r.get(f"fve_{j}", 0.0) for j in js],
                   0.7, color=color, edgecolor="white", linewidth=0.4)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(np.arange(5), jlab, fontsize=7)
            if s == 0:
                ax.set_ylabel(f"{tag}\nvariance", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.set_title(f"seed {s}", fontsize=8, loc="left")
    # interventions, one dot per seed
    axd = fig.add_subplot(gs[2, :])
    keys = [("none", "none"), ("fablate_spin_half", "remove spin ½"),
            ("fablate_spin_one", "remove spin 1"),
            ("even_projection", "center average")]
    for vi, (variant, color, mk) in enumerate(
            [("su2", BLUE, "o"), ("so3", ORANGE, "s")]):
        for ki, (key, lab) in enumerate(keys):
            vals = []
            for s in range(5):
                path = SELECTED / f"data/su2/{variant}_N2048_f0.01_s{s}/step_100000/causal_100000.json"
                d = json.loads(path.read_text())
                assert d['step'] == 100000
                vals.append(d["interventions"][key]["task_acc"])
            x = ki + (vi - 0.5) * 0.18 + np.linspace(-0.05, 0.05, 5)
            axd.plot(x, vals, mk, ms=4.5, color=color, alpha=0.85, lw=0,
                     label={"su2": "SU(2) model", "so3": "SO(3) model"}[variant]
                     if ki == 0 else None)
    axd.axhline(1 / 64, color=INK2, lw=0.8, ls=":")
    axd.text(3.42, 1 / 64 + 0.04, "chance", fontsize=7.5, color=INK2)
    axd.set_xticks(range(4), [l for _, l in keys], fontsize=8)
    axd.set_ylim(-0.05, 1.1)
    axd.set_ylabel("ranking accuracy")
    axd.legend(frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.31), ncol=2, borderaxespad=0)
    axd.set_title("interventions, one marker per seed", fontsize=8.5,
                  loc="left")
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(OUTPUT / "figure_seeds.pdf")
    fig.savefig(OUTPUT / "figure_seeds.png", dpi=200)
    print("saved paper/figure_seeds.{pdf,png}")


seeds_figure()
