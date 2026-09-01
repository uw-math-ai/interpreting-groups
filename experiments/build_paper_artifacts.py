"""Build every headline paper artifact from the evidence bundle (G0.5).

Reads ONLY the bundle (plus top-level analysis files the bundle names,
verified against their recorded SHA-256) and writes machine-readable table
data, TeX table bodies, and a hash manifest. Output is deterministic: two
invocations on the same bundle produce byte-identical files.

Usage:
    uv run python -m experiments.build_paper_artifacts \
        --bundle summary/results_bundle.json --output paper/generated
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from core.envelope import sha256_file

ROOT = Path(__file__).resolve().parent.parent

SEED_CONDITIONS = {
    "SU(2)": {"runs": [f"su2/su2_N2048_f0.01_s{s}" for s in range(5)],
              "dominant_fve": "fve_0.5", "label": "spin 1/2"},
    "SO(3)": {"runs": [f"su2/so3_N2048_f0.01_s{s}" for s in range(5)],
              "dominant_fve": "fve_1.0", "label": "spin 1"},
    "SU(3)": {"runs": [f"su3/su3_N2048_f0.01_s{s}" for s in range(5)],
              "dominant_fve": "fve_10", "label": "3 + 3bar"},
    "PSU(3)": {"runs": [f"su3/psu3_N2048_f0.01_s{s}" for s in range(5)],
               "dominant_fve": "fve_11", "label": "adjoint 8"},
    "Spin(5)": {"runs": [f"spin5/spin5_N2048_f0.01_s{s}" for s in range(5)],
                "dominant_fve": "fve_4", "label": "spinor 4"},
    "SO(5)": {"runs": [f"spin5/so5_N2048_f0.01_s{s}" for s in range(5)],
              "dominant_fve": "fve_5", "label": "vector 5",
              "extra_fve": "fve_10"},
}

SO5_STRESS = [
    ("f = 0.01, s0 (200k extension)", "spin5/so5_N2048_f0.01_s0"),
    ("K = 256", "spin5/so5_N2048_f0.01_K256_s0"),
    ("weight decay 0.1", "spin5/so5_N2048_f0.01_wd0.1_s0"),
    ("f = 0.03", "spin5/so5_N2048_f0.03_s0"),
]


def mean_sd(values):
    m = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return round(m, 4), round(sd, 4)


def fmt(m, sd, digits=3):
    return f"${m:.{digits}f} \\pm {sd:.{digits}f}$"


def seed_table(bundle):
    rows = {}
    for cond, spec in SEED_CONDITIONS.items():
        runs = [bundle["runs"][r] for r in spec["runs"]
                if r in bundle["runs"]]
        if not runs:
            continue
        accs = [r["final_metrics"]["acc_T2"] for r in runs]
        fves = [r["final_metrics"][spec["dominant_fve"]] for r in runs]
        row = {"n_seeds": len(runs),
               "seeds": sorted(r["seed"] for r in runs),
               "run_ids": [r["run_id"] for r in runs],
               "dominant_subspace": spec["label"],
               "acc_T2": dict(zip(("mean", "sd"), mean_sd(accs))),
               "dominant_fve": dict(zip(("mean", "sd"), mean_sd(fves)))}
        if "extra_fve" in spec:
            extra = [r["final_metrics"][spec["extra_fve"]] for r in runs]
            row["adjoint_fve"] = dict(zip(("mean", "sd"), mean_sd(extra)))
        rows[cond] = row
    return rows


def so5_stress_table(bundle):
    rows = []
    for label, run_id in SO5_STRESS:
        e = bundle["runs"].get(run_id)
        if e is None:
            continue
        fm = e["final_metrics"]
        rows.append({"condition": label, "run_id": run_id,
                     "step": fm.get("step"),
                     "acc_T2": round(fm["acc_T2"], 4),
                     "fve_vector5": round(fm["fve_5"], 4),
                     "fve_adjoint10": round(fm["fve_10"], 4)})
    return rows


def p1b_table(bundle, root):
    ref = bundle["analyses"]["p1b_reanalysis_wd1.0"]
    path = root / ref["path"]
    if sha256_file(path) != ref["sha256"]:
        raise SystemExit(f"{path} does not match the bundle hash; "
                         "regenerate the bundle first")
    payload = json.loads(path.read_text())
    stats = payload["results"]["paired_statistics"]
    keep = ("margin_at_matched_norm", "final_margin", "final_total_norm",
            "final_margin_per_norm", "final_T2_accuracy",
            "final_train_loss")
    out = {}
    for k in keep:
        s = stats[k]
        out[k] = {
            "paired_diffs_block_minus_frame":
                [round(d, 4) for d in s["paired_diffs_block_minus_frame"]],
            "mean": round(s["mean"], 4),
            "p_one_sided": round(s["p_one_sided"], 4),
            "p_two_sided": round(s["p_two_sided"], 4),
            "ci95": [round(v, 4) for v in s["ci95"]],
        }
    return out


def tex_seeds(rows):
    lines = []
    for cond, r in rows.items():
        cells = [cond, r["dominant_subspace"],
                 fmt(r["acc_T2"]["mean"], r["acc_T2"]["sd"]),
                 fmt(r["dominant_fve"]["mean"], r["dominant_fve"]["sd"])]
        if "adjoint_fve" in r:
            cells[-1] += (" / "
                          + fmt(r["adjoint_fve"]["mean"],
                                r["adjoint_fve"]["sd"]))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def tex_so5(rows):
    lines = []
    for r in rows:
        lines.append(" & ".join([
            r["condition"], f"${r['acc_T2']:.3f}$",
            f"${r['fve_vector5']:.3f}$",
            f"${r['fve_adjoint10']:.3f}$"]) + r" \\")
    return "\n".join(lines) + "\n"


def tex_p1b(stats):
    pretty = {
        "margin_at_matched_norm": "margin at matched norm",
        "final_margin": "final margin",
        "final_total_norm": "final total norm",
        "final_margin_per_norm": "final margin/norm",
        "final_T2_accuracy": "final T2 accuracy",
        "final_train_loss": "final train loss",
    }
    lines = []
    for k, s in stats.items():
        diffs = ", ".join(f"{d:+.3f}"
                          for d in s["paired_diffs_block_minus_frame"])
        lines.append(" & ".join([
            pretty[k], f"${s['mean']:+.3f}$", diffs,
            f"${s['p_one_sided']:.4f}$",
            f"$[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]$"]) + r" \\")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default=str(ROOT
                                           / "summary/results_bundle.json"))
    p.add_argument("--output", default=str(ROOT / "paper/generated"))
    args = p.parse_args()

    bundle_path = Path(args.bundle)
    bundle = json.loads(bundle_path.read_text())
    root = bundle_path.resolve().parent.parent
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "source_bundle_sha256": sha256_file(bundle_path),
        "bundle_code_commit": bundle["code_commit"],
        "seed_statistics": seed_table(bundle),
        "so5_stress": so5_stress_table(bundle),
        "p1b_wd1.0_paired": p1b_table(bundle, root),
    }

    files = {
        "tables.json": json.dumps(tables, indent=2, sort_keys=True) + "\n",
        "tab_seeds.tex": tex_seeds(tables["seed_statistics"]),
        "tab_so5.tex": tex_so5(tables["so5_stress"]),
        "tab_p1b.tex": tex_p1b(tables["p1b_wd1.0_paired"]),
    }
    for name, content in files.items():
        (out / name).write_text(content)

    manifest = {"source_bundle": str(bundle_path),
                "source_bundle_sha256": tables["source_bundle_sha256"],
                "files": {n: sha256_file(out / n) for n in sorted(files)}}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(files) + 1} artifacts to {out}")
    if not math.isfinite(
            tables["p1b_wd1.0_paired"]["final_margin"]["mean"]):
        raise SystemExit("non-finite value in generated tables")


if __name__ == "__main__":
    main()
