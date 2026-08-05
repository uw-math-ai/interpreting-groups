"""Aggregate seed replications of the SU(2)/SO(3) cells.

For each cell (variant x frac) collects, across seeds, the final logged metrics
(at the last step <= --at_step, so 200k pilot runs are compared at the same
horizon as 100k seed runs) and, when present, the causal JSONs. Prints a
mean +- sd table and writes summary/seed_summary.json.

Usage:
    uv run python -m experiments.su2_seed_summary [--at_step 100000]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/su2"


def collect(variant, frac, at_step):
    rows = []
    for d in sorted(DATA.glob(f"{variant}_N2048_f{frac}_s*")):
        m = re.fullmatch(rf"{variant}_N2048_f{frac}_s(\d+)", d.name)
        if not m or not (d / "log.jsonl").exists():
            continue
        specs = [json.loads(l) for l in open(d / "log.jsonl") if "fve_0.5" in l]
        specs = [s for s in specs if s["step"] <= at_step]
        if not specs:
            continue
        s = specs[-1]
        row = {"seed": int(m.group(1)), "step": s["step"],
               "T1": s["acc_T1"], "T2": s["acc_T2"],
               "fve_half": s["fve_0.5"], "fve_one": s["fve_1.0"],
               "fve_3half": s["fve_1.5"], "fve_two": s["fve_2.0"],
               "pm_cos": s["pm_cos_mean"], "r2_q": s["r2_q"],
               "r2_qq": s["r2_qq"]}
        causal_files = sorted(d.glob("causal_*.json"))
        if causal_files:
            c = json.loads(causal_files[-1].read_text())
            iv = c["interventions"]
            row["D_c"] = iv["none"]["center_disc"]
            row["D_c_no_half"] = iv["fablate_spin_half"]["center_disc"]
            row["task_no_half"] = iv["fablate_spin_half"]["task_acc"]
            row["task_no_one"] = iv["fablate_spin_one"]["task_acc"]
            key = "r2_upto_j0.5" if variant == "su2" else "r2_upto_j1.0"
            row["char_r2"] = c["character_fit"][key]
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--at_step", type=int, default=100_000)
    args = p.parse_args()

    out = {}
    for variant in ["su2", "so3"]:
        for frac in ["0.002", "0.01"]:
            rows = collect(variant, frac, args.at_step)
            if not rows:
                continue
            cell = f"{variant}_f{frac}"
            out[cell] = rows
            keys = [k for k in rows[0] if k not in ("seed", "step")
                    and all(k in r for r in rows)]
            print(f"\n== {cell}  (n = {len(rows)} seeds, "
                  f"steps {sorted(set(r['step'] for r in rows))})")
            for k in keys:
                v = np.array([r[k] for r in rows], dtype=float)
                print(f"  {k:<14} {v.mean():+.4f} +- {v.std():.4f}   "
                      f"[{v.min():+.4f}, {v.max():+.4f}]")
    dest = ROOT / "summary/seed_summary.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
