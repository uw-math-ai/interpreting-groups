"""Export a compact, tracked results bundle (plan P0).

Walks every run directory under data/{su2,su3,spin4,spin5}/, collecting:
    - config.json (verbatim);
    - the final evaluation record and the final spectroscopy record from
      log.jsonl;
    - every analysis JSON (causal_*.json, mechanism_*.json, fusion_*.json);
    - SHA-256 of the latest checkpoint (identity without shipping weights);
    - environment lock identity (SHA-256 of uv.lock, torch/numpy versions).

Output: summary/results_bundle.json (tracked in git). Headline tables in
summary/report.md are regenerable from this bundle without checkpoints.

Usage: uv run python -m experiments.export_bundle
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_run(d):
    out = {}
    cfg = d / "config.json"
    if cfg.exists():
        out["config"] = json.loads(cfg.read_text())
    log = d / "log.jsonl"
    if log.exists():
        recs = [json.loads(l) for l in open(log) if l.strip()]
        evals = [r for r in recs if "acc_T2" in r or "mse_T2" in r]
        specs = [r for r in recs if any(k.startswith("fve") for k in r)]
        if evals:
            out["final_eval"] = evals[-1]
        if specs:
            out["final_spec"] = specs[-1]
        out["n_eval_records"] = len(evals)
    for js in sorted(d.glob("*.json")):
        if js.name == "config.json":
            continue
        stem = js.stem
        if stem.split("_")[0] in ("causal", "mechanism", "fusion", "twosided",
                                  "hk"):
            out[stem] = json.loads(js.read_text())
    ckpts = sorted(d.glob("ckpt_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
        out["ckpt"] = {"name": ckpts[-1].name,
                       "sha256": sha256(ckpts[-1]),
                       "bytes": ckpts[-1].stat().st_size}
    return out


def main():
    bundle = {"environment": {
        "torch": torch.__version__, "numpy": np.__version__,
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
    }}
    for fam in ("su2", "su3", "spin4", "spin5"):
        base = ROOT / "data" / fam
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            run = collect_run(d)
            if run:
                bundle[f"{fam}/{d.name}"] = run
    dest = ROOT / "summary/results_bundle.json"
    dest.write_text(json.dumps(bundle, indent=1))
    n = len(bundle) - 1
    print(f"wrote {dest} ({n} runs, {dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
