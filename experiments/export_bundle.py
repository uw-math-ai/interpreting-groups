"""Export the versioned, validated evidence bundle (handoff G0.4).

Walks every run directory under data/{su2,su3,spin4,spin5}/ and emits
summary/results_bundle.json with schema results-bundle v2. A run is
COMPLETE when it has a config.json and a checkpoint at its configured step
budget; complete runs carry their effective config (checkpoint-embedded
when present), training seed, final metrics, checkpoint inventory with the
final checkpoint's SHA-256, the checkpoint-selection rule, and every
analysis JSON in the run directory labeled with its schema (envelope
schema when present, otherwise the explicit legacy label). Runs still
training are listed under in_progress with no metrics.

Usage:
    uv run python -m experiments.export_bundle              # regenerate
    uv run python -m experiments.export_bundle --validate   # check only
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import torch

from core.envelope import LEGACY_SCHEMA, _environment, sha256_file
from core.provenance import CKPT_SELECTION_RULE, git_commit

ROOT = Path(__file__).resolve().parent.parent

# Run ids that the bundle must contain for the current manuscript claims.
# so5_N2048_f0.01_s0 is also the 200k-extension stress cell (its step
# budget was raised to 200000 in place and it was resumed).
REQUIRED_RUNS = (
    [f"su2/so3_so3frame_f0.01_s{s}" for s in range(5)]
    + [f"su2/so3_so3block_f0.01_s{s}" for s in range(5)]
    + [f"spin5/spin5_N2048_f0.01_s{s}" for s in range(5)]
    + [f"spin5/so5_N2048_f0.01_s{s}" for s in range(5)]
    + ["spin5/so5_N2048_f0.01_K256_s0", "spin5/so5_N2048_f0.01_wd0.1_s0",
       "spin5/so5_N2048_f0.03_s0"]
)

TOP_LEVEL_ANALYSES = (
    "summary/p1b_reanalysis_wd1.0.json",
    "summary/p1b_trajectories_wd1.0.json",
    "summary/seed_summary.json",
)

ANALYSIS_PREFIXES = ("causal", "mechanism", "fusion", "twosided", "hk")


def analysis_schema(payload):
    if isinstance(payload, dict) and "envelope" in payload:
        env = payload["envelope"]
        return env.get("schema_name", LEGACY_SCHEMA), env.get(
            "schema_version")
    return LEGACY_SCHEMA, None


def collect_run(d, group):
    cfg_path = d / "config.json"
    if not cfg_path.exists():
        return None, "no config.json"
    cfg_json = json.loads(cfg_path.read_text())
    steps = cfg_json.get("steps")
    final_ckpt = d / f"ckpt_{steps}.pt"
    if steps is None or not final_ckpt.exists():
        return None, f"no final checkpoint at step budget {steps}"

    state = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "config" in state:
        config, config_source = state["config"], "checkpoint"
    else:
        config, config_source = cfg_json, "config.json"

    recs = [json.loads(l) for l in open(d / "log.jsonl") if l.strip()]
    evals = [r for r in recs if "acc_T2" in r or "mse_T2" in r]
    specs = [r for r in recs if any(k.startswith("fve") for k in r)]
    final_metrics = {}
    if specs:
        final_metrics.update(specs[-1])
    if evals:
        final_metrics.update(evals[-1])

    analyses = {}
    for js in sorted(d.glob("*.json")):
        if js.name == "config.json":
            continue
        if js.stem.split("_")[0] not in ANALYSIS_PREFIXES:
            continue
        payload = json.loads(js.read_text())
        name, version = analysis_schema(payload)
        analyses[js.stem] = {"schema_name": name, "schema_version": version,
                             "results": payload}

    ckpts = sorted(d.glob("ckpt_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    entry = {
        "run_id": f"{group}/{d.name}",
        "group": group,
        "config": config,
        "config_source": config_source,
        "seed": config.get("seed"),
        "final_metrics": final_metrics,
        "n_eval_records": len(evals),
        "checkpoints": [c.name for c in ckpts],
        "final_checkpoint": {"name": final_ckpt.name,
                             "sha256": sha256_file(final_ckpt),
                             "bytes": final_ckpt.stat().st_size},
        "ckpt_selection_rule": cfg_json.get("ckpt_selection_rule",
                                            CKPT_SELECTION_RULE),
    }
    entry["analyses"] = analyses
    return entry, None


def build_bundle(root=ROOT):
    runs, in_progress, skipped = {}, {}, []
    for group in ("su2", "su3", "spin4", "spin5"):
        base = root / "data" / group
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            entry, why = collect_run(d, group)
            run_id = f"{group}/{d.name}"
            if entry is not None:
                runs[run_id] = entry
            elif why == "no config.json":
                skipped.append({"run_id": run_id, "reason": why})
            else:
                in_progress[run_id] = {
                    "note": why,
                    "config_source": "config.json",
                }

    analyses = {}
    for rel in TOP_LEVEL_ANALYSES:
        p = root / rel
        if not p.exists():
            continue
        name, version = analysis_schema(json.loads(p.read_text()))
        analyses[Path(rel).stem] = {"path": rel, "sha256": sha256_file(p),
                                    "schema_name": name,
                                    "schema_version": version}

    queues = {}
    for qf in sorted((root / "data").glob("queue_v*_cmds.txt")):
        queues[qf.stem] = {"path": str(qf.relative_to(root)),
                           "sha256": sha256_file(qf),
                           "commands": qf.read_text().splitlines()}
    for qlog in sorted((root / "data").glob("queue_v*.log")):
        queues.setdefault(qlog.stem + "_log", {})
        queues[qlog.stem + "_log"] = {"path": str(qlog.relative_to(root)),
                                      "sha256": sha256_file(qlog)}

    missing = [r for r in REQUIRED_RUNS if r not in runs]
    env = _environment()
    env["uv_lock_sha256"] = sha256_file(root / "uv.lock")
    return {
        "schema_name": "results-bundle",
        "schema_version": 2,
        "generated_at_utc":
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "environment": env,
        "corrections": "summary/config_corrections.json",
        "required_run_ids": list(REQUIRED_RUNS),
        "missing_required": missing,
        "runs": runs,
        "in_progress": in_progress,
        "skipped": skipped,
        "analyses": analyses,
        "queues": queues,
    }


def validate_bundle(bundle):
    """Return a list of defects; empty means valid."""
    problems = []
    if bundle.get("schema_name") != "results-bundle":
        problems.append("wrong or missing schema_name")
    if bundle.get("schema_version") != 2:
        problems.append("wrong or missing schema_version")
    for run_id, e in bundle.get("runs", {}).items():
        loc = f"runs[{run_id}]"
        if not e.get("run_id"):
            problems.append(f"{loc}: missing run_id")
        if e.get("seed") is None:
            problems.append(f"{loc}: missing seed")
        if not e.get("final_metrics"):
            problems.append(f"{loc}: missing final metrics")
        if not e.get("final_checkpoint", {}).get("sha256"):
            problems.append(f"{loc}: missing final checkpoint hash")
        if not e.get("ckpt_selection_rule"):
            problems.append(f"{loc}: missing checkpoint-selection rule")
        for stem, a in e.get("analyses", {}).items():
            if not a.get("schema_name"):
                problems.append(f"{loc}: analysis {stem} missing schema")
    for name, a in bundle.get("analyses", {}).items():
        if not a.get("schema_name"):
            problems.append(f"analyses[{name}]: missing schema")
        if not a.get("sha256"):
            problems.append(f"analyses[{name}]: missing sha256")
    if bundle.get("missing_required"):
        # recorded, not fatal: the bundle states explicitly what is absent
        pass
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default=str(ROOT / "summary/results_bundle.json"))
    p.add_argument("--validate", action="store_true",
                   help="validate an existing bundle without regenerating")
    p.add_argument("--out", default=None,
                   help="write path (defaults to --bundle)")
    args = p.parse_args()

    if args.validate:
        bundle = json.loads(Path(args.bundle).read_text())
        problems = validate_bundle(bundle)
        for pr in problems:
            print(f"INVALID: {pr}", file=sys.stderr)
        if problems:
            sys.exit(1)
        print(f"{args.bundle}: valid results-bundle v2, "
              f"{len(bundle['runs'])} runs, "
              f"{len(bundle['in_progress'])} in progress, "
              f"missing_required={bundle['missing_required']}")
        return

    bundle = build_bundle()
    problems = validate_bundle(bundle)
    if problems:
        for pr in problems:
            print(f"INVALID: {pr}", file=sys.stderr)
        sys.exit(1)
    dest = Path(args.out or args.bundle)
    dest.write_text(json.dumps(bundle, indent=1, default=float))
    print(f"wrote {dest} ({len(bundle['runs'])} runs, "
          f"{len(bundle['in_progress'])} in progress, "
          f"{dest.stat().st_size / 1e6:.1f} MB)")
    if bundle["missing_required"]:
        print(f"missing required runs: {bundle['missing_required']}")


if __name__ == "__main__":
    main()
