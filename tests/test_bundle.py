"""Permanent tests for the evidence bundle (handoff G0.4)."""
import json
from pathlib import Path

import pytest

from experiments.export_bundle import (REQUIRED_RUNS, build_bundle,
                                       validate_bundle)

REPO = Path(__file__).resolve().parent.parent


def test_validation_rejects_defective_entries():
    bundle = {"schema_name": "results-bundle", "schema_version": 2,
              "runs": {"su2/x": {"run_id": "su2/x", "seed": None,
                                 "final_metrics": {},
                                 "final_checkpoint": {},
                                 "ckpt_selection_rule": "",
                                 "analyses": {"a": {}}}},
              "analyses": {}, "missing_required": []}
    problems = validate_bundle(bundle)
    joined = "\n".join(problems)
    for needle in ("missing seed", "missing final metrics",
                   "missing final checkpoint hash",
                   "missing checkpoint-selection rule", "missing schema"):
        assert needle in joined


@pytest.mark.skipif(not (REPO / "data/su2").exists(),
                    reason="data/ not present in this checkout")
def test_bundle_regenerates_and_contains_required_runs(tmp_path):
    bundle = build_bundle()
    assert validate_bundle(bundle) == []
    out = tmp_path / "bundle.json"
    out.write_text(json.dumps(bundle, default=float))
    reloaded = json.loads(out.read_text())
    assert validate_bundle(reloaded) == []
    for run_id in REQUIRED_RUNS:
        assert (run_id in reloaded["runs"]
                or run_id in reloaded["missing_required"])
    assert not reloaded["missing_required"], reloaded["missing_required"]
    # the ten P1B runs carry checkpoint-embedded effective configs
    for enc, m in (("so3frame", 6), ("so3block", 9)):
        e = reloaded["runs"][f"su2/so3_{enc}_f0.01_s0"]
        assert e["config_source"] == "checkpoint"
        assert e["config"]["m"] == m
    # top-level analyses are labeled with their schema
    assert reloaded["analyses"]["p1b_reanalysis_wd1.0"][
        "schema_name"] == "p1b-reanalysis"
