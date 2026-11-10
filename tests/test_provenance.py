"""Permanent tests for run-config provenance (handoff G0.2)."""
import argparse
import json
from pathlib import Path

import pytest

from core.provenance import (IMMUTABLE_KEYS, check_resume_compat,
                             write_config)

REPO = Path(__file__).resolve().parent.parent


def _args(**kw):
    ns = argparse.Namespace(variant="so3", encoding="so3frame", loss="infonce",
                            m=32, N=2048, frac=0.01, sigma=0.2, K=64, seed=0,
                            d_enc=128, batch=4096)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_effective_config_is_serialized(tmp_path):
    """The config on disk must carry the effective value, with the requested
    value preserved separately when they differ."""
    args = _args()
    requested = {"m": args.m}
    args.m = 6                      # what get_encoding('so3frame') resolves to
    write_config(tmp_path, args, requested)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["m"] == 6
    assert cfg["requested_overrides"] == {"m": 32}
    assert cfg["ckpt_selection_rule"]
    assert cfg["launch_command"]
    assert cfg["code_commit"]


def test_resume_mismatch_exits_nonzero():
    """An incompatible resume must fail before any training state loads."""
    state = {"config": {"m": 6, "encoding": "so3frame", "seed": 0}}
    with pytest.raises(SystemExit) as e:
        check_resume_compat(state, _args(m=9, encoding="so3frame"))
    assert e.value.code == 1


def test_resume_without_embedded_config_is_refused():
    with pytest.raises(SystemExit) as e:
        check_resume_compat({"model": {}}, _args())
    assert e.value.code == 2


def test_resume_match_passes():
    args = _args(m=6)
    state = {"config": {k: getattr(args, k) for k in IMMUTABLE_KEYS
                        if hasattr(args, k)}}
    check_resume_compat(state, args)   # must not raise


@pytest.mark.skipif(
    not (REPO / "data/su2/so3_so3frame_f0.01_s0/ckpt_60000.pt").exists(),
    reason="historical checkpoints not present in this checkout")
def test_historical_p1b_effective_widths():
    """Checkpoint configs carry m=6 (frame) / m=9 (block), and the stale
    serialized configs are covered by the correction record."""
    import torch
    for enc, m_eff in [("so3frame", 6), ("so3block", 9)]:
        ck = REPO / f"data/su2/so3_{enc}_f0.01_s0/ckpt_60000.pt"
        st = torch.load(ck, map_location="cpu", weights_only=False)
        assert st["config"]["m"] == m_eff
        assert st["model"]["enc.0.weight"].shape[1] == m_eff
    rec = json.loads((REPO / "summary/config_corrections.json").read_text())
    runs = {c["run"] for c in rec["corrections"]}
    assert len(runs) == 10 and rec["schema_version"] == 1
