"""Permanent tests for the analysis provenance envelope (handoff G0.3)."""
import pytest

from core.envelope import (make_envelope, read_analysis, validate_envelope,
                           write_analysis)


def _env(**kw):
    return make_envelope("unit-test-analysis", 1, analysis_file=__file__,
                         analysis_args={"alpha": 1.0}, training_seed=0,
                         analysis_seed=123, **kw)


def test_envelope_has_required_fields():
    env = _env()
    for k in ("schema_name", "schema_version", "generated_at_utc",
              "code_commit", "analysis_source_sha256", "run_config_sha256",
              "training_seed", "analysis_seed", "analysis_args",
              "environment"):
        assert k in env
    assert env["environment"]["python"]
    assert len(env["analysis_source_sha256"]) == 64


def test_validation_rejects_missing_key():
    env = _env()
    del env["code_commit"]
    with pytest.raises(ValueError):
        validate_envelope(env)


def test_write_read_roundtrip(tmp_path):
    p = tmp_path / "analysis.json"
    write_analysis(p, {"metric": 0.5}, _env())
    d = read_analysis(p)
    assert d["results"]["metric"] == 0.5
    assert d["envelope"]["schema_name"] == "unit-test-analysis"


def test_read_rejects_legacy_payload(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text('{"metric": 0.5}')
    with pytest.raises(ValueError):
        read_analysis(p)
