"""Permanent tests for the deterministic paper-artifact builder (G0.5)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "summary/results_bundle.json"


def _build(out_dir):
    subprocess.run(
        [sys.executable, "-m", "experiments.build_paper_artifacts",
         "--bundle", str(BUNDLE), "--output", str(out_dir)],
        cwd=REPO, check=True, capture_output=True)


@pytest.mark.skipif(not BUNDLE.exists(), reason="bundle not present")
def test_double_build_is_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _build(a)
    _build(b)
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


@pytest.mark.skipif(not BUNDLE.exists(), reason="bundle not present")
def test_manifest_hashes_match_files(tmp_path):
    from core.envelope import sha256_file
    _build(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["files"], "manifest lists no files"
    for name, digest in manifest["files"].items():
        assert sha256_file(tmp_path / name) == digest, name
    tables = json.loads((tmp_path / "tables.json").read_text())
    assert "SU(2)" in tables["seed_statistics"]
    assert tables["p1b_wd1.0_paired"]["margin_at_matched_norm"]["mean"] > 0
