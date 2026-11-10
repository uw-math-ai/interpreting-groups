"""Versioned provenance envelope for analysis outputs (handoff G0.3).

Every new analysis JSON is written as

    {"envelope": {...}, "results": {...}}

through ``write_analysis``. The envelope carries schema identity, code and
input hashes, seeds, complete analysis arguments, and the software/device
environment. Legacy outputs written before this module existed are labeled
``legacy-pre-envelope-0`` by the bundle exporter and remain immutable.
"""
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

from core.provenance import git_commit

LEGACY_SCHEMA = "legacy-pre-envelope-0"

REQUIRED_KEYS = (
    "schema_name", "schema_version", "generated_at_utc", "code_commit",
    "analysis_source_sha256", "source_checkpoint", "run_config_sha256",
    "training_seed", "analysis_seed", "analysis_args", "environment",
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _environment():
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["device"] = (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu")
    except Exception:
        env["torch"] = env["cuda"] = env["device"] = None
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        env["numpy"] = None
    return env


def make_envelope(schema_name, schema_version, *, analysis_file,
                  analysis_args, ckpt_path=None, run_config_path=None,
                  training_seed=None, analysis_seed=None):
    """Build a validated envelope dict.

    ``analysis_file`` is the ``__file__`` of the analysis script.
    ``ckpt_path`` and ``run_config_path`` may be None for analyses that do
    not read a checkpoint (e.g. synthetic calibrations).
    """
    ckpt = None
    if ckpt_path is not None:
        ckpt = {"path": str(ckpt_path), "sha256": sha256_file(ckpt_path)}
    env = {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "generated_at_utc":
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "analysis_source_sha256": sha256_file(analysis_file),
        "source_checkpoint": ckpt,
        "run_config_sha256": (sha256_file(run_config_path)
                              if run_config_path is not None else None),
        "training_seed": training_seed,
        "analysis_seed": analysis_seed,
        "analysis_args": dict(analysis_args),
        "environment": _environment(),
    }
    validate_envelope(env)
    return env


def validate_envelope(env):
    missing = [k for k in REQUIRED_KEYS if k not in env]
    if missing:
        raise ValueError(f"envelope missing keys: {missing}")
    if not env["schema_name"] or env["schema_version"] is None:
        raise ValueError("envelope schema_name/schema_version must be set")
    if env["analysis_args"] is None:
        raise ValueError("envelope analysis_args must be a dict")


def write_analysis(path, results, envelope):
    """Validate the envelope and write {'envelope':..., 'results':...}."""
    validate_envelope(envelope)
    payload = {"envelope": envelope, "results": results}
    Path(path).write_text(json.dumps(payload, indent=2, default=float))
    return payload


def read_analysis(path):
    d = json.loads(Path(path).read_text())
    if "envelope" not in d:
        raise ValueError(f"{path} has no envelope (legacy schema?)")
    validate_envelope(d["envelope"])
    return d
