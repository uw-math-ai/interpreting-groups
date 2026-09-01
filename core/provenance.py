"""Run-configuration provenance shared by all trainers (handoff G0.2).

Contract:
  * every derived config field is resolved BEFORE serialization;
  * requested and effective values are stored separately when they differ;
  * resume compares immutable effective fields against the checkpoint config
    and exits nonzero BEFORE loading any training state on a mismatch;
  * the checkpoint-selection rule and normalized launch command are recorded.
"""
import json
import shlex
import subprocess
import sys
from pathlib import Path

# Effective fields that a resumed run must share with its checkpoint. Fields
# absent from either side are ignored so the list can be shared across the
# su2/su3/spin4/spin5 trainers.
IMMUTABLE_KEYS = (
    "variant", "encoding", "loss", "m", "N", "frac", "sigma", "excl_radius",
    "K", "alpha", "enc_seed", "activation", "d_enc", "enc_hidden",
    "mlp_hidden", "mlp_depth", "batch", "seed", "hk_t", "bl_js", "bl_coefs",
)

CKPT_SELECTION_RULE = (
    "training saves ckpt_<step>.pt every ckpt_every steps and at the final "
    "step; unless an analysis states otherwise it uses the final checkpoint "
    "at the configured step budget"
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str:
    """Current commit hash with a +dirty marker, or 'unknown'."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=_REPO_ROOT, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=_REPO_ROOT, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def write_config(save_dir, args, requested_overrides=None):
    """Serialize the effective config plus provenance fields.

    Call this only after every derived field (e.g. the encoding width m) has
    been resolved on ``args``. ``requested_overrides`` maps field name to the
    originally requested value for each field whose effective value differs.
    Returns the dict that was written.
    """
    cfg = dict(vars(args))
    cfg["requested_overrides"] = dict(requested_overrides or {})
    cfg["ckpt_selection_rule"] = CKPT_SELECTION_RULE
    cfg["launch_command"] = shlex.join(sys.argv)
    cfg["code_commit"] = git_commit()
    Path(save_dir, "config.json").write_text(json.dumps(cfg, indent=2,
                                                        default=str))
    return cfg


def check_resume_compat(state, args, immutable_keys=IMMUTABLE_KEYS):
    """Refuse to resume from an incompatible checkpoint.

    ``state`` is the loaded checkpoint dict (before any state_dict is applied
    to the model or optimizer). Exits with status 1 on any mismatch of an
    immutable effective field, and with status 2 if the checkpoint carries no
    embedded config at all.
    """
    old = state.get("config")
    if old is None:
        print("resume refused: checkpoint has no embedded config",
              file=sys.stderr)
        raise SystemExit(2)
    bad = []
    for k in immutable_keys:
        if k in old and hasattr(args, k):
            new_v = getattr(args, k)
            if old[k] != new_v:
                bad.append((k, old[k], new_v))
    if bad:
        for k, o, n in bad:
            print(f"resume mismatch: {k}: checkpoint={o!r} requested={n!r}",
                  file=sys.stderr)
        raise SystemExit(1)
