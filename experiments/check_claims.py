"""Evidence-freeze claim checker (handoff Section 11).

Validates summary/claim_ledger.json against the manuscript and the
results bundle: every ledger entry's anchor text must appear in
paper/main.tex (whitespace-normalized), every referenced bundle key must
exist, and every entry must carry status, seeds, uncertainty method, and
locations. The paper build script runs this first and aborts on failure,
so a quantitative claim cannot outrun its evidence entry.
"""
import json
import re
import sys
from pathlib import Path

REQUIRED = ("id", "text_anchor", "status", "bundle_keys", "seeds",
            "checkpoint_rule", "uncertainty", "locations")
STATUSES = {"theorem", "measured", "inferred", "conjecture", "limitation"}


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    ledger = json.loads(Path("summary/claim_ledger.json").read_text())
    bundle = json.loads(Path("summary/results_bundle.json").read_text())
    tex = norm(Path("paper/main.tex").read_text())
    known = set(bundle.get("runs", {})) | set(bundle.get("analyses", {}))
    errors = []
    for e in ledger["claims"]:
        missing = [k for k in REQUIRED if k not in e]
        if missing:
            errors.append(f"{e.get('id','?')}: missing fields {missing}")
            continue
        if e["status"] not in STATUSES:
            errors.append(f"{e['id']}: bad status {e['status']}")
        if norm(e["text_anchor"]) not in tex:
            errors.append(f"{e['id']}: anchor text not found in main.tex")
        for k in e["bundle_keys"]:
            if k not in known:
                errors.append(f"{e['id']}: bundle key '{k}' not in bundle")
    if errors:
        print("CLAIM CHECK FAILED:", file=sys.stderr)
        for err in errors:
            print("  " + err, file=sys.stderr)
        sys.exit(1)
    print(f"claim check passed: {len(ledger['claims'])} claims verified "
          f"against {len(known)} bundle entries")


if __name__ == "__main__":
    main()
