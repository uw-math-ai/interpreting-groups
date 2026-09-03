#!/bin/bash
# Evidence-gated paper build: regenerate artifacts from the bundle,
# verify the claim ledger, then compile. Fails closed.
set -e
cd "$(dirname "$0")/.."
uv run python -m experiments.build_paper_artifacts \
    --bundle summary/results_bundle.json --output paper/generated
uv run python -m experiments.check_claims
cd paper && ./tectonic main.tex
