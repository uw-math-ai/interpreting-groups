# Interpreting continuous-group composition

This repository trains small encoder-factorized networks to identify
products in compact Lie groups presented through fixed generic encodings,
and reverse engineers the representations they learn. The experimental
program covers representation selection across ten group forms drawn from
SU(2), SU(3), Spin(4), and Spin(5) and their central quotients, causal
mechanism tests (function-space ablations, activation patching, the
two-sided translation test), frame-versus-coefficient-block cost
comparisons on SO(3), and heat-kernel spectral analyses. Results are
tracked in a validated evidence bundle from which the paper artifacts
regenerate.

## Setup

The environment is locked. From a clean checkout:

    uv sync --frozen

Run the tests with either invocation. Both must pass:

    uv run pytest tests/ -q
    uv run python -m pytest tests/ -q

## Smoke run

A short training run to verify the stack end to end (about two minutes on
a GPU, and it writes only to the scratch directory):

    uv run python -m experiments.su2_train --variant su2 --N 2048 \
        --frac 0.01 --seed 0 --steps 2000 --save_dir data/su2/smoke_s0

Each run directory receives a config.json with the effective
configuration, requested overrides, checkpoint-selection rule, launch
command, and code commit. Resuming an incompatible run fails before any
training state loads.

## Analyses

Analysis scripts write JSON with a versioned provenance envelope
(core/envelope.py). Example, the P1B frame-versus-block reanalysis:

    uv run python -m experiments.p1b_analysis --cell wd1.0 \
        --frame-glob 'data/su2/so3_so3frame_f0.01_s{s}' \
        --block-glob 'data/su2/so3_so3block_f0.01_s{s}' \
        --out summary/p1b_reanalysis_wd1.0.json

## Evidence bundle and paper artifacts

The tracked bundle records every complete run with its effective config,
seed, final metrics, final-checkpoint SHA-256, checkpoint-selection rule,
and schema-labeled analyses:

    uv run python -m experiments.export_bundle             # regenerate
    uv run python -m experiments.export_bundle --validate  # check only

Paper tables and figure inputs build from the bundle alone, with no reads
under data/, and are byte-for-byte deterministic:

    uv run python -m experiments.build_paper_artifacts \
        --bundle summary/results_bundle.json --output paper/generated

The paper compiles with tectonic (the binary is gitignored, fetch any
0.15+ build):

    cd paper && ./tectonic main.tex

## Data and run policy

data/ holds checkpoints and logs and is not tracked. summary/ holds
lightweight tracked results, including the bundle, the config-correction
record, and analysis outputs. Preregistrations for new experiment grids
are committed under preregistration/ before launch.

Run directories follow variant_encoding_wd_frac_seed, for example
data/su2/so3_so3frame_wd0.1_f0.01_s2 (the wd segment is omitted at the
default weight decay 1.0). Training uses about 1.3 GB of GPU memory per
concurrent run at batch 4096, so a 16 GB T4 sustains four concurrent
runs. Queue scripts under experiments/run_queue_v*.sh launch detached
sweeps.

## Legacy: finite groups

The original finite-group grokking project (Cayley tables, geometric
irreps, polytope readouts) lives in RESULTS.md, with the SU(2) public
summary in RESULTS_SU2.md. The finite-group training code predates the
continuous-group program and is unmaintained.
