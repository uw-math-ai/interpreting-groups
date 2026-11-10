# Compact Lie group composition

Code and results for *To Infinite and Beyond: How Neural Networks Learn Compact Lie Groups, a Mechanistic Study*, presented at the Interp4Discovery workshop at NeurIPS 2026.

[Read the paper on OpenReview](https://openreview.net/forum?id=fosUuKztgI).

**Authors:** Junaid Hasan and Michael Ruofan Zeng. Supported by the UW Math AI Lab.

## Get started

Install [uv](https://docs.astral.sh/uv/), then run these commands from the repository root:

```sh
uv sync --frozen --group dev
uv run python -m pytest tests -q
```

The environment uses Python 3.11 or later and PyTorch 2.11. Training uses a CUDA GPU. Logs and analysis results are included; trained weights are not. One test skips when historical weights are absent.

## Reproduce figures and tables

```sh
uv run python analysis/build_tables.py
uv run python figures/make_figures.py
```

These commands use the included results and need no GPU or checkpoints. Tables are written to `analysis/verified_*.json` and CSV files; figures are written to `figures/` as PDF and PNG.

Geometry figures are included. To regenerate them from trained weights, use `experiments/embedding_viz.py`. The architecture diagram is editable in `figures/figure_architecture.tikz`.

## Train a model

Use a new output directory to keep the archived results intact:

```sh
uv run python -m experiments.su2_train --variant su2 --N 2048 --frac 0.01 \
  --seed 0 --steps 100000 --save_dir runs/su2_s0
uv run python -m experiments.su2_train --variant so3 --N 2048 --frac 0.01 \
  --seed 0 --steps 100000 --save_dir runs/so3_s0
```

Other groups use `experiments.su3_train`, `experiments.spin4_train`, or `experiments.spin5_train`. Each trainer provides `--help`; the archived `data/` configurations record the settings used in the paper.

Analyze the resulting checkpoints with:

```sh
uv run python -m experiments.su2_causal --run_dir runs/su2_s0
uv run python -m experiments.su2_mechanism --run_dir runs/su2_s0
uv run python -m experiments.twosided --run_dir runs/so3_s0
```

## SO(5) follow-up

The paper reports a three-seed preprocessing comparison. A completed extension includes five paired baseline/whitened seeds and three rotation controls, with the original results preserved. Whitening improves frozen test accuracy by about 4.2 percentage points. See the [results and reproduction instructions](analysis/so5_followup/README.md).

## Find your way around

- `core/`: group operations, observations, and model components.
- `experiments/`: trainers and mechanistic analyses.
- `data/`: archived training configurations and logs.
- `analysis/`: verified results, checkpoint diagnostics, and table generation.
- `figures/`: paper figures and their source code.
- `preregistration/`: experiment plans and decision rules.

See [VALIDATION.md](VALIDATION.md) for verification details and [RELEASE_NOTES.md](RELEASE_NOTES.md) for changes. `main` and `nips_2026_workshop` contain the paper code; `exploratory` preserves the earlier finite-group work.

## License

[Apache License 2.0](LICENSE). Copyright (c) 2026 Junaid Hasan and Michael Ruofan Zeng. The MIT notice for code from earlier releases is preserved in [LICENSES/MIT.txt](LICENSES/MIT.txt); see [NOTICE](NOTICE).
