# Compact Lie group composition

Code and lightweight results for *To Infinite and Beyond: How Neural Networks Learn Compact Lie Groups, a Mechanistic Study*.

Camera-ready release for the Interp4Discovery workshop at NeurIPS 2026.

Authors: Junaid Hasan and Michael Ruofan Zeng. This work was supported by the UW Math AI Lab.

This camera-ready update includes the group-family trainers, mechanistic analyses, corrected figure generation, and verified statistics. Checkpoints and Python environments are not included.

## Setup and checks

From this directory:

```sh
uv sync --frozen --group dev
uv run python -m pytest tests -q
```

The lockfile is the full experiment environment, with PyTorch 2.11. The paper-specific test selection covers signed Haar estimation, group conventions, SO(5) observations, polynomial composition, provenance, and analysis envelopes. A checkpoint-dependent provenance test skips when historical weights are absent. Unrelated legacy modular-table tests are outside this compact release.

Group self-tests are also available as `python -m core.su2`, `core.su3`, `core.spin4`, and `core.spin5`.

## Reproduce the figures and tables

```sh
uv run python analysis/build_tables.py
uv run python figures/make_figures.py
```

These commands require no GPU or checkpoints. They read archived training logs and uniform-checkpoint analysis results, and regenerate the numerical paper figures: `figure1`, `figure2`, `figure_seeds`, `figureA1`, and `figure_spectral`, as PDF and PNG. Full-precision tables are written to `analysis/verified_*.json` and CSV files.

The geometry figures are included as PDFs. Their checkpoint-based source is `experiments/embedding_viz.py`; regenerating those illustrations requires the relevant trained weights in the original `data/` run directories. Create `paper/` before running that script.

## Layout

- `core/`: group operations, exact coefficients, frozen observations, scorer, and provenance helpers.
- `experiments/`: training, causal interventions, patching, translation tests, Haar coefficients, cost analyses, polynomial controls, and SO(5) diagnostics.
- `data/`: archived configurations and logs for the reported runs. No weights are included.
- `analysis/selected_checkpoints/`: rerun diagnostics with explicit checkpoint steps.
- `analysis/`: verified aggregates, spectral target scales, matched controls, table builder, and supporting analyses.
- `figures/`: figure generator and paper figure files.
- `preregistration/`: the relevant cost, polynomial, and SO(5) decision rules.
- `summary/config_corrections.json`: the original correction ledger for historical explicit-feature widths.

## Fresh training

Use a new output directory to preserve the archived results:

```sh
uv run python -m experiments.su2_train --variant su2 --N 2048 --frac 0.01 \
  --seed 0 --steps 100000 --save_dir runs/su2_s0
uv run python -m experiments.su2_train --variant so3 --N 2048 --frac 0.01 \
  --seed 0 --steps 100000 --save_dir runs/so3_s0
```

Other families use `experiments.su3_train` (`su3` or `psu3`), `experiments.spin4_train` (`spin4`, `so4`, `leftq`, or `pso4`), and `experiments.spin5_train` (`spin5` or `so5`). Intrinsic SO(5) observations use `--variant so5 --encoding so5rf --m 48 --frac 0.03`.

Main SU(2)/SO(3) settings use 32 observations, batch 4096, noise 0.2, and exclusion radius 0.5. SU(3) uses 32 complex probes (64 real observations), batch 2048, noise 0.15, and exclusion radius 0.7. Spin(4) uses 48 observations, batch 2048, noise 0.2, and exclusion radius 0.7. Spin(5) uses 48 observations, batch 2048, noise 0.15, and exclusion radius 0.7. All use 64 candidates and the default learning rate 0.001 and weight decay 1.0.

SO(3)'s landmark observation is `exp(alpha * abs(dot(q,h)))`; the alternative random-MLP and function-value quotient maps use antipodal averaging.

## Mechanistic analyses from trained weights

```sh
uv run python -m experiments.su2_causal --run_dir runs/su2_s0
uv run python -m experiments.su2_mechanism --run_dir runs/su2_s0
uv run python -m experiments.twosided --run_dir runs/so3_s0
uv run python -m experiments.hk_haar --run runs/heatkernel_s0
```

The archived five-seed table diagnostics use step 100k, exact positives for causal ranking, and noisy positives for T2. The figure generator reads the verified aggregates rather than selecting diagnostic files by filename order. The rank-matched random control removes four neural directions on SU(2) and nine on SO(3).

`analysis/check_translation_controls.py --work . --out new_controls.json --device cpu --include-exact-controls` reproduces the matched initialization comparison when the corresponding 100k checkpoint weights are supplied in the matching `analysis/selected_checkpoints/` folders. The shipped results are in `analysis/translation_controls_matched.json`.

## Interpretation of the cost and spectral results

The extra-layer cost cell fails its preregistered accuracy floor and is excluded from supporting cost evidence. The four weight-decay cells and doubled-width cell remain eligible. The measured norm reduction in the base comparison is 14.3%, with a 31.2% gain in margin per norm.

The spectral plot uses signed Haar coefficients divided by the corresponding standardized target coefficients, restoring the actual training-target standard deviation. It uses no per-seed median rescaling and displays all seed points. The older sequential QR variance profile is a different diagnostic.

MIT license; see `LICENSE`.
