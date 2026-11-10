# SO(5) preprocessing and sector interventions

The camera-ready follow-up compares three paired seeds of the averaged SO(5) observation with a fixed invertible whitening, at the same 100k-step training budget and data fraction 0.03. An orthogonal rotation provides a one-seed control. The original three-seed plan is retained here. A later extension is separate from these completed results.

Frozen noisy T2 improves from 0.7328 ± 0.0074 to 0.7747 ± 0.0070, with paired gains 0.0415–0.0425. Independent 8192-example accuracy is 0.7272 ± 0.0032 versus 0.7769 ± 0.0092. The rotation control gives 0.7368 on the independent set. The vector/adjoint mixture remains. Whitening changes optimizer and regularization geometry together with input covariance.

The function-space intervention jointly fits the 25 vector and 100 adjoint coefficients on 16384 independent Haar elements. Removing adjoint content from both encoder loci raises noisy accuracy from 0.7288 to 0.8769 in the three fresh averaged models and from 0.7763 to 0.8972 in the whitened models. Removing vector content reduces accuracy near chance. This diagnostic uses exact group coefficients at inference. Complete archived and fresh results, including shuffled-fit controls, are included as JSON.

The lightweight configurations and original training logs are in `data/followup_so5/`; checkpoints are not bundled. Retrain into a separate directory with:

```sh
uv run python analysis/so5_followup/reproduce.py --mode identity --seed 0
uv run python analysis/so5_followup/reproduce.py --mode whiten --seed 0
uv run python analysis/so5_followup/reproduce.py --mode rotate --seed 0
```

Run identity and whiten for seeds 0, 1, and 2. Use `--mode intrinsic` for the additional fresh intrinsic-observation control. The wrapper refuses to overwrite an existing run. `--dry-run` prints the exact command, and `--steps` permits a short execution check.

After retraining the eight conditions listed in `fresh_cases.json`, rerun the interventions:

```sh
uv run python analysis/so5_followup/sector_interventions.py \
  --source . --cases-json analysis/so5_followup/fresh_cases.json \
  --out analysis/so5_followup/rerun_fresh_interventions.json \
  --n-fit 16384 --n-eval 8192 --batch 512 --device cuda
```

`archived_cases.json` lists the ten original archived models, using their conventional checkpoint locations. For those comparisons, use `--n-eval 4096`. Every evaluation uses common examples and candidate sets across interventions.
