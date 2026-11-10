# SO(5) preprocessing and sector interventions

The camera-ready follow-up compares three paired seeds of the averaged SO(5) observation with a fixed invertible whitening, at the same 100k-step training budget and data fraction 0.03. An orthogonal rotation provides a one-seed control. The original three-seed plan is retained here. The completed five-seed extension is in [extension/](extension/); the original three-seed results below are preserved for comparison with the camera-ready paper.

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

## Completed seed extension

The extension was declared after observing the initial three paired seeds. It adds baseline and whitening seeds 3–4 and rotation seeds 1–2, completing every run at 100k steps. [OpenReview paper](https://openreview.net/forum?id=fosUuKztgI).

| Condition | Seeds | Frozen T2 | Independent T2 | Accuracy after adjoint removal |
| --- | ---: | ---: | ---: | ---: |
| Untreated | 5 | 0.7319 ± 0.0059 | 0.7287 ± 0.0032 | 0.8767 ± 0.0011 |
| Whitened | 5 | 0.7736 ± 0.0058 | 0.7771 ± 0.0072 | 0.8968 ± 0.0013 |
| Rotated | 3 | 0.7355 ± 0.0079 | 0.7306 ± 0.0049 | 0.8676 ± 0.0023 |

All uncertainties are population standard deviations across the observed seeds. Frozen-T2 whitening gains are positive in every paired seed, ranging from 0.0408 to 0.0425. The independent test and sector-intervention columns use separate shared 8192-example sets. Removing vector content still reduces accuracy near chance. The vector/adjoint mixture persists; whitening does not close the performance gap to the intrinsic observation.

Train identity and whiten for seeds 0–4, rotate for seeds 0–2, and the intrinsic control at seed 0 with the wrapper above. `extension/cases.json` lists all fourteen checkpoints. The declared extension plan, original logs, per-checkpoint verification, complete intervention results, and full-precision aggregates are included. The earlier three-seed files are unchanged.

Verify new test accuracy and sector FVE after retraining:

```sh
uv run python analysis/so5_followup/verify_checkpoints.py \
  --source . --cases-json analysis/so5_followup/extension/cases.json \
  --out analysis/so5_followup/extension/rerun_verification.json --device cuda
```

Run the intervention command above with `--cases-json analysis/so5_followup/extension/cases.json` to evaluate the full extension.
