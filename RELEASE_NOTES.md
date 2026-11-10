# Interp4Discovery — NeurIPS 2026 workshop release

This release contains the compact-group composition paper's code and lightweight evidence. `main` opens on the current paper release, `nips_2026_workshop` preserves the named submission snapshot, and `exploratory` retains the earlier finite-group code and galleries. Future submission releases can use separate branches.

The release includes the four group-family trainers, mechanistic analysis entrypoints, five-seed logs, uniform-checkpoint diagnostics, preregistered controls, corrected figure generation, and full-precision summary tables. Checkpoints and Python environments are excluded.

Camera-ready corrections include the 14.3% parameter-norm reduction, exclusion of the failed extra-layer cost cell, common 100k diagnostic horizons, matched-rank random ablations, explicit observation-map settings, and signed Haar plots normalized using the actual training-target scale.

The project metadata uses a generic package name so that automatic redaction of the source repository name does not modify the lockfile's package identity.

See `README.md` for reproduction commands and `VALIDATION.md` for checks. `UPDATE_MANIFEST.json` records file hashes. Further fresh replications and exploratory follow-ups remain separate from the archived five-seed statistics in this release.
