# Validation of this update

- Paper-specific tests: **30 passed, 1 skipped**. The skipped test requires historical checkpoint weights, which are intentionally absent from the compact bundle.
- `analysis/build_tables.py`: passes using only packaged logs and JSON diagnostics.
- `figures/make_figures.py`: passes using only packaged inputs; writes all five numerical figure families.
- Checkpoint files and virtual environments are excluded.

These checks were run with the supplied locked PyTorch 2.11 environment. They validate this compact release. The fuller historical handoff contains additional legacy tests; three of those cannot collect without an unrelated omitted module, and they are not included in this paper-specific selection.
