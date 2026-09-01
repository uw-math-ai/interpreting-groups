# Submission package: MATH-AI 2026 (NeurIPS workshop)

Paper: "Neural networks trained on compact group composition learn the
global form of the group."

## Contents
- `main.tex`     -- the paper (4 content pages + references + appendices)
- `refs.bib`     -- bibliography (arXiv identifiers included in every entry)
- `neurips_2024.sty` -- NeurIPS style file (replace with the 2026 file when
                    the workshop releases it; no document changes needed)
- `figure1.pdf`  -- Figure 1, regenerated from tracked results by
                    `make_figure.py` (reads `../summary/` and `../data/`)
- `main.pdf`     -- compiled camera-ready draft

## Build
Compiled with Tectonic 0.15 (`tectonic main.tex`); any TeX Live with
pdflatex + bibtex also works.

## Before submission
1. Replace the author block in `main.tex` (currently the lab placeholder)
   with the author list, and anonymize if the final call requires it
   (`\usepackage{neurips_2024}` without the `final` option).
2. Swap in the official 2026 style file when available.
3. Every number in the paper regenerates from `../summary/results_bundle.json`.
