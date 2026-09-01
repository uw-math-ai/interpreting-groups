# Submission package: Interpretability for Discovery (NeurIPS 2026 workshop)

Paper: "Neural networks trained on compact group composition learn the
global form of the group."

CFP: https://interpretability4discovery.github.io/cfp.html
Deadline: September 2, 2026 (AoE). Double-blind, non-archival.
Limit: 5 pages of main text (references and appendices excluded);
this draft uses ~4.6, leaving headroom for the +1 camera-ready page.

## Contents
- `main.tex`      -- the paper (main text + appendices A-C: proofs,
                     experimental details, complete result tables and the
                     dynamics figure)
- `refs.bib`      -- bibliography (arXiv identifiers in every entry)
- `neurips_2024.sty` -- stand-in style file in submission (anonymous) mode;
                     swap in the official NeurIPS 2026 workshop template
                     from the CFP when convenient -- no document changes
- `figure1.pdf`, `figureA1.pdf` -- figures, regenerated from tracked
                     results by `make_figure.py` (reads `../summary/`,
                     `../data/`)
- `main.pdf`      -- compiled draft (anonymous, line numbers on)

## Build
Compiled with Tectonic 0.15 (`tectonic main.tex`); any TeX Live with
pdflatex + bibtex also works.

## Before submission
1. The draft is anonymized (no names, affiliations, or acknowledgments;
   no repository URLs). Attach code + `summary/results_bundle.json` as
   anonymized supplementary material.
2. Swap in the official 2026 workshop style file from the CFP zip.
