This project gathers information about DPA decisions on the GDPR for comparative analysis and statistical analysis. We are at the very first step of prepping data.

Current structure:

- `/raw-data` contains a CSV file with all decisions and their machine translations, pooled from MD files in subfolder. Each decision is given an ID.
- `/analyzed-decisions` contains AI-processed decisions where each one has 68 fields extracted using AI. The prompt given to AI is an important file explaining the values, and is stored as `.md` in the folder.
- Master file is CSV; each answer is delimited with newline.
- For convenience, the same responses (raw, uncleaned) are also included as JSON.
- Therefore, `master-analyzed-data-unclean.csv` is the most vital file in the project.
- We must assume additional rows may be added over time, so all our processing must be documented and reproducible.
- `/resources` contains various resources and utilities — for instance, a list of ISIC abbreviations, codebooks, etc.
- `/scripts` is where we save processing scripts, each with its own README. NB: `data-cleaning.md` outlines our current idea of a plan and the scripts under `/scripts` have produced outputs in `/outputs`. THESE ARE JUST TESTS AND SUGGESTED APPROACHES, and must be scrutinized for efficiency and effectiveness, as well as best data science practices and academic rigor.

## Phase 0 Omni-Scan

See `docs/phase0_omniscan.md` for the comprehensive Omni-Scan workflow that builds the full feature universe, coverage ledger, and baseline analytics across all structured variables. After generating the cleaned wide CSV and long tables, run:

```bash
python -m scripts.cli omniscan \
  --wide-csv outputs/cleaned_wide.csv \
  --long-tables-dir outputs/long_tables \
  --output-dir outputs/analysis/omniscan
```

This produces feature metadata, nested-CV LightGBM/CatBoost benchmarks with SHAP/SAGE importances, specification curves, stability-selection probabilities, Model-X knockoff selections, CRT-based jurisdiction residual screens, bipartite network diagnostics, and risk-band parity tables.
