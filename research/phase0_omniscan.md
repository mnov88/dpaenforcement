# Phase 0 – Omni-Scan Implementation Notes

This document summarises the tooling delivered for the omni-scan stage of the GDPR evenness programme. The goal is to cover **all** structured decision facts, establish baseline predictive diagnostics, and surface early signals of conditional non-uniformity before building factual twins.

## Feature universe

- The omni-scan loader reads the cleaned wide dataset and harmonises jurisdiction codes (`UK → GB`, `EL → GR`).
- Any record flagged with `*_exclusivity_conflict=1` is excluded from modelling support to avoid contradictory annotations.
- For each `*_status`/`*_coverage_status` column we:
  - Preserve `DISCUSSED`, `NOT_MENTIONED`, `NOT_APPLICABLE`, and `MISSING` states as one-hot indicators.
  - Mask underlying indicators or numeric values unless the status is `DISCUSSED`.
- Categorical responses (including organisation size/type, ISIC, case origin, Article 33/34 flags, cross-border status) are expanded to binaries. Numeric facts retain floating point representations.
- A JSON catalogue (`features_universe.json`) logs every engineered feature with its source column, datatype, and legal block (breach facts, sensitive data, mitigations, jurisdiction, temporal, etc.).
- The coverage ledger records per-feature observation rates, mean/variance, and the upstream status mix to provide the "no-feature-left-behind" checklist.

## Predictive baselines

- LightGBM (or CatBoost / HistGradientBoosting as fallback) runs nested 5×5 CV across all outcomes: fines (`fine_positive`, `fine_log1p`, `fine_eur`), `enforcement_severity_index`, and every `q53_powers_*` indicator.
- SHAP summaries and interaction values feed the feature × outcome heatmap and the top-interaction catalogue. Block-level importances aggregate SHAP weight by legal family.
- SAGE importances are approximated through loss-based permutation deltas with classification handled on predicted probabilities.
- Jurisdiction-conditioned SHAP averages capture which features drive predictions within each country/DPA.

## Specification stability

- Penalised GLMs sweep fixed-effect toggles (country/DPA), sector dummies, temporal controls, winsorisation, and penalty type (lasso vs elastic net). Results are collated in `specification_curve.csv`.
- Stability selection repeatedly subsamples the data and re-fits lasso/elastic-net models to derive selection probabilities. Model-X knockoff filters (with synthetic knockoffs) control FDR at 10% and yield the robust driver list.

## High-dimensional fairness checks

- Cross-fitted residuals from boosted models feed conditional randomisation tests for both country and DPA effects. P-values (FDR ready) and effect estimates are stored in `crt_results.csv` and `jurisdiction_effects.csv`.
- Residual means plus 95% CIs provide the preliminary leniency/severity map, while heterogeneity markers highlight robust drivers implicated in disparities.

## Network & distributional diagnostics

- Feature–outcome importance edges form a bipartite graph; Louvain clustering surfaces communities for the constellation map.
- Model risk scores produce ventile risk bands. For each band, the workflow reports mean outcomes by jurisdiction and runs KS/Wasserstein contrasts across high-volume countries (`risk_band_parity.csv`, `risk_band_distribution.csv`).

All artefacts live under `outputs/evenness/omniscan/`. Use `python -m scripts.evenness.cli phase-zero` after generating `outputs/cleaned_wide_latest.csv` to refresh the full omni-scan suite.
