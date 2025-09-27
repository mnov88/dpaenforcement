# GDPR Enforcement Evenness Protocol

## Overview

This protocol operationalises the research goal of testing whether highly similar GDPR cases are treated evenly within and across jurisdictions, quantifying conditional disparities, and identifying institutional versus fact-driven drivers. It relies exclusively on structured questionnaire data covering 2018–2025 decisions and is designed to be reproducible via the CLI utilities under `scripts/evenness`.

## Data Foundations

- **Source**: `outputs/cleaned_wide_latest.csv` produced by the ingestion/cleaning pipeline in `scripts/cli.py`.
- **Scope**: cases with structured answers Q1–Q68, including derived measures (`fine_log1p`, `enforcement_severity_index`, `n_principles_*`, etc.).
- **Guardrails**:
  - Respect `*_coverage_status`, `*_status`, and `*_exclusivity_conflict` columns. Indicator columns are nulled when coverage ≠ `DISCUSSED` or conflicts are flagged, preventing false positives from schema echoes.
  - Treat `NOT_MENTIONED`, `NOT_APPLICABLE`, and `NONE` indicators as legally meaningful categories instead of NA.
  - Multi-select prefixes used in the facts vector: `q21_breach_types`, `q25_sensitive_data`, `q46_vuln`, `q47_remedial`, `q53_powers`.
  - Turnover variables (`turnover_*`) are only included through the selection-correction channel in robustness checks.
  - UK mapping is preserved; ES/IT skew is handled in robustness reweighting (country-year weights).

## Facts Vector (pre-outcome)

| Domain | Columns / Construction |
| --- | --- |
| Breach & specials | `breach_case`, indicators from `q21_*`, `q25_*` (status-aware). |
| Vulnerable groups | `q46_vuln_*` indicators (discussed-only). |
| Remedial actions | `q47_remedial_*` indicators. |
| Key powers | `q53_powers_*` indicators to observe severity signals. |
| Sector & size | `isic_section`, `isic_code`, `organization_size_tier` (from `q10`), `organization_type` (`raw_q8`). |
| Case origin | `case_origin` derived from `raw_q15`. |
| Time | `decision_year`, `decision_quarter`, `days_since_gdpr`, `decision_year_bucket` (2018–19 / 2020–21 / 2022–23 / 2024–25). |
| Geography | `country_code`, `dpa_name_canonical`, helper `country_year` for weights. |
| Controls | `n_principles_violated`, `n_corrective_measures`, `severity_measures_present`, `remedy_only_case`. |

Outcomes analysed: `fine_positive`, `fine_eur`, `fine_log1p`, `enforcement_severity_index`, and key corrective powers (from `q53_*`).

## Matching Strategy

1. **Exact / Coarsened Keys**
   - `breach_case`, special data indicators (Article 9/10/Neither), `q46_vuln_CHILDREN`, `decision_year_bucket`, `isic_section`, and (for cross-country) `organization_type`.
2. **Gower Nearest Neighbour Layer**
   - Numeric: `days_since_gdpr`, `n_principles_violated`, `n_corrective_measures`.
   - Categorical: within-country adds `organization_size_tier`, `organization_type`, `country_code`, `dpa_name_canonical`; cross-country excludes geography to permit international matches.
   - Calipers: 0.30 (within), 0.25 (cross). Up to 3 (within) or 5 (cross) nearest neighbours per stratum.
3. **Outputs & Diagnostics**
   - Matched edge lists saved to `outputs/evenness/matches_within.csv` and `.../matches_cross.csv`.
   - Standardised mean difference (SMD) balance charts produced via `plot_balance`.
   - Common support assessed by reviewing caliper-trimmed distances in the edge list.

## Conditional Disparity Modelling

1. **Baseline Models (facts-only plus jurisdiction fixed effects)**
   - Logistic GLM (`fine_positive ~ facts + C(country) + C(DPA)`), cluster-robust SEs by DPA.
   - Weighted least squares for `fine_log1p` and `enforcement_severity_index`.
2. **Mixed-Effects Specification**
   - Linear mixed model with random intercepts `(1 | DPA)` and variance components `(1 | country)`; optional random slopes on `q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY` and `q46_vuln_CHILDREN` to probe heterogeneous weighting of sensitive data/vulnerability.
3. **Interaction Scan**
   - Country × driver interactions tested individually, ranked by ΔAIC / LR statistic.
4. **Outputs**
   - JSON summaries under `outputs/evenness/models/` (`logistic.json`, `linear.json`, `mixed.json`).
   - Interaction scan CSV for top drivers.

## Leniency / Severity Index

1. Regress outcomes (`fine_log1p`) on facts-only covariates to generate residuals.
2. Fit partial pooling models:
   - `(1 + drivers | DPA)` for DPA-level residualisation.
   - `(1 | country)` for country-level pooling.
3. Export DPA and country posterior means with 95% CIs to `outputs/evenness/leniency_index.csv` and plot the map to `.../leniency_map.png`.
4. Variance components summarised (incl. ICC) via `outputs/evenness/variance_components.csv` and optional bar plot.

## Variance & Driver Attribution

- Intraclass correlations computed from mixed models (country vs. DPA vs. residual).
- Oaxaca–Blinder decomposition between selected country pairs (e.g., high-volume vs. low-volume) saved per comparison.
- Interaction scans highlight which factual drivers vary most across jurisdictions.

## Robustness Package

Executed with `python -m scripts.evenness.cli robustness`:

| Scenario | Adjustment |
| --- | --- |
| `reweight_country_year` | Equalises influence across country-year cells. |
| `heckman_turnover` | Applies control-function correction before introducing turnover. |
| `discussed_only` | Drops rows where required prefixes are not discussed. |
| `winsorize_fines` | Winsorises `fine_log1p` at the 99th percentile (and symmetrically). |
| `quantile_75` / `quantile_90` | Quantile regression for τ=0.75/0.90 on `fine_log1p`. |

All scenario outputs are JSON with model coefficients and audit notes placed under `outputs/evenness/robustness/`.

Additional manual checks recommended:
- Randomisation inference via shuffling `country_code` within match strata (edge list contains `stratum_key` for convenience).
- Tail diagnostics using `quantile` scenarios.
- Calibration of logistic model (`statsmodels` `GLMResults` supports `.predict()` for calibration curves).

## Predictive Cross-Check

- Gradient boosting models (classification for `fine_positive`, regression for `fine_log1p`) confirm non-linearities and interaction patterns.
- SHAP summaries exported as PNGs and JSON metrics saved alongside model runs.
- Serves as a validation of drivers identified in linear/mixed frameworks; not used for causal claims.

## Command Summary

```bash
# 1. Build fact matrix
python -m scripts.evenness.cli prepare-data --out outputs/evenness/fact_matrix.parquet

# 2. Build matches and review balance
python -m scripts.evenness.cli build-matches --balance-plot outputs/evenness/balance.png

# 3. Fit conditional models
python -m scripts.evenness.cli fit-models

# 4. Generate leniency map & variance components
python -m scripts.evenness.cli leniency-index --icc-plot outputs/evenness/icc.png

# 5. Oaxaca between two jurisdictions
python -m scripts.evenness.cli decompose --group-a FR --group-b DE

# 6. Robustness battery
python -m scripts.evenness.cli robustness

# 7. Predictive SHAP cross-check
python -m scripts.evenness.cli predictive --outcome fine_positive
```

## Deliverables

1. **Methodology & Tooling** – This protocol, CLI documentation, and configuration files (see `scripts/evenness` package).
2. **Reproducible Environment** – `research/evenness_environment.yml` capturing Python dependencies.
3. **Execution Artefacts** – Fact matrix, match lists, model summaries, leniency map, variance tables, decomposition outputs, robustness JSONs, and SHAP plots under `outputs/evenness/` (created by CLI commands above).
4. **Insights Report Template** – See `research/evenness_insights_report.md` (companion document) for structuring final findings (evenness verdicts, marginal effects, random-effect profiles, policy simulations).

## Notes

- All scripts default to reading the latest cleaned wide CSV; override `--wide-csv` to point to frozen snapshots.
- The CLI honours guardrails automatically; no manual filtering of status columns is required.
- Caliper and neighbour counts are exposed via `scripts/evenness/config.py` for experiment tuning.
- For reproducibility, set `PYTHONHASHSEED` and supply a `--random-state` flag (future enhancement) if deterministic sampling is required beyond current deterministic operations.
