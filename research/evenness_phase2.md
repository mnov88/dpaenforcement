# Evenness Phase 2: Uniformity Testing Workflow

## Purpose
Phase 2 evaluates whether GDPR enforcement outcomes are conditionally independent of jurisdiction once we control for the structured case facts assembled during Phase 1. The workflow operationalises the programme charter by combining doubly robust residual diagnostics, matched-pair disparity tests, distributional equality checks, calibration summaries, and hierarchical variance decomposition.

## Inputs
* `outputs/evenness/X_full.parquet` – Full cohort design matrix with country-year weights.
* `outputs/evenness/X_timeobs.parquet` – Time-observed subset with inverse-probability weights.
* Twin artefacts from Phase 1 (`twins_cem.parquet`, `twins_gower_within.parquet`, `twins_gower_cross.parquet`, `twins_riskbands.parquet`).

## Command
Run with the project environment activated:

```bash
python -m scripts.evenness.cli phase-two --n-splits 3
```

Arguments:
* `--n-splits`: number of cross-fitting folds for the nuisance models (defaults to 5; use 3 for faster iteration).
* `--random-state`: optional seed to reproduce fold assignments.

## Workflow Overview
1. **Feature preparation** – Drops outcome/status columns, retains numerical facts, and builds aligned weights for full and time-observed cohorts.
2. **Cross-fitted nuisance models** – Ridge (continuous) and logistic SGD (binary) models fitted via cross-fitting to obtain predictions and orthogonalised residuals for each supported outcome.
3. **Jurisdiction residual tests** – Weighted least squares with DPA-clustered covariance to estimate country and DPA residual effects, alongside joint Wald tests (Benjamini–Hochberg FDR applied).
4. **Paired disparity statistics** – Constructs weighted difference-in-differences across near-twin (Gower) and rule-based (CEM) pairings. Supports both continuous outcomes (t-tests with CIs) and binary outcomes (McNemar statistics with FDR control).
5. **Distributional equality diagnostics** – Within CEM strata and risk-band ventiles, compares outcome distributions across jurisdictions using Kolmogorov–Smirnov (asymptotic) and Earth Mover Distance. Reports weighted quantile contrasts (τ = 0.25/0.5/0.75/0.9).
6. **Calibration panels** – Assesses facts-only risk calibration across risk bands (mean observed vs predicted fine incidence and log-fine bias, plus Brier scores).
7. **Mixed-effects variance components** – Fits hierarchical models with country and DPA random intercepts plus random slopes on key drivers to extract intra-class correlations and slope variances.

## Outputs
All artefacts are written under `outputs/evenness/uniformity/`:

| File | Description |
| --- | --- |
| `residuals.parquet` | Cross-fitted predictions and residuals for each outcome, cohort tag, and jurisdiction. |
| `jurisdiction_effects.csv` | Residual mean effects (country & DPA) with standard errors, t-statistics, and FDR-adjusted p-values. |
| `joint_tests.csv` | Joint Wald/cluster-robust CRT statistics for each outcome and cohort. |
| `paired_tests.csv` | Weighted disparity summaries for each twin set, grouping level, and outcome. |
| `distribution_tests.csv` | KS/EMD tests within CEM strata and risk bands. |
| `quantile_contrasts.csv` | Weighted quantile comparisons (τ = 0.25/0.5/0.75/0.9). |
| `calibration.csv` | Risk-band calibration diagnostics (means, Brier, log-bias). |
| `variance_components.csv` | Random-effect variances and intra-class correlations for mixed-effects models. |

The CLI prints a completion summary showing row counts for the major artefacts.

## Diagnostics & Quality Checks
* Residual models drop jurisdictions with fewer than two residual observations to avoid singular covariance warnings.
* Paired statistics automatically fall back to reporting weighted mean differences (with NA intervals) when only a single pair is available, preventing `DescrStatsW` runtime warnings.
* KS tests are forced to the asymptotic method with SciPy runtime warnings suppressed for divide-by-zero cases.
* All p-values exposed in public tables include Benjamini–Hochberg FDR corrections for transparency.

## Interpretation Tips
* Use `jurisdiction_effects.csv` to build leniency/severity maps; effects are residualised (facts removed) and comparable across outcomes.
* `paired_tests.csv` surfaces direction and magnitude of disparities; inspect `net_up_minus_down` for binary outcomes to gauge asymmetry.
* `distribution_tests.csv` and `quantile_contrasts.csv` reveal whether entire distributions differ even when means align; focus on high ventiles to understand tail behaviour.
* `variance_components.csv` provides intra-class correlation coefficients indicating the share of variance attributable to countries vs DPAs.
* Calibration outputs highlight whether risk bands are well-calibrated; large Brier scores or log-bias values flag mismatches between predicted and observed severity.

## Runtime Notes
* Expect the full run to take several minutes; the heaviest sections are the distributional comparisons and mixed-effects modelling.
* The workflow now guards against the warning storms noted in the Phase 2 TODOs. If new jurisdictions appear with extremely low support, consider re-running Phase 1 to regenerate balanced twins before re-launching Phase 2.
