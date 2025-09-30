# Sanction Driver Methodology

This memo documents the workflow used to analyse drivers of DPA sanction selection. The pipeline
is implemented in `scripts/analysis/sanction_driver_pipeline.py` and is designed to run on top of the
existing cleaned feature matrix (`outputs/analysis/feature_matrix.parquet`).

## 1. Environment and inputs
- Activate the project virtualenv before running (`source .venv/bin/activate`).
- Required inputs:
  - `outputs/analysis/feature_matrix.parquet` and `feature_matrix_metadata.json`.
  - Optional latent components in `outputs/analysis/interaction/latent_scores.parquet`.
- Outputs are written to `outputs/analysis/sanction_drivers/` (new files overwrite prior runs).

## 2. Feature engineering
- Features draw from structured Q1–Q68 indicators with status fields filtered out (`META_SUFFIXES`).
- Aggregated counts for breach types, mitigations, aggravating/mitigating factors, vulnerabilities and
  remedial actions are computed to capture contextual intensity while controlling leakage.
- Numeric predictors (violations, corrective measures, turnover, counts) are z-standardised.
- Categorical controls: `country_group` and `isic_section` are one-hot encoded with drop-first coding.
- Outcomes engineered from Article 58(2) flags:
  - Binary sanction types: `power_fine_flag`, `power_warning_flag`, `power_reprimand_flag`, `power_none_flag`.
  - Ordinal severity (`severity_rank`) re-derived from the power bundle when missing.
  - Fine amounts `fine_log1p` conditioned on fine-imposed cases.

## 3. Modelling stack
1. **Logistic GLMs** (Binomial with logit link) for each sanction flag.
   - Predictors: enforcement scale (`turnover_log1p`), culpability proxies (violations, corrective measures),
     procedural levers (Art. 33/34 flags, initiation channel), sensitive data flags, aggravating/mitigating
     themes, sector and region controls.
   - Marginal effects computed via `result.get_margeff()`.
   - Calibration curve & fairness tables produced for `power_fine_flag`.
2. **Ordinal logit** (statsmodels `OrderedModel`) for severity rank using the same predictor set.
3. **Conditional fine model**: OLS on `fine_log1p` restricted to fined observations with identical controls.
4. **Gradient boosting** (scikit-learn `GradientBoosting*`) for SHAP-based diagnostics across
   `power_fine_flag`, `power_warning_flag`, `fine_log1p`, and `severity_rank`.
5. **TreeSHAP** to compute global importance and interaction maps; stored as CSV/PNG/Parquet.
6. **Random-forest CATE** approximations: twin regressors estimate treatment effects for
   Art. 33 timeliness, breach/ex-officio triggers, mitigation cooperation, and sensitive data.
7. **Policy explorer**: counterfactual perturbations for a representative case using the fine logit.

## 4. Execution
Run the full stack after activating the virtualenv:

```bash
source .venv/bin/activate
python -m scripts.analysis.sanction_driver_pipeline
```

Runtime on a laptop (no GPU) is ~4 minutes and produces:
- Model summaries (`*_logit_summary.txt`, `ordinal_model_summary.txt`, `conditional_fine_summary.txt`).
- Coefficient and marginal-effect tables (`logit_coefficients_all.csv`, `*_marginal_effects.csv`).
- SHAP diagnostics under `outputs/analysis/sanction_drivers/shap/`.
- Calibration chart (`calibration_power_fine_flag.png`) and fairness ledger (`fairness_summary.csv`).
- Causal forest summary (`causal_forest_summary.csv`).
- Policy simulation table (`policy_scenarios.csv`).
- `pipeline_snapshot.json` with run metadata.

## 5. Reproducibility notes
- Perfect-separation warnings stem from rare categories (EU institutions, specific ISIC sections). Results are
  retained with caution; high-variance coefficients are flagged in the report.
- SHAP interaction matrices default to LightGBM-compatible formats; Gradient Boosting works with TreeSHAP
  without additional parameters.
- The pipeline is idempotent: successive runs refresh all artefacts in `outputs/analysis/sanction_drivers/`.
