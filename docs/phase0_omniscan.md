# Phase 0 Omni-Scan Blueprint

The omni-scan pipeline provides the "big-eye" baseline required before building twin cohorts and causal designs.  It covers feature construction, predictive benchmarks, specification sweeps, fairness diagnostics, and network/risk-band perspectives.  The process is implemented in `scripts/analysis/omniscan.py` and exposed as a CLI command.

## Command

```bash
python -m scripts.cli omniscan \
  --wide-csv outputs/cleaned_wide.csv \
  --long-tables-dir outputs/long_tables \
  --output-dir outputs/analysis/omniscan
```

The command defaults to the cleaned-wide CSV and long tables produced by the ingestion pipeline and writes all artefacts under `outputs/analysis/omniscan/`.

## Feature Universe

1. **Load & Filter** – ingest the cleaned wide table, drop any decision containing an exclusivity conflict flag.
2. **Typed indicators** – preserve booleans as binary indicators and expand every `_status` field into one-hot indicators (e.g., `q53_powers_status__DISCUSSED`).
3. **Categorical expansion** – widen nominal columns (countries, DPAs, initiation type, etc.) via one-hot encoding with names like `country_code__DE`.
4. **Multi-select binaries** – pivot each long table (breach types, vulnerable groups, corrective powers, etc.) into decision-level indicator columns `table__TOKEN`.
5. **Metadata** – record the origin/source/kind for every feature and emit the catalogue as `features_universe.json`.

The coverage ledger (`coverage_ledger.csv`) reports per-feature observed rate, non-zero rate, variance, and extrema, while `no_feature_left_behind.md` summarises counts by source and highlights any columns with zero support.

## Baseline Models & Importances

For every outcome (fines, severity indices, and each corrective power option if available):

- Standardise inputs, run nested CV LightGBM (ROC-AUC or negative MSE scoring), and refit best model on full data.
- Fit CatBoost as a secondary gradient boosting baseline.
- Compute global SHAP importances, SHAP interaction effects, and SAGE values for the LightGBM model.
- Aggregate SHAP importances by legal block (rights, breach, Article 33/34, geography, mitigations, etc.) and output block-level weight tables.
- Produce jurisdiction-conditioned SHAP summaries (`country_shap_*.csv`, `dpa_shap_*.csv`).

Outputs: `baseline_metrics.csv`, `shap_importances.csv`, `shap_interactions.csv`, `block_importance.csv`, `sage_importance.csv`.

## Specification Curve & Stability Selection

- Grid over sector/time/country fixed-effect toggles and outcome winsorisation, fitting elastic-net (continuous) or logistic elastic-net (binary) models to produce `specification_curve.csv`.
- Bootstrap-based stability selection using L1-regularised logistic/linear models (100 resamples).  Feature selection probabilities per outcome are saved as `stability_<outcome>.csv`, with a 0.1 probability threshold feeding `robust_driver_list.json`.

## Model-X Knockoffs

Gaussian knockoffs are generated on the scaled design matrix for each outcome.  Logistic or lasso statistics are used depending on the outcome type, yielding FDR-controlled driver sets (`knockoff_selected.json`).

## Fairness Lens (CRT/DML)

- Cross-fitted gradient boosting residuals create orthogonalised outcomes and jurisdiction residuals.
- A permutation-based Conditional Randomisation Test checks correlation between outcome and jurisdiction residuals across 500 draws (Benjamini–Hochberg adjusted).  Results land in `crt_results.csv`.
- The same residuals feed a leniency/severity map with means, counts, standard deviations, and 95% CIs per country (`leniency_map.csv`).

## Network & Risk-Band Diagnostics

- Construct a feature–outcome bipartite graph weighted by mean absolute SHAP scores and detect Louvain communities (`bipartite_edges.csv`, `network_communities.csv`).
- Train a fine-positive risk model, bucket predictions into ventiles, and compute within-band country means plus pairwise KS statistics (`risk_band_parity.csv`).

## Dependencies

The omni-scan relies on the following Python packages:

- `pandas`, `numpy`
- `scikit-learn`
- `lightgbm`
- `catboost`
- `shap`
- `sage-importance`
- `knockpy`
- `networkx`
- `scipy`

These requirements are listed in `requirements-omniscan.txt` for reproducible environments.

## Outputs

Running the pipeline populates `outputs/analysis/omniscan/` with:

- `features_universe.json`
- `coverage_ledger.csv`
- `no_feature_left_behind.md`
- `baseline_metrics.csv`, `shap_importances.csv`, `shap_interactions.csv`, `block_importance.csv`, `sage_importance.csv`
- `country_shap_<outcome>.csv`, `dpa_shap_<outcome>.csv`
- `specification_curve.csv`
- `stability_<outcome>.csv`, `robust_driver_list.json`
- `knockoff_selected.json`
- `crt_results.csv`, `leniency_map.csv`
- `bipartite_edges.csv`, `network_communities.csv`
- `risk_band_parity.csv`

These artefacts satisfy the Phase 0 acceptance checks: coverage for all structured variables, global importances, stability/knockoff driver lists, CRT-based jurisdictional screening, community structure, and risk-band parity reporting.
