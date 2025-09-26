# Sanctions Drivers Analysis

## Data
Cleaned dataset from outputs/cleaned_wide.csv with long tables for measures.

## Methods
Multilevel-ready features; baseline logistic for admin fine; gradient boosting with SHAP if available.

## Results (summary)
- Logistic AUC (admin fine): 0.9260075772421451
- Standardized effects (top):
  - severity_measures_present: 3.381
  - turnover_log1p: 1.012
  - n_principles_violated: 0.411
  - n_principles_discussed: -0.227
  - breach_case: -0.173
  - n_corrective_measures: -0.070
