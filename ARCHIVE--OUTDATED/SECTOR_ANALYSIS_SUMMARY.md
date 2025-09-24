### Private vs Public Sector: Hypothesis Test Summary

Generated: 2025-09-21

#### Sample
- n (private/BUSINESS): 39
- n (public/PUBLIC_AUTHORITY): 25

#### Primary outcomes
- Fine amounts (EUR):
  - Mann–Whitney p=0.179; Cliff’s delta=-0.235 (95% CI -0.577, 0.117).
  - Median private €20,700 vs public €69,000; Δ median ≈ -€48,300 (95% CI -€85,100, -€4,600).
  - Sensitivity (excluding zero fines): p≈0.059.

- Fine likelihood (any fine vs none):
  - OR=1.99 (private vs public), 95% CI 0.68–5.84; p=0.209.

Interpretation: With this sample, private entities do not show statistically significant differences in fine amounts or fine likelihood at α=0.05, though medians suggest lower fines for private and the zero-excluded sensitivity is marginal.

#### Sanction mix (Fisher tests with BH-FDR)
- No sanction type differences remain significant after FDR (all q≥0.30).
  - Reprimand: p=0.050; q=0.301 (more common among private in this sample but not FDR-significant).

#### Key binary/context differences (Fisher tests with BH-FDR)
- Sensitive data involvement (`A16_SensitiveData`):
  - OR=9.78 (public vs private), 95% CI 2.97–32.24; p<0.001; q≈0.00019 (FDR-significant).
  - Public sector cases are far more likely to involve special-category data.
- Cross-border (`A5_CrossBorder`): p=0.080; q=0.120 (not significant after FDR).
- Data transfers (`A21_DataTransfers`): p=0.552; q=0.552.

#### Caveats
- Small sample sizes and skewed fines; interpret effect sizes with CIs.
- Exploratory scope with FDR control; confirmatory analyses should pre-register outcomes.

#### Suggested next steps
- Fit adjusted models:
  - log fine (EUR) ~ sector + cross-border + sensitive data + log subjects + negligence + cooperation + year.
  - fine likelihood ~ same covariates (logistic regression).
- Repeat sensitivity excluding zero fines and winsorizing top 1%.
- Add year fixed effects to control for temporal shifts.


