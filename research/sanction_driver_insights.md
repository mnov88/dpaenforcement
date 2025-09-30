# GDPR Sanction Driver Insights

_Date:_ 2025-09-30  |  _Pipeline:_ `python -m scripts.analysis.sanction_driver_pipeline`

## 1. Model diagnostics
- Fined cases represent 57% of the 1,998-decision feature matrix; warning-only cases 21%, reprimands 33%, and remedy-only/no-action the remainder.
- Logistic classifiers for sanction flags achieve AUC 0.97 (`outputs/analysis/sanction_drivers/gb_power_fine_flag_metrics.json`) and 0.88 for warnings. Gradient-boosted SHAP profiles corroborate the GLM effect ordering.
- Ordinal logit convergence succeeded with 1,998 observations; conditional fine OLS fits 1,138 fined matters (R²=0.41 per `conditional_fine_summary.txt`).

## 2. Drivers of administrative fines
- **Scale & gravity dominate**:
  - Each s.d. rise in corrective measures increases fine probability by +6.5pp (`power_fine_flag_marginal_effects.csv`).
  - Turnover (log) adds +10.5pp; every additional principle violated adds +5.2pp; additional breach types (+3.9pp).
- **Aggravating factors**:
  - Explicit negligence references (+16.2pp) and prior infringement history (+15.9pp) are the strongest qualitative levers.
  - Financial-vulnerability mitigations are rarely protective; employee-specific cases add +6.8pp.
- **Sensitive data nuance**:
  - Article 10 (criminal) processing reduces fine probability by −29pp, mirroring DPAs' hesitancy to escalate without lawful basis clarity (often ending in reprimands or referrals).
- **Regional control**: EU-based cases show +20pp higher sanction propensity than non-EU after covariate adjustment, reflecting heavier caseload concentration.
- Calibration remains tight (Brier 0.11, `calibration_metrics.json`), though certain ISIC sections (manufacturing, transport) sit near 95% predicted sanction rates (`fairness_summary.csv`).

## 3. Warning & no-sanction patterns
- Warning logit stresses **corrective depth** ( +6.1pp ) but ex-officio launches reduce warning likelihood (−7.9pp), signalling DPAs escalate to fines once they self-initiate major probes.
- Children-focused cases obtain +4.0pp warning probability, often alongside mandated remedial programmes.
- No-sanction models lack significant predictors; the outcome remains sparse in modern enforcement, typically tied to procedural dismissals.

## 4. Sanction severity and fine amounts
- Ordinal logit (`ordinal_coefficients.csv`) highlights the severity ladder:
  - Corrective-measure count (+2.32 log-odds) and negligence language (+1.27) sharply shift matters into fine-plus tiers.
  - Breach notifications (+0.78) correlate with more extensive orders, reflecting high-profile incidents.
  - Criminal-data cases (−1.91) frequently cap severity, aligning with the lower sanction propensity above.
- Conditional fine OLS (`conditional_fine_coefficients.csv`) shows:
  - Turnover elasticity: +0.53 log fine per s.d.
  - Ex-officio investigations (+0.86 log points) and financially vulnerable groups (+1.48) drive fine size even after sanction selection.
  - Additional aggravating factor mentions (+0.39) increase fines, while mitigation cooperation trims them (−0.35).

## 5. Non-linear insights (SHAP)
- Fine probability SHAP importances (`shap/power_fine_flag_shap_importance.csv`) confirm the GLM ordering: corrective measures, aggravating-factor count, negligence tokens, and turnover dominate, followed by discussion breadth and employee focus.
- Severity SHAP maps emphasise corrective breadth, aggravating totals, and mitigation cooperation, while latent Art. 33 submission signals contribute moderately.
- Warning SHAP surfaces sectoral clustering (public administration, transport) and ex-officio dampening, matching marginal effects.

## 6. Causal forest stress tests
- Uplift estimates (`causal_forest_summary.csv`):
  - Breach-notification triggers add +0.31 sanction probability (IQR 0.53) relative to complaints, even after conditioning on covariates.
  - Ex-officio cases show neutral-to-slightly negative uplift (−0.04), consistent with escalations to corrective remedies rather than fines alone.
  - Cooperation with the DPA during mitigation yields +0.07, indicating post-incident engagement coexists with sanction issuance (DPAs may both sanction and monitor compliance).

## 7. Fairness and sector focus
- Predicted vs. actual fine shares by country group stay within ±1pp for EU, EEA-non EU, and non-EEA clusters (`fairness_summary.csv`).
- Sector disparities persist: manufacturing (ISIC C) and information (J) average >0.9 predicted sanction rates, whereas public administration (O) averages 0.45, aligning with DPAs’ tendency to prescribe corrective orders for public bodies.

## 8. Policy what-ifs
- Counterfactual probes (`policy_scenarios.csv`) show high base sanction probability (≈0.99) for the representative case. Piling on aggravating language pushes fines near certainty (0.998), whereas timeliness or breach-notification toggles barely dent the outcome. Interpretation: once substantive violations and aggravating histories accumulate, procedural levers cannot pull the case below the fine threshold.

## 9. Key takeaways
1. **Procedural narratives matter but cannot replace substantive breaches**: negligence language, prior infringements, and employee exposure dominate sanction selection, overshadowing notification timeliness.
2. **Mitigation cooperation shifts severity rather than selection**: cooperation raises fine probability slightly yet reduces fine magnitude, indicating DPAs balance deterrence with remedial oversight.
3. **Sectoral parity gaps** emerge in capital-intensive industries—manufacturing and transport face near-certain fines, suggesting targeted compliance support may be warranted.
4. **Criminal-data handling remains an exception**: when investigations focus on Article 10 data, DPAs often de-escalate to remedial/warning outcomes unless additional violations pile up.

Supporting artefacts live under `outputs/analysis/sanction_drivers/`. Use them alongside SHAP plots for presentation-ready visuals.
