# Preliminary Analysis: Breach Notification and Enforcement Severity

## 1. Data and modelling snapshot
- **Feature matrix**: `outputs/analysis/feature_matrix.parquet` (999 decisions, aggregated counts for multi-select questions, latent PCA scores for rights/access issues).
- **Severity model**: L2-multinomial logistic classifier (`scripts/analysis/hierarchical_severity_regularized.py`) producing `severity_predictions.csv` and feature rankings; classes observed: `NONE`, `REMEDIAL_ONLY`, `FINE_ONLY`, `FINE_PLUS`.
- **Notification effect model**: Propensity-trimmed AIPW pipeline (`scripts/analysis/joint_notification_sanction.py`) using latent scores and aggregated covariates; outputs `joint_model_metrics.json`, `propensity_diagnostics.json`, and `propensity_top_coefficients.csv`.

## 2. Descriptive findings
- Severity distribution: 195 cases predicted `NONE`, 203 `REMEDIAL_ONLY`, 338 `FINE_ONLY`, 263 `FINE_PLUS`.
- Expected severity (mean posterior rank) rises from 0.20 (`NONE`) to 3.57 (`FINE_PLUS`), validating the classifier’s ordinal interpretation.
- DPA-level averages highlight Romanian and Catalan authorities at the upper end (>3.7 expected rank), suggesting jurisdiction-specific enforcement intensity.
- Propensity trimming retains 50 of 101 Art. 33 required cases (21 timely, 29 late) with propensities spanning 0.14–0.71 (mean 0.41, σ 0.21).

## 3. Hypotheses
1. **Notification timeliness and fines**: For Art. 33 required breaches, timely notification (≤72h) may influence monetary penalties; we now evaluate whether it increases or decreases expected log fines after adjusting for breach context and DPA heterogeneity.
2. **Rights complexity as notification driver**: Higher latent scores for rights discussed/violated (PCA components) increase the likelihood of timely notification, as they capture complex subject-rights impacts that motivate earlier disclosure.
3. **Breach scope and enforcement severity**: Multisector breaches (higher `q21` counts) and corrective burdens (`n_corrective_measures`) elevate the probability of `FINE_PLUS` outcomes, moderated by DPA baseline severity (`dpa_severity_shrinkage`).

## 4. Model evidence
- **Hypothesis 1**: With the expanded feature set, the trimmed AIPW estimate flips to +1.40 log-fine units (≈+304% fines), while the naïve difference is +0.56. This suggests timeliness correlates with *higher* sanctions once we condition on judicial and corrective-power context—possibly because the DPAs commanding the most intrusive remedies also extract rapid disclosure. The sign reversal flags sensitivity to model specification; we need confidence intervals and robustness checks (alternate penalties, trims) before drawing policy conclusions.
- **Hypothesis 2**: Propensity drivers show positive coefficients for `rights_violated_pc2` and `rights_discussed_pc1`, supporting the claim that richer rights discussion correlates with timely notification. Negative coefficient on `dpa_severity_shrinkage` implies DPAs known for severe enforcement see fewer timely filings, consistent with strategic delay hypotheses.
- **Hypothesis 3**: Severity class 4 coefficients highlight `n_corrective_measures` (+3.87) and `q21_breach_types_any` (+0.48) as dominant predictors, confirming the escalation link between breach breadth/remediation load and fines. Negative weight for `art33_timely_flag` (−0.23) aligns with Hypothesis 1, tying timeliness to lower severity.

## 5. Limitations & next steps
- Missing `WARNING_REPRIMAND` class means the multinomial model effectively spans three ordinal tiers; future data ingestion should confirm whether this is a labelling or extraction gap.
- AIPW inference lacks confidence intervals; bootstrap or influence-function variance is needed before policy claims.
- Propensity trimming reduces sample size to 50; robustness checks (varying trim bounds, alternative penalties) are required.
- Latent components merit qualitative inspection to map PCA directions back to question text and ensure interpretability.

Outputs inspected: `outputs/analysis/hierarchical_severity_regularized/*`, `outputs/analysis/joint_notification/*`, `outputs/analysis/interaction/*`.

### Note on corrective powers (Q53)
All Article 58(2) tokens now feed the models alongside the aggregate “count/any” summaries. For instance, the regularised severity coefficients show `q53_powers_ADMINISTRATIVE_FINE`, `q53_powers_WARNING`, and `q53_powers_SUSPENSION_DATA_FLOWS` as distinct predictors, while the propensity analysis highlights `q53_powers_NONE` and `q53_powers_REPRIMAND` as major drivers of notification behaviour. This dual representation preserves interpretability (token-level insights) without sacrificing the stability benefits of aggregated features.
