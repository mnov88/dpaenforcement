# Preliminary Analysis: Breach Notification and Enforcement Severity

## 1. Data and modelling snapshot
- **Feature matrix**: `outputs/analysis/feature_matrix.parquet` (999 decisions) with both aggregate counts and per-token indicators for multi-select questions, including Article 58(2) corrective powers and case-initiation channels.
- **Severity model**: Regularised multinomial logistic classifier (`scripts/analysis/hierarchical_severity_regularized.py`) yielding `severity_predictions.csv` and ranked coefficients (`top_coefficients.csv`). Observed classes: `NONE`, `REMEDIAL_ONLY`, `FINE_ONLY`, `FINE_PLUS`.
- **Notification model**: Propensity-trimmed AIPW workflow (`scripts/analysis/joint_notification_sanction.py`) using latent PCA scores, aggregated features, and individual power indicators; outputs `joint_model_metrics.json`, `propensity_diagnostics.json`, and `propensity_top_coefficients.csv`.

## 2. Descriptive findings
- **Severity distribution**: 195 cases predicted `NONE`, 203 `REMEDIAL_ONLY`, 338 `FINE_ONLY`, 263 `FINE_PLUS`.
- **Expected ranks**: Mean severity ranges from 0.02 (`NONE`) to 3.97 (`FINE_PLUS`), confirming the classifier’s ordinal behaviour under penalisation.
- **Initiation mix** (Q15): 745 complaints, 92 breach notifications, 58 ex officio cases, 20 referrals, and a pooled `LOW_FREQUENCY` bucket (media, joint investigations, follow-ups, other) covering 12 decisions. Breach notifications deliver the highest mean log fines (7.46) and severity expectations (≈3.04); ex officio cases follow (6.83 / 2.69), while the pooled low-frequency bucket drops to 5.96 / 1.67. Complaint-driven cases sit near 6.22 log fines and 2.68 in expected severity.
- **Jurisdictional intensity**: Romanian and Catalan authorities continue to average >3.8 in expected severity, reflecting consistent enforcement posture.
- **Propensity overlap**: Trimming retains 50 of 101 Art. 33-required cases (21 timely, 29 late) within the 0.10–0.90 band (mean 0.40, σ 0.23).

## 3. Hypotheses
1. **Notification timeliness and fines**: Timely notification (≤72 h) may alter expected log fines after conditioning on breach context, initiation channel, and DPA heterogeneity.
2. **Rights complexity as notification driver**: Latent rights-discussed/violated components correlate with timely filings, signalling richer subject-right impacts.
3. **Breach scope and enforcement severity**: Multisector breaches and heavier corrective measures raise the probability of `FINE_PLUS` outcomes, modulated by each DPA’s baseline severity.
4. **Initiation channel and sanctions**: Ex officio, breach-notification, and media-triggered investigations generate higher fines/severity than complaint-only cases.

## 4. Model evidence
- **Hypothesis 1**: The trimmed AIPW estimate is +1.40 log-fine units (≈+304% fines), versus a naïve difference of +0.56. Timeliness now coincides with higher sanctions once initiation and corrective powers enter the model—suggesting self-reported or high-profile breaches are severe enough to draw rapid disclosures *and* heavier remedies. The sign reversal relative to earlier runs underscores sensitivity.
- **Bootstrap diagnostics**: 300 bootstrap replicates yield a mean AIPW of +1.42 with a very wide 95% CI (−10.9, +23.4) under the 0.10–0.90 trim, reflecting extreme leverage from a handful of severe cases. A tighter 0.20–0.80 trim produces a mean of +0.12 (CI −3.35, +3.04) across 36 cases, indicating the effect is not statistically distinguishable from zero once overlap improves. Further work should stabilise the estimator before drawing policy conclusions.
- **Hypothesis 2**: Propensity coefficients remain dominated by `rights_violated_pc2` (+0.91) and `rights_discussed_pc1` (+0.67), while `dpa_severity_shrinkage` (−0.82) and `q53_powers_REPRIMAND` (−0.56) reduce the odds of timeliness. The presence of `q53_powers_NONE` (+0.59) indicates lighter corrective portfolios accompany quicker filings.
- **Hypothesis 3**: Class-4 severity drivers are led by `q53_powers_ADMINISTRATIVE_FINE` (+1.68), `n_corrective_measures` (+1.50), and additional powers (`COMPLY_WITH_DATA_SUBJECT_REQUESTS`, `BRING_PROCESSING_INTO_COMPLIANCE`, `WARNING`). This reinforces the link between broad remedy suites and high severity scores.
- **Hypothesis 4**: After pooling scarce categories, breach notifications (7.46 log fines, 3.04 severity) and ex officio investigations (6.83 / 2.69) still dominate complaints (6.22 / 2.68); the pooled `LOW_FREQUENCY` bucket falls to 5.96 / 1.67 and referrals remain at 3.92 / 2.10. Proactive or high-visibility triggers therefore retain their association with harsher sanctions.

## 5. Limitations & next steps
- No `WARNING_REPRIMAND` class observed yet; the severity model effectively spans three ordinal tiers pending data review.
- AIPW results now include bootstrap intervals, which remain wide under the default trim (mean ≈0.13, 95 % CI −15.8 to 13.8); future work should incorporate overlap diagnostics into the trimming strategy (or adopt targeted shrinkage) before reporting definitive effects. Under a tighter 0.20–0.80 trim the bootstrap mean drops to ≈−0.04 with CI (−3.12, 2.79), underscoring the need for more overlap before causal claims.
- Initiation categories with very small counts need pooling or Bayesian shrinkage before strong claims.
- Qualitative inspection of latent PCA directions (rights/access issues) would bolster interpretability and ensure the components align with doctrinal narratives.

Outputs referenced: `outputs/analysis/hierarchical_severity_regularized/*`, `outputs/analysis/joint_notification/*`, `outputs/analysis/interaction/*`, `outputs/analysis/diagnostics/*`.

### Note on corrective powers (Q53)
All Article 58(2) tokens continue to feed the models alongside aggregate “count/any” summaries. Individual powers now appear explicitly among the dominant coefficients (e.g., fines, warnings, compliance orders), offering granular levers for enforcement storytelling while preserving the stability benefits of aggregated metrics.
