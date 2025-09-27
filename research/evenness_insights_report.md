# GDPR Enforcement Evenness – Insights Report Template

## Executive Summary
- **Headline verdicts** on evenness within and across jurisdictions.
- Key quantitative findings (effect sizes, variance shares, uncertainty intervals).
- Policy-relevant recommendations based on simulated what-if adjustments.

## Data & Matching Diagnostics
- Dataset snapshot (time span, number of decisions, coverage filters applied).
- Matching coverage table (within-country vs. cross-country pairs, average distances).
- Balance diagnostics (embed/export SMD chart) and notes on common support.

## Conditional Disparity Results
- Logistic model marginal effects on `fine_positive` (with clustered CIs).
- OLS / mixed-model coefficients for `fine_log1p` and `enforcement_severity_index`.
- Highlight heterogeneity revealed by random slopes or interaction scans (country × driver effects).

## Leniency / Severity Profiles
- Leniency map (DPA-level residuals with CIs) and key outliers.
- Country-level shrinkage estimates and variance decomposition (ICC chart).
- Discussion of whether disparities are fact-driven or institutional.

## Driver Attribution & Decomposition
- Oaxaca–Blinder summaries for focal country pairs (explained vs. unexplained gaps).
- Ranking of top discrepancy drivers (from interaction scan / SHAP interactions).
- Narrative on vulnerable groups, sensitive data, and remedial action weighting.

## Policy What-If Explorer
- Simulated outcome shifts when toggling specific drivers (e.g., removing vulnerable group, switching initiation channel).
- Stress tests for turnover inclusion using the selection-corrected scenario.
- Cross-jurisdiction benchmarking: how lenient/severe jurisdictions converge under harmonised fact patterns.

## Robustness Checks
- Table summarising each robustness scenario, model shifts, and qualitative verdict (stable / notable change).
- Notes on calibration, randomisation inference, and tail behaviour.

## Appendix
- Methodological references (link back to `research/evenness_analysis_protocol.md`).
- CLI command log and environment hash (`conda env export` or `pip freeze`).
- Data caveats (missing statuses, exclusivity conflicts, language issues).
