# Breach Notification Modelling Blueprint

This blueprint documents the modelling scaffolds that will consume `outputs/analysis/feature_matrix.parquet` for multivariate breach notification studies.

## 1. Baseline diagnostics
- Load the feature matrix and confirm `art33_*`, `art34_*`, and the full `q53_powers_*` indicator set are populated for every decision.
- Generate summary tables (counts, proportions, cross-tabs) for enforcement powers, notification decisions, and severity outcomes; persist under `outputs/analysis/diagnostics/`.
- Validate overlap for causal estimators by inspecting propensity score distributions across strata defined by organisational class (`q10_org_class_*`), breach vectors (`q21_breach_types_*`), mitigation actions (`q28_mitigations_*`), and vulnerability flags (`q46_vuln_*`).

## 2. Hierarchical ordinal / multinomial severity models
- Treat the exercised powers (`q53_powers_*`) as the outcome family; encode mutually exclusive states (warning, reprimand, fine, none, combinations) leveraging the derived columns (`power_*_flag`).
- Specify multilevel regressions with random intercepts and slopes for `dpa_name_canonical` and `country_group` to capture institutional heterogeneity.
- Include notification levers (`art33_*`, `art34_*`, `subjects_notified_flag`) and breach context (`q21_*`, `q25_*`, `q47_*`) as covariates; evaluate model fit via WAIC/LOO when estimating in a Bayesian framework.

## 3. Joint treatment-outcome estimation
- Build simultaneous models for (a) timeliness (`art33_late_flag`) and (b) sanction magnitude (fine indicators, `fine_log1p`) using seemingly unrelated regression or structural equation modelling.
- Implement doubly-robust estimators (AIPW, causal forest) fed by the feature matrix to estimate combined treatment effects of timely Art. 33 notification, subject notification, and remedial actions.
- Report sensitivity analyses (Rosenbaum bounds, leave-one-DPA-out re-estimation) to quantify robustness of causal claims.

## 4. Interaction and latent structure analysis
- Construct co-occurrence networks for multi-select tokens, starting with breach types, vulnerabilities, remedial actions, and exercised powers; detect communities to surface bundled risk patterns.
- Apply factor analysis or topic models to rights discussed/violated (`q56_*`, `q57_*`) and access/ADM issues (`q58_*`, `q59_*`) to derive latent constructs moderating enforcement severity.
- Integrate latent scores into the hierarchical and causal models, re-running diagnostics to ensure stability.

## 5. Reproducibility & outputs
- Version any modelling scripts under `scripts/analysis/` with CLI entry points that accept the feature matrix path and emit artefacts to `outputs/analysis/`.
- After each modelling iteration, regenerate `phase3completed.patch` to capture updated evenness artefacts alongside new diagnostic exports.
- Document findings in dedicated research memos, highlighting how power combinations interact with notification behaviour across the full Q1–Q68 covariate space.
