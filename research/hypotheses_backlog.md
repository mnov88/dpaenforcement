## GDPR Enforcement — Hypotheses & Analysis Backlog

Purpose: actionable, academically interesting hypotheses grounded in current outputs, with clear why, methods, and steps. Use DISCUSSED-only fields unless a hypothesis explicitly targets status effects. Regenerate artifacts with CLI; never hand-edit `outputs/`.

---

### 1) Rights taxonomy elasticity → sanction severity

**Why:** Moves beyond “any fine” to substantive mapping from violated principles/rights to sanction intensity; underexplored in GDPR literature.

**Hypothesis:** Controlling for facts and country/DPA, both the count and mix of rights/principles violated show a monotone, convex relationship with the enforcement severity index; security/lawfulness families exhibit higher elasticities.

**Inputs:** `outputs/cleaned_wide_latest.csv`, `outputs/long_tables_latest/rights_*`, `outputs/analysis/diagnostics/top_propensity_features.md`.

**Method:** Mixed-effects GLM/OLS with country and DPA effects; spline on rights counts; cluster-robust SE; SHAP sanity check.

**Steps:**
1. Aggregate rights long tables to per-case counts by family + total principles violated.
2. Fit severity ~ splines(n_principles_violated) + rights_families + controls + C(country)+C(DPA).
3. Export partial dependence with CIs and an elasticity table.

**Accept/Falsify:** Significant positive slope(s) after FDR; convexity holds. Placebo with permuted rights shows null.

---

### 2) Remedial depth moderates severity unevenly across DPAs

**Why:** Phase-three driver scans highlight remedial coverage; quantifying DPA-specific moderation informs harmonization.

**Hypothesis:** The marginal mitigation from deeper remedial actions differs across DPAs; lenient DPAs weight remedial depth more.

**Inputs:** `outputs/cleaned_wide_latest.csv`, `outputs/evenness/phase_three/{country_interactions.csv,dpa_interactions.csv}`.

**Method:** WLS with DPA-clustered SEs; interactions (DPA × remedial_depth); marginal-effects forest plot.

**Steps:**
1. Derive remedial depth (counts/intensity) from `q47_*`.
2. Fit severity ~ remedial_depth × C(DPA) + controls + weights.
3. Plot per-DPA marginal effects; BH-FDR adjust.

**Accept/Falsify:** Between-DPA variance > 0; ≥3 DPAs significant after FDR.

---

### 3) Sector disparities after conditioning on facts

**Why:** Early coefficients suggest ISIC-specific effects; policy relevance for proportionality.

**Hypothesis:** After controls and FEs, ISIC sections show heterogeneous severity residuals; finance/public administration differ from baseline.

**Inputs:** `outputs/cleaned_wide_latest.csv`, `exports/ml_ready/feature_metadata.json`.

**Method:** Mixed effects with ISIC fixed effects; FDR across sections.

**Steps:** Fit severity ~ facts + C(ISIC_section) + (1|country) + (1|DPA); export effects.

**Accept/Falsify:** ≥2 sections |z|>2.58 (FDR-corrected).

---

### 4) Channel selection bias (complaint vs breach notification vs ex officio)

**Why:** Initiation channel may proxy unobservables; matched designs clarify magnitude.

**Hypothesis:** Relative to complaints, ex officio increases severity; breach-notification differences attenuate within twins.

**Inputs:** `outputs/evenness/twins_*`, `outputs/cleaned_wide_latest.csv`, `outputs/analysis/diagnostics/case_initiation_summary.md`.

**Method:** Weighted matched-pairs (Gower/CEM) DID; McNemar/t-tests; multinomial GPS sensitivity.

**Steps:** Compute within-twin differences by channel; test and FDR-adjust.

**Accept/Falsify:** Significant ex officio lift within twins; attenuated vs unconditional gap.

---

### 5) “Status culture” predicts residual severity

**Why:** Jurisdictional DISCUSSED/NOT_MENTIONED patterns may proxy institutional diligence and shape outcomes.

**Hypothesis:** Higher status-intensity (share DISCUSSED across key facts/rights) predicts higher residual severity; NOT_MENTIONED prevalence predicts leniency.

**Inputs:** `outputs/analysis/stage1_data_brief.md`, `outputs/cleaned_wide_latest.csv` (status columns).

**Method:** Build status-intensity index; regress residual severity on index with country/DPA RE; correlate with jurisdiction residuals.

**Steps:** Construct index; fit models; compute correlations with `jurisdiction_effects.csv`.

**Accept/Falsify:** ρ>0.2 (p<0.05) between index and residual severity; robust to controls.

---

### 6) Cross-border coordination centrality and sanction intensity

**Why:** Networked enforcement (lead/co-concerned) may correlate with sanction posture.

**Hypothesis:** Higher centrality in cross-border networks associates with higher severity and multi-power usage.

**Inputs:** `exports/graph_data/cross_border_network.graphml`, `exports/graph_data/neo4j_import/*`, wide data.

**Method:** Compute network centrality; join to case/DPA; mixed effects.

**Steps:** Centrality metrics → merge → regress; export effects.

**Accept/Falsify:** Positive, FDR-significant centrality coefficients on severity or `q53_powers_*`.

---

### 7) Article constellation risks (Art. 32 with 33/34)

**Why:** Co-citation may capture compound duty failures with policy relevance.

**Hypothesis:** Art. 32 co-occurring with 33 or 34 predicts materially higher severity vs singular citations, beyond controls.

**Inputs:** `exports/graph_data/decision_article_bipartite.*`, `exports/graph_data/violation_cooccurrence.*`.

**Method:** Construct co-occurrence features; regress severity with FEs.

**Steps:** Engineer interactions; fit models; export table.

**Accept/Falsify:** Interaction (Art32 × (Art33∨Art34)) > 0, FDR-significant.

---

### 8) Temporal regime shifts and calibration

**Why:** Time-FE sensitivity suggests evolving enforcement; quantify period effects.

**Hypothesis:** Post-2021 periods show different calibration (Brier/log-bias) and altered timing penalties.

**Inputs:** `outputs/evenness/uniformity/calibration.csv`, decision dates in wide data.

**Method:** Split-period calibration panels; re-estimate timing RD by period.

**Steps:** Period-split; compute calibration metrics; re-run RD; compare.

**Accept/Falsify:** Significant period deltas; timing LATE differs by ≥0.1.

---

### 9) Subject-notification heterogeneity (vulnerable / special data)

**Why:** Aggregate AIPW is imprecise; targeted benefits may exist in high-risk strata.

**Hypothesis:** For vulnerable subjects and/or Art. 9 data, proper subject notification reduces fine probability ≥5 pp (ATT) with CIs excluding 0.

**Inputs:** wide data (Art34 required/fulfilled, vulnerable, Art9), Phase 1/2 outputs.

**Method:** AIPW/DoubleML within strata; overlap diagnostics.

**Steps:** Stratify; estimate ATT/ATE; check overlap and CIs.

**Accept/Falsify:** ATT reduction ≥5 pp with 95% CI excluding 0 in strata.

---

### 10) Calibration gaps as policy signal

**Why:** Low-risk ventiles over-predict fines; identify missing mitigators or under-enforcement pockets.

**Hypothesis:** Ventile miscalibration clusters align with missing mitigator proxies (cooperation/history) or specific DPAs.

**Inputs:** `outputs/evenness/uniformity/calibration.csv`, wide mitigations, DPA IDs.

**Method:** Ventile-level error decomposition by DPA and mitigations; random-effects meta-analysis.

**Steps:** Compute errors by ventile×DPA; regress on mitigations; extract DPA RE share.

**Accept/Falsify:** Significant association with mitigator proxies; DPA RE explains >10% variance.

---

## Quick-win priorities (2–5 hours)

1. Remedial depth × DPA marginal effects (Item 2)
2. Article constellation risks (Item 7)
3. Channel disparities within twins (Item 4)

## Repro commands

Refresh cleaned data:
```bash
python3 -m scripts.cli clean-wide \
  --input-csv analyzed-decisions/master-analyzed-data-unclean.csv \
  --out-csv outputs/cleaned_wide.csv \
  --validation-report outputs/validation_report.json
```

Evenness foundations and uniformity:
```bash
python -m scripts.evenness.cli phase-one
python -m scripts.evenness.cli phase-two --n-splits 3
```

Phase three (ensure statsmodels compatibility for decompositions/levers):
```bash
python -m scripts.evenness.cli phase-three --outcome fine_log1p
```

## Acceptance rubric

- Direction and significance consistent across baseline + robustness spec.
- FDR-adjusted p-values for multiple comparisons.
- Plots/tables saved under `outputs/analysis/` or `outputs/evenness/phase_three/` via scripts.



