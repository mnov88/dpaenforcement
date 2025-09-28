# Evenness Pipeline — Interpretation & Policy Implications (2025-03-02)

## Diagnoses from Phases 1–2

1. **Uniform treatment is statistically rejected.** Conditional randomization tests yield vanishing p-values for both country and DPA effects across fines, confirming that observed outcomes still embed jurisdiction-specific signals after conditioning on facts.【F:research/evenness_execution_report.md†L15-L19】
2. **Severity gaps concentrate in specific authorities.** The country-level table highlights consistent outliers (e.g., GB leniency on euro fines but elevated positive rates, HU severity on log fines with lower positive probabilities), underscoring asymmetric enforcement postures that warrant supervisory review.【F:research/evenness_execution_report.md†L21-L41】
3. **Matched cohorts expose distributional failures.** Within CEM strata, certain country pairs (e.g., ES vs FR) display complete distribution shifts, signalling that even among tightly matched cases, sanction levels diverge materially.【F:research/evenness_execution_report.md†L42-L43】
4. **Calibration remains poor for predicted risk bands.** Early ventiles show zero realized fines despite non-zero predicted probabilities, implying that fact-only models overstate sanction likelihoods for low-risk clusters.【F:research/evenness_execution_report.md†L44-L45】

## Policy and Oversight Implications

* **Prioritise bilateral audits.** Country pairs with large explained gaps (e.g., ES–FR, GB–HU) should be queued for joint supervisory sessions to reconcile sanction frameworks and document mitigating considerations that justify divergences.【F:research/evenness_execution_report.md†L21-L43】
* **Revisit sanction calibration guidance.** Persistent over-prediction among low-risk ventiles suggests either under-enforcement or missing mitigating variables (e.g., cooperation, compliance history). Regulators should benchmark sanction matrices against observed outcomes to restore calibration credibility.【F:research/evenness_execution_report.md†L44-L45】
* **Institutionalise status-aware case matching.** The twin exports demonstrate balanced cohorts are feasible; embedding them into supervisory dashboards would allow DPAs to self-monitor parity against near-twin jurisdictions in real time.【F:research/evenness_execution_report.md†L5-L13】

## Technical Backlog (Phase 3)

* **Statsmodels regression incompatibilities.** Oaxaca–Blinder routines in statsmodels ≥0.14 drop the `weights` argument, blocking decomposition outputs; without resolution (downgrade or adapter shim), explained/unexplained gap attribution and policy lever estimation remain stalled.【F:research/evenness_execution_report.md†L47-L50】
* **Long-running robustness jobs.** Attempted monkey-patching of Oaxaca resolved the TypeError but led to prolonged runs that required manual termination, signalling the need for tighter convergence controls or smaller pilot cohorts before full recomputation.【F:research/evenness_execution_report.md†L47-L50】

## Immediate Next Steps

1. Patch or vendor-lock the Oaxaca interface (e.g., custom wrapper or pinning statsmodels 0.13.x within a Python 3.11 environment) and re-run Phase 3 end-to-end to populate decomposition and policy outputs.【F:research/evenness_execution_report.md†L47-L50】
2. Extend Phase 2 reporting with DPA-level heatmaps and narrative annotations drawn from the matched strata to surface human-readable explanations for major disparities.【F:research/evenness_execution_report.md†L21-L43】
3. Incorporate calibration diagnostics into supervisory briefings, including thresholds where predicted risk exceeds realized fines, to motivate policy harmonisation or data enrichment discussions.【F:research/evenness_execution_report.md†L44-L45】

