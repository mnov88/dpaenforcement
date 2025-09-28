# Phase 3 – Explanation, Policy Levers & Robustness Synthesis

This note documents the implementation of Phase 3 for the GDPR enforcement evenness study. Building on the facts-only design matrices and uniformity diagnostics from Phases 1 and 2, the final phase explains observed disparities, quantifies policy-controllable levers, and assembles an auditable toolkit for decision-makers.

## Inputs

* `outputs/evenness/X_full.parquet` and `X_timeobs.parquet` – design matrices from Phase 1.
* Twin artefacts (`twins_cem.parquet`, `twins_gower_within.parquet`, `twins_gower_cross.parquet`, `twins_riskbands.parquet`).
* Phase 2 uniformity outputs under `outputs/evenness/uniformity/` (residuals, disparity tables, calibration panels).

## Command

Run the full workflow (after activating the project environment):

```bash
python -m scripts.evenness.cli phase-three --outcome fine_log1p
```

### Key Flags

* `--outcome` (default `fine_log1p`): outcome used for the interaction scan/driver ranking. Policy levers are always estimated on both `fine_positive` and `fine_log1p`.

## Workflow Components

1. **Driver Attribution** – Builds a facts-only regression (`outcome ~ facts + C(country) + C(DPA)`) and scans country × driver and DPA × driver interactions. ΔAIC, likelihood-ratio statistics, and FDR-adjusted p-values identify jurisdictions that weight facts differently.
2. **Gap Decomposition** – Runs Oaxaca–Blinder decompositions for the highest-volume country pairs (top six combinations), reporting explained vs. unexplained shares with standard errors and cohort sizes.
3. **Policy Levers** –
   * *72-hour notification timing*: local-linear quasi-RD using `art33_delay_amount` with automatic centring and triangular kernel weighting. Outputs reduced-form and local ATE estimates alongside placebo cut-offs.
   * *Subject notification*: AIPW/DoubleML estimator restricted to `art34_required=YES` cases with overlap diagnostics (min/max propensity, effective sample size) and both ATE/ATT effect sizes.
   * Plots effect magnitudes and stores placebo summaries.
4. **Randomisation Inference** – Permutes jurisdiction labels within CEM strata to validate whether observed within-stratum disparities exceed chance under the null.
5. **Robustness Suite** – Reuses the scenarios defined in `scripts/evenness/config.py` (country-year reweighting, turnover selection correction, discussed-only filter, winsorisation, quantile regressions) and consolidates parameter movements into a single summary table.
6. **Reporting & Playbook** – Renders a human-readable insights report and a succinct “playbook” enumerating priority drivers, lever magnitudes, and robustness verdicts. Captures the Python environment snapshot to support reproducibility audits.

## Outputs

All artefacts are saved under `outputs/evenness/phase_three/`:

| File | Description |
| --- | --- |
| `driver_leaderboard.csv` | Combined country/DPA interaction rankings with FDR-adjusted p-values. |
| `country_interactions.csv`, `dpa_interactions.csv` | Raw interaction scan tables per jurisdictional level. |
| `decomposition_summary.csv` | Oaxaca–Blinder decomposition results (effect splits, standard errors, sample sizes). |
| `policy/lever_estimates.csv` | Policy lever estimates (reduced-form, local ATE, AIPW ATE/ATT) with diagnostics. |
| `policy/rd_placebos.csv` | Placebo cut-off checks for the timing quasi-RD. |
| `policy/lever_effects.png` | Visual summary of ATE/LATE estimates with 95% confidence intervals. |
| `randomization_inference.csv` | Permutation test statistics and p-values. |
| `robustness_summary.csv` | Compact summary of robustness scenarios (type, notes, key parameter shifts). |
| `insights_report.md` | Narrative synthesis covering drivers, decompositions, levers, and robustness verdicts. |
| `playbook.md` | Actionable checklist for policy teams (top drivers + lever magnitudes). |
| `environment.txt` | Captured Python environment (version + `pip list`) for audit trails. |

## Diagnostics & Acceptance Criteria

* Interaction scans fail gracefully when a term is absent, and all p-values are Benjamini–Hochberg adjusted.
* Decompositions include uncertainty (standard errors) and sample sizes; country pairs are selected by observed volume.
* RD outputs include first-stage jumps and placebo offsets to validate bandwidth sensitivity. Notification effects report overlap diagnostics and ATT/ATE values.
* Randomisation inference uses 200 permutations with deterministic seeding for reproducibility.
* Robustness summary highlights shifts in sensitive-data and vulnerable-group coefficients.
* Reports note weighting, bandwidth, and tail-handling choices alongside environment details, satisfying the reproducibility requirement.

## Usage Notes

* The CLI reuses Phase 1 facts; re-run `phase-one` if new columns or cohorts are introduced.
* If RD windows lack support, the module automatically skips the estimate and logs the empty rows—review placebo outputs to confirm support.
* The environment snapshot is generated via `python -m pip list`; ensure the virtual environment is active when running Phase 3.

With Phase 3 in place, the evenness toolkit now offers end-to-end coverage: fact harmonisation (Phase 1), conditional disparity testing (Phase 2), and explanatory/policy synthesis (Phase 3).
