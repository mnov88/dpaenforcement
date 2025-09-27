# Phase 1 – Foundation and "Twin" Construction

This note documents the Phase 1 implementation for the GDPR evenness study. The phase delivers facts-only design matrices, harmonised identifiers, and multiple "twin" constructions that enable apples-to-apples outcome comparisons in later phases.

## Data preparation

* Inputs come from `outputs/cleaned_wide_latest.csv`. Only structured fields with a `*_status=DISCUSSED` flag are used; `NOT_MENTIONED` and `NOT_APPLICABLE` states are preserved as explicit indicators.
* Rows with any `*_exclusivity_conflict=1` flag are dropped. Country codes are harmonised to ISO-3166 two-letter codes with a log of the applied mappings.
* Derived helper fields include:
  * Days since GDPR (converted from UTC decision timestamps).
  * Coarsened bins for `n_principles_discussed`, `n_principles_violated`, and `n_corrective_measures`.
  * Multi-select "signatures" for breach types, remedial measures, sensitive data, and vulnerable data subjects.
  * Country-year weights to temper Spain/Italy volume and a time-observed indicator.

The CLI command below generates the artefacts:

```bash
python -m scripts.evenness.cli phase-one
```

Key outputs (written under `outputs/evenness/`) are:

* `X_full.parquet` – full-fact design matrix (facts + outcomes).
* `X_timeobs.parquet` – time-observed subset with inverse-probability weights.
* `twins_cem.parquet`, `twins_gower_within.parquet`, `twins_gower_cross.parquet`, `twins_riskbands.parquet`.
* `twin_balance_diagnostics.csv` – post-match standardised differences.
* Coverage summaries in `coverage/` and common-support plots in `support/`.
* `country_harmonization_log.csv`.

## Balance diagnostics

Post-match balance meets the Phase 1 acceptance threshold (|SMD| ≤ 0.10):

| Twin set       | Mean SMD | Max SMD | Min SMD |
|----------------|---------:|--------:|--------:|
| CEM            |   0.000  |  0.000  |  0.000  |
| Gower (within) |   0.009  |  0.088  | -0.025  |
| Gower (cross)  |   0.012  |  0.100  | -0.041  |
| Risk bands     |   0.000  |  0.001  | -0.000  |

CEM strata provide 71 matched cohorts (median size 2, maximum 32) covering 2–5 jurisdictions per stratum. Gower near-twins retain 6,746 within-country and 4,645 cross-country edges under a 0.03 caliper with status hard-matching. Risk-band twins bucket cases into 20 ventiles of modelled fine risk.

## Usage notes

* Re-running the CLI regenerates all artefacts; outputs are not versioned in Git to keep the repo lean.
* The new `phase-one` subcommand also prints a summary of cohort sizes for quick verification.
* Logistic components use `liblinear` with convergence warnings suppressed; runtime is ~2 minutes on the provided hardware.

These foundations enable Phase 2 to test conditional uniformity while ensuring matched cohorts are balanced on all structured facts.
