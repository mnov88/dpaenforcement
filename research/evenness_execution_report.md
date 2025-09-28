# Evenness Pipeline Run — Results Summary (2025-03-02)

## Phase 1 — Foundation & Matching Outputs

* Phase 1 completed on the cleaned wide dataset, producing 1,962 cases in the full cohort, 491 time-observed cases, 6,746 within-country near-twin edges, 4,518 cross-country near-twin edges, and 20 risk ventiles for risk-band analysis.【91a253†L1-L7】
* Balance diagnostics confirm all standardized mean differences well below the 0.10 tolerance across the coarsened exact match, Gower, and risk-band cohorts (largest absolute value < 7.1e-04).【F:outputs/evenness/twin_balance_diagnostics.csv†L1-L20】【F:outputs/evenness/twin_balance_diagnostics.csv†L199-L210】
* Harmonization successfully reconciled 24 country codes directly and flagged UNCLEAR or NOT_DISCUSSED cases for review, ensuring consistent jurisdiction identifiers ahead of uniformity tests.【F:outputs/evenness/country_harmonization_log.csv†L1-L29】
* Coverage exports show multi-country strata in the rule-based twins, supporting cross-jurisdiction comparisons (e.g., stratum 377 spans three countries over 12 cases; stratum 828 covers five countries over 18 cases).【F:outputs/evenness/coverage/cem_coverage.csv†L1-L100】

## Phase 2 — Uniformity Testing Highlights

* Conditional randomization tests strongly reject uniformity: country-level residual effects are significant (e.g., F≈1.24×10³, p≈4.44×10⁻¹⁵³ for fine_eur) and DPA-level effects remain jointly significant across all outcomes in both full and IPW-weighted cohorts.【F:outputs/evenness/uniformity/joint_tests.csv†L1-L10】
* Representative country effects reveal systematic severity/leniency gaps. The table below lists the five most statistically significant jurisdictions per outcome (full cohort):

  | Jurisdiction | Outcome | Effect | Std. Err. | p-value | 95% CI | N |
  |--------------|---------|-------:|----------:|--------:|-------:|---:|
  | ES | fine_eur | -3.70e6 | 7.91e5 | <0.001 | [-5.25e6, -2.15e6] | 1,499 |
  | GB | fine_eur | -1.10e7 | 5.21e5 | <0.001 | [-1.20e7, -9.94e6] | 1,499 |
  | HU | fine_eur | 8.05e6 | 9.21e5 | <0.001 | [6.25e6, 9.85e6] | 1,499 |
  | IT | fine_eur | -1.52e7 | 6.61e5 | <0.001 | [-1.65e7, -1.39e7] | 1,499 |
  | LU | fine_eur | -3.68e7 | 0.53 | <0.001 | [-3.68e7, -3.68e7] | 1,499 |
  | GB | fine_log1p | 0.379 | 0.030 | <0.001 | [0.319, 0.438] | 1,499 |
  | HU | fine_log1p | 1.981 | 0.117 | <0.001 | [1.751, 2.210] | 1,499 |
  | IT | fine_log1p | -1.225 | 0.049 | <0.001 | [-1.321, -1.128] | 1,499 |
  | MT | fine_log1p | 1.348 | 0.195 | <0.001 | [0.967, 1.729] | 1,499 |
  | RO | fine_log1p | -2.251 | 0.299 | <0.001 | [-2.837, -1.666] | 1,499 |
  | GB | fine_positive | 0.420 | 0.043 | <0.001 | [0.336, 0.505] | 1,761 |
  | HU | fine_positive | -0.226 | 0.065 | <0.001 | [-0.353, -0.100] | 1,761 |
  | IT | fine_positive | 0.071 | 0.019 | <0.001 | [0.034, 0.109] | 1,761 |
  | LT | fine_positive | -0.111 | 0.031 | <0.001 | [-0.171, -0.051] | 1,761 |
  | LU | fine_positive | 0.604 | 0.018 | <0.001 | [0.569, 0.639] | 1,761 |

  (Source: `uniformity/jurisdiction_effects.csv` filtered to full-cohort country effects.)【F:outputs/evenness/uniformity/jurisdiction_effects.csv†L1-L20】【F:outputs/evenness/uniformity/jurisdiction_effects.csv†L217-L236】
* Distributional parity fails in several matched strata. For example, in rule-based stratum 25 (ES vs FR), fine amounts differ sharply (KS=1.0, EMD=710,000 EUR) alongside log-fine disparities (KS=1.0, EMD=2.24).【F:outputs/evenness/uniformity/distribution_tests.csv†L1-L7】
* Calibration curves show pronounced over-prediction of fines: the first nine risk ventiles post cross-fitting have mean predicted fine_positive probabilities between 3.2e-4 and 1.87e-2, yet observed rates remain zero, indicating insufficient alignment even after conditioning on facts.【F:outputs/evenness/uniformity/calibration.csv†L1-L10】

## Phase 3 — Partial Completion and Blockers

* Interaction scans executed successfully, yielding a driver leaderboard where multinational status, breach case type, and discussion indicators surface as top jurisdiction-specific drivers (ΔAIC ≤ -10.96 with FDR-controlled p-values <0.01).【F:outputs/evenness/phase_three/driver_leaderboard.csv†L1-L10】
* Downstream Phase 3 modules failed under statsmodels 0.14.5: `OaxacaBlinder` rejects the `weights` argument, halting decompositions and keeping lever/robustness outputs empty despite reruns.【cbf0e9†L1-L12】【325a5e†L1-L2】【14c8b0†L1-L2】【e5049f†L1-L2】
* Subsequent attempts to patch the Oaxaca class hung during recomputation, necessitating manual termination and leaving policy lever estimates and randomization inference unpopulated for this run.【5457e4†L1-L2】【e5049f†L1-L2】

