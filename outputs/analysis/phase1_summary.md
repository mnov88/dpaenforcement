# Phase 1: Extended Causal Analysis Results

## Summary Statistics

- Total Breach Cases: 208
- Cases With Timing Data: 208
- Cases With Notification Data: 208
- Cases With Channel Data: 208
- Vulnerable Cases: 99.0
- Remedial Cases: 169.0

## Initiation Channel Effects

*No channel effects estimated due to insufficient sample size*

## Enhanced Timing Analysis (Fuzzy RD)

- fuzzy_rd_enforcement_severity_index: 0.3023 (95% CI 0.2152, 0.3499)

## Heterogeneous Effects

| Subgroup | Outcome | Estimate | 95% CI | N Treated | N Control |
|----------|---------|--------::|--------|----------:|----------:|
| vulnerable_True | timing_fine_positive | -0.0322 | (-0.1345, 0.2059) | 12 | 37 |
| remedial_True | timing_severity_index | 0.0736 | (-0.1414, 0.2904) | 18 | 58 |

## Notes

- Channel effects compare each initiation method vs COMPLAINT (reference)
- Fuzzy RD uses ordered delay categories with local polynomial estimation
- Heterogeneous effects examine differential impacts by vulnerability/remediation profiles
- All estimates use AIPW (Augmented Inverse Probability Weighting) for robustness