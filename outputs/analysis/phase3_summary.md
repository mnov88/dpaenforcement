# Phase 3: Robustness and Sensitivity Analysis Results

## Overview

This phase tests the robustness of our causal estimates to methodological choices
and potential confounding, following academic best practices for policy analysis.

## Robustness Summary

| Test | Original | Robust | Difference | Rel Change | Passes |
|------|--------:|-------:|-----------:|-----------:|-------:|
| Time Fixed Effects | 0.1000 | 0.0242 | -0.0758 | -75.84% | ⚠️ |

**Time Fixed Effects**: Compares time-FE with IPW vs full sample specification

## Country Sensitivity Analysis

*Country sensitivity analysis not completed*

## Time Controls Sensitivity

- **Time-FE specification**: 0.0242
- **No time-FE specification**: 0.1000

- **Robust to time controls**: No

## Selection Correction

*Selection correction analysis not completed*

## Notes

- **Robustness thresholds**: <30% change for country weights, <50% for time controls
- **Selection correction**: Required for turnover-based analyses (Article 83(2))
- **ES/IT dominance**: 44% of dataset requires reweighting for generalizability
- **Time controls**: IPW adjusts for 75% missing decision dates