#!/usr/bin/env python3
"""
Hypothesis Test: Cases with breach notifications have significantly higher fine rates

This script tests whether GDPR enforcement cases involving data breaches
have statistically significantly higher rates of fines imposed.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("=" * 80)
print("HYPOTHESIS TEST: Breach Notification Cases Have Higher Fine Rates")
print("=" * 80)

df = pd.read_csv('/home/user/dpaenforcement/outputs/cleaned_wide.csv')

print(f"\nTotal records in dataset: {len(df)}")

# Examine breach_case field
print(f"\n--- Breach Case Distribution ---")
print(df['breach_case'].value_counts(dropna=False))

# Clean and prepare data
# breach_case appears to be binary (0/1) based on initial exploration
df_analysis = df[df['breach_case'].isin([0, 1, '0', '1'])].copy()
df_analysis['breach_case'] = df_analysis['breach_case'].astype(int)
df_analysis['fine_positive'] = df_analysis['fine_positive'].astype(int)

print(f"\nRecords with valid breach_case indicator: {len(df_analysis)}")

# ============================================================================
# PART 1: Fine Rate Analysis (Binary: Was a fine imposed?)
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: FINE RATE ANALYSIS (Was a fine imposed?)")
print("=" * 80)

# Create contingency table
breach_fined = df_analysis[(df_analysis['breach_case'] == 1) & (df_analysis['fine_positive'] == 1)].shape[0]
breach_not_fined = df_analysis[(df_analysis['breach_case'] == 1) & (df_analysis['fine_positive'] == 0)].shape[0]
no_breach_fined = df_analysis[(df_analysis['breach_case'] == 0) & (df_analysis['fine_positive'] == 1)].shape[0]
no_breach_not_fined = df_analysis[(df_analysis['breach_case'] == 0) & (df_analysis['fine_positive'] == 0)].shape[0]

contingency_table = np.array([
    [breach_fined, breach_not_fined],
    [no_breach_fined, no_breach_not_fined]
])

print("\n--- Contingency Table ---")
print(f"                        Fine Imposed    No Fine")
print(f"Breach Cases:           {breach_fined:>10}      {breach_not_fined:>7}")
print(f"Non-Breach Cases:       {no_breach_fined:>10}      {no_breach_not_fined:>7}")

# Calculate fine rates
total_breach = breach_fined + breach_not_fined
total_no_breach = no_breach_fined + no_breach_not_fined

breach_fine_rate = breach_fined / total_breach if total_breach > 0 else 0
no_breach_fine_rate = no_breach_fined / total_no_breach if total_no_breach > 0 else 0

print(f"\n--- Fine Rates ---")
print(f"Breach cases:     {breach_fine_rate:.1%} ({breach_fined}/{total_breach})")
print(f"Non-breach cases: {no_breach_fine_rate:.1%} ({no_breach_fined}/{total_no_breach})")
print(f"Difference:       {(breach_fine_rate - no_breach_fine_rate)*100:+.1f} percentage points")

# Relative risk
if no_breach_fine_rate > 0:
    relative_risk = breach_fine_rate / no_breach_fine_rate
    print(f"Relative Risk:    {relative_risk:.2f}x")
else:
    relative_risk = float('inf')
    print(f"Relative Risk:    undefined (no fines in non-breach cases)")

# Chi-square test
print(f"\n--- Chi-Square Test ---")
chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom:   {dof}")
print(f"P-value:              {p_chi2:.6f}")
print(f"Expected frequencies:\n{expected}")

# Fisher's exact test (more robust for small samples)
print(f"\n--- Fisher's Exact Test ---")
odds_ratio, p_fisher = fisher_exact(contingency_table)
print(f"Odds Ratio:           {odds_ratio:.4f}")
print(f"P-value:              {p_fisher:.6f}")

# Interpretation
alpha = 0.05
print(f"\n--- Interpretation (α = {alpha}) ---")
if p_chi2 < alpha:
    print(f"✓ Chi-square test: SIGNIFICANT (p = {p_chi2:.6f} < {alpha})")
else:
    print(f"✗ Chi-square test: NOT significant (p = {p_chi2:.6f} >= {alpha})")

if p_fisher < alpha:
    print(f"✓ Fisher's exact: SIGNIFICANT (p = {p_fisher:.6f} < {alpha})")
else:
    print(f"✗ Fisher's exact: NOT significant (p = {p_fisher:.6f} >= {alpha})")

# ============================================================================
# PART 2: Fine Amount Analysis (For cases where fines were imposed)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: FINE AMOUNT ANALYSIS (Among cases with fines)")
print("=" * 80)

# Filter to only cases with positive fines
df_fined = df_analysis[df_analysis['fine_positive'] == 1].copy()
df_fined['fine_eur'] = pd.to_numeric(df_fined['fine_eur'], errors='coerce')

breach_fines = df_fined[df_fined['breach_case'] == 1]['fine_eur'].dropna()
no_breach_fines = df_fined[df_fined['breach_case'] == 0]['fine_eur'].dropna()

print(f"\n--- Sample Sizes ---")
print(f"Breach cases with fines:     {len(breach_fines)}")
print(f"Non-breach cases with fines: {len(no_breach_fines)}")

if len(breach_fines) > 0 and len(no_breach_fines) > 0:
    print(f"\n--- Descriptive Statistics (Fine Amounts in EUR) ---")
    print(f"\n{'Metric':<25} {'Breach Cases':>18} {'Non-Breach Cases':>18}")
    print("-" * 65)
    print(f"{'Mean':.<25} {breach_fines.mean():>18,.0f} {no_breach_fines.mean():>18,.0f}")
    print(f"{'Median':.<25} {breach_fines.median():>18,.0f} {no_breach_fines.median():>18,.0f}")
    print(f"{'Std Dev':.<25} {breach_fines.std():>18,.0f} {no_breach_fines.std():>18,.0f}")
    print(f"{'Min':.<25} {breach_fines.min():>18,.0f} {no_breach_fines.min():>18,.0f}")
    print(f"{'Max':.<25} {breach_fines.max():>18,.0f} {no_breach_fines.max():>18,.0f}")
    print(f"{'25th Percentile':.<25} {breach_fines.quantile(0.25):>18,.0f} {no_breach_fines.quantile(0.25):>18,.0f}")
    print(f"{'75th Percentile':.<25} {breach_fines.quantile(0.75):>18,.0f} {no_breach_fines.quantile(0.75):>18,.0f}")

    # Mann-Whitney U test (non-parametric, robust to non-normality)
    print(f"\n--- Mann-Whitney U Test (non-parametric) ---")
    stat_mw, p_mw = mannwhitneyu(breach_fines, no_breach_fines, alternative='two-sided')
    print(f"U statistic:    {stat_mw:.2f}")
    print(f"P-value:        {p_mw:.6f}")

    # Also run one-sided test (breach > non-breach)
    stat_mw_greater, p_mw_greater = mannwhitneyu(breach_fines, no_breach_fines, alternative='greater')
    print(f"\nOne-sided test (breach > non-breach):")
    print(f"P-value:        {p_mw_greater:.6f}")

    # Welch's t-test (robust to unequal variances)
    print(f"\n--- Welch's t-test (parametric) ---")
    stat_t, p_t = ttest_ind(breach_fines, no_breach_fines, equal_var=False)
    print(f"t statistic:    {stat_t:.4f}")
    print(f"P-value:        {p_t:.6f}")

    # Log-transformed comparison (since fine distributions are typically right-skewed)
    print(f"\n--- Log-Transformed Fine Analysis ---")
    log_breach_fines = np.log1p(breach_fines)
    log_no_breach_fines = np.log1p(no_breach_fines)

    stat_t_log, p_t_log = ttest_ind(log_breach_fines, log_no_breach_fines, equal_var=False)
    print(f"Mean log(fine+1) - Breach:     {log_breach_fines.mean():.4f}")
    print(f"Mean log(fine+1) - Non-breach: {log_no_breach_fines.mean():.4f}")
    print(f"t statistic:                   {stat_t_log:.4f}")
    print(f"P-value:                       {p_t_log:.6f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((breach_fines.std()**2 + no_breach_fines.std()**2) / 2)
    cohens_d = (breach_fines.mean() - no_breach_fines.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"\n--- Effect Size ---")
    print(f"Cohen's d:      {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"Interpretation: {effect_interp}")

    # Interpretation
    print(f"\n--- Interpretation (α = {alpha}) ---")
    if p_mw < alpha:
        print(f"✓ Mann-Whitney U: SIGNIFICANT (p = {p_mw:.6f} < {alpha})")
    else:
        print(f"✗ Mann-Whitney U: NOT significant (p = {p_mw:.6f} >= {alpha})")

    if p_t < alpha:
        print(f"✓ Welch's t-test: SIGNIFICANT (p = {p_t:.6f} < {alpha})")
    else:
        print(f"✗ Welch's t-test: NOT significant (p = {p_t:.6f} >= {alpha})")
else:
    print("\nInsufficient data for fine amount comparison.")

# ============================================================================
# PART 3: Summary and Conclusion
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY AND CONCLUSION")
print("=" * 80)

print(f"""
HYPOTHESIS: Cases with breach notifications have significantly higher fine rates

FINDINGS:

1. FINE RATE (Probability of receiving a fine):
   - Breach cases:     {breach_fine_rate:.1%} fine rate
   - Non-breach cases: {no_breach_fine_rate:.1%} fine rate
   - Difference:       {(breach_fine_rate - no_breach_fine_rate)*100:+.1f} percentage points
   - Chi-square p-value: {p_chi2:.6f}
   - Fisher's exact p-value: {p_fisher:.6f}
""")

if len(breach_fines) > 0 and len(no_breach_fines) > 0:
    print(f"""2. FINE AMOUNT (When a fine is imposed):
   - Breach cases median:     €{breach_fines.median():,.0f}
   - Non-breach cases median: €{no_breach_fines.median():,.0f}
   - Breach cases mean:       €{breach_fines.mean():,.0f}
   - Non-breach cases mean:   €{no_breach_fines.mean():,.0f}
   - Mann-Whitney p-value:    {p_mw:.6f}
   - Effect size (Cohen's d): {cohens_d:.4f} ({effect_interp})
""")

# Final verdict
print("CONCLUSION:")
if p_chi2 < alpha and p_fisher < alpha:
    if breach_fine_rate > no_breach_fine_rate:
        print("  ✓ SUPPORTED: Breach cases have SIGNIFICANTLY HIGHER fine rates.")
    else:
        print("  ✗ REJECTED: Breach cases have SIGNIFICANTLY LOWER fine rates.")
else:
    print("  ✗ NOT SUPPORTED: No statistically significant difference in fine rates.")

if len(breach_fines) > 0 and len(no_breach_fines) > 0:
    if p_mw < alpha:
        if breach_fines.median() > no_breach_fines.median():
            print("  ✓ Breach cases also have significantly HIGHER fine amounts.")
        else:
            print("  ✗ Breach cases have significantly LOWER fine amounts.")
    else:
        print("  - No significant difference in fine amounts between groups.")

print("\n" + "=" * 80)

# Save results to CSV
results = {
    'metric': [
        'total_records',
        'valid_breach_records',
        'breach_cases_total',
        'breach_cases_fined',
        'breach_fine_rate',
        'non_breach_cases_total',
        'non_breach_cases_fined',
        'non_breach_fine_rate',
        'rate_difference_pct_points',
        'relative_risk',
        'chi2_statistic',
        'chi2_p_value',
        'fisher_odds_ratio',
        'fisher_p_value',
        'breach_fines_n',
        'breach_fines_mean',
        'breach_fines_median',
        'non_breach_fines_n',
        'non_breach_fines_mean',
        'non_breach_fines_median',
        'mannwhitney_p_value',
        'cohens_d',
        'hypothesis_supported'
    ],
    'value': [
        len(df),
        len(df_analysis),
        total_breach,
        breach_fined,
        breach_fine_rate,
        total_no_breach,
        no_breach_fined,
        no_breach_fine_rate,
        (breach_fine_rate - no_breach_fine_rate) * 100,
        relative_risk if relative_risk != float('inf') else None,
        chi2,
        p_chi2,
        odds_ratio,
        p_fisher,
        len(breach_fines) if len(breach_fines) > 0 else None,
        breach_fines.mean() if len(breach_fines) > 0 else None,
        breach_fines.median() if len(breach_fines) > 0 else None,
        len(no_breach_fines) if len(no_breach_fines) > 0 else None,
        no_breach_fines.mean() if len(no_breach_fines) > 0 else None,
        no_breach_fines.median() if len(no_breach_fines) > 0 else None,
        p_mw if len(breach_fines) > 0 and len(no_breach_fines) > 0 else None,
        cohens_d if len(breach_fines) > 0 and len(no_breach_fines) > 0 else None,
        (p_chi2 < 0.05 and p_fisher < 0.05 and breach_fine_rate > no_breach_fine_rate)
    ]
}

results_df = pd.DataFrame(results)
results_df.to_csv('/home/user/dpaenforcement/outputs/breach_fine_hypothesis_results.csv', index=False)
print(f"\nResults saved to: /home/user/dpaenforcement/outputs/breach_fine_hypothesis_results.csv")
