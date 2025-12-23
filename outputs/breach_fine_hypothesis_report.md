# Hypothesis Test: Breach Cases and Fine Rates

## Hypothesis

**Cases with data breaches have significantly higher fine rates than non-breach cases.**

## Dataset

- **Total records analyzed**: 999 GDPR enforcement decisions
- **Breach cases**: 97 (9.7%)
- **Non-breach cases**: 902 (90.3%)

## Key Findings

### 1. Fine Rate Analysis

| Metric | Breach Cases | Non-Breach Cases |
|--------|-------------|------------------|
| Total cases | 97 | 902 |
| Cases with fines | 64 | 337 |
| **Fine rate** | **66.0%** | **37.4%** |

**Difference: +28.6 percentage points**

Breach cases are **1.77x more likely** to result in a fine compared to non-breach cases.

### 2. Statistical Significance

| Test | Statistic | P-value | Significant? |
|------|-----------|---------|--------------|
| Chi-square | 28.67 | < 0.000001 | Yes |
| Fisher's exact | OR = 3.25 | < 0.000001 | Yes |

Both tests confirm the difference is **highly statistically significant** (p < 0.001).

The **odds ratio of 3.25** means breach cases have more than 3x the odds of receiving a fine compared to non-breach cases.

### 3. Fine Amount Comparison (When Fines Are Imposed)

| Metric | Breach Cases (n=64) | Non-Breach Cases (n=337) |
|--------|---------------------|--------------------------|
| Mean | €1,195,973 | €645,776 |
| Median | €25,000 | €15,000 |
| 25th percentile | €5,000 | €3,000 |
| 75th percentile | €125,000 | €70,000 |
| Maximum | €35,000,000 | €75,000,000 |

**Mann-Whitney U test p-value**: 0.057 (marginally non-significant)

While breach cases show higher median fines (€25,000 vs €15,000), the difference is not statistically significant at α=0.05.

## Conclusion

### Hypothesis: SUPPORTED

**Breach cases have significantly higher fine rates:**

- Breach cases: 66.0% receive fines
- Non-breach cases: 37.4% receive fines
- The difference is highly statistically significant (p < 0.001)
- Odds ratio: 3.25x higher odds of fine for breach cases

**Fine amounts show a trend but are not significantly different:**

- Breach cases have higher median fines (€25,000 vs €15,000)
- The Mann-Whitney U test is marginally non-significant (p = 0.057)
- Effect size is negligible (Cohen's d = 0.11)

## Interpretation

Data protection authorities are significantly more likely to impose fines in cases involving data breaches. This aligns with the GDPR's emphasis on security (Article 32) and breach notification requirements (Articles 33-34).

The finding that fine *amounts* are not significantly different suggests that once a DPA decides to impose a fine, the breach vs. non-breach nature of the case may be less determinative of the amount than other factors (severity, number of affected individuals, organization size, etc.).

---

*Analysis performed: 2025-12-23*
*Data source: cleaned_wide.csv (999 GDPR enforcement decisions)*
