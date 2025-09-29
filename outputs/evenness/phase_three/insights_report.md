# Phase 3 – Explanation & Policy Synthesis

## Driver Attribution

| term                                    |   delta_aic |   lr_stat |      pvalue | group_field        |   pvalue_fdr |
|:----------------------------------------|------------:|----------:|------------:|:-------------------|-------------:|
| q47_remedial_coverage__DISCUSSED        |   -10.5554  |   46.5554 | 0.00024606  | country_code       |  0.000984239 |
| n_corrective_measures                   |    -7.26651 |   45.2665 | 0.00062889  | country_code       |  0.00167704  |
| ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE |    -2.82648 |   30.8265 | 0.00586192  | country_code       |  0.0117238   |
| n_principles_violated                   |     3.18693 |   36.8131 | 0.0123207   | country_code       |  0.0155538   |
| q47_remedial_coverage__DISCUSSED        |   -26.2947  |  116.295  | 3.17206e-08 | dpa_name_canonical |  2.53765e-07 |
| ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE |     9.42409 |   54.5759 | 0.00769626  | dpa_name_canonical |  0.012314    |
| n_corrective_measures                   |    27.7371  |   78.2629 | 0.0136096   | dpa_name_canonical |  0.0155538   |
| n_principles_violated                   |    46.5818  |   69.4182 | 0.144887    | dpa_name_canonical |  0.144887    |


## Predictive SHAP Attributions

| feature                                                |   mean_abs_shap |
|:-------------------------------------------------------|----------------:|
| REMEDY_ONLY_CASE_YES                                   |        1.79218  |
| REMEDY_ONLY_CASE_NO                                    |        1.41431  |
| N_CORRECTIVE_MEASURES_BIN_0                            |        1.13485  |
| n_corrective_measures                                  |        0.709549 |
| n_principles_violated                                  |        0.337927 |
| CASE_ORIGIN_COMPLAINT                                  |        0.214012 |
| ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE                |        0.208784 |
| ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE,MULTINATIONAL  |        0.145574 |
| ORGANIZATION_TYPE_NATURAL_PERSON                       |        0.130878 |
| ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE, MULTINATIONAL |        0.129029 |


## Gap Decomposition

|   explained |   unexplained |   overall |   explained_se |   unexplained_se |   overall_se | group_a   | group_b   | outcome    |   n_obs |
|------------:|--------------:|----------:|---------------:|-----------------:|-------------:|:----------|:----------|:-----------|--------:|
|     2.71348 |     -0.504831 |   2.20865 |            nan |              nan |          nan | ES        | IT        | fine_log1p |     843 |
|     3.20896 |     -2.02413  |   1.18483 |            nan |              nan |          nan | ES        | RO        | fine_log1p |     700 |
|     4.33958 |     -1.81574  |   2.52384 |            nan |              nan |          nan | ES        | IS        | fine_log1p |     656 |
|     1.9904  |     -0.562702 |   1.42769 |            nan |              nan |          nan | ES        | GR        | fine_log1p |     633 |
|     1.06507 |      1.32463  |   2.38969 |            nan |              nan |          nan | ES        | NO        | fine_log1p |     639 |


## Policy Lever Estimates

_No data available._


## Randomisation Inference

| outcome       |   observed_stat |   perm_mean |   perm_std |    pvalue |
|:--------------|----------------:|------------:|-----------:|----------:|
| fine_positive |        0.347251 |    0.342132 |   0.077213 | 0.545455  |
| fine_log1p    |      141.627    |  114.121    |   5.50567  | 0.0909091 |


## Robustness Summary

| scenario              | type     |   nobs |   logistic_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY |   linear_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY |   logistic_q46_vuln_CHILDREN |   linear_q46_vuln_CHILDREN | notes                                                                     |   quantile |
|:----------------------|:---------|-------:|---------------------------------------------------------:|-------------------------------------------------------:|-----------------------------:|---------------------------:|:--------------------------------------------------------------------------|-----------:|
| reweight_country_year | glm_ols  |   1962 |                                                      nan |                                                    nan |                          nan |                        nan | Applied country-year reweighting                                          |     nan    |
| heckman_turnover      | glm_ols  |   1962 |                                                      nan |                                                    nan |                          nan |                        nan | Skipped turnover correction (turnover_log1p missing); Linear model failed |     nan    |
| discussed_only        | glm_ols  |   1962 |                                                      nan |                                                    nan |                          nan |                        nan | Filtered to discussed-only records; Linear model failed                   |     nan    |
| winsorize_fines       | glm_ols  |   1962 |                                                      nan |                                                    nan |                          nan |                        nan | Winsorized fine_log1p at q=0.99; Linear model failed                      |     nan    |
| quantile_75           | quantile |   1719 |                                                      nan |                                                    nan |                          nan |                        nan |                                                                           |       0.75 |
| quantile_90           | quantile |   1719 |                                                      nan |                                                    nan |                          nan |                        nan |                                                                           |       0.9  |