# Uniform Treatment Playbook
### Priority Drivers
| group_field        | term                                    |   pvalue_fdr |
|:-------------------|:----------------------------------------|-------------:|
| dpa_name_canonical | q47_remedial_coverage__DISCUSSED        |  2.53765e-07 |
| country_code       | q47_remedial_coverage__DISCUSSED        |  0.000984239 |
| country_code       | n_corrective_measures                   |  0.00167704  |
| country_code       | ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE |  0.0117238   |
| country_code       | n_principles_violated                   |  0.0155538   |

### Robustness Diagnostics
| scenario              | type     |   nobs |   logistic_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY |   linear_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY |   logistic_q46_vuln_CHILDREN |   linear_q46_vuln_CHILDREN | notes                                                                     |   quantile |
|:----------------------|:---------|-------:|---------------------------------------------------------:|-------------------------------------------------------:|-----------------------------:|---------------------------:|:--------------------------------------------------------------------------|-----------:|
| reweight_country_year | glm_ols  |   1962 |                                                      nan |                                                    nan |                          nan |                        nan | Applied country-year reweighting                                          |     nan    |
| heckman_turnover      | glm_ols  |   1962 |                                                      nan |                                                    nan |                          nan |                        nan | Skipped turnover correction (turnover_log1p missing); Linear model failed |     nan    |
| discussed_only        | glm_ols  |   1962 |                                                      nan |                                                    nan |                          nan |                        nan | Filtered to discussed-only records; Linear model failed                   |     nan    |
| winsorize_fines       | glm_ols  |   1962 |                                                      nan |                                                    nan |                          nan |                        nan | Winsorized fine_log1p at q=0.99; Linear model failed                      |     nan    |
| quantile_75           | quantile |   1719 |                                                      nan |                                                    nan |                          nan |                        nan |                                                                           |       0.75 |
| quantile_90           | quantile |   1719 |                                                      nan |                                                    nan |                          nan |                        nan |                                                                           |       0.9  |