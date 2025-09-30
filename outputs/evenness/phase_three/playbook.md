# Uniform Treatment Playbook
### Priority Drivers
```
       group_field                                       term  pvalue_fdr
      country_code      ORGANIZATION_SIZE_TIER_NOT_APPLICABLE    0.049128
dpa_name_canonical     q21_breach_types_status__NOT_MENTIONED    0.049128
      country_code q25_sensitive_data_coverage__NOT_MENTIONED    0.049128
      country_code     q21_breach_types_status__NOT_MENTIONED    0.049128
dpa_name_canonical q25_sensitive_data_coverage__NOT_MENTIONED    0.049128
```

### Robustness Diagnostics
```
             scenario     type   nobs  logistic_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY  linear_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY  logistic_q46_vuln_CHILDREN  linear_q46_vuln_CHILDREN                                                                     notes  quantile
reweight_country_year  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN                                          Applied country-year reweighting       NaN
     heckman_turnover  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN Skipped turnover correction (turnover_log1p missing); Linear model failed       NaN
       discussed_only  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN                   Filtered to discussed-only records; Linear model failed       NaN
      winsorize_fines  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN                      Winsorized fine_log1p at q=0.99; Linear model failed       NaN
          quantile_75 quantile 1855.0                                                     NaN                                                   NaN                         NaN                       NaN                                                                                0.75
          quantile_90 quantile 1855.0                                                     NaN                                                   NaN                         NaN                       NaN                                                                                0.90
```