# Phase 3 – Explanation & Policy Synthesis

## Driver Attribution

```
                                         term  delta_aic   lr_stat   pvalue        group_field  pvalue_fdr
        ORGANIZATION_SIZE_TIER_NOT_APPLICABLE  -6.128579 28.128579 0.003094       country_code    0.049128
   q25_sensitive_data_coverage__NOT_MENTIONED  -3.849159  5.849159 0.015585       country_code    0.049128
       q21_breach_types_status__NOT_MENTIONED  -3.773647  5.773647 0.016268       country_code    0.049128
                  q47_remedial_POLICY_CHANGES   8.239468 25.760532 0.078935       country_code    0.177604
q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY   8.480654 27.519346 0.069756       country_code    0.177604
                            q46_vuln_CHILDREN  10.614520 23.385480 0.137117       country_code    0.246810
                        n_corrective_measures  10.799320 27.200680 0.100065       country_code    0.200130
      ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE  13.861426 14.138574 0.439439       country_code    0.564993
                        n_principles_violated  17.020945 22.979055 0.289824       country_code    0.401295
       q21_breach_types_status__NOT_MENTIONED  -3.860496  5.860496 0.015484 dpa_name_canonical    0.049128
```


## Predictive SHAP Attributions

```
                                   feature  mean_abs_shap
                      REMEDY_ONLY_CASE_YES       1.620918
                     n_corrective_measures       1.037333
                       REMEDY_ONLY_CASE_NO       0.978743
               N_CORRECTIVE_MEASURES_BIN_0       0.517895
                     CASE_ORIGIN_COMPLAINT       0.297804
                     n_principles_violated       0.267231
               q47_remedial_POLICY_CHANGES       0.236663
q25_sensitive_data_coverage__NOT_MENTIONED       0.201011
                        Q47_SIGNATURE_NONE       0.170212
   ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE       0.159786
```


## Gap Decomposition

```
 explained  unexplained  overall  explained_se  unexplained_se  overall_se group_a group_b    outcome  n_obs
  2.678168    -0.752294 1.925874           NaN             NaN         NaN      ES      IT fine_log1p    848
  2.702165    -1.526628 1.175537           NaN             NaN         NaN      ES      RO fine_log1p    717
  4.196068    -0.574972 3.621096           NaN             NaN         NaN      ES      IS fine_log1p    660
  1.458870    -0.268844 1.190027           NaN             NaN         NaN      ES      GR fine_log1p    644
  1.018389    -0.158663 0.859725           NaN             NaN         NaN      ES      NO fine_log1p    644
```


## Policy Lever Estimates

_No data available._


## Randomisation Inference

```
      outcome  observed_stat  perm_mean  perm_std   pvalue
fine_positive       1.094248   0.749189  0.097879 0.090909
   fine_log1p     127.449374 105.809640 13.899449 0.090909
```


## Robustness Summary

```
             scenario     type   nobs  logistic_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY  linear_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY  logistic_q46_vuln_CHILDREN  linear_q46_vuln_CHILDREN                                                                     notes  quantile
reweight_country_year  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN                                          Applied country-year reweighting       NaN
     heckman_turnover  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN Skipped turnover correction (turnover_log1p missing); Linear model failed       NaN
       discussed_only  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN                   Filtered to discussed-only records; Linear model failed       NaN
      winsorize_fines  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN                      Winsorized fine_log1p at q=0.99; Linear model failed       NaN
          quantile_75 quantile 1855.0                                                     NaN                                                   NaN                         NaN                       NaN                                                                                0.75
          quantile_90 quantile 1855.0                                                     NaN                                                   NaN                         NaN                       NaN                                                                                0.90
```