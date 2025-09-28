# Phase 3 – Explanation & Policy Synthesis

## Driver Attribution

```
                                   term  delta_aic    lr_stat       pvalue        group_field   pvalue_fdr
  ORGANIZATION_SIZE_TIER_NOT_APPLICABLE -36.256513  56.256513 1.837831e-08       country_code 9.189155e-08
       q47_remedial_coverage__DISCUSSED -21.852208  55.852208 4.996442e-06       country_code 1.249110e-05
ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE -14.710374  38.710374 1.174414e-04       country_code 2.348828e-04
                  n_corrective_measures -11.617415  49.617415 1.491695e-04       country_code 2.486159e-04
                  n_principles_violated   2.602966  37.397034 1.048273e-02       country_code 1.310341e-02
       q47_remedial_coverage__DISCUSSED -29.163010 115.163010 1.655722e-08 dpa_name_canonical 9.189155e-08
  ORGANIZATION_SIZE_TIER_NOT_APPLICABLE -28.039624  60.039624 5.153895e-07 dpa_name_canonical 1.717965e-06
ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE  -6.101422  58.101422 3.001950e-04 dpa_name_canonical 4.288500e-04
                  n_corrective_measures  25.782976  74.217024 1.469320e-02 dpa_name_canonical 1.632578e-02
                  n_principles_violated  46.423272  67.576728 1.594327e-01 dpa_name_canonical 1.594327e-01
```


## Predictive SHAP Attributions

```
                                              feature  mean_abs_shap
                                 REMEDY_ONLY_CASE_YES       1.917784
                                  REMEDY_ONLY_CASE_NO       1.360648
                          N_CORRECTIVE_MEASURES_BIN_0       1.075239
                                n_corrective_measures       0.854009
                                n_principles_violated       0.418716
              ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE       0.246710
ORGANIZATION_SIZE_TIER_LARGE_ENTERPRISE,MULTINATIONAL       0.228760
                                CASE_ORIGIN_COMPLAINT       0.220090
                                   Q47_SIGNATURE_NONE       0.212775
                                      days_since_gdpr       0.091599
```


## Gap Decomposition

```
 explained  unexplained  overall  explained_se  unexplained_se  overall_se group_a group_b    outcome  n_obs
  2.693593    -0.455183 2.238410           NaN             NaN         NaN      ES      IT fine_log1p    767
  3.104737    -1.989918 1.114818           NaN             NaN         NaN      ES      RO fine_log1p    652
  4.304937    -1.931327 2.373610           NaN             NaN         NaN      ES      IS fine_log1p    635
  1.987129    -0.594479 1.392650           NaN             NaN         NaN      ES      GR fine_log1p    618
  1.077732     1.345933 2.423665           NaN             NaN         NaN      ES      NO fine_log1p    625
```


## Policy Lever Estimates

_No data available._


## Randomisation Inference

```
      outcome  observed_stat  perm_mean  perm_std   pvalue
fine_positive       2.574179   2.127920  0.169260 0.090909
   fine_log1p     125.664198 103.211665  4.671151 0.090909
```


## Robustness Summary

```
             scenario     type   nobs  logistic_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY  linear_q25_sensitive_data_ARTICLE_9_SPECIAL_CATEGORY  logistic_q46_vuln_CHILDREN  linear_q46_vuln_CHILDREN                                                                     notes  quantile
reweight_country_year  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN                                          Applied country-year reweighting       NaN
     heckman_turnover  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN Skipped turnover correction (turnover_log1p missing); Linear model failed       NaN
       discussed_only  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN                   Filtered to discussed-only records; Linear model failed       NaN
      winsorize_fines  glm_ols 1962.0                                                     NaN                                                   NaN                         NaN                       NaN                      Winsorized fine_log1p at q=0.99; Linear model failed       NaN
          quantile_75 quantile 1534.0                                                     NaN                                                   NaN                         NaN                       NaN                                                                                0.75
          quantile_90 quantile 1534.0                                                     NaN                                                   NaN                         NaN                       NaN                                                                                0.90
```