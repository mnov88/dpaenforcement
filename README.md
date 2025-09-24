## GDPR DPA Responses Parser

This repository includes a small Python script that aggregates AI-extracted answers from JSON files into analysis-friendly CSVs.

### What it does

- Reads all `results_*.json` files under `raw_data/responses/`.
- Each JSON file contains an array of objects with:
  - `ID`, `English_Translation`, `response` (68-line answer block), and other metadata.
- Parses the 68 answers from `response` into named columns.
- Normalizes values by stripping prefixes like `TYPE:`, `ENUM:`, `MULTI_SELECT:`, `FORMAT:` and reduces `country_code` from `ISO_3166-1_ALPHA-2: FR` to `FR`.
- Coerces numeric fields:
  - `fine_amount_eur` (Answer 37)
  - `annual_turnover_eur` (Answer 38)
  into plain numeric strings when possible (e.g., `600000`, `84000000000`).
- Sorts rows naturally by `ID` (e.g., `France_2` before `France_12`).
- Writes two CSVs to `raw_data/responses/`:
  - `parsed_responses.csv`: includes metadata + answers
  - `parsed_responses_min.csv`: includes only `ID` + answers (no metadata)

### Prerequisites

- Python 3.8+ (no external dependencies)

### How to run

1. Ensure your JSON results are in `raw_data/responses/` and named like `results_YYYYMMDD_*.json` (the script matches `results_*.json`).
2. Execute:

```
python3 parse_responses_to_csv.py
```

3. Outputs will appear as:
   - `raw_data/responses/parsed_responses.csv`
   - `raw_data/responses/parsed_responses_min.csv`

If your data directory is different, edit the `RESPONSES_DIR` constant at the top of `parse_responses_to_csv.py`.

### Columns

- Full CSV columns (in order):
  - Metadata (9): `ID`, `English_Translation`, `error`, `success`, `model_used`, `markdown_file`, `input_tokens`, `output_tokens`, `total_tokens`
  - Answers (68):
    - `country_code`, `dpa_name`, `issue_date`, `is_appeal`, `appeal_outcome`, `multiple_defendants`, `primary_defendant_name`, `defendant_status`, `defendant_role`, `org_classifications`, `public_sector_level`, `isic_sector`, `turnover_mentioned`, `turnover_range`, `initiation_method`, `breach_discussed`, `art33_required`, `breach_notified`, `notified_within_72h`, `delay_length`, `breach_type`, `breach_cause`, `harm_materialized`, `affected_subjects`, `special_or_criminal_data`, `subjects_notified`, `art34_required`, `mitigating_actions`, `notification_failures_effect`, `art5_principles_discussed`, `art5_principles_violated`, `art6_bases_discussed`, `legal_bases_relied`, `consent_issues`, `legitimate_interest_outcome`, `art56_summary`, `fine_amount_eur`, `annual_turnover_eur`, `hit_caps`, `violation_duration`, `aggravating_factors`, `mitigating_factors`, `harm_documented`, `economic_benefit`, `cooperation_level`, `vulnerable_subjects`, `remedial_actions`, `first_time_violation`, `cross_border`, `other_measures`, `financial_consideration`, `fine_calc_summary`, `corrective_powers`, `processing_limitation_scope`, `compliance_deadline`, `ds_rights_discussed`, `ds_rights_violated`, `access_issues`, `adm_issues`, `dpo_appointment`, `dpo_issues`, `jurisdiction_complexity`, `data_transfers_discussed`, `transfer_violations_issues`, `precedent_significance`, `references_other_cases`, `edpb_references`, `case_summary`

The minimal CSV excludes the 9 metadata columns and keeps `ID` + the 68 answer columns.

### Normalization details

- Prefix stripping on answer values: `TYPE:`, `ENUM:`, `MULTI_SELECT:`, `FORMAT:` are removed where present.
- `country_code` is post-processed to keep only the ISO code, e.g., `FR`, `SE`.
- Numeric coercion removes spaces/currency symbols where possible; if a numeric cannot be inferred, the original token is left as-is (or left blank if explicitly `null`).

### Verifying outputs

You can quickly check that prefixes were fully removed:

```
grep -nE "\b(TYPE:|ENUM:|MULTI_SELECT:|FORMAT:|ISO_3166-1_ALPHA-2)" raw_data/responses/parsed_responses_min.csv | head -n 10 | cat
```

Expect no matches.

### Troubleshooting

- If the script reports 0 rows written, confirm the JSON files are valid arrays and contain `response` fields.
- If IDs do not sort as expected, ensure they follow the pattern `Prefix_Number` (e.g., `France_12`).
- If you need to exclude rows (e.g., `success != true`), add filtering where indicated in `main()`.

This project gathers information about DPA decisions on the GDPR for comparative analysis and statistical analysis.
Currently supports Norwegian and Swedish DPA decisions with comprehensive multi-country analytical capabilities.
The analysis is done in Python, as well as using any other tools which are well-known and widely used for the purposes of analysis.
First, we must decide which kind of analysis to start with. Phase1Plan.md contains the plan for the initial analysis. We must then decide on the tool stack. We should make a careful research plan, then carru out step by step, installing tools as needed, reviewing results, and iterating until we have a working analysis. We should track progress meticously. We must make results and analysis logged and human readable and generate charts.

# Data structure

The data is stored in a CSV file with the following columns:
ID,A1_Country,A2_Authority,A3_DecisionDate,A4_CaseTrigger,A5_CrossBorder,A6_DPARole,A7_IsAppeal,A8_AppealSuccess,A9_PriorInfringements,A10_DefendantCount,A11_DefendantName,A12_DefendantRole,A13_InstitutionalIdentity,A14_EconomicSector,A15_DefendantCategory,A16_SensitiveData,A17_SensitiveDataTypes,A18_Cookies,A19_VulnerableSubjects,A20_LegalBasis,A21_DataTransfers,A22_TransferMechanism,A23_HighRiskProcessing,A24_PowerImbalance,A25_SubjectsAffected,A26_InfringementDuration,A27_DamageEstablished,A28_DPIARequired,A29_SecurityMeasures,A30_DataMinimization,A31_SubjectRightsRequests,A32_RightsInvolved,A33_TransparencyObligations,A34_GDPREvaluated,A35_GDPRViolated,A36_NoInfringement,A37_LegalBasisInvalid,A38_NegligenceEstablished,A39_ManagementAuthorization,A40_DefendantCooperation,A41_MitigatingMeasures,A42_PriorNonCompliance,A43_FinancialBenefit,A44_PriorMeasuresConsidered,A45_SanctionType,A46_FineAmount,A47_FineCurrency,A48_EDPBGuidelines,A49_FineCalculationFactors,A50_CaseSummary

## Methodology (Academic Rigor)

- Variable typing follows documented schema. Binary fields treat `UNKNOWN/N_A` as missing; multi-select fields are expanded to one-hot indicators in `enhanced_data_cleaning.py`.
- Univariate: descriptive stats with CIs where applicable, normality checks (Shapiro; log-normal for positive), robust outliers via MAD and IQR; IsolationForest with dynamic contamination. Dates parsed with `dayfirst=True`.
- Bivariate: categorical×categorical uses χ²; for 2×2, Fisher’s exact is used; Cramér's V reported. Continuous×categorical uses Kruskal–Wallis (plus ANOVA for reference) with eta-squared. Continuous×continuous reports Pearson and Spearman.
- Multiple comparisons: Benjamini–Hochberg FDR within each family (cat×cat, cont×cat, cont×cont) with q-values reported.
- Reproducibility: results include timestamps, dataset dimensions, and analysis metadata; random components seeded.
- Caveats: Exploratory analyses; interpret inferential results using q-values and effect sizes; small-sample or sparse tables prefer exact tests.

## Usage

```bash
# Quality Assessment (supports --input for any country)
python data_quality_assessment.py --input dataNorway.csv
python data_quality_assessment.py --input dataSweden.csv

# Schema Validation (supports --input for any country)
python gdpr_schema_validation.py --input dataNorway.csv
python gdpr_schema_validation.py --input dataSweden.csv

# Data Cleaning (supports --input, --quality-threshold, --no-remove-low-quality)
python enhanced_data_cleaning.py --input dataNorway.csv --quality-threshold 0.3
python enhanced_data_cleaning.py --input dataSweden.csv --quality-threshold 0.3

# Multi-Format Export (auto-detects latest cleaned file)
python multi_format_exporter.py
```

### Analysis Scripts
```bash
# Univariate analysis (supports --data for any cleaned dataset)
python univariate_analyzer.py --data dataNorway_cleaned_YYYYMMDD_HHMMSS.csv
python univariate_analyzer.py --data dataSweden_cleaned_YYYYMMDD_HHMMSS.csv
python univariate_analyzer.py --data dataNordics_cleaned_combined.csv

# Bivariate analysis (optionally pass univariate results)
python bivariate_analyzer.py --data dataNorway_cleaned_YYYYMMDD_HHMMSS.csv --univariate univariate_analysis_results_YYYYMMDD_HHMMSS.json
python bivariate_analyzer.py --data dataSweden_cleaned_YYYYMMDD_HHMMSS.csv

# Sector analysis (private vs public)
python sector_analysis.py --data dataNorway_cleaned_YYYYMMDD_HHMMSS.csv
python sector_analysis.py --data dataSweden_cleaned_YYYYMMDD_HHMMSS.csv
```

### CLI Arguments Reference
- **data_quality_assessment.py**:
  - `--input PATH` - Input CSV file path (required)
- **gdpr_schema_validation.py**:
  - `--input PATH` - Input CSV file path (required)
- **enhanced_data_cleaning.py**:
  - `--input PATH` - Input CSV file path (required)
  - `--quality-threshold FLOAT` - Quality threshold 0-1 for row filtering (default: 0.3)
  - `--no-remove-low-quality` - Disable automatic removal of low-quality rows
- **Analysis scripts** (univariate_analyzer.py, bivariate_analyzer.py, sector_analysis.py):
  - `--data PATH` - Cleaned dataset file path (auto-detects latest if not specified)
  - `--univariate PATH` - Previous univariate results for bivariate analysis (optional)

### Derived features (after cleaning)
- Article indicators: `A34_Art_*`, `A35_Art_*` (1 if present, else NaN)
- One-hot for enums: `A4_*`, `A6_*`, `A8_*`, `A9_*`, `A12_*`, `A15_*`, `A27-*`, `A28-*`, `A29-*`, `A30-*`, `A33-*`, `A37-*` ... `A44-*`, `A49-*`
- Y/N binaries: `{field}_bin` for A5, A7, A16, A18, A21, A31, A36, A48 (UNKNOWN -> NaN)
- Tri-state positive bins: `{field}_pos_bin` for A23, A24, A27, A28, A29, A30, A33, A37–A43
- Multi-select one-hots (fixed vocab): `SensitiveType_*`, `VulnerableSubject_*`, `LegalBasis_*`, `TransferMech_*`, `Right_*`, `Sanction_*`
- Parsed fields: `A13_SNA_Code`, `A13_SNA_Desc`; `A14_ISIC_Code`, `A14_ISIC_Desc`, `A14_ISIC_Level`
- Numericized counts: `A25_*` (min/max/midpoint/is_range)
- Duration: `A26_Duration_Months`
- Currency normalization: `A46_FineAmount_EUR`
- Provenance: `dataset_source`

### Combine multiple countries
```python
import pandas as pd
df_no = pd.read_csv('dataNorway_cleaned_YYYYMMDD_HHMMSS.csv')
df_se = pd.read_csv('dataSweden_cleaned_YYYYMMDD_HHMMSS.csv')
cols = sorted(set(df_no.columns).union(df_se.columns))
combined = pd.concat([df_no.reindex(columns=cols), df_se.reindex(columns=cols)], ignore_index=True)
combined.to_csv('dataNordics_cleaned_combined.csv', index=False)
```

### Multi-Format Export
The project supports exporting cleaned data to multiple formats with comprehensive metadata:

```bash
# Export latest cleaned dataset to all formats
python multi_format_exporter.py

# Outputs:
# - CSV: Standard format for statistical analysis
# - Excel: Multi-sheet workbook with data dictionary and summary statistics
# - Parquet: Efficient binary format for big data pipelines
# - JSON: Web-ready format with embedded metadata
```

**Export Features**:
- Automatic detection of latest cleaned files
- Comprehensive metadata preservation
- Data dictionary generation
- Processing audit trails
- Multi-country support
- Versioned output files

Column data was gathered using the following questionnaire:
**Question 1:** What is the country of the deciding DPA?\n\n**Answer 1:** `[AT|BE|BG|HR|CY|CZ|DE|DK|EE|ES|FI|FR|GR|HU|IE|IS|IT|LI|LT|LU|LV|MT|NL|NO|PL|PT|RO|SE|SI|SK|EU|UNKNOWN]`\n\n**Question 2:** What is the full name of the deciding authority?\n\n**Answer 2:** `Free text (e.g., \"Datatilsynet\")`\n\n**Question 3:** What was the date of the decision?\n\n**Answer 3:** `Date in DD-MM-YYYY format`\n\n## **Section 2: Procedural Context**\n\n**Question 4:** What was the explicit trigger of the case?\n\n**Answer 4:** `[COMPLAINT|BREACH_NOTIFICATION|EX_OFFICIO_INVESTIGATION|OTHER|UNKNOWN]`\n\n**Question 5:** Is this a cross-border case?\n\n**Answer 5:** `[Y|N|UNKNOWN]`\n\n**Question 6:** If cross-border, what was the role of the deciding DPA?\n\n**Answer 6:** `[Lead_Authority|Concerned_Authority|N_A]`\n\n**Question 7:** Is this case an appeal of a prior decision?\n\n**Answer 7:** `[Y|N]`\n\n**Question 8:** If yes to Q7, was the appeal successful?\n\n**Answer 8:** `[Successful|Unsuccessful|N_A]`\n\n**Question 9:** Did the decision explicitly reference previous infringements by the same defendant?\n\n**Answer 9:** `[PRIOR_INFRINGER_PRESENT|DISCUSSED_NO_PRIOR_FOUND|NOT_DISCUSSED|UNKNOWN]`\n\n## **Section 3: Defendant Information**\n\n**Question 10:** How many defendants were there in this case?\n\n**Answer 10:** `An integer (e.g., 1)`\n\n**Question 11:** What is the name of the primary defendant?\n\n**Answer 11:** `Free text (e.g., \"Global Tech Inc.\")`\n\n**Question 12:** What was the role of the primary defendant?\n\n**Answer 12:** `[Controller|Processor|Joint_Controller|UNKNOWN]`\n\n**Question 13:** What is the institutional identity (SNA) of the defendant is mentioned or can be inferred with high confidence?\n\n**Answer 13:** `Provide the SNA code and its description (e.g., \"S.11 - Non-financial corporations\"); UNKNOWN if low confidence`\n\n**Question 14:** What is the economic sector/activity (ISIC Rev.4) of the defendant which is mentioned or can be inferred with high confidence?\n\n**Answer 14:** `Provide the ISIC code, its description, and its classification level (e.g., \"Code: 6201, Description: Computer programming activities, Level: 4\"); UNKNOWN if low confidence even after generalizing a level`\n\n**Question 15:** What is the explicitly identified top-level category of the primary defendant?\n\n**Answer 15:** `[PUBLIC_AUTHORITY|BUSINESS|NON_PROFIT|INDIVIDUAL|OTHER|UNKNOWN]`\n\n## **Section 4: Data Processing Context -- based on explicit DPA statements**\n\n**Question 16:** Did the case involve processing of sensitive/special category data?\n\n**Answer 16:** `[Y|N|UNKNOWN]`\n\n**Question 17:** If yes to Q16, what type(s) of sensitive data were involved?\n\n**Answer 17:** `Select one or more: [Health|Biometric|Genetic|Racial_Ethnic|Political|Religious|Philosophical|Trade_Union|Sexual|Criminal|Other|N_A]`\n\n**Question 18:** Did the case involve cookies or similar tracking technologies under the ePrivacy Directive?\n\n**Answer 18:** `[Y|N|UNKNOWN]`\n\n**Question 19:** Were vulnerable data subjects explicitly mentioned?\n\n**Answer 19:** `Select one or more: [Children|Elderly|Employees|Patients|Students|Other_Vulnerable|None_Mentioned|UNKNOWN]`\n\n**Question 20:** What legal basis was relied upon by the defendant?\n\n**Answer 20:** `Select one or more: [Consent|Contract|Legal_Obligation|Vital_Interests|Public_Task|Legitimate_Interest|Not_Specified|UNKNOWN]`\n\n**Question 21:** Did the case involve international data transfers?\n\n**Answer 21:** `[Y|N|UNKNOWN]`\n\n**Question 22:** If yes to Q21, what transfer mechanism was used?\n\n**Answer 22:** `Select one or more: [Adequacy_Decision|Standard_Contractual_Clauses|Binding_Corporate_Rules|Certification|Derogation|None_Specified|N_A]`\n\n**Question 23:** Was this explicitly characterized as high-risk processing?\n\n**Answer 23:** `[Y|N|Not_Assessed|UNKNOWN]`\n\n**Question 24:** Was a power imbalance between controller and data subjects noted?\n\n**Answer 24:** `[Y|N|Not_Discussed|UNKNOWN]`\n\n## **Section 5: Scale and Impact**\n\n**Question 25:** How many data subjects were affected?\n\n**Answer 25:** `An integer, or ranges like \"1000-5000\", or \"UNKNOWN\"`\n\n**Question 26:** What was the duration of the infringement?\n\n**Answer 26:** `Free text (e.g., \"6 months\", \"ongoing since 2018\")`\n\n**Question 27:** Was material or non-material damage to data subjects established?\n\n**Answer 27:** `[Material_Damage|Non_Material_Damage|Both|None|Not_Assessed|UNKNOWN]`\n\n## **Section 6: Technical and Organizational Measures**\n\n**Question 28:** Was a Data Protection Impact Assessment (DPIA) required?\n\n**Answer 28:** `[Required_And_Conducted|Required_Not_Conducted|Not_Required|Not_Assessed|UNKNOWN]`\n\n**Question 29:** Were technical and organizational security measures evaluated?\n\n**Answer 29:** `[Adequate|Inadequate|Not_Evaluated|UNKNOWN]`\n\n**Question 30:** Was data minimization principle explicitly evaluated?\n\n**Answer 30:** `[Compliant|Violated|Not_Evaluated|UNKNOWN]`\n\n## **Section 7: Data Subject Rights**\n\n**Question 31:** Did the case involve data subject rights requests?\n\n**Answer 31:** `[Y|N|UNKNOWN]`\n\n**Question 32:** If yes to Q31, which rights were involved?\n\n**Answer 32:** `Select one or more: [Access|Rectification|Erasure|Portability|Objection|Restrict_Processing|Automated_Decision_Making|N_A]`\n\n**Question 33:** Were transparency obligations (information provision) evaluated?\n\n**Answer 33:** `[Compliant|Violated|Not_Evaluated|UNKNOWN]`\n\n## **Section 8: Compliance Evaluation**\n\n**Question 34:** Which GDPR articles were evaluated for compliance?\n\n**Answer 34:** `A list of article numbers, separated by commas (e.g., \"Art. 5(1)(a), Art. 6, Art. 32\")`\n\n**Question 35:** Which GDPR articles were deemed to be violated?\n\n**Answer 35:** `A list of article numbers, separated by commas (e.g., \"Art. 5(1)(f), Art. 13\")`\n\n**Question 36:** Was there an explicit finding of no infringement?\n\n**Answer 36:** `[Y|N]`\n\n**Question 37:** Was the legal basis directly deemed invalid or insufficient?\n\n**Answer 37:** `[Y|N|Not_Evaluated|UNKNOWN]`\n\n## **Section 9: Defendant Conduct and Mitigating/Aggravating Factors**\n\n**Question 38:** Was negligence explicitly established?\n\n**Answer 38:** `[Y|N|Not_Assessed|UNKNOWN]`\n\n**Question 39:** Was high-level management authorization of unlawful processing established?\n\n**Answer 39:** `[Y|N|Not_Assessed|UNKNOWN]`\n\n**Question 40:** Did the defendant cooperate with the investigation?\n\n**Answer 40:** `[Cooperative|Partially_Cooperative|Uncooperative|Not_Discussed|UNKNOWN]`\n\n**Question 41:** Did the defendant implement mitigating measures?\n\n**Answer 41:** `[Before_Investigation|During_Investigation|None|Not_Discussed|UNKNOWN]`\n\n**Question 42:** Was non-compliance with previous DPA orders established?\n\n**Answer 42:** `[Y|N|Not_Assessed|UNKNOWN]`\n\n**Question 43:** Did the defendant derive financial benefit from the infringement?\n\n**Answer 43:** `[Y|N|Not_Assessed|UNKNOWN]`\n\n**Question 44:** Were measures taken prior to this finding considered in the decision?\n\n**Answer 44:** `[Mitigating_Factor|Aggravating_Factor|Neutral|Not_Considered|N_A]`\n\n## **Section 10: Sanctions and Fine Calculation**\n\n**Question 45:** What type of sanction or corrective measure was imposed?\n\n**Answer 45:** `Select one or more: [Fine|Warning|Reprimand|Compliance_Order|Processing_Ban_Temporary|Processing_Ban_Permanent|Data_Flow_Suspension|Other_Corrective_Measure|None]`\n\n**Question 46:** If a fine was imposed, what was the cumulative amount?\n\n**Answer 46:** `An integer, with no commas or currency symbols (e.g., 100000)`\n\n**Question 47:** If a fine was imposed, what was the currency?\n\n**Answer 47:** `A three-letter currency code (e.g., \"EUR\")`\n\n**Question 48:** Did the decision explicitly reference EDPB guidelines on fines?\n\n**Answer 48:** `[Y|N|UNKNOWN]`\n\n**Question 49:** Were fine calculation factors from the GDPR assessed largely systematically and  individually in the decision?\n\n**Answer 49:** `[Y|Partially|N|N_A]`\n\n**Question 50:** Provide a short summary of the case skimmable to a busy expert - key facts, issue, DPA reasoning.\n\n**Answer 50:** `Free text, 3 sentences maximum`

# Column values

# Field Allowed Values Overview

**ID**: Unique identifier  
**A1_Country**: AT|BE|BG|HR|CY|CZ|DE|DK|EE|ES|FI|FR|GR|HU|IE|IS|IT|LI|LT|LU|LV|MT|NL|NO|PL|PT|RO|SE|SI|SK|EU|UNKNOWN  
**A2_Authority**: Free text (authority name)  
**A3_DecisionDate**: DD-MM-YYYY date format  
**A4_CaseTrigger**: COMPLAINT|BREACH_NOTIFICATION|EX_OFFICIO_INVESTIGATION|OTHER|UNKNOWN  
**A5_CrossBorder**: Y|N|UNKNOWN  
**A6_DPARole**: Lead_Authority|Concerned_Authority|N_A  
**A7_IsAppeal**: Y|N  
**A8_AppealSuccess**: Successful|Unsuccessful|N_A  
**A9_PriorInfringements**: PRIOR_INFRINGER_PRESENT|DISCUSSED_NO_PRIOR_FOUND|NOT_DISCUSSED|UNKNOWN  
**A10_DefendantCount**: Integer (number of defendants)  
**A11_DefendantName**: Free text (defendant name)  
**A12_DefendantRole**: Controller|Processor|Joint_Controller|UNKNOWN  
**A13_InstitutionalIdentity**: SNA code + description or UNKNOWN  
**A14_EconomicSector**: ISIC code + description + level or UNKNOWN  
**A15_DefendantCategory**: PUBLIC_AUTHORITY|BUSINESS|NON_PROFIT|INDIVIDUAL|OTHER|UNKNOWN  
**A16_SensitiveData**: Y|N|UNKNOWN  
**A17_SensitiveDataTypes**: Health|Biometric|Genetic|Racial_Ethnic|Political|Religious|Philosophical|Trade_Union|Sexual|Criminal|Other|N_A (multiple selection)  
**A18_Cookies**: Y|N|UNKNOWN  
**A19_VulnerableSubjects**: Children|Elderly|Employees|Patients|Students|Other_Vulnerable|None_Mentioned|UNKNOWN (multiple selection)  
**A20_LegalBasis**: Consent|Contract|Legal_Obligation|Vital_Interests|Public_Task|Legitimate_Interest|Not_Specified|UNKNOWN (multiple selection)  
**A21_DataTransfers**: Y|N|UNKNOWN  
**A22_TransferMechanism**: Adequacy_Decision|Standard_Contractual_Clauses|Binding_Corporate_Rules|Certification|Derogation|None_Specified|N_A (multiple selection)  
**A23_HighRiskProcessing**: Y|N|Not_Assessed|UNKNOWN  
**A24_PowerImbalance**: Y|N|Not_Discussed|UNKNOWN  
**A25_SubjectsAffected**: Integer|ranges (e.g., "1000-5000")|UNKNOWN  
**A26_InfringementDuration**: Free text (duration description)  
**A27_DamageEstablished**: Material_Damage|Non_Material_Damage|Both|None|Not_Assessed|UNKNOWN  
**A28_DPIARequired**: Required_And_Conducted|Required_Not_Conducted|Not_Required|Not_Assessed|UNKNOWN  
**A29_SecurityMeasures**: Adequate|Inadequate|Not_Evaluated|UNKNOWN  
**A30_DataMinimization**: Compliant|Violated|Not_Evaluated|UNKNOWN  
**A31_SubjectRightsRequests**: Y|N|UNKNOWN  
**A32_RightsInvolved**: Access|Rectification|Erasure|Portability|Objection|Restrict_Processing|Automated_Decision_Making|N_A (multiple selection)  
**A33_TransparencyObligations**: Compliant|Violated|Not_Evaluated|UNKNOWN  
**A34_GDPREvaluated**: Comma-separated list of GDPR article numbers  
**A35_GDPRViolated**: Comma-separated list of GDPR article numbers  
**A36_NoInfringement**: Y|N  
**A37_LegalBasisInvalid**: Y|N|Not_Evaluated|UNKNOWN  
**A38_NegligenceEstablished**: Y|N|Not_Assessed|UNKNOWN  
**A39_ManagementAuthorization**: Y|N|Not_Assessed|UNKNOWN  
**A40_DefendantCooperation**: Cooperative|Partially_Cooperative|Uncooperative|Not_Discussed|UNKNOWN  
**A41_MitigatingMeasures**: Before_Investigation|During_Investigation|None|Not_Discussed|UNKNOWN  
**A42_PriorNonCompliance**: Y|N|Not_Assessed|UNKNOWN  
**A43_FinancialBenefit**: Y|N|Not_Assessed|UNKNOWN  
**A44_PriorMeasuresConsidered**: Mitigating_Factor|Aggravating_Factor|Neutral|Not_Considered|N_A  
**A45_SanctionType**: Fine|Warning|Reprimand|Compliance_Order|Processing_Ban_Temporary|Processing_Ban_Permanent|Data_Flow_Suspension|Other_Corrective_Measure|None (multiple selection)  
**A46_FineAmount**: Integer (no commas/currency symbols)  
**A47_FineCurrency**: Three-letter currency code (e.g., EUR)  
**A48_EDPBGuidelines**: Y|N|UNKNOWN  
**A49_FineCalculationFactors**: Y|Partially|N|N_A  
**A50_CaseSummary**: Free text (max 3 sentences)