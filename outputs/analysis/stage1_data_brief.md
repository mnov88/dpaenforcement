# Stage 1 coverage summary

## Status distributions
### breach_types_status
- NOT_APPLICABLE: 1461
- DISCUSSED: 365
- MISSING: 81
- NOT_DISCUSSED: 69
- UNCLEAR: 8
- MIXED_STATUS: 1
- NOT_MENTIONED: 1

### special_data_status
- NOT_APPLICABLE: 1197
- DISCUSSED: 607
- MISSING: 81
- NOT_DISCUSSED: 70
- NOT_MENTIONED: 22
- UNCLEAR: 8
- MIXED_STATUS: 1

### vulnerable_status
- DISCUSSED: 1046
- NONE_MENTIONED: 859
- NOT_MENTIONED: 81

### remedial_status
- DISCUSSED: 969
- NONE_MENTIONED: 927
- NOT_MENTIONED: 81
- NOT_APPLICABLE: 9

## Country harmonisation
- Mapped 'COUNTRY OF THE DECIDING AUTHORITY: IRELAND (IE)' ? IE
- UK codes harmonised to GB

## Weight notes
- `country_weight` rescales each country to equal total weight across the sample
- `country_year_weight` equalises country-year cells where decision_year is observed

## Storage format
- Saved design matrices to CSV and Parquet to preserve downstream compatibility
