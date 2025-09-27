# Format Conversion Review

## Context
- Cleaning pipeline emphasizes non-destructive handling of typed missingness and preservation of legal nuance across every transformation stage.【F:data-cleaning.md†L1-L135】【F:scripts/README.md†L1-L103】
- Advanced export scripts extend the cleaned wide and long tables into Parquet, Arrow, graph/network, statistical package, and machine-learning friendly formats.

## Key Findings

### 1. Parquet export mutates canonical data when preparing partitions
- Missing partition values are filled in-place with sentinel values (`'UNKNOWN'`/`-999`) before writing the dataset, overwriting the canonical DataFrame copy that should preserve typed missingness.【F:scripts/export/parquet_export.py†L36-L55】
- Rebuilding the Arrow table after this mutation discards the rich schema metadata that `_create_arrow_table` attaches, leading to metadata loss in the emitted partitioned dataset.【F:scripts/export/parquet_export.py†L42-L63】【F:scripts/export/parquet_export.py†L128-L156】
- Impact: downstream consumers receive corrupted values for `country_group`, `decision_year`, etc., breaking the non-destructive guarantee and masking true null/`pd.NA` states that encode legal nuance.

### 2. Graph export treats nullable booleans as plain truthy values
- Node attributes such as `breach_case`, `severity_measures`, and the derived `cross_border` flag coerce nullable booleans via `bool(...)`, which raises on `pd.NA` or silently treats "unknown" as `False`.【F:scripts/export/graph_export.py†L88-L109】
- Cross-border detection relies on `row.get('q49')`/`row.get('q62')` even though the cleaned dataset exposes `raw_q49`/`raw_q62`, so the per-node `cross_border` attribute never reflects the detected cooperation markers.【F:scripts/export/graph_export.py†L223-L256】

- Impact: exporters either crash on legitimate `pd.NA` values or emit misleading graph attributes, erasing information about uncertain or cross-border cases that analysts rely on.

### 3. ML export collapses typed missingness and targets during imputation
- Geographic, case-type, and status-derived features call `.astype(int)` on nullable Boolean/Categorical comparisons without guarding for `pd.NA`, forcing ambiguous states into implicit zeros or raising at runtime.【F:scripts/export/ml_export.py†L76-L117】
- `_combine_features` fills every numeric column—including targets such as `fine_amount_log`—with the median, eliminating legal distinctions between "not assessed" and genuine numeric values and causing leakage in downstream modeling.【F:scripts/export/ml_export.py†L233-L253】
- Impact: the ML-ready outputs both destroy the legal semantics around missingness and blur supervised targets, inviting biased training sets contrary to the pipeline's guarantees.

## Recommended Action Plan

1. **Partition-safe Parquet writing (priority: high)**
   - Clone partition columns solely for path derivation (e.g., `partition_values = df[col].fillna('__MISSING__')`) while keeping the stored column untouched, or use Arrow datasets with computed partition expressions.【F:scripts/export/parquet_export.py†L36-L63】
   - Persist the metadata-rich schema by writing the table returned from `_create_arrow_table` directly via `pq.write_table`/`pq.write_to_dataset` after updating only the partition paths.【F:scripts/export/parquet_export.py†L42-L63】【F:scripts/export/parquet_export.py†L128-L156】
   - Add regression tests confirming null counts remain identical between the source CSV and Parquet outputs.

2. **Robust nullable handling in graph exports (priority: high)**
   - Replace `bool(...)` coercions with explicit checks (`pd.isna` guards, `.astype('boolean')`) and propagate `None` where legal status is unknown, matching the cleaning guidelines.【F:data-cleaning.md†L57-L105】【F:scripts/export/graph_export.py†L88-L109】
   - Align cross-border detection with the available columns (`raw_q49`, `raw_q62`) and document how mixed-status cases are represented in edge attributes.【F:scripts/export/graph_export.py†L223-L256】
   - Extend metadata export to flag counts of indeterminate nodes so analysts know when legal meaning is unavailable.

3. **Maintain typed missingness in ML exports (priority: high)**
   - Convert nullable comparisons using `.astype('Int64')` after filling with zero or better, emit paired "missing" indicators, and avoid overwriting targets during imputation.【F:scripts/export/ml_export.py†L76-L117】【F:scripts/export/ml_export.py†L233-L253】
   - Provide configuration to leave NaNs intact (letting downstream pipelines decide) and surface masks derived from `_status` columns to preserve legal nuance.【F:data-cleaning.md†L57-L105】
   - Add validation that the distribution of status codes/NaNs matches the source prior to writing train/test splits.

4. **Cross-format consistency checks (priority: medium)**
   - Implement automated round-trip comparisons (CSV → export → read-back) for key fields to detect discrepancies in counts, null ratios, and categorical levels across Parquet, Arrow, and ML datasets.
   - Document expected sentinel values (if any) and ensure they never collide with real enumerations (e.g., avoid `UNKNOWN` if it's a legitimate response option).

5. **Metadata parity across exports (priority: medium)**
   - Ensure every export reuses a shared metadata builder so that legal warnings, question descriptions, and typed missingness notes remain synchronized across Parquet, Arrow, stats, and ML outputs.【F:scripts/export/parquet_export.py†L128-L199】【F:scripts/export/arrow_export.py†L120-L212】
   - Publish a changelog entry describing the safeguards so stakeholders understand the preservation guarantees.

Addressing these items will keep advanced exports aligned with the project's non-destructive mandate and protect analysts from silent data loss.
