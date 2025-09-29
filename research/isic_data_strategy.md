# ISIC Integration Strategy for DPA Enforcement Analytics

## 1. Data landscape and current assets
- **Source repositories** – Raw decisions and AI annotations live in `raw-data/` and `analyzed-decisions/`; both feed the cleaning pipeline via the merged canonical CSV in `raw-data/LATEST_MASTER_ONLY_USE_THIS_MERGED.csv` and the legacy AI extract at `analyzed-decisions/master-analyzed-data-unclean.csv`.【F:README.md†L9-L16】
- **Cleaning and analytics scripts** – The CLI workflow (`scripts/cli.py`) orchestrates enum whitelist building, wide-table cleaning, long-table emission, and QA steps before higher-level analytics run.【F:scripts-readme.md†L5-L32】
- **ISIC reference** – A structured Rev.4 lookup table under `resources/ISIC_Rev_4_english_structure.txt` provides section letters (A–U) and granular class descriptions to attach sector metadata to responses.【F:resources/ISIC_Rev_4_english_structure.txt†L1-L40】
- **Evenness phases** – Phases 0–3 span omni-scan diagnostics, factual matching, uniformity tests, and policy-oriented modelling via `scripts/evenness/cli.py` and its subcommands.【F:README.md†L5-L7】【F:scripts-readme.md†L35-L66】【F:scripts/evenness/cli.py†L255-L325】

## 2. Existing data cleaning relevant to ISIC
- **Schema-aware ingestion** – `clean_csv_to_wide` parses each response, strips schema echoes, normalises countries/DPAs, and captures raw answers for Q1–Q68 alongside parser metadata, ensuring reproducible provenance for later ISIC enrichment.【F:scripts/clean/wide_output.py†L152-L352】
- **ISIC parsing** – When Q12 contains a code, the pipeline strips any template residue, looks up the value against the Rev.4 index, and records the normalised code, description, and section letter so downstream analytics can use harmonised sectors.【F:scripts/clean/wide_output.py†L268-L333】【F:scripts/clean/isic_map.py†L20-L55】
- **Multi-select expansion** – Enumerated and multi-select questions are expanded into systematic indicator columns with coverage/status metadata, providing ready-made one-hot features for sectoral cross-tabs (e.g., organisation class, breach types, powers applied).【F:scripts/clean/wide_output.py†L181-L406】
- **Numeric and textual derived fields** – Fine and turnover values are converted to floats/logs with validation flags, text answers are normalised with language heuristics, and aggregate counts such as `n_principles_violated` are computed for modelling.【F:scripts/clean/wide_output.py†L250-L406】

## 3. Proposed ISIC data preparation workflow
1. **Validate and normalise codes**
   - Run the existing wide-table cleaner on the latest merged source, ensuring the enum whitelist is current. This guarantees Q12 values pass through `strip_schema_echo` and the ISIC index lookup, flagging unparseable or multi-sector entries for manual review.【F:scripts/clean/wide_output.py†L268-L389】
   - Augment the validation report by tagging decisions where `isic_section` remains blank but `isic_code` was provided, signalling codes that may need reference-table updates.
2. **Enhance the ISIC reference**
   - Extend `IsicIndex` to record 2-digit division names alongside 4-digit classes, enabling multi-level aggregation (Section → Division → Group). The existing loader already collapses to divisions if a class is missing, so adding division descriptors will align text labels across resolutions.【F:scripts/clean/isic_map.py†L20-L55】
   - Version the lookup file (e.g., include a `isic_reference_version` column in outputs) to track updates over time for reproducibility.
3. **Derive hierarchical features**
   - In the wide dataset, create binary indicators for each ISIC section (A–U) and optional division-level features via a new helper (e.g., `expand_isic_hierarchy`). Store these in `outputs/cleaned_wide.csv` so they feed both descriptive tables and the evenness toolkit.
   - Capture multi-sector cases by splitting additional codes into a long table (`decision_id`, `isic_code`, `isic_level`) for network analyses and sector co-occurrence studies, following the guidance in `data-cleaning.md` for long-form structuring.【F:data-cleaning.md†L153-L209】
4. **Bridge to organisational context**
   - Pair ISIC features with existing organisation class indicators (`q10_org_class_*`) to build combined profiles (e.g., public authority in Section O). This can be implemented during feature matrix construction to maintain tidy semantics.【F:scripts/clean/wide_output.py†L181-L406】【F:scripts/analysis/build_feature_matrix.py†L12-L172】
5. **Quality assurance hooks**
   - Expand the validation JSON with ISIC-specific flags (`missing_section`, `ambiguous_multi_sector`) and surface summary stats in `qa_summary` so gaps are visible before modelling.

## 4. Format conversions and deliverables
- **Wide table** – Continue exporting `outputs/cleaned_wide.csv` with enriched ISIC fields plus any new binary indicators and hierarchy columns. Maintain the schema echo flags and raw question copies for audit trails.【F:scripts/clean/wide_output.py†L152-L422】
- **Long-form sector table** – Introduce `outputs/long_tables/isic_assignments.csv` generated by an updated `emit-long` command, mirroring the existing approach for rights and powers. This supports tidy analyses and joins in notebooks/dashboards.【F:data-cleaning.md†L177-L209】
- **Feature matrix** – Ensure `scripts.analysis.build_feature_matrix` ingests the new section/division indicators and exposes them in `feature_matrix.parquet` for modelling and report generation.【F:scripts/analysis/build_feature_matrix.py†L12-L172】

## 5. Analytical strategy leveraging ISIC across phases
### Phase 0 – Omni-scan (big picture)
- Use the expanded ISIC indicators to inspect feature coverage and importance in the omni-scan outputs (`coverage_ledger.csv`, SHAP summaries). Pay attention to sparsity by section to decide if pooling (e.g., combining low-frequency divisions) is warranted.【F:scripts-readme.md†L45-L66】
- Evaluate cross-links between ISIC sections and high-impact features (sensitive data types, corrective powers) using interaction maps to spot sector-specific enforcement patterns.

### Phase 1 – Matching foundation (comparative baselines)
- Incorporate ISIC section/division flags into the matching feature set where feasible, enabling within-sector comparisons of fines and remedies. Check balance diagnostics to ensure matches are not dominated by high-frequency sectors.【F:scripts/evenness/cli.py†L120-L136】
- For low-sample sections, rely on pooled indicators or supplement with organisation-class proxies to maintain overlap.

### Phase 2 – Uniformity tests (calibration)
- When reviewing residuals and parity metrics, segment the outputs by ISIC section to assess whether enforcement uniformity holds across industries. Store sector-specific calibration plots to identify outlier sectors needing qualitative follow-up.【F:scripts/evenness/cli.py†L271-L279】

### Phase 3 – Explanation and policy (multivariate focus)
- Ensure ISIC section binaries flow into the driver leaderboard, decompositions, and policy levers so the toolkit reports sector differentials explicitly. The Phase 3 runner already scans columns prefixed with `ISIC_SECTION_`; populate these during feature preparation to benefit from FDR-adjusted effect estimates.【F:scripts/evenness/phase_three.py†L28-L102】【F:scripts/evenness/cli.py†L281-L324】
- Use Oaxaca-Blinder decomposition by sector groups (e.g., compare Sections J vs G) to quantify component contributions to fine disparities, leveraging `cmd_decompose` with `group_col="isic_section"`.

### Complementary analytics
- Feed the enriched feature matrix into breach-notification diagnostics and hierarchical severity models, treating ISIC section as both a fixed effect and a grouping variable for heterogeneity tests.【F:scripts/analysis/build_feature_matrix.py†L12-L172】【F:scripts/evenness/cli.py†L138-L175】
- Build sector-by-remedy dashboards (counts, fines, notification behaviours) using the long ISIC table to support qualitative case studies alongside quantitative models.

## 6. Operational recommendations
- **Version control** – Check `resources/ISIC_Rev_4_english_structure.txt` into data lineage tracking whenever updated and note the version in outputs to maintain reproducibility.【F:resources/ISIC_Rev_4_english_structure.txt†L1-L40】
- **Documentation** – Update `data-cleaning.md` and `scripts-readme.md` with any new ISIC-specific commands or columns so collaborators understand how to regenerate artefacts.【F:data-cleaning.md†L153-L209】【F:scripts-readme.md†L5-L74】
- **Testing** – Extend `tests/test_parser_and_clean.py` with fixtures covering multi-sector inputs and section-level fallbacks to guard against regressions in the ISIC mapping logic.
- **Pipeline automation** – Consider adding a dedicated CLI switch (e.g., `python -m scripts.cli enrich-isic`) that refreshes the hierarchy features and validation summaries after the wide clean completes, simplifying reruns for analysts.

With this approach, ISIC data becomes a first-class dimension across descriptive, comparative, and multivariate analyses, enabling consistent sector insights from ingestion through Phase 3 policy evaluations.
