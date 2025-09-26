 Guiding principles

- Preserve legal nuance: treat “NOT_DISCUSSED,” “NOT_MENTIONED,” “NOT_APPLICABLE,” “UNCLEAR,” “NOT_DETERMINED,” and “NO_FINE_IMPOSED” as distinct states rather than generic missing values. Absence of discussion is not evidence of absence of a violation.

- Ensure reproducibility and auditability: retain both the raw answers and normalized representations. Log every transformation, with validation and conflict flags.

- Build for both EDA and advanced analysis: provide tidy single-row-per-decision tables plus normalized “long”/relational tables for multi-selects and entities; produce derived features for modeling while keeping original categorical richness.

- Validate against questionnaire logic and legal reality: run cross-field consistency checks, but never “auto-correct” to change legal meaning; instead flag improbable or contradictory states.

A. Ingestion and parsing (exact format)

1) Identify individual decisions:

- The data may come as a concatenation of decision blocks. Split records whenever a new “Answer 1:” appears, after first trimming surrounding quotes and normalizing line endings.

- Requirements: each record should contain Answer 1 through Answer 68. If not, mark record as incomplete; allow partial records with a completeness flag.

2) Extract (Answer N, value) pairs:

- Regex to capture: ^"?Answer\s+(\d+):\s*(.+?)\s*$ (line-anchored). Strip enclosing quotes and trailing commas.

- For Answer 1 (country): the value prefix includes a descriptor (ISO_3166-1_ALPHA-2:). Normalize by extracting the code after the final colon, e.g., “GR” from “ISO_3166-1_ALPHA-2: GR”.

3) Normalize value delimiters:

- Multi-select answers: split on commas, trim whitespace, preserve original order; also deduplicate tokens if repeated.

- Normalize case strictly to the enumerated vocabulary (upper snake case as in the schema). If an unrecognized token appears, keep it as raw_token and flag validation_error.

4) Attach metadata:

- Store per-record: source_id, ingestion_timestamp, parser_version, line_count, completeness, and any parser_warnings.

B. Canonical typing and categorical harmonization

5) Apply declared types to each question:

- TYPE:STRING fields (Q2, Q12, Q36, Q52, Q67, Q68): store both raw and normalized (Unicode NFC; trim; collapse internal runs of whitespace to single space; preserve punctuation; detect and keep language if possible). Do not lemmatize or casefold by default; retain original capitalization for legal citations.

- TYPE:NUMBER fields (Q37 fine, Q38 turnover): cast to decimal EUR; allow 0 for “no fine” (Q37) but store a boolean no_fine = (Q37 == 0). For Q38, null allowed (not mentioned). Store an additional numeric_valid flag and raw_value if parsing fails.

- Dates (Q3): parse YYYY-MM-DD, else leave as null and set date_status = NOT_DISCUSSED (distinct from null); create derived year, quarter, and ISO week. Validate plausible ranges (2018+ for GDPR decisions, unless legacy). If NOT_DISCUSSED, do not coerce to null without noting status.

- Enumerations: map exactly to allowed ENUM labels. Values like EU and UNCLEAR for Q1 are allowed; validate against whitelist.

6) Multi-select expansion into binary indicators:

- For each multi-select question, create:

- A long table: decision_id, question_id, option, present (1/0).

- A wide matrix of boolean columns (one per option), e.g., Q30_SECURITY = 1 if “SECURITY” selected.

- Preserve logical exclusivities when applicable (see D.9 below).

7) Retain typed missingness:

- Maintain two layers:

- value columns for substantive answers (e.g., Q31_SECURITY_violation = 1/0/null).

- paired status columns (e.g., Q31_status) carrying codes: DISCUSSED, NONE_VIOLATED, NOT_DETERMINED, NOT_APPLICABLE, NOT_MENTIONED, NOT_DISCUSSED, UNCLEAR.

- This allows modeling that correctly distinguishes “no finding” from “not assessed” and “not applicable.”

C. Legal logic and interpretive safeguards

8) Preserve the legal distinction between discussion, finding, and absence:

- Example (Articles 5/6):

- Q30 (discussed principles) indicates scope of assessment.

- Q31 (violated principles) indicates findings.

- If a principle is not in Q30 but is in Q31, flag a consistency_warning but do not override; the text may embed findings outside a labeled “discussion” section.

- Never infer “no violation” from “NOT_DISCUSSED” or “NOT_MENTIONED.”

- Example (Rights, Q56–Q57):

- NONE_DISCUSSED ≠ NONE_VIOLATED. Treat NONE_VIOLATED as an explicit negative finding; treat NONE_DISCUSSED as unassessed.

- Breach logic (Q16–Q29):

- If Q16 = NO and later answers include breach-specific content (e.g., Q25 ARTICLE_9_SPECIAL_CATEGORY), do not coerce; flag contradiction. Some decisions discuss hypotheticals.

9) Encode exclusivities and precedence carefully:

- For mutually exclusive options like NOT_APPLICABLE or NONE_MENTIONED in multi-select:

- If they co-occur with substantive options, retain substantive options, set na_conflict_flag = 1, and set the status to MIXED/CONTRADICTORY for auditing.

- For appeal logic:

- If Q4 = NO, Q5 should be NOT_APPLICABLE. If not, flag.

- For fines and caps:

- If Q37 = 0 but Q39 indicates hitting a cap, flag contradiction.

- If Q38 (turnover) is null, Q39 = HIT_4PCT_TURNOVER_CAP is unverifiable; keep as reported but set verification_possible = 0.

D. Cross-field consistency and validation checks

10) Logical checks to flag (not fix):

- Breach notification chain:

- If Q17 = YES_REQUIRED and Q18 = NOT_APPLICABLE → flag.

- If Q18 indicates submission and Q16 = NO → flag (unless decision discusses hypothetical compliance).

- Article 34 vs 33:

- If Q27 = YES_REQUIRED but Q16 ≠ YES → flag.

- Data subject notifications:

- If Q26 = YES_NOTIFIED but Q16 = NO → flag.

- DPO:

- If Q60 = YES_REQUIRED_NOT_APPOINTED and Q61 does not include NO_DPO_APPOINTED → flag.

- Enforcement:

- If Q53 includes ADMINISTRATIVE_FINE then Q37 should be > 0 unless a lawful fine of 0 is explicitly imposed; otherwise flag.

- Cross-border:

- If Q49 = YES_LEAD_AUTHORITY_CASE or Q62 indicates cross-border, ensure country code is EEA/EU or NON_EU_ENTITY_INVOLVED is stated; else flag.

E. Normalization, mapping, and enrichment

11) Country and region:

- Map Q1 to ISO alpha-2 and attach EEA/EU membership, region (Eurostat), and language(s) for later stratification. Allow EU (EDPS/EDPB) and EEA non-EU (IS, LI, NO) as given.

12) DPA names:

- Normalize Q2 against a controlled DPA registry (canonical English name and native name), attach DPA_id. Retain raw if no match; log fuzzy match score.

13) Sector mapping:

- Q12 ISIC Rev.4 code: validate format, attach 2-digit section and high-level sector. If multiple codes appear, keep the first as primary and list others in a secondary field; flag multi-sector.

14) Financial normalization:

- Q37 fine and Q38 turnover already in EUR. For modeling, also compute:

- fine_positive = I(Q37 > 0).

- fine_log1p = ln(1 + Q37).

- turnover_log1p if Q38 not null.

- fine_to_turnover_ratio if both available.

- Q14 turnover_range: convert to numeric approximation (use midpoints) in a separate derived field turnover_range_midpoint_eur, with a range_imputed flag.

15) Time features:

- From Q3 date: derive year, quarter, month_index; also create decision_age if needed for time-series analysis.

- If Q40 violation duration is categorical, map to an ordinal scale (e.g., DAYS=1, WEEKS=2, MONTHS=3, YEARS=4, ONGOING=5) but keep the original category.

16) Articles and factors expansion:

- Expand Q30–Q35, Q41–Q42, Q46–Q47, Q50, Q53–Q56, Q58–Q59, Q61, Q64 into long-form association tables: decision_id, code, role (e.g., discussed/violated; aggravating/mitigating), to support graph/network and co-occurrence analysis.

F. Granular structuring for tidy and relational use

17) Create core tables:

- decisions (one row per decision): single-valued fields, numeric features, primary outcomes (fine amount, presence of 58(2) measures), time and geography.

- defendants (one row per decision for the “primary defendant” captured; include Q6 “multiple defendants” as a flag; allow later extension to many-to-many).

- multi_select tables:

- article_5_discussed, article_5_violated, article_6_discussed, legal_basis_relied_on, consent_issues, li_test_outcome (single), breach_types, vulnerable_groups, corrective_powers, corrective_scopes, rights_discussed, rights_violated, access_issues, adm_issues, dpo_issues, transfer_violations, aggravating_factors, mitigating_factors, other_measures.

- text_summaries: Q36, Q52, Q67, Q68, with fields raw_text, normalized_text, token_count, language.

- missingness_status: wide table mapping each question to a status code for quick filtering.

18) Longitudinal and categorical roll-ups:

- For EDA convenience, add counts:

- n_principles_discussed, n_principles_violated, n_rights_discussed, n_rights_violated, n_corrective_measures, n_aggravating, n_mitigating.

- Severity proxies:

- severity_measures_present = I(any of REPRIMAND, WARNING, LIMITATION_PROHIBITION_OF_PROCESSING, PROCESSING_BAN, DATA_DELETION_ORDER, ADMINISTRATIVE_FINE).

- remedy_only_case = I(no fine and at least one non-fine corrective measure).

- breach_case = I(Q16 == YES).

G. EDA readiness (basic)

19) Profiles and distributions:

- Frequency tables by DPA, country, year/quarter.

- Fine distributions: histograms, log-scale, stratified by sector, organizational size (Q10), cross-border (Q49/Q62).

- Co-occurrence matrices: Article 5/6 discussed vs violated, rights discussed vs violated, 83(2) aggravating vs mitigating patterns.

- Heatmaps of measures (Q53) by case type (breach_case vs rights_case).

20) Missingness and interpretive caution:

- Visualize missingness by status type (NOT_DISCUSSED vs NOT_APPLICABLE vs UNCLEAR vs NOT_MENTIONED vs NOT_DETERMINED) to avoid the pitfall “absence as exculpation.”

- Always stratify EDA by discussion status when comparing violation rates.

H. Advanced analysis readiness

21) Predictive modeling features:

- Target examples: fine_positive, fine_log1p; presence of ADMINISTRATIVE_FINE; presence of LIMITATION/PROHIBITION; severity composite.

- Predictors: sector, role (controller/processor), organization size, cross-border status, rights involved, principles violated, 83(2) factors (aggravating/mitigating), breach presence and characteristics, DPO non-compliance, international transfer issues, previous infringements.

- Avoid leakage: ensure that targets do not directly appear as predictors (e.g., do not use ADMINISTRATIVE_FINE to predict fine_positive).

22) Legal-sound feature engineering:

- Discussion/violation dual features: include both discussed_X and violated_X with typed-missingness dummies so models don’t treat “not discussed” as “no.”

- Harm encoding: Q43 with types; keep non-material vs material distinct.

- Economic benefit: Q44; maintain quantified vs unquantified difference.

23) Networks and knowledge graphs:

- Entity graph: DPA —[decided]→ decision —[involves]→ defendant; decision —[discusses]→ principle/right/article/factor; decision —[exercises]→ corrective_power.

- Cross-border graph: decisions connected by lead/concerned authority roles.

24) NLP embeddings for research retrieval:

- Compute embeddings for Q36, Q52, Q68, and Q67 references; store vectors separately.

- Preserve raw and normalized texts to link results back to original language and phrasing.

I. Conflict handling, quality flags, and documentation

25) Conflict matrix:

- For each decision, maintain a validation summary: list of contradictions (e.g., fine=0 with ADMINISTRATIVE_FINE), unverifiable claims (caps with unknown turnover), exclusivity violations (NOT_APPLICABLE with substantive options), date anomalies.

- Do not drop records due to conflicts; use flags to filter or weight in analysis.

26) Versioning and provenance:

- Keep data_dictionary.json describing each variable, possible states, and transformations.

- Record parser_version, enum_whitelist_version, and mapping tables versions (ISO/ISIC).

J. Specific transformations/derivations by section (selected highlights)

- Core Metadata (Q1–Q5): map country to EEA/EU; extract appeal flags and outcomes; build case_class = appeal/non-appeal.

- Defendant profile (Q6–Q14): normalize size classes; set public_sector flag; sector_section from ISIC; build size_sector composite; turnover_range_midpoint_eur (imputed).

- Breach (Q15–Q29): breach_case flag; chain-compliance flags (Art. 33 compliance, timeliness, delay bucket, Art. 34 requirement); special data flags (Art.9/10); mitigating_actions count.

- Articles 5/6 (Q30–Q36): discussed vs violated matrices; legal_basis_relied_on vs legal_basis_discussed; consent_issues and LI balancing outcome; textual summary Q36.

- Fines (Q37–Q52): fine metrics, cap claims, duration ordinal, 83(2) aggravating/mitigating matrices; harm and benefit distinctions; cooperation level ordinal; vulnerable groups; other corrective measures; financial situation treatment; summary text Q52.

- Enforcement (Q53–Q55): 58(2) powers matrix; scope and deadlines normalized to ordinal (IMMEDIATE=0, WITHIN_MONTH=1, etc.).

- Rights (Q56–Q59): rights discussed/violated matrices; access and ADM issues expansions.

- DPO (Q60–Q61): requirement and compliance; issues matrix; DPO_noncompliant flag.

- International (Q62–Q64): jurisdictional complexity normalization; transfers discussed/violations; transfer_issues matrix (e.g., Schrems II, safeguards).

- Precedent (Q65–Q68): precedent level ordinal; references count/class; EDPB references parsed (extract guideline numbers); expert summary stored.

K. Examples of careful legal handling

- Do not treat Q30 = NONE_DISCUSSED as absence of principle relevance; it is absence of discussion. Only Q31 = NONE_VIOLATED represents an explicit non-violation finding.

- If Q56 includes rights discussed but Q57 = NONE_VIOLATED, that is a negative finding on those rights; still, presence of discussion informs salience in the case type.

- If Q37 = 0 with significant corrective measures (Q53 non-fine powers), classify as remedy_only_case; do not imply leniency due to lack of mention of certain articles.

L. Practical parsing notes for the sample format

- Strip leading/trailing quotes around entire blocks; handle escaped sequences (e.g., “Guidelines 5/2019”).

- Split by ‘\n’ and then capture Answer lines; ignore empty lines.

- The sample shows two blocks concatenated; ensure segmentation by new “Answer 1:” and reset counters.

- Tolerate occasional trailing commas, mixed “NOT_APPLICABLE” plus substantive tokens; these become contradictions with flags.

M. Outputs to produce immediately

- Cleaned wide table (one row per decision) with:

- All single-value fields normalized and typed.

- Numeric and time derivatives.

- Status columns for typed missingness.

- Quality flags.

- Normalized long tables for multi-selects and factors.

- Dictionaries for ISO, ISIC, DPA canonical names.

- A validation report summarizing contradictions and completeness.

Concise overview table

- Step: Ingest and segment records

- What we do: Split text at each “Answer 1:” block; trim quotes; capture Answer N/value pairs.

- Why it matters: Ensures one decision per record; avoids mixing decisions.

- Step: Parse and type-cast fields

- What we do: Apply schema types; cast numbers/dates; normalize ENUM/MULTI_SELECT tokens.

- Why it matters: Enables numeric EDA and correct categorical analysis.

- Step: Preserve typed missingness

- What we do: Store status codes (NOT_DISCUSS/NOT_APPL/UNCLEAR/etc.) alongside values.

- Why it matters: Avoids conflating “not discussed” with “no violation.”

- Step: Expand multi-selects

- What we do: Create long tables and wide binary matrices per option.

- Why it matters: Supports granular co-occurrence, network analysis, and modeling.

- Step: Legal consistency checks

- What we do: Cross-validate chains (breach→Art.33→Art.34), appeal logic, fines vs caps, DPO logic.

- Why it matters: Flags contradictions without overwriting legal meaning.

- Step: Geographic and sector enrichment

- What we do: Map ISO country→EEA/EU/region; ISIC code→section/sector.

- Why it matters: Enables stratified EDA by geography and sector.

- Step: Financial normalization

- What we do: Create fine_positive, fine_log1p, turnover_log1p, fine_to_turnover; midpoint for ranges.

- Why it matters: Stabilizes distributions, supports comparative analysis.

- Step: Time features

- What we do: Extract year/quarter; ordinalize duration.

- Why it matters: Trend analysis, seasonality, and duration effects.

- Step: Derived legal features

- What we do: breach_case, remedy_only_case, severity proxies; counts of discussed/violated/factors.

- Why it matters: Immediate EDA filters and modeling inputs.

- Step: NLP-ready text fields

- What we do: Normalize and store Q36/Q52/Q67/Q68; compute embeddings (optional).

- Why it matters: Semantic retrieval and qualitative analysis.

- Step: Data structures

- What we do: decisions table + relational tables (factors, rights, powers, issues).

- Why it matters: Supports both tidy EDA and advanced, graph-based analysis.

- Step: Validation and documentation

- What we do: Emit validation report; maintain data dictionary and versioning.

- Why it matters: Reproducibility and academic rigor; transparent handling of anomalies.