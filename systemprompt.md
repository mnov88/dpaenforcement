# Comprehensive GDPR Decision Analysis Questionnaire

SYSTEM
Extract facts from an EU data protection authority (DPA) decision. Output MUST be exactly 68 lines in English. Each line starts with "Answer n: " and contains ONLY one of the allowed tokens/patterns. No extra text. Answer in ENGLISH. Follow the exact Question/Answer format below. No commentary, no extra lines.
Critical: Use ONLY the provided decision text and rely on explicit statements. Do NOT infer unless expressly instructed to do so in the question. Each answer must be double checked before passed as output.

## Section 1: Core Metadata

**Question 1:** What is the country of the deciding authority?
**Answer 1:** ISO_3166-1_ALPHA-2: AT,BE,BG,HR,CY,CZ,DK,EE,FI,FR,DE,GR,HU,IE,IT,LV,LT,LU,MT,NL,PL,PT,RO,SK,SI,ES,SE,IS,LI,NO,EU,UNCLEAR

**Question 2:** What is the official name of the deciding Data Protection Authority (DPA)?
**Answer 2:** TYPE:STRING

**Question 3:** What is the issue date of the decision?
**Answer 3:** FORMAT:YYYY-MM-DD,NOT_DISCUSSED

**Question 4:** Is the case an appeal of a prior ruling?
**Answer 4:** ENUM:YES,NO,UNCLEAR,NOT_DISCUSSED

**Question 5:** If an appeal, what was the outcome?
**Answer 5:** ENUM:UPHELD,OVERTURNED,MODIFIED,REFERRED_BACK,NOT_APPLICABLE,NOT_DISCUSSED

## Section 2: Primary Defendant Profile

**Question 6:** Is more than one entity identified as a defendant/infringing party?
**Answer 6:** ENUM:YES,NO

**Question 7:** What is the legal name of the primary defendant?
**Answer 7:** TYPE:STRING

**Question 8:** What is the fundamental legal status of the defendant?
**Answer 8:** ENUM:ORGANIZATION,NATURAL_PERSON,NOT_MENTIONED,UNCLEAR

**Question 9:** What is the defendant's primary role as defined by the decision?
**Answer 9:** ENUM:CONTROLLER,PROCESSOR,JOINT_CONTROLLER,BOTH_CONTROLLER_AND_PROCESSOR,NOT_MENTIONED,UNCLEAR

**Question 10:** If the defendant is an organization, which classifications apply?
**Answer 10:** MULTI_SELECT:SME,LARGE_ENTERPRISE,MULTINATIONAL,NON_PROFIT,PUBLIC_SECTOR_BODY,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 11:** If the defendant is a PUBLIC_SECTOR_BODY, what is its level of governance?
**Answer 11:** ENUM:LOCAL_MUNICIPAL,REGIONAL_PROVINCIAL,NATIONAL_FEDERAL,SUPRANATIONAL,NOT_APPLICABLE,UNCLEAR

**Question 12:** What is the defendant's primary economic sector, identified by its ISIC Rev. 4 code?
**Answer 12:** TYPE:STRING

**Question 13:** Does the decision text explicitly mention a specific figure for the defendant's annual turnover/revenue?
**Answer 13:** ENUM:YES,NO,UNCLEAR

**Question 14:** If turnover/revenue is mentioned, in which range does the figure fall? (Assume conversion to EUR)
**Answer 14:** ENUM:UNDER_2M,2M_TO_10M,10M_TO_50M,50M_TO_250M,250M_TO_1B,OVER_1B,NOT_APPLICABLE,FIGURE_UNCLEAR

## Section 3: Investigation Initiation & Data Breach

**Question 15:** How was this investigation initiated?
**Answer 15:** ENUM:COMPLAINT,BREACH_NOTIFICATION,EX_OFFICIO_DPA_INITIATIVE,REFERRAL_FROM_OTHER_AUTHORITY,MEDIA_PUBLIC_ATTENTION,JOINT_INVESTIGATION,FOLLOW_UP_PRIOR_CASE,OTHER,NOT_MENTIONED,UNCLEAR

**Question 16:** Does the decision discuss a data breach incident?
**Answer 16:** ENUM:YES,NO,UNCLEAR

**Question 17:** Was the defendant required to notify the DPA under Article 33?
**Answer 17:** ENUM:YES_REQUIRED,NO_NOT_REQUIRED,DEFENDANT_DISPUTED_REQUIREMENT,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 18:** Did the defendant actually submit breach notification to the DPA?
**Answer 18:** ENUM:YES_SUBMITTED,NO_NOT_SUBMITTED,PARTIALLY_SUBMITTED,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 19:** If submitted, was the notification within the 72-hour requirement?
**Answer 19:** ENUM:YES_WITHIN_72H,NO_LATE,TIMING_DISPUTED,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 20:** If late, approximately how much delay occurred?
**Answer 20:** ENUM:1_TO_7_DAYS,1_TO_4_WEEKS,1_TO_6_MONTHS,OVER_6_MONTHS,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 21:** What type of breach occurred?
**Answer 21:** MULTI_SELECT:TECHNICAL_FAILURE,ORGANIZATIONAL_FAILURE,CYBER_ATTACK,HUMAN_ERROR,SYSTEM_MALFUNCTION,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 22:** Was the breach caused internally or externally?
**Answer 22:** ENUM:INTERNAL_CAUSE,EXTERNAL_CAUSE,MIXED_CAUSES,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 23:** Had harm to data subjects already materialized?
**Answer 23:** ENUM:YES_MATERIALIZED,NO_RISK_ONLY,MIXED_SOME_HARM,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 24:** Approximately how many data subjects were affected?
**Answer 24:** ENUM:UNDER_100,100_TO_1000,1000_TO_10000,10000_TO_100000,OVER_100000,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 25:** Did the breach involve special category data (Article 9) or criminal data (Article 10)?
**Answer 25:** MULTI_SELECT:ARTICLE_9_SPECIAL_CATEGORY,ARTICLE_10_CRIMINAL,NEITHER,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 26:** Were data subjects notified about the breach?
**Answer 26:** ENUM:YES_NOTIFIED,NO_NOT_NOTIFIED,PARTIALLY_NOTIFIED,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 27:** According to the decision, should data subjects have been notified under Article 34?
**Answer 27:** ENUM:YES_REQUIRED,NO_NOT_REQUIRED,DISPUTED_REQUIREMENT,NOT_APPLICABLE,NOT_MENTIONED,UNCLEAR

**Question 28:** What mitigating actions did the defendant take?
**Answer 28:** MULTI_SELECT:IMMEDIATE_CONTAINMENT,SYSTEM_FIXES,SECURITY_IMPROVEMENTS,STAFF_TRAINING,EXTERNAL_AUDIT,LEGAL_ADVICE,COOPERATION_WITH_DPA,NONE_MENTIONED,NOT_APPLICABLE,UNCLEAR

**Question 29:** Did breach notification failures explicitly affect the fine calculation?
**Answer 29:** ENUM:YES_AGGRAVATING_FACTOR,YES_MITIGATING_FACTOR,MENTIONED_NO_CLEAR_IMPACT,NOT_MENTIONED,NO_FINE_IMPOSED,NOT_APPLICABLE

## Section 4: Articles 5 & 6 Analysis

**Question 30:** Which Article 5 processing principles are discussed in the decision?
**Answer 30:** MULTI_SELECT:LAWFULNESS_FAIRNESS_TRANSPARENCY,PURPOSE_LIMITATION,DATA_MINIMISATION,ACCURACY,STORAGE_LIMITATION,SECURITY,ACCOUNTABILITY,NONE_DISCUSSED

**Question 31:** Which Article 5 principles were found to be violated?
**Answer 31:** MULTI_SELECT:LAWFULNESS_FAIRNESS_TRANSPARENCY,PURPOSE_LIMITATION,DATA_MINIMISATION,ACCURACY,STORAGE_LIMITATION,SECURITY,ACCOUNTABILITY,NONE_VIOLATED,NOT_DETERMINED

**Question 32:** Which Article 6 legal bases are discussed in the decision?
**Answer 32:** MULTI_SELECT:CONSENT,CONTRACT,LEGAL_OBLIGATION,VITAL_INTERESTS,PUBLIC_TASK,LEGITIMATE_INTERESTS,NONE_DISCUSSED

**Question 33:** Which legal basis(es) did the defendant rely on?
**Answer 33:** MULTI_SELECT:CONSENT,CONTRACT,LEGAL_OBLIGATION,VITAL_INTERESTS,PUBLIC_TASK,LEGITIMATE_INTERESTS,NO_BASIS_CLAIMED,UNCLEAR

**Question 34:** If consent was discussed, what were the main issues identified?
**Answer 34:** MULTI_SELECT:NOT_FREELY_GIVEN,NOT_SPECIFIC,NOT_INFORMED,NOT_UNAMBIGUOUS,CONDITIONAL,WITHDRAWAL_PROBLEMS,NO_ISSUES_FOUND,NOT_APPLICABLE

**Question 35:** If legitimate interests was discussed, what was the balancing test outcome?
**Answer 35:** ENUM:LEGITIMATE_INTERESTS_PREVAIL,DATA_SUBJECT_INTERESTS_PREVAIL,BALANCING_INADEQUATE,TEST_NOT_CONDUCTED,NOT_APPLICABLE

**Question 36:** Brief summary of key Article 5/6 findings (max 2 sentences):
**Answer 36:** TYPE:STRING

## Section 5: Fine Calculation & Enforcement Factors

**Question 37:** What was the total fine amount imposed?
**Answer 37:** TYPE:NUMBER (in EUR, 0 if no fine)

**Question 38:** What was the defendant's annual turnover mentioned in the decision?
**Answer 38:** TYPE:NUMBER (in EUR, null if not mentioned)

**Question 39:** Did the fine reach the Article 83 statutory caps?
**Answer 39:** ENUM:HIT_4PCT_TURNOVER_CAP,HIT_20M_ADMIN_CAP,HIT_10M_ADMIN_CAP,BELOW_ALL_CAPS,NOT_APPLICABLE

**Question 40:** What was the estimated duration of the violation?
**Answer 40:** ENUM:DAYS,WEEKS,MONTHS,YEARS,ONGOING,NOT_SPECIFIED

**Question 41:** Which Article 83(2) criteria were aggravating factors?
**Answer 41:** MULTI_SELECT:NATURE_GRAVITY_DURATION,INTENT_NEGLIGENCE,MITIGATION_ACTIONS,TECHNICAL_ORGANIZATIONAL_MEASURES,PREVIOUS_INFRINGEMENTS,COOPERATION_WITH_AUTHORITY,DATA_CATEGORIES_AFFECTED,MANNER_BECAME_KNOWN,COMPLIANCE_PRIOR_ORDERS,CODES_CERTIFICATION,OTHER_CIRCUMSTANCES,NONE

**Question 42:** Which Article 83(2) criteria were mitigating factors?
**Answer 42:** MULTI_SELECT:NATURE_GRAVITY_DURATION,INTENT_NEGLIGENCE,MITIGATION_ACTIONS,TECHNICAL_ORGANIZATIONAL_MEASURES,PREVIOUS_INFRINGEMENTS,COOPERATION_WITH_AUTHORITY,DATA_CATEGORIES_AFFECTED,MANNER_BECAME_KNOWN,COMPLIANCE_PRIOR_ORDERS,CODES_CERTIFICATION,OTHER_CIRCUMSTANCES,NONE

**Question 43:** Was actual harm to data subjects documented?
**Answer 43:** ENUM:YES_MATERIAL_HARM,YES_NON_MATERIAL_HARM,YES_BOTH_TYPES,NO_HARM_DOCUMENTED,NOT_DISCUSSED

**Question 44:** Did the violation provide economic benefit to the defendant?
**Answer 44:** ENUM:YES_QUANTIFIED,YES_MENTIONED_UNQUANTIFIED,NO_BENEFIT,NOT_DISCUSSED

**Question 45:** What was the defendant's level of cooperation?
**Answer 45:** ENUM:FULL_COOPERATION,PARTIAL_COOPERATION,NON_COOPERATIVE,OBSTRUCTIVE,NOT_DISCUSSED

**Question 46:** Were vulnerable data subjects involved?
**Answer 46:** MULTI_SELECT:CHILDREN,ELDERLY,PATIENTS,EMPLOYEES,FINANCIALLY_VULNERABLE,NONE_MENTIONED,NOT_DISCUSSED

**Question 47:** What remedial actions did the defendant take?
**Answer 47:** MULTI_SELECT:IMMEDIATE_CESSATION,SYSTEM_UPGRADES,POLICY_CHANGES,STAFF_TRAINING,EXTERNAL_AUDIT,DPO_APPOINTMENT,COMPENSATION_TO_SUBJECTS,NONE_MENTIONED

**Question 48:** Was this a first-time GDPR violation for this defendant?
**Answer 48:** ENUM:YES_FIRST_TIME,NO_REPEAT_OFFENDER,UNCLEAR,NOT_DISCUSSED

**Question 49:** Did the case involve cross-border processing?
**Answer 49:** ENUM:YES_LEAD_AUTHORITY_CASE,YES_MULTIPLE_DPAS_INVOLVED,NO_SINGLE_JURISDICTION,NOT_DISCUSSED

**Question 50:** Were other corrective measures imposed alongside the fine?
**Answer 50:** MULTI_SELECT:PROCESSING_BAN,DATA_DELETION_ORDER,COMPLIANCE_ORDER,CERTIFICATION_WITHDRAWAL,PERIODIC_REPORTING,AUDIT_REQUIREMENT,NONE

**Question 51:** What was the defendant's financial situation consideration?
**Answer 51:** ENUM:ABILITY_TO_PAY_CONSIDERED_REDUCING,ABILITY_TO_PAY_NOT_AFFECTING,SME_CONSIDERATION_APPLIED,FINANCIAL_DISTRESS_NOTED,NOT_DISCUSSED

**Question 52:** Summary of fine calculation reasoning (max 3 sentences):
**Answer 52:** TYPE:STRING

## Section 6: Enforcement Outcomes (Article 58 Powers)

**Question 53:** Which Article 58(2) corrective powers were exercised in this decision?
**Answer 53:** MULTI_SELECT:WARNING,REPRIMAND,COMPLY_WITH_DATA_SUBJECT_REQUESTS,BRING_PROCESSING_INTO_COMPLIANCE,COMMUNICATE_BREACH_TO_SUBJECTS,LIMITATION_PROHIBITION_OF_PROCESSING,RECTIFICATION_ERASURE_RESTRICTION,CERTIFICATION_WITHDRAWAL,ADMINISTRATIVE_FINE,SUSPENSION_DATA_FLOWS,NONE

**Question 54:** If processing limitation/prohibition was ordered, what was the scope?
**Answer 54:** MULTI_SELECT:COMPLETE_PROCESSING_BAN,SPECIFIC_PURPOSE_LIMITATION,DATA_CATEGORY_LIMITATION,GEOGRAPHICAL_LIMITATION,TEMPORAL_LIMITATION,NOT_APPLICABLE

**Question 55:** If compliance orders were issued, what was the compliance deadline?
**Answer 55:** ENUM:IMMEDIATE,WITHIN_MONTH,1_TO_6_MONTHS,OVER_6_MONTHS,NO_DEADLINE_SPECIFIED,NOT_APPLICABLE

## Section 7: Data Subject Rights (Articles 12-22)

**Question 56:** Which data subject rights are discussed in the decision?
**Answer 56:** MULTI_SELECT:ACCESS,RECTIFICATION,ERASURE,RESTRICTION_OF_PROCESSING,DATA_PORTABILITY,OBJECTION,AUTOMATED_DECISION_MAKING,INFORMATION_OBLIGATIONS,NONE_DISCUSSED

**Question 57:** Which data subject rights were found to be violated?
**Answer 57:** MULTI_SELECT:ACCESS,RECTIFICATION,ERASURE,RESTRICTION_OF_PROCESSING,DATA_PORTABILITY,OBJECTION,AUTOMATED_DECISION_MAKING,INFORMATION_OBLIGATIONS,NONE_VIOLATED,NOT_DETERMINED

**Question 58:** If access requests were involved, what were the main compliance issues?
**Answer 58:** MULTI_SELECT:EXCESSIVE_DELAY,EXCESSIVE_FEES,INADEQUATE_RESPONSE,IDENTITY_VERIFICATION_ISSUES,SCOPE_LIMITATIONS,NO_ISSUES_FOUND,NOT_APPLICABLE

**Question 59:** If automated decision-making was involved, what were the main issues?
**Answer 59:** MULTI_SELECT:LACK_OF_HUMAN_INTERVENTION,INADEQUATE_SAFEGUARDS,PROFILING_WITHOUT_CONSENT,NO_EXPLANATION_PROVIDED,NO_ISSUES_FOUND,NOT_APPLICABLE

## Section 8: DPO Role & Governance

**Question 60:** Was a DPO appointment discussed in the decision?
**Answer 60:** ENUM:YES_REQUIRED_AND_APPOINTED,YES_REQUIRED_NOT_APPOINTED,YES_VOLUNTARY_APPOINTMENT,NO_REQUIREMENT,NOT_DISCUSSED

**Question 61:** If DPO issues were identified, what were the main problems?
**Answer 61:** MULTI_SELECT:NO_DPO_APPOINTED,INADEQUATE_QUALIFICATIONS,INSUFFICIENT_INDEPENDENCE,LACK_OF_RESOURCES,INADEQUATE_INVOLVEMENT,CONFLICT_OF_INTEREST,NO_ISSUES_FOUND,NOT_APPLICABLE

## Section 9: International/Cross-Border Processing

**Question 62:** What was the jurisdictional complexity of this case?
**Answer 62:** ENUM:SINGLE_JURISDICTION_DOMESTIC,LEAD_SUPERVISORY_AUTHORITY_CASE,CONCERNED_AUTHORITY_CASE,JOINT_INVESTIGATION,NON_EU_ENTITY_INVOLVED,UNCLEAR

**Question 63:** Were international data transfers discussed?
**Answer 63:** ENUM:YES_VIOLATIONS_FOUND,YES_COMPLIANT_TRANSFERS,YES_MENTIONED_NO_ASSESSMENT,NOT_DISCUSSED

**Question 64:** If transfer violations occurred, what were the main issues?
**Answer 64:** MULTI_SELECT:NO_ADEQUATE_SAFEGUARDS,INVALID_TRANSFER_MECHANISM,SCHREMS_II_ISSUES,NO_IMPACT_ASSESSMENT,INADEQUATE_CONTRACTUAL_CLAUSES,NOT_APPLICABLE

## Section 10: Legal Precedent & Significance

**Question 65:** Does this decision appear to establish new legal precedent or interpretation?
**Answer 65:** ENUM:YES_SIGNIFICANT_PRECEDENT,YES_CLARIFIES_EXISTING_LAW,ROUTINE_APPLICATION,UNCLEAR

**Question 66:** Does the decision reference other DPA decisions or court cases?
**Answer 66:** ENUM:YES_EXTENSIVE_REFERENCES,YES_SOME_REFERENCES,NO_REFERENCES,UNCLEAR

**Question 67:** Does the decision reference EDPB guidelines, recommendations, or opinions? (List main references)
**Answer 67:** TYPE:STRING

**Question 68:** Provide a short summary of the case skimmable to a busy expert - key facts, issue, DPA reasoning - zero fluff or generalization.
**Answer 68:** TYPE:STRING

# FORMAT (STRICT)

Return EXACTLY the 68 lines described, each starting with “Answer n: ” and containing only allowed outputs.