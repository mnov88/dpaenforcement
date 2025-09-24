#!/usr/bin/env python3
import csv
import json
import os
import re
import argparse
from glob import glob
from typing import Dict, List, Tuple, Optional

RESPONSES_DIR = "/Users/milos/Desktop/dpaenforcement/raw_data/responses"
OUTPUT_CSV = os.path.join(RESPONSES_DIR, "parsed_responses.csv")
MIN_OUTPUT_CSV = os.path.join(RESPONSES_DIR, "parsed_responses_min.csv")

# Define column names for the 68 answers (concise, analysis-friendly)
ANSWER_COLUMNS = [
    (1, "country_code"),
    (2, "dpa_name"),
    (3, "issue_date"),
    (4, "is_appeal"),
    (5, "appeal_outcome"),
    (6, "multiple_defendants"),
    (7, "primary_defendant_name"),
    (8, "defendant_status"),
    (9, "defendant_role"),
    (10, "org_classifications"),
    (11, "public_sector_level"),
    (12, "isic_sector"),
    (13, "turnover_mentioned"),
    (14, "turnover_range"),
    (15, "initiation_method"),
    (16, "breach_discussed"),
    (17, "art33_required"),
    (18, "breach_notified"),
    (19, "notified_within_72h"),
    (20, "delay_length"),
    (21, "breach_type"),
    (22, "breach_cause"),
    (23, "harm_materialized"),
    (24, "affected_subjects"),
    (25, "special_or_criminal_data"),
    (26, "subjects_notified"),
    (27, "art34_required"),
    (28, "mitigating_actions"),
    (29, "notification_failures_effect"),
    (30, "art5_principles_discussed"),
    (31, "art5_principles_violated"),
    (32, "art6_bases_discussed"),
    (33, "legal_bases_relied"),
    (34, "consent_issues"),
    (35, "legitimate_interest_outcome"),
    (36, "art56_summary"),
    (37, "fine_amount_eur"),
    (38, "annual_turnover_eur"),
    (39, "hit_caps"),
    (40, "violation_duration"),
    (41, "aggravating_factors"),
    (42, "mitigating_factors"),
    (43, "harm_documented"),
    (44, "economic_benefit"),
    (45, "cooperation_level"),
    (46, "vulnerable_subjects"),
    (47, "remedial_actions"),
    (48, "first_time_violation"),
    (49, "cross_border"),
    (50, "other_measures"),
    (51, "financial_consideration"),
    (52, "fine_calc_summary"),
    (53, "corrective_powers"),
    (54, "processing_limitation_scope"),
    (55, "compliance_deadline"),
    (56, "ds_rights_discussed"),
    (57, "ds_rights_violated"),
    (58, "access_issues"),
    (59, "adm_issues"),
    (60, "dpo_appointment"),
    (61, "dpo_issues"),
    (62, "jurisdiction_complexity"),
    (63, "data_transfers_discussed"),
    (64, "transfer_violations_issues"),
    (65, "precedent_significance"),
    (66, "references_other_cases"),
    (67, "edpb_references"),
    (68, "case_summary"),
]

META_COLUMNS = [
    "ID",
    "English_Translation",
    "error",
    "success",
    "model_used",
    "markdown_file",
    "input_tokens",
    "output_tokens",
    "total_tokens",
]

ALL_COLUMNS: List[str] = META_COLUMNS + [name for _, name in ANSWER_COLUMNS]
MIN_COLUMNS: List[str] = ["ID"] + [name for _, name in ANSWER_COLUMNS]

ANSWER_LINE_RE = re.compile(r"^Answer\s+(\d+)\s*:\s*(.*)$")
PREFIXES = (
    "TYPE:",
    "ENUM:",
    "MULTI_SELECT:",
    "FORMAT:",
)


def parse_answers(response_text: Optional[str]) -> Dict[int, str]:
    answers: Dict[int, str] = {}
    if not response_text:
        return answers
    lines = response_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        m = ANSWER_LINE_RE.match(line)
        if not m:
            continue
        num_str, value = m.group(1), m.group(2)
        try:
            num = int(num_str)
        except ValueError:
            continue
        cleaned = " ".join(value.strip().split())
        answers[num] = cleaned
    return answers


def strip_known_prefixes(value: str) -> str:
    v = value.strip()
    changed = True
    while changed and v:
        changed = False
        for p in PREFIXES:
            if v.startswith(p):
                v = v[len(p):].lstrip()
                changed = True
    return v


def normalize_value(answer_num: int, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = strip_known_prefixes(value)
    if answer_num == 1:
        m = re.search(r"ISO_3166-1_ALPHA-2\s*:\s*(.*)$", v)
        if m:
            v = m.group(1).strip()
    if v.lower() in {"type:null", "null"}:
        return None
    return v if v != "" else None


def coerce_number(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    if v.lower() == "null":
        return None
    v_clean = re.sub(r"[€,\$\s]", "", v)
    if re.fullmatch(r"-?\d+(\.\d+)?", v_clean):
        return v_clean
    m = re.search(r"(-?\d+(?:\.\d+)?)", v)
    if m:
        return m.group(1)
    return None


def natural_id_key(id_value: str) -> Tuple[str, int]:
    if not id_value:
        return ("", 0)
    m = re.match(r"^(.*?)(\d+)$", id_value)
    if m:
        prefix, num = m.group(1), int(m.group(2))
        return (prefix, num)
    return (id_value, 0)


def read_json_file(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse AI responses JSON into CSVs")
    p.add_argument("--only-success", action="store_true", help="Include only records where success is true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    json_paths = sorted(glob(os.path.join(RESPONSES_DIR, "results_*.json")))
    records: List[Dict[str, Optional[str]]] = []

    for jp in json_paths:
        try:
            data = read_json_file(jp)
        except Exception as e:
            print(f"WARN: Failed to read {jp}: {e}")
            continue
        if not isinstance(data, list):
            print(f"WARN: {jp} does not contain a JSON array; skipping")
            continue
        for obj in data:
            if not isinstance(obj, dict):
                continue
            if args.only_success:
                succ = obj.get("success")
                # Accept true/True/"true"/"True"
                if not (succ is True or str(succ).lower() == "true"):
                    continue
            rec: Dict[str, Optional[str]] = {}
            for k in META_COLUMNS:
                v = obj.get(k)
                rec[k] = None if v is None else str(v)
            answers = parse_answers(obj.get("response"))
            for num, col_name in ANSWER_COLUMNS:
                raw = answers.get(num)
                norm = normalize_value(num, raw)
                if num in (37, 38):
                    rec[col_name] = coerce_number(norm) if norm is not None else None
                else:
                    rec[col_name] = norm
            records.append(rec)

    records.sort(key=lambda r: natural_id_key(r.get("ID") or ""))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    with open(MIN_OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MIN_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in MIN_COLUMNS})

    print(f"Wrote {len(records)} rows to {OUTPUT_CSV}")
    print(f"Wrote {len(records)} rows to {MIN_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
