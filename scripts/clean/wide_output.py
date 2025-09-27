from __future__ import annotations

import csv
import json
from math import log1p
from pathlib import Path
from typing import Dict, Any, List, Tuple

from scripts.clean.typing_status import (
    parse_date_field,
    normalize_country,
    parse_number,
    split_multiselect,
    derive_multiselect_status,
    detect_exclusivity_conflict,
    NumericParseResult,
)
from scripts.clean.isic_map import IsicIndex
from scripts.clean.geo_enrich import enrich_country, normalize_dpa_name
from scripts.clean.text_norm import normalize_text, detect_language_heuristic
from scripts.clean.enum_validate import EnumWhitelist
from scripts.parser.ingest import parse_record


SEVERITY_MEASURES = {
    "WARNING",
    "REPRIMAND",
    "LIMITATION_PROHIBITION_OF_PROCESSING",
    "PROCESSING_BAN",
    "DATA_DELETION_ORDER",
    "ADMINISTRATIVE_FINE",
}

COUNTRY_WHITELIST = {
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR","HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE","IS","LI","NO","EU","UNCLEAR"
}

SCHEMA_ECHO_PREFIXES = ("TYPE:", "ENUM:", "MULTI_SELECT:")

FINE_OUTLIER_HIGH = 1e10
TURNOVER_OUTLIER_HIGH = 1e12

# Multi-select fields to emit systematically: (Qkey, prefix)
MULTI_FIELDS: List[Tuple[str, str]] = [
    ("Q30", "q30_discussed"),
    ("Q31", "q31_violated"),
    ("Q32", "q32_bases"),
    ("Q41", "q41_aggrav"),
    ("Q42", "q42_mitig"),
    ("Q46", "q46_vuln"),
    ("Q47", "q47_remedial"),
    ("Q50", "q50_other_measures"),
    ("Q53", "q53_powers"),
    ("Q54", "q54_scopes"),
    ("Q56", "q56_rights_discussed"),
    ("Q57", "q57_rights_violated"),
    ("Q58", "q58_access_issues"),
    ("Q59", "q59_adm_issues"),
    ("Q61", "q61_dpo_issues"),
    ("Q64", "q64_transfer_violations"),
]

RAW_QUESTION_EXPORT: Tuple[str, ...] = (
    "Q1",
    "Q4",
    "Q5",
    "Q16",
    "Q17",
    "Q18",
    "Q21",
    "Q26",
    "Q27",
    "Q30",
    "Q31",
    "Q32",
    "Q33",
    "Q34",
    "Q35",
    "Q37",
    "Q38",
    "Q39",
    "Q41",
    "Q42",
    "Q46",
    "Q49",
    "Q53",
    "Q54",
    "Q56",
    "Q57",
    "Q58",
    "Q59",
    "Q60",
    "Q61",
    "Q62",
    "Q64",
)

def _is_schema_echo(value: str) -> bool:
    v = (value or "").strip()
    return any(v.startswith(p) for p in SCHEMA_ECHO_PREFIXES)


def _clean_country_code(code: str) -> Tuple[str, bool, bool]:
    c = (code or "").upper()
    mapped = False
    if c == "UK":
        c = "GB"
        mapped = True
    whitelist_ok = c in COUNTRY_WHITELIST
    return c, mapped, whitelist_ok
def clean_csv_to_wide(input_csv: Path, out_csv: Path, validation_report: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    validation_report.parent.mkdir(parents=True, exist_ok=True)

    isic_path = Path("resources/ISIC_Rev_4_english_structure.txt")
    isic_idx = IsicIndex.load_from_file(isic_path) if isic_path.exists() else None
    wl_path = Path("resources/enum_whitelist.json")
    whitelist = EnumWhitelist.load(wl_path) if wl_path.exists() else EnumWhitelist({})

    with input_csv.open(newline="", encoding="utf-8") as f_in, \
         out_csv.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)

        # Base fieldnames
        fieldnames: List[str] = [
            "decision_id",
            "ingest_parser_version", "ingest_line_count", "ingest_question_count", "ingest_missing_questions", "ingest_warning_count", "ingest_warnings",
            "country_code", "country_status", "country_group", "country_mapped_from_uk", "country_whitelist_ok",
            "dpa_name_raw", "dpa_name_canonical",
            "decision_date_raw", "decision_date", "decision_date_status", "decision_date_error", "decision_year", "decision_quarter",
            # Numerics with typed status/raw
            "fine_eur", "fine_numeric_valid", "fine_positive", "fine_log1p", "fine_outlier_flag", "fine_raw", "fine_status", "fine_error",
            "turnover_eur", "turnover_numeric_valid", "turnover_positive", "turnover_log1p", "turnover_outlier_flag", "turnover_raw", "turnover_status", "turnover_error",
            "fine_to_turnover_ratio",
            # ISIC
            "isic_code", "isic_desc", "isic_section",
            # Derived counts and flags
            "n_principles_discussed", "n_principles_violated", "n_corrective_measures",
            "severity_measures_present", "remedy_only_case", "breach_case",
            # Text fields (normalized)
            "q36_text", "q36_lang", "q36_tokens",
            "q52_text", "q52_lang", "q52_tokens",
            "q67_text", "q67_lang", "q67_tokens",
            "q68_text", "q68_lang", "q68_tokens",
            # Cleaning flags
            "schema_echo_flag", "schema_echo_fields",
        ]
        for qkey in RAW_QUESTION_EXPORT:
            fieldnames.append(f"raw_{qkey.lower()}")
        # Track allowed tokens per multi-select question for downstream expansion
        multi_allowed_map: Dict[str, List[str]] = {}
        for qkey, prefix in MULTI_FIELDS:
            allowed_tokens = whitelist.allowed_tokens(qkey[1:] if qkey.startswith("Q") else qkey)
            multi_allowed_map[qkey] = allowed_tokens

            # Add systematic multi-select columns (coverage/status metadata + exclusivity)
            fieldnames.extend(
                [
                    f"{prefix}_coverage_status",
                    f"{prefix}_known",
                    f"{prefix}_unknown",
                    f"{prefix}_status",
                    f"{prefix}_exclusivity_conflict",
                ]
            )

            # Add per-option boolean indicators (1/0/blank for not mentioned)
            for token in allowed_tokens:
                fieldnames.append(f"{prefix}_{token}")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        validations: list[Dict[str, Any]] = []

        for row in reader:
            decision_id = row.get("ID")
            resp = (row.get("response", "") or "")
            parsed = parse_record(resp)
            answers = parsed["answers"]
            metadata = parsed["metadata"]

            schema_echo_flag = 0
            schema_echo_fields: List[str] = []

            country_code_raw, country_status = normalize_country(answers.get("Q1", ""))
            country_code_mapped, mapped_from_uk, whitelist_ok = _clean_country_code(country_code_raw)
            country_code_final, country_group = enrich_country(country_code_mapped)

            dpa_name_raw = (answers.get("Q2", "") or "").strip()
            dpa_name_clean = dpa_name_raw if not _is_schema_echo(dpa_name_raw) else ""
            if dpa_name_clean == "" and dpa_name_raw:
                schema_echo_flag = 1
                schema_echo_fields.append("Q2")
            dpa_name_canonical = normalize_dpa_name(dpa_name_clean)

            dpr = parse_date_field(answers.get("Q3", ""))
            decision_date = dpr.value.isoformat() if dpr.value else ""
            decision_year = dpr.value.year if dpr.value else ""
            decision_quarter = (f"Q{((dpr.value.month-1)//3)+1}" if dpr.value else "")

            # Numerics: keep raw strings and status
            fine_parsed: NumericParseResult = parse_number(answers.get("Q37", ""))
            turnover_parsed: NumericParseResult = parse_number(answers.get("Q38", ""))
            fine_eur = fine_parsed.value
            turnover_eur = turnover_parsed.value
            fine_log = f"{log1p(fine_eur):.6f}" if fine_eur is not None else ""
            turnover_log = f"{log1p(turnover_eur):.6f}" if turnover_eur is not None else ""
            fine_outlier = 1 if (fine_eur or 0) > FINE_OUTLIER_HIGH else 0
            turnover_outlier = 1 if (turnover_eur or 0) > TURNOVER_OUTLIER_HIGH else 0
            fine_to_turnover_ratio = ""
            if (fine_eur is not None) and (turnover_eur is not None) and turnover_eur > 0:
                fine_to_turnover_ratio = f"{fine_eur/turnover_eur:.8f}"

            # Derived counts placeholders
            n_principles_discussed = 0
            n_principles_violated = 0
            n_corrective_measures = 0

            # ISIC
            isic_raw = (answers.get("Q12", "") or "").strip()
            if _is_schema_echo(isic_raw):
                schema_echo_flag = 1
                schema_echo_fields.append("Q12")
                isic_raw = ""
            isic_code = isic_raw
            isic_desc = ""
            isic_section = ""
            if isic_idx is not None and isic_raw:
                entry, ok = isic_idx.lookup(isic_raw)
                if ok and entry is not None:
                    isic_code = entry.code
                    isic_desc = entry.description
                    isic_section = entry.section or ""

            # Text normalization
            q36_norm, q36_tok, _ = normalize_text(answers.get("Q36", "") or "")
            q52_norm, q52_tok, _ = normalize_text(answers.get("Q52", "") or "")
            q67_norm, q67_tok, _ = normalize_text(answers.get("Q67", "") or "")
            q68_norm, q68_tok, _ = normalize_text(answers.get("Q68", "") or "")
            q36_lang = detect_language_heuristic(q36_norm)
            q52_lang = detect_language_heuristic(q52_norm)
            q67_lang = detect_language_heuristic(q67_norm)
            q68_lang = detect_language_heuristic(q68_norm)

            base_row = {
                "decision_id": decision_id,
                "ingest_parser_version": metadata.get("parser_version", ""),
                "ingest_line_count": metadata.get("line_count", 0),
                "ingest_question_count": metadata.get("question_count", 0),
                "ingest_missing_questions": ";".join(metadata.get("missing_questions", [])),
                "ingest_warning_count": len(metadata.get("warnings", [])),
                "ingest_warnings": ";".join(metadata.get("warnings", [])),
                "country_code": country_code_final or "",
                "country_status": country_status,
                "country_group": country_group,
                "country_mapped_from_uk": 1 if mapped_from_uk else 0,
                "country_whitelist_ok": 1 if whitelist_ok else 0,
                "dpa_name_raw": dpa_name_raw,
                "dpa_name_canonical": dpa_name_canonical,
                "decision_date_raw": dpr.raw,
                "decision_date": decision_date,
                "decision_date_status": dpr.status,
                "decision_date_error": dpr.error or "",
                "decision_year": decision_year,
                "decision_quarter": decision_quarter,
                "fine_eur": f"{fine_eur:.6f}" if fine_eur is not None else "",
                "fine_numeric_valid": 1 if fine_parsed.valid else 0,
                "fine_positive": 1 if (fine_eur or 0) > 0 else 0,
                "fine_log1p": fine_log,
                "fine_outlier_flag": fine_outlier,
                "fine_raw": fine_parsed.raw,
                "fine_status": fine_parsed.status,
                "fine_error": fine_parsed.error or "",
                "turnover_eur": f"{turnover_eur:.6f}" if turnover_eur is not None else "",
                "turnover_numeric_valid": 1 if turnover_parsed.valid else 0,
                "turnover_positive": 1 if (turnover_eur or 0) > 0 else 0,
                "turnover_log1p": turnover_log,
                "turnover_outlier_flag": turnover_outlier,
                "turnover_raw": turnover_parsed.raw,
                "turnover_status": turnover_parsed.status,
                "turnover_error": turnover_parsed.error or "",
                "fine_to_turnover_ratio": fine_to_turnover_ratio,
                "isic_code": isic_code,
                "isic_desc": isic_desc,
                "isic_section": isic_section,
                "n_principles_discussed": n_principles_discussed,
                "n_principles_violated": n_principles_violated,
                "n_corrective_measures": n_corrective_measures,
                "severity_measures_present": 0,
                "remedy_only_case": 0,
                "breach_case": 1 if answers.get("Q16") == "YES" else 0,
                "q36_text": q36_norm, "q36_lang": q36_lang, "q36_tokens": q36_tok,
                "q52_text": q52_norm, "q52_lang": q52_lang, "q52_tokens": q52_tok,
                "q67_text": q67_norm, "q67_lang": q67_lang, "q67_tokens": q67_tok,
                "q68_text": q68_norm, "q68_lang": q68_lang, "q68_tokens": q68_tok,
                "schema_echo_flag": schema_echo_flag,
                "schema_echo_fields": ";".join(sorted(set(schema_echo_fields))),
            }
            for qkey in RAW_QUESTION_EXPORT:
                base_row[f"raw_{qkey.lower()}"] = (answers.get(qkey, "") or "").strip()

            # Populate systematic multi-selects
            for qkey, prefix in MULTI_FIELDS:
                ms_parsed = split_multiselect(answers.get(qkey, ""))
                tokens = [t for t in ms_parsed.tokens if not _is_schema_echo(t)]
                unknown, known = whitelist.validate_tokens(qkey, tokens)
                status = derive_multiselect_status(qkey, tokens)
                exclusivity = detect_exclusivity_conflict(tokens)
                coverage_status = ms_parsed.status
                allowed_tokens = multi_allowed_map.get(qkey, [])
                known_set = set(known)

                base_row[f"{prefix}_coverage_status"] = coverage_status
                base_row[f"{prefix}_known"] = ",".join(known)
                base_row[f"{prefix}_unknown"] = ",".join(unknown)
                base_row[f"{prefix}_status"] = status
                base_row[f"{prefix}_exclusivity_conflict"] = exclusivity

                for token in allowed_tokens:
                    col_name = f"{prefix}_{token}"
                    if coverage_status == "NOT_MENTIONED":
                        base_row[col_name] = ""
                    else:
                        base_row[col_name] = 1 if token in known_set else 0

                # Aggregate counts/flags
                if qkey == "Q30":
                    base_row["n_principles_discussed"] = len([t for t in known if t])
                if qkey == "Q31":
                    base_row["n_principles_violated"] = len([t for t in known if t and not t.startswith("NONE_") and t != "NOT_DETERMINED"]) if known else 0
                if qkey == "Q53":
                    base_row["n_corrective_measures"] = len([t for t in known if t and t != "NONE"]) if known else 0
                    base_row["severity_measures_present"] = 1 if any(t in SEVERITY_MEASURES for t in known) else 0
                    base_row["remedy_only_case"] = 1 if (fine_eur or 0) == 0 and any(t for t in known if t and t != "NONE" and t != "ADMINISTRATIVE_FINE") else 0

            writer.writerow(base_row)

            flags = []
            q53_raw = answers.get("Q53", "") or ""
            if "ADMINISTRATIVE_FINE" in q53_raw and not (fine_eur or 0) > 0:
                flags.append("admin_fine_but_zero_amount")
            if fine_parsed.status == "PARSE_ERROR":
                flags.append("fine_parse_error")
            if turnover_parsed.status == "PARSE_ERROR":
                flags.append("turnover_parse_error")
            validations.append({"decision_id": decision_id, "flags": flags})

    validation_report.write_text(json.dumps(validations, ensure_ascii=False, indent=2), encoding="utf-8")
