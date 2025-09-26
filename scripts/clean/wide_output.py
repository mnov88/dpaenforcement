from __future__ import annotations

import csv
import json
from math import log1p
from pathlib import Path
from typing import Dict, Any, List, Tuple

from scripts.clean.typing_status import parse_date_field, normalize_country, parse_number, split_multiselect
from scripts.clean.isic_map import IsicIndex
from scripts.clean.geo_enrich import enrich_country, normalize_dpa_name
from scripts.clean.text_norm import normalize_text, detect_language_heuristic


SEVERITY_MEASURES = {
    "WARNING",
    "REPRIMAND",
    "LIMITATION_PROHIBITION_OF_PROCESSING",
    "PROCESSING_BAN",
    "DATA_DELETION_ORDER",
    "ADMINISTRATIVE_FINE",
}

# Allowed country codes from the prompt (Q1)
COUNTRY_WHITELIST = {
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR","HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE","IS","LI","NO","EU","UNCLEAR"
}

SCHEMA_ECHO_PREFIXES = ("TYPE:", "ENUM:", "MULTI_SELECT:")

# Outlier thresholds (EUR)
FINE_OUTLIER_HIGH = 1e10
TURNOVER_OUTLIER_HIGH = 1e12


def _answers_from_response(resp: str) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    for ln in resp.splitlines():
        if ln.startswith("Answer "):
            try:
                idx = ln.index(":")
                left = ln[len("Answer "):idx].strip()
                val = ln[idx + 1 :].strip()
                answers[f"Q{left}"] = val
            except Exception:
                pass
    return answers


def _is_schema_echo(value: str) -> bool:
    v = (value or "").strip()
    return any(v.startswith(p) for p in SCHEMA_ECHO_PREFIXES)


def _clean_country_code(code: str) -> Tuple[str, bool, bool]:
    # Returns (mapped_code, mapped_from_uk_flag, whitelist_ok)
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

    # Load ISIC index lazily
    isic_path = Path("resources/ISIC_Rev_4_english_structure.txt")
    isic_idx = IsicIndex.load_from_file(isic_path) if isic_path.exists() else None

    with input_csv.open(newline="", encoding="utf-8") as f_in, \
         out_csv.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)

        fieldnames: List[str] = [
            "decision_id",
            "country_code", "country_status", "country_group", "country_mapped_from_uk", "country_whitelist_ok",
            "dpa_name_raw", "dpa_name_canonical",
            "decision_date", "decision_date_status", "decision_year", "decision_quarter",
            "fine_eur", "fine_numeric_valid", "fine_positive", "fine_log1p", "fine_outlier_flag",
            "turnover_eur", "turnover_numeric_valid", "turnover_log1p", "turnover_outlier_flag",
            "fine_to_turnover_ratio",
            "legal_bases_discussed", "legal_bases_status",
            "principles_violated", "principles_violated_status",
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
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        validations: list[Dict[str, Any]] = []

        for row in reader:
            decision_id = row.get("ID")
            resp = row.get("response", "")
            answers = _answers_from_response(resp)

            schema_echo_flag = 0
            schema_echo_fields: List[str] = []

            # Country and geo enrichment
            country_code_raw, country_status = normalize_country(answers.get("Q1", ""))
            country_code_mapped, mapped_from_uk, whitelist_ok = _clean_country_code(country_code_raw)
            country_code_final, country_group = enrich_country(country_code_mapped)

            # DPA name raw and canonical (clean schema echo)
            dpa_name_raw = (answers.get("Q2", "") or "").strip()
            dpa_name_clean = dpa_name_raw if not _is_schema_echo(dpa_name_raw) else ""
            if dpa_name_clean == "" and dpa_name_raw:
                schema_echo_flag = 1
                schema_echo_fields.append("Q2")
            dpa_name_canonical = normalize_dpa_name(dpa_name_clean)

            # Decision date
            dpr = parse_date_field(answers.get("Q3", ""))
            decision_date = dpr.value.isoformat() if dpr.value else ""
            decision_year = dpr.value.year if dpr.value else ""
            decision_quarter = (f"Q{((dpr.value.month-1)//3)+1}" if dpr.value else "")

            # Fine and turnover (robust parsing)
            fine_eur, fine_valid, _ = parse_number(answers.get("Q37", ""))
            turnover_eur, turnover_valid, _ = parse_number(answers.get("Q38", ""))
            fine_log = f"{log1p(fine_eur):.6f}" if fine_eur is not None else ""
            turnover_log = f"{log1p(turnover_eur):.6f}" if turnover_eur is not None else ""
            fine_outlier = 1 if (fine_eur or 0) > FINE_OUTLIER_HIGH else 0
            turnover_outlier = 1 if (turnover_eur or 0) > TURNOVER_OUTLIER_HIGH else 0
            fine_to_turnover_ratio = ""
            if (fine_eur is not None) and (turnover_eur is not None) and turnover_eur > 0:
                fine_to_turnover_ratio = f"{fine_eur/turnover_eur:.8f}"

            # Multi-selects (clean schema echo)
            li_discussed, li_status = split_multiselect(answers.get("Q32", ""))
            li_discussed_clean = [t for t in li_discussed if not _is_schema_echo(t)]
            if li_discussed and len(li_discussed_clean) < len(li_discussed):
                schema_echo_flag = 1
                schema_echo_fields.append("Q32")

            p_discussed, _ = split_multiselect(answers.get("Q30", ""))
            p_discussed_clean = [t for t in p_discussed if not _is_schema_echo(t)]
            if p_discussed and len(p_discussed_clean) < len(p_discussed):
                schema_echo_flag = 1
                schema_echo_fields.append("Q30")

            p_violated, p_violated_status = split_multiselect(answers.get("Q31", ""))
            p_violated_clean = [t for t in p_violated if not _is_schema_echo(t)]
            if p_violated and len(p_violated_clean) < len(p_violated):
                schema_echo_flag = 1
                schema_echo_fields.append("Q31")

            corrective_powers, _ = split_multiselect(answers.get("Q53", ""))
            corrective_powers_clean = [t for t in corrective_powers if not _is_schema_echo(t)]
            if corrective_powers and len(corrective_powers_clean) < len(corrective_powers):
                schema_echo_flag = 1
                schema_echo_fields.append("Q53")

            # Derived counts and flags (use cleaned lists)
            n_principles_discussed = len([t for t in p_discussed_clean if t])
            n_principles_violated = len([t for t in p_violated_clean if t and not t.startswith("NONE_") and t != "NOT_DETERMINED"]) if p_violated_clean else 0
            n_corrective_measures = len([t for t in corrective_powers_clean if t and t != "NONE"]) if corrective_powers_clean else 0
            severity_measures_present = 1 if any(t in SEVERITY_MEASURES for t in corrective_powers_clean) else 0
            remedy_only_case = 1 if (fine_eur or 0) == 0 and any(t for t in corrective_powers_clean if t and t != "NONE" and t != "ADMINISTRATIVE_FINE") else 0
            breach_case = 1 if answers.get("Q16") == "YES" else 0

            # ISIC mapping (Q12) with schema-echo cleaning
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

            # Text normalization (Q36, Q52, Q67, Q68)
            q36_norm, q36_tok, q36_chars = normalize_text(answers.get("Q36", "") or "")
            q52_norm, q52_tok, q52_chars = normalize_text(answers.get("Q52", "") or "")
            q67_norm, q67_tok, q67_chars = normalize_text(answers.get("Q67", "") or "")
            q68_norm, q68_tok, q68_chars = normalize_text(answers.get("Q68", "") or "")
            q36_lang = detect_language_heuristic(q36_norm)
            q52_lang = detect_language_heuristic(q52_norm)
            q67_lang = detect_language_heuristic(q67_norm)
            q68_lang = detect_language_heuristic(q68_norm)

            writer.writerow({
                "decision_id": decision_id,
                "country_code": country_code_final or "",
                "country_status": country_status,
                "country_group": country_group,
                "country_mapped_from_uk": 1 if mapped_from_uk else 0,
                "country_whitelist_ok": 1 if whitelist_ok else 0,
                "dpa_name_raw": dpa_name_raw,
                "dpa_name_canonical": dpa_name_canonical,
                "decision_date": decision_date,
                "decision_date_status": dpr.status,
                "decision_year": decision_year,
                "decision_quarter": decision_quarter,
                "fine_eur": f"{fine_eur:.6f}" if fine_eur is not None else "",
                "fine_numeric_valid": 1 if fine_valid else 0,
                "fine_positive": 1 if (fine_eur or 0) > 0 else 0,
                "fine_log1p": fine_log,
                "fine_outlier_flag": fine_outlier,
                "turnover_eur": f"{turnover_eur:.6f}" if turnover_eur is not None else "",
                "turnover_numeric_valid": 1 if turnover_valid else 0,
                "turnover_log1p": turnover_log,
                "turnover_outlier_flag": turnover_outlier,
                "fine_to_turnover_ratio": fine_to_turnover_ratio,
                "legal_bases_discussed": ",".join(li_discussed_clean),
                "legal_bases_status": li_status,
                "principles_violated": ",".join(p_violated_clean),
                "principles_violated_status": p_violated_status,
                "isic_code": isic_code,
                "isic_desc": isic_desc,
                "isic_section": isic_section,
                "n_principles_discussed": n_principles_discussed,
                "n_principles_violated": n_principles_violated,
                "n_corrective_measures": n_corrective_measures,
                "severity_measures_present": severity_measures_present,
                "remedy_only_case": remedy_only_case,
                "breach_case": breach_case,
                "q36_text": q36_norm, "q36_lang": q36_lang, "q36_tokens": q36_tok,
                "q52_text": q52_norm, "q52_lang": q52_lang, "q52_tokens": q52_tok,
                "q67_text": q67_norm, "q67_lang": q67_lang, "q67_tokens": q67_tok,
                "q68_text": q68_norm, "q68_lang": q68_lang, "q68_tokens": q68_tok,
                "schema_echo_flag": schema_echo_flag,
                "schema_echo_fields": ";".join(sorted(set(schema_echo_fields))),
            })

            # Minimal validations per plan (keep prior ones)
            flags = []
            if answers.get("Q53", "").find("ADMINISTRATIVE_FINE") >= 0 and not (fine_eur or 0) > 0:
                flags.append("admin_fine_but_zero_amount")
            validations.append({
                "decision_id": decision_id,
                "flags": flags,
            })

    validation_report.write_text(json.dumps(validations, ensure_ascii=False, indent=2), encoding="utf-8")
