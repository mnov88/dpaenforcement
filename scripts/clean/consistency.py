from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any

from scripts.parser.ingest import parse_record
from scripts.clean.typing_status import parse_number


CONSISTENCY_QUESTIONS = {
    "Q1",
    "Q4",
    "Q5",
    "Q16",
    "Q17",
    "Q18",
    "Q26",
    "Q27",
    "Q37",
    "Q38",
    "Q39",
    "Q49",
    "Q53",
    "Q60",
    "Q61",
    "Q62",
}


def _resolve_input_format(fieldnames: List[str] | None, requested: str) -> str:
    if requested != "auto":
        return requested
    fieldnames = fieldnames or []
    if "response" in fieldnames:
        return "raw"
    if any(fn.startswith("raw_q") for fn in fieldnames if fn):
        return "wide"
    raise ValueError("Unable to detect input format for consistency checks")


def run_consistency_checks(input_csv: Path, report_json: Path, input_format: str = "auto") -> None:
    report: List[Dict[str, Any]] = []
    with input_csv.open(newline="", encoding="utf-8") as f_in:
        r = csv.DictReader(f_in)
        fmt = _resolve_input_format(r.fieldnames, input_format)
        for row in r:
            decision_id = row.get("decision_id") or row.get("ID")
            if fmt == "raw":
                parsed = parse_record((row.get("response", "") or ""))
                a = parsed["answers"]
            else:
                a = {q: row.get(f"raw_{q.lower()}", "") for q in CONSISTENCY_QUESTIONS}
            flags: List[str] = []

            # Breach notification chain
            if a.get("Q17") == "YES_REQUIRED" and a.get("Q18") == "NOT_APPLICABLE":
                flags.append("q17_yes_required_but_q18_not_applicable")
            if a.get("Q18", "").startswith("YES_") and a.get("Q16") == "NO":
                flags.append("q18_submitted_but_q16_no_breach")

            # Article 34 vs 33
            if a.get("Q27") == "YES_REQUIRED" and a.get("Q16") != "YES":
                flags.append("q27_yes_required_but_q16_not_yes")

            # Data subject notifications
            if a.get("Q26") == "YES_NOTIFIED" and a.get("Q16") == "NO":
                flags.append("q26_yes_notified_but_q16_no_breach")

            # DPO
            if a.get("Q60") == "YES_REQUIRED_NOT_APPOINTED" and "NO_DPO_APPOINTED" not in (a.get("Q61", "")):
                flags.append("q60_required_not_appointed_but_q61_missing_no_dpo")

            # Enforcement vs fine amount
            fine_result = parse_number(a.get("Q37", ""))
            fine_value = fine_result.value or 0.0
            if "ADMINISTRATIVE_FINE" in (a.get("Q53", "") or ""):
                if fine_value <= 0:
                    flags.append("admin_fine_present_but_fine_zero_or_invalid")
            if fine_result.status == "PARSE_ERROR":
                flags.append("fine_amount_parse_error")

            # Appeal logic
            if a.get("Q4") == "NO" and a.get("Q5") != "NOT_APPLICABLE":
                flags.append("appeal_outcome_present_but_q4_no")

            # Caps vs fine logic
            cap = a.get("Q39")
            if fine_value == 0 and cap and cap.startswith("HIT_"):
                flags.append("cap_hit_with_zero_fine")
            # Turnover cap verification not possible if turnover missing
            turnover_result = parse_number(a.get("Q38", ""))
            if turnover_result.status == "PARSE_ERROR":
                flags.append("turnover_parse_error")
            if cap == "HIT_4PCT_TURNOVER_CAP" and not (turnover_result.raw or "").strip():
                flags.append("turnover_cap_claim_unverifiable_turnover_missing")

            # Cross-border checks
            if a.get("Q49") in ("YES_LEAD_AUTHORITY_CASE", "YES_MULTIPLE_DPAS_INVOLVED") or a.get("Q62") in (
                "LEAD_SUPERVISORY_AUTHORITY_CASE",
                "CONCERNED_AUTHORITY_CASE",
                "JOINT_INVESTIGATION",
            ):
                # If country not EU/EEA and no NON_EU_ENTITY_INVOLVED
                if a.get("Q62") != "NON_EU_ENTITY_INVOLVED" and a.get("Q1", "").endswith(": EU") is False:
                    # We cannot fully verify country group here; flag for review
                    flags.append("cross_border_claim_requires_geo_verification")

            if flags:
                report.append({"decision_id": decision_id, "flags": flags})

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
