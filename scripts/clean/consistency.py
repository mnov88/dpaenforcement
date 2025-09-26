from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any


def _answers_from_response(resp: str) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    for ln in (resp or "").splitlines():
        if ln.startswith("Answer "):
            try:
                idx = ln.index(":")
                left = ln[len("Answer "):idx].strip()
                val = ln[idx + 1 :].strip()
                answers[f"Q{left}"] = val
            except Exception:
                continue
    return answers


def run_consistency_checks(input_csv: Path, report_json: Path) -> None:
    report: List[Dict[str, Any]] = []
    with input_csv.open(newline="", encoding="utf-8") as f_in:
        r = csv.DictReader(f_in)
        for row in r:
            decision_id = row.get("ID")
            a = _answers_from_response(row.get("response", ""))
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
            if "ADMINISTRATIVE_FINE" in (a.get("Q53", "")):
                try:
                    fine = float(a.get("Q37", "") or 0)
                except Exception:
                    fine = 0.0
                if fine <= 0:
                    flags.append("admin_fine_present_but_fine_zero_or_invalid")

            # Cross-border
            if a.get("Q49") in ("YES_LEAD_AUTHORITY_CASE",) or a.get("Q62") in (
                "LEAD_SUPERVISORY_AUTHORITY_CASE",
                "CONCERNED_AUTHORITY_CASE",
                "JOINT_INVESTIGATION",
            ):
                # No hard fail here; just note unverifiable if country code is outside EEA/EU (not checked now)
                pass

            if flags:
                report.append({"decision_id": decision_id, "flags": flags})

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
