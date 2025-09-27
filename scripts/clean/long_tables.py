from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from scripts.clean.typing_status import (
    split_multiselect,
    derive_multiselect_status,
    normalize_multiselect_tokens,
)
from scripts.clean.enum_validate import EnumWhitelist
from scripts.clean.schema_echo import strip_schema_echo
from scripts.parser.validators import dedupe_preserve_order
from scripts.parser.ingest import parse_record

LONG_TABLE_QUESTIONS = {
    "Q10",
    "Q21",
    "Q25",
    "Q28",
    "Q30",
    "Q31",
    "Q32",
    "Q33",
    "Q34",
    "Q35",
    "Q41",
    "Q42",
    "Q43",
    "Q44",
    "Q45",
    "Q46",
    "Q53",
    "Q54",
    "Q56",
    "Q57",
    "Q58",
    "Q59",
    "Q61",
    "Q64",
}


def _resolve_input_format(fieldnames: List[str] | None, requested: str) -> str:
    if requested != "auto":
        return requested
    fieldnames = fieldnames or []
    if "response" in fieldnames:
        return "raw"
    if any(fn.startswith("raw_q") for fn in fieldnames if fn):
        return "wide"
    raise ValueError("Unable to detect input format for long table emission")


class LongEmitter:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        wl_path = Path("resources/enum_whitelist.json")
        self.whitelist = EnumWhitelist.load(wl_path) if wl_path.exists() else EnumWhitelist({})

    def _emit(self, filename: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def _add_tokens(self, out: List[Dict[str, str]], decision_id: str, qkey: str, table_name: str, tokens: List[str], status: str) -> None:
        unknown, known = self.whitelist.validate_tokens(qkey, tokens)
        for t in known:
            out.append({"decision_id": decision_id, "option": t, "status": status, "token_status": "KNOWN"})
        for t in unknown:
            out.append({"decision_id": decision_id, "option": t, "status": status, "token_status": "UNKNOWN"})

    def emit_from_csv(self, input_csv: Path, input_format: str = "auto") -> None:
        with input_csv.open(newline="", encoding="utf-8") as f_in:
            r = csv.DictReader(f_in)
            fmt = _resolve_input_format(r.fieldnames, input_format)

            tables: Dict[str, List[Dict[str, str]]] = {
                "defendant_classifications.csv": [],
                "article_5_discussed.csv": [],
                "article_5_violated.csv": [],
                "article_6_discussed.csv": [],
                "legal_basis_relied_on.csv": [],
                "consent_issues.csv": [],
                "li_test_outcome.csv": [],
                "breach_types.csv": [],
                "special_data_categories.csv": [],
                "mitigating_actions.csv": [],
                "vulnerable_groups.csv": [],
                "corrective_powers.csv": [],
                "corrective_scopes.csv": [],
                "rights_discussed.csv": [],
                "rights_violated.csv": [],
                "access_issues.csv": [],
                "adm_issues.csv": [],
                "dpo_issues.csv": [],
                "transfer_violations.csv": [],
                "aggravating_factors.csv": [],
                "mitigating_factors.csv": [],
                "harm_outcomes.csv": [],
                "benefit_outcomes.csv": [],
                "cooperation_levels.csv": [],
            }

            for row in r:
                decision_id = row.get("decision_id") or row.get("ID")
                if fmt == "raw":
                    parsed = parse_record((row.get("response", "") or ""))
                    answers = parsed["answers"]
                else:
                    answers = {q: row.get(f"raw_{q.lower()}", "") for q in LONG_TABLE_QUESTIONS}

                def add_multiselect(qkey: str, table_name: str) -> None:
                    ms_parsed = split_multiselect(answers.get(qkey, ""))
                    tokens: List[str] = []
                    for token in ms_parsed.tokens:
                        cleaned_tokens, _ = normalize_multiselect_tokens(qkey, token)
                        tokens.extend(t for t in cleaned_tokens if t)
                    tokens = dedupe_preserve_order(tokens)
                    if not tokens:
                        status = derive_multiselect_status(qkey, ms_parsed.tokens)
                        if status != "NOT_MENTIONED":
                            tables[table_name].append({
                                "decision_id": decision_id,
                                "option": status,
                                "status": status,
                                "token_status": "STATUS_ONLY",
                            })
                        return
                    status = derive_multiselect_status(qkey, tokens)
                    self._add_tokens(tables[table_name], decision_id, qkey, table_name, tokens, status)

                def add_single(qkey: str, table_name: str) -> None:
                    val = (answers.get(qkey, "") or "").strip()
                    val, _ = strip_schema_echo(val)
                    if not val:
                        return
                    unknown, known = self.whitelist.validate_tokens(qkey, [val])
                    if known:
                        tables[table_name].append({"decision_id": decision_id, "option": known[0], "status": "DISCUSSED", "token_status": "KNOWN"})
                    if unknown:
                        tables[table_name].append({"decision_id": decision_id, "option": unknown[0], "status": "DISCUSSED", "token_status": "UNKNOWN"})

                add_multiselect("Q10", "defendant_classifications.csv")
                add_multiselect("Q21", "breach_types.csv")
                add_multiselect("Q25", "special_data_categories.csv")
                add_multiselect("Q28", "mitigating_actions.csv")
                add_multiselect("Q30", "article_5_discussed.csv")
                add_multiselect("Q31", "article_5_violated.csv")
                add_multiselect("Q32", "article_6_discussed.csv")
                add_multiselect("Q33", "legal_basis_relied_on.csv")
                add_multiselect("Q34", "consent_issues.csv")
                add_single("Q35", "li_test_outcome.csv")

                add_multiselect("Q46", "vulnerable_groups.csv")

                add_multiselect("Q53", "corrective_powers.csv")
                add_multiselect("Q54", "corrective_scopes.csv")

                add_multiselect("Q56", "rights_discussed.csv")
                add_multiselect("Q57", "rights_violated.csv")
                add_multiselect("Q58", "access_issues.csv")
                add_multiselect("Q59", "adm_issues.csv")

                add_multiselect("Q61", "dpo_issues.csv")

                add_multiselect("Q64", "transfer_violations.csv")

                add_multiselect("Q41", "aggravating_factors.csv")
                add_multiselect("Q42", "mitigating_factors.csv")
                add_multiselect("Q43", "harm_outcomes.csv")
                add_multiselect("Q44", "benefit_outcomes.csv")
                add_multiselect("Q45", "cooperation_levels.csv")

            for filename, rows in tables.items():
                self._emit(filename, rows, ["decision_id", "option", "status", "token_status"]) 
