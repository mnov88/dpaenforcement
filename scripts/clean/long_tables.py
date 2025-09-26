from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from scripts.clean.typing_status import split_multiselect

SCHEMA_ECHO_PREFIXES = ("TYPE:", "ENUM:", "MULTI_SELECT:")


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


def _is_schema_echo(value: str) -> bool:
    v = (value or "").strip()
    return any(v.startswith(p) for p in SCHEMA_ECHO_PREFIXES)


class LongEmitter:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _emit(self, filename: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def emit_from_csv(self, input_csv: Path) -> None:
        with input_csv.open(newline="", encoding="utf-8") as f_in:
            r = csv.DictReader(f_in)

            tables: Dict[str, List[Dict[str, str]]] = {
                "article_5_discussed.csv": [],
                "article_5_violated.csv": [],
                "article_6_discussed.csv": [],
                "legal_basis_relied_on.csv": [],
                "consent_issues.csv": [],
                "li_test_outcome.csv": [],
                "breach_types.csv": [],
                "vulnerable_groups.csv": [],
                "corrective_powers.csv": [],
                "rights_discussed.csv": [],
                "rights_violated.csv": [],
            }

            for row in r:
                decision_id = row.get("ID")
                answers = _answers_from_response(row.get("response", ""))

                def add_multiselect(qkey: str, table_name: str) -> None:
                    tokens, status = split_multiselect(answers.get(qkey, ""))
                    tokens = [t for t in tokens if not _is_schema_echo(t)]
                    if not tokens:
                        return
                    for t in tokens:
                        tables[table_name].append({
                            "decision_id": decision_id,
                            "option": t,
                            "status": status,
                        })

                def add_single(qkey: str, table_name: str) -> None:
                    val = (answers.get(qkey, "") or "").strip()
                    if not val or _is_schema_echo(val):
                        return
                    tables[table_name].append({
                        "decision_id": decision_id,
                        "option": val,
                        "status": "DISCUSSED",
                    })

                add_multiselect("Q30", "article_5_discussed.csv")
                add_multiselect("Q31", "article_5_violated.csv")
                add_multiselect("Q32", "article_6_discussed.csv")
                add_multiselect("Q33", "legal_basis_relied_on.csv")
                add_multiselect("Q34", "consent_issues.csv")
                add_single("Q35", "li_test_outcome.csv")
                add_multiselect("Q21", "breach_types.csv")
                add_multiselect("Q46", "vulnerable_groups.csv")
                add_multiselect("Q53", "corrective_powers.csv")
                add_multiselect("Q56", "rights_discussed.csv")
                add_multiselect("Q57", "rights_violated.csv")

            for filename, rows in tables.items():
                self._emit(filename, rows, ["decision_id", "option", "status"])
