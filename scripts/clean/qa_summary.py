from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Keep in sync with MULTI_FIELDS prefixes in wide_output.py
MULTI_PREFIXES: List[Tuple[str, str]] = [
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


def create_qa_summary(wide_csv: Path, out_csv: Path, top_k: int = 5) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, object]] = {}

    with wide_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # Precompute columns existence
        cols = set(r.fieldnames or [])

        counters: Dict[str, Dict[str, object]] = {}
        for _, prefix in MULTI_PREFIXES:
            known_col = f"{prefix}_known"
            unknown_col = f"{prefix}_unknown"
            status_col = f"{prefix}_status"
            if not ({known_col, unknown_col, status_col} <= cols):
                continue
            counters[prefix] = {
                "total_rows": 0,
                "rows_with_known": 0,
                "rows_with_unknown": 0,
                "total_unknown_tokens": 0,
                "status_counts": Counter(),
                "unknown_token_counts": Counter(),
            }

        for row in r:
            for _, prefix in MULTI_PREFIXES:
                if prefix not in counters:
                    continue
                known_val = row.get(f"{prefix}_known", "") or ""
                unknown_val = row.get(f"{prefix}_unknown", "") or ""
                status_val = (row.get(f"{prefix}_status", "") or "").strip()

                c = counters[prefix]
                c["total_rows"] += 1
                if known_val.strip():
                    c["rows_with_known"] += 1
                if unknown_val.strip():
                    c["rows_with_unknown"] += 1
                    tokens = [t for t in unknown_val.split(",") if t]
                    c["total_unknown_tokens"] += len(tokens)
                    c["unknown_token_counts"].update(tokens)
                if status_val:
                    c["status_counts"][status_val] += 1

        # Convert counters to serializable dicts
        for prefix, c in counters.items():
            status_counts = dict(c["status_counts"])  # type: ignore
            top_unknown = c["unknown_token_counts"].most_common(top_k)  # type: ignore
            stats[prefix] = {
                "field_prefix": prefix,
                "total_rows": c["total_rows"],
                "rows_with_known": c["rows_with_known"],
                "rows_with_unknown": c["rows_with_unknown"],
                "total_unknown_tokens": c["total_unknown_tokens"],
                "status_counts_json": json.dumps(status_counts, ensure_ascii=False),
                "top_unknown_tokens_json": json.dumps(top_unknown, ensure_ascii=False),
            }

    # Write summary CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f_out:
        fieldnames = [
            "field_prefix",
            "total_rows",
            "rows_with_known",
            "rows_with_unknown",
            "total_unknown_tokens",
            "status_counts_json",
            "top_unknown_tokens_json",
        ]
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        for prefix, row in stats.items():
            w.writerow(row)
