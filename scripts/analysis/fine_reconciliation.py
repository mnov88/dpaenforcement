"""Utilities for comparing and reconciling GDPR fine amounts.

This module reads the AI-extracted wide dataset together with the
human-curated annotations and provides helpers to:

* compute record-level comparisons between both sources; and
* overwrite the AI values with human annotations when conflicts are found.

The script exposes a small CLI with two subcommands::

    python -m scripts.analysis.fine_reconciliation compare
    python -m scripts.analysis.fine_reconciliation apply --output-csv outputs/cleaned_wide_with_human.csv

Both commands default to the canonical dataset locations within the
repository but allow overriding the paths for ad-hoc investigations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.clean.typing_status import NumericParseResult, parse_number
from scripts.clean.wide_output import FINE_OUTLIER_HIGH

# Approximate currency conversion rates to EUR.  These figures do not need to
# be perfect – they only provide a deterministic, order-of-magnitude reference
# for reconciling datasets that mix local currencies with Euro-denominated
# amounts.
CURRENCY_TO_EUR: Dict[str, float] = {
    "EUR": 1.0,
    "GBP": 1.17,
    "SEK": 0.087,
    "DKK": 0.134,
    "NOK": 0.089,
    "PLN": 0.23,
    "RON": 0.20,
    "BGN": 0.51,
    "HUF": 0.0026,
    "CZK": 0.041,
    "ISK": 0.0068,
    "HRK": 0.133,
    "CHF": 1.05,
    "USD": 0.92,
}

_CURRENCY_SYMBOLS: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("EUR", ("EUR", "€", "EURO", "EUROS")),
    ("GBP", ("GBP", "£", "POUND", "POUNDS", "STERLING")),
    ("SEK", ("SEK", "SWEDISH KR", "KRONOR")),
    ("DKK", ("DKK", "DANISH KR")),
    ("NOK", ("NOK", "NORWEGIAN KR", "KRONER")),
    ("PLN", ("PLN", "ZŁ", "ZL", "ZLOTY", "ZLOTI")),
    ("RON", ("RON", "LEI")),
    ("BGN", ("BGN", "LEV", "LEVA")),
    ("HUF", ("HUF", "FT", "FORINT")),
    ("CZK", ("CZK", "KČ", "KCS")),
    ("ISK", ("ISK", "KRÓNA", "KRONA")),
    ("HRK", ("HRK", "KUNA", "KN")),
    ("CHF", ("CHF", "FRANC")),
    ("USD", ("USD", "$", "DOLLAR")),
)

_CURRENCY_CODE_RE = re.compile(r"\b([A-Z]{3})\b")
_ID_FIELD_FALLBACKS = ("ID", "\ufeffID", "DecisionID")


@dataclass
class FineRecord:
    """Structured representation of a single fine observation."""

    decision_id: str
    amount_eur: Optional[float]
    amount_native: Optional[float]
    currency: Optional[str]
    raw: str
    source: str  # "ai" or "human"


@dataclass
class FineComparison:
    """Record-by-record comparison payload."""

    decision_id: str
    ai_amount_eur: Optional[float]
    ai_raw: str
    human_amount_eur: Optional[float]
    human_amount_native: Optional[float]
    human_currency: Optional[str]
    human_raw: str
    status: str
    difference_eur: Optional[float]


@dataclass
class ComparisonSummary:
    """Aggregated counts derived from the comparison output."""

    total: int
    matches: int
    conflicts: int
    ai_missing: int
    human_missing: int
    missing_both: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "matches": self.matches,
            "conflicts": self.conflicts,
            "ai_missing": self.ai_missing,
            "human_missing": self.human_missing,
            "missing_both": self.missing_both,
        }


def _detect_currency(raw: str) -> Optional[str]:
    if not raw:
        return None
    upper = raw.upper()
    for currency, tokens in _CURRENCY_SYMBOLS:
        if any(token in upper for token in tokens):
            return currency
    match = _CURRENCY_CODE_RE.search(upper)
    if match and match.group(1) in CURRENCY_TO_EUR:
        return match.group(1)
    return None


def _parse_numeric(value: str) -> Optional[float]:
    result: NumericParseResult = parse_number(value)
    return result.value if result.valid and result.value is not None else None


def _extract_decision_id(row: Mapping[str, str]) -> Optional[str]:
    for key in _ID_FIELD_FALLBACKS:
        if key in row and row[key]:
            return row[key]
    return None


def _format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def load_ai_dataset(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_ai_fines(rows: Iterable[Mapping[str, str]]) -> Dict[str, FineRecord]:
    fines: Dict[str, FineRecord] = {}
    for row in rows:
        decision_id = row.get("decision_id")
        if not decision_id:
            continue
        fine_eur = _parse_numeric(row.get("fine_eur", ""))
        fines[decision_id] = FineRecord(
            decision_id=decision_id,
            amount_eur=fine_eur,
            amount_native=fine_eur,  # AI data is already expressed in EUR.
            currency="EUR" if fine_eur is not None else None,
            raw=row.get("fine_raw", ""),
            source="ai",
        )
    return fines


def load_human_fines(path: Path) -> Dict[str, FineRecord]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fines: Dict[str, FineRecord] = {}
        for row in reader:
            decision_id = _extract_decision_id(row)
            if not decision_id:
                continue
            raw_fine = (row.get("Fine") or "").strip()
            numeric_raw = row.get("Fine_Numeric") or ""
            amount_native = None
            if numeric_raw.strip():
                amount_native = _parse_numeric(numeric_raw)
            if amount_native is None and raw_fine:
                amount_native = _parse_numeric(raw_fine)
            currency = _detect_currency(raw_fine)
            amount_eur: Optional[float] = None
            if amount_native is not None and currency:
                rate = CURRENCY_TO_EUR.get(currency)
                if rate:
                    amount_eur = amount_native * rate
            elif amount_native is not None:
                # Assume EUR when the annotation provides a number but no
                # explicit currency.  This mirrors the historical dataset
                # where Euro amounts were often recorded without a suffix.
                amount_eur = amount_native
                currency = "EUR"
            fines[decision_id] = FineRecord(
                decision_id=decision_id,
                amount_eur=amount_eur,
                amount_native=amount_native,
                currency=currency,
                raw=raw_fine or numeric_raw.strip(),
                source="human",
            )
    return fines


def compare_fines(
    ai_fines: Mapping[str, FineRecord],
    human_fines: Mapping[str, FineRecord],
    *,
    tolerance: float = 1.0,
) -> List[FineComparison]:
    decision_ids = set(ai_fines) | set(human_fines)
    comparisons: List[FineComparison] = []
    for decision_id in sorted(decision_ids):
        ai_record = ai_fines.get(decision_id)
        human_record = human_fines.get(decision_id)
        ai_amount = ai_record.amount_eur if ai_record else None
        ai_raw = ai_record.raw if ai_record else ""
        human_amount = human_record.amount_eur if human_record else None
        human_native = human_record.amount_native if human_record else None
        human_currency = human_record.currency if human_record else None
        human_raw = human_record.raw if human_record else ""
        difference = None
        status = "MISSING_BOTH"
        if ai_amount is None and human_amount is None:
            status = "MISSING_BOTH"
        elif ai_amount is None:
            status = "AI_MISSING"
            difference = None if human_amount is None else abs(human_amount)
        elif human_amount is None:
            status = "HUMAN_MISSING"
            difference = None
        else:
            difference = abs(ai_amount - human_amount)
            status = "MATCH" if difference <= tolerance else "CONFLICT"
        comparisons.append(
            FineComparison(
                decision_id=decision_id,
                ai_amount_eur=ai_amount,
                ai_raw=ai_raw,
                human_amount_eur=human_amount,
                human_amount_native=human_native,
                human_currency=human_currency,
                human_raw=human_raw,
                status=status,
                difference_eur=difference,
            )
        )
    return comparisons


def summarise_comparisons(comparisons: Sequence[FineComparison]) -> ComparisonSummary:
    total = len(comparisons)
    matches = sum(1 for c in comparisons if c.status == "MATCH")
    conflicts = sum(1 for c in comparisons if c.status == "CONFLICT")
    ai_missing = sum(1 for c in comparisons if c.status == "AI_MISSING")
    human_missing = sum(1 for c in comparisons if c.status == "HUMAN_MISSING")
    missing_both = sum(1 for c in comparisons if c.status == "MISSING_BOTH")
    return ComparisonSummary(
        total=total,
        matches=matches,
        conflicts=conflicts,
        ai_missing=ai_missing,
        human_missing=human_missing,
        missing_both=missing_both,
    )


def apply_human_overrides(
    rows: Sequence[Mapping[str, str]],
    human_fines: Mapping[str, FineRecord],
) -> Tuple[List[Dict[str, str]], List[str]]:
    """Return updated rows and a log of overridden decision IDs."""

    updated_rows: List[Dict[str, str]] = []
    overrides: List[str] = []
    for row in rows:
        row_copy = dict(row)
        decision_id = row_copy.get("decision_id")
        if decision_id and decision_id in human_fines:
            human_record = human_fines[decision_id]
            amount_eur = human_record.amount_eur
            if amount_eur is not None:
                overrides.append(decision_id)
                row_copy["fine_eur"] = _format_float(amount_eur)
                row_copy["fine_numeric_valid"] = "1"
                row_copy["fine_positive"] = "1" if amount_eur > 0 else "0"
                row_copy["fine_log1p"] = _format_float(math.log1p(amount_eur))
                row_copy["fine_outlier_flag"] = (
                    "1" if amount_eur > FINE_OUTLIER_HIGH else "0"
                )
                row_copy["fine_raw"] = human_record.raw
                row_copy["fine_status"] = "DISCUSSED"
                row_copy["fine_error"] = ""
                # Maintain the fine-to-turnover ratio if turnover is available.
                turnover = _parse_numeric(row_copy.get("turnover_eur", ""))
                if turnover and turnover > 0:
                    ratio = amount_eur / turnover
                    row_copy["fine_to_turnover_ratio"] = f"{ratio:.8f}"
                else:
                    row_copy["fine_to_turnover_ratio"] = row_copy.get(
                        "fine_to_turnover_ratio", ""
                    )
                row_copy.setdefault("fine_amount_human_native", "")
                row_copy.setdefault("fine_amount_human_eur", "")
                row_copy.setdefault("fine_currency_human", "")
                if human_record.amount_native is not None:
                    row_copy["fine_amount_human_native"] = _format_float(
                        human_record.amount_native
                    )
                if human_record.amount_eur is not None:
                    row_copy["fine_amount_human_eur"] = _format_float(
                        human_record.amount_eur
                    )
                if human_record.currency:
                    row_copy["fine_currency_human"] = human_record.currency
        updated_rows.append(row_copy)
    return updated_rows, overrides


def write_csv(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write CSV with no rows")
    fieldnames: List[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_comparison_csv(
    comparisons: Sequence[FineComparison],
    path: Path,
) -> None:
    fieldnames = [
        "decision_id",
        "ai_amount_eur",
        "ai_raw",
        "human_amount_eur",
        "human_amount_native",
        "human_currency",
        "human_raw",
        "status",
        "difference_eur",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for comp in comparisons:
            writer.writerow(
                {
                    "decision_id": comp.decision_id,
                    "ai_amount_eur": _format_float(comp.ai_amount_eur),
                    "ai_raw": comp.ai_raw,
                    "human_amount_eur": _format_float(comp.human_amount_eur),
                    "human_amount_native": _format_float(
                        comp.human_amount_native
                    ),
                    "human_currency": comp.human_currency or "",
                    "human_raw": comp.human_raw,
                    "status": comp.status,
                    "difference_eur": _format_float(comp.difference_eur),
                }
            )


def _cli_compare(args: argparse.Namespace) -> int:
    ai_rows = load_ai_dataset(args.ai_csv)
    ai_fines = load_ai_fines(ai_rows)
    human_fines = load_human_fines(args.human_csv)
    comparisons = compare_fines(ai_fines, human_fines, tolerance=args.tolerance)
    summary = summarise_comparisons(comparisons)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    if args.out_csv:
        write_comparison_csv(comparisons, args.out_csv)
    if args.summary_out:
        with args.summary_out.open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2, sort_keys=True)
    if args.limit and args.limit > 0:
        conflicts = [c for c in comparisons if c.status == "CONFLICT"]
        conflicts.sort(key=lambda c: (c.difference_eur or 0), reverse=True)
        preview = conflicts[: args.limit]
        if preview:
            print("Top conflicts (decision_id, ai_eur, human_eur, diff):")
            for comp in preview:
                print(
                    f"  {comp.decision_id}: {comp.ai_amount_eur} vs {comp.human_amount_eur} (Δ={comp.difference_eur})"
                )
    return 0


def _cli_apply(args: argparse.Namespace) -> int:
    ai_rows = load_ai_dataset(args.ai_csv)
    human_fines = load_human_fines(args.human_csv)
    updated_rows, overrides = apply_human_overrides(ai_rows, human_fines)
    write_csv(updated_rows, args.output_csv)
    if overrides:
        print(f"Updated {len(overrides)} records from human annotations.")
    else:
        print("No overrides were applied.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare", help="Generate a comparison between AI and human fines"
    )
    compare_parser.add_argument(
        "--ai-csv",
        type=Path,
        default=Path("outputs/cleaned_wide.csv"),
        help="Path to the AI-generated wide dataset.",
    )
    compare_parser.add_argument(
        "--human-csv",
        type=Path,
        default=Path("raw-data/all_gdpr_fines_raw_human_annotations.csv"),
        help="Path to the human annotated fines.",
    )
    compare_parser.add_argument(
        "--out-csv",
        type=Path,
        help="Optional path to write the detailed comparison table.",
    )
    compare_parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional path to write the aggregate summary as JSON.",
    )
    compare_parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Difference tolerance (in EUR) treated as a match.",
    )
    compare_parser.add_argument(
        "--limit",
        type=int,
        help="Print the top-N conflicting rows to stdout.",
    )
    compare_parser.set_defaults(func=_cli_compare)

    apply_parser = subparsers.add_parser(
        "apply", help="Overwrite AI fines with human annotations"
    )
    apply_parser.add_argument(
        "--ai-csv",
        type=Path,
        default=Path("outputs/cleaned_wide.csv"),
        help="Path to the AI-generated wide dataset.",
    )
    apply_parser.add_argument(
        "--human-csv",
        type=Path,
        default=Path("raw-data/all_gdpr_fines_raw_human_annotations.csv"),
        help="Path to the human annotated fines.",
    )
    apply_parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination for the reconciled dataset.",
    )
    apply_parser.set_defaults(func=_cli_apply)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
