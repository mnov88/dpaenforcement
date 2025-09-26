from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import List, Optional, Tuple


DATE_FMT = "%Y-%m-%d"


@dataclass
class DateParseResult:
    raw: str
    value: Optional[datetime]
    status: str  # DISCUSSED | NOT_DISCUSSED | NOT_MENTIONED | PARSE_ERROR
    error: Optional[str] = None


ISO_PREFIX = "ISO_3166-1_ALPHA-2:"

_STATUS_TOKENS = {
    "NOT_DISCUSSED",
    "NOT_MENTIONED",
    "NOT_APPLICABLE",
}


def parse_date_field(raw: str) -> DateParseResult:
    raw_norm = (raw or "").strip()
    if not raw_norm:
        return DateParseResult(raw="", value=None, status="NOT_MENTIONED")
    if raw_norm in _STATUS_TOKENS:
        return DateParseResult(raw=raw_norm, value=None, status=raw_norm)
    try:
        dt = datetime.strptime(raw_norm, DATE_FMT).replace(tzinfo=UTC)
        return DateParseResult(raw=raw_norm, value=dt, status="DISCUSSED")
    except Exception as exc:
        return DateParseResult(raw=raw_norm, value=None, status="PARSE_ERROR", error=str(exc))


def normalize_country(raw: str) -> Tuple[Optional[str], str]:
    # Extract the country code after the final colon per spec
    raw = (raw or "").strip()
    if not raw:
        return None, "NOT_MENTIONED"
    if raw.startswith(ISO_PREFIX):
        try:
            code = raw.split(":")[-1].strip()
            return code, "DISCUSSED"
        except Exception:
            return None, "UNCLEAR"
    return raw, "DISCUSSED"


_NUM_ALLOWED = re.compile(r"[^0-9eE+\-\., ]+")


def _sanitize_numeric_string(raw: str) -> str:
    # Remove currency symbols/letters and keep digits, signs, decimal/thousand separators, exponents
    s = _NUM_ALLOWED.sub("", raw or "").strip()
    # Remove spaces and commas used as thousand separators
    s = s.replace(" ", "").replace(",", "")
    # If there are multiple dots, assume thousand separators and remove all but last
    if s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]
    return s


@dataclass
class NumericParseResult:
    raw: str
    value: Optional[float]
    status: str  # DISCUSSED | NOT_MENTIONED | NOT_DISCUSSED | NOT_APPLICABLE | PARSE_ERROR | NEGATIVE_VALUE
    valid: bool
    error: Optional[str] = None


_NUMERIC_STATUS_TOKENS = {
    "NOT_DISCUSSED",
    "NOT_MENTIONED",
    "NOT_APPLICABLE",
    "NONE",
}


def parse_number(raw: str) -> NumericParseResult:
    raw_norm = (raw or "").strip()
    if not raw_norm:
        return NumericParseResult(raw="", value=None, status="NOT_MENTIONED", valid=True)
    if raw_norm in _NUMERIC_STATUS_TOKENS:
        return NumericParseResult(raw=raw_norm, value=None, status=raw_norm, valid=True)
    s = _sanitize_numeric_string(raw_norm)
    if not s:
        return NumericParseResult(
            raw=raw_norm,
            value=None,
            status="PARSE_ERROR",
            valid=False,
            error="sanitized_numeric_string_empty",
        )
    try:
        val = float(s)
    except Exception as exc:
        return NumericParseResult(
            raw=raw_norm,
            value=None,
            status="PARSE_ERROR",
            valid=False,
            error=str(exc),
        )
    if val < 0:
        return NumericParseResult(
            raw=raw_norm,
            value=None,
            status="NEGATIVE_VALUE",
            valid=False,
            error="value_lt_zero",
        )
    return NumericParseResult(raw=raw_norm, value=val, status="DISCUSSED", valid=True)


@dataclass
class MultiSelectParseResult:
    raw: str
    tokens: List[str]
    status: str


def split_multiselect(raw: str) -> MultiSelectParseResult:
    raw_norm = (raw or "").strip()
    if not raw_norm:
        return MultiSelectParseResult(raw="", tokens=[], status="NOT_MENTIONED")
    tokens = [t.strip() for t in raw_norm.split(",") if t.strip()]
    status = "DISCUSSED" if tokens else "NOT_MENTIONED"
    return MultiSelectParseResult(raw=raw_norm, tokens=tokens, status=status)


EXCLUSIVE_MARKERS = {
    "NOT_APPLICABLE",
    "NONE_MENTIONED",
    "NONE_DISCUSSED",
    "NONE",
    "NOT_DETERMINED",
}


def derive_multiselect_status(qkey: str, tokens: List[str]) -> str:
    if not tokens:
        return "NOT_MENTIONED"
    tset = set(tokens)
    if qkey in {"Q31", "Q57"}:
        if "NONE_VIOLATED" in tset:
            return "NONE_VIOLATED"
        if "NOT_DETERMINED" in tset:
            return "NOT_DETERMINED"
    if "NOT_APPLICABLE" in tset and all(t == "NOT_APPLICABLE" for t in tokens):
        return "NOT_APPLICABLE"
    if "NONE_MENTIONED" in tset and all(t == "NONE_MENTIONED" for t in tokens):
        return "NONE_MENTIONED"
    if qkey == "Q30" and "NONE_DISCUSSED" in tset and all(t == "NONE_DISCUSSED" for t in tokens):
        return "NONE_DISCUSSED"
    return "DISCUSSED"


def detect_exclusivity_conflict(tokens: List[str]) -> int:
    if not tokens:
        return 0
    tset = set(tokens)
    markers = tset.intersection(EXCLUSIVE_MARKERS)
    if not markers:
        return 0
    substantive = [t for t in tokens if t not in EXCLUSIVE_MARKERS]
    return 1 if substantive else 0
