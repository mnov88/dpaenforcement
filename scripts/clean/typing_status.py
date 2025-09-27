from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import List, Optional, Tuple

from .schema_echo import strip_schema_echo


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
_SCHEMA_TOKEN_RE = re.compile(r"(?:^|\s)(?:TYPE|ENUM|MULTI_SELECT):[A-Z0-9_]+", re.IGNORECASE)


def _strip_schema_tokens(raw: str) -> str:
    """Remove schema-echo artefacts like ``TYPE:NUMBER`` before sanitizing.

    These tokens are emitted by the questionnaire schema and should not be
    interpreted as part of the numeric value.
    """

    return _SCHEMA_TOKEN_RE.sub(" ", raw)


def _sanitize_numeric_string(raw: str) -> str:
    # Remove currency symbols/letters and keep digits, signs, decimal/thousand separators, exponents
    s = _NUM_ALLOWED.sub("", raw or "").strip()
    # Remove spaces and commas used as thousand separators
    s = s.replace(" ", "").replace(",", "")
    # If there are multiple dots, assume thousand separators and remove all but last
    if s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]
    # Drop stray leading exponent markers introduced by schema tokens/currency strings
    s = s.lstrip("eE")
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
    if raw_norm.upper() == "NO":
        return NumericParseResult(raw=raw_norm, value=None, status="NOT_MENTIONED", valid=True)
    if raw_norm.lower() == "null":
        return NumericParseResult(raw=raw_norm, value=None, status="NOT_MENTIONED", valid=True)
    stripped = _strip_schema_tokens(raw_norm)
    stripped_norm = stripped.strip()
    if stripped_norm.lower() == "null":
        return NumericParseResult(raw=raw_norm, value=None, status="NOT_MENTIONED", valid=True)
    s = _sanitize_numeric_string(stripped_norm)
    if not s:
        if _SCHEMA_TOKEN_RE.search(raw_norm):
            return NumericParseResult(raw=raw_norm, value=None, status="NOT_MENTIONED", valid=True)
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
    "NONE_VIOLATED",
}


CANONICAL_MULTI_OPTIONS = {
    "Q43": [
        "YES_MATERIAL_HARM",
        "YES_NON_MATERIAL_HARM",
        "YES_BOTH_TYPES",
        "NO_HARM_DOCUMENTED",
        "NO",
        "NOT_DISCUSSED",
        "UNCLEAR",
    ],
    "Q44": [
        "YES_QUANTIFIED",
        "YES_MENTIONED_UNQUANTIFIED",
        "NO_BENEFIT",
        "NO",
        "NOT_DISCUSSED",
        "UNCLEAR",
    ],
    "Q45": [
        "FULL_COOPERATION",
        "PARTIAL_COOPERATION",
        "NON_COOPERATIVE",
        "OBSTRUCTIVE",
        "NO",
        "NOT_DISCUSSED",
        "UNCLEAR",
    ],
}


def derive_multiselect_status(qkey: str, tokens: List[str]) -> str:
    if not tokens:
        return "NOT_MENTIONED"

    if detect_exclusivity_conflict(tokens):
        return "MIXED_CONTRADICTORY"

    tset = set(tokens)
    if qkey in {"Q31", "Q57"}:
        if tset == {"NONE_VIOLATED"}:
            return "NONE_VIOLATED"
        if tset == {"NOT_DETERMINED"}:
            return "NOT_DETERMINED"
    if tset == {"NOT_APPLICABLE"}:
        return "NOT_APPLICABLE"
    if tset == {"NONE_MENTIONED"}:
        return "NONE_MENTIONED"
    if qkey == "Q30" and tset == {"NONE_DISCUSSED"}:
        return "NONE_DISCUSSED"
    return "DISCUSSED"


def detect_exclusivity_conflict(tokens: List[str]) -> int:
    if not tokens:
        return 0
    markers = [t for t in tokens if t in EXCLUSIVE_MARKERS]
    if not markers:
        return 0
    substantive_present = any(t not in EXCLUSIVE_MARKERS for t in tokens)
    if substantive_present:
        return 1
    unique_markers = set(markers)
    return 1 if len(unique_markers) > 1 else 0


def _expand_canonical_sequence(qkey: str, token: str) -> List[str]:
    canonicals = CANONICAL_MULTI_OPTIONS.get(qkey)
    if not canonicals:
        return [token]

    remaining = token
    results: List[str] = []
    ordered = sorted(canonicals, key=len, reverse=True)

    while remaining:
        matched = False
        for cand in ordered:
            if remaining == cand:
                results.append(cand)
                remaining = ""
                matched = True
                break
            if remaining.startswith(f"{cand}_"):
                results.append(cand)
                remaining = remaining[len(cand) + 1 :]
                matched = True
                break
        if not matched:
            return [token]

    deduped: List[str] = []
    for cand in results:
        if cand not in deduped:
            deduped.append(cand)
    return deduped or [token]


def normalize_multiselect_tokens(qkey: str, token: str) -> Tuple[List[str], bool]:
    """Normalise a raw multi-select option into canonical tokens."""

    cleaned, had_schema = strip_schema_echo(token)
    cleaned = (cleaned or "").strip()
    if not cleaned:
        return [], had_schema

    if ":" in cleaned:
        prefix, suffix = cleaned.rsplit(":", 1)
        if any(ch.isspace() for ch in prefix):
            cleaned = suffix.strip()

    cleaned = cleaned.replace("-", "_")
    cleaned = cleaned.replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    cleaned = cleaned.upper()

    if not cleaned:
        return [], had_schema

    return _expand_canonical_sequence(qkey, cleaned), had_schema
