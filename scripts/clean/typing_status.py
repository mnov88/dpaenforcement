from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, List, Optional, Tuple


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


@dataclass
class EnumParseResult:
    raw: str
    value: str
    status: str
    note: str = ""


_ENUM_STATUS_DIRECT = {
    "NOT_MENTIONED",
    "NOT_DISCUSSED",
    "NOT_APPLICABLE",
}

_ENUM_TOKEN_RE = re.compile(r"[A-Z0-9_]+")


def _normalize_enum_token(token: str) -> str:
    t = (token or "").strip().upper()
    if not t:
        return ""
    t = t.replace("-", "_").replace("/", "_")
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"__+", "_", t)
    return t.strip("_")


def _enum_status_from_value(value: str) -> str:
    if not value:
        return "NOT_MENTIONED"
    if value in _ENUM_STATUS_DIRECT:
        return value
    if value == "UNCLEAR":
        return "UNCLEAR"
    return "DISCUSSED"


def parse_enum_field(
    raw: str,
    allowed_tokens: List[str],
    aliases: Optional[Dict[str, str]] = None,
) -> EnumParseResult:
    raw_norm = (raw or "").strip()
    if not raw_norm:
        return EnumParseResult(raw="", value="", status="NOT_MENTIONED", note="empty")

    allowed_lookup = {token: token for token in allowed_tokens}
    allowed_norm = {_normalize_enum_token(token): token for token in allowed_tokens}

    alias_norm: Dict[str, str] = {}
    if aliases:
        for key, target in aliases.items():
            normalized_key = _normalize_enum_token(key)
            if normalized_key:
                alias_norm[normalized_key] = target

    direct_norm = _normalize_enum_token(raw_norm)
    if direct_norm in allowed_norm:
        value = allowed_norm[direct_norm]
        return EnumParseResult(
            raw=raw_norm,
            value=value,
            status=_enum_status_from_value(value),
            note="direct",
        )
    if direct_norm in alias_norm:
        value = alias_norm[direct_norm]
        return EnumParseResult(
            raw=raw_norm,
            value=value,
            status=_enum_status_from_value(value),
            note="alias_direct",
        )

    if ":" in raw_norm:
        trailing = raw_norm.split(":")[-1].strip()
        trailing_norm = _normalize_enum_token(trailing)
        if trailing_norm in allowed_norm:
            value = allowed_norm[trailing_norm]
            return EnumParseResult(
                raw=raw_norm,
                value=value,
                status=_enum_status_from_value(value),
                note="trailing_segment",
            )
        if trailing_norm in alias_norm:
            value = alias_norm[trailing_norm]
            return EnumParseResult(
                raw=raw_norm,
                value=value,
                status=_enum_status_from_value(value),
                note="trailing_alias",
            )

    # Token sweep across uppercase words/underscores
    tokens = [tok for tok in _ENUM_TOKEN_RE.findall(raw_norm.upper()) if tok in allowed_lookup]
    if tokens:
        value = allowed_lookup[tokens[-1]]
        token_set = set(tokens)
        allowed_set = set(allowed_lookup)
        if allowed_set and token_set.issuperset(allowed_set):
            return EnumParseResult(
                raw=raw_norm,
                value=value,
                status=_enum_status_from_value(value),
                note="menu_echo",
            )
        unique_tokens = list(dict.fromkeys(tokens))
        if len(unique_tokens) > 1:
            return EnumParseResult(
                raw=raw_norm,
                value=value,
                status="MIXED_CONTRADICTORY",
                note="multiple_tokens",
            )
        return EnumParseResult(
            raw=raw_norm,
            value=value,
            status=_enum_status_from_value(value),
            note="token_match",
        )

    alias_tokens = [tok for tok in _ENUM_TOKEN_RE.findall(raw_norm.upper()) if tok in alias_norm]
    if alias_tokens:
        value = alias_norm[alias_tokens[-1]]
        unique_alias = list(dict.fromkeys(alias_tokens))
        if len(unique_alias) > 1:
            return EnumParseResult(
                raw=raw_norm,
                value=value,
                status="MIXED_CONTRADICTORY",
                note="multiple_alias_tokens",
            )
        return EnumParseResult(
            raw=raw_norm,
            value=value,
            status=_enum_status_from_value(value),
            note="alias_token",
        )

    if allowed_tokens:
        return EnumParseResult(raw=raw_norm, value="", status="UNPARSEABLE", note="unmatched")
    return EnumParseResult(raw=raw_norm, value="", status="UNKNOWN", note="no_allowed_tokens")


EXCLUSIVE_MARKERS = {
    "NOT_APPLICABLE",
    "NONE_MENTIONED",
    "NONE_DISCUSSED",
    "NOT_DISCUSSED",
    "NONE",
    "NOT_DETERMINED",
    "NONE_VIOLATED",
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
    if tset == {"NOT_DISCUSSED"}:
        return "NOT_DISCUSSED"
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
