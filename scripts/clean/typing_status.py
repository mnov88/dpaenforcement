from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Dict, Optional, Tuple


DATE_FMT = "%Y-%m-%d"


@dataclass
class DateParseResult:
    value: Optional[datetime]
    status: str  # DISCUSSED | NOT_DISCUSSED


ISO_PREFIX = "ISO_3166-1_ALPHA-2:"


def parse_date_field(raw: str) -> DateParseResult:
    raw = (raw or "").strip()
    if raw == "NOT_DISCUSSED":
        return DateParseResult(value=None, status="NOT_DISCUSSED")
    try:
        dt = datetime.strptime(raw, DATE_FMT).replace(tzinfo=UTC)
        return DateParseResult(value=dt, status="DISCUSSED")
    except Exception:
        return DateParseResult(value=None, status="NOT_DISCUSSED")


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


def parse_number(raw: str) -> Tuple[Optional[float], bool, str]:
    # Returns (value, valid, status)
    raw = (raw or "").strip()
    if raw == "":
        return None, True, "NOT_MENTIONED"
    s = _sanitize_numeric_string(raw)
    try:
        val = float(s)
        if val < 0:
            return None, False, "DISCUSSED"
        return val, True, "DISCUSSED"
    except Exception:
        return None, False, "DISCUSSED"


def split_multiselect(raw: str) -> Tuple[list[str], str]:
    if not raw:
        return [], "NOT_MENTIONED"
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    return tokens, "DISCUSSED" if tokens else "NOT_MENTIONED"
