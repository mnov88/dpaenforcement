from __future__ import annotations

from typing import Optional, Tuple

# Minimal EU/EEA membership maps (can extend later)
EU_ALPHA2 = {
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR","HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"
}
EEA_NON_EU = {"IS", "LI", "NO"}


def enrich_country(code: Optional[str]) -> Tuple[str, str]:
    code = (code or "").upper()
    if not code:
        return "", "UNKNOWN"
    if code in EU_ALPHA2:
        return code, "EU"
    if code in EEA_NON_EU:
        return code, "EEA_NON_EU"
    if code == "EU":
        return code, "EU_INSTITUTION"
    return code, "NON_EEA"


def normalize_dpa_name(raw: str) -> str:
    # Placeholder: return raw for now; later map to canonical registry
    return (raw or "").strip()
