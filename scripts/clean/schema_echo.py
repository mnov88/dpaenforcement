"""Utilities for handling schema echo artefacts in raw AI answers."""

from __future__ import annotations

from typing import Tuple


SCHEMA_ECHO_PREFIXES = ("TYPE:", "ENUM:", "MULTI_SELECT:", "FORMAT:")


def is_schema_echo(value: str) -> bool:
    """Return True when ``value`` consists solely of a schema echo prefix."""

    v = (value or "").strip()
    return any(v.startswith(prefix) for prefix in SCHEMA_ECHO_PREFIXES)


def strip_schema_echo(value: str) -> Tuple[str, bool]:
    """Remove a leading schema echo prefix if present.

    Parameters
    ----------
    value:
        Raw answer token or field exported from the AI response.

    Returns
    -------
    tuple[str, bool]
        A tuple containing the cleaned value (with the prefix removed and
        surrounding whitespace stripped) and a boolean indicating whether any
        prefix was removed.
    """

    had_schema = False

    def clean_token(token: str) -> str:
        nonlocal had_schema
        t = token.strip()
        for prefix in SCHEMA_ECHO_PREFIXES:
            if t.startswith(prefix):
                had_schema = True
                return t[len(prefix) :].lstrip()
        return t

    v = (value or "").strip()
    if not v:
        return "", False

    if "," in v:
        cleaned_tokens = [clean_token(part) for part in v.split(",")]
        if had_schema:
            filtered = [token for token in cleaned_tokens if token]
            return ", ".join(filtered), True
        return v, False

    cleaned = clean_token(v)
    return cleaned, had_schema

