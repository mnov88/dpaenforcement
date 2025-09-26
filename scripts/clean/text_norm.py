from __future__ import annotations

import re
import unicodedata
from typing import Dict, Tuple

_ZW_CHARS = [
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\ufeff",  # BOM
]
_ZW_RE = re.compile("|".join(map(re.escape, _ZW_CHARS)))
_WS_RE = re.compile(r"\s+")


def normalize_text(raw: str) -> Tuple[str, int, int]:
    """Normalize to NFC, strip zero-widths, collapse whitespace.

    Returns (normalized_text, token_count, char_count).
    """
    text = raw or ""
    text = unicodedata.normalize("NFC", text)
    text = _ZW_RE.sub("", text)
    text = text.strip()
    text = _WS_RE.sub(" ", text)
    tokens = [t for t in text.split(" ") if t]
    return text, len(tokens), len(text)


def detect_language_heuristic(text: str) -> str:
    # Very light heuristic: if mostly ASCII -> ENGLISH_LIKELY else UNKNOWN
    if not text:
        return "UNKNOWN"
    ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
    return "ENGLISH_LIKELY" if ascii_ratio > 0.95 else "UNKNOWN"
