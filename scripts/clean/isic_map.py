from __future__ import annotations

import csv
import hashlib
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TOKEN_SPLIT_RE = re.compile(r"(?:,|;|\n|\bor\b|\band\b|\||/)+", re.IGNORECASE)
PREFIX_RE = re.compile(r"^(SECTION|SEC\.?|DIVISION|DIV\.?|GROUP|CLASS)\s+", re.IGNORECASE)
PAREN_RE = re.compile(r"\([^)]*\)")


@dataclass
class IsicEntry:
    code: str
    description: str
    section: Optional[str]  # Letter A..U
    section_description: Optional[str]
    division: Optional[str]
    division_description: Optional[str]
    group: Optional[str]
    group_description: Optional[str]


class IsicIndex:
    def __init__(self) -> None:
        self.code_to_entry: Dict[str, IsicEntry] = {}
        self.section_descriptions: Dict[str, str] = {}
        self.division_descriptions: Dict[str, str] = {}
        self.group_descriptions: Dict[str, str] = {}
        self.reference_path: Optional[Path] = None
        self.reference_version: Optional[str] = None

    @staticmethod
    def load_from_file(path: Path) -> "IsicIndex":
        idx = IsicIndex()
        idx.reference_path = path
        try:
            idx.reference_version = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
        except FileNotFoundError:
            idx.reference_version = None
        current_section: Optional[str] = None
        current_section_desc: Optional[str] = None
        current_division: Optional[str] = None
        current_division_desc: Optional[str] = None
        current_group: Optional[str] = None
        current_group_desc: Optional[str] = None
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if not row or len(row) < 2:
                    continue
                code = row[0].strip().strip('"')
                desc = row[1].strip().strip('"')
                if not code:
                    continue
                # A section row is a single uppercase letter
                if len(code) == 1 and code.isalpha() and code.isupper():
                    current_section = code
                    current_section_desc = desc
                    current_division = None
                    current_division_desc = None
                    current_group = None
                    current_group_desc = None
                    idx.section_descriptions[code] = desc
                    idx.code_to_entry[code] = IsicEntry(
                        code=code,
                        description=desc,
                        section=code,
                        section_description=desc,
                        division=None,
                        division_description=None,
                        group=None,
                        group_description=None,
                    )
                    continue
                # Regular numeric code (e.g., 62, 6209, 0111, etc.)
                if code.isdigit():
                    if len(code) == 2:
                        current_division = code
                        current_division_desc = desc
                        current_group = None
                        current_group_desc = None
                        idx.division_descriptions[code] = desc
                    elif len(code) == 3:
                        current_group = code
                        current_group_desc = desc
                        idx.group_descriptions[code] = desc
                    entry = IsicEntry(
                        code=code,
                        description=desc,
                        section=current_section,
                        section_description=current_section_desc,
                        division=code if len(code) == 2 else current_division,
                        division_description=current_division_desc,
                        group=code if len(code) == 3 else current_group,
                        group_description=current_group_desc,
                    )
                else:
                    entry = IsicEntry(
                        code=code,
                        description=desc,
                        section=current_section,
                        section_description=current_section_desc,
                        division=current_division,
                        division_description=current_division_desc,
                        group=current_group,
                        group_description=current_group_desc,
                    )
                idx.code_to_entry[code] = entry
        return idx

    def lookup(self, raw_code: str) -> Tuple[Optional[IsicEntry], bool]:
        if not raw_code:
            return None, False
        raw_code = raw_code.strip().strip('"')
        # Prefer exact match
        entry = self.code_to_entry.get(raw_code)
        if entry:
            return entry, True
        # Try normalization: if 4-digit class not found, try 2-digit division
        if len(raw_code) >= 2 and raw_code[:2] in self.code_to_entry:
            return self.code_to_entry.get(raw_code[:2]), True
        return None, False

    def parse_codes(self, raw_value: str) -> Tuple[List[IsicEntry], List[str]]:
        candidates = parse_isic_codes(raw_value)
        entries: List[IsicEntry] = []
        invalid: List[str] = []
        for candidate in candidates:
            entry, ok = self.lookup(candidate)
            if ok and entry is not None:
                entries.append(entry)
            else:
                invalid.append(candidate)
        return entries, invalid


def parse_isic_codes(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    tokens = TOKEN_SPLIT_RE.split(raw_value)
    candidates: List[str] = []
    for token in tokens:
        candidates.extend(_extract_candidates_from_token(token))
    seen: set[str] = set()
    ordered: List[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _extract_candidates_from_token(token: str) -> List[str]:
    token = token.strip()
    if not token:
        return []
    # Remove schema echoes and descriptive wrappers
    token = PAREN_RE.sub(" ", token)
    token = token.replace("–", "-").replace("—", "-")
    token = PREFIX_RE.sub("", token)
    token = token.strip()
    if not token:
        return []

    # Cut off after the first colon or dash to isolate code-like prefix
    for delimiter in (":", "-", "—", "–"):
        if delimiter in token:
            token = token.split(delimiter, 1)[0].strip()
    token = token.strip()
    if not token:
        return []

    upper = token.upper().strip()
    compressed = upper.replace(" ", "")
    compressed = compressed.replace(".", "")
    compressed = compressed.replace(" ", "")  # non-breaking space

    matches: List[str] = []
    if len(upper) == 1 and upper in string.ascii_uppercase:
        matches.append(upper)
        return matches

    combo = re.match(r"^([A-U])\s*(\d{1,4})$", upper)
    if combo:
        matches.append(combo.group(2))
        return matches

    combo_no_space = re.match(r"^([A-U])(\d{1,4})$", compressed)
    if combo_no_space:
        matches.append(combo_no_space.group(2))
        return matches

    digits_only = "".join(ch for ch in token if ch.isdigit())
    if digits_only:
        digits_only = digits_only[:4]
        if 1 <= len(digits_only) <= 4:
            matches.append(digits_only)

    if matches:
        return matches

    # As a fallback, if the token looks like a solitary letter code at the start
    head = compressed[:1]
    if head and head in string.ascii_uppercase and len(compressed) == 1:
        return [head]

    return []
