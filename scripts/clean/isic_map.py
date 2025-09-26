from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class IsicEntry:
    code: str
    description: str
    section: Optional[str]  # Letter A..U


class IsicIndex:
    def __init__(self) -> None:
        self.code_to_entry: Dict[str, IsicEntry] = {}

    @staticmethod
    def load_from_file(path: Path) -> "IsicIndex":
        idx = IsicIndex()
        current_section: Optional[str] = None
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
                    # Store section as an entry too
                    idx.code_to_entry[code] = IsicEntry(code=code, description=desc, section=code)
                    continue
                # Regular numeric code (e.g., 62, 6209, 0111, etc.)
                idx.code_to_entry[code] = IsicEntry(code=code, description=desc, section=current_section)
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
