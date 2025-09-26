from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


class EnumWhitelist:
    def __init__(self, questions: Dict[str, dict]):
        self.questions = questions or {}

    @staticmethod
    def load(path: Path) -> "EnumWhitelist":
        data = json.loads(path.read_text(encoding="utf-8"))
        return EnumWhitelist(data.get("questions", {}))

    def allowed_tokens(self, qnum: str) -> List[str]:
        spec = self.questions.get(qnum)
        if not spec:
            return []
        if spec.get("kind") in ("ENUM", "MULTI_SELECT"):
            return list(spec.get("options", []))
        return []

    def validate_tokens(self, qnum: str, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """Return (unknown_tokens, known_tokens) given a question number like '30' or 'Q30'."""
        qkey = qnum[1:] if qnum.startswith("Q") else qnum
        allowed = set(self.allowed_tokens(qkey))
        if not allowed:
            return ([], tokens)
        unknown = [t for t in tokens if t not in allowed]
        known = [t for t in tokens if t in allowed]
        return unknown, known
