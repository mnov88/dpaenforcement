from typing import List


def normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def dedupe_preserve_order(tokens: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
