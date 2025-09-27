import re
from typing import Dict, Optional

ANSWER_SPEC_RE = re.compile(r'^\*\*Answer\s+(\d+)\:\*\*\s*(.+)$')


def _parse_enum_or_multi(value: str) -> Optional[Dict[str, object]]:
    value = value.strip()
    if value.startswith("ENUM:"):
        options = [t.strip() for t in value[len("ENUM:"):].split(',') if t.strip()]
        return {"kind": "ENUM", "options": options}
    if value.startswith("MULTI_SELECT:"):
        options = [t.strip() for t in value[len("MULTI_SELECT:"):].split(',') if t.strip()]
        return {"kind": "MULTI_SELECT", "options": options}
    if value.startswith("TYPE:"):
        t = value[len("TYPE:"):].strip()
        return {"kind": "TYPE", "type": t}
    if value.startswith("FORMAT:"):
        return {"kind": "FORMAT", "format": value[len("FORMAT:"):].strip()}
    return None


def build_enum_whitelist(prompt_text: str, *, source_path: Optional[str] = None) -> Dict[str, object]:
    """Extract per-question specifications from the questionnaire prompt.

    Returns a dict with a `questions` map and a `source` metadata.
    """
    lines = [ln.strip() for ln in prompt_text.splitlines() if ln.strip()]
    questions: Dict[str, Dict[str, object]] = {}

    for i in range(len(lines) - 1):
        m = ANSWER_SPEC_RE.match(lines[i])
        if not m:
            continue
        qnum = m.group(1)
        remainder = m.group(2)
        parsed = _parse_enum_or_multi(remainder)
        if parsed is None:
            nxt = lines[i + 1]
            parsed = _parse_enum_or_multi(nxt) or {"kind": "RAW", "raw": remainder}
        questions[qnum] = parsed

    return {
        "questions": questions,
        "source": {
            "extracted_from": source_path or "data-extraction-prompt-sent-to-ai.md",
            "parser_version": "0.1.0",
        },
    }
