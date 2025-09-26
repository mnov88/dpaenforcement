import re
from typing import Dict, List

ANSWER_LINE_RE = re.compile(r'^"?Answer\s+(\d+):\s*(.+?)\s*$', re.MULTILINE)
BLOCK_SPLIT_RE = re.compile(r'(?=^"?Answer\s+1:\s*)', re.MULTILINE)


def segment_records(text: str) -> List[str]:
    """Split concatenated blocks at each new Answer 1: line.

    Trims leading/trailing whitespace and ignores empty segments.
    """
    if not text:
        return []
    segments = [s.strip() for s in BLOCK_SPLIT_RE.split(text) if s.strip()]
    # Keep only segments that begin with Answer 1:
    segments = [s for s in segments if s.startswith("Answer 1:") or s.startswith('"Answer 1:')]
    return segments


def parse_record(record_text: str) -> Dict[str, object]:
    """Parse a single record's Answer N: value lines into dict.

    Returns a dict with keys: answers (dict) and metadata (dict).
    """
    answers: Dict[str, str] = {}
    warnings: List[str] = []

    for m in ANSWER_LINE_RE.finditer(record_text):
        num = m.group(1)
        val = m.group(2).strip().strip('\u200b')
        answers[f"Q{num}"] = val

    completeness = len(answers) == 68
    if not completeness:
        missing = [f"Q{i}" for i in range(1, 69) if f"Q{i}" not in answers]
        warnings.append(
            f"incomplete_record: missing={','.join(missing[:10])}{'...' if len(missing)>10 else ''}"
        )

    metadata = {
        "line_count": len(record_text.splitlines()),
        "completeness": completeness,
        "warnings": warnings,
    }

    return {"answers": answers, "metadata": metadata}
