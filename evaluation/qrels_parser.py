"""
Parsers for NTCIR CSAT topic and qrels files.

Topic file format (TSV):  qid  <TAB>  query_text
Qrels file format (TREC): qid  0  segment_id  relevance_grade
"""

from pathlib import Path


def load_topics(path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                topics[parts[0].strip()] = parts[1].strip()
    return topics


def load_qrels(path: Path, min_grade: int = 1) -> dict[str, set[str]]:
    """Load TREC-format qrels. Only segments with grade >= min_grade are relevant."""
    qrels: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, grade = parts[0], parts[1], parts[2], int(parts[3])
            if grade >= min_grade:
                qrels.setdefault(qid, set()).add(doc_id)
    return qrels
