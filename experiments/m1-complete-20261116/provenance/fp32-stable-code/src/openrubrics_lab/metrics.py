"""Strict decisions, abstentions, position bias, and denominator-safe metrics."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict


def normalize_label(value: str) -> str | None:
    value = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return {"a": "A", "b": "B", "response a": "A", "response b": "B",
            "assistant 1": "A", "assistant 2": "B"}.get(value)


def parse_winner(text: str) -> str | None:
    # Ignore quoted examples, ambiguous/repeated decisions and reasoning prefixes.
    text = unicodedata.normalize("NFKC", text).replace("**", "").strip()
    matches = re.findall(r"^\s*Winner:\s*(?:Response\s+)?([AB])\s*[.!]?\s*$", text, re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1:
        return None
    return matches[0].upper()


def majority(winners: list[str | None]) -> str | None:
    """Invalid generations count in the denominator; require a strict majority."""
    counts = Counter(winners)
    for label in ("A", "B"):
        if counts[label] > len(winners) / 2:
            return label
    return None


def wilson(correct: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = correct / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0.0, center-half), min(1.0, center+half)]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "accuracy": None, "majority_accuracy": None, "invalid_rate": None}
    n = len(rows)
    winners = [[parse_winner(t) for t in r["predictions"]] for r in rows]
    votes = [majority(w) for w in winners]
    first = sum(bool(w) and w[0] == r["label"] for w, r in zip(winners, rows))
    correct = sum(v == r["label"] for v, r in zip(votes, rows))
    total = sum(map(len, winners))
    valid = sum(x is not None for w in winners for x in w)
    groups = defaultdict(list)
    for r, v in zip(rows, votes):
        if r.get("group_id") is not None:
            groups[r["group_id"]].append(v == r["label"])
    return {
        "n": n, "generations": total, "accuracy": first/n, "majority_accuracy": correct/n,
        "majority_wilson95": wilson(correct, n),
        "invalid_rate": 1-valid/total if total else None,
        "abstention_rate": sum(v is None for v in votes)/n,
        "choose_a_rate_valid": sum(x == "A" for w in winners for x in w)/valid if valid else None,
        "group_all_correct": sum(all(v) for v in groups.values())/len(groups) if groups else None,
        "groups": len(groups),
    }


def format_reward(completion: str, rubric: str) -> float:
    """0..0.5 reward; accepts both numbered and Criterion-style source rubrics."""
    criteria = re.findall(r"(?:^|\n)\s*(?:Criterion\s+)?(\d+)[.:\s].*?\[(Hard Rule|Principle)\]", rubric)
    checks = [bool(criteria), parse_winner(completion) is not None]
    for section in ("--- Compliance Check ---", "--- Analysis ---", "--- Final Judgment ---",
                    "Gatekeeper Reasoning:", "Identified Gatekeeper Criterion:",
                    "Aggregation Summary:", "Justification:"):
        checks.append(section in completion)
    for label in ("A", "B"):
        block = re.search(rf"\*\*Response {label}:\*\*(.*?)(?=\*\*Response|--- Final Judgment|\Z)",
                          completion, re.DOTALL)
        checks.append(bool(block) and bool(criteria) and all(
            re.search(rf"Criterion\s+{idx}\s*\[{kind}\]:\s*Justification:\s*\S", block[1])
            for idx, kind in criteria))
    return 0.5 * sum(checks) / len(checks)
