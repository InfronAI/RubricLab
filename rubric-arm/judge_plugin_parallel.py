"""Reward plugin that trusts the policy model as the judge.

It parses the policy's completion, extracts the declared winner, and compares
it with the dataset label. Matching predictions get +1 reward, mismatches get
-1. The format reward intentionally mirrors the original training code.
"""

from __future__ import annotations

from typing import List, Sequence
import re

from swift.plugin import ORM, orms


def parse_winner_from_response(judge_response: str) -> str:
    for line in judge_response.split("\n"):
        if line.strip().startswith("Winner:"):
            return line.split("Winner:")[1].strip()
    return "Unknown"


def _normalize_label(label: str) -> str:
    label = (label or "").strip().lower()
    if "response" in label:
        label = label.replace("response", "")
    label = label.replace(" ", "")
    if label == "a":
        return "a"
    if label == "b":
        return "b"
    return "unknown"


def extract_criteria_from_rubric(rubric_text: str) -> List[str]:
    """
    Extract criterion names from rubric text, excluding the trailing colon.
    Expected format:
      Criterion 1 [Hard Rule]: ...
      Criterion 2 [Hard Rule]: ...
      Criterion 3 [Principle]: ...
    """
    pattern = r"^(Criterion\s+\d+.*?):"
    named = re.findall(pattern, rubric_text, flags=re.M)
    if named:
        return named
    # Released OpenRubric-v2 rubrics use numbered sentences with trailing tags.
    numbered = re.findall(r"^\s*(\d+)\.\s+.*?\[(Hard Rule|Principle)\]", rubric_text, re.M)
    return [f"Criterion {number} [{kind}]" for number, kind in numbered]


class PairwiseJudgeReward(ORM):
    """Parse the policy completion directly and return +1/-1 rewards."""

    def __call__(self,
                 completions: Sequence[str],
                 winner: Sequence[str],
                 **_: object) -> List[float]:
        rewards: List[float] = []
        for completion, gold_winner in zip(completions, winner):
            predicted = parse_winner_from_response(completion)
            pred_label = _normalize_label(predicted)
            gold_label = _normalize_label(gold_winner)
            reward = 1.0 if pred_label != "unknown" and pred_label == gold_label else -1.0
            rewards.append(reward)
        return rewards


orms["pairwise_judge_reward"] = PairwiseJudgeReward


class FormatReward(ORM):
    """Parse the policy completion directly and return format rewards."""

    def __call__(self,
                 completions: Sequence[str],
                 rubric: Sequence[str],
                 **_: object) -> List[float]:
        rewards: List[float] = []
        for completion, rub in zip(completions, rubric):
            criteria = extract_criteria_from_rubric(rub)
            error = 0.0
            if not criteria:
                error += 1.0
            pattern = (
                r"--- Compliance Check ---\s*(.*?)"
                r"--- Analysis ---\s*(.*?)"
                r"--- Final Judgment ---\s*(.*)$"
            )
            m = re.search(pattern, completion, re.S)
            if not m:
                error += 1.0
            else:
                compliance, analysis, final = m.groups()
                if not re.search(r"Gatekeeper Reasoning:\s*\S+", compliance):
                    error += 1.0
                if not re.search(r"Identified Gatekeeper Criterion:\s*\S+", compliance):
                    error += 1.0
                ra_match = re.search(r"\*\*Response A:\*\*\s*(.*?)(?=\*\*Response B:\*\*)", analysis, re.S)
                rb_match = re.search(r"\*\*Response B:\*\*\s*(.*)$", analysis, re.S)
                if not ra_match:
                    error += 1.0
                if not rb_match:
                    error += 1.0

                def check_response_block(block_text: str, label: str) -> List[str]:
                    sub_errors = []
                    for crit in criteria:
                        crit_pattern = (
                            r"-\s*"
                            + re.escape(crit)
                            + r":\s*Justification:\s*\S+"
                        )
                        if not re.search(crit_pattern, block_text):
                            sub_errors.append(
                                f"Missing criterion '{crit}' under {label}."
                            )
                    return sub_errors

                if ra_match:
                    resp_a = ra_match.group(1)
                    if not re.search(r"-\s*Criterion .*?:\s*Justification:\s*\S+", resp_a):
                        error += 1.0
                    else:
                        sub_errs = check_response_block(resp_a, "Response A")
                        if sub_errs:
                            error += 1.0
                if rb_match:
                    resp_b = rb_match.group(1)
                    if not re.search(r"-\s*Criterion .*?:\s*Justification:\s*\S+", resp_b):
                        error += 1.0
                    else:
                        sub_errs = check_response_block(resp_b, "Response B")
                        if sub_errs:
                            error += 1.0
                if not re.search(r"Aggregation Summary:\s*\S+", final):
                    error += 1.0
                if not re.search(r"Justification:\s*\S+", final):
                    error += 1.0
                if not re.search(r"Winner:\s*(Response A|Response B)\b", final):
                    error += 1.0

            reward = (1 - error / 10.0) * 0.5
            rewards.append(reward)
        return rewards


orms["pairwise_judge_format_reward"] = FormatReward
