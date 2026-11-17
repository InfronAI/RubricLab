import re


judge_generation_template_pairwise = (
    "You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' "
    "based on a given instruction and a rubric. You will conduct this evaluation in distinct "
    "phases as outlined below.\n\n"
    "### Phase 1: Compliance Check Instructions\n"
    "First, identify the single most important, objective 'Gatekeeper Criterion' from the rubric.\n"
    "- **A rule is objective (and likely a Gatekeeper) if it can be verified without opinion. "
    "Key examples are: word/paragraph limits, required output format (e.g., JSON validity), "
    "required/forbidden sections, or forbidden content.**\n"
    "- **Conversely, a rule is subjective if it requires interpretation or qualitative judgment. "
    "Subjective rules about quality are NOT Gatekeepers. Examples include criteria like \"be creative,\" "
    "\"write clearly,\" \"be engaging,\" or \"use a professional tone.\"**\n"
    "Think step-by-step to determine this single most important Gatekeeper, then write a 1–2 sentence explanation of your decision.\n\n"
    "### Phase 2: Analyze Each Response\n"
    "Next, for each Gatekeeper Criterion and all other criteria in the rubric, evaluate each "
    "response item by item.\n"
    "For each item, think step-by-step and cite concrete evidence from the response before assigning your judgment.\n\n"
    "### Phase 3: Final Judgment Instructions\n"
    "Based on the results from the previous phases, determine the winner using these simple rules. "
    "Provide a final justification explaining your decision first and then give your decision.\n"
    "Think step-by-step to aggregate the findings and make the decision; keep the reasoning explicit and concise.\n\n"
    "---\n"
    "### REQUIRED OUTPUT FORMAT\n"
    "You must follow this exact output format below.\n\n"
    "--- Compliance Check ---\n"
    "Gatekeeper Reasoning: <1–2 sentences citing the relevant rubric text>\n"
    "Identified Gatekeeper Criterion: <e.g., Criterion 1: Must be under 50 words.>\n\n"
    "--- Analysis ---\n"
    "**Response A:**\n"
    "- Criterion 1 [Hard Rule]: Justification: <...>\n"
    "- Criterion 2 [Hard Rule]: Justification: <...>\n"
    "- Criterion 3 [Principle]: Justification: <...>\n"
    "- ... (and so on for all other criteria)\n\n"
    "**Response B:**\n"
    "- Criterion 1 [Hard Rule]: Justification: <...>\n"
    "- Criterion 2 [Hard Rule]: Justification: <...>\n"
    "- Criterion 3 [Principle]: Justification: <...>\n"
    "- ... (and so on for all other criteria)\n\n"
    "--- Final Judgment ---\n"
    "Aggregation Summary: <1–3 sentences explaining how Gatekeeper and other criteria led to the decision>\n"
    "Justification: <...>\n"
    "Winner: <Response A / Response B>\n\n\n"
    "Task to Evaluate:\n"
    "Instruction:\n{instruction}\n\n"
    "Rubric:\n{rubric}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}"
).strip()


def clean_judge_format(rubric_text: str) -> str:
    """Remove possible <output> tags and extra whitespace."""
    if not rubric_text:
        return rubric_text
    cleaned = re.sub(r'<output>\s*', '', rubric_text)
    cleaned = re.sub(r'\s*</output>', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def make_judge_prompt_pairwise(instruction: str, rubric: str, responses: list[str]) -> str:
    resa, resb = responses
    return judge_generation_template_pairwise.format(
        instruction=instruction or "",
        rubric=rubric or "",
        response_a=resa or "",
        response_b=resb or ""
    )
