import re


rubrics_generation_template_listwise = (
    "You are an expert in pedagogy and critical thinking. Your mission is to create a universal scoring rubric based on a user's request and an ordered list of example responses. "
    "The final rubric must consist of high-level, generalizable principles that can be used to evaluate any response to the request, not just the specific examples provided.\n\n"
    "====================================================================\n"
    "Methodology – A Three-Step Process for Principled Rubric Design\n"
    "====================================================================\n\n"
    "1. Step 1: Extract Explicit Requirements.  \n"
    "   - Meticulously analyze the <request> tag to identify all direct commands and constraints (e.g., length, format, style).  \n"
    "   - These requirements are *non-negotiable hard rules* that must appear in the rubric.  \n"
    "   - They should be clearly labeled as [Hard Rule] in the final output.  \n\n"
    "2. Step 2: Analyze the Ordered Examples for Specific Differences.  \n"
    "   - Study the <responses> list, which is sorted in *descending preference* (earlier responses are BETTER).  \n"
    "   - Identify concrete qualities that make higher-ranked responses superior to lower-ranked ones. Consider both adjacent comparisons (i vs. i+1) and contrasts between the top responses and the rest.  \n"
    "   - It is acceptable at this stage to note topic-specific observations (e.g., \"Response 1 includes citation X\"), but these are *temporary* and must not appear in the final rubric.  \n"
    "   - Every such observation must then be abstracted in Step 3.  \n\n"
    "3. Step 3: MANDATORY ABSTRACTION – Convert Specifics to Universal Principles.  \n"
    "   - This is the most critical step. For each observation from Step 2, ask:  \n"
    "     **\"What is the universal principle of high-quality communication, reasoning, or pedagogy that this specific difference demonstrates?\"**  \n"
    "   - Convert each observation into a principle that applies across any domain, not just the provided examples.  \n"
    "   - Any rubric item that references concrete facts, names, events, topics, or response indices (e.g., \"Response #1\") is INVALID.  \n"
    "   - All such principles must be labeled as [Principle] in the final output.  \n\n"
    "====================================================================\n"
    "Strict Guidelines for Final Output\n"
    "====================================================================\n\n"
    "- **Abstraction is Mandatory:**  \n"
    "  Every rubric item must be a universal principle. If any rubric still contains topic-specific references (e.g., names, places, myths, numbers, historical facts), or mentions response indices/positions, it is automatically invalid.\n\n"
    "- **Two Distinct Categories:**  \n"
    "  - [Hard Rule]: Derived strictly from explicit requirements in the <request>.  \n"
    "  - [Principle]: Derived from abstracted differences in Step 3.  \n\n"
    "- **Comprehensiveness:**  \n"
    "  The rubric must cover all critical aspects implied by the request and examples, including explicit requirements and implicit quality standards.\n\n"
    "- **Conciseness & Uniqueness:**  \n"
    "  Each rubric must capture a distinct evaluation criterion. Overlapping or redundant criteria must be merged into a single rubric. Wording must be precise and free of repetition.\n\n"
    "- **Format Requirements:**  \n"
    "  - Use a numbered list.  \n"
    "  - Each item starts with \"The response…\" phrased in third person.  \n"
    "  - Append [Hard Rule] or [Principle] at the end of each item.  \n"
    "  - Do not include reasoning, explanations, or examples in the final output—only the rubrics.\n\n"
    "- **Validation Check Before Output:**  \n"
    "  Before presenting the final list, verify:  \n"
    "  1. Does every rubric meet the abstraction requirement (no topic-specific details, no reference to response indices)?  \n"
    "  2. Are all hard rules from Step 1 included?  \n"
    "  3. Are all principles unique and non-overlapping?  \n"
    "  4. Is the list written entirely in third person, concise, and consistent?\n\n"
    "====================================================================\n"
    "Final Output Format\n"
    "====================================================================\n\n"
    "1. The response … [Hard Rule]  \n"
    "2. The response … [Principle]  \n"
    "3. The response … [Principle]  \n"
    "... (continue until all rules and principles are listed)\n\n"
    "====================================================================\n\n"
    "<request>\n"
    "{request}\n"
    "</request>\n\n"
    "<context>\n"
    "{context}\n"
    "</context>\n\n"
    "<responses>\n"
    "{responses}\n"
    "</responses>\n"
).strip()


def clean_rubric_format(rubric_text: str) -> str:
    """Remove possible <output> tags and extra whitespace."""
    if not rubric_text:
        return rubric_text
    cleaned = re.sub(r'<output>\s*', '', rubric_text)
    cleaned = re.sub(r'\s*</output>', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _escape_braces(s: str) -> str:
    """Escape braces before str.format to avoid KeyError from { } in text."""
    if not s:
        return ""
    return s.replace("{", "{{").replace("}", "}}")


def make_rubric_prompt_listwise(request: str, responses: list[str],  context: str = "") -> str:
    combined_responses = make_response_list(responses)
    return rubrics_generation_template_listwise.format(
        request=_escape_braces(request or ""),
        context=_escape_braces(context or ""),
        responses=_escape_braces(combined_responses or ""),
    )

def make_response_list(responses: list[str]) -> str:
    combined_response = ""
    for i, response in enumerate(responses):
        response = response.strip()
        combined_response += f"Rank {i+1}. {response}\n\n"

    return combined_response.strip()
