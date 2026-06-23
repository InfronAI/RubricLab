"""Shared helpers for the RubricRM evaluation scripts."""
import os


# Released RubricRM checkpoints (Hugging Face Hub).
MODELS = {
    "8": {
        "rubric": "OpenRubrics/RubricRM-8B-Rubric-v2",
        "judge": "OpenRubrics/RubricRM-8B-Judge-v2",
    },
    "4": {
        "rubric": "OpenRubrics/RubricRM-4B-Rubric-v2",
        "judge": "OpenRubrics/RubricRM-4B-Judge-v2",
    },
}

# Keywords that mark a model name as a hosted-API baseline (served through
# api.py) rather than a local / Hugging Face checkpoint loaded with vLLM.
API_KEYWORDS = ("gpt", "gemini", "deepseek", "claude", "o1", "o3", "o4")


def is_api_model(model_name: str) -> bool:
    if os.path.exists(model_name):
        return False
    return any(k in model_name.lower() for k in API_KEYWORDS)


def resolve_models(size: str, rubric_override=None, judge_override=None):
    """Return (rubric_model, judge_model) for a given size, honoring overrides."""
    rubric = rubric_override or MODELS[size]["rubric"]
    judge = judge_override or MODELS[size]["judge"]
    return rubric, judge
