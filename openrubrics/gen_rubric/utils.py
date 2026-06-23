import os 
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_results_path(model_name: str, dataset_slug: str) -> str:
    # dedicated location for rubrics: rubrics/<dataset>/<model>.jsonl
    datasets_path = os.path.join(root_path, "rubrics", dataset_slug)
    os.makedirs(datasets_path, exist_ok=True)
    filename = f"{dataset_slug}-{model_name.replace('/', '_')}.jsonl"
    return os.path.join(datasets_path, filename)


