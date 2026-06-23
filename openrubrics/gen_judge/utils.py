import os 
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_results_path(model_name: str, dataset_slug: str) -> str:
    datasets_path = os.path.join(root_path, "judges", dataset_slug)
    os.makedirs(datasets_path, exist_ok=True)
    filename = f"{dataset_slug}-{model_name.replace('/', '_')}.jsonl"
    return os.path.join(datasets_path, filename)


def parse_winner(judge_response: str) -> str:
    """Parse the winner from the judge response string."""

    LABEL = {
        "A": "a",
        "B": "b",
        "Assistant 1": "a",
        "Assistant 2": "b",
        "Response A": "a",
        "Response B": "b",
        "Assistant A": "a",
        "Assistant B": "b",
    }

    lines = judge_response.split('\n')
    for line in lines:
        if line.strip().startswith('Winner:'):
            winner = line.split('Winner:')[1].strip()
            return LABEL.get(winner, 'Unknown')

    return 'Unknown'
