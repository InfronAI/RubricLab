import json
import os
import re
import argparse
import numpy as np
from typing import Optional, List
import unicodedata
from collections import Counter


LABEL = {
    "A": "response A",
    "B": "response B",
    "Assistant 1": "response A",
    "Assistant 2": "response B",
}


def parse_winner_from_response(judge_response: str) -> str:
    """Parse the winner from the judge response string."""
    text = normalize_text(judge_response)
    matches = re.findall(r'^\s*Winner:\s*(?:Response\s+)?([AB])\s*[.!]?\s*$', text, re.M | re.I)
    return f"Response {matches[0].upper()}" if len(matches) == 1 else "Unknown"


def voting(winners: List[str]) -> str:
    vote_count = Counter(pred.strip().lower() for pred in winners)
    for label in ("response a", "response b"):
        if vote_count[label] > len(winners) / 2:
            return label
    return "unknown"


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("​", "").replace("﻿", "").replace(" ", " ")
    s = s.replace("**", "")
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy from the prediction JSON files written by the eval scripts."
    )
    parser.add_argument("folder", nargs="?", default="./outputs/ours[8B]",
                        help="Directory containing the prediction JSON files.")
    args = parser.parse_args()

    folder_path = args.folder

    file_paths = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            file_paths.append(os.path.join(folder_path, file))
    file_paths.sort()

    for file_path in file_paths:
        print(file_path.split("/")[-1])
        with open(file_path, 'r') as f:
            file = json.load(f)

        scores = []
        level = {}
        pairs = {}

        scores_mv = []
        level_mv = {}
        pairs_mv = {}

        for i in range(len(file)):
            if 'label' not in file[i]:
                assert "rubric" in file[i] and "judge" in file[i]
                continue

            label = LABEL[file[i]['label']]
            winners = [parse_winner_from_response(pred) for pred in file[i]['prediction']]
            is_corrects = [winner.lower() == label.lower() for winner in winners]
            scores += is_corrects

            mv_winner = voting(winners)
            is_corrects_mv = mv_winner.lower() == label.lower()
            scores_mv.append(is_corrects_mv)

            if "rmbench" in file_path:
                level.setdefault(file[i]['difficulty'], [])
                level[file[i]['difficulty']] += is_corrects
                level_mv.setdefault(file[i]['difficulty'], [])
                level_mv[file[i]['difficulty']] += [is_corrects_mv]

            if "rewardbench2" in file_path:
                sample_id, chosen_id, rejected_id = file[i]['id'].split("_")
                pairs.setdefault(sample_id, {})
                pairs_mv.setdefault(sample_id, {})
                pairs[sample_id].setdefault(chosen_id, [])
                pairs[sample_id][chosen_id] += is_corrects
                pairs_mv[sample_id].setdefault(chosen_id, [])
                pairs_mv[sample_id][chosen_id] += [is_corrects_mv]

        if level:
            for k, v in level.items():
                print(k, (np.mean(v) * 100).round(1), len(v))

        if level_mv:
            for k, v in level_mv.items():
                print("Majority Voting: ", k, (np.mean(v) * 100).round(1), len(v))

        if pairs:
            accs = []
            for k, v in pairs.items():
                for k2, v2 in v.items():
                    accs.append(float(np.min(v2)))
            print("Pairwise accuracy:", (np.mean(accs) * 100).round(1), len(accs))

        if pairs_mv:
            accs_mv = []
            for k, v in pairs_mv.items():
                for k2, v2 in v.items():
                    accs_mv.append(float(np.min(v2)))
            print("Majority Voting - Pairwise accuracy: ", (np.mean(accs_mv) * 100).round(1), len(accs_mv))
        else:
            print("Prev: ", (np.mean(scores) * 100).round(1), len(scores))
            print("Majority Voting: ", (np.mean(scores_mv) * 100).round(1), len(scores_mv))
        print("\n")


if __name__ == "__main__":
    main()
