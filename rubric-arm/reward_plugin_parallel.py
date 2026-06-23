"""External pairwise reward plugin for rubric-generator GRPO training.

The training semantics in this file intentionally match the original research
code. The policy completion is treated as the candidate rubric, an external
pairwise judge is asked to choose between Response A and Response B, and the
reward is +1 when that judge agrees with the dataset winner and -1 otherwise.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Dict, List, Sequence, Tuple

import requests
from swift.plugin import ORM, orms


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


def create_pairwise_judge_prompt(instruction: str, response_a: str, response_b: str, rubric: str) -> str:
    return f"""You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' based on a given instruction and a rubric. You will conduct this evaluation in distinct phases as outlined below.

Phase 1: Compliance Check Instructions
First, identify the single most important, objective 'Gatekeeper Criterion' from the rubric.

A rule is objective (and likely a Gatekeeper) if it can be verified without opinion. Key examples are: word/paragraph limits, required output format (e.g., JSON validity), required/forbidden sections, or forbidden content.
Conversely, a rule is subjective if it requires interpretation or qualitative judgment. Subjective rules about quality are NOT Gatekeepers. Examples include criteria like "be creative," "write clearly," "be engaging," or "use a professional tone."
Phase 2: Analyze Each Response
Next, for each Gatekeeper Criterion and all other criteria in the rubric, evaluate each response item by item.

Phase 3: Final Judgment Instructions
Based on the results from the previous phases, determine the winner using these simple rules. Provide a final justification explaining your decision first and then give your decision.

REQUIRED OUTPUT FORMAT
You must follow this exact output format below.

--- Compliance Check ---
Identified Gatekeeper Criterion: <e.g., Criterion 1: Must be under 50 words.>

--- Analysis ---
Response A:

Criterion 1 [Hard Rule]: Justification: <...>
Criterion 2 [Hard Rule]: Justification: <...>
Criterion 3 [Principle]: Justification: <...>
... (and so on for all other criteria)
Response B:

Criterion 1 [Hard Rule]: Justification: <...>
Criterion 2 [Hard Rule]: Justification: <...>
Criterion 3 [Principle]: Justification: <...>
... (and so on for all other criteria)
--- Final Judgment ---
Justification: <...>
Winner: <Response A / Response B>

Task to Evaluate:
Instruction:
{instruction}

Rubric:
{rubric}

Response A:
{response_a}

Response B:
{response_b}
"""


def parse_winner_from_response(judge_response: str) -> str:
    for line in judge_response.split("\n"):
        if line.strip().startswith("Winner:"):
            return line.split("Winner:")[1].strip()
    return "Unknown"


class PairwiseJudgeReward(ORM):
    """
    Supports two reward backends:
      - server: a custom /score endpoint that accepts batched instances.
      - openai: a vLLM/OpenAI-compatible chat completions endpoint.

    Environment variables:
      PAIRWISE_REWARD_ENDPOINT: e.g. http://127.0.0.1:8002/v1 or
        http://127.0.0.1:8000/score.
      PAIRWISE_REWARD_MODEL: model id for OpenAI-compatible mode.
      PAIRWISE_REWARD_API_KEY: bearer token if authentication is required.
      PAIRWISE_REWARD_WORKERS: concurrent request count, default 32.
      PAIRWISE_REWARD_SERVER_BATCH: /score batch size, default 1024.
      PAIRWISE_REWARD_TIMEOUT: request timeout in seconds, default 60.
    """

    def __init__(self,
                 endpoint: str | None = None,
                 timeout: float | None = None,
                 api_key: str | None = None,
                 model_name: str | None = None) -> None:
        super().__init__()
        self.endpoint = endpoint or os.environ.get("PAIRWISE_REWARD_ENDPOINT", "http://127.0.0.1:8000")
        self.timeout = timeout or float(os.environ.get("PAIRWISE_REWARD_TIMEOUT", "60"))
        self._session = requests.Session()
        self._api_key = api_key or os.environ.get("PAIRWISE_REWARD_API_KEY", "EMPTY")
        self._model_name = model_name or os.environ.get("PAIRWISE_REWARD_MODEL")

        self._workers = int(os.environ.get("PAIRWISE_REWARD_WORKERS", "32"))
        self._server_batch = int(os.environ.get("PAIRWISE_REWARD_SERVER_BATCH", "1024"))

        base = self.endpoint.rstrip("/")
        self._mode = "server" if base.endswith("/score") else "openai"
        if self._mode == "openai":
            self._configure_openai_backend(base)

    def _configure_openai_backend(self, base: str) -> None:
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        self._models_url = f"{base}/models"
        self._chat_url = f"{base}/chat/completions"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key or 'EMPTY'}",
        }
        if self._model_name is None:
            resp = self._session.get(self._models_url, headers=self._headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            try:
                self._model_name = data["data"][0]["id"]
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Failed to fetch model id from {self._models_url}: {data}") from exc

    def _score_via_server(self,
                          completions: Sequence[str],
                          instruction: Sequence[str],
                          response_a: Sequence[str],
                          response_b: Sequence[str],
                          winner: Sequence[str]) -> List[float]:

        def _post_batch(batch_payload: List[Dict]) -> List[float]:
            r = self._session.post(self.endpoint, json={"instances": batch_payload}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            rewards = data.get("rewards")
            if rewards is None:
                raise RuntimeError(f"Reward service returned invalid payload: {data}")
            return [float(x) if x is not None else -1.0 for x in rewards]

        n = len(completions)
        if n == 0:
            return []

        payload = [{
            "instruction": instr,
            "rubric": rubric,
            "response_a": ans_a,
            "response_b": ans_b,
            "winner": gold_winner,
        } for rubric, instr, ans_a, ans_b, gold_winner in zip(
            completions, instruction, response_a, response_b, winner)]

        if n <= self._server_batch or self._workers <= 1:
            return _post_batch(payload)

        results: List[float] = [-1.0] * n
        futures = []
        with ThreadPoolExecutor(max_workers=self._workers) as ex:
            for i in range(0, n, self._server_batch):
                sub = payload[i:i + self._server_batch]
                futures.append((i, ex.submit(_post_batch, sub)))
            for i, fut in futures:
                part = fut.result()
                results[i:i + len(part)] = part
        return results

    def _score_via_openai(self,
                          completions: Sequence[str],
                          instruction: Sequence[str],
                          response_a: Sequence[str],
                          response_b: Sequence[str],
                          winner: Sequence[str]) -> List[float]:

        def _build_payload(prompt: str) -> Dict[str, object]:
            return {
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024,
                "n": 1,
                "extra_body": {"enable_thinking": False},
            }

        def _one_request(idx: int, prompt: str, expected: str) -> Tuple[int, float]:
            try:
                payload = _build_payload(prompt)
                r = self._session.post(self._chat_url, headers=self._headers, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                judge_output = data["choices"][0]["message"]["content"]
                predicted = parse_winner_from_response(judge_output)
                pred_label = _normalize_label(predicted)
                exp_label = _normalize_label(expected)
                reward = 1.0 if (pred_label != "unknown" and pred_label == exp_label) else -1.0
                return idx, reward
            except Exception:
                return idx, -1.0

        n = len(completions)
        if n == 0:
            return []

        prompts = [
            create_pairwise_judge_prompt(instr, ans_a, ans_b, rubric)
            for rubric, instr, ans_a, ans_b in zip(completions, instruction, response_a, response_b)
        ]
        results: List[float] = [-1.0] * n
        with ThreadPoolExecutor(max_workers=self._workers) as ex:
            futures = [ex.submit(_one_request, i, p, w) for i, (p, w) in enumerate(zip(prompts, winner))]
            for fut in as_completed(futures):
                i, rew = fut.result()
                results[i] = rew
        return results

    def __call__(self,
                 completions: Sequence[str],
                 instruction: Sequence[str],
                 response_a: Sequence[str],
                 response_b: Sequence[str],
                 winner: Sequence[str],
                 **kwargs) -> List[float]:
        if self._mode == "server":
            return self._score_via_server(completions, instruction, response_a, response_b, winner)
        return self._score_via_openai(completions, instruction, response_a, response_b, winner)


orms["pairwise_judge_reward"] = PairwiseJudgeReward
