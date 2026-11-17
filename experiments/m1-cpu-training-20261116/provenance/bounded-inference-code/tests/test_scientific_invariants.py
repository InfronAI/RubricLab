"""Protect scientific correctness: leakage, masks, rewards, and actual gradient direction."""
import pytest
import torch

from openrubrics_lab.data import prompt_key, split_by_prompt, validate_pair
from openrubrics_lab.metrics import format_reward, majority, parse_winner, summarize
from openrubrics_lab.prompts import contrastive_prompt, judge_prompt, rubric_prompt
from openrubrics_lab.training import advantages, dpo_objective, grpo_objective


@pytest.mark.parametrize("text,expected", [
    ("Winner: Response A", "A"), ("**Winner: Response B**", "B"),
    ("Winner: A", "A"), ("Winner: Response C", None),
    ("Winner: Response A or Response B", None),
    ("Winner: A\nWinner: B", None), ("Quoted Winner: A", None), ("", None),
])
def test_strict_winner(text, expected):
    assert parse_winner(text) == expected


def test_vote_requires_majority_of_all_samples():
    assert majority(["A", "A", None, None, None]) is None
    assert majority(["A", "A", "A", "B", None]) == "A"
    assert majority([]) is None
    assert majority(["A", "B"]) is None


def test_metrics_do_not_drop_failed_outputs():
    rows = [{"label": "A", "predictions": ["Winner: A", "bad", "bad"]},
            {"label": "B", "predictions": ["Winner: A", "Winner: B", "Winner: B"]}]
    result = summarize(rows)
    assert result["accuracy"] == 0.5
    assert result["majority_accuracy"] == 0.5
    assert result["invalid_rate"] == pytest.approx(1/3)
    assert summarize([])["accuracy"] is None


def test_rb2_group_metric_requires_all_rejections_defeated():
    rows = [{"label": "A", "predictions": [pred], "group_id": group} for group, pred in
            [("prompt:chosen:1", "Winner: A"), ("prompt:chosen:1", "Winner: B"),
             ("prompt:chosen:2", "Winner: A")]]
    result = summarize(rows)
    assert result["accuracy"] == pytest.approx(2/3)
    assert result["group_all_correct"] == 0.5


def test_prompt_disjoint_split_and_normalization():
    rows = [{"instruction": x} for x in ["Same   prompt", "same prompt", "two", "three", "four"]]
    train, val = split_by_prompt(rows, 1)
    assert not {prompt_key(r["instruction"]) for r in train} & {prompt_key(r["instruction"]) for r in val}
    assert len(train)+len(val) == len(rows)


def test_prompts_preserve_json_without_double_escaping():
    instruction = 'Return {"key": 1}'
    for style in ("compact", "original"):
        assert instruction in rubric_prompt(instruction, style)
        assert instruction in contrastive_prompt(instruction, ["good", "bad"], style)
        assert instruction in judge_prompt(instruction, "rules", "a", "b", style)


def test_invalid_pair_rejected():
    with pytest.raises(ValueError):
        validate_pair({"instruction": "q", "response_a": "same", "response_b": "same", "label": "A"})


def test_grpo_zero_variance_and_direction():
    assert torch.equal(advantages(torch.ones(3)), torch.zeros(3))
    adv = advantages(torch.tensor([0., 1., 1.]))
    assert adv.mean().item() == pytest.approx(0, abs=1e-6)
    logp = torch.tensor([-2., -1.], requires_grad=True)
    old = logp.detach().clone()
    loss, kl = grpo_objective(logp, old, old, 1.)
    loss.backward()
    assert kl.item() == pytest.approx(0)
    assert (logp.grad < 0).all()  # gradient descent increases likelihood of rewarded tokens


def test_grpo_ratio_clipping_and_positive_kl():
    current = torch.tensor([0.0], requires_grad=True)
    old = torch.tensor([-2.0])
    loss, kl = grpo_objective(current, old, old, 1., beta=0)
    assert loss.item() == pytest.approx(-1.28)
    assert kl.item() > 0
    loss.backward()
    assert current.grad.item() == 0  # positive advantage cannot increase past the clip


def test_dpo_gradient_prefers_chosen():
    chosen = torch.tensor([-2., -1.], requires_grad=True)
    rejected = torch.tensor([-3., -2.], requires_grad=True)
    loss, margin = dpo_objective(chosen, rejected, chosen.detach(), rejected.detach(), sft_weight=0)
    assert loss.item() == pytest.approx(0.69314718)
    assert margin.item() == 0
    loss.backward()
    assert (chosen.grad < 0).all()
    assert (rejected.grad > 0).all()


def test_format_reward_handles_published_numbered_rubrics():
    rubric = "1. The response is concise. [Hard Rule]\n2. The response is clear. [Principle]"
    completion = """--- Compliance Check ---
Gatekeeper Reasoning: Length is explicit.
Identified Gatekeeper Criterion: Criterion 1
--- Analysis ---
**Response A:**
- Criterion 1 [Hard Rule]: Justification: short.
- Criterion 2 [Principle]: Justification: clear.
**Response B:**
- Criterion 1 [Hard Rule]: Justification: long.
- Criterion 2 [Principle]: Justification: unclear.
--- Final Judgment ---
Aggregation Summary: A complies.
Justification: A is better.
Winner: Response A"""
    assert format_reward(completion, rubric) == 0.5
    assert format_reward("", rubric) < 0.1
