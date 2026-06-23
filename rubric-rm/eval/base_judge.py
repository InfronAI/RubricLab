import asyncio
from typing import List, Optional, Any
from tqdm.asyncio import tqdm_asyncio

from api import generate_response

# JUDGE_PROMPT_TEMPLATE = (# define the system prompt as a tuple of strings for readability and maintainability
#     f"You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' "
#     f"based on a given instruction and a rubric. You will conduct this evaluation in distinct "
#     f"phases as outlined below.\n\n"
#     f"### Phase 1: Compliance Check Instructions\n"
#     f"First, identify the single most important, objective 'Gatekeeper Criterion' from the rubric.\n"
#     f"- **A rule is objective (and likely a Gatekeeper) if it can be verified without opinion. "
#     f"Key examples are: word/paragraph limits, required output format (e.g., JSON validity), "
#     f"required/forbidden sections, or forbidden content.**\n"
#     f"- **Conversely, a rule is subjective if it requires interpretation or qualitative judgment. "
#     f"Subjective rules about quality are NOT Gatekeepers. Examples include criteria like \"be creative,\" "
#     f"\"write clearly,\" \"be engaging,\" or \"use a professional tone.\"**\n\n"
#     f"### Phase 2: Analyze Each Response\n"
#     f"Next, for each Gatekeeper Criterion and all other criteria in the rubric, evaluate each "
#     f"response item by item.\n\n"
#     f"### Phase 3: Final Judgment Instructions\n"
#     f"Based on the results from the previous phases, determine the winner using these simple rules. "
#     f"Provide a final justification explaining your decision first and then give your decision.\n\n"
#     f"---\n"
#     f"### REQUIRED OUTPUT FORMAT\n"
#     f"You must follow this exact output format below.\n\n"
#     f"--- Compliance Check ---\n"
#     f"Identified Gatekeeper Criterion: <e.g., Criterion 1: Must be under 50 words.>\n\n"
#     f"--- Analysis ---\n"
#     f"**Response A:**\n"
#     f"- Criterion 1 [Hard Rule]: Justification: <...>\n"
#     f"- Criterion 2 [Hard Rule]: Justification: <...>\n"
#     f"- Criterion 3 [Principle]: Justification: <...>\n"
#     f"- ... (and so on for all other criteria)\n\n"
#     f"**Response B:**\n"
#     f"- Criterion 1 [Hard Rule]: Justification: <...>\n"
#     f"- Criterion 2 [Hard Rule]: Justification: <...>\n"
#     f"- Criterion 3 [Principle]: Justification: <...>\n"
#     f"- ... (and so on for all other criteria)\n\n"
#     f"--- Final Judgment ---\n"
#     f"Justification: <...>\n"
#     f"Winner: <Response A / Response B>\n\n\n"
#     f"Task to Evaluate:\n"
#     "Instruction:\n{instruction}\n\n"
#     "Rubric:\n{rubric}\n\n"
#     "Response A:\n{response_a}\n\n"
#     "Response B:\n{response_b}"
# )

JUDGE_PROMPT_TEMPLATE = (
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
    f"Think step-by-step to determine this single most important Gatekeeper, then write a 1–2 sentence explanation of your decision.\n\n"

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
    # "Aggregation Summary: <Provide a detailed, step-by-step explanation (3–6 sentences) of how the Gatekeeper and other criteria led to the decision>\n"
    "Aggregation Summary: <1–3 sentences explaining how Gatekeeper and other criteria led to the decision>\n"
    "Justification: <...>\n"
    "Winner: <Response A / Response B>\n\n\n"
    "Task to Evaluate:\n"
    "Instruction:\n{instruction}\n\n"
    "Rubric:\n{rubric}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}"
).strip()



class Judge:
    def __init__(self, model, tokenizer, sp):        
        self.model = model
        self.tokenizer = tokenizer
        self.sp = sp
        self.can_enable_think = True

    def gen_judge(self, instructions, rurbics, responses_a, responses_b):
        prompts = []
        for instruction, rubric, response_a, response_b in zip(instructions, rurbics, responses_a, responses_b):
            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                instruction=instruction,
                rubric=rubric,
                response_a=response_a,
                response_b=response_b
            )
            prompts.append(judge_prompt)

        chat_inputs = [self.build_chat_inputs(prompt) for prompt in prompts]
        outputs = self.model.generate(chat_inputs, self.sp)
        generations = [o.outputs[0].text for o in outputs]

        return generations

    def build_chat_inputs(self, user_text: str) -> str:
        messages = [
            {"role": "user", "content": user_text},
        ]

        if self.can_enable_think:
            try:
                message =  self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except Exception as e:
                message = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                )

                self.can_enable_think = False
        else:
            message = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
            )
        
        return message
    


class APIJudge:
    """
    API version of the Judge (interface aligned with the vLLM Judge):

    - __init__(model, tokenizer, sp, ...)
        model: str, e.g. "gpt-4.1" / "gemini" / "deepseek"
        tokenizer/sp: kept for compatibility with external scripts; sp.max_tokens is used if present
    - gen_judge(instructions, rurbics, responses_a, responses_b) -> List[str]
        synchronous wrapper: uses asyncio.run internally
    - gen_judge_async(...) -> List[str]
        async version: if you are in an async pipeline, await this directly
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[Any] = None,
        sp: Optional[Any] = None,
        *,
        concurrency: int = 10,
        default_max_tokens: int = 4096,
        show_pbar: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sp = sp

        # IMPORTANT: do NOT store asyncio.Semaphore on self (event-loop bound)
        self.concurrency = concurrency
        self.default_max_tokens = default_max_tokens
        self.show_pbar = show_pbar

        # keep for interface parity
        self.can_enable_think = False

    def _get_max_tokens(self) -> int:
        mt = getattr(self.sp, "max_tokens", None)
        if isinstance(mt, int) and mt > 0:
            return mt
        return self.default_max_tokens

    def build_chat_inputs(self, user_text: str) -> str:
        # the API path uses user_text directly
        return user_text

    async def _judge_one(
        self,
        instruction: str,
        rubric: str,
        response_a: str,
        response_b: str,
        sample_id: int,
        semaphore: asyncio.Semaphore,
    ) -> str:
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            instruction=instruction,
            rubric=rubric,
            response_a=response_a,
            response_b=response_b,
        )
        user_prompt = self.build_chat_inputs(judge_prompt)

        resp = await generate_response(
            user_prompt=user_prompt,
            model_name=self.model,
            max_tokens=self._get_max_tokens(),
            semaphore=semaphore,
            sample_id=sample_id,
        )

        if not resp:
            return ""
        return (resp.get("text") or "").strip()

    async def gen_judge_async(
        self,
        instructions: List[str],
        rurbics: List[str],
        responses_a: List[str],
        responses_b: List[str],
    ) -> List[str]:
        n = min(len(instructions), len(rurbics), len(responses_a), len(responses_b))
        if n == 0:
            return []

        # Create semaphore in CURRENT event loop
        semaphore = asyncio.Semaphore(self.concurrency)

        coros = [
            self._judge_one(
                instruction=instructions[i],
                rubric=rurbics[i],
                response_a=responses_a[i],
                response_b=responses_b[i],
                sample_id=i,
                semaphore=semaphore,
            )
            for i in range(n)
        ]

        if self.show_pbar:
            return await tqdm_asyncio.gather(
                *coros, desc=f"Judge({self.model})", leave=False
            )
        return await asyncio.gather(*coros)

    def gen_judge(
        self,
        instructions: List[str],
        rurbics: List[str],
        responses_a: List[str],
        responses_b: List[str],
    ) -> List[str]:
        """
        synchronous interface: for the current eval scripts (for loop + synchronous calls)
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "APIJudge.gen_judge() called inside a running event loop. "
                "Use: await gen_judge_async(...) instead."
            )

        return asyncio.run(self.gen_judge_async(instructions, rurbics, responses_a, responses_b))