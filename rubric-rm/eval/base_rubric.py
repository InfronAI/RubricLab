import asyncio
from typing import List, Optional, Any
from tqdm.asyncio import tqdm_asyncio



RUBRIC_PROMPT_TEMPLATE = (
    "Your task is to extract a set of rubric-style instructions from a user's request.\n"
    "These rubrics will be used as evaluation criteria to check if a response fully meets the request.\n"
    "Every rubric item must be a universal principle. If any rubric still contains topic-specific references (e.g., names, places, myths, numbers, historical facts), it is automatically invalid.\n"
    "\n"
    "- **Two Distinct Categories:**\n"
    "  - [Hard Rule]: Derived strictly from explicit requirements stated in the <request> (format, length, structure, forbidden/required elements, etc.).\n"
    "  - [Principle]: Derived by abstracting any concrete cues into domain-agnostic quality criteria (e.g., clarity, correctness, sound reasoning, pedagogy).\n"
    "\n"
    "- **Comprehensiveness:**\n"
    "  The rubric must cover all critical aspects implied by the request and examples, including explicit requirements and implicit quality standards.\n"
    "\n"
    "- **Conciseness & Uniqueness:**\n"
    "  Each rubric must capture a distinct evaluation criterion. Overlapping or redundant criteria must be merged into a single rubric. Wording must be precise and free of repetition.\n"
    "\n"
    "- **Format Requirements:**\n"
    "  - Use a numbered list.\n"
    "  - Each item starts with \"The response\" phrased in third person.\n"
    "  - Append [Hard Rule] or [Principle] at the end of each item.\n"
    "  - Do not include reasoning, explanations, or examples in the final output—only the rubrics.\n"
    "\n"
    "Here is the request:\n"
    "{prompt}\n"
    "\n"
    "Please generate the rubrics for the above request."
)


class Rubric:
    def __init__(self, model, tokenizer, sp):        
        self.model = model
        self.tokenizer = tokenizer
        self.sp = sp
        self.can_enable_think = True

    def gen_rubric(self, instructions):
        prompts = [RUBRIC_PROMPT_TEMPLATE.format(prompt=inst) for inst in instructions]
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
    

class APIRubric:
    """
    API version of the Rubric generator (interface aligned with the vLLM Rubric):

    - __init__(model, tokenizer, sp, ...)
        model: str, e.g. "gpt-4.1" / "gemini" / "deepseek"
        tokenizer/sp: kept for compatibility with external scripts, but unused
    - gen_rubric(instructions) -> List[str]
        synchronous wrapper: uses asyncio.run internally
    - gen_rubric_async(instructions) -> List[str]
        async version: if you are in an async pipeline, await this directly
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[Any] = None,
        sp: Optional[Any] = None,
        *,
        max_tokens: int = 4096,
        concurrency: int = 10,
        show_pbar: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sp = sp
        self.max_tokens = max_tokens

        # IMPORTANT: do NOT store asyncio.Semaphore on self (event-loop bound)
        self.concurrency = concurrency
        self.show_pbar = show_pbar

        # keep for interface parity with vLLM class
        self.can_enable_think = False

    def build_chat_inputs(self, user_text: str) -> str:
        # the API path uses user_text directly
        return user_text

    async def _gen_one(self, inst: str, sample_id: int, semaphore: asyncio.Semaphore) -> str:
        prompt = RUBRIC_PROMPT_TEMPLATE.format(prompt=inst)
        user_prompt = self.build_chat_inputs(prompt)

        from api import generate_response

        resp = await generate_response(
            user_prompt=user_prompt,
            model_name=self.model,
            max_tokens=self.max_tokens,
            semaphore=semaphore,
            sample_id=sample_id,
        )

        if not resp:
            return ""
        return (resp.get("text") or "").strip()

    async def gen_rubric_async(self, instructions: List[str]) -> List[str]:
        if not instructions:
            return []

        # Create semaphore in CURRENT event loop
        semaphore = asyncio.Semaphore(self.concurrency)

        coros = [self._gen_one(inst, i, semaphore) for i, inst in enumerate(instructions)]

        if self.show_pbar:
            return await tqdm_asyncio.gather(
                *coros, desc=f"RubricGen({self.model})", leave=False
            )
        return await asyncio.gather(*coros)

    def gen_rubric(self, instructions: List[str]) -> List[str]:
        """
        synchronous interface: for the current eval scripts (for loop + synchronous calls)
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "APIRubric.gen_rubric() called inside a running event loop. "
                "Use: await gen_rubric_async(...) instead."
            )

        return asyncio.run(self.gen_rubric_async(instructions))