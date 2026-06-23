import os

from typing import List, Dict, Any, Optional
import asyncio

from openai import AsyncAzureOpenAI, RateLimitError, APIError
from tqdm.asyncio import tqdm_asyncio

# boto3 / botocore (Bedrock) and google-genai (Gemini) are imported lazily inside
# their backend functions so the default path doesn't require every SDK installed.

global azure_endpoint
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")

global client
client = None

global bedrock_client
bedrock_client = None

global gemini_client
gemini_client = None


async def generate_response(
    user_prompt: str,
    model_name: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    sample_id: int,
) -> Optional[Dict[str, Any]]:
    if "deepseek" in model_name:
        model_name = "deepseek.v3-v1:0"
        return await generate_response_deepseek(
            user_prompt, model_name, max_tokens, semaphore, sample_id)
    elif "gemini" in model_name:
        model_name = "gemini-2.5-flash-lite"
        return await generate_response_gemini(
            user_prompt, model_name, max_tokens, semaphore, sample_id)
    else:
        return await generate_response_gpt(
            user_prompt, model_name, max_tokens, semaphore, sample_id)


async def generate_response_gpt(
    # client: AsyncOpenAI,
    user_prompt: str,
    model_name: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    sample_id: int,
) -> Optional[Dict[str, Any]]:

    # lazy but persistent client initialization
    global client
    if client is None:
        client = AsyncAzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=azure_endpoint,
            api_key=os.getenv("OPENAIKEY"),
            )

    async with semaphore:
        messages = [{"role": "user", "content": user_prompt}]

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    # temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                return {"sample_id": sample_id, "text": text}
            except RateLimitError:
                wait = 2 ** (attempt + 1)
                wait = min(wait, 60)
                print(f"Rate limit, sample {sample_id}: retry after {wait}s")
                await asyncio.sleep(wait)
            except APIError as e:
                print(f"Sample {sample_id} API error: {e}")
                break
            except Exception as e:
                print(f"Sample {sample_id} other error: {e}")
                break
        print(f"Sample {sample_id} failed after {max_retries} attempts")

        return None


async def generate_response_deepseek(
    user_prompt: str,
    model_name: str, # e.g. "deepseek.v3-v1:0"
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    sample_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Call a DeepSeek model asynchronously through Amazon Bedrock (boto3).
    """
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError

    global bedrock_client # use a separate global client for Bedrock
    if bedrock_client is None:
        # configuration from the reference example
        custom_config = Config(connect_timeout=840, read_timeout=840)
        # make sure AWS credentials are configured in the environment (e.g. AWS_ACCESS_KEY_ID, etc.)
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            config=custom_config,
            region_name='us-west-2' # from the reference example
        )

    async with semaphore:
        # 1. Format the input expected by the Bedrock converse API

        # Bedrock expects content to be a list containing a "text" key
        messages = [
            {
                "role": "user",
                "content": [{"text": user_prompt}]
            }
        ]

        # Bedrock inference configuration
        inference_config = {
            "temperature": 0.0,  # match the deterministic intent of the GPT example
            "maxTokens": max_tokens
        }

        # Bedrock system prompt (if needed)
        # The GPT function interface has no system_prompt, so we pass an empty list
        system_prompts = []

        # 2. Mirror the retry logic of the GPT function
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # 3. Run the synchronous boto3 call asynchronously
                response = await asyncio.to_thread(
                    bedrock_client.converse,
                    modelId=model_name,
                    messages=messages,
                    system=system_prompts,
                    inferenceConfig=inference_config
                )

                # 4. Extract the response text
                # Bedrock's response content is a list; iterate to find 'text'
                text = ""
                for content in response['output']['message']['content']:
                    if 'text' in content:
                        text = content['text'].strip()
                        break  # take the first text block

                if not text:
                    # the model returned no text content
                    raise Exception("No text content found in response")

                return {"sample_id": sample_id, "text": text}

            except ClientError as e:
                # 5. Handle Boto3 errors (mirroring RateLimitError and APIError)

                # 'ThrottlingException' is the Boto3/AWS rate-limit error
                if e.response['Error']['Code'] == 'ThrottlingException':
                    wait = 2 ** (attempt + 1)
                    wait = min(wait, 60)
                    print(f"Bedrock Rate limit, sample {sample_id}: retry after {wait}s")
                    await asyncio.sleep(wait)
                else:
                    # other Boto3 client errors (similar to APIError)
                    print(f"Sample {sample_id} Boto3 ClientError: {e}")
                    break # exit the retry loop

            except Exception as e:
                # 6. Handle all other exceptions
                print(f"Sample {sample_id} other error: {e}")
                break # exit the retry loop

        # 7. If all retries failed
        print(f"Sample {sample_id} failed after {max_retries} attempts")
        return None


async def generate_response_gemini(
    user_prompt: str,
    model_name: str,  # e.g. "gemini-2.5-flash-lite"
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    sample_id: int,
    api_key: Optional[str] = None,  # falls back to env GOOGLE_API_KEY if not given
) -> Optional[Dict[str, Any]]:
    """
    Call a Gemini model asynchronously through google-genai (ai.google.dev style).
    The control flow mirrors generate_response_deepseek:
      - semaphore controls concurrency
      - max_retries=5
      - exponential backoff only on rate-limit errors (2**(attempt+1), capped at 60s)
      - other errors fail immediately
      - returns {"sample_id": sample_id, "text": text} on success, None on failure
    """
    from google import genai
    from google.api_core import exceptions as gex

    global gemini_client

    # 0) initialize the client (mirrors deepseek's bedrock_client init)
    if gemini_client is None:
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not provided and GOOGLE_API_KEY is not set."
            )
        gemini_client = genai.Client(api_key=api_key)

    async with semaphore:
        # 1) assemble the Gemini request params (mirrors deepseek's messages + inference_config)
        # Gemini takes contents=prompt here; the generation config goes in `config`
        config = {
            "temperature": 0.0,
            "max_output_tokens": max_tokens,
            "candidate_count": 1,
        }

        # 2) same retry logic as deepseek
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # 3) run the synchronous call asynchronously (mirrors asyncio.to_thread(bedrock_client.converse, ...))
                resp = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=model_name,
                    contents=user_prompt,
                    config=config,
                )

                # 4) extract the response text (mirrors deepseek's response['output']...)
                text = (getattr(resp, "text", "") or "").strip()

                # fallback: if resp.text is empty, try reading from candidates.parts
                if not text:
                    try:
                        for c in (getattr(resp, "candidates", None) or []):
                            content = getattr(c, "content", None)
                            for p in (getattr(content, "parts", None) or []):
                                t = getattr(p, "text", None)
                                if t:
                                    text = str(t).strip()
                                    break
                            if text:
                                break
                    except Exception:
                        pass

                if not text:
                    raise Exception("No text content found in response")

                return {"sample_id": sample_id, "text": text}

            except (gex.TooManyRequests, gex.ResourceExhausted) as e:
                # 5) handle rate limiting (mirrors deepseek's ThrottlingException)
                wait = 2 ** (attempt + 1)
                wait = min(wait, 60)
                print(f"Gemini Rate limit, sample {sample_id}: retry after {wait}s")
                await asyncio.sleep(wait)

            except Exception as e:
                # 6) all other exceptions (mirrors deepseek's other error)
                print(f"Sample {sample_id} other error: {e}")
                break

        # 7) all retries failed
        print(f"Sample {sample_id} failed after {max_retries} attempts")
        return None
