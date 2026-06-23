import os

from typing import Dict, Any, Optional
import asyncio

from openai import AsyncAzureOpenAI, RateLimitError, APIError

# Optional API baselines (DeepSeek via Bedrock, Gemini via google-genai) are
# imported lazily inside their respective functions so that the common path
# (local / Hugging Face checkpoints served with vLLM) does not require boto3 or
# google-genai to be installed.

# Azure OpenAI endpoint for the GPT API baselines. Set via env var.
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
    model_name: str,  # e.g. "deepseek.v3-v1:0"
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    sample_id: int,
) -> Optional[Dict[str, Any]]:
    """Call a DeepSeek model through Amazon Bedrock (boto3)."""
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError

    global bedrock_client
    if bedrock_client is None:
        custom_config = Config(connect_timeout=840, read_timeout=840)
        # AWS credentials must be configured in the environment.
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            config=custom_config,
            region_name=os.getenv("AWS_REGION", "us-west-2"),
        )

    async with semaphore:
        messages = [
            {
                "role": "user",
                "content": [{"text": user_prompt}]
            }
        ]
        inference_config = {
            "temperature": 0.0,
            "maxTokens": max_tokens
        }
        system_prompts = []

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    bedrock_client.converse,
                    modelId=model_name,
                    messages=messages,
                    system=system_prompts,
                    inferenceConfig=inference_config
                )

                text = ""
                for content in response['output']['message']['content']:
                    if 'text' in content:
                        text = content['text'].strip()
                        break

                if not text:
                    raise Exception("No text content found in response")

                return {"sample_id": sample_id, "text": text}

            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    wait = 2 ** (attempt + 1)
                    wait = min(wait, 60)
                    print(f"Bedrock Rate limit, sample {sample_id}: retry after {wait}s")
                    await asyncio.sleep(wait)
                else:
                    print(f"Sample {sample_id} Boto3 ClientError: {e}")
                    break

            except Exception as e:
                print(f"Sample {sample_id} other error: {e}")
                break

        print(f"Sample {sample_id} failed after {max_retries} attempts")
        return None


async def generate_response_gemini(
    user_prompt: str,
    model_name: str,  # e.g. "gemini-2.5-flash-lite"
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    sample_id: int,
    api_key: Optional[str] = None,  # falls back to env GOOGLE_API_KEY
) -> Optional[Dict[str, Any]]:
    """Call a Gemini model through google-genai (ai.google.dev style)."""
    from google import genai
    from google.api_core import exceptions as gex

    global gemini_client

    if gemini_client is None:
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not provided and GOOGLE_API_KEY is not set."
            )
        gemini_client = genai.Client(api_key=api_key)

    async with semaphore:
        config = {
            "max_output_tokens": max_tokens,
            "candidate_count": 1,
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=model_name,
                    contents=user_prompt,
                    config=config,
                )

                text = (getattr(resp, "text", "") or "").strip()

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

            except (gex.TooManyRequests, gex.ResourceExhausted):
                wait = 2 ** (attempt + 1)
                wait = min(wait, 60)
                print(f"Gemini Rate limit, sample {sample_id}: retry after {wait}s")
                await asyncio.sleep(wait)

            except Exception as e:
                print(f"Sample {sample_id} other error: {e}")
                break

        print(f"Sample {sample_id} failed after {max_retries} attempts")
        return None
