import os

from typing import List, Dict, Any, Optional
import asyncio

from openai import AsyncAzureOpenAI, RateLimitError, APIError
from tqdm.asyncio import tqdm_asyncio

# boto3 / botocore are imported lazily inside generate_response_deepseek so that
# the default path (GPT / o4-mini via Azure) does not require boto3 installed.


azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")

global client
client = None

global bedrock_client
bedrock_client = None

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
