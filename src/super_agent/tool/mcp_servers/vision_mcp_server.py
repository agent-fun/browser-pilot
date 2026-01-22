import base64
import os
import random
import time
from typing import Awaitable, Callable, Optional, TypeVar

import asyncio
import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from fastmcp import FastMCP
from google import genai
from google.genai import types
from openai import OpenAI

from src.super_agent.tool.logger import bootstrap_logger

load_dotenv(verbose=True)

# Anthropic / OpenAI / Gemini configuration
ENABLE_CLAUDE_VISION = os.getenv("ENABLE_CLAUDE_VISION", "false").lower() == "true"
ENABLE_OPENAI_VISION = os.getenv("ENABLE_OPENAI_VISION", "false").lower() == "true"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

SANDBOX_PATH_MARKER = "home/user"

_ResultT = TypeVar("_ResultT")


# Initialize FastMCP server
mcp = FastMCP("vision-mcp-server")
logger = bootstrap_logger()


def is_sandbox_path(path_or_url: str) -> bool:
    """Return True if the given path or URL refers to a sandbox-only location."""
    return SANDBOX_PATH_MARKER in path_or_url


def ensure_https_url(url: str) -> str:
    """Normalize a URL to https if possible (Claude requires https)."""
    if url.startswith("http://"):
        return url.replace("http://", "https://", 1)
    if not url.startswith("https://"):
        return "https://" + url
    return url


async def guess_mime_type_from_path(file_path: str) -> str:
    """Guess the MIME type based on the file extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".gif":
        return "image/gif"
    if ext == ".webp":
        return "image/webp"
    # Default to JPEG if unknown
    return "image/jpeg"


async def guess_mime_media_type_from_extension(file_path: str) -> str:
    """Backward-compatible wrapper around guess_mime_type_from_path."""
    return await guess_mime_type_from_path(file_path)


async def _async_retry_with_exponential_backoff(
    operation: Callable[[], Awaitable[_ResultT]],
    *,
    max_attempts: int,
    delay_factor: int = 4,
    failure_message_builder: Optional[Callable[[int, Exception], _ResultT]] = None,
) -> _ResultT:
    """Run an async operation with exponential backoff.

    This is a light abstraction over the previous retry loops, preserving the
    same number of attempts and delay pattern while centralizing the logic.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            if attempt == max_attempts:
                if failure_message_builder is not None:
                    return failure_message_builder(max_attempts, exc)
                raise

            # Use the same 4**attempt pattern as the original implementation.
            await asyncio.sleep(delay_factor**attempt)

    # Should not be reachable, but keep a defensive fallback.
    if failure_message_builder is not None and last_exception is not None:
        return failure_message_builder(max_attempts, last_exception)
    raise RuntimeError("Retry helper exhausted without result or failure message.")


async def call_claude_vision(image_path_or_url: str, question: str) -> str:
    """Call Claude vision API."""
    request_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": None,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    try:
        if os.path.exists(image_path_or_url):  # Check if the file exists locally
            with open(image_path_or_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                request_messages[0]["content"][0]["source"] = dict(
                    type="base64",
                    media_type=await guess_mime_media_type_from_extension(
                        image_path_or_url
                    ),
                    data=image_data,
                )
        elif is_sandbox_path(image_path_or_url):
            return "The visual_question_answering tool cannot access to sandbox file, please use the local path provided by original instruction"
        else:  # Otherwise, assume it's a URL
            # Convert to https URL for Claude vision API
            url = ensure_https_url(image_path_or_url)

            request_messages[0]["content"][0]["source"] = dict(type="url", url=url)

        async def _send_claude_request() -> str:
            client = Anthropic(
                api_key=ANTHROPIC_API_KEY,
                base_url=ANTHROPIC_BASE_URL,
            )
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4096,
                messages=request_messages,
            )
            answer_text = response.content[0].text

            # Check if response.text is None or empty after stripping
            if answer_text is None or answer_text.strip() == "":
                raise Exception("Response text is None or empty")  # noqa: TRY002

            return answer_text

        def _build_claude_failure_message(max_retries: int, error: Exception) -> str:
            return (
                "Visual Question Answering (Claude Client) failed after "
                f"{max_retries} retries: {error}\n"
            )

        max_retries = 5
        return await _async_retry_with_exponential_backoff(
            _send_claude_request,
            max_attempts=max_retries,
            failure_message_builder=_build_claude_failure_message,
        )

    except Exception as exc:  # noqa: BLE001
        return f"[ERROR]: Claude Error: {exc}"


async def call_openai_vision(image_path_or_url: str, question: str) -> str:
    """Call OpenAI vision API."""
    try:
        if os.path.exists(image_path_or_url):  # Check if the file exists locally
            with open(image_path_or_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                mime_type = await guess_mime_media_type_from_extension(
                    image_path_or_url
                )
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                }
        elif is_sandbox_path(image_path_or_url):
            return "The visual_question_answering tool cannot access to sandbox file, please use the local path provided by original instruction"
        else:  # Otherwise, assume it's a URL
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_path_or_url},
            }

        messages_for_llm = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    image_content,
                ],
            }
        ]

        async def _send_openrouter_vision_request() -> str:
            client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
            )

            response = client.chat.completions.create(
                model="google/gemini-2.5-pro",
                messages=messages_for_llm,
            )

            answer_text = response.choices[0].message.content

            if answer_text is None or answer_text.strip() == "":
                raise Exception("Response text is None or empty")  # noqa: TRY002

            return answer_text

        def _build_openrouter_failure_message(max_retries: int, error: Exception) -> str:
            return (
                "Visual Question Answering (Client) failed after "
                f"{max_retries} retries: {error}\n"
            )

        max_retries = 5
        return await _async_retry_with_exponential_backoff(
            _send_openrouter_vision_request,
            max_attempts=max_retries,
            failure_message_builder=_build_openrouter_failure_message,
        )

    except Exception as exc:  # noqa: BLE001
        return f"[ERROR]: OpenAI Error: {exc}"


async def call_gemini_vision(image_path_or_url: str, question: str) -> str:
    """Call Gemini vision API."""
    try:
        mime_type = await guess_mime_type_from_path(image_path_or_url)
        if os.path.exists(image_path_or_url):  # Check if the file exists locally
            with open(image_path_or_url, "rb") as image_file:
                image_data = image_file.read()
                image = types.Part.from_bytes(
                    data=image_data,
                    mime_type=mime_type,
                )
        elif is_sandbox_path(image_path_or_url):
            return "The visual_question_answering tool cannot access to sandbox file, please use the local path provided by original instruction"
        else:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # Simple retry for requests.get: 4 total attempts (1 initial + 3 retries)
            max_attempts = 4
            for attempt in range(max_attempts):
                try:
                    response = requests.get(image_path_or_url, headers=headers)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    image_data = response.content
                    break
                except Exception as e:
                    if attempt == max_attempts - 1:  # Last attempt
                        raise e
                    # Wait time: 5s, 15s, 60s for retries 1, 2, 3
                    wait_times = [5, 15, 60]
                    await asyncio.sleep(wait_times[attempt])

            image = types.Part.from_bytes(
                data=image_data,
                mime_type=mime_type,
            )
    except Exception as e:
        return f"[ERROR]: Failed to get image data {image_path_or_url}: {e}.\nNote: The visual_question_answering tool cannot access to sandbox file, please use the local path provided by original instruction or http url. If you are using http url, make sure it is an image file url."

    retry_count = 0
    max_retry = 3  # 3 retries with smart timing to avoid thundering herd
    while retry_count <= max_retry:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)

            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    image,
                    types.Part(text=question),
                ],
                # config=types.GenerateContentConfig(temperature=0.1),
            )

            # Check if response.text is None or empty after stripping
            if response.text is None or response.text.strip() == "":
                raise Exception("Response text is None or empty")

            return response.text

        except Exception as e:
            # Only retry for rate limit and server errors, or empty response
            if (
                "503" in str(e)
                or "429" in str(e)
                or "500" in str(e)
                or "Response text is None or empty" in str(e)
            ):
                retry_count += 1
                if retry_count > max_retry:
                    return f"[ERROR]: Gemini Error after {retry_count} retries: {e}"

                # Rate limit is per minute, spread 5 requests across different minute windows
                if retry_count == 1:
                    # First retry: wait 60-300 seconds to spread across 4 minute windows
                    wait_time = random.randint(60, 300)
                elif retry_count == 2:
                    # Second retry: wait 60-180 seconds to try different window
                    wait_time = random.randint(60, 180)
                else:
                    # Third retry: fixed 60 seconds - ensure crossing minute boundary
                    wait_time = 60

                await asyncio.sleep(wait_time)
            else:
                return f"[ERROR]: Gemini Error: {e}"


@mcp.tool()
async def visual_question_answering(image_path_or_url: str, question: str) -> str:
    """This tool is used to ask question about an image or a video and get the answer with Gemini vision language models. It also automatically performs OCR (text extraction) on the image for additional context.

    Args:
        image_path_or_url: The path of the image file locally or its URL. Files from sandbox are not supported.
        question: The question to ask about the image. This tool performs bad on reasoning-required questions.

    Returns:
        The concatenated answers from Gemini vision model, including both VQA responses and OCR results.
    """

    ocr_prompt = """You are a meticulous text extraction specialist. Your task is to carefully scan the entire image and extract ALL visible text with maximum accuracy.

IMPORTANT INSTRUCTIONS:
1. **Scan systematically** - Look at every corner, edge, and area of the image multiple times
2. **Extract ALL text** - Include headers, labels, captions, fine print, watermarks, signs, and any other text elements
3. **Preserve formatting** - Maintain line breaks, spacing, and text hierarchy as they appear
4. **Include numbers and symbols** - Extract all numerical values, symbols, and special characters
5. **Double-check your work** - Review the entire image again to ensure nothing was missed
6. **Describe any unclear, partially visible, or ambiguous text** - If any text is blurry, cut off, partly obscured, or otherwise difficult to read, **describe it as best as possible, even if you are unsure or cannot fully recognize it**.

Remember: Your extraction will be used by someone who cannot see the image themselves. Any possible guess, uncertainty, or ambiguity should be reported in words rather than left out, so that nothing is omitted or lost.

Return only the extracted text content, maintaining the original formatting and structure as much as possible. If there is no text in the image, respond with 'No text found'. If there are areas where text may exist but is unreadable or ambiguous, describe these as well."""

    ocr_text = await call_openai_vision(image_path_or_url, ocr_prompt)

    vqa_prompt = f"""You are a highly attentive visual analysis assistant. Your task is to carefully examine the image and provide a thorough, accurate answer to the question.

IMPORTANT INSTRUCTIONS:
1. **Look at the image multiple times** - Take your time to observe all details, objects, people, text, colors, spatial relationships, and any subtle elements
2. **Cross-reference with OCR data** - Carefully compare what you see visually with the extracted text to ensure consistency
3. **Think step-by-step** - Break down your analysis into logical steps before providing your final answer, especially for complex and multi-object recognition questions
4. **Consider multiple perspectives** - Look at the image from different angles and consider various interpretations, especially for multi-object recognition questions
5. **Double-check your observations** - Verify your initial impressions by looking again at specific areas, especially for complex and multi-object recognition questions
6. **Be precise and detailed** - Provide specific details rather than general observations
7. **Report all visible or possible content, even if uncertain or ambiguous** - If you notice anything that is blurry, partly obscured, difficult to recognize, or of uncertain importance, **describe it in words instead of omitting it**. Do not leave out any possible content, even if you are unsure.

Remember: Your analysis will be used by someone who cannot see the image themselves. Any possible guess, uncertainty, or ambiguity should be reported in words rather than left out, so that nothing is omitted or lost.
        
The OCR result of this image is as follows (may be incomplete or missing some text):
{ocr_text}

Question to answer: {question}

Please provide a comprehensive analysis that demonstrates careful observation and thoughtful reasoning, including any possible, uncertain, or ambiguous elements you notice.
If its an animal, follow the nomenclature rules as stated below: 

NOMENCLATURE RULES (critical):
- Use the **simplest common name** only. **Do NOT** include species/region qualifiers unless explicitly asked. eg. when it's an 'Atlantic puffin', simply proceed on with 'puffin'.
- Prefer the **broader category** when multiple species are plausible.
- If unsure of species, state the generic name and note uncertainty only if the question asks for species-level ID.
"""
    #  eg. when it's an Atlantic puffin, reply with just puffin.
    # Before answering, carefully analyze both the question and the image. Identify and briefly list potential subtle or easily overlooked VQA pitfalls or ambiguities that could arise in interpreting this question or image (e.g., confusing similar objects, missing small details, misreading text, ambiguous context, etc.). For each, suggest a method or strategy to avoid or mitigate these issues. Only after this analysis, proceed to answer the question, providing a thorough and detailed observation and reasoning process.

    vqa_answer = await call_openai_vision(image_path_or_url, vqa_prompt)
    logger.info("Visual Question Answering tool called via OpenRouter (Gemini model)")
    logger.info(f"OCR results: {ocr_text}")
    logger.info(f"VQA results: {vqa_answer}")
    return f"OCR results:\n{ocr_text}\n\nVQA result:\n{vqa_answer}"


# The tool visual_audio_youtube_analyzing only support single YouTube URL as input for now, though GEMINI can support multiple URLs up to 10 per request.
@mcp.tool()
async def visual_audio_youtube_analyzing(
    url: str, question: str = "", provide_transcribe: bool = False
) -> str:
    """Analyzes public YouTube video audiovisual content to answer questions or provide transcriptions. This tool processes both audio tracks and visual frames from YouTube videos. This tool could be primarily used when analyzing YouTube video content. Only supports YouTube Video URLs containing youtube.com/watch, youtube.com/shorts, or youtube.com/live for now.

    Args:
        url: The YouTube video URL.
        question: The specific question about the video content. Use timestamp format MM:SS or MM:SS-MM:SS if needed to specify a specific time (e.g., 01:45, 03:20-03:45). Leave empty if only requesting transcription.
        provide_transcribe: When set to true, returns a complete timestamped transcription of both spoken content and visual elements throughout the video.

    Returns:
        The answer to the question or the transcription of the video.
    """
    if OPENROUTER_API_KEY == "":
        return "[ERROR]: OPENROUTER_API_KEY is not set, visual_audio_youtube_analyzing tool is not available."

    if (
        "youtube.com/watch" not in url
        and "youtube.com/shorts" not in url
        and "youtube.com/live" not in url
    ):
        return f"[ERROR]: Invalid URL: '{url}'. YouTube Video URL must contain youtube.com/watch, youtube.com/shorts, or youtube.com/live"

    if question == "" and not provide_transcribe:
        return "[ERROR]: You must provide a question to ask about the video content or set provide_transcribe to True."

    # client = genai.Client(api_key=GEMINI_API_KEY)
    client = OpenAI(
                    api_key=OPENROUTER_API_KEY,
                    base_url=OPENROUTER_BASE_URL,
                )

            
    logger.info("Visual Audio YouTube Analyzing tool called via OpenRouter (Gemini model)")
    logger.info(f"URL: {url}")

    if provide_transcribe:
        # prompt from GEMINI official document
        prompt = "Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions."
        retry_count = 0
        max_retry = 3  # 3 retries with smart timing to avoid thundering herd
        while retry_count <= max_retry:
            try:
                # transcribe_response = client.models.generate_content(
                #     model="gemini-3-pro-preview",
                #     contents=types.Content(
                #         parts=[
                #             types.Part(file_data=types.FileData(file_uri=url)),
                #             types.Part(text=prompt),
                #         ]
                #     ),
                # )
                response = client.chat.completions.create(
                    model="google/gemini-2.5-pro",
                    messages=[
                            {
                                "role": "user",
                                "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "video_url",
                                    "video_url": {
                                    "url": url
                                    }
                                }
                                ]
                        }
                    ]
                )

                # Check if response.text is None or empty after stripping
                transcribe_response = response.choices[0].message.content
                if (
                    transcribe_response is None
                    or transcribe_response.strip() == ""
                ):
                    logger.error(f"Response text is None or empty for URL: {url}")
                    raise Exception("Response text is None or empty")

                transcription_output = (
                    "Transcription:\n\n" + transcribe_response + "\n\n"
                )
                break
            except Exception as e:
                # Handle 400 error specifically for video length issues
                if "exceeds the maximum number of tokens" in str(e):
                    transcription_output = f"[ERROR]: Failed to transcribe the video: {str(e)}. This is due to the video being too long to process."
                    break
                # Only 503 error need to retry, or empty response
                elif (
                    "400" in str(e)
                    or "503" in str(e)
                    or "429" in str(e)
                    or "500" in str(e)
                    or "Response text is None or empty" in str(e)
                ):
                    retry_count += 1
                    if retry_count > max_retry:
                        transcription_output = f"[ERROR]: Failed to transcribe the video after {retry_count} retries: {str(e)}"
                        break

                    # Rate limit is per minute, spread 5 requests across different minute windows
                    if retry_count == 1:
                        # First retry: wait 60-300 seconds to spread across 4 minute windows
                        wait_time = random.randint(60, 300)
                    elif retry_count == 2:
                        # Second retry: wait 60-180 seconds to try different window
                        wait_time = random.randint(60, 180)
                    else:
                        # Third retry: fixed 60 seconds - ensure crossing minute boundary
                        wait_time = 60

                    await asyncio.sleep(wait_time)
                else:
                    transcription_output = (
                        f"[ERROR]: Failed to transcribe the video: {str(e)}"
                    )
                    break
    else:
        transcription_output = ""

    if question != "":
        prompt = f"Answer the following question: {question}"
        retry_count = 0
        max_retry = 3  # 3 retries with smart timing to avoid thundering herd
        while retry_count <= max_retry:
            try:
                # response = client.models.generate_content(
                #     model="gemini-3-pro-preview",
                #     contents=types.Content(
                #         parts=[
                #             types.Part(file_data=types.FileData(file_uri=url)),
                #             types.Part(text=prompt),
                #         ]
                #     ),
                # )
                response = client.chat.completions.create(
                    model="google/gemini-2.5-pro",
                    messages=[
                            {
                                "role": "user",
                                "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "video_url",
                                    "video_url": {
                                    "url": url
                                    }
                                }
                                ]
                        }
                    ]
                )
                response_txt = response.choices[0].message.content
                # Check if response.text is None or empty after stripping
                if response_txt is None or response_txt.strip() == "":
                    raise Exception("Response text is None or empty")

                answer_output = (
                    "Answer of the question: "
                    + question
                    + "\n\n"
                    + response_txt
                    + "\n\n"
                )
                break
            except Exception as e:
                # Handle 400 error specifically for video length issues
                if "exceeds the maximum number of tokens" in str(e):
                    transcription_output = f"[ERROR]: Failed to transcribe the video: {str(e)}. This is due to the video being too long to process."
                    break
                # Only 503 error need to retry, or empty response
                elif (
                    "400" in str(e)
                    or "503" in str(e)
                    or "429" in str(e)
                    or "500" in str(e)
                    or "Response text is None or empty" in str(e)
                ):
                    retry_count += 1
                    if retry_count > max_retry:
                        answer_output = f"[ERROR]: Failed to answer the question after {retry_count} retries: {str(e)}"
                        break

                    # Rate limit is per minute, spread 5 requests across different minute windows
                    if retry_count == 1:
                        # First retry: wait 60-300 seconds to spread across 4 minute windows
                        wait_time = random.randint(60, 300)
                    elif retry_count == 2:
                        # Second retry: wait 60-180 seconds to try different window
                        wait_time = random.randint(60, 180)
                    else:
                        # Third retry: fixed 60 seconds - ensure crossing minute boundary
                        wait_time = 60

                    await asyncio.sleep(wait_time)
                else:
                    answer_output = f"[ERROR]: Failed to answer the question: {str(e)}"
                    break
    else:
        answer_output = ""

    hint = "\n\nHint: Large videos may trigger rate limits causing failures. If you need more website information rather than video visual content itself (such as video subtitles, titles, descriptions, key moments), you can also call tool `scrape_website` tool."
    logger.info(f"Transcription output: {transcription_output}")
    logger.info(f"Answer output: {answer_output}")
    return transcription_output + answer_output + hint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vision MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport method: 'stdio' or 'sse' (default: sse)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to use when running with SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8932,
        help="Port to use when running with SSE transport (default: 8932)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)