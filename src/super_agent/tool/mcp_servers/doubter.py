import asyncio
import os
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv(verbose=True)

import google.generativeai as genai
from fastmcp import FastMCP
from openai import AsyncOpenAI

BOUNDARY = 7
DOUBTER_MODEL = os.getenv("DOUBTER_MODEL", "anthropic/claude-sonnet-4.5")
DOUBTER_SCORE_MODEL = os.getenv("DOUBTER_SCORE_MODEL", "google/gemini-2.5-pro")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

INSTRUCTION = dedent(
    """\
You should first analyze this flow of actions taken by the agent.
As you are a naturally doubtful tool, try to spot out places within the flow of actions where there might be a logical flow or a missing step.
If there are any flaws within the flow of action, I need you to tell me that I need to run the agent again and suggest me an improvement I can make to the flow of actions to be taken by the agent.
Not all tasks will have flaws, some might be perfect in which case you should suggest no improvements.
But do keep in mind that the rest might not be, and you need to discern out these flawed sequence of actions.
The history shown to you is all the action taken so far right before the verification step which is going to happen now. The key mistake in the code is not that the verification step hasn't run yet even though the previous steps show that the verification step haven't run yet. That's because you are going to run now. SO DONT POINT OUT THAT AS AN ERROR.

Here is the history of the flow of actions:\n
"""
)

SCORE_INSTRUCTION = dedent(
    f"""\
You will be given a reflection of the past actions taken by the agent by another LLM.
I need you to look at the reflection done by this LLM and give an overall score from 0 to 10.
The score reflects the past actions taken by the original agent. If you give it a score of 0, it means the past actions taken were illogical and full of errors and flaws, requiring a reset of the plans and another run by the agent.
A score of 10 means the past actions were perfect, logically sound and robust enough that the agent does not need to perform it again.
After deciding on the score, give me a decision whether to run the agent again one more time. The boundary score of whether to rerun is {BOUNDARY}.
Any score lower than this requires the agent to run again.
Any score higher or equal to this does not require the agent to run again as it means the past actions were logically sound enough.

Give me a score, the respective decision, and the reasoning behind that decision, after considering the reflection of the past actions shown below:\n
Return STRICT JSON only (no extra text):
{{
  "score": <number 0-10>,
  "rerun": <true|false>,
  "reason": "<short reasoning>",
  "improvement": "<concrete improvement to the plan if rerun>"
}}
Reflection text:\n
"""
)

mcp = FastMCP("doubter-mcp-server")

def _build_chat_client() -> AsyncOpenAI | None:
    """Select the chat completion backend based on configured credentials."""
    if OPENROUTER_API_KEY:
        headers: dict[str, str] = {}
        if OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        if OPENROUTER_APP_NAME:
            headers["X-Title"] = OPENROUTER_APP_NAME
        client_kwargs: dict[str, object] = {
            "api_key": OPENROUTER_API_KEY,
            "base_url": OPENROUTER_BASE_URL,
            "timeout": 60,
        }
        if headers:
            client_kwargs["default_headers"] = headers
        return AsyncOpenAI(**client_kwargs)
    if OPENAI_API_KEY:
        return AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=60,
        )
    return None


_chat_client = _build_chat_client()


def _extract_gemini_text(response) -> str:
    if hasattr(response, "text") and response.text:
        return response.text
    candidates = getattr(response, "candidates", []) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    if parts:
        return "\n".join(parts)
    raise RuntimeError("Gemini response contained no text")


async def _call_chat_model(model: str, prompt: str, max_retries: int = 3) -> str:
    if _chat_client is None:
        raise RuntimeError(
            "No OpenAI-compatible API key configured. Provide OPENROUTER_API_KEY or OPENAI_API_KEY."
        )

    for attempt in range(1, max_retries + 1):
        try:
            response = await _chat_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            if not response or not response.choices:
                raise RuntimeError("Empty response from OpenAI")
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if not content:
                raise RuntimeError("Response contained no message content")
            return content
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Model call failed after {attempt} attempts: {exc}"
                ) from exc
            await asyncio.sleep(min(2 ** attempt, 8))


async def _call_gemini_model(model: str, prompt: str, max_retries: int = 3) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                lambda: genai.GenerativeModel(model).generate_content(prompt)
            )
            return _extract_gemini_text(response)
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Gemini model call failed after {attempt} attempts: {exc}"
                ) from exc
            await asyncio.sleep(min(2 ** attempt, 8))


async def _call_model(model: str, prompt: str, max_retries: int = 3) -> str:
    model_lower = model.lower()
    if model_lower.startswith("gemini"):
        return await _call_gemini_model(model, prompt, max_retries)
    return await _call_chat_model(model, prompt, max_retries)


@mcp.tool()
async def doubter(history: str) -> str:
    """Critically review the agent's action history and decide if a rerun is needed."""
    if not history or not history.strip():
        return "[ERROR]: 'history' is required."

    history = history.strip()

    try:
        critique = await _call_model(DOUBTER_MODEL, INSTRUCTION + history)
        score_and_decision = await _call_model(
            DOUBTER_SCORE_MODEL, SCORE_INSTRUCTION + critique
        )
    except RuntimeError as exc:
        return f"[ERROR]: {exc}"

    return (
        "The history of the agent's actions:\n"
        f"{history}\n\n"
        "Doubter review:\n"
        f"{critique}\n\n"
        "Score & decision:\n"
        f"{score_and_decision}\n"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
