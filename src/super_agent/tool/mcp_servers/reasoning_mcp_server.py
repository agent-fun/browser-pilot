import os
from dotenv import load_dotenv
load_dotenv(verbose=True)

from anthropic import Anthropic
from fastmcp import FastMCP
from openai import OpenAI
import asyncio

from src.super_agent.tool.logger import bootstrap_logger

ANTHROPIC_API_KEY =  os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL =  os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
OPENROUTER_API_KEY =  os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL =  os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = "anthropic/claude-3-7-sonnet:thinking"
ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"
REQUEST_TIMEOUT = 30

# Initialize FastMCP server
mcp = FastMCP("reasoning-mcp-server")

# Initialize logger 
logger = bootstrap_logger()

async def _retry_with_backoff(max_attempts: int, coro_factory):
    """Execute a coroutine factory with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as error:
            if attempt >= max_attempts:
                raise error
            await asyncio.sleep(5 * (2**attempt))


async def _call_openrouter(messages_for_llm):
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    def _do_call():
        return client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages_for_llm,
            extra_body={},
            timeout=REQUEST_TIMEOUT,
        )

    response = await _retry_with_backoff(5, lambda: asyncio.to_thread(_do_call))
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty response from reasoning (OpenRouter)")
    return content


async def _call_anthropic(messages_for_llm):
    client = Anthropic(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL)

    def _do_call():
        return client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=21000,
            thinking={
                "type": "enabled",
                "budget_tokens": 19000,
            },
            messages=messages_for_llm,
            stream=False,
            timeout=REQUEST_TIMEOUT,
        )

    response = await _retry_with_backoff(5, lambda: asyncio.to_thread(_do_call))
    content = response.content[-1].text
    if not content or not content.strip():
        raise ValueError("Empty response from reasoning (Anthropic)")
    return content


@mcp.tool()
async def reasoning(question: str) -> str:
    """You can use this tool use solve hard math problem, puzzle, riddle and IQ test question that requries a lot of chain of thought efforts.
    DO NOT use this tool for simple and obvious question.

    Args:
        question: The complex question or problem requiring step-by-step reasoning. Should include all relevant information needed to solve the problem..

    Returns:
        The answer to the question.
    """

    messages_for_llm = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question,
                }
            ],
        }
    ]

    if not question or not question.strip():
        logger.error(f"Question cannot be empty for reasoning")
        return "[ERROR]: Question cannot be empty for reasoning."

    if not OPENROUTER_API_KEY and not ANTHROPIC_API_KEY:
        logger.error(f"No API key configured for reasoning (need OPENROUTER_API_KEY or ANTHROPIC_API_KEY).")
        return "[ERROR]: No API key configured for reasoning (need OPENROUTER_API_KEY or ANTHROPIC_API_KEY)."

    if OPENROUTER_API_KEY:
        try:
            logger.info(f"Calling reasoning (OpenRouter Client) with question: {question}")
            return await _call_openrouter(messages_for_llm)
        except Exception as e:
            return f"[ERROR]: Reasoning (OpenRouter Client) failed: {e}\n"

    if ANTHROPIC_API_KEY:
        try:
            logger.info(f"Calling reasoning (Anthropic Client) with question: {question}")
            return await _call_anthropic(messages_for_llm)
        except Exception as e:
            return f"[ERROR]: Reasoning (Anthropic Client) failed: {e}\n"
            


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reasoning MCP Server")
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
        default=8934,
        help="Port to use when running with SSE transport (default: 8934)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)
