import asyncio
import logging
import sys
from fastmcp import FastMCP
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters  # (already imported in config.py)
# import your existing tool class
from src.super_agent.tool.mcp_servers.auto_browser import *
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)
from src.super_agent.tool.browser.utils.logger import logger as browser_logger

mcp = FastMCP("browser-use-server")


@mcp.tool()
async def auto_browser_use(task: str) -> str:
    """Automate a browser to complete a high-level task.

    Args:
        task: Description of the overall task of what to do in the browser.

    Returns:
        Extracted content as a plain string.
    """
    if not task or not task.strip():
        return "[ERROR]: 'task' is required."

    try:
        # Change the model you wish to use
        # gemini - for gemini family
        # gpt - for gpt family
        # claude - for claude family

        model_id = "claude"  # Change this to the desired model family

        assert model_id in ["gemini", "gpt", "claude"], f"Model ID: {model_id} not in authorized model_id family"
        if model_id == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
        elif model_id == "gpt":
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif model_id == "claude":
            api_key = os.getenv("OPENROUTER_API_KEY")

        result = await browse(
            model_id=model_id,
            api_key=api_key,
            task=task,
        )
        return result

    except Exception as e:
        return f"[ERROR]: Exception {e} occurred in auto_browser_use."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Browser Use MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport method: 'stdio' or 'sse' (default: stdio)",
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
        default=8930,
        help="Port to use when running with SSE transport (default: 8930)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)
