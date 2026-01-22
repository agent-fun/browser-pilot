import os
from dotenv import load_dotenv
from fastmcp import FastMCP

# 复用你们项目里现有的 browse() 实现
from src.super_agent.tool.mcp_servers.auto_browser import browse

load_dotenv(verbose=True)

mcp = FastMCP("browser-use-server")


@mcp.tool()
async def auto_browser_use(task: str) -> str:
    """Automate a browser to complete a high-level task."""
    if not task or not task.strip():
        return "[ERROR]: 'task' is required."

    # 你原逻辑：按 model family 取 key
    model_id = os.getenv("BROWSER_USE_MODEL_ID", "claude").strip().lower()
    assert model_id in ["gemini", "gpt", "claude"], f"Invalid model_id: {model_id}"

    if model_id == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    else:
        # gpt/claude 都走 openrouter
        api_key = os.getenv("OPENROUTER_API_KEY")

    try:
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

    parser = argparse.ArgumentParser("Minimal Browser Use MCP Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8930)
    args = parser.parse_args()

    # SSE 启动
    mcp.run(transport="sse", host=args.host, port=args.port)
