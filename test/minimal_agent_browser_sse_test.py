import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ====== 让 repo root & examples 可 import（按你原来的方式）======
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # 你可以按实际放置位置调整
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
for p in [REPO_ROOT, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ====== 你们框架依赖 ======
from src.super_agent.agent.super_react_agent import SuperReActAgent
from src.super_agent.agent.super_config import SuperAgentFactory
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.param import Param
from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig
from openjiuwen.core.runner.runner import Runner, resource_mgr

# ====== 配置：SSE URL ======
AUTO_BROWSER_SSE_URL = os.getenv("AUTO_BROWSER_SSE_URL", "http://127.0.0.1:8930/sse").strip()
SERVER_NAME = "browser-use-server"

# ====== LLM 配置（你们 SuperReActAgent 用于“决定何时调用工具”）=====
API_BASE = os.getenv("API_BASE", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic/claude-3.7-sonnet")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")


def create_model_config() -> ModelConfig:
    model_info = BaseModelInfo(
        api_key=API_KEY,
        api_base=API_BASE,
        model=MODEL_NAME,
        timeout=60,
    )
    return ModelConfig(model_provider=MODEL_PROVIDER, model_info=model_info)


async def register_browser_tool_as_localfunction() -> LocalFunction:
    """注册 SSE MCP server，并包装 auto_browser_use 成 LocalFunction。"""
    tool_mgr = resource_mgr.tool()

    ok_list = await tool_mgr.add_tool_servers([
        ToolServerConfig(
            server_name=SERVER_NAME,
            client_type="sse",
            params=AUTO_BROWSER_SSE_URL,
        )
    ])
    if not ok_list or not ok_list[0]:
        raise RuntimeError(f"Failed to add MCP server: {SERVER_NAME}")

    async def _call_auto_browser_use(task: str):
        # MCP tool id = "{server}.{tool}"
        tool_id = f"{SERVER_NAME}.auto_browser_use"
        result = await Runner.run_tool(tool_id, {"task": task})
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result

    return LocalFunction(
        name="auto_browser_use",
        description="Use a real browser (Playwright) to complete a web task and return extracted text.",
        params=[Param(name="task", description="High-level browser task", param_type="string", required=True)],
        func=_call_auto_browser_use,
    )


async def main():
    # 1) 启动 Runner
    await Runner.start()

    # 2) 注册 browser tool
    browser_tool = await register_browser_tool_as_localfunction()

    # 3) 创建最小 agent config
    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="minimal_browser_agent",
        agent_type="main",
        agent_version="1.0",
        description="Minimal agent with only auto_browser_use tool (SSE MCP).",
        model=create_model_config(),
        prompt_template=[
            {
                "role": "system",
                "content": (
                    "You are a minimal browsing agent.\n"
                    "When you need to use the web, call the tool `auto_browser_use` with a clear step-by-step task.\n"
                    "After you get tool output, answer the user succinctly.\n"
                    f"Current time: {datetime.now().isoformat()}\n"
                ),
            }
        ],
        max_iteration=6,
        max_tool_calls_per_turn=1,
        enable_o3_hints=False,
        enable_o3_final_answer=False,
        enable_todo_plan=False,
    )

    agent = SuperReActAgent(agent_config=agent_config, tools=[browser_tool], workflows=None)

    # 4) Query 测试（你可以加更多）
    queries = [
        # "打开 https://example.com ，提取页面标题(title)和页面内第一个 H1 文本。",
        # "打开 https://www.iana.org/domains/reserved ，在页面中找到 example.com 的说明并返回对应段落。",
        "6001 Lancaster Ave in Philadelphia, Pennsylvania, used to be home to a restaurant called \"Zeke's Mainline BBQ.\"  According to Google Street View, and prior to 2023, what color was the word \"BBQ\" on the building's sign? Express your answer in all caps."
    ]

    for i, q in enumerate(queries, 1):
        print("\n" + "=" * 80)
        print(f"[TEST #{i}] Query:\n{q}")
        result = await agent.invoke({"query": q})
        print(f"\n[TEST #{i}] Agent result keys: {list(result.keys())}")
        print(f"\n[TEST #{i}] Output:\n{result.get('output', result)}")

    # 5) 停止 Runner
    await Runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
