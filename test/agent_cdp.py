import os
import sys
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import re
import hashlib

load_dotenv()

# # ====== 让 repo root & examples 可 import======
# CURRENT_DIR = os.path.dirname(__file__)
# REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) 
# EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
# for p in [REPO_ROOT, EXAMPLES_DIR]:
#     if p not in sys.path:
#         sys.path.insert(0, p)
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR            # ✅ agent_cdp.py 在 repo 根目录时
EXAMPLES_DIR = REPO_ROOT / "src"

for p in (str(REPO_ROOT), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.super_agent.agent.super_react_agent import SuperReActAgent
from src.super_agent.agent.super_config import SuperAgentFactory
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.param import Param
from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig
from openjiuwen.core.runner.runner import Runner, resource_mgr

# ====== 配置：SSE URL & server name（要和 ToolServerConfig.server_name 一致） ======
AUTO_BROWSER_SSE_URL = os.getenv("AUTO_BROWSER_SSE_URL", "http://127.0.0.1:8930/sse").strip()
SERVER_NAME = os.getenv("BROWSER_MCP_SERVER_NAME", "browser-use-cdp-server").strip()


API_BASE = os.getenv("API_BASE", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic/claude-sonnet-4.5")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")

def make_tool_server_config(server_name: str, client_type: str, params):
    """
    兼容新版本 ToolServerConfig:
    - server_path 必填
    - params 必须 dict
    """
    if client_type == "sse":
        # 你传进来通常是 url(str)
        if isinstance(params, str):
            server_path = params
            server_params = {}
        elif isinstance(params, dict):
            # 兼容：你也可以传 {"server_path": url, "params": {...}}
            server_path = params.get("server_path") or params.get("url")
            inner = params.get("params")
            server_params = inner if isinstance(inner, dict) else {
                k: v for k, v in params.items() if k not in ("server_path", "url", "params")
            }
        else:
            raise TypeError(f"SSE params must be str or dict, got {type(params)}")

        return ToolServerConfig(
            server_name=server_name,
            server_path=server_path,
            client_type="sse",
            params=server_params,
        )

    # 你这个脚本里暂时只用 sse，这里先留扩展位
    return ToolServerConfig(
        server_name=server_name,
        server_path=str(client_type),
        client_type=client_type,
        params=params if isinstance(params, dict) else {"raw": params},
    )

_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

def sanitize_tool_name(raw: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "tool"
    if len(name) > 128:
        h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        name = name[:119] + "_" + h
    return name

async def wrap_tools_with_safe_names(tools: list[LocalFunction]) -> list[LocalFunction]:
    """
    让 LLM 看到的 tool.name 合法（不含 . 等），但实际执行仍调用原始 MCP tool_id（raw_tool.name）
    """
    wrapped: list[LocalFunction] = []
    used = set()

    for raw_tool in tools:
        raw_tool_id = getattr(raw_tool, "name", "")  # e.g. "browser-use-cdp-server.auto_browser_use"
        safe_name = sanitize_tool_name(raw_tool_id)  # e.g. "browser-use-cdp-server_auto_browser_use"

        # 防止 sanitize 后重名
        if safe_name in used:
            h = hashlib.md5(raw_tool_id.encode("utf-8")).hexdigest()[:6]
            safe_name = sanitize_tool_name(f"{safe_name}_{h}")
        used.add(safe_name)

        if not _TOOL_NAME_RE.match(safe_name):
            raise ValueError(f"Bad tool name: raw={raw_tool_id} safe={safe_name}")

        async def _wrapper_factory(_raw_tool_id: str):
            async def _wrapper(**kwargs):
                result = await Runner.run_tool(_raw_tool_id, kwargs)
                if isinstance(result, dict) and "result" in result:
                    return result["result"]
                return result
            return _wrapper

        wrapped_tool = LocalFunction(
            name=safe_name,
            description=getattr(raw_tool, "description", "") or f"Wrapped MCP tool {raw_tool_id}",
            params=getattr(raw_tool, "params", None),
            func=await _wrapper_factory(raw_tool_id),
        )

        # 可选：记录映射方便 debug
        wrapped_tool._raw_tool_id = raw_tool_id
        wrapped.append(wrapped_tool)

    return wrapped


def create_model_config() -> ModelConfig:
    model_info = BaseModelInfo(
        api_key=API_KEY,
        api_base=API_BASE,
        model=MODEL_NAME,
        timeout=60,
    )
    return ModelConfig(model_provider=MODEL_PROVIDER, model_info=model_info)


# -----------------------------
# MCP 工具封装：把 server 上所有 tools 自动注册为 LocalFunction
# -----------------------------
def _jsonschema_type_to_param_type(t: str) -> str:
    # openjiuwen 的 Param.param_type 你们常用 string/integer/boolean
    if t in {"string", "integer", "number", "boolean", "object", "array"}:
        return t
    return "string"


async def _runner_tool_call(tool_id: str, args: dict):
    """
    Runner.run_tool 返回一般是 {"result": "..."}。
    这里统一拆出来；如果不是 dict，就原样返回。
    """
    result = await Runner.run_tool(tool_id, args)
    if isinstance(result, dict) and "result" in result:
        return result["result"]
    return result


async def register_mcp_server_and_wrap_all_tools() -> list[LocalFunction]:
    """
    1) add_tool_servers 注册 SSE MCP server
    2) Runner.list_tools(server_name) 拿 schema
    3) 自动生成每个 tool 的 LocalFunction wrapper
    """
    tool_mgr = resource_mgr.tool()

    ok_list = await tool_mgr.add_tool_servers([
        # ToolServerConfig(
        #     server_name=SERVER_NAME,
        #     client_type="sse",
        #     params=AUTO_BROWSER_SSE_URL,
        # )
        make_tool_server_config(SERVER_NAME, "sse", AUTO_BROWSER_SSE_URL)
    ])
    if not ok_list or not ok_list[0]:
        raise RuntimeError(f"Failed to add MCP server: {SERVER_NAME} @ {AUTO_BROWSER_SSE_URL}")

    tool_infos = await Runner.list_tools(SERVER_NAME)  # -> list[McpToolInfo]
    if not tool_infos:
        raise RuntimeError(
            f"Runner.list_tools('{SERVER_NAME}') returned empty/None. "
            f"Check MCP server tool metadata (especially tool description must be string)."
        )
    local_tools: list[LocalFunction] = []

    for info in tool_infos:
        # 注意：安装版 openjiuwen 使用 input_schema，而不是 schema（schema 是 Pydantic 方法）
        schema = getattr(info, "input_schema", None) or getattr(info, "inputSchema", None) or {}
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        params_def: list[Param] = []
        for pname, pinfo in props.items():
            ptype = _jsonschema_type_to_param_type(pinfo.get("type", "string"))
            params_def.append(
                Param(
                    name=pname,
                    description=pinfo.get("description", ""),
                    param_type=ptype,
                    required=pname in required,
                )
            )

        tool_name = info.name
        tool_desc = getattr(info, "description", "") or f"MCP tool {tool_name} from {SERVER_NAME}"

        async def _make_call(_tool_name: str):
            async def _call(**kwargs):
                tool_id = f"{SERVER_NAME}.{_tool_name}"
                return await _runner_tool_call(tool_id, kwargs)
            return _call

        async_func = await _make_call(tool_name)

        local_tools.append(
            LocalFunction(
                name=tool_name,
                description=tool_desc,
                params=params_def,
                func=async_func,
            )
        )

    return local_tools


# -----------------------------
# 额外提供一个“高层一键 QA”工具：browser_quick_qa
# -----------------------------
def build_browser_quick_qa_tool() -> LocalFunction:
    """
    对 agent 来说一次调用搞定：
      browser_start -> session_new(url) -> qa(question) -> session_close(session_id)
    优点：你保持 max_tool_calls_per_turn=1 也能做网页 QA。
    """

    async def browser_quick_qa(url: str, question: str, use_current_tab: bool = False) -> str:
        # 1) start (idempotent)
        _ = await _runner_tool_call(f"{SERVER_NAME}.browser_start", {})

        # 2) new session
        raw = await _runner_tool_call(
            f"{SERVER_NAME}.session_new",
            {"url": url, "use_current_tab": use_current_tab},
        )

        # MCP server 返回的是 json 字符串（我给你的 server 是 json.dumps）
        # 这里做一次 parse，拿到 session_id
        session_id = None
        try:
            sess = json.loads(raw) if isinstance(raw, str) else raw
            session_id = sess.get("session_id")
        except Exception:
            session_id = None

        # 3) QA
        qa_raw = await _runner_tool_call(
            f"{SERVER_NAME}.qa",
            {"question": question, "session_id": session_id},
        )

        # 4) close (只关闭自己创建的 tab；如果 use_current_tab=True 且 external tab，默认不会关)
        try:
            _ = await _runner_tool_call(
                f"{SERVER_NAME}.session_close",
                {"session_id": session_id, "force": False},
            )
        except Exception:
            pass

        return qa_raw

    return LocalFunction(
        name="browser_quick_qa",
        description=(
            "Open a webpage and answer a question using the page content. "
            "Internally: browser_start -> session_new(url) -> qa(question) -> session_close."
        ),
        params=[
            Param(name="url", description="Webpage URL to open", param_type="string", required=True),
            Param(name="question", description="Question to answer from the page", param_type="string", required=True),
            Param(
                name="use_current_tab",
                description="Attach to current tab (CDP). Safer default is false (new tab).",
                param_type="boolean",
                required=False,
            ),
        ],
        func=browser_quick_qa,
    )

async def register_browser_run_task_tool() -> LocalFunction:
    tool_mgr = resource_mgr.tool()
    ok_list = await tool_mgr.add_tool_servers([make_tool_server_config(SERVER_NAME, "sse", AUTO_BROWSER_SSE_URL)])
    if not ok_list or not ok_list[0]:
        raise RuntimeError(f"Failed to add MCP server: {SERVER_NAME} @ {AUTO_BROWSER_SSE_URL}")

    async def _call(task: str) -> str:
        # ✅ 一次调用：server 端用 browser_use.Agent 跑完整长链路
        tool_id = f"{SERVER_NAME}.run_task"
        result = await Runner.run_tool(tool_id, {"task": task, "use_current_tab": False, "close": True})
        text = result["result"] if isinstance(result, dict) and "result" in result else result
        return text

    return LocalFunction(
        name="browser_run_task",
        description=(
            "Run a complex multi-step browsing task in ONE call. "
            "Supports clicks/drag/screenshot/Q&A. Prefer this tool for any web interaction."
        ),
        params=[Param(name="task", description="High-level browsing task", param_type="string", required=True)],
        func=_call,
    )

async def main():
    # 1) 启动 Runner
    await Runner.start()

    try:
        # 2) 注册 MCP server & 自动包装所有 tools
        mcp_tools = await register_mcp_server_and_wrap_all_tools()
        mcp_tools = await wrap_tools_with_safe_names(mcp_tools) 
        # 3) 额外加一个 high-level 便捷工具
        quick_qa_tool = build_browser_quick_qa_tool()
        
        browser_tool = await register_browser_run_task_tool()

        # 4) 创建 agent config
        agent_config = SuperAgentFactory.create_main_agent_config(
            agent_id="minimal_browser_agent_new_mcp",
            agent_type="main",
            agent_version="1.0",
            description="Minimal agent with browser-use CDP MCP tools (SSE).",
            model=create_model_config(),
            prompt_template=[
                {
                    "role": "system",
                    "content": (
                        "You are a browsing agent.\n"
                        "When the user needs ANY web interaction (open pages, click, drag, screenshot, extract info), "
                        "CALL THE TOOL `browser_run_task(task)` EXACTLY ONCE.\n"
                        "Write a clear step-by-step task in the tool input, including URLs.\n"
                        "After tool returns, answer the user succinctly.\n"
                        f"Current time: {datetime.now().isoformat()}\n"
                    ),
                }
            ],
            max_iteration=8,
            max_tool_calls_per_turn=1,  # ✅ 你可以保持 1，因为我们提供了 browser_quick_qa
            enable_o3_hints=False,
            enable_o3_final_answer=False,
            enable_todo_plan=False,
        )

        # 5) 实例化 agent（工具 = MCP tools + quick tool）
        tools = [quick_qa_tool] + mcp_tools
        # agent = SuperReActAgent(agent_config=agent_config, tools=tools, workflows=None)
        agent = SuperReActAgent(agent_config=agent_config, tools=[browser_tool], workflows=None)
        # 6) Query 测试
        queries = [
            # 简单：直接网页 QA（推荐走 browser_quick_qa）
            "请打开 https://example.com ，并回答：页面标题是什么？同时给出页面内 H1 文本。",
            # # 更明确：也可以让 agent 显式使用 quick_qa
            # "Use browser_quick_qa to open https://www.iana.org/domains/reserved and answer: "
            # "What does the page say about example.com? Provide a short evidence snippet.",
            # # 复杂任务：如果想让 agent 分步操作，也能做到，但在 max_tool_calls_per_turn=1 下会慢
            # # 建议之后把 max_tool_calls_per_turn 提到 2~3，或在 MCP server 里加一个 run_task(task) 的高层工具
            # "Open https://example.com, take a screenshot, then answer: what is the main message on the page?"
        ]

        for i, q in enumerate(queries, 1):
            print("\n" + "=" * 80)
            print(f"[TEST #{i}] Query:\n{q}")
            result = await agent.invoke({"query": q})
            print(f"\n[TEST #{i}] Agent result keys: {list(result.keys())}")
            print(f"\n[TEST #{i}] Output:\n{result.get('output', result)}")

    finally:
        # 7) 停止 Runner（Windows + SSE 可能偶发 cancel scope 报错，你原来已经见过）
        try:
            await Runner.stop()
        except RuntimeError as e:
            if "cancel scope" in str(e):
                print("[WARN] Ignore SSE shutdown RuntimeError:", e)
            else:
                raise


if __name__ == "__main__":
    async def _runner():
        try:
            await main()
        finally:
            # 给底层 anyio/httpx_sse 一点时间优雅退出
            await asyncio.sleep(0.1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_runner())
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

