#!/usr/bin/env python
# coding: utf-8

# python -m uvicorn examples.super_agent.api.server:app --host 127.0.0.1 --port 8000 --reload

"""
API runtime for agent_cdp (CDP browsing agent):
- init once: start Runner, register MCP SSE tools, build main agent
- multi-turn: bind ContextManager per session_id
- stream: SSE event generator
- non-stream: run_once
- reset/shutdown
"""

import asyncio
import base64
import json
import os
import re
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ====== Ensure repo root & examples importable (same style as agent_cdp.py) ======
from pathlib import Path as _Path
import sys as _sys

_CURRENT_DIR = _Path(__file__).resolve().parents[3]  # .../examples/super_agent/agent -> repo root
_REPO_ROOT = _CURRENT_DIR
_EXAMPLES_DIR = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_EXAMPLES_DIR)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from src.super_agent.agent.super_react_agent import SuperReActAgent
from src.super_agent.agent.super_config import SuperAgentFactory
from src.super_agent.agent.context_manager import ContextManager
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.param import Param
from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig
from openjiuwen.core.runner.runner import Runner, resource_mgr
from mcp import StdioServerParameters

# -----------------------------
# Env / Config (keep aligned with agent_cdp.py)
# -----------------------------
# Load .env file from super_agent directory
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

AUTO_BROWSER_SSE_URL = os.getenv("AUTO_BROWSER_SSE_URL", "http://127.0.0.1:8930/sse").strip()
SERVER_NAME = os.getenv("BROWSER_MCP_SERVER_NAME", "browser-use-cdp-server").strip()

API_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic/claude-sonnet-4.5").strip()
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter").strip()

# session artifacts dir (screenshots)
SESSION_DATA_DIR = Path(os.getenv("SUPER_AGENT_SESSION_DIR", "./.super_agent_sessions")).resolve()
SESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# MCP Tool Python environment (for stdio servers)
# -----------------------------
TOOLING_VENV_DIR = os.getenv(
    "MCP_TOOL_VENV_DIR",
    os.path.join(str(_REPO_ROOT), "src", "super_agent", "tool", ".venv-tool"),
)
if os.name == "nt":
    _tooling_python = os.path.join(TOOLING_VENV_DIR, "Scripts", "python.exe")
else:
    _tooling_python = os.path.join(TOOLING_VENV_DIR, "bin", "python")

MCP_TOOL_PYTHON = _tooling_python if os.path.exists(_tooling_python) else _sys.executable
MCP_TOOL_CWD = str(_REPO_ROOT)

_pythonpath_entries = [str(_REPO_ROOT), str(_EXAMPLES_DIR)]
_existing_pythonpath = os.getenv("PYTHONPATH")
if _existing_pythonpath:
    _pythonpath_entries.append(_existing_pythonpath)
MCP_TOOL_ENV_BASE = {"PYTHONPATH": os.pathsep.join(_pythonpath_entries)}


def build_tool_env(extra: dict | None = None) -> dict[str, str]:
    env = dict(MCP_TOOL_ENV_BASE)
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            env[key] = str(value)
    return env


# -----------------------------
# MCP Tool Groups Configuration (reasoning, searching, vqa)
# -----------------------------
MCP_TOOL_GROUPS = {
    "tool-reasoning": {
        "server_name": "reasoning-mcp-server",
        "client_type": "stdio",
        "params": StdioServerParameters(
            command=MCP_TOOL_PYTHON,
            args=["-u", "-m", "src.super_agent.tool.mcp_servers.reasoning_mcp_server", "--transport", "stdio"],
            env=build_tool_env({
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
                "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL"),
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
                "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL"),
            }),
            cwd=MCP_TOOL_CWD,
        ),
    },
    "tool-searching": {
        "server_name": "searching-mcp-server",
        "client_type": "stdio",
        "params": StdioServerParameters(
            command=MCP_TOOL_PYTHON,
            args=["-u", "-m", "src.super_agent.tool.mcp_servers.searching_mcp_server", "--transport", "stdio"],
            env=build_tool_env({
                "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
                "JINA_API_KEY": os.getenv("JINA_API_KEY"),
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
                "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
            }),
            cwd=MCP_TOOL_CWD,
        ),
    },
    "tool-vqa": {
        "server_name": "vision-mcp-server",
        "client_type": "stdio",
        "params": StdioServerParameters(
            command=MCP_TOOL_PYTHON,
            args=["-u", "-m", "src.super_agent.tool.mcp_servers.vision_mcp_server", "--transport", "stdio"],
            env=build_tool_env({
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
                "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
                "ENABLE_CLAUDE_VISION": "true",
                "ENABLE_OPENAI_VISION": "true",
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
                "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL"),
            }),
            cwd=MCP_TOOL_CWD,
        ),
    },
    "tool-selfevolution": {
    "server_name": "selfevolution-mcp-server",
    "client_type": "stdio",
    "params": StdioServerParameters(
        command=MCP_TOOL_PYTHON,  
        args=["-u", "-m", "src.super_agent.tool.mcp_servers.selfevolution"],
        env=build_tool_env({
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
            "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),

            "DOUBTER_MODEL": os.getenv("DOUBTER_MODEL", "anthropic/claude-sonnet-4.5"),
            "DOUBTER_SCORE_MODEL": os.getenv("DOUBTER_SCORE_MODEL", "google/gemini-2.5-pro"),
            "BOUNDARY": os.getenv("BOUNDARY", "7"),
        }),
        cwd=MCP_TOOL_CWD,
    ),
}
}

# per session system prompt marker (optional)
SYSTEM_MARKER = "[FRONTEND_SYSTEM_PROMPT]\n"

# Language-specific response instruction
LANGUAGE_INSTRUCTION = {
    "en": "Please respond in English.",
    "zh": "请用中文回答。",
}
LANGUAGE_MARKER = "[LANGUAGE_INSTRUCTION]\n"

# -----------------------------
# Global singletons
# -----------------------------
_MAIN_AGENT: Optional[SuperReActAgent] = None
_INIT_LOCK = asyncio.Lock()

# (session_id, agent_id) -> {"context": ContextManager, "last_idx": int}
_SESSION_CTX: Dict[Tuple[str, str], Dict[str, Any]] = {}
_SESSION_LOCK = asyncio.Lock()


# =============================
# Helpers: ToolServerConfig compat
# =============================
def make_tool_server_config(server_name: str, client_type: str, params):
    """
    兼容新版本 ToolServerConfig:
    - server_path 必填
    - params 必须 dict
    """
    if client_type == "sse":
        if isinstance(params, str):
            server_path = params
            server_params = {}
        elif isinstance(params, dict):
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

    return ToolServerConfig(
        server_name=server_name,
        server_path=str(client_type),
        client_type=client_type,
        params=params if isinstance(params, dict) else {"raw": params},
    )


# =============================
# Helpers: safe tool names (avoid dots etc.)
# =============================
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

async def wrap_tools_with_safe_names(tools: List[LocalFunction]) -> List[LocalFunction]:
    """
    让 LLM 看到的 tool.name 合法（不含 . 等），但实际执行仍调用原始 MCP tool_id
    """
    wrapped: List[LocalFunction] = []
    used = set()

    for raw_tool in tools:
        raw_tool_id = getattr(raw_tool, "name", "")
        safe_name = sanitize_tool_name(raw_tool_id)

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
        wrapped_tool._raw_tool_id = raw_tool_id  # debug mapping
        wrapped.append(wrapped_tool)

    return wrapped


# =============================
# Model config (same as agent_cdp.py)
# =============================
def create_model_config() -> ModelConfig:
    if not API_KEY:
        raise RuntimeError("Missing API_KEY / OPENROUTER_API_KEY env var")
    model_info = BaseModelInfo(
        api_key=API_KEY,
        api_base=API_BASE,
        model=MODEL_NAME,
        timeout=300,
    )
    return ModelConfig(model_provider=MODEL_PROVIDER, model_info=model_info)


# =============================
# MCP tool registration: list tools and build LocalFunction wrappers
# =============================
def _jsonschema_type_to_param_type(t: str) -> str:
    if t in {"string", "integer", "number", "boolean", "object", "array"}:
        return t
    return "string"

async def _runner_tool_call(tool_id: str, args: dict):
    result = await Runner.run_tool(tool_id, args)
    if isinstance(result, dict) and "result" in result:
        return result["result"]
    return result

async def register_mcp_server_and_wrap_all_tools() -> List[LocalFunction]:
    tool_mgr = resource_mgr.tool()

    ok_list = await tool_mgr.add_tool_servers([
        make_tool_server_config(SERVER_NAME, "sse", AUTO_BROWSER_SSE_URL)
    ])
    if not ok_list or not ok_list[0]:
        raise RuntimeError(f"Failed to add MCP server: {SERVER_NAME} @ {AUTO_BROWSER_SSE_URL}")

    tool_infos = await Runner.list_tools(SERVER_NAME)
    if not tool_infos:
        raise RuntimeError(
            f"Runner.list_tools('{SERVER_NAME}') returned empty/None. "
            f"Check MCP server tool metadata."
        )

    local_tools: List[LocalFunction] = []
    for info in tool_infos:
        # 注意：安装版 openjiuwen 使用 input_schema，而不是 schema（schema 是 Pydantic 方法）
        # 同时兼容 inputSchema (驼峰命名)
        schema = getattr(info, "input_schema", None) or getattr(info, "inputSchema", None) or {}

        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        params_def: List[Param] = []
        for pname, pinfo in props.items():
            pinfo = pinfo or {}
            ptype = _jsonschema_type_to_param_type(pinfo.get("type", "string"))
            params_def.append(
                Param(
                    name=pname,
                    description=pinfo.get("description", "") or "",
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


async def register_all_mcp_tool_groups(agent: SuperReActAgent) -> Dict[str, List[LocalFunction]]:
    """
    使用 SuperReActAgent 的 create_mcp_tools 方法注册 MCP_TOOL_GROUPS 中定义的所有工具组。
    这样可以正确处理 StdioServerParameters 参数。
    返回: {"tool-reasoning": [LocalFunction, ...], "tool-searching": [...], "tool-vqa": [...]}
    """
    tool_groups: Dict[str, List[LocalFunction]] = {}

    for group_name, cfg in MCP_TOOL_GROUPS.items():
        print(f"[INFO] Registering MCP tool group: {group_name} ({cfg['server_name']})")
        try:
            tools = await agent.create_mcp_tools(
                server_name=cfg["server_name"],
                client_type=cfg["client_type"],
                params=cfg["params"],
            )
            # 统一 wrap 成安全名字（去掉点号等非法字符）
            tools = await wrap_tools_with_safe_names(tools)
            tool_groups[group_name] = tools
            print(f"[INFO] Registered {len(tools)} tools for {group_name}")
        except Exception as e:
            print(f"[WARN] Failed to register tool group {group_name}: {e}")
            import traceback
            traceback.print_exc()
            tool_groups[group_name] = []

    return tool_groups


def build_browser_quick_qa_tool() -> LocalFunction:
    """
    一次调用搞定：
      browser_start -> session_new(url) -> qa(question) -> session_close(session_id)
    方便在 max_tool_calls_per_turn=1 时也能网页 QA。
    """
    async def browser_quick_qa(url: str, question: str, use_current_tab: bool = False) -> str:
        _ = await _runner_tool_call(f"{SERVER_NAME}.browser_start", {})

        raw = await _runner_tool_call(
            f"{SERVER_NAME}.session_new",
            {"url": url, "use_current_tab": use_current_tab},
        )

        session_id = None
        try:
            sess = json.loads(raw) if isinstance(raw, str) else raw
            session_id = sess.get("session_id")
        except Exception:
            session_id = None

        qa_raw = await _runner_tool_call(
            f"{SERVER_NAME}.qa",
            {"question": question, "session_id": session_id},
        )

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
    async def _call(task: str, session_id: str) -> str:
        tool_id = f"{SERVER_NAME}.run_task"
        result = await Runner.run_tool(
            tool_id,
            {
                "task": task,
                "session_id": session_id,     # 固定复用同一个工作 tab
                "use_current_tab": False,     # 推荐：同浏览器里新建“专用 tab”，不抢用户正在看的 tab
                "close": False,               # 多轮不要关
                "max_steps": 30,
            },
        )
        return result["result"] if isinstance(result, dict) and "result" in result else result

    return LocalFunction(
        name="browser_run_task",
        description="Run a complex browsing task in ONE call using the shared CDP Chrome; multi-turn stable.",
        params=[
            Param(name="task", description="High-level browsing task", param_type="string", required=True),
            Param(name="session_id", description="Chat session id for sticky browser tab reuse", param_type="string", required=True),
        ],
        func=_call,
    )

# =============================
# Session helpers: bind context + image saving
# =============================
def _bind_session_context(agent: SuperReActAgent, session_id: str) -> None:
    key = (session_id, agent._agent_config.id)
    session_data = _SESSION_CTX.get(key)
    if session_data is None:
        cm = ContextManager(llm=agent._llm, max_history_length=agent._context_manager.max_history_length)
        session_data = {"context": cm, "last_idx": 0}
        _SESSION_CTX[key] = session_data
    agent._context_manager = session_data["context"]

def _session_dir(session_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", session_id)[:200]
    d = SESSION_DATA_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d

def _guess_ext_from_mime(mime: Optional[str]) -> str:
    if not mime:
        return ".png"
    m = mime.lower()
    if "png" in m:
        return ".png"
    if "jpeg" in m or "jpg" in m:
        return ".jpg"
    if "webp" in m:
        return ".webp"
    return ".png"

def _strip_data_url(data: str) -> Tuple[Optional[str], str]:
    if not data.startswith("data:"):
        return None, data
    m = re.match(r"^data:([^;]+);base64,(.*)$", data, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None, data
    return m.group(1), m.group(2)

def _save_image_part(session_id: str, idx: int, data: str, mime_type: Optional[str]) -> Path:
    if data.startswith("http://") or data.startswith("https://"):
        p = _session_dir(session_id) / f"webshot_{idx:02d}.url.txt"
        p.write_text(data, encoding="utf-8")
        return p

    mime_from_dataurl, payload = _strip_data_url(data)
    mime = mime_type or mime_from_dataurl
    ext = _guess_ext_from_mime(mime)

    raw = base64.b64decode(payload)
    p = _session_dir(session_id) / f"webshot_{idx:02d}{ext}"
    p.write_bytes(raw)
    return p

def _part_get(part: Any, key: str, default=None):
    if isinstance(part, dict):
        return part.get(key, default)
    return getattr(part, key, default)

def _flatten_message_to_text_and_images(session_id: str, content: Any) -> Tuple[str, List[str]]:
    """
    content can be:
      - str
      - list of {type:"text"/"image"} (dict or pydantic-like)
    """
    if isinstance(content, str):
        return content, []

    text_chunks: List[str] = []
    images: List[str] = []
    idx = 1

    for part in (content or []):
        ptype = _part_get(part, "type")
        if ptype == "text":
            t = _part_get(part, "text", "")
            if t and str(t).strip():
                text_chunks.append(str(t))
        elif ptype == "image":
            data = _part_get(part, "data", "")
            mime_type = _part_get(part, "mime_type", None)
            if data and str(data).strip():
                fp = _save_image_part(session_id, idx, str(data), mime_type)
                images.append(str(fp))
                idx += 1

    return "\n".join(text_chunks).strip(), images


# =============================
# Init once: build main agent (CDP tools + reasoning/searching/vqa tools)
# =============================
async def init_agents_once() -> SuperReActAgent:
    global _MAIN_AGENT

    async with _INIT_LOCK:
        if _MAIN_AGENT is not None:
            return _MAIN_AGENT

        await Runner.start()

        # 1) CDP browser MCP tools (SSE)
        mcp_tools = await register_mcp_server_and_wrap_all_tools()
        mcp_tools = await wrap_tools_with_safe_names(mcp_tools)

        browser_tool = await register_browser_run_task_tool()
        
        # quick_qa_tool = build_browser_quick_qa_tool()
        # ✅ 不把 run_task 暴露给 LLM，强制走 browser_run_task
        BLOCK = {"browser-use-cdp-server_run_task"}  # 你也可以顺便 block browser_start，让 wrapper 自己 ensure_started
        mcp_tools = [t for t in mcp_tools if getattr(t, "name", "") not in BLOCK]

        tools = [browser_tool] + mcp_tools
        DOUBTER_TOOL_NAME = "selfevolution-mcp-server_selfevolution"
        # 2) 先创建 agent config
        agent_config = SuperAgentFactory.create_main_agent_config(
            agent_id="agent_cdp_api",
            agent_type="main",
            agent_version="1.0",
            description="CDP browsing agent (API runtime) with MCP tools, plus reasoning/searching/vqa capabilities.",
            model=create_model_config(),
            prompt_template=[
                {
                    "role": "system",
                    "content": (
                        "You are a powerful multi-capability agent with:\n"
                        "1) Browser control tools (CDP) for web interaction\n"
                        "2) Reasoning tools for complex problem solving\n"
                        "3) Searching tools for web search\n"
                        "4) VQA (Vision QA) tools for image understanding\n\n"
                        "5) Selfevolution tool to reflect on your own actions and results, ensuring high-quality outcomes. But only use it ONCE.\n\n"
                        "Tool usage rules (STRICT):\n"
                        "- Before any tool call, first output a brief numbered plan under a \"#PLAN#\" header (2-5 steps). Do not repeat the plan in the final answer unless asked.\n"
                        "- When the user needs ANY web interaction (open pages, click, drag, screenshot, extract info), CALL THE TOOL `browser_run_task(task)` EXACTLY ONCE.\n"
                        "- Write a clear step-by-step task in the tool input, including URLs.\n"
                        "- For complex reasoning tasks, use reasoning tools.\n"
                        "- For web search, use searching tools.\n"
                        "- For image/vision questions, use VQA tools.\n"
                        "- If the user request requires ANY web interaction, you MUST call browser_run_task(task, session_id) once to execute.\n"
                        "- Before you provide the final answer to end the agent, you MUST call {DOUBTER_TOOL_NAME}(history) ONCE and ONLY ONCE to reflect on:\n"
                        "  (a) the plan you executed,\n"
                        "  (b) the tool output you got,\n"
                        "  (c) whether steps are missing or illogical.\n"
                        "- Build 'history' as a compact plain text with sections:\n"
                        "[USER_QUESTION]\n"
                        "[PLAN_EXECUTED]\n"
                        "[TOOL_CALL_INPUT]\n"
                        "[TOOL_OUTPUT] (paste key JSON fields only; do NOT dump huge logs)\n"
                        "[WHAT_WE_LEARNED]\n"
                        "- If the selfevolution result indicates score < BOUNDARY or says rerun is needed:\n"
                        "  1) Provide a improved plan (what to change),\n"
                        "  2) Rerun the whole agent with the improved plan (same session_id), but ONLY RERUN IT ONCE\n"
                        "  3) If {DOUBTER_TOOL_NAME} doesn't deem its necessary to re-run or you already have re-runned, just end the agent and provide the final answer.\n"
                        "- Do at most ONE rerun. If still low score, answer with best effort and explain uncertainty.\n"
                        "- In CDP mode, DO NOT close the user's existing tabs. Use session_new() (new tab) by default.\n"
                        "Return final answer succinctly.\n"
                        f"Current time: {datetime.now().isoformat()}\n"
                    ),
                }
            ],
            max_iteration=15,
            max_tool_calls_per_turn=1,
            enable_o3_hints=False,
            enable_o3_final_answer=False,
            enable_todo_plan=False,
        )

        # 3) 先创建 agent（使用 CDP 工具）
        _MAIN_AGENT = SuperReActAgent(agent_config=agent_config, tools=tools, workflows=None)

        # 4) 使用 agent 的 create_mcp_tools 方法注册额外工具组 (stdio)
        extra_tool_groups = await register_all_mcp_tool_groups(_MAIN_AGENT)
        for group_name, group_tools in extra_tool_groups.items():
            if group_tools:
                print(f"[INFO] Adding {len(group_tools)} tools from {group_name}")
                _MAIN_AGENT.add_tools(group_tools)

        total_tools = len(_MAIN_AGENT._tools) if hasattr(_MAIN_AGENT, '_tools') else len(tools)
        print(f"[INFO] Agent initialized with {total_tools} tools total")
        return _MAIN_AGENT


# =============================
# Non-stream: run once (optional multi-turn via session_id)
# =============================
async def run_once(
    session_id: str,
    user_message_content: Any,   # str or multimodal parts
    system_prompt: Optional[str] = None,
    language: Optional[str] = None,  # "en" or "zh"
    history: Optional[List[Dict]] = None,  # Frontend-managed history (stateless mode)
) -> Dict[str, Any]:
    """
    One-shot call:
    - binds session context (or uses frontend history if provided)
    - invokes agent
    - returns {reply, history}

    Args:
        session_id: Session identifier
        user_message_content: User message content (text or multimodal)
        system_prompt: Optional system prompt to inject
        language: Response language ("en" or "zh")
        history: Optional frontend-managed history. If provided, uses stateless mode.
                 If None or empty, uses backend session management (original mode).
    """
    agent = await init_agents_once()

    # Determine mode: frontend history (stateless) vs backend session (stateful)
    use_frontend_history = history is not None and len(history) > 0

    async with _SESSION_LOCK:
        if use_frontend_history:
            # ===== Frontend history mode (stateless) =====
            agent._context_manager = ContextManager.from_history(
                history=history,
                llm=agent._llm,
                max_history_length=agent._context_manager.max_history_length
            )
        else:
            # ===== Backend session mode (stateful, original behavior) =====
            _bind_session_context(agent, session_id)

        # Upsert system prompt (applies to both modes)
        if system_prompt:
            sp = f"{SYSTEM_MARKER}{system_prompt}".strip()
            agent._context_manager.upsert_system_message(sp, SYSTEM_MARKER)

        # 设置语言指令 (applies to both modes)
        if language and language in LANGUAGE_INSTRUCTION:
            lang_instruction = f"{LANGUAGE_MARKER}{LANGUAGE_INSTRUCTION[language]}"
            agent._context_manager.upsert_system_message(lang_instruction, LANGUAGE_MARKER)

    user_text, image_paths = _flatten_message_to_text_and_images(session_id, user_message_content)

    query = (user_text or "").strip()
    if image_paths:
        joined = "\n".join(f"- {p}" for p in image_paths)
        query += f"\n\n[网页截图已保存到本地：]\n{joined}"
    query = query.strip() or " "

    file_path = image_paths[0] if image_paths else None

    result = await agent.invoke({"query": query, "file_path": file_path})
    reply = result.get("output", "")

    history = agent._context_manager.get_history() or []

    return {
        "session_id": session_id,
        "reply": reply,
        "history": history,
        "result": result,
    }


# =============================
# Stream: SSE generator (front-end friendly)
# =============================
async def run_turn_stream(
    session_id: str,
    user_message_content: Any,
    system_prompt: Optional[str] = None,
    language: Optional[str] = None,  # "en" or "zh"
    history: Optional[List[Dict]] = None,  # Frontend-managed history (stateless mode)
    request: Optional[Any] = None,  # FastAPI Request object for disconnect detection
):
    """
    SSE event async generator.
    Emits:
      - session_start
      - assistant_message/tool_message/system_message/user_message (from history)
      - assistant_final
      - error

    Args:
        session_id: Session identifier
        user_message_content: User message content (text or multimodal)
        system_prompt: Optional system prompt to inject
        language: Response language ("en" or "zh")
        history: Optional frontend-managed history. If provided, uses stateless mode.
                 If None or empty, uses backend session management (original mode).
        request: FastAPI Request object for disconnect detection
    """
    agent = await init_agents_once()

    # Determine mode: frontend history (stateless) vs backend session (stateful)
    use_frontend_history = history is not None and len(history) > 0

    # Get session context key (used for backend mode)
    key = (session_id, agent._agent_config.id)

    # session_data will be set based on mode
    session_data = None

    async with _SESSION_LOCK:
        if use_frontend_history:
            # ===== Frontend history mode (stateless) =====
            # Create fresh ContextManager from frontend-provided history
            agent._context_manager = ContextManager.from_history(
                history=history,
                llm=agent._llm,
                max_history_length=agent._context_manager.max_history_length
            )
            # Create a temporary session_data dict for last_idx tracking
            session_data = {"context": agent._context_manager, "last_idx": 0}
        else:
            # ===== Backend session mode (stateful, original behavior) =====
            _bind_session_context(agent, session_id)
            # Get reference to session data inside lock to ensure it exists
            session_data = _SESSION_CTX[key]

        # Upsert system prompt (applies to both modes)
        if system_prompt:
            sp = f"{SYSTEM_MARKER}{system_prompt}".strip()
            agent._context_manager.upsert_system_message(sp, SYSTEM_MARKER)

        # 设置语言指令 (applies to both modes)
        if language and language in LANGUAGE_INSTRUCTION:
            lang_instruction = f"{LANGUAGE_MARKER}{LANGUAGE_INSTRUCTION[language]}"
            agent._context_manager.upsert_system_message(lang_instruction, LANGUAGE_MARKER)

    # event queue (agent internal events, if your SuperReActAgent supports callback)
    event_queue: asyncio.Queue = asyncio.Queue()

    async def event_callback(event: dict):
        await event_queue.put(event)

    # safe: not all versions have set_event_callback
    if hasattr(agent, "set_event_callback"):
        agent.set_event_callback(event_callback)

    yield {"type": "session_start", "data": {"session_id": session_id}}

    user_text, image_paths = _flatten_message_to_text_and_images(session_id, user_message_content)

    query = (user_text or "").strip()
    if image_paths:
        joined = "\n".join(f"- {p}" for p in image_paths)
        query += f"\n\n[网页截图已保存到本地：]\n{joined}"
    query = query.strip() or " "

    file_path = image_paths[0] if image_paths else None

    task = asyncio.create_task(agent.invoke({"query": query, "file_path": file_path}))

    # Retrieve last_idx from session context to avoid re-streaming old messages
    last_idx = session_data.get("last_idx", 0)

    try:
        import time
        last_ping = time.monotonic()
        while True:
            # Check if client disconnected
            if request and await request.is_disconnected():
                print(f"[INFO] Client disconnected for session {session_id}, cancelling agent task...")
                break

            # ✅ 每 5 秒发个心跳，防止 SSE 被中间层断开
            if time.monotonic() - last_ping > 5:
                yield {"type": "ping", "data": {"ts": datetime.now().isoformat()}}
                last_ping = time.monotonic()
            
            # drain agent events first
            while True:
                try:
                    event = event_queue.get_nowait()
                    yield {"type": event.get("type", "agent_event"), "data": event}
                except asyncio.QueueEmpty:
                    break

            # stream new messages from history
            hist = agent._context_manager.get_history() or []
            while last_idx < len(hist):
                msg = hist[last_idx]
                last_idx += 1
                role = msg.get("role")

                if role == "assistant":
                    yield {"type": "assistant_message", "data": msg}
                elif role == "tool":
                    yield {"type": "tool_message", "data": msg}
                elif role == "system":
                    yield {"type": "system_message", "data": msg}
                elif role == "user":
                    yield {"type": "user_message", "data": msg}
                else:
                    yield {"type": "message", "data": msg}

            # Update last_idx in session context after processing messages
            if session_data:
                session_data["last_idx"] = last_idx

            if task.done():
                # final drain
                while True:
                    try:
                        event = event_queue.get_nowait()
                        yield {"type": event.get("type", "agent_event"), "data": event}
                    except asyncio.QueueEmpty:
                        break

                try:
                    result = task.result()
                except Exception as e:
                    yield {"type": "error", "data": {"error": str(e)}}
                    break

                # Check if agent reported internal error
                result_type = result.get("result_type", "answer")
                if result_type == "error":
                    error_msg = result.get("output", "Unknown error occurred")
                    print(f"[ERROR] Agent completed with error for session {session_id}: {error_msg}")
                    yield {"type": "error", "data": {"error": error_msg}}
                    break

                yield {
                    "type": "assistant_final",
                    "data": {
                        "reply": result.get("output", ""),
                        "result_type": result_type,
                    },
                }
                break

            await asyncio.sleep(0.15)
    finally:
        # Cancel agent task if still running (disconnect or error)
        if not task.done():
            print(f"[INFO] Cancelling agent task for session {session_id}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                print(f"[INFO] Agent task cancelled successfully for session {session_id}")
            except Exception as e:
                print(f"[WARN] Error during task cancellation for session {session_id}: {e}")

        # Clear event callback
        if hasattr(agent, "set_event_callback"):
            agent.set_event_callback(None)


# =============================
# reset / shutdown
# =============================
async def reset_session(session_id: str) -> None:
    keys = [k for k in list(_SESSION_CTX.keys()) if k[0] == session_id]
    for k in keys:
        _SESSION_CTX.pop(k, None)

    d = _session_dir(session_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

async def shutdown_runner():
    try:
        await Runner.stop()
    except RuntimeError as e:
        if "cancel scope" in str(e):
            print("Ignore MCP SSE shutdown RuntimeError during Runner.stop:", e)
        else:
            raise
