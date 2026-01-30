#!/usr/bin/env python
# coding: utf-8

import asyncio
import base64
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

from src.super_agent.agent.super_react_agent import SuperReActAgent
from src.super_agent.agent.super_config import SuperAgentFactory
from src.super_agent.agent.prompt_templates_ori import (
    get_main_agent_system_prompt,
    get_browsing_agent_system_prompt,
    get_coding_agent_system_prompt,
)
from src.super_agent.agent.context_manager import ContextManager
from openjiuwen.core.runner.runner import Runner

# 复用你 test 脚本里的 bootstrap（保持你现在能跑的方式）
from src.super_agent.test.super_react_agent_test_run import (
    create_model_config,
    ensure_autobrowser_sse_server,
    AUTO_BROWSER_TRANSPORT,
    build_mcp_tool_groups,
    OPENAI_API_KEY,
)

_MAIN_AGENT: Optional[SuperReActAgent] = None
_BROWSING_AGENT: Optional[SuperReActAgent] = None
_CODING_AGENT: Optional[SuperReActAgent] = None
_INIT_LOCK = asyncio.Lock()

# =============== Session Stores ===============
# (session_id, agent_id) -> ContextManager
_SESSION_CTX: Dict[Tuple[str, str], ContextManager] = {}
_SESSION_LOCK = asyncio.Lock()

# per session system prompt marker
SYSTEM_MARKER = "[FRONTEND_SYSTEM_PROMPT]\n"

# where to store screenshots
SESSION_DATA_DIR = Path(os.getenv("SUPER_AGENT_SESSION_DIR", "./.super_agent_sessions")).resolve()
SESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- utils: session ctx -----------------

def _bind_session_context(agent: SuperReActAgent, session_id: str) -> None:
    """
    Bind session context so multi-turn works:
    same session_id will reuse the same ContextManager (history)
    """
    key = (session_id, agent._agent_config.id)
    cm = _SESSION_CTX.get(key)
    if cm is None:
        cm = ContextManager(llm=agent._llm, max_history_length=agent._context_manager.max_history_length)
        _SESSION_CTX[key] = cm
    agent._context_manager = cm


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
    """
    data:image/png;base64,xxxx -> (image/png, xxxx)
    else -> (None, data)
    """
    if not data.startswith("data:"):
        return None, data
    m = re.match(r"^data:([^;]+);base64,(.*)$", data, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None, data
    return m.group(1), m.group(2)


def _save_image_part(session_id: str, idx: int, data: str, mime_type: Optional[str]) -> Path:
    """
    Save image to local disk.
    - if http(s) url: save a .url.txt
    - else: treat as dataURL or raw base64
    """
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
from typing import Any, Dict, List, Tuple, Optional

def _part_get(part: Any, key: str, default=None):
    """
    Safely get fields from either:
      - dict
      - Pydantic model (v2): has attribute access, and may have model_dump
      - other objects with attributes
    """
    if isinstance(part, dict):
        return part.get(key, default)
    # Pydantic object / normal object
    return getattr(part, key, default)

def _flatten_message_to_text_and_images(session_id: str, content: Any) -> Tuple[str, List[str]]:
    """
    content can be:
      - str
      - list of TextPart/ImagePart (pydantic objects) OR dicts
    Return:
      text, [saved_image_paths]
    """
    if isinstance(content, str):
        return content, []

    text_chunks: List[str] = []
    images: List[str] = []
    idx = 1

    for part in content:
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


# def _flatten_message_to_text_and_images(session_id: str, content: Any) -> Tuple[str, List[str]]:
#     """
#     content:
#       - str
#       - list[{type:"text"...} | {type:"image"...}]
#     Returns:
#       text, [saved_image_paths]
#     """
#     if isinstance(content, str):
#         return content, []

#     text_chunks: List[str] = []
#     images: List[str] = []
#     idx = 1

#     for part in content:
#         # pydantic model 或 dict 都兼容
#         ptype = getattr(part, "type", None) or part.get("type")
#         if ptype == "text":
#             t = getattr(part, "text", None) or part.get("text", "")
#             if t:
#                 text_chunks.append(t)
#         elif ptype == "image":
#             data = getattr(part, "data", None) or part.get("data", "")
#             mime_type = getattr(part, "mime_type", None) or part.get("mime_type")
#             if data:
#                 fp = _save_image_part(session_id, idx, data, mime_type)
#                 images.append(str(fp))
#                 idx += 1

#     return "\n".join([x for x in text_chunks if x.strip()]).strip(), images


# ----------------- init agents once -----------------

async def init_agents_once() -> Tuple[SuperReActAgent, SuperReActAgent, SuperReActAgent]:
    """
    初始化 Runner + MCP servers + agents + tools + sub agents
    只初始化一次，供 API 多次请求复用
    """
    global _MAIN_AGENT, _BROWSING_AGENT, _CODING_AGENT

    async with _INIT_LOCK:
        if _MAIN_AGENT is not None:
            return _MAIN_AGENT, _BROWSING_AGENT, _CODING_AGENT

        if AUTO_BROWSER_TRANSPORT == "sse":
            ensure_autobrowser_sse_server()

        await Runner.start()

        main_agent_config = SuperAgentFactory.create_main_agent_config(
            agent_id="super_react_main_mcp",
            agent_type="main",
            agent_version="1.0",
            description="Main MCP agent with multiple tool groups",
            model=create_model_config(),
            prompt_template=[{"role": "system", "content": get_main_agent_system_prompt(datetime.now())}],
            max_iteration=10,
            max_tool_calls_per_turn=1,
            enable_o3_hints=True,
            o3_api_key=OPENAI_API_KEY,
            enable_o3_final_answer=True,
            enable_todo_plan=True,
        )

        browsing_agent_config = SuperAgentFactory.create_main_agent_config(
            agent_id="agent-browsing",
            agent_type="sub",
            agent_version="1.0",
            description="Browsing specialist sub-agent",
            model=create_model_config(),
            prompt_template=[{"role": "system", "content": get_browsing_agent_system_prompt(datetime.now())}],
            max_iteration=10,
            max_tool_calls_per_turn=1,
            enable_o3_hints=True,
            o3_api_key=OPENAI_API_KEY,
            enable_o3_final_answer=True,
            enable_todo_plan=False,
        )

        coding_agent_config = SuperAgentFactory.create_main_agent_config(
            agent_id="agent-coding",
            agent_type="sub",
            agent_version="1.0",
            description="Coding specialist sub-agent",
            model=create_model_config(),
            prompt_template=[{"role": "system", "content": get_coding_agent_system_prompt(datetime.now())}],
            max_iteration=10,
            max_tool_calls_per_turn=1,
            enable_o3_hints=True,
            o3_api_key=OPENAI_API_KEY,
            enable_o3_final_answer=True,
            enable_todo_plan=False,
        )

        main_agent = SuperReActAgent(agent_config=main_agent_config, tools=None, workflows=None)
        browsing_agent = SuperReActAgent(agent_config=browsing_agent_config, tools=None, workflows=None)
        coding_agent = SuperReActAgent(agent_config=coding_agent_config, tools=None, workflows=None)

        tool_groups = await build_mcp_tool_groups(main_agent)

        MAIN_AGENT_TOOL_GROUPS = ["tool-reasoning"]
        CODING_AGENT_TOOL_GROUPS = ["tool-code", "tool-vqa", "tool-reading"]
        BROWSING_AGENT_TOOL_GROUPS = ["tool-searching", "tool-vqa", "tool-reading", "tool-autobrowser", "tool-transcribe"]

        for group in MAIN_AGENT_TOOL_GROUPS:
            if group in tool_groups:
                main_agent.add_tools(tool_groups[group])

        for group in BROWSING_AGENT_TOOL_GROUPS:
            if group in tool_groups:
                browsing_agent.add_tools(tool_groups[group])

        for group in CODING_AGENT_TOOL_GROUPS:
            if group in tool_groups:
                coding_agent.add_tools(tool_groups[group])

        main_agent.register_sub_agent("agent-browsing", browsing_agent)
        main_agent.register_sub_agent("agent-coding", coding_agent)

        _MAIN_AGENT, _BROWSING_AGENT, _CODING_AGENT = main_agent, browsing_agent, coding_agent
        return _MAIN_AGENT, _BROWSING_AGENT, _CODING_AGENT


# ----------------- SSE: run one turn -----------------

async def run_turn_stream(
    session_id: str,
    user_message_content: Any,           # req.message.content
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict]] = None,  # Frontend-managed history (stateless mode)
):
    """
    SSE event async generator.
    Uses event callback + polling ContextManager.history while main_agent.invoke() is running.
    Now supports streaming events from both main agent and sub-agents (iteration, tool execution).

    Args:
        session_id: Session identifier
        user_message_content: User message content (text or multimodal)
        system_prompt: Optional system prompt to inject
        history: Optional frontend-managed history. If provided, uses stateless mode.
                 If None or empty, uses backend session management (original mode).
    """

    main_agent, browsing_agent, coding_agent = await init_agents_once()

    # Determine mode: frontend history (stateless) vs backend session (stateful)
    use_frontend_history = history is not None and len(history) > 0

    async with _SESSION_LOCK:
        if use_frontend_history:
            # ===== Frontend history mode (stateless) =====
            # Create fresh ContextManagers from frontend-provided history
            main_agent._context_manager = ContextManager.from_history(
                history=history,
                llm=main_agent._llm,
                max_history_length=main_agent._context_manager.max_history_length
            )
            browsing_agent._context_manager = ContextManager.from_history(
                history=history,
                llm=browsing_agent._llm,
                max_history_length=browsing_agent._context_manager.max_history_length
            )
            coding_agent._context_manager = ContextManager.from_history(
                history=history,
                llm=coding_agent._llm,
                max_history_length=coding_agent._context_manager.max_history_length
            )
        else:
            # ===== Backend session mode (stateful, original behavior) =====
            _bind_session_context(main_agent, session_id)
            _bind_session_context(browsing_agent, session_id)
            _bind_session_context(coding_agent, session_id)

        # Upsert system prompt (applies to both modes)
        if system_prompt:
            sp = f"{SYSTEM_MARKER}{system_prompt}".strip()
            main_agent._context_manager.upsert_system_message(sp, SYSTEM_MARKER)
            browsing_agent._context_manager.upsert_system_message(sp, SYSTEM_MARKER)
            coding_agent._context_manager.upsert_system_message(sp, SYSTEM_MARKER)

    # Create event queue for agent status updates
    event_queue: asyncio.Queue = asyncio.Queue()

    async def event_callback(event: dict):
        """Callback to receive events from agents (main + sub)"""
        await event_queue.put(event)

    # Set event callback on main agent (will propagate to sub-agents)
    main_agent.set_event_callback(event_callback)

    # tell client session started
    yield {"type": "session_start", "data": {"session_id": session_id}}

    # parse user text + screenshots
    user_text, image_paths = _flatten_message_to_text_and_images(session_id, user_message_content)

    # Make the prompt explicit: this is a webpage screenshot
    query = (user_text or "").strip()
    if image_paths:
        # You only pass the first image to file_path (your agent signature),
        # but we also mention all paths in text for traceability.
        joined = "\n".join(f"- {p}" for p in image_paths)
        query += f"\n\n[网页截图已保存到本地：]\n{joined}"
    query = query.strip() or " "

    file_path = image_paths[0] if image_paths else None

    # launch invoke in background
    task = asyncio.create_task(main_agent.invoke({"query": query, "file_path": file_path}))

    # stream new history items
    last_idx = 0

    while True:
        # First, drain all pending events from the queue
        while True:
            try:
                event = event_queue.get_nowait()
                # Yield agent status events (iteration_start, tool_executing, tool_completed, tool_error)
                yield {"type": event["type"], "data": event}
            except asyncio.QueueEmpty:
                break

        # Then check history for new messages
        hist = main_agent._context_manager.get_history() or []

        while last_idx < len(hist):
            msg = hist[last_idx]
            last_idx += 1

            role = msg.get("role")

            if role == "assistant":
                yield {"type": "assistant_message", "data": msg}
            elif role == "tool":
                # tool result messages (your agent adds tool results as role=tool)
                yield {"type": "tool_message", "data": msg}
            elif role == "system":
                # usually not necessary to show, but keep it for debugging
                yield {"type": "system_message", "data": msg}
            elif role == "user":
                # optional: normally front-end already knows user input
                yield {"type": "user_message", "data": msg}
            else:
                yield {"type": "message", "data": msg}

        if task.done():
            # Drain any remaining events before finishing
            while True:
                try:
                    event = event_queue.get_nowait()
                    yield {"type": event["type"], "data": event}
                except asyncio.QueueEmpty:
                    break

            try:
                result = task.result()
            except Exception as e:
                yield {"type": "error", "data": {"error": str(e)}}
                break

            yield {
                "type": "assistant_final",
                "data": {
                    "reply": result.get("output", ""),
                    "result_type": result.get("result_type", "answer"),
                },
            }
            break

        await asyncio.sleep(0.15)

    # Clear the event callback when done
    main_agent.set_event_callback(None)


async def reset_session(session_id: str) -> None:
    """
    Clear contexts + delete screenshots for this session
    """
    keys = [k for k in list(_SESSION_CTX.keys()) if k[0] == session_id]
    for k in keys:
        _SESSION_CTX.pop(k, None)

    d = _session_dir(session_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


async def shutdown_runner():
    """
    服务关闭时调用，释放 Runner
    """
    try:
        await Runner.stop()
    except RuntimeError as e:
        if "cancel scope" in str(e):
            print("Ignore MCP SSE shutdown RuntimeError during Runner.stop:", e)
        else:
            raise
