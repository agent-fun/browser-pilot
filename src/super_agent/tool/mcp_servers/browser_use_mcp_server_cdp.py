# python browser_use_mcp_server_cdp.py --host 127.0.0.1 --port 8930
import os
import json
import uuid
import asyncio
import argparse
from typing import Optional, Dict, Any, List
import httpx
import time
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
import uvicorn
from fastapi import FastAPI, Request
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount

from browser_use import Browser, Agent
from browser_use import ChatBrowserUse, ChatOpenAI, ChatAnthropic  
import inspect
from urllib.parse import urlparse


def _filter_kwargs(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return kwargs

def _extract_final_text(history_or_result) -> str:
    # 尽量兼容不同版本的 history/result 结构
    for key in ("final_result", "final_output", "result", "output"):
        if hasattr(history_or_result, key):
            v = getattr(history_or_result, key)
            try:
                v = v() if callable(v) else v
            except Exception:
                pass
            if v:
                return str(v)

    # browser_use 有时是 history.extracted_content()
    try:
        if hasattr(history_or_result, "extracted_content"):
            chunks = history_or_result.extracted_content()
            if chunks:
                chunks = [c.strip() for c in chunks if c and str(c).strip()]
                if chunks:
                    return chunks[-1]
    except Exception:
        pass

    return str(history_or_result)

async def _agent_run(agent, max_steps: int):
    # 兼容 agent.run(max_steps=...) 和 agent.run()
    try:
        sig = inspect.signature(agent.run)
        if "max_steps" in sig.parameters:
            return await agent.run(max_steps=max_steps)
    except Exception:
        pass
    return await agent.run()

def normalize_http_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return u
    u = u.strip()
    if not u:
        return None
    if urlparse(u).scheme == "":
        return "http://" + u
    return u

# ----------------------------
# LLM builder (browser_use-native)
# ----------------------------
def build_llm():
    # browser_use_mcp_server_cdp.py

    # api_key = os.getenv("OPENROUTER_API_KEY")
    # if api_key:
    #     model = os.getenv("BROWSER_USE_LLM_MODEL", "gpt-4.1-mini")
    #     base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"  # ✅ 默认补上
    #     kwargs = {"model": model, "api_key": api_key, "base_url": base_url}
    #     return ChatOpenAI(**kwargs)

    """
    Priority:
      1) BROWSER_USE_API_KEY -> ChatBrowserUse()
      2) OPENAI/OPENROUTER (OpenAI-compatible) -> ChatOpenAI(...)
      3) ANTHROPIC_API_KEY -> ChatAnthropic(...)
    """
    # 1) Browser-Use Cloud model (optional)
    if os.getenv("BROWSER_USE_API_KEY"):
        # 官方文档：设置 BROWSER_USE_API_KEY 后可直接 ChatBrowserUse() 使用 :contentReference[oaicite:2]{index=2}
        return ChatBrowserUse()

    # 2) OpenAI-compatible (OpenAI / OpenRouter / any OpenAI-style gateway)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        model = os.getenv("BROWSER_USE_LLM_MODEL", "google/gemini-2.5-pro")  
        base_url = os.getenv("OPENROUTER_BASE_URL")  # 例如 OpenRouter: https://openrouter.ai/api/v1
        kwargs = {"model": model, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    # 3) Anthropic direct
    if os.getenv("ANTHROPIC_API_KEY"):
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4.5")
        return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))

    raise RuntimeError(
        "No LLM credentials found. Set one of:\n"
        "- BROWSER_USE_API_KEY (Browser-Use Cloud)\n"
        "- OPENAI_API_KEY (+ optional BROWSER_USE_LLM_MODEL / LLM_BASE_URL)\n"
        "- OPENROUTER_API_KEY (+ optional BROWSER_USE_LLM_MODEL / LLM_BASE_URL)\n"
        "- ANTHROPIC_API_KEY (+ optional ANTHROPIC_MODEL)\n"
    )


# ----------------------------
# QA Output Schema (structured)
# ----------------------------
class QAResult(BaseModel):
    answer: str = Field(..., description="Final answer to the question.")
    evidence: list[str] = Field(default_factory=list, description="Short quotes/snippets from the page supporting the answer.")
    url: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)


# ----------------------------
# Browser Service (singleton)
# ----------------------------
class BrowserService:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._browser: Optional[Browser] = None
        self._llm = build_llm()

        # sessions: session_id -> { "page": <Page>, "external": bool }
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._active_session_id: Optional[str] = None
        
        self._cdp_ready_timeout = float(os.getenv("CDP_READY_TIMEOUT_SEC", "12"))
        self._cdp_ready_poll = float(os.getenv("CDP_READY_POLL_SEC", "0.3"))
        self._reconnect_once_guard = asyncio.Lock()  # 防止并发反复重连
        
    async def _restart_browser(self, reason: str = "") -> None:
        # 只做“断开并清空”，不去 kill 用户 Chrome
        old = self._browser
        self._browser = None
        self._sessions = {}
        self._active_session_id = None
        try:
            if old is not None and (not self._is_cdp_mode()):
                await old.stop()
        except Exception:
            pass

    async def _ensure_ready_or_reconnect(self) -> Dict[str, Any]:
        """
        确保 browser_use 内部 CDP client 已初始化。
        若遇到 'CDP client not initialized' / 连接断开，自动重连一次。
        """
        try:
            return await self.ensure_started()
        except Exception as e:
            msg = str(e)
            if "CDP client not initialized" not in msg:
                raise

        # 触发重连（只重连一次，避免风暴）
        async with self._reconnect_once_guard:
            try:
                await self._restart_browser("cdp_not_initialized")
            except Exception:
                pass
            return await self.ensure_started()

    async def _bring_to_front_safe(page):
        try:
            fn = getattr(page, "bring_to_front", None)
            if callable(fn):
                return await fn()
            inner = getattr(page, "_page", None) or getattr(page, "page", None)
            fn2 = getattr(inner, "bring_to_front", None)
            if callable(fn2):
                return await fn2()
        except Exception:
            pass
        
    def _cdp_url(self) -> Optional[str]:
        return normalize_http_url(os.getenv("BROWSER_USE_CDP_URL"))

    def _is_cdp_mode(self) -> bool:
        return bool(self._cdp_url())

    async def _probe_cdp_http(self, cdp_url: str) -> Dict[str, Any]:
        """
        Probe CDP HTTP endpoint: /json/version and /json/list
        Useful to confirm SSH tunnel + Chrome remote debugging is reachable.
        """
        base = cdp_url.rstrip("/")
        out: Dict[str, Any] = {"cdp_url": cdp_url, "json_version_ok": False, "json_list_ok": False}
        timeout = httpx.Timeout(3.0, connect=3.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.get(f"{base}/json/version")
                out["json_version_status"] = r.status_code
                if r.status_code == 200:
                    out["json_version_ok"] = True
                    out["json_version"] = r.json()
            except Exception as e:
                out["json_version_error"] = repr(e)

            try:
                r = await client.get(f"{base}/json/list")
                out["json_list_status"] = r.status_code
                if r.status_code == 200:
                    out["json_list_ok"] = True
                    data = r.json()
                    # avoid dumping huge payload
                    out["targets_count"] = len(data) if isinstance(data, list) else None
            except Exception as e:
                out["json_list_error"] = repr(e)

        return out

    async def _wait_cdp_ready(self) -> Dict[str, Any]:
        """
        Wait until browser_use's CDP client is ready.
        This prevents: 'CDP client not initialized - browser may not be connected yet'
        """
        assert self._browser is not None
        cdp = self._cdp_url()
        if not cdp:
            return {"cdp_ready": True, "mode": "local", "pid": os.getpid()}

        probe = await self._probe_cdp_http(cdp)

        last_err = None
        deadline = time.monotonic() + self._cdp_ready_timeout
        while time.monotonic() < deadline:
            try:
                page = await self._browser.get_current_page()
                _ = await page.get_url()
                return {
                    "cdp_ready": True,
                    "mode": "cdp",
                    "pid": os.getpid(),
                    "probe": probe,
                }
            except Exception as e:
                last_err = e
                await asyncio.sleep(self._cdp_ready_poll)

        raise RuntimeError(f"CDP not ready after {self._cdp_ready_timeout}s. last_err={last_err!r}, probe={probe}")

    async def ensure_started(self) -> Dict[str, Any]:
        async with self._lock:
            # 已有 browser：先检查 ready
            if self._browser is not None:
                try:
                    ready = await self._wait_cdp_ready()
                    return {"started": True, **ready}
                except Exception:
                    # browser 对象在，但内部连接坏了：重建
                    await self._restart_browser("wait_cdp_ready_failed")

            cdp = self._cdp_url()
            headless = os.getenv("BROWSER_HEADLESS", "false").lower() in {"1", "true", "yes"}

            if cdp:
                self._browser = Browser(cdp_url=cdp, headless=headless, is_local=False)
            else:
                self._browser = Browser(headless=headless)

            await self._browser.start()
            ready = await self._wait_cdp_ready()
            return {"mode": "cdp" if self._is_cdp_mode() else "local", "started": True, **ready}

    async def _get_page(self, session_id: Optional[str]):
        if self._browser is None:
            await self.ensure_started()
        assert self._browser is not None

        sid = session_id or self._active_session_id
        if not sid or sid not in self._sessions:
            raise RuntimeError("No active session. Call session_new(...) first.")

        return sid, self._sessions[sid]["page"], self._sessions[sid]["external"]

    async def session_new(self, url: Optional[str] = None, use_current_tab: bool = False, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Default: create a NEW tab and only control that tab (safe in CDP mode).
        If use_current_tab=True: attach to current tab as external (won't be closed unless force=True).
        """
        await self._ensure_ready_or_reconnect()
        assert self._browser is not None

        # # 记住创建前的“前台页”（通常就是插件前端页）
        # frontend_page = None
        # if not use_current_tab:
        #     try:
        #         frontend_page = await self._browser.get_current_page()
        #     except Exception:
        #         frontend_page = None
            
        async with self._lock:
            sid = (session_id or uuid.uuid4().hex[:12])

            if use_current_tab:
                page = await self._browser.get_current_page()
                external = True
            else:
                page = await self._browser.new_page(url=url)
                external = False

            self._sessions[sid] = {"page": page, "external": external}
            self._active_session_id = sid
        # # 新建工作 tab 后，把插件前端 tab 拉回前台（避免“占据插件界面”）
        # if frontend_page is not None:
        #     await self._bring_to_front_safe(frontend_page)
            
            try:
                title = await page.get_title()
            except Exception:
                title = None
            try:
                cur_url = await page.get_url()
            except Exception:
                cur_url = url

            return {"session_id": sid, "external": external, "title": title, "url": cur_url, "pid": os.getpid()}

    async def session_switch(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            if session_id not in self._sessions:
                raise RuntimeError(f"Unknown session_id={session_id}")
            self._active_session_id = session_id
            return {"active_session_id": self._active_session_id, "pid": os.getpid()}

    async def session_list(self) -> Dict[str, Any]:
        out = []
        for sid, info in self._sessions.items():
            page = info["page"]
            external = info["external"]
            try:
                url = await page.get_url()
            except Exception:
                url = None
            out.append({"session_id": sid, "external": external, "url": url, "active": sid == self._active_session_id})
        return {"sessions": out, "pid": os.getpid()}

    async def session_close(self, session_id: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        sid = session_id or self._active_session_id
        if not sid or sid not in self._sessions:
            return {"closed": False, "reason": "no such session", "pid": os.getpid()}

        async with self._lock:
            info = self._sessions[sid]
            page = info["page"]
            external = info["external"]

            if external and not force:
                return {"closed": False, "reason": "external tab; set force=True to close", "pid": os.getpid()}

            # Actor: close_page(page) :contentReference[oaicite:7]{index=7}
            await self._browser.close_page(page)  # closes only this tab
            del self._sessions[sid]

            if self._active_session_id == sid:
                self._active_session_id = next(iter(self._sessions.keys()), None)

            return {"closed": True, "session_id": sid, "active_session_id": self._active_session_id, "pid": os.getpid()}

    # ---- actions ----
    async def goto(self, url: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid, page, _ = await self._get_page(session_id)
        # Actor: page.goto(url) :contentReference[oaicite:8]{index=8}
        await page.goto(url)
        title = None
        try:
            title = await page.get_title()
        except Exception:
            pass
        return {"session_id": sid, "url": await page.get_url(), "title": title, "pid": os.getpid()}

    async def click_css(self, selector: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid, page, _ = await self._get_page(session_id)
        # Actor: page.click(selector) :contentReference[oaicite:9]{index=9}
        await page.click(selector)
        return {"session_id": sid, "clicked": selector, "pid": os.getpid()}

    async def click_prompt(self, prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid, page, _ = await self._get_page(session_id)
        # Actor: get_element_by_prompt(prompt, llm) -> element :contentReference[oaicite:10]{index=10}
        el = await page.get_element_by_prompt(prompt, llm=self._llm)
        await el.click()
        return {"session_id": sid, "clicked_prompt": prompt, "pid": os.getpid()}

    async def fill_css(self, selector: str, text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid, page, _ = await self._get_page(session_id)
        # Actor: page.fill(selector, text) :contentReference[oaicite:11]{index=11}
        await page.fill(selector, text)
        return {"session_id": sid, "filled": selector, "pid": os.getpid()}

    async def fill_prompt(self, prompt: str, text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid, page, _ = await self._get_page(session_id)
        el = await page.get_element_by_prompt(prompt, llm=self._llm)
        await el.fill(text)
        return {"session_id": sid, "filled_prompt": prompt, "pid": os.getpid()}

    async def screenshot(self, session_id: Optional[str] = None, full_page: bool = False) -> Dict[str, Any]:
        sid, page, _ = await self._get_page(session_id)
        # Actor: page.screenshot(full_page=...) returns base64 string :contentReference[oaicite:12]{index=12}
        b64 = await page.screenshot()
        return {"session_id": sid, "full_page": full_page, "image_base64": b64, "pid": os.getpid()}

    async def qa(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid, page, _ = await self._get_page(session_id)
        url = None
        title = None
        try:
            url = await page.get_url()
        except Exception:
            pass
        try:
            title = await page.get_title()
        except Exception:
            pass

        # Actor: page.extract_content(prompt, structured_output, llm) :contentReference[oaicite:13]{index=13}
        prompt = (
            "Answer the user's question using ONLY the current webpage.\n"
            "Return short evidence snippets (quotes) from the page.\n"
            f"Question: {question}"
        )
        result: QAResult = await page.extract_content(prompt=prompt, structured_output=QAResult, llm=self._llm)
        result.url = result.url or url
        result.title = result.title or title
        return {"session_id": sid, "qa": result.model_dump(), "pid": os.getpid()}

   # ----------------------------
    # ✅ NEW: run_task (one tool-call for long multi-step tasks)
    # ----------------------------
    async def run_task(
        self,
        task: str,
        session_id: Optional[str] = None,
        max_steps: int = 40,
        start_url: Optional[str] = None,
        use_current_tab: bool = False,
        close: bool = False,  # ✅ 默认 False：多轮不关 tab
        return_screenshot: bool = False,
        screenshot_full_page: bool = False,
    ) -> Dict[str, Any]:
        if not task or not task.strip():
            raise ValueError("task is required")

        await self._ensure_ready_or_reconnect()
        assert self._browser is not None

        # 1) 获取/创建稳定工作页：如果给了 session_id，就固定复用该 session 的 tab
        page = None
        external = False
        
        # 先在锁里只做“是否存在”的判断，绝不在锁里 await session_new
        need_create = False
        async with self._lock:
            if session_id and session_id in self._sessions:
                page = self._sessions[session_id]["page"]
                external = self._sessions[session_id]["external"]
            else:
                need_create = True
                
        # 创建 session 放到锁外（避免死锁）
        if need_create:
            sess = await self.session_new(
                url=start_url or "about:blank",
                use_current_tab=use_current_tab,
                session_id=session_id,
            )
            session_id = sess["session_id"]

            async with self._lock:
                page = self._sessions[session_id]["page"]
                external = self._sessions[session_id]["external"]

        # async with self._lock:
        #     if session_id and session_id in self._sessions:
        #         page = self._sessions[session_id]["page"]
        #         external = self._sessions[session_id]["external"]
        #     else:
        #         # session 不存在：创建一个“工作 tab”
        #         sess = await self.session_new(
        #             url=start_url or "about:blank",
        #             use_current_tab=use_current_tab,
        #             session_id=session_id,
        #         )
        #         created_here = True
        #         page = self._sessions[sess["session_id"]]["page"]
        #         external = self._sessions[sess["session_id"]]["external"]
        #         session_id = sess["session_id"]

        # 2) 运行 Agent：如果中途再报 CDP not initialized，自动重连并重试一次
        async def _run_once():
            safety_hint = (
                "\n\nIMPORTANT:\n"
                "- You are controlling a real browser.\n"
                "- Prefer staying in the current tab unless the task explicitly requires opening a new tab.\n"
                "- Do NOT close the browser.\n"
                "- When working with online Excel spreadsheets, please carefully check that the cells you enter are in the same column as the target cells. Also, pay attention to the header text columns and align the numbers with each other. Don't modify text cells.\n"
                "- When done, return the final answer clearly.\n"
            )
            agent_task = task.strip() + safety_hint

            # 3) “吃满功能”参数：用 introspection 过滤（0.11.2/不同版本都不炸）
            max_actions_per_step = int(os.getenv("BROWSER_USE_MAX_ACTIONS_PER_STEP", "15"))
            max_input_tokens = int(os.getenv("BROWSER_USE_MAX_INPUT_TOKENS", "128000"))
            max_failures = int(os.getenv("BROWSER_USE_MAX_FAILURES", "3"))
            retry_delay = int(os.getenv("BROWSER_USE_RETRY_DELAY_SEC", "2"))
            planner_interval = int(os.getenv("BROWSER_USE_PLANNER_INTERVAL", "1"))
            save_conv = os.getenv("BROWSER_USE_SAVE_CONVERSATION_PATH")  # 可选
            gen_gif = os.getenv("BROWSER_USE_GENERATE_GIF", "false").lower() in {"1","true","yes"}
            extend_system_message = os.getenv(
                "BROWSER_USE_EXTEND_SYSTEM_MESSAGE",
                "Be robust to popups/login walls. Use the current page content to answer precisely."
            )

            agent_kwargs = dict(
                task=agent_task,
                llm=self._llm,
                browser=self._browser,
                use_vision=True,
                use_vision_for_planner=True,
                validate_output=True,

                planner_llm=self._llm,
                planner_interval=planner_interval,
                page_extraction_llm=self._llm,

                max_actions_per_step=max_actions_per_step,
                max_input_tokens=max_input_tokens,
                max_failures=max_failures,
                retry_delay=retry_delay,

                save_conversation_path=save_conv,
                generate_gif=gen_gif,
                extend_system_message=extend_system_message,
            )

            agent = Agent(**_filter_kwargs(Agent.__init__, agent_kwargs))

            # 4) run
            history = await _agent_run(agent, max_steps=max_steps)
            return history

        try:
            try:
                history = await _run_once()
            except Exception as e:
                if "CDP client not initialized" in str(e):
                    await self._ensure_ready_or_reconnect()
                    history = await _run_once()
                else:
                    raise

            final_text = _extract_final_text(history)

            shot_b64 = None
            if return_screenshot and page is not None:
                try:
                    shot_b64 = await page.screenshot(full_page=screenshot_full_page)
                except Exception:
                    shot_b64 = None

            cur_url = None
            title = None
            try:
                cur_url = await page.get_url()
            except Exception:
                pass
            try:
                title = await page.get_title()
            except Exception:
                pass

            return {
                "ok": True,
                "mode": "cdp" if self._is_cdp_mode() else "local",
                "pid": os.getpid(),
                "session_id": session_id,
                "final": final_text,
                "page": {"url": cur_url, "title": title},
                "screenshot_base64": shot_b64,
                "external": external,
            }

        finally:
            # 多轮默认不 close；如果你显式 close=True，才会关
            if close and (page is not None) and (not external):
                try:
                    await self._browser.close_page(page)
                except Exception:
                    pass
                # 同步清 session
                async with self._lock:
                    if session_id in self._sessions:
                        self._sessions.pop(session_id, None)
                        if self._active_session_id == session_id:
                            self._active_session_id = None

    async def shutdown(self) -> Dict[str, Any]:
        """
        CDP mode: only close sessions created by this server; do NOT attempt to kill the user's Chrome.
        Local mode: stop browser session.
        """
        async with self._lock:
            browser = self._browser
            sessions_snapshot = list(self._sessions.items())
            self._sessions = {}
            self._active_session_id = None
            self._browser = None
            
        if browser is None:
            return {"shutdown": True, "reason": "not started", "pid": os.getpid()}

            # # close all INTERNAL sessions (external only if force via session_close)
            # for sid in list(self._sessions.keys()):
            #     info = self._sessions.get(sid)
            #     if not info:
            #         continue
            #     if info["external"]:
            #         continue
            #     try:
            #         await self.session_close(sid, force=True)
            #     except Exception:
            #         pass
        # 关掉我们自己 session 创建的 tab（external 不动）
        for sid, info in sessions_snapshot:
            try:
                if info.get("external"):
                    continue
                await browser.close_page(info["page"])
            except Exception:
                pass

        if not self._is_cdp_mode():
            try:
                await browser.stop()
            except Exception:
                pass

            # self._browser = None
            # self._sessions.clear()
            # self._active_session_id = None
        return {"shutdown": True, "mode": "cdp" if self._is_cdp_mode() else "local", "pid": os.getpid()}


# ----------------------------
# MCP Server
# ----------------------------
mcp = FastMCP("browser-use-cdp-server")
svc = BrowserService()


@mcp.tool()
async def browser_start() -> str:
    """Start browser service (local launch or attach via CDP). Returns mode/started as JSON string."""
    return json.dumps(await svc.ensure_started(), ensure_ascii=False)

@mcp.tool()
async def run_task(
    task: str,
    session_id: Optional[str] = None,
    max_steps: int = 40,
    start_url: Optional[str] = None,
    use_current_tab: bool = False,
    close: bool = False,
    return_screenshot: bool = False,
    screenshot_full_page: bool = False,
) -> str:
    """
    Run a long multi-step complex browser task in ONE tool call.
    Use browser_use.Agent internally (supports clicking, scrolling, drag/drop, screenshots, QA, etc).
    Returns JSON with final text and optional screenshot.
    """
    out = await svc.run_task(
        task=task,
        session_id=session_id,
        max_steps=max_steps,
        start_url=start_url,
        use_current_tab=use_current_tab,
        close=close,
        return_screenshot=return_screenshot,
        screenshot_full_page=screenshot_full_page,
    )
    return json.dumps(out, ensure_ascii=False)

@mcp.tool()
async def session_new(url: Optional[str] = None, use_current_tab: bool = False) -> str:
    """Create a browsing session. Default opens a NEW tab (safe for CDP). Returns session info as JSON string."""
    return json.dumps(await svc.session_new(url=url, use_current_tab=use_current_tab), ensure_ascii=False)


@mcp.tool()
async def session_switch(session_id: str) -> str:
    """Switch active session to session_id. Returns active session id as JSON string."""
    return json.dumps(await svc.session_switch(session_id=session_id), ensure_ascii=False)


@mcp.tool()
async def session_list() -> str:
    """List sessions with url/external/active flags. Returns list as JSON string."""
    return json.dumps(await svc.session_list(), ensure_ascii=False)


@mcp.tool()
async def session_close(session_id: Optional[str] = None, force: bool = False) -> str:
    """Close a session tab. In CDP mode, external tab won't be closed unless force=True. Returns result as JSON string."""
    return json.dumps(await svc.session_close(session_id=session_id, force=force), ensure_ascii=False)


@mcp.tool()
async def goto(url: str, session_id: Optional[str] = None) -> str:
    """Navigate current session page to url. Returns new url/title as JSON string."""
    return json.dumps(await svc.goto(url=url, session_id=session_id), ensure_ascii=False)


@mcp.tool()
async def click_css(selector: str, session_id: Optional[str] = None) -> str:
    """Click an element by CSS selector on current session page. Returns clicked selector as JSON string."""
    return json.dumps(await svc.click_css(selector=selector, session_id=session_id), ensure_ascii=False)


@mcp.tool()
async def click_prompt(prompt: str, session_id: Optional[str] = None) -> str:
    """Click an element matched by natural-language prompt (LLM-assisted). Returns clicked prompt as JSON string."""
    return json.dumps(await svc.click_prompt(prompt=prompt, session_id=session_id), ensure_ascii=False)


@mcp.tool()
async def fill_css(selector: str, text: str, session_id: Optional[str] = None) -> str:
    """Fill an input by CSS selector with text. Returns filled selector as JSON string."""
    return json.dumps(await svc.fill_css(selector=selector, text=text, session_id=session_id), ensure_ascii=False)


@mcp.tool()
async def fill_prompt(prompt: str, text: str, session_id: Optional[str] = None) -> str:
    """Fill an input matched by natural-language prompt with text (LLM-assisted). Returns filled prompt as JSON string."""
    return json.dumps(await svc.fill_prompt(prompt=prompt, text=text, session_id=session_id), ensure_ascii=False)


@mcp.tool()
async def screenshot(session_id: Optional[str] = None, full_page: bool = False) -> str:
    """Take a screenshot (base64) for current session page. Returns base64 string in JSON."""
    return json.dumps(await svc.screenshot(session_id=session_id, full_page=full_page), ensure_ascii=False)


@mcp.tool()
async def qa(question: str, session_id: Optional[str] = None) -> str:
    """Answer a question using ONLY current webpage content, with evidence snippets. Returns QA JSON."""
    return json.dumps(await svc.qa(question=question, session_id=session_id), ensure_ascii=False)

# @mcp.tool()
# async def browser_quick_qa(url: str, question: str, use_current_tab: bool = False, close: bool = True) -> str:
#     """
#     One-shot: start browser -> open tab -> QA -> (optional) close tab.
#     This avoids session state issues across multiple SSE calls.
#     Returns JSON string: {start, session, qa}
#     """
#     start = await svc.ensure_started()
#     session = await svc.session_new(url=url, use_current_tab=use_current_tab)
#     sid = session["session_id"]
#     qa_out = await svc.qa(question=question, session_id=sid)
#     if close:
#         await svc.session_close(session_id=sid, force=False)
#     return json.dumps({"start": start, "session": session, "qa": qa_out}, ensure_ascii=False)

@mcp.tool()
async def shutdown() -> str:
    """Shutdown browser tool. CDP mode: only closes tabs created by this server. Local mode: stop browser."""
    return json.dumps(await svc.shutdown(), ensure_ascii=False)


def build_app() -> FastAPI:
    app = FastAPI(title="Browser Use MCP (SSE)")

    # 和你 agent 端 /messages/?session_id=... 的调用方式对齐
    sse = SseServerTransport("/messages/")

    # POST /messages 用于客户端发消息（openjiuwen runner 就是这么打的）
    app.router.routes.append(Mount("/messages", app=sse.handle_post_message))

    # GET /sse 用于建立 SSE 连接
    @app.get("/sse")
    async def sse_endpoint(request: Request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as (
            read_stream,
            write_stream,
        ):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Browser Use MCP Server (SSE)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8930)
    args = parser.parse_args()

    app = build_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("Minimal Browser Use MCP Server (CDP)")
#     parser.add_argument("--transport", choices=["stdio", "sse"], default="sse")
#     parser.add_argument("--host", type=str, default="127.0.0.1")
#     parser.add_argument("--port", type=int, default=8930)
#     args = parser.parse_args()

#     if args.transport == "stdio":
#         mcp.run(transport="stdio")
#     else:
#         mcp.run(transport="sse")


#     # if args.transport == "stdio":
#     #     mcp.run(transport="stdio")
#     # else:
#     #     mcp.run(transport="sse", host=args.host, port=args.port)
