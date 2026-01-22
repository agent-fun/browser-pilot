import os
import base64
import subprocess
import atexit
import httpx
import signal
import inspect
from urllib.parse import urlsplit
from dotenv import load_dotenv
load_dotenv(verbose=True)
import sys, socket, time

# Ensure browser-use never reuses an existing Playwright instance; set before importing library.
os.environ["BROWSER_USE_REUSE"] = "0"

from browser_use import Agent, Browser, BrowserConfig
from src.super_agent.tool.browser import Controller
from src.super_agent.tool.browser.action_memory import BrowserSessionContext
import requests
from src.super_agent.tool.logger import bootstrap_logger
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.super_agent.tool.browser.litellm import LiteLLMModel
from openai import AsyncOpenAI

from src.super_agent.tool.mcp_servers import model_manager
from pathlib import Path
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from browser_use.browser.context import BrowserContext
import asyncio

PROXIES = None
VERIFY_SSL = False if os.environ.get("PPLX_VERIFY_SSL", "0").lower() in {"", "0", "false", "no"} else True
DEFAULT_TIMEOUT = float(os.environ.get("PPLX_HTTP_TIMEOUT", 30))
RETRY_MAX_TRIES = int(os.environ.get("PPLX_HTTP_RETRIES", 3))

API_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_VISION_IMAGE_BYTES = 4_800_000  # stay under common 5MB vision payload limits
logger = bootstrap_logger()
def _get_env_timeout(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

AUTO_BROWSER_TIMEOUT_SECONDS = _get_env_timeout("AUTO_BROWSER_TIMEOUT_SECONDS", 0)
AUTO_BROWSER_CANCEL_GRACE_SECONDS = _get_env_timeout("AUTO_BROWSER_CANCEL_GRACE_SECONDS", 10)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_project_root():
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists():
            return candidate
    return current.parents[-1]

def assemble_project_path(path):
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = get_project_root() / path_obj
    return path_obj


def find_chrome_executable():
    """Find Chrome executable at common Windows locations."""
    if os.name != "nt":
        return None
    
    common_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def get_browseruse_user_data_dir():
    """Get the browser-use user data directory path."""
    return str(Path.home() / ".config" / "browseruse" / "profiles" / "default")


def _is_google_maps_url(url: str | None) -> bool:
    """Return True when URL belongs to Google Maps (including /maps paths)."""
    if not url:
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return (
        host.endswith("google.com")
        and ("maps." in host or path.startswith("/maps"))
    )


def _patch_browser_context_screenshot():
    """Extend screenshot handling to wait longer before timing out."""
    if getattr(BrowserContext, "_miroflow_patched_screenshot", False):
        return

    async def _capture(page, *, full_page: bool, timeout_ms: int, quality: int):
        try:
            return await page.screenshot(
                full_page=full_page,
                animations="disabled",
                caret="initial",
                timeout=timeout_ms,
                type="jpeg",
                quality=quality,
            )
        except PlaywrightTimeoutError:
            logger.warning("Screenshot timed out; retrying with relaxed options.")
            try:
                return await page.screenshot(
                    full_page=full_page,
                    animations=None,
                    caret="initial",
                    timeout=timeout_ms,
                    type="jpeg",
                    quality=quality,
                )
            except PlaywrightTimeoutError:
                logger.error("Screenshot timed out on retry; falling back to minimal capture.")
                return None

    async def take_screenshot(self, full_page: bool = False) -> str:  # type: ignore[override]
        page = await self.get_agent_current_page()

        timeout_ms = 60_000
        current_url = getattr(page, "url", None)
        skip_network_idle = _is_google_maps_url(current_url)

        if skip_network_idle:
            # Google Maps keeps the network busy forever; waiting for networkidle blocks steps.
            try:
                await page.wait_for_load_state(timeout=min(15_000, timeout_ms))
            except PlaywrightTimeoutError:
                logger.debug("Load wait timed out on Google Maps; proceeding with screenshot.")
        else:
            try:
                await page.wait_for_load_state(state="networkidle", timeout=timeout_ms)
            except PlaywrightTimeoutError:
                try:
                    await page.wait_for_load_state(timeout=timeout_ms)
                except PlaywrightTimeoutError:
                    logger.warning("Load state wait timed out before screenshot; continuing anyway.")

        # Try a reasonably high JPEG quality first; degrade if payload exceeds limits.
        screenshot = await _capture(page, full_page=full_page, timeout_ms=timeout_ms, quality=80)

        if screenshot and len(screenshot) > MAX_VISION_IMAGE_BYTES:
            for quality in (65, 50, 40, 30, 20):
                candidate = await _capture(
                    page, full_page=full_page, timeout_ms=timeout_ms, quality=quality
                )
                if candidate and len(candidate) <= MAX_VISION_IMAGE_BYTES:
                    screenshot = candidate
                    break

        if screenshot and len(screenshot) > MAX_VISION_IMAGE_BYTES and full_page:
            # Drop full_page to shrink the shot if it is still too large.
            for quality in (65, 50, 40, 30, 20):
                candidate = await _capture(
                    page, full_page=False, timeout_ms=timeout_ms, quality=quality
                )
                if candidate and len(candidate) <= MAX_VISION_IMAGE_BYTES:
                    screenshot = candidate
                    break

        if screenshot is None:
            screenshot = await _capture(page, full_page=False, timeout_ms=0, quality=40) or b""

        return base64.b64encode(screenshot).decode("utf-8")

    BrowserContext.take_screenshot = take_screenshot  # type: ignore[assignment]
    BrowserContext._miroflow_patched_screenshot = True  # type: ignore[attr-defined]


# _patch_browser_context_screenshot()


def _clear_browser_use_reuse_marker():
    """Remove reuse marker so browser_use won't attach to an existing Playwright instance."""
    try:
        reuse_file = Path.home() / ".browser_use" / "reuse.json"
        if reuse_file.exists():
            reuse_file.unlink()
    except Exception as e:
        logger.debug(f"Failed to clear browser_use reuse marker: {e}")


class AutoBrowserUseTool():
    def __init__(self,
                api_key,
                model_id: str = "gpt-5",
                timeout_seconds: float | None = None,
                cancel_grace_seconds: float | None = None,
                 ):

        super(AutoBrowserUseTool, self).__init__()

        self.api_key = api_key
        self.model_id = model_id if model_id else DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else AUTO_BROWSER_TIMEOUT_SECONDS
        self.cancel_grace_seconds = cancel_grace_seconds if cancel_grace_seconds is not None else AUTO_BROWSER_CANCEL_GRACE_SECONDS
        self.url = API_URL
        self.http_server_path = assemble_project_path("./browser/http_server")
        self.http_save_path = assemble_project_path("./browser/http_server/local")
        os.makedirs(self.http_save_path, exist_ok=True)

        self.http_server_port = self._find_free_port()  # 新增：可配置端口
        self.server_proc = None       # 新增：保存进程句柄

        self.browser = None
        self._started_at: float | None = None

        self._init_pdf_server()

    def _find_free_port(self, start=8080):
        s = socket.socket()
        for p in range(start, start+200):
            try:
                s.bind(("127.0.0.1", p))
                s.close()
                return p
            except OSError:
                continue
        return 0

    def _is_port_open(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _init_pdf_server(self):
        # 若端口被占用则直接认为可用；否则尝试启动
        if not self._is_port_open(self.http_server_port):
            cmd = [sys.executable, "-m", "http.server", str(self.http_server_port)]
        python_exec = sys.executable
        server_proc = subprocess.Popen(
            # ["python", "-m", "http.server", "8080"],
            [python_exec, "-m", "http.server", str(self.http_server_port)],
            cwd=self.http_server_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # preexec_fn=None,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.server_proc = server_proc
        
        # 健康检查：最多等 5 秒
        for _ in range(25):
            time.sleep(0.2)
            if self._is_port_open(self.http_server_port):
                break

        @atexit.register
        def shutdown_server():
            print("Shutting down http.server...")
            if self.server_proc and self.server_proc.poll() is None:
                try:
                    if os.name == "nt":
                        server_proc.terminate()
                    else:
                        server_proc.send_signal(signal.SIGTERM)
                    server_proc.wait(timeout=5)
                except Exception as e:
                    # print("Force killing server...")
                    server_proc.kill()

    def _format_session_visit_summary(self, session_context: BrowserSessionContext | None) -> str:
        """Render a human-readable summary of sites and actions visited this run."""
        if not session_context:
            return "Visited sites summary: No website activity recorded during this session."
        memory = getattr(session_context, "memory", None)
        if not memory or not hasattr(memory, "get_session_observations"):
            return "Visited sites summary: Tracking is not available."
        site_actions = memory.get_session_observations()
        if not site_actions:
            return "Visited sites summary: No website interactions recorded during this run."
        lines: list[str] = ["Visited sites summary:"]
        for idx, (site, observations) in enumerate(site_actions.items(), start=1):
            lines.append(f"{idx}. {site}")
            for observation in observations:
                flattened = " ".join(
                    part.strip() for part in observation.splitlines() if part.strip()
                )
                lines.append(f"   - {flattened}")
        return "\n".join(lines)

    async def _close_context_and_tabs(self, ctx) -> None:
        """Close all pages within a Playwright context, then the context itself."""
        if not ctx:
            return
        pages = getattr(ctx, "pages", None)
        if pages:
            for page in list(pages):
                try:
                    await page.close()
                except Exception as e:
                    logger.debug(f"Error closing page: {e}")
        close_ctx = getattr(ctx, "close", None)
        if callable(close_ctx):
            try:
                maybe_ctx = close_ctx()
                if inspect.isawaitable(maybe_ctx):
                    await maybe_ctx
            except Exception as e:
                logger.debug(f"Error closing context: {e}")

    async def _safe_close_browser(self) -> None:
        """Close all tabs and the browser instance (best-effort)."""
        browser = getattr(self, "browser", None)
        if browser is None:
            return

        # Collect likely contexts (browser, browser.session, etc.)
        contexts = []
        for holder in (browser, getattr(browser, "session", None)):
            if not holder:
                continue
            for attr in ("context", "playwright_context", "playwright_browser"):
                ctx = getattr(holder, attr, None)
                if ctx and ctx not in contexts:
                    contexts.append(ctx)
            ctxs = getattr(holder, "contexts", None) or getattr(holder, "browser_contexts", None)
            if ctxs:
                for ctx in ctxs:
                    if ctx and ctx not in contexts:
                        contexts.append(ctx)

        for ctx in contexts:
            try:
                await self._close_context_and_tabs(ctx)
            except Exception as e:
                logger.debug(f"Error while closing context: {e}")

        close_fn = getattr(browser, "close", None)
        if callable(close_fn):
            try:
                maybe_result = close_fn()
                if inspect.isawaitable(maybe_result):
                    await maybe_result
            except Exception as e:
                logger.warning(f"Error closing browser instance: {e}")

        self.browser = None

    async def _browser_task(self, task):
        try:
            _clear_browser_use_reuse_marker()
            controller = Controller(http_save_path=self.http_save_path, http_server_port=self.http_server_port)
            storage_root = assemble_project_path("./website_database")
            task_id = os.getenv("MIROFLOW_CURRENT_TASK_ID") or os.getenv("MIROFLOW_TASK_ID")
            if task_id:
                storage_path = storage_root / f"{task_id}_site_action_memory.jsonl"
            else:
                storage_path = storage_root / "site_action_memory.jsonl"
            session_context = BrowserSessionContext(
                storage_path=storage_path,
                session_id=task_id,
                # Persist browsing history across sub-agent invocations within the same task.
                load_existing=False,
            )
        except asyncio.CancelledError:
            # Propagate cancellation so upstream timeout logic can stop the run.
            raise
        except Exception as e:
            return f"Cannot instantiate controller: {e}"
            
        try:
            # Not working with the browser-use because of ainvoke introduction
            """client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.url,
                http_client=httpx.AsyncClient(verify=False),
            )
            model = LiteLLMModel(
                    model_id=self.model_id,
                    http_client=client,
                    custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
            )"""
            
            # We might need to create a model registry - if not instantiating a model everytime we use browser-use agent, it will take too long to run 
            planner_model = None
            if 'gemini' in self.model_id: 
                os.environ['GOOGLE_API_KEY'] = self.api_key
                fast_model = ChatGoogleGenerativeAI(
                    model = 'gemini-2.5-pro', # gemini flash very fast but not very good performance-wise: very poor/repetitive decisions
                    google_api_key = self.api_key, 
                    # convert_system_message_to_human=True,
                    temperature=0.7,
                    max_output_tokens = 8192,
                    timeout=60,
                )
                slow_model = ChatGoogleGenerativeAI(
                    model = 'gemini-2.5-pro', 
                    google_api_key = self.api_key, 
                    # convert_system_message_to_human=True,
                    temperature=0.3,
                    max_output_tokens = 8192,
                    timeout=60,
                )
                planner_model = slow_model
                # Using openrouter instead of google api
                """os.environ['OPENROUTER_API_KEY'] = self.api_key
                fast_model = ChatOpenAI(
                    model = 'google/gemini-2.5-flash', 
                    api_key = self.api_key,
                    base_url = "https://openrouter.ai/api/v1",
                    timeout = 30,
                    max_retries=5,
                    http_client = httpx.Client(verify=False),
                    http_async_client = httpx.AsyncClient(verify=False)
                )
                slow_model = ChatOpenAI(
                    model = 'google/gemini-2.5-pro', 
                    api_key = self.api_key,
                    base_url = "https://openrouter.ai/api/v1",
                    timeout = 30,
                    max_retries=5,
                    http_client = httpx.Client(verify=False),
                    http_async_client = httpx.AsyncClient(verify=False)
                )"""
                
            elif 'gpt' in self.model_id:
                os.environ['OPENROUTER_API_KEY'] = self.api_key
                fast_model = ChatOpenAI(
                    model = 'google/gemini-2.5-flash', 
                    api_key = self.api_key,
                    base_url = "https://openrouter.ai/api/v1",
                    timeout = 30,
                    max_retries=2,
                    http_client = httpx.Client(verify=False),
                    http_async_client = httpx.AsyncClient(verify=False)
                )
                slow_model = ChatOpenAI(
                    model = 'gpt-5', 
                    api_key = self.api_key,
                    base_url = self.url,
                    timeout = 30,
                    max_retries=2,
                    http_client = httpx.Client(verify=False),
                    http_async_client = httpx.AsyncClient(verify=False)
                )
                planner_model = slow_model

            elif 'claude' in self.model_id:
                os.environ['OPENROUTER_API_KEY'] = self.api_key
                fast_model = ChatOpenAI(
                    model = 'google/gemini-2.5-flash',
                    api_key = self.api_key,
                    base_url = "https://openrouter.ai/api/v1",
                    timeout = 60,
                    max_retries=2,
                    http_client = httpx.Client(verify=False),
                    http_async_client = httpx.AsyncClient(verify=False),
                    temperature = 0.8,
                )
                slow_model = ChatOpenAI(
                    model = 'google/gemini-2.5-pro', 
                    api_key = self.api_key,
                    base_url = "https://openrouter.ai/api/v1",
                    timeout = 60,
                    max_retries=2,
                    http_client = httpx.Client(verify=False),
                    http_async_client = httpx.AsyncClient(verify=False),
                    temperature = 0.8,
                )
                planner_model = ChatOpenAI(
                    model = 'anthropic/claude-sonnet-4.5', 
                    api_key = self.api_key,
                    base_url = "https://openrouter.ai/api/v1",
                    timeout = 60,
                    max_retries=2,
                    http_client = httpx.Client(verify=False),
                    http_async_client = httpx.AsyncClient(verify=False),
                    temperature = 0.8,
                )

            else:
                raise Exception("Model not defined!")

            # Tried to create model registry, but unable to
            # model = model_manager.registered_models[self.model_id]
            controller.bind_memory_summarizer(fast_model)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.info("Cannot instantiate model")
            return f"Error while instantiating model for browser use: {e}"

        # Code to check model has the .ainvoke attribute
        """try:
            _ = await model.ainvoke("ping")
        except Exception as e:
            return f"LLM probe failed: {e}"""

        #launch_channel = "chrome"
        # Fallback to 'chromium' if Chrome channel isn't available
        #if os.name == "nt" and not os.environ.get("BROWSER_USE_ALLOW_CHROME_CHANNEL", ""):
        #    launch_channel = "chromium"

        # Find Chrome executable dynamically
        chrome_path = find_chrome_executable()
        if not chrome_path:
            logger.warning("Chrome executable not found at common locations. Trying without chrome_instance_path.")
        
        # Get user data directory dynamically
        user_data_dir = get_browseruse_user_data_dir()
        os.makedirs(user_data_dir, exist_ok=True)

        # Build BrowserConfig with conditional chrome_instance_path
        browser_config_kwargs = {
                # chrome_binary_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe", # This code should not be working
                # Let Chrome control window size via launch args (start maximized)
                "viewport": None,
                "headless": False,
                "channel": 'chrome',
                "keep_alive": False,
                "disable_security": True,
                "navigation_timeout_ms": 8000,       # 导航超时 ↑
                "screenshot_timeout_ms": 15000,       # 截图超时 ↑
                "wait_until": "domcontentloaded",
                "user_data_dir": user_data_dir,  # Dynamically determined path
                "launch_args": [
                    f"--user-data-dir={user_data_dir}",
                    "--profile-directory=Default",
                    "--start-maximized",
                    "--enable-gpu",
                    "--ignore-gpu-blocklist",
                    "--enable-accelerated-2d-canvas",
                    "--enable-gpu-rasterization",
                    "--enable-features=CanvasOopRasterization,UseSkiaRenderer",
                    # On Windows, ANGLE D3D11 is best for WebGL:
                    "--use-angle=d3d11",
                    # If d3d11 is problematic on your machine, try one of:
                    # "--use-angle=gl", "--use-gl=desktop"
                    # "--high-dpi-support=1",
                    # "--disable-renderer-backgrounding",
                    # "--disable-background-timer-throttling",
                    # "--disable-backgrounding-occluded-windows",
                    "--no-first-run",
                    "--no-default-browser-check",
                    # "--disable-features=IsolateOrigins,site-per-process",
                ],
                "new_context_kwargs": {
                    "ignore_https_errors": True,
                    # Use the full window size Playwright creates (maximized)
                    "viewport": None,
                    "locale": "en-US",
                    "timezone_id": "Asia/Singapore",
                    "permissions": ["clipboard-read","clipboard-write", "geolocation"],
                },
                "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.7444.163 Safari/537.36',
        }
        
        # Only include chrome_instance_path if Chrome was found
        if chrome_path:
            browser_config_kwargs["chrome_instance_path"] = chrome_path
        
        browser = Browser(
            config=BrowserConfig(**browser_config_kwargs)
        )
        self.browser = browser
        # This is to ensure go_back function can run - might slow down google maps/street view but it shouldn't be significant problem
        context_config = browser.config.new_context_config
        context_config.minimum_wait_page_load_time = 3
        context_config.wait_for_network_idle_page_load_time = 3.0
        context_config.maximum_wait_page_load_time = 5.0
        context_config.wait_between_actions = 2
        logger.info("Going to run the agent")
        try:
            browser_agent = Agent(
                task=task,
                llm=fast_model,
                enable_memory=True,
                controller=controller,
                page_extraction_llm=slow_model,
                planner_llm = planner_model,
                browser=browser,      
                max_failures = 3,
                context=session_context,
                max_actions_per_step = 4,
                use_vision = True,
                use_vision_for_planner=True,
                planner_interval = 4,
                generate_gif=False,
                save_conversation_path=None,
                validate_output=False,

            )
            session_context.bind_message_manager(browser_agent.message_manager)
            logger.info("===Browser running===")
            history = await browser_agent.run(max_steps=20)
            """contents = []
            contents.extend(r.extracted_content or "" for r in history.action_results())
            contents.append(f"steps={history.number_of_steps()}")
            contents.append(f"is_done={history.is_done()}")
            return "\n".join(contents)"""
            contents = [
                entry.strip()
                for entry in history.extracted_content()
                if entry and entry.strip()
            ]
            if not contents:
                reuse_count = session_context.memory.existing_site_hits
                if reuse_count:
                    logger.info(
                        "Browser-use reused site memory %d time(s) during this run.",
                        reuse_count,
                    )
                visits_summary = self._format_session_visit_summary(session_context)
                return (
                    "AutoBrowser agent returned no extracted content.\n\n"
                    f"Site memory reuse count: {reuse_count}\n\n{visits_summary}"
                )

            reuse_count = session_context.memory.existing_site_hits
            if reuse_count:
                logger.info(
                    "Browser-use reused site memory %d time(s) during this run.",
                    reuse_count,
                )
            visits_summary = self._format_session_visit_summary(session_context)
            final_text = contents[-1]
            final_text = (
                f"{final_text}\n\nSite memory reuse count: {reuse_count}\n\n{visits_summary}"
            )
            return final_text
        except asyncio.CancelledError:
            logger.warning("Browser task cancelled; propagating.")
            raise
        except Exception as e:
            return f"Agent couldn't run due to : {e}"
        finally:
            # Always tear down tabs and browser when the run ends.
            try:
                await self._safe_close_browser()
            except Exception as e:
                logger.debug(f"Error during browser shutdown: {e}")

    async def forward(self, task: str):
        """
        Automatically browse the web and extract information based on a given task.

        Args:
            task: The task to perform
        Returns:
            ToolResult with the task result
        """
        self._started_at = time.monotonic()
        return await self._browser_task(task)

async def browse(model_id: str, api_key: str, task: str, timeout: float = AUTO_BROWSER_TIMEOUT_SECONDS) -> str:
    """
    Convenience wrapper to run AutoBrowserUseTool in one call with timeout.
    
    Args:
        model_id: Model identifier (gemini, gpt, claude)
        api_key: API key for the model
        task: The browsing task to perform
        timeout: Maximum time in seconds (default uses AUTO_BROWSER_TIMEOUT_SECONDS env, 15 minutes if unset)
    
    Returns:
        Task result or timeout error message
    """
    tool = None
    try:
        tool = AutoBrowserUseTool(
            api_key=api_key,
            model_id=model_id,
            timeout_seconds=timeout,
            cancel_grace_seconds=AUTO_BROWSER_CANCEL_GRACE_SECONDS,
        )
        result = await tool.forward(task)
        return result

    except Exception as e:
        logger.error(f"Error in browse function: {e}", exc_info=True)
        return f"Browser task failed: {str(e)}"
        
    finally:
        # Cleanup: destroy the tool instance
        if tool is not None:
            await _cleanup_browser_tool(tool)
            del tool


async def _cleanup_browser_tool(tool: AutoBrowserUseTool):
    """
    Clean up browser tool resources.
    
    Args:
        tool: The AutoBrowserUseTool instance to clean up
    """
    try:
        # Terminate the HTTP server if it's running
        if hasattr(tool, 'server_proc') and tool.server_proc is not None:
            if tool.server_proc.poll() is None:  # Process is still running
                logger.info(f"Terminating HTTP server on port {tool.http_server_port}" )
                try:
                    if os.name == "nt":
                        tool.server_proc.terminate()
                    else:
                        tool.server_proc.send_signal(signal.SIGTERM)
                    
                    # Wait for graceful shutdown
                    try:
                        tool.server_proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.warning("HTTP server didn't terminate gracefully, killing it")
                        tool.server_proc.kill()
                        tool.server_proc.wait()
                except Exception as e:
                    logger.error(f"Error terminating HTTP server: {e}")
                    try:
                        tool.server_proc.kill()
                    except:
                        pass
        try:
            # Close any remaining pages and the browser itself.
            await tool._safe_close_browser()
        except Exception as e:
            logger.error(f"Error closing browser instance: {e}")

        # Additional cleanup if needed (browser instances, temp files, etc.)
        # You can add more cleanup logic here if necessary
        
    except Exception as e:
        logger.error(f"Error during browser tool cleanup: {e}", exc_info=True)

# Just for sanity check to see if tools are registered
if __name__=='__main__':
    import asyncio
    def _find_free_port(start=8080):
        s = socket.socket()
        for p in range(start, start+200):
            try:
                s.bind(("127.0.0.1", p))
                s.close()
                return p
            except OSError:
                continue
        return 0

    async def main():
        tmp = AutoBrowserUseTool(
            os.getenv("GEMINI_API_KEY"), 
            'gemini-2.5-flash'
            )
        result = await tmp.forward("""Go to Google Maps and search for coordinates 46.4192624,14.642093. Then switch to
                             Street View mode. Make sure to check if there are images from May 2021 available (you may need
                             to click on a clock icon or 'See more dates' to access historical imagery). If May 2021 is
                             available, select that date. Then orient the view slightly left of North (look at the compass
                             and turn slightly counter-clockwise from North). Describe what you see, particularly focusing
                             on any signs on the right side of the road from this perspective.""")
        # print(result)
    
    asyncio.run(main())
