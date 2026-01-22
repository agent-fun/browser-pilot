# auto_browser_cdp.py
from __future__ import annotations

import os
import base64
import subprocess
import atexit
import signal
import inspect
import socket
import time
import asyncio
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
load_dotenv(verbose=True)

# Ensure browser-use never reuses an existing Playwright instance; set before importing browser_use.
os.environ.setdefault("BROWSER_USE_REUSE", "0")

# --- browser_use imports (version compatible) ---
from browser_use import Agent, Browser

# BrowserProfile is kept for backward compatibility in some versions
try:
    from browser_use import BrowserProfile  # newer docs mention BrowserProfile (legacy)
except Exception:
    BrowserProfile = None  # type: ignore

# Use browser_use's model wrappers (NOT langchain_openai.ChatOpenAI)
try:
    from browser_use import ChatOpenAI
except Exception:
    # some versions may place wrappers elsewhere; keep a fallback if needed
    from browser_use.llm import ChatOpenAI  # type: ignore

# --- your repo imports (keep your existing behavior) ---
from src.super_agent.tool.browser import Controller
from src.super_agent.tool.browser.action_memory import BrowserSessionContext
from src.super_agent.tool.logger import bootstrap_logger

logger = bootstrap_logger()

MAX_VISION_IMAGE_BYTES = 4_800_000  # keep under common 5MB vision limits


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists():
            return candidate
    return current.parents[-1]


def assemble_project_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def find_chrome_executable() -> str | None:
    """Find Chrome executable at common Windows locations."""
    if os.name != "nt":
        return None
    common_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return None


def get_browseruse_user_data_dir() -> str:
    # keep your original default
    return str(Path.home() / ".config" / "browseruse" / "profiles" / "default")


def _is_google_maps_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    return host.endswith("google.com") and ("maps." in host or path.startswith("/maps"))


class AutoBrowserUseTool:
    """
    Close-to-your-original structure, but:
    - BrowserConfig removed -> use Browser(...) / BrowserProfile fallback
    - Use browser_use.ChatOpenAI wrapper to avoid 'provider' error
    - CDP mode auto switch: if env BROWSER_USE_CDP_URL is set -> attach to remote Chrome via CDP
    - CDP mode cleanup: do NOT call browser.close(); only close pages created during this run
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "openrouter",
        timeout_seconds: float | None = None,
        cancel_grace_seconds: float | None = None,
    ):
        self.api_key = api_key
        self.model_id = model_id

        self.timeout_seconds = float(timeout_seconds or os.getenv("AUTO_BROWSER_TIMEOUT_SECONDS", "0") or 0)
        self.cancel_grace_seconds = float(cancel_grace_seconds or os.getenv("AUTO_BROWSER_CANCEL_GRACE_SECONDS", "10") or 10)

        # PDF server (keep)
        self.http_server_path = assemble_project_path("./browser/http_server")
        self.http_save_path = assemble_project_path("./browser/http_server/local")
        os.makedirs(self.http_save_path, exist_ok=True)

        self.http_server_port = int(os.getenv("AUTO_BROWSER_HTTP_PORT", "0")) or self._find_free_port()
        self.server_proc: subprocess.Popen | None = None

        self.browser: Browser | None = None
        self._started_at: float | None = None

        # CDP mode switch
        self.cdp_url = (os.getenv("BROWSER_USE_CDP_URL") or "").strip() or None
        self._cdp_mode = self.cdp_url is not None

        self._init_pdf_server()

    # ---------------- PDF server ----------------
    def _find_free_port(self, start: int = 8080) -> int:
        s = socket.socket()
        for p in range(start, start + 200):
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

    def _init_pdf_server(self) -> None:
        if self.http_server_port <= 0:
            logger.warning("AUTO_BROWSER_HTTP_PORT is invalid; skip pdf server.")
            return

        if self._is_port_open(self.http_server_port):
            return

        python_exec = os.getenv("AUTO_BROWSER_PYTHON") or os.sys.executable
        self.server_proc = subprocess.Popen(
            [python_exec, "-m", "http.server", str(self.http_server_port)],
            cwd=str(self.http_server_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        for _ in range(25):
            time.sleep(0.2)
            if self._is_port_open(self.http_server_port):
                break

        @atexit.register
        def _shutdown_server():
            if self.server_proc and self.server_proc.poll() is None:
                try:
                    if os.name == "nt":
                        self.server_proc.terminate()
                    else:
                        self.server_proc.send_signal(signal.SIGTERM)
                    self.server_proc.wait(timeout=5)
                except Exception:
                    try:
                        self.server_proc.kill()
                    except Exception:
                        pass

    # ---------------- LLM ----------------
    def _build_llm(self, model: str, base_url: str, temperature: float, timeout: int = 60):
        """
        Use browser_use.ChatOpenAI to avoid: 'ChatOpenAI' object has no attribute 'provider'
        Docs: use ChatOpenAI wrapper; supports OpenAI-compatible endpoints/custom URL. :contentReference[oaicite:2]{index=2}
        """
        return ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=2,
        )

    # ---------------- Browser (local launch / CDP attach) ----------------
    def _make_browser(self) -> Browser:
        chrome_path = find_chrome_executable()
        user_data_dir = os.getenv("BROWSER_USE_USER_DATA_DIR") or get_browseruse_user_data_dir()
        os.makedirs(user_data_dir, exist_ok=True)

        # NOTE: browser-use new API uses Browser(..., args=[...], user_data_dir=..., profile_directory=...)
        # Your original launch_args -> args mapping.
        args = [
            f"--user-data-dir={user_data_dir}",
            "--profile-directory=Default",
            "--start-maximized",
            "--enable-gpu",
            "--ignore-gpu-blocklist",
            "--enable-accelerated-2d-canvas",
            "--enable-gpu-rasterization",
            "--enable-features=CanvasOopRasterization,UseSkiaRenderer",
            "--use-angle=d3d11",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        # Permissions: normalize your old ("clipboard-read","clipboard-write") into the browser-use default style
        # browser-use docs default uses 'clipboardReadWrite'; keep your geolocation. :contentReference[oaicite:3]{index=3}
        permissions = ["clipboardReadWrite", "geolocation"]

        browser_kwargs = dict(
            headless=False,
            channel="chrome",                 # local launch
            executable_path=chrome_path,      # may be None; Browser should handle
            user_data_dir=user_data_dir,
            profile_directory="Default",
            args=args,
            disable_security=True,
            user_agent=os.getenv(
                "BROWSER_USE_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
            ),
            locale="en-US",
            timezone_id="Asia/Singapore",
            permissions=permissions,
            # keep_alive=False in local mode so it exits cleanly; CDP mode we won't close anyway.
            keep_alive=False,
        )

        # CDP attach mode: connect to existing Chrome instance on client. :contentReference[oaicite:4]{index=4}
        if self._cdp_mode:
            # Some versions accept cdp_url in Browser(...); if not, fallback to BrowserProfile below.
            try:
                return Browser(cdp_url=self.cdp_url, **browser_kwargs)  # type: ignore[arg-type]
            except TypeError:
                if BrowserProfile is None:
                    raise
                profile = BrowserProfile(cdp_url=self.cdp_url, **browser_kwargs)  # type: ignore
                return Browser(browser_profile=profile)  # type: ignore

        # Local launch
        try:
            return Browser(**browser_kwargs)
        except TypeError:
            # fallback to BrowserProfile
            if BrowserProfile is None:
                raise
            profile = BrowserProfile(**browser_kwargs)  # type: ignore
            return Browser(browser_profile=profile)  # type: ignore

    # ---------------- “Only close new tabs” for CDP mode ----------------
    def _iter_contexts_and_pages(self):
        """
        Best-effort introspection to find Playwright contexts/pages under browser_use.Browser wrapper.
        Keep it defensive: browser_use internal structure may change.
        """
        b = self.browser
        if b is None:
            return [], []

        contexts = []
        pages = []

        # probe both browser and browser.session holders
        holders = [b, getattr(b, "session", None)]
        for h in holders:
            if not h:
                continue

            # possible context containers
            for attr in ("context", "playwright_context"):
                ctx = getattr(h, attr, None)
                if ctx and ctx not in contexts:
                    contexts.append(ctx)

            ctxs = getattr(h, "contexts", None) or getattr(h, "browser_contexts", None)
            if ctxs:
                for ctx in list(ctxs):
                    if ctx and ctx not in contexts:
                        contexts.append(ctx)

        # collect pages
        for ctx in contexts:
            try:
                ps = getattr(ctx, "pages", None)
                if ps:
                    for p in list(ps):
                        if p and p not in pages:
                            pages.append(p)
            except Exception:
                continue

        return contexts, pages

    def _snapshot_pages(self) -> set[int]:
        _, pages = self._iter_contexts_and_pages()
        return {id(p) for p in pages}

    async def _close_new_pages_only(self, before_ids: set[int]) -> None:
        """
        In CDP mode, DO NOT call browser.close().
        Only close pages created during this run (difference by object id).
        """
        _, pages = self._iter_contexts_and_pages()
        for p in pages:
            if id(p) not in before_ids:
                try:
                    await p.close()
                except Exception:
                    pass

    async def _safe_close_browser_local(self) -> None:
        """
        Local mode: safe to close everything.
        """
        b = self.browser
        if b is None:
            return
        close_fn = getattr(b, "close", None)
        if callable(close_fn):
            try:
                maybe = close_fn()
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception:
                pass
        self.browser = None

    # ---------------- main run ----------------
    async def _browser_task(self, task: str) -> str:
        # Controller / memory (keep)
        try:
            controller = Controller(http_save_path=str(self.http_save_path), http_server_port=self.http_server_port)

            storage_root = assemble_project_path("./website_database")
            task_id = os.getenv("MIROFLOW_CURRENT_TASK_ID") or os.getenv("MIROFLOW_TASK_ID")
            storage_path = storage_root / (f"{task_id}_site_action_memory.jsonl" if task_id else "site_action_memory.jsonl")

            session_context = BrowserSessionContext(
                storage_path=storage_path,
                session_id=task_id,
                load_existing=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return f"Cannot instantiate controller/memory: {e}"

        # LLM (keep your “fast/slow/planner” pattern but with browser_use.ChatOpenAI)
        try:
            # OpenAI compatible endpoint:
            # - if you use OpenRouter: base_url=https://openrouter.ai/api/v1 , model="anthropic/claude-..."
            # - if you use OpenAI: base_url=https://api.openai.com/v1 , model="gpt-5"
            base_url = os.getenv("BROWSER_USE_LLM_BASE_URL") or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"

            fast_model_name = os.getenv("BROWSER_USE_FAST_MODEL", "google/gemini-2.5-flash")
            slow_model_name = os.getenv("BROWSER_USE_SLOW_MODEL", "google/gemini-2.5-pro")
            planner_model_name = os.getenv("BROWSER_USE_PLANNER_MODEL", "anthropic/claude-sonnet-4.5")

            fast_model = self._build_llm(fast_model_name, base_url=base_url, temperature=0.8, timeout=60)
            slow_model = self._build_llm(slow_model_name, base_url=base_url, temperature=0.3, timeout=60)
            planner_model = self._build_llm(planner_model_name, base_url=base_url, temperature=0.3, timeout=60)

            controller.bind_memory_summarizer(fast_model)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return f"Error while instantiating LLMs: {e}"

        # Browser (local vs CDP)
        before_page_ids: set[int] = set()
        try:
            self.browser = self._make_browser()
            if self._cdp_mode:
                # Snapshot current pages so we only close new ones later
                before_page_ids = self._snapshot_pages()

            logger.info("=== Browser-use Agent running (cdp_mode=%s) ===", self._cdp_mode)

            browser_agent = Agent(
                task=task,
                llm=fast_model,
                enable_memory=True,
                controller=controller,
                page_extraction_llm=slow_model,
                planner_llm=planner_model,
                browser=self.browser,
                max_failures=3,
                context=session_context,
                max_actions_per_step=4,
                use_vision=True,
                use_vision_for_planner=True,
                planner_interval=4,
                generate_gif=False,
                save_conversation_path=None,
                validate_output=False,
            )

            session_context.bind_message_manager(browser_agent.message_manager)

            history = await browser_agent.run(max_steps=int(os.getenv("BROWSER_USE_MAX_STEPS", "30")))

            # Extract last meaningful content
            contents = []
            try:
                for entry in history.extracted_content():
                    if entry and entry.strip():
                        contents.append(entry.strip())
            except Exception:
                pass

            if contents:
                return contents[-1]
            return "AutoBrowser agent returned no extracted content."

        except asyncio.CancelledError:
            logger.warning("Browser task cancelled; propagating.")
            raise
        except Exception as e:
            return f"Agent couldn't run due to: {e}"
        finally:
            # Cleanup policy:
            # - CDP mode: only close new tabs, do NOT browser.close()
            # - Local mode: close the whole browser instance
            try:
                if self._cdp_mode:
                    await self._close_new_pages_only(before_page_ids)
                else:
                    await self._safe_close_browser_local()
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")

    async def forward(self, task: str) -> str:
        self._started_at = time.monotonic()
        return await self._browser_task(task)


async def browse(task: str, api_key: str | None = None) -> str:
    """
    Convenience wrapper.
    If api_key is None, fallback to OPENROUTER_API_KEY / OPENAI_API_KEY.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return "[ERROR]: Missing OPENROUTER_API_KEY or OPENAI_API_KEY."

    tool = None
    try:
        tool = AutoBrowserUseTool(api_key=key)
        return await tool.forward(task)
    finally:
        # ensure pdf server process is terminated
        if tool and tool.server_proc and tool.server_proc.poll() is None:
            try:
                tool.server_proc.terminate()
            except Exception:
                pass
