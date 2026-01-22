from email import message
import os
from sqlite3 import Date
from dotenv import load_dotenv
load_dotenv(verbose=True)
import time
import enum
import json
import re
from typing import TypeVar, cast, Any
from patchright.async_api import ElementHandle, Page
from pydantic import BaseModel
import requests
import urllib.parse
from urllib.parse import quote, urlsplit, urlunsplit, urlparse, parse_qs
import asyncio
from browser_use import Agent
from browser_use.agent.views import ActionModel, ActionResult
from browser_use.browser.context import BrowserContext
from browser_use import BrowserConfig, Browser
from browser_use.controller.service import Controller as BrowserUseController
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import (
    ClickElementAction,
    CloseTabAction,
    DoneAction,
    DragDropAction,
    GoToUrlAction,
    InputTextAction,
    NoParamsAction,
    OpenTabAction,
    Position,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
from typing import Literal, Optional
from browser_use.utils import time_execution_sync
from openai import AsyncOpenAI

from src.super_agent.tool.browser.local_proxy import PROXY_URL, proxy_env
from src.super_agent.tool.browser.tools import Tool, ToolResult
from src.super_agent.tool.browser.utils.logger import logger
from src.super_agent.tool.browser.action_memory import BrowserSessionContext

Context = TypeVar('Context')

class ZoomToPointParams(BaseModel):
    # Choose ONE way to specify the target:
    selector: Optional[str] = None         # e.g. "canvas", "img[alt*='sign']"
    text: Optional[str] = None             # visible text near the target
    x: Optional[float] = None              # normalized [0..1] viewport x
    y: Optional[float] = None              # normalized [0..1] viewport y

    steps: int = 6                         # how many zoom increments
    strategy: Literal["wheel","double_click","controls","auto"] = "auto"
    center_first: bool = True              # click once to set focus before zoom
    step_wait_ms: int = 180                # delay between increments

class FindArchiveURLAction(BaseModel):
    url: str
    date: str

class PDFAction(BaseModel):
    action: str # download, scroll_down, scroll_up, jump
    # download
    pdf_url: str | None
    save_name: str | None
    # jump
    page_number: int | None
    # scroll
    pixels: int | None
    # search
    search_text: str | None

class FindInPageAction(BaseModel):
    query: str
    match_case: bool = False
    whole_word: bool = False
    max_snippets: int = 100
    highlight: bool = False  # set True if you want <mark> injection

class VideoAction(BaseModel):
    action: str # jump
    # jump
    video_url: str | None
    time: int | None

class EnsureEarliestSVParams(BaseModel):
    max_clicks: int = 80   # how many “Previous/Earlier” clicks to try
    try_drag: bool = True  # also drag slider fully left as a fallback

class TimelineArrowParams(BaseModel):
    # 'left' ≈ earlier imagery, 'right' ≈ later imagery
    direction: Literal["left", "right", "earlier", "later", "prev", "next"] = "left"
    steps: int = 5                 # how many arrow clicks
    wait_ms: int = 250             # pause between clicks
    ensure_timeline_open: bool = True  # try to open the time widget first

async def call_openai_api(prompt: str) -> str:
    """Call OpenAI API directly and return the response content."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    
    return response.choices[0].message.content

class Controller(BrowserUseController):
    def __init__(
            self,
            exclude_actions: list[str] = [],
            output_model: type[BaseModel] | None = None,
            http_save_path: str = None,
            http_server_port: int = 8080,
    ):
        self.http_save_path = http_save_path
        self.http_server_port = http_server_port

        self.registry = Registry[Context](exclude_actions)
        self._memory_summarizer_llm: Any | None = None
        self._baseline_wait_profile: dict[str, float] | None = None
        self._current_wait_profile: str = "default"
        self._wait_overrides: dict[str, dict[str, float]] = {
            "google_maps": {
                "minimum_wait_page_load_time": 0.2,
                "wait_for_network_idle_page_load_time": 0.0,
                "maximum_wait_page_load_time": 5.0,
                "wait_between_actions": 0.2,
            }
        }


        """Register all default browser actions"""

        if output_model is not None:
            # Create a new model that extends the output model with success parameter
            class ExtendedOutputModel(BaseModel):  # type: ignore
                success: bool = True
                data: output_model  # type: ignore

            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet  completely finished (success=False), because last step is reached',
                param_model=ExtendedOutputModel,
            )
            async def done(params: ExtendedOutputModel):
                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return ActionResult(is_done=True, success=params.success, extracted_content=json.dumps(output_dict))
        else:

            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet  completely finished (success=False), because last step is reached',
                param_model=DoneAction,
            )
            async def done(params: DoneAction):
                return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

        #=== Custom code - Kim ===#

        @self.registry.action(
        '📷 Take a screenshot of the current page.',
        )
        async def take_screenshot(browser: BrowserContext):
            page = await browser.get_current_page()

            # Generate a safe filename from the URL
            short_url = re.sub(r'^https?://(?:www\.)?|/$', '', page.url)
            slug = re.sub(r'[^a-zA-Z0-9]+', '-', short_url).strip('-').lower()

            # Optional: cap slug length or hash to avoid OS filename limits (uncomment to use).
            if len(slug) > 64:
                slug = slug[:64]
            if not slug: 
                slug = "screenshot"
            # Alternatively, generate a short hash instead of a long slug:
            # import hashlib
            # slug = hashlib.sha256(short_url.encode('utf-8')).hexdigest()[:16]
            sanitized_filename = f'{slug}.jpg'

            # Save screenshot
            await page.screenshot(path=sanitized_filename, full_page=False, timeout = 120_000)

            saved_screenshot_message = f'Took a screenshot and saved {page.url} to ./{sanitized_filename}'
            logger.info(saved_screenshot_message)
            return ActionResult(
                extracted_content=saved_screenshot_message,
                success=True,
                include_in_memory=True,
            )

        """# Code to combat task id #3
        @self.registry.action("Find multiple keyword and obtain its frequency in current page", param_model=FindInPageAction)
        async def find_in_page(params: FindInPageAction, browser: BrowserContext):
            page = await browser.get_current_page()

            text = await page.inner_text("body")   # get visible text only
            query = params.query
            flags = 0 if params.match_case else re.IGNORECASE
            pattern = r"\b{}\b".format(re.escape(query)) if params.whole_word else re.escape(query)

            matches = list(re.finditer(pattern, text, flags))
            count = len(matches)

            snippets = []
            for m in matches[: params.max_snippets]:
                i = m.start()
                start = max(0, i - 40)
                end = min(len(text), i + len(query) + 40)
                snippets.append(text[start:i] + "[" + m.group() + "]" + text[i+len(query):end])

            msg = f"🔎 Found {count} match(es) for '{query}'"
            logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                data={"count": count, "snippets": snippets, "query": query},
                include_in_memory=True
            )
            
            async def _focus_point():
                try:
                    await page.mouse.move(cx, cy)
                    await page.mouse.click(cx, cy)   # sets focus point in Street View
                    await page.wait_for_timeout(120)
                except Exception:
                    pass

            async def _zoom_wheel():
                count = 0
                for _ in range(max(1, params.steps)):
                    await page.mouse.move(cx, cy)
                    await page.mouse.wheel(0, -400)  # negative = zoom in
                    count += 1
                    await page.wait_for_timeout(params.step_wait_ms)
                return count

            async def _zoom_double_click():
                count = 0
                for _ in range(max(1, params.steps)):
                    await page.mouse.dblclick(cx, cy, delay=30)
                    count += 1
                    await page.wait_for_timeout(params.step_wait_ms)
                return count

            async def _zoom_controls():
                count = 0
                # controls usually live on the right; ensure they’re visible
                try:
                    vw = await page.evaluate("() => window.innerWidth")
                    vh = await page.evaluate("() => window.innerHeight")
                    await page.mouse.move(vw-40, vh/2)
                    await page.wait_for_timeout(120)
                except Exception:
                    pass
                for _ in range(max(1, params.steps)):
                    clicked = False
                    for sel in [
                        "button[aria-label*='Zoom in']",
                        "[role='button'][aria-label*='Zoom in']",
                        "div[aria-label*='Zoom in']",
                    ]:
                        try:
                            btn = page.locator(sel).first
                            if await btn.count() > 0 and await btn.is_visible():
                                await btn.click()
                                clicked = True
                                break
                        except Exception:
                            pass
                    if not clicked:
                        break
                    count += 1
                    await page.wait_for_timeout(params.step_wait_ms)
                return count

            if params.center_first:
                await _focus_point()

            strategy = params.strategy
            performed = 0
            logs = []

            async def run_strategy(name):
                nonlocal performed
                if name == "wheel":
                    c = await _zoom_wheel(); logs.append(("wheel", c)); performed += c
                elif name == "double_click":
                    c = await _zoom_double_click(); logs.append(("double_click", c)); performed += c
                elif name == "controls":
                    c = await _zoom_controls(); logs.append(("controls", c)); performed += c

            if strategy == "auto":
                # Try wheel → double-click → controls
                await run_strategy("wheel")
                if performed == 0:
                    await run_strategy("double_click")
                if performed == 0:
                    await run_strategy("controls")
            else:
                await run_strategy(strategy)
            
            msg = f"🔎 Zoomed in at ({cx},{cy}) using {', '.join(f'{n}×{c}' for n,c in logs if c)}."
            logger.info(msg)
            return ActionResult(
                extracted_content=f"🔎 Zoomed in at ({cx},{cy}) using {', '.join(f'{n}×{c}' for n,c in logs if c)}.",
                data={"x": cx, "y": cy, "log": logs},
                include_in_memory=True,
            )"""

        @self.registry.action(
            "Google Maps: Click the Street View timeline arrow (left=earlier, right=later) N times.",
            param_model=TimelineArrowParams,
            domains=["google.com", "maps.google.com"],
        )
        async def sv_click_timeline_arrow(params: TimelineArrowParams, browser: BrowserContext):
            page = await browser.get_current_page()

            async def _open_timeline():
                # make sure the timeline/time widget is open (if available)
                for selector in [
                    "button[aria-label*='See more dates']",
                    "button[aria-label*='Time']",
                    "button[aria-label*='Timeline']",
                    "div[aria-label*='See more dates']",
                ]:
                    try:
                        page_locator = page.locator(selector).first
                        if await page_locator.count() > 0 and await page_locator.is_visible():
                            await page_locator.click()
                            await page.wait_for_timeout(250)
                            return
                    except Exception:
                        pass

            async def _reveal_arrows():
                # arrows sometimes appear on hover near the bottom timeline strip
                try:
                    viewport_width = await page.evaluate("() => window.innerWidth")
                    viewport_height = await page.evaluate("() => window.innerHeight")
                    await page.mouse.move(int(viewport_width/2), max(0, viewport_height - 40))
                    await page.wait_for_timeout(150)
                except Exception:
                    pass

            def _want_right():
                direction = params.direction.lower()
                return direction in ("right", "later", "next")

            def _selectors_for_direction():
                # try most specific first, then fallbacks
                if _want_right():
                    return [
                        "g-scrolling-carousel button[aria-label*='Next']",
                        "button[aria-label*='Next']",
                        "button[aria-label*='Later']",
                        "[role='button'][aria-label*='Next']",
                        "[role='button'][aria-label*='Later']",
                    ]
                else:
                    return [
                        "g-scrolling-carousel button[aria-label*='Previous']",
                        "button[aria-label*='Previous']",
                        "button[aria-label*='Earlier']",
                        "[role='button'][aria-label*='Previous']",
                        "[role='button'][aria-label*='Earlier']",
                    ]

            async def _read_date_label() -> str:
                try:
                    date_label = await page.locator("[aria-live='polite']").first.text_content(timeout=600)
                    return (date_label or "").strip()
                except Exception:
                    return ""

            # (1) optionally open the timeline
            if params.ensure_timeline_open:
                await _open_timeline()

            # (2) reveal the arrows if they’re hover-gated
            await _reveal_arrows()

            # (3) locate the arrow button we need
            arrow = None
            for selector in _selectors_for_direction():
                try:
                    page_locator = page.locator(selector).last if _want_right() else page.locator(selector).first
                    if await page_locator.count() > 0:
                        arrow = page_locator
                        break
                except Exception:
                    continue

            num_clicks = 0
            before = await _read_date_label()

            # (4) click the arrow repeatedly
            if arrow:
                for _ in range(max(1, params.steps)):
                    try:
                        await arrow.click()
                        num_clicks += 1
                        await page.wait_for_timeout(params.wait_ms)
                    except Exception:
                        # try re-revealing and continue
                        await _reveal_arrows()
                        try:
                            await arrow.click()
                            num_clicks += 1
                            await page.wait_for_timeout(params.wait_ms)
                        except Exception:
                            break
            else:
                # (5) fallback: focus slider and use keyboard arrows
                try:
                    page_slider = page.locator("[role='slider']").first
                    await page_slider.wait_for(state="visible", timeout=1200)
                    await page_slider.focus()
                    key = "ArrowRight" if _want_right() else "ArrowLeft"
                    for _ in range(max(1, params.steps)):
                        await page.keyboard.press(key)
                        num_clicks += 1
                        await page.wait_for_timeout(params.wait_ms)
                except Exception:
                    pass

            after = await _read_date_label()

            return ActionResult(
                extracted_content=(
                    f"➡️ Timeline arrow clicked {num_clicks}x towards "
                    f"{'RIGHT/later' if _want_right() else 'LEFT/earlier'}."
                    f" Date label: '{before}' → '{after}'"
                ),
                data={"clicks": num_clicks, "direction": "right" if _want_right() else "left", "before": before, "after": after},
                include_in_memory=True,
            )

        # --- Street View helpers ---------------------------------------------
        async def _sv_open_timeline(self, page):
            for selector in [
                "button[aria-label*='See more dates']",
                "button[aria-label*='Time']",
                "button[aria-label*='Timeline']",
                "div[aria-label*='See more dates']",
            ]:
                try:
                    page_locator = page.locator(selector).first
                    if await page_locator.count() > 0:
                        await page_locator.click()
                        await page.wait_for_timeout(250)
                        return
                except Exception:
                    pass

        async def _sv_get_slider_vals(self, page):
            try:
                page_slider = page.locator("[role='slider']").first
                await page_slider.wait_for(state="visible", timeout=1200)
                page_slider_min_value = await page_slider.get_attribute("aria-valuemin")
                page_slider_now_value = await page_slider.get_attribute("aria-valuenow")
                page_slider_max_value = await page_slider.get_attribute("aria-valuemax")
                to_int = lambda x: int(x) if x is not None else None
                return page_slider, to_int(page_slider_min_value), to_int(page_slider_now_value), to_int(page_slider_max_value)
            except Exception:
                return None, None, None, None

        async def _sv_read_date_label(self, page) -> str:
            try:
                date = await page.locator("[aria-live='polite']").first.text_content(timeout=600)
                return (date or "").strip()
            except Exception:
                return ""

        async def _sv_click_previous_until_stable(self, page, max_clicks: int = 80):
            previous_label = await self._sv_read_date_label(page)
            num_clicks = 0
            for _ in range(max_clicks):
                clicked = False
                for selector in ["button[aria-label*='Previous']", "button[aria-label*='Earlier']"]:
                    try:
                        button_locator = page.locator(selector).last
                        if await button_locator.count() > 0 and not await button_locator.is_disabled():
                            await button_locator.click()
                            clicked = True
                            num_clicks += 1
                            await page.wait_for_timeout(200)
                            break
                    except Exception:
                        pass
                current_label = await self._sv_read_date_label(page)
                if not clicked or current_label == previous_label:
                    break
                previous_label = current_label
            return num_clicks, previous_label

        async def _sv_drag_fully_left(self, page):
            try:
                slider = page.locator("[role='slider']").first
                bounding_box = await slider.bounding_box()
                if not bounding_box:
                    return False
                y = bounding_box["y"] + bounding_box["height"] / 2
                await page.mouse.move(bounding_box["x"] + bounding_box["width"] * 0.8, y)
                await page.mouse.down()
                await page.mouse.move(bounding_box["x"] - bounding_box["width"] * 2, y, steps=20)
                await page.mouse.up()
                await page.wait_for_timeout(250)
                return True
            except Exception:
                return False

        async def _sv_verify_earliest_available_imagery(self, page) -> dict:
            """
            Returns: {"earliest": bool, "label": str, "vmin": int|None, "vnow": int|None, "clicks": int}
            """
            await self._sv_open_timeline(page)

            # Try 'Home' (often jumps to earliest)
            try:
                slider = page.locator("[role='slider']").first
                await slider.wait_for(state="visible", timeout=1500)
                await slider.focus()
                await page.keyboard.press("Home")
                await page.wait_for_timeout(250)
            except Exception:
                pass

            num_clicks, label_after_clicks = await self._sv_click_previous_until_stable(page)

            slider, value_min, value_now, value_max = await self._sv_get_slider_vals(page)

            # If still not at earliest, drag the handle hard left
            if slider is not None and value_min is not None and value_now is not None and value_now > value_min:
                if await self._sv_drag_fully_left(page):
                    _, value_min, value_now, _ = await self._sv_get_slider_vals(page)

            label_final = await self._sv_read_date_label(page)
            earliest = (value_min is not None and value_now is not None and value_now == value_min)

            return {
                "earliest": earliest,
                "label": label_final or label_after_clicks,
                "vmin": value_min,
                "vnow": value_now,
                "clicks": num_clicks,
            }


        #=== End of Custom Code ===#

        async def _has_google_captcha(page) -> bool:
            page_html = (await page.content()).lower()
            if "detected unusual traffic" in page_html or "recaptcha" in page_html:
                return True
            # 更稳：检查 iframe
            for frame in page.frames:
                frame_url = (frame.url or "").lower()
                if "recaptcha" in frame_url or "bframe" in frame_url:
                    return True
            return False
        
        @self.registry.action('Search the query in DuckDuckGo', param_model=SearchGoogleAction)
        async def search_duckduckgo(params: SearchGoogleAction, browser: BrowserContext):
            page = await browser.get_current_page()
            import urllib.parse
            q = urllib.parse.quote(params.query)
            await page.goto(f'https://duckduckgo.com/?q={q}')
            await page.wait_for_load_state()
            duckduckgo_searched_for_message = f'🔍  Searched for "{params.query}" in DuckDuckGo'
            logger.info(duckduckgo_searched_for_message)
            return ActionResult(extracted_content=duckduckgo_searched_for_message, include_in_memory=True)
        
        # Basic Navigation Actions
        @self.registry.action(
            'Search the query in Google in the current tab, the query should be a search query like humans search in Google, concrete and not vague or super long. More the single most important items. ',
            param_model=SearchGoogleAction,
        )
        async def search_google(params: SearchGoogleAction, browser: BrowserContext):
            page = await browser.get_current_page()

            # 1) 先去 NCR，避免地区重定向
            await page.goto('https://www.google.com/ncr?hl=en', wait_until='domcontentloaded')
            await page.wait_for_load_state('networkidle')

            # 2) 若出现同意/隐私提示，自动接受
            try:
                if "consent" in page.url or await page.locator('form[action*="consent"]').count() > 0:
                    # 常见按钮文案覆盖
                    for button_text in ["I agree", "Accept all", "Agree to all", "同意", "接受全部"]:
                        page_locator = page.get_by_role("button", name=button_text, exact=False)
                        if await page_locator.count() > 0:
                            await page_locator.first.click()
                            await page.wait_for_load_state('networkidle')
                            break
            except Exception:
                pass

            # 3) 真正搜索（避免 udm=14 引发不稳定，强制英文）
            from urllib.parse import quote
            encoded = quote(params.query)
            await page.goto(f'https://www.google.com/search?q={encoded}&hl=en&num=10', wait_until='domcontentloaded')
            await page.wait_for_load_state('networkidle')

            # 4) 等搜索结果区域出现（多种选择器兜底）
            for selector in ['div#search', 'div[role="main"]', '#rso']:
                try:
                    await page.wait_for_selector(selector, timeout=8000)
                    break
                except Exception:
                    continue

            # 5) 稳定一下（给截图留点缓冲）
            await page.wait_for_timeout(500)

            if await _has_google_captcha(page):
                captcha_detected_message = "🛑 Google CAPTCHA detected — falling back to DuckDuckGo."
                logger.info(captcha_detected_message)
                # 直接改用 DuckDuckGo
                await search_duckduckgo(params, browser)
                return ActionResult(extracted_content=captcha_detected_message, include_in_memory=True)
            
            google_searched_for_message = f'🔍  Searched for "{params.query}" in Google'
            logger.info(google_searched_for_message)
            return ActionResult(extracted_content=google_searched_for_message, include_in_memory=True)

        @self.registry.action('Open ORCID profile directly if query contains an ORCID iD', param_model=SearchGoogleAction)
        async def open_orcid_if_present(params: SearchGoogleAction, browser: BrowserContext):
            page = await browser.get_current_page()
            import re
            orcid_match = re.search(r"\b(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])\b", params.query)
            if orcid_match:
                orcid = orcid_match.group(1)
                orcid_url = f"https://orcid.org/{orcid}"
                await page.goto(orcid_url, wait_until='domcontentloaded')
                await page.wait_for_load_state('networkidle')
                orcid_opened_message = f"🔗  Opened ORCID profile: {orcid_url}"
                logger.info(orcid_opened_message)
                return ActionResult(extracted_content=orcid_opened_message, include_in_memory=True)
            return ActionResult(extracted_content="No ORCID id found in query.", include_in_memory=False)

        @self.registry.action('Navigate to URL in the current tab', param_model=GoToUrlAction)
        async def go_to_url(params: GoToUrlAction, browser: BrowserContext):
            page = await browser.get_current_page()
            await page.goto(params.url)
            await page.wait_for_load_state()
            url_navigation_message = f'🔗  Navigated to {params.url}'
            logger.info(url_navigation_message)
            return ActionResult(extracted_content=url_navigation_message, include_in_memory=True)

        @self.registry.action('Wayback Machine for finding the archive URL of a given URL, with a specified date', param_model=FindArchiveURLAction)
        async def find_archive_url(params: FindArchiveURLAction, browser: BrowserContext):
            no_timestamp_url = f"https://archive.org/wayback/available?url={params.url}"
            archive_url = no_timestamp_url + f"&timestamp={params.date}"

            response = requests.get(archive_url).json()
            response_notimestamp = requests.get(no_timestamp_url).json()

            if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
                closest = response["archived_snapshots"]["closest"]
                logger.info(f"Archive found! {closest}")

            elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp[
                "archived_snapshots"]:
                closest = response_notimestamp["archived_snapshots"]["closest"]
                logger.info(f"Archive found! {closest}")
            else:
                return ActionResult(
                    extracted_content = "❌  No archive URL found for the given URL and date.",
                    include_in_memory = True,
                )

            target_url = closest["url"]
            return ActionResult(
                extracted_content = f"🕰️  Found archive URL: {target_url}",
                include_in_memory = True,
            )

        @self.registry.action('Go back', param_model=NoParamsAction)
        async def go_back(_: NoParamsAction, browser: BrowserContext):
            def _normalize(url: str | None) -> tuple[str, str]:
                if not url:
                    return "", ""
                try:
                    parts = urlsplit(url)
                    host = parts.netloc.lower()
                    path = parts.path.rstrip("/") or "/"
                    return host, path
                except Exception:
                    return url, ""

            page_before = await browser.get_agent_current_page()
            previous_url = page_before.url
            prev_norm = _normalize(previous_url)

            await browser.go_back()
            await asyncio.sleep(browser.config.minimum_wait_page_load_time)

            page_after = await browser.get_agent_current_page()
            current_url = page_after.url
            current_norm = _normalize(current_url)

            if current_norm == prev_norm:
                # Attempt a JS history.back fallback first
                try:
                    await page_after.evaluate("window.history.back();")
                    await asyncio.sleep(browser.config.minimum_wait_page_load_time)
                    page_after = await browser.get_agent_current_page()
                    current_url = page_after.url
                    current_norm = _normalize(current_url)
                except Exception as history_exc:
                    logger.debug(f"JS history.back fallback failed: {history_exc}")

            if current_norm == prev_norm:
                # Try navigating directly to document.referrer if available
                try:
                    referrer = await page_after.evaluate("document.referrer || ''")
                except Exception as ref_exc:
                    logger.debug(f"Unable to read document.referrer: {ref_exc}")
                    referrer = ""

                if referrer and referrer != current_url:
                    try:
                        await page_after.goto(referrer, wait_until="domcontentloaded")
                        await asyncio.sleep(browser.config.minimum_wait_page_load_time)
                        page_after = await browser.get_agent_current_page()
                        current_url = page_after.url
                        current_norm = _normalize(current_url)
                    except Exception as ref_nav_exc:
                        logger.debug(f"Navigation to document.referrer failed: {ref_nav_exc}")
            if current_norm == prev_norm:
                # If no history is available (e.g., link opened a new tab), try returning to a previous tab
                session = await browser.get_session()
                context = getattr(session, "context", None)
                open_pages = list(getattr(context, "pages", []) or [])

                if len(open_pages) > 1:
                    try:
                        await page_after.close()
                        await asyncio.sleep(0.2)
                        await browser.switch_to_tab(-1)
                        await asyncio.sleep(browser.config.minimum_wait_page_load_time)
                        page_after = await browser.get_agent_current_page()
                        current_url = page_after.url
                        current_norm = _normalize(current_url)
                        msg = (
                            "Go back had no history; closed the current tab and returned to the previous tab "
                            f"({current_url})."
                        )
                        logger.info(msg)
                        return ActionResult(extracted_content=msg, include_in_memory=True)
                    except Exception as tab_exc:
                        logger.warning(
                            "Fallback to close tab after failed go back encountered an error: %s", tab_exc
                        )

                msg = f"Go back requested but browser remained on {current_url}."
                logger.warning(msg)
                return ActionResult(
                    extracted_content=msg,
                    success=False,
                    include_in_memory=True,
                )
            else:
                msg = f"Navigated back from {previous_url} to {current_url}."
                logger.info(msg)
                return ActionResult(
                    extracted_content=msg,
                    success=True,
                    include_in_memory=True,
                )

        # wait for x seconds
        @self.registry.action('Wait for x seconds default 3')
        async def wait(seconds: int = 3):
            waiting_for_message = f'🕒  Waiting for {seconds} seconds'
            logger.info(waiting_for_message)
            await asyncio.sleep(seconds)
            return ActionResult(extracted_content=waiting_for_message, include_in_memory=True)


        @self.registry.action('Click element by index', param_model=ClickElementAction)
        async def click_element_by_index(params: ClickElementAction, browser: BrowserContext):
            """重写版本，结构略有调整。"""
            session = await browser.get_session()
            selector_map = await browser.get_selector_map()
            
            # 验证元素索引是否存在
            if params.index not in selector_map:
                raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')
            
            element_node = await browser.get_dom_element_by_index(params.index)
            pages_before_click = len(session.context.pages)
            
            # 在尝试点击之前检查元素是否为文件上传器
            if await browser.is_file_uploader(element_node):
                uploader_message = f'Index {params.index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files'
                logger.info(uploader_message)
                return ActionResult(extracted_content=uploader_message, include_in_memory=True)
            
            try:
                # 执行点击操作
                download_path = await browser._click_element_node(element_node)
                
                # 根据结果构建消息
                if download_path:
                    result_message = f'💾  Downloaded file to {download_path}'
                else:
                    element_text = element_node.get_all_text_till_next_clickable_element(max_depth=2)
                    result_message = f'🖱️  Clicked button with index {params.index}: {element_text}'
                
                logger.info(result_message)
                logger.debug(f'Element xpath: {element_node.xpath}')
                
                # 检查是否打开了新标签页并处理
                pages_after_click = len(session.context.pages)
                if pages_after_click > pages_before_click:
                    tab_notification = 'New tab opened - switching to it'
                    result_message += f' - {tab_notification}'
                    logger.info(tab_notification)
                    await browser.switch_to_tab(-1)
                
                return ActionResult(extracted_content=result_message, include_in_memory=True)
            except Exception as e:
                logger.warning(f'Element not clickable with index {params.index} - most likely the page changed')
                return ActionResult(error=str(e))


        @self.registry.action(
            'Input text into a input interactive element',
            param_model=InputTextAction,
        )
        async def input_text(params: InputTextAction, browser: BrowserContext, has_sensitive_data: bool = False):
            if params.index not in await browser.get_selector_map():
                raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

            element_node = await browser.get_dom_element_by_index(params.index)
            await browser._input_text_element_node(element_node, params.text)
            if not has_sensitive_data:
                input_content_message = f'⌨️  Input {params.text} into index {params.index}'
            else:
                input_content_message = f'⌨️  Input sensitive data into index {params.index}'
            logger.info(input_content_message)
            logger.debug(f'Element xpath: {element_node.xpath}')
            return ActionResult(extracted_content=input_content_message, include_in_memory=True)

        # Save PDF
        @self.registry.action(
            'Save the current page as a PDF file. This action is equivalent to taking a screenshot.',
        )
        async def save_pdf(browser: BrowserContext):
            page = await browser.get_current_page()
            short_url = re.sub(r'^https?://(?:www\.)?|/$', '', page.url)
            slug = re.sub(r'[^a-zA-Z0-9]+', '-', short_url).strip('-').lower()

            # Optional: cap slug length or hash to avoid OS filename limits (uncomment to use).
            if len(slug) > 64:
                slug = slug[:64]
            # Alternatively, generate a short hash instead of a long slug:
            # import hashlib
            # slug = hashlib.sha256(short_url.encode('utf-8')).hexdigest()[:16]
            sanitized_filename = f'{slug}.pdf'

            await page.emulate_media(media='screen')
            await page.pdf(path=sanitized_filename, format='A4', print_background=False)
            saving_pdf_message = f'Saving page with URL {page.url} as PDF to ./{sanitized_filename}'
            logger.info(saving_pdf_message)
            return ActionResult(extracted_content=saving_pdf_message, include_in_memory=True)
        @self.registry.action(
            """Interact with Youtube videos in the browser.
* `action`: The action must be one of the following: jump. The `action` is required.
- jump: Jump to a specific time in the video. `video_url` and `time` are required. `video_url` is the URL of the video and `time` is the time in seconds to jump to.
""",
            param_model=VideoAction,
        )
        async def video_viewer(params: VideoAction, browser: BrowserContext):
            """Custom polling and extracted seek logic."""
            # Helper functions (nested for closure access)
            def _parse_youtube_video_id(url: str) -> str | None:
                """Extract YouTube video ID from various URL formats."""
                import re
                # Try standard URL patterns first (watch?v=, /shorts/, youtu.be/)
                url_pattern = r"(?:v=|/shorts/|youtu\.be/)([0-9A-Za-z_-]{11})"
                url_match = re.search(url_pattern, url)
                if url_match:
                    return url_match.group(1)
                # Try direct video ID pattern
                direct_pattern = r"^([0-9A-Za-z_-]{11})$"
                direct_match = re.fullmatch(direct_pattern, url)
                if direct_match:
                    return direct_match.group(1)
                return None

            async def _wait_for_player_ready_polling(page: Page, timeout_ms: int = 30000) -> bool:
                """Wait for player to be ready using polling with exponential backoff."""
                import time
                start_time = time.time()
                check_interval = 100  # Start with 100ms
                max_interval = 1000   # Cap at 1 second
                
                while (time.time() - start_time) * 1000 < timeout_ms:
                    try:
                        is_ready = await page.evaluate("() => window.playerReady === true")
                        if is_ready:
                            return True
                    except Exception:
                        pass
                    await asyncio.sleep(check_interval / 1000)
                    check_interval = min(check_interval * 1.5, max_interval)
                return False

            async def _seek_to_time_and_verify(page: Page, target_time: int) -> float:
                """Seek to target time, play briefly to unlock, pause, and return actual time."""
                # Mute for autoplay compatibility
                await page.evaluate("() => { try { if (window.player) window.player.mute(); } catch(e){} }")
                
                # Seek to target
                await page.evaluate(f"() => {{ if (window.ytCtl) window.ytCtl.seek({target_time}); }}")
                
                # Brief play to unlock video frame
                await page.evaluate("() => { if (window.ytCtl) window.ytCtl.play(); }")
                await asyncio.sleep(0.4)
                
                # Pause and get current time
                await page.evaluate("() => { if (window.ytCtl) window.ytCtl.pause(); }")
                current_time = await page.evaluate("() => window.ytCtl ? window.ytCtl.now() : 0")
                
                return float(current_time)
            
            page = await browser.get_current_page()
            
            if params.action != "jump":
                return ActionResult(
                    error=f"❌  Unsupported action: {params.action}",
                    include_in_memory=True,
                )
            
            # Validate parameters inline (too simple to extract)
            if not hasattr(params, 'video_url') or params.video_url is None:
                return ActionResult(
                    error="❌  Video URL is required for jump action.",
                    include_in_memory=True,
                )
            
            if not hasattr(params, 'time') or params.time is None:
                return ActionResult(
                    error="❌  Time is required for jump action.",
                    include_in_memory=True,
                )
            
            # Extract video ID using helper
            video_id = _parse_youtube_video_id(params.video_url)
            if not video_id:
                return ActionResult(
                    error="❌  Could not parse YouTube video id from URL.",
                    include_in_memory=True,
                )
            
            # Navigate to local player
            player_url = f"http://localhost:{self.http_server_port}/video_viewer/player.html?v={video_id}"
            await page.goto(player_url, wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle")
            
            # Wait for player readiness using custom polling
            player_ready = await _wait_for_player_ready_polling(page, timeout_ms=30000)
            if not player_ready:
                return ActionResult(
                    error="❌  Player failed to initialize within timeout period.",
                    include_in_memory=True,
                )
            
            # Seek to target time and verify
            target_time = int(params.time)
            actual_time = await _seek_to_time_and_verify(page, target_time)
            
            result_message = f"🎥  Jumped to {target_time}s via local player. Current time: {actual_time:.2f}s"
            logger.info(result_message)
            return ActionResult(extracted_content=result_message, include_in_memory=True)
        
        @self.registry.action(
            """Download pdf from a given URL and interact with PDF in the browser. PDF must be downloaded and opened in browser first before it can scroll or jump to a page.
* `action`: download | scroll_down | scroll_up | jump | search
 - download: `pdf_url` + `save_name` required
 - jump: `page_number` required
 - search: `search_text` required
""",
            param_model=PDFAction,
        )
        async def pdf_viewer(params: PDFAction, browser: BrowserContext):
            page = await browser.get_current_page()

            # -------- helpers --------
            def _download_pdf(url: str, save_path: str, timeout: int = 60):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 若 URL 含空格，先做最小合法化
                if " " in url:
                    # 只对 path/query 进行编码，scheme+host 不动
                    parts = urlsplit(url)
                    url = urlunsplit((
                        parts.scheme,
                        parts.netloc,
                        quote(parts.path),
                        quote(parts.query, safe="=&?"),
                        parts.fragment
                    ))
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(save_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                return save_path

            async def _download_pdf_async(url: str, save_path: str):
                # ✨ 避免阻塞事件循环
                return await asyncio.to_thread(_download_pdf, url, save_path)
            
            def _extract_pdf_from_viewer_url(url: str) -> str | None:
                """从 pdf.js 的 viewer.html?file=... 里提取真实 PDF URL"""
                if "viewer.html" not in url:
                    return None
                pdf_query = urlparse(url).query
                pdf_params = parse_qs(pdf_query)
                pdf_file_vals = pdf_params.get("file")
                if not pdf_file_vals:
                    return None
                # 只取第一个，必要时再 unquote
                return pdf_file_vals[0]
    
            async def _open_local_pdfviewer(save_name: str) -> bool:
                # 本地 pdf.js viewer
                local_url = f"http://localhost:{self.http_server_port}/pdf_viewer/viewer.html?file=../local/{quote(save_name)}"
                try:
                    await page.goto(local_url, wait_until="domcontentloaded", timeout=20000)
                    await page.wait_for_selector("#pageNumber", timeout=20000)
                    return True
                except Exception:
                    return False

            async def _open_online_pdfviewer(pdf_url: str) -> bool:
                # 在线 pdf.js viewer
                remote = "https://mozilla.github.io/pdf.js/web/viewer.html?file=" + quote(pdf_url, safe='')
                try:
                    await page.goto(remote, wait_until="domcontentloaded", timeout=25000)
                    await page.wait_for_selector("#pageNumber", timeout=25000)
                    return True
                except Exception:
                    return False

            def _with_page_fragment(url: str, fragment: str) -> str:
                page_parts = list(urlsplit(url))
                # 保留已有片段的基础上设置/覆盖
                page_parts[4] = fragment
                return urlunsplit(page_parts)

            async def _is_pdfjs() -> bool:
                # pdf.js 里一定有 #pageNumber / #viewerContainer
                try:
                    await page.wait_for_selector("#pageNumber", timeout=1000)
                    return True
                except Exception:
                    return False

            async def _jump_pdfjs(page_no: int):
                await page.fill("#pageNumber", str(page_no))
                await page.keyboard.press("Enter")
                await asyncio.sleep(0.4)

            async def _jump_native(page_no: int):
                base = page.url.split("#")[0]
                await page.goto(_with_page_fragment(base, f"page={page_no}"), wait_until="load")

            # -------- main --------
            action = params.action

            # DOWNLOAD
            if action == "download":
                if not getattr(params, 'pdf_url', None):
                    return ActionResult(error="❌  PDF URL is required for download action.", include_in_memory=True)

                pdf_url = params.pdf_url
                save_name = params.save_name or "downloaded_file.pdf"
                save_path = os.path.join(self.http_save_path, save_name)

                try:
                    _download_pdf(pdf_url, save_path)
                except Exception as e:
                    return ActionResult(error=f"❌  PDF download failed: {e}", include_in_memory=True)

                # 优先尝试本地 pdf.js；不行则在线 pdf.js；再不行直接打开原 PDF
                opened_pdf = await _open_local_pdfviewer(save_name)
                mode = "local-pdfjs" if opened_pdf else None
                if not opened_pdf:
                    opened_pdf= await _open_online_pdfviewer(pdf_url)
                    mode = "online-pdfjs" if opened_pdf else None
                if not opened_pdf:
                    await page.goto(pdf_url, wait_until="load")
                    mode = "native"

                downloaded_pdf_message = f"📄  Downloaded PDF → {save_path}. Open mode: {mode}"
                logger.info(downloaded_pdf_message)
                return ActionResult(extracted_content=downloaded_pdf_message, include_in_memory=True)

            # SCROLL DOWN
            if action == "scroll_down":
                pixels = getattr(params, 'pixels', None)
                if await _is_pdfjs():
                    if pixels:
                        await page.evaluate(f"document.querySelector('#viewerContainer').scrollTop += {int(pixels)}")
                    else:
                        await page.click("#next")
                else:
                    # 原生预览器：用 PageDown 兜底
                    await page.keyboard.press("PageDown")
                await page.wait_for_load_state()
                return ActionResult(extracted_content="📄  Scrolled down.", include_in_memory=True)

            # SCROLL UP
            if action == "scroll_up":
                pixels = getattr(params, 'pixels', None)
                if await _is_pdfjs():
                    if pixels:
                        await page.evaluate(f"document.querySelector('#viewerContainer').scrollTop -= {int(pixels)}")
                    else:
                        await page.click("#previous")
                else:
                    await page.keyboard.press("PageUp")
                await page.wait_for_load_state()
                return ActionResult(extracted_content="📄  Scrolled up.", include_in_memory=True)

            # JUMP
            if action == "jump":
                if getattr(params, 'page_number', None) is None:
                    return ActionResult(error="❌  Page number is required for jump action.", include_in_memory=True)

                current_page_number = int(params.page_number)
                if await _is_pdfjs():
                    await _jump_pdfjs(current_page_number)
                    await page.wait_for_load_state()
                    pdf_jumped_message = f"📄  Jumped to page {current_page_number} (pdf.js)."
                else:
                    await _jump_native(current_page_number)
                    pdf_jumped_message = f"📄  Jumped to page {current_page_number} (native)."
                logger.info(pdf_jumped_message)
                return ActionResult(extracted_content=pdf_jumped_message, include_in_memory=True)

            # SEARCH
            if action == "search":
                if not getattr(params, 'search_text', None):
                    return ActionResult(error="❌  Search text is required for search action.", include_in_memory=True)

                query = params.search_text

                # 若非 pdf.js，自动切到在线 pdf.js 再搜
                if not await _is_pdfjs():
                    current_pdf = page.url.split("#")[0]
                    real_pdf = _extract_pdf_from_viewer_url(current_pdf)
                    if real_pdf:
                        switched = await _open_online_pdfviewer(real_pdf)
                    else:
                        #  不在 viewer.html，直接尝试把“当前 URL 当作 PDF”送入在线 pdf.js
                        switched = await _open_online_pdfviewer(current_pdf)
                    if not switched:
                        # 兜底：用浏览器原生查找（没有计数）
                        try:
                            await page.keyboard.press("Control+f")
                            # 常见的 find 框定位可能各浏览器不同，这里仅触发快捷键 + 回车
                            await page.keyboard.type(query)
                            await page.keyboard.press("Enter")
                            native_find_message = f"🔎 Used browser native find for '{query}' (not PDF.js)."
                            logger.info(native_find_message)
                            return ActionResult(extracted_content=native_find_message, include_in_memory=True)
                        except Exception as e:
                            return ActionResult(error=f"❌  Search requires pdf.js viewer and native find failed: {e}", include_in_memory=True)

                # pdf.js 搜索
                await page.click("#viewFind")
                await page.fill("#findInput", query)
                await page.keyboard.press("Enter")
                await page.wait_for_timeout(300)
                try:
                    count_text = await page.inner_text("#findResultsCount")
                except Exception:
                    count_text = ""
                m = re.search(r"of (\d+)", count_text)
                if not m:
                    message = f"No matches found for '{query}'."
                else:
                    total = m.group(1)
                    await page.click("#findNext")
                    message = f"📄  Found {total} matches for '{query}' and moved to first."
                logger.info(message)
                return ActionResult(extracted_content=message, include_in_memory=True)

            return ActionResult(error=f"Unknown pdf action: {action}", include_in_memory=True)

        # Tab Management Actions
        @self.registry.action('Switch tab', param_model=SwitchTabAction)
        async def switch_tab(params: SwitchTabAction, browser: BrowserContext):
            await browser.switch_to_tab(params.page_id)
            # Wait for tab to be ready
            page = await browser.get_current_page()
            await page.wait_for_load_state()
            switched_tab_message = f'🔄  Switched to tab {params.page_id}'
            logger.info(switched_tab_message)
            return ActionResult(extracted_content=switched_tab_message, include_in_memory=True)

        @self.registry.action('Open url in new tab', param_model=OpenTabAction)
        async def open_tab(params: OpenTabAction, browser: BrowserContext):
            await browser.create_new_tab(params.url)
            opened_tab_message = f'🔗  Opened new tab with {params.url}'
            logger.info(opened_tab_message)
            return ActionResult(extracted_content=opened_tab_message, include_in_memory=True)

        @self.registry.action('Close an existing tab', param_model=CloseTabAction)
        async def close_tab(params: CloseTabAction, browser: BrowserContext):
            await browser.switch_to_tab(params.page_id)
            page = await browser.get_current_page()
            current_page_url = page.url
            await page.close()
            closed_tab_message = f'❌  Closed tab #{params.page_id} with url {current_page_url}'
            logger.info(closed_tab_message)
            return ActionResult(extracted_content=closed_tab_message, include_in_memory=True)


        @self.registry.action(
            'Extract page content to retrieve specific information from the page, e.g. all company names, a specific description, all information about, links with companies in structured format or simply links',
        )
        async def extract_content(
                goal: str, should_strip_link_urls: bool, browser: BrowserContext
        ):
            page = await browser.get_current_page()
            import markdownify

            # 确定需要剥离的标签
            tags_to_strip = ['a', 'img'] if should_strip_link_urls else []

            # 将主页面内容转换为 markdown
            main_page_content = markdownify.markdownify(await page.content(), strip=tags_to_strip)

            # 手动追加 iframe 文本内容，使其可被 LLM 读取（包括跨域 iframe）
            for iframe in page.frames:
                if iframe.url != page.url and not iframe.url.startswith('data:'):
                    try:
                        iframe_html = await iframe.content()
                        iframe_markdown = markdownify.markdownify(iframe_html)
                        main_page_content += f'\n\nIFRAME {iframe.url}:\n{iframe_markdown}'
                    except Exception as e:
                        logger.debug(f'无法提取 iframe 内容 {iframe.url}: {e}')

            # 使用 f-string 构建提示词
            extraction_prompt = (
                f'Your task is to extract the content of the page. '
                f'You will be given a page and a goal and you should extract all relevant information around this goal from the page. '
                f'If the goal is vague, summarize the page. Respond in json format. '
                f'Extraction goal: {goal}, Page: {main_page_content}'
            )

            try:
                # 使用 OpenAI API 直接调用
                llm_response_content = await call_openai_api(extraction_prompt)
                result_message = f'📄  Extracted from page\n: {llm_response_content}\n'
                logger.info(result_message)
                return ActionResult(extracted_content=result_message, include_in_memory=True)
            except Exception as e:
                logger.debug(f'Error extracting content: {e}')
                # 如果 LLM 提取失败，返回原始内容
                fallback_message = f'📄  Extracted from page\n: {main_page_content}\n'
                logger.info(fallback_message)
                return ActionResult(extracted_content=fallback_message)

        @self.registry.action(
            'Scroll down the page by pixel amount - if no amount is specified, scroll down one page',
            param_model=ScrollAction,
        )
        async def scroll_down(params: ScrollAction, browser: BrowserContext):
            page = await browser.get_current_page()
            if params.amount is not None:
                await page.evaluate(f'window.scrollBy(0, {params.amount});')
                scrolled_down_amount = f'{params.amount} pixels'
            else:
                await page.evaluate('window.scrollBy(0, window.innerHeight / 2);')
                scrolled_down_amount = 'half a page'

            scrolled_down_message = f'🔍  Scrolled down the page by {scrolled_down_amount}'
            logger.info(scrolled_down_message)
            return ActionResult(
                extracted_content=scrolled_down_message,
                include_in_memory=True,
            )

        # scroll up
        @self.registry.action(
            'Scroll up the page by pixel amount - if no amount is specified, scroll up one page',
            param_model=ScrollAction,
        )
        async def scroll_up(params: ScrollAction, browser: BrowserContext):
            page = await browser.get_current_page()
            if params.amount is not None:
                await page.evaluate(f'window.scrollBy(0, -{params.amount});')
                scrolled_up_amount = f'{params.amount} pixels'
            else:
                await page.evaluate('window.scrollBy(0, -window.innerHeight / 2);')
                scrolled_up_amount = 'half a page'

            scrolled_up_message = f'🔍  Scrolled up the page by {scrolled_up_amount}'
            logger.info(scrolled_up_message)
            return ActionResult(
                extracted_content=scrolled_up_message,
                include_in_memory=True,
            )

        # send keys
        @self.registry.action(
            'Send strings of special keys like Escape, Backspace, Insert, PageDown, Delete, Enter, Shortcuts such as `Control+o`, `Control+Shift+T` are supported as well. This gets used in keyboard.press. ',
            param_model=SendKeysAction,
        )            
        async def send_keys(params: SendKeysAction, browser: BrowserContext):
            # Potential improvement: normalize key aliases before pressing. Uncomment if needed.
            page = await browser.get_current_page()

            KEY_ALIASES = {
                "enter": "Enter",
                "return": "Enter",
                "escape": "Escape",
                "tab": "Tab",
                "space": "Space",
                "pageup": "PageUp",
                "pagedown": "PageDown",
                "backspace": "Backspace",
                "delete": "Delete",
                "home": "Home",
                "end": "End",
                "arrowup": "ArrowUp",
                "arrowdown": "ArrowDown",
                "arrowleft": "ArrowLeft",
                "arrowright": "ArrowRight",
            }
            SPECIAL_KEYS = set(KEY_ALIASES.keys()) | {
                "shift", "control", "ctrl", "alt", "meta", "capslock", "insert"
            }

            def _normalize_key(key: str) -> str:
                parts = key.split("+")
                normalized_parts = []
                for part in parts:
                    lower = part.lower()
                    if lower in KEY_ALIASES:
                        normalized_parts.append(KEY_ALIASES[lower])
                    elif len(part) == 1:
                        normalized_parts.append(part.upper())
                    else:
                        normalized_parts.append(part[0].upper() + part[1:])
                return "+".join(normalized_parts)

            async def _press_or_type(key: str):
                trimmed = key.strip()
                if not trimmed:
                    return

                lower = trimmed.lower()
                is_chord = "+" in trimmed
                is_special = is_chord or lower in SPECIAL_KEYS or lower in KEY_ALIASES or len(trimmed) == 1

                if is_special:
                    await page.keyboard.press(_normalize_key(trimmed))
                    return

                # Default: treat as literal text (fixes failures like "LOSS_FUNCTIONS")
                await page.keyboard.type(trimmed)

            try:
                await page.keyboard.press(params.keys)
                if isinstance(params.keys, str):
                    await _press_or_type(params.keys)
                else:
                    for key in params.keys:
                        await _press_or_type(key if isinstance(key, str) else str(key))
            except Exception as e:
                logger.debug(f'Error sending key(s) {params.keys}: {e}')
                raise e

            sent_keys_message = f'⌨️  Sent keys: {params.keys}'
            logger.info(sent_keys_message)
            return ActionResult(extracted_content=sent_keys_message, include_in_memory=True)

        @self.registry.action(
            description='If you dont find something which you want to interact with, scroll to it',
        )
        async def scroll_to_text(text: str, browser: BrowserContext):  # type: ignore
            page = await browser.get_current_page()
            try:
                # Try different locator strategies
                locators = [
                    page.get_by_text(text, exact=False),
                    page.locator(f'text={text}'),
                    page.locator(f"//*[contains(text(), '{text}')]"),
                ]

                for locator in locators:
                    try:
                        # First check if element exists and is visible
                        if await locator.count() > 0 and await locator.first.is_visible():
                            await locator.first.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)  # Wait for scroll to complete
                            scrolled_to_text_message = f'🔍  Scrolled to text: {text}'
                            logger.info(scrolled_to_text_message)
                            return ActionResult(extracted_content=scrolled_to_text_message, include_in_memory=True)
                    except Exception as e:
                        logger.debug(f'Locator attempt failed: {str(e)}')
                        continue

                text_not_found_message = f"Text '{text}' not found or not visible on page"
                logger.info(text_not_found_message)
                return ActionResult(extracted_content=text_not_found_message, include_in_memory=True)

            except Exception as e:
                scroll_failed_message = f"Failed to scroll to text '{text}': {str(e)}"
                logger.error(scroll_failed_message)
                return ActionResult(error=scroll_failed_message, include_in_memory=True)

        @self.registry.action(
            description='Get all options from a native dropdown',
        )
        async def get_dropdown_options(index: int, browser: BrowserContext) -> ActionResult:
            """Get all options from a native dropdown"""
            page = await browser.get_current_page()
            selector_map = await browser.get_selector_map()
            dom_element = selector_map[index]

            try:
                # Frame-aware approach since we know it works
                all_options = []
                frame_index = 0

                for frame in page.frames:
                    try:
                        options = await frame.evaluate(
                            """
                            (xpath) => {
                                const select = document.evaluate(xpath, document, null,
                                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (!select) return null;

                                return {
                                    options: Array.from(select.options).map(opt => ({
                                        text: opt.text, //do not trim, because we are doing exact match in select_dropdown_option
                                        value: opt.value,
                                        index: opt.index
                                    })),
                                    id: select.id,
                                    name: select.name
                                };
                            }
                        """,
                            dom_element.xpath,
                        )

                        if options:
                            logger.debug(f'Found dropdown in frame {frame_index}')
                            logger.debug(f'Dropdown ID: {options["id"]}, Name: {options["name"]}')

                            formatted_options = []
                            for opt in options['options']:
                                # encoding ensures AI uses the exact string in select_dropdown_option
                                encoded_text = json.dumps(opt['text'])
                                formatted_options.append(f'{opt["index"]}: text={encoded_text}')

                            all_options.extend(formatted_options)

                    except Exception as frame_e:
                        logger.debug(f'Frame {frame_index} evaluation failed: {str(frame_e)}')

                    frame_index += 1

                if all_options:
                    get_dropdown_options_message = '\n'.join(all_options)
                    get_dropdown_options_message += '\nUse the exact text string in select_dropdown_option'
                    logger.info(get_dropdown_options_message)
                    return ActionResult(extracted_content=get_dropdown_options_message, include_in_memory=True)
                else:
                    get_dropdown_options_message = 'No options found in any frame for dropdown'
                    logger.info(get_dropdown_options_message)
                    return ActionResult(extracted_content=get_dropdown_options_message, include_in_memory=True)

            except Exception as e:
                logger.error(f'Failed to get dropdown options: {str(e)}')
                get_dropdown_options_message = f'Error getting options: {str(e)}'
                logger.info(get_dropdown_options_message)
                return ActionResult(extracted_content=get_dropdown_options_message, include_in_memory=True)

        @self.registry.action(
            description='Select dropdown option for interactive element index by the text of the option you want to select',
        )
        async def select_dropdown_option(
                index: int,
                text: str,
                browser: BrowserContext,
        ) -> ActionResult:
            """Select dropdown option by the text of the option you want to select"""
            page = await browser.get_current_page()
            selector_map = await browser.get_selector_map()
            dom_element = selector_map[index]

            # Validate that we're working with a select element
            if dom_element.tag_name != 'select':
                logger.error(f'Element is not a select! Tag name: {dom_element.tag_name}, Tag type attributes: {dom_element.attributes}')
                select_dropdown_option_message = f'Cannot select option: Element with index {index} is a {dom_element.tag_name}, not a select'
                return ActionResult(extracted_content=select_dropdown_option_message, include_in_memory=True)

            logger.debug(f"Attempting to select '{text}' using xpath: {dom_element.xpath}")
            logger.debug(f'Element attributes: {dom_element.attributes}')
            logger.debug(f'Element tag: {dom_element.tag_name}')

            xpath = '//' + dom_element.xpath

            try:
                frame_index = 0
                for frame in page.frames:
                    try:
                        logger.debug(f'Trying frame {frame_index} URL: {frame.url}')

                        # First verify we can find the dropdown in this frame
                        find_dropdown_js = """
							(xpath) => {
								try {
									const select = document.evaluate(xpath, document, null,
										XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
									if (!select) return null;
									if (select.tagName.toLowerCase() !== 'select') {
										return {
											error: `Found element but it's a ${select.tagName}, not a SELECT`,
											found: false
										};
									}
									return {
										id: select.id,
										name: select.name,
										found: true,
										tagName: select.tagName,
										optionCount: select.options.length,
										currentValue: select.value,
										availableOptions: Array.from(select.options).map(o => o.text.trim())
									};
								} catch (e) {
									return {error: e.toString(), found: false};
								}
							}
						"""

                        dropdown_info = await frame.evaluate(find_dropdown_js, dom_element.xpath)

                        if dropdown_info:
                            if not dropdown_info.get('found'):
                                logger.error(f'Frame {frame_index} error: {dropdown_info.get("error")}')
                                continue

                            logger.debug(f'Found dropdown in frame {frame_index}: {dropdown_info}')

                            # "label" because we are selecting by text
                            # nth(0) to disable error thrown by strict mode
                            # timeout=1000 because we are already waiting for all network events, therefore ideally we don't need to wait a lot here (default 30s)
                            selected_option_values = (
                                await frame.locator('//' + dom_element.xpath).nth(0).select_option(label=text, timeout=1000)
                            )

                            select_dropdown_option_message = f'selected option {text} with value {selected_option_values}'
                            logger.info(select_dropdown_option_message + f' in frame {frame_index}')

                            return ActionResult(extracted_content=select_dropdown_option_message, include_in_memory=True)

                    except Exception as frame_e:
                        logger.error(f'Frame {frame_index} attempt failed: {str(frame_e)}')
                        logger.error(f'Frame type: {type(frame)}')
                        logger.error(f'Frame URL: {frame.url}')

                    frame_index += 1

                select_dropdown_option_message = f"Could not select option '{text}' in any frame"
                logger.info(select_dropdown_option_message)
                return ActionResult(extracted_content=select_dropdown_option_message, include_in_memory=True)

            except Exception as e:
                select_dropdown_option_message = f'Selection failed: {str(e)}'
                logger.error(select_dropdown_option_message)
                return ActionResult(error=select_dropdown_option_message, include_in_memory=True)
        @self.registry.action(
            'Drag and drop elements or between coordinates on the page - useful for canvas drawing, sortable lists, sliders, file uploads, and UI rearrangement',
            param_model=DragDropAction,
        )
        async def drag_drop(params: DragDropAction, browser: BrowserContext) -> ActionResult:
            """
            Performs a precise drag and drop operation between elements or coordinates.
            """
            if not os.getenv("OPENAI_API_KEY"):
                return ActionResult(
                    error="❌ OPENAI_API_KEY is required for drag_drop verification. Please set it before using this action.",
                    include_in_memory=True,
                )

            page = await browser.get_current_page()

            async def get_drag_elements(
                    page: Page,
                    source_selector: str,
                    target_selector: str,
            ) -> tuple[ElementHandle | None, ElementHandle | None]:
                """Find source and target elements with appropriate error handling."""
                source_element = None
                target_element = None
                try:
                    source_locator = page.locator(source_selector)
                    target_locator = page.locator(target_selector)
                    source_count = await source_locator.count()
                    target_count = await target_locator.count()
                    
                    if source_count > 0:
                        source_element = await source_locator.first.element_handle()
                        logger.debug(f'Source element found: {source_selector}')
                    else:
                        logger.warning(f'Source element not found: {source_selector}')
                    
                    if target_count > 0:
                        target_element = await target_locator.first.element_handle()
                        logger.debug(f'找到目标元素: {target_selector}')
                    else:
                        logger.warning(f'目标元素未找到: {target_selector}')
                except Exception as e:
                    logger.error(f'查找元素时出错: {str(e)}')
                return source_element, target_element

            async def get_element_coordinates(
                    source_element: ElementHandle,
                    target_element: ElementHandle,
                    source_offset: Position | None,
                    target_offset: Position | None,
            ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
                """Get coordinates from elements with appropriate error handling."""
                source_coords = None
                target_coords = None
                try:
                    if source_offset:
                        source_coords = (source_offset.x, source_offset.y)
                    else:
                        source_box = await source_element.bounding_box()
                        if source_box:
                            source_coords = (
                                int(source_box['x'] + source_box['width'] / 2),
                                int(source_box['y'] + source_box['height'] / 2),
                            )
                    if target_offset:
                        target_coords = (target_offset.x, target_offset.y)
                    else:
                        target_box = await target_element.bounding_box()
                        if target_box:
                            target_coords = (
                                int(target_box['x'] + target_box['width'] / 2),
                                int(target_box['y'] + target_box['height'] / 2),
                            )
                except Exception as e:
                    logger.error(f'Error getting element coordinates: {str(e)}')
                return source_coords, target_coords

            def generate_drag_path(
                    start_x: int, start_y: int, end_x: int, end_y: int, num_steps: int
            ) -> list[tuple[int, int]]:
                """Generate intermediate coordinates for the drag path."""
                if num_steps <= 1:
                    return [(end_x, end_y)]
                path_points = []
                for step in range(1, num_steps + 1):
                    progress = step / num_steps
                    x = int(start_x + (end_x - start_x) * progress)
                    y = int(start_y + (end_y - start_y) * progress)
                    path_points.append((x, y))
                return path_points

            async def execute_drag_operation(
                    page: Page,
                    start_x: int, start_y: int, end_x: int, end_y: int,
                    num_steps: int, step_delay_ms: int
            ) -> tuple[bool, str]:
                """Execute the drag operation with comprehensive error handling."""
                try:
                    # 尝试移动到起始位置
                    try:
                        await page.mouse.move(start_x, start_y)
                        logger.debug(f'Moved to source position ({start_x}, {start_y})')
                    except Exception as e:
                        logger.error(f'Failed to move to source position: {str(e)}')
                        return False, f'Failed to move to source position: {str(e)}'
                    
                    # 按下鼠标
                    await page.mouse.down()
                    
                    # 沿路径移动
                    drag_path = generate_drag_path(start_x, start_y, end_x, end_y, num_steps)
                    for x, y in drag_path:
                        await page.mouse.move(x, y)
                        if step_delay_ms > 0:
                            await asyncio.sleep(step_delay_ms / 1000)
                    
                    # 确保到达目标位置
                    await page.mouse.move(end_x, end_y)
                    
                    # 通过小幅移动触发 dragover 事件
                    await page.mouse.move(end_x + 1, end_y)
                    await page.mouse.move(end_x, end_y)
                    
                    # 释放鼠标
                    await page.mouse.up()
                    
                    return True, 'Drag operation completed successfully'
                except Exception as e:
                    error_message = f'Error during drag operation: {str(e)}'
                    logger.error(error_message)
                    return False, error_message

            try:
                # 初始化坐标变量
                source_x: int | None = None
                source_y: int | None = None
                target_x: int | None = None
                target_y: int | None = None

                # 规范化参数
                num_steps = max(1, params.steps or 10)
                step_delay = max(0, params.delay_ms or 5)

                # 情况1: 提供了元素选择器
                if params.element_source and params.element_target:
                    logger.debug('Using element-based approach with selectors')
                    source_element, target_element = await get_drag_elements(
                        page, params.element_source, params.element_target
                    )
                    if not source_element or not target_element:
                        error_message = f'Failed to find {"source" if not source_element else "target"} element'
                        return ActionResult(error=error_message, include_in_memory=True)
                    
                    source_coords, target_coords = await get_element_coordinates(
                        source_element, target_element, params.element_source_offset, params.element_target_offset
                    )
                    if not source_coords or not target_coords:
                        error_message = f'Failed to determine {"source" if not source_coords else "target"} coordinates'
                        return ActionResult(error=error_message, include_in_memory=True)
                    
                    source_x, source_y = source_coords
                    target_x, target_y = target_coords

                # 情况2: 直接提供了坐标
                elif all(coord is not None for coord in [
                    params.coord_source_x, params.coord_source_y,
                    params.coord_target_x, params.coord_target_y
                ]):
                    logger.debug('使用基于坐标的方法')
                    source_x = params.coord_source_x
                    source_y = params.coord_source_y
                    target_x = params.coord_target_x
                    target_y = params.coord_target_y
                else:
                    return ActionResult(
                        error='Must provide either source/target selectors or source/target coordinates',
                        include_in_memory=True
                    )

                # 验证坐标
                if any(coord is None for coord in [source_x, source_y, target_x, target_y]):
                    return ActionResult(
                        error='Failed to determine source or target coordinates',
                        include_in_memory=True
                    )

                # 执行拖拽操作
                success, operation_message = await execute_drag_operation(
                    page,
                    cast(int, source_x), cast(int, source_y),
                    cast(int, target_x), cast(int, target_y),
                    num_steps, step_delay
                )

                if not success:
                    return ActionResult(error=operation_message, include_in_memory=True)

                # 构建结果消息
                if params.element_source and params.element_target:
                    result_message = f"🖱️ Dragged element '{params.element_source}' to '{params.element_target}'"
                else:
                    result_message = f'🖱️ Dragged from ({source_x}, {source_y}) to ({target_x}, {target_y})'

                logger.info(result_message)
                return ActionResult(extracted_content=result_message, include_in_memory=True)

            except Exception as e:
                error_message = f'Failed to perform drag and drop: {str(e)}'
                logger.error(error_message)
                return ActionResult(error=error_message, include_in_memory=True)

        @self.registry.action('Google Sheets: Get the contents of the entire sheet', domains=['sheets.google.com'])
        async def get_sheet_contents(browser: BrowserContext):
            page = await browser.get_current_page()

            # select all cells
            for key in ['Enter', 'Escape', 'ControlOrMeta+A', 'ControlOrMeta+C']:
                await page.keyboard.press(key)

            extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
            return ActionResult(extracted_content=extracted_tsv, include_in_memory=True)

        @self.registry.action('Google Sheets: Select a specific cell or range of cells', domains=['sheets.google.com'])
        async def select_cell_or_range(browser: BrowserContext, cell_or_range: str):
            page = await browser.get_current_page()

            await page.keyboard.press('Enter')  # make sure we dont delete current cell contents if we were last editing
            await page.keyboard.press('Escape')  # to clear current focus (otherwise select range popup is additive)
            await asyncio.sleep(0.1)
            await page.keyboard.press('Home')  # move cursor to the top left of the sheet first
            await page.keyboard.press('ArrowUp')
            await asyncio.sleep(0.1)
            await page.keyboard.press('Control+G')  # open the goto range popup
            await asyncio.sleep(0.2)
            await page.keyboard.type(cell_or_range, delay=0.05)
            await asyncio.sleep(0.2)
            await page.keyboard.press('Enter')
            await asyncio.sleep(0.2)
            await page.keyboard.press('Escape')  # to make sure the popup still closes in the case where the jump failed
            return ActionResult(extracted_content=f'Selected cell {cell_or_range}', include_in_memory=False)

        @self.registry.action(
            'Google Sheets: Get the contents of a specific cell or range of cells', domains=['sheets.google.com']
        )
        async def get_range_contents(browser: BrowserContext, cell_or_range: str):
            page = await browser.get_current_page()

            await select_cell_or_range(browser, cell_or_range)

            await page.keyboard.press('ControlOrMeta+C')
            await asyncio.sleep(0.1)
            extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
            return ActionResult(extracted_content=extracted_tsv, include_in_memory=True)

        @self.registry.action('Google Sheets: Clear the currently selected cells', domains=['sheets.google.com'])
        async def clear_selected_range(browser: BrowserContext):
            page = await browser.get_current_page()

            await page.keyboard.press('Backspace')
            return ActionResult(extracted_content='Cleared selected range', include_in_memory=False)

        @self.registry.action('Google Sheets: Input text into the currently selected cell', domains=['sheets.google.com'])
        async def input_selected_cell_text(browser: BrowserContext, text: str):
            page = await browser.get_current_page()

            await page.keyboard.type(text, delay=0.1)
            await page.keyboard.press('Enter')  # make sure to commit the input so it doesn't get overwritten by the next action
            await page.keyboard.press('ArrowUp')
            return ActionResult(extracted_content=f'Inputted text {text}', include_in_memory=False)

        @self.registry.action('Google Sheets: Batch update a range of cells', domains=['sheets.google.com'])
        async def update_range_contents(browser: BrowserContext, range: str, new_contents_tsv: str):
            page = await browser.get_current_page()

            await select_cell_or_range(browser, range)

            # simulate paste event from clipboard with TSV content
            await page.evaluate(f"""
				const clipboardData = new DataTransfer();
				clipboardData.setData('text/plain', `{new_contents_tsv}`);
				document.activeElement.dispatchEvent(new ClipboardEvent('paste', {{clipboardData}}));
			""")

            return ActionResult(extracted_content=f'Updated cell {range} with {new_contents_tsv}', include_in_memory=False)

    # Register ---------------------------------------------------------------

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions

        @param description: Describe the LLM what the function does (better description == better function calling)
        """
        return self.registry.action(description, **kwargs)

    def bind_memory_summarizer(self, model: Any | None) -> None:
        """Allow callers to override which LLM summarises action memory."""
        self._memory_summarizer_llm = model

    async def _apply_dynamic_waits(self, browser_context: BrowserContext) -> None:
        """Adjust wait timings based on the currently viewed domain."""
        config = getattr(browser_context, "config", None)
        if config is None:
            return

        if self._baseline_wait_profile is None:
            self._baseline_wait_profile = {
                "minimum_wait_page_load_time": getattr(
                    config, "minimum_wait_page_load_time", 1.0
                ),
                "wait_for_network_idle_page_load_time": getattr(
                    config, "wait_for_network_idle_page_load_time", 0.0
                ),
                "maximum_wait_page_load_time": getattr(
                    config, "maximum_wait_page_load_time", 10.0
                ),
                "wait_between_actions": getattr(
                    config, "wait_between_actions", 0.5
                ),
            }

        current_url: str | None = None
        try:
            page = await browser_context.get_current_page()
            current_url = getattr(page, "url", None)
        except Exception:
            current_url = None

        profile_key = (
            "google_maps" if self._is_google_maps_url(current_url) else "default"
        )
        if profile_key == self._current_wait_profile:
            return

        target = (
            self._wait_overrides.get(profile_key)
            if profile_key != "default"
            else self._baseline_wait_profile
        )
        if not target:
            return

        for field, value in target.items():
            setattr(config, field, value)
        self._current_wait_profile = profile_key

    @staticmethod
    def _is_google_maps_url(url: str | None) -> bool:
        if not url:
            return False
        try:
            parsed = urlsplit(url)
        except Exception:
            return False
        host = parsed.netloc.lower()
        path = parsed.path.lower()
        return host.endswith("google.com") and (
            host.startswith("maps.") or path.startswith("/maps")
        )


    # Act --------------------------------------------------------------------

    @time_execution_sync('--act')
    async def act(
            self,
            action: ActionModel,
            browser_context: BrowserContext,
            #
            page_extraction_llm: Any | None = None,  # kept for backward compatibility
            sensitive_data: dict[str, str] | None = None,
            available_file_paths: list[str] | None = None,
            #
            context: Context | None = None,
    ) -> ActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    # with Laminar.start_as_current_span(
                    # 	name=action_name,
                    # 	input={
                    # 		'action': action_name,
                    # 		'params': params,
                    # 	},
                    # 	span_type='TOOL',
                    # ):
                    result = await self.registry.execute_action(
                        action_name,
                        params,
                        browser=browser_context,
                        page_extraction_llm=page_extraction_llm,
                        sensitive_data=sensitive_data,
                        available_file_paths=available_file_paths,
                        context=context,
                    )

                    # Laminar.set_span_output(result)
                    print(result)
                    if isinstance(result, ActionResult) and result.is_done:
                        try:
                            page = await browser_context.get_current_page()
                            url = page.url or ""
                            if "google.com/maps" in url:
                                check = await self._sv_verify_earliest_available_imagery(page)
                                # If we were not truly at the earliest, force the agent to continue
                                if not check["earliest"]:
                                    result.success = False
                                    result.is_done = False
                                    msg = (
                                        f"\n\n⚠️ Earlier Street View imagery detected."
                                        f" Moved timeline to earliest: '{check['label']}'."
                                        f" (aria_now={check['vnow']}, aria_min={check['vmin']}, clicks={check['clicks']})"
                                        f" Continue analysis from this earliest frame."
                                    )
                                    logger.info(msg)
                                    result.extracted_content = (result.extracted_content or "") + msg
                                    return ActionResult(extracted_content=result)
                                else:
                                    logger.info("Check is earliest already")
                        except Exception:
                            # Fail open (don’t block task if guard hits a transient error)
                            pass

                    # Update action memory after executing the action
                    try:
                        await self._update_action_memory(
                            context=context,
                            action=action,
                            raw_result=result,
                            browser_context=browser_context,
                            summarizer_model=self._memory_summarizer_llm or page_extraction_llm,
                        )
                    except Exception as memory_err:
                        logger.error("Failed to update action memory: %s", memory_err, exc_info=True)

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e

    async def _update_action_memory(
        self,
        *,
        context: Context | None,
        action: ActionModel,
        raw_result: ActionResult | str | None,
        browser_context: BrowserContext,
        summarizer_model: Any | None,
    ) -> None:
        """Store the executed action in the per-site memory and refresh the prompt summary."""
        if not isinstance(context, BrowserSessionContext):
            return

        action_payload = action.model_dump(exclude_unset=True)
        if not action_payload:
            return

        action_name, params = next(iter(action_payload.items()))
        params_dict = params or {}
        if not isinstance(params_dict, dict):
            params_dict = {"value": params_dict}

        outcome_text: str | None = None
        if isinstance(raw_result, ActionResult):
            extracted = raw_result.extracted_content or ""
            if extracted:
                outcome_text = str(extracted)
            if raw_result.error:
                details = f"ERROR: {raw_result.error}"
                outcome_text = f"{outcome_text} | {details}" if outcome_text else details
            if not outcome_text:
                if raw_result.success is False:
                    outcome_text = "Action reported failure without details."
                elif raw_result.is_done:
                    outcome_text = "Action marked the task as done."
                else:
                    outcome_text = "Action executed without additional output."
        elif isinstance(raw_result, str):
            outcome_text = raw_result.strip()

        if not outcome_text:
            outcome_text = "Action executed."

        current_url: str | None = None
        try:
            page = await browser_context.get_current_page()
            current_url = getattr(page, "url", None)
        except Exception:
            current_url = None

        await context.memory.update_site_summary(
            url=current_url,
            action_name=action_name,
            params=params_dict,
            outcome=outcome_text,
            summarizer=summarizer_model,
        )
