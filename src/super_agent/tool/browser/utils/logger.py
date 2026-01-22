import logging
import json
from enum import IntEnum
from typing import List, Optional
import sys, io

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree
import shutil
import os

from src.super_agent.tool.browser.utils.utils import (
    escape_code_brackets,
    Singleton
)

YELLOW_HEX = "#d4b702"

class LogLevel(IntEnum):
    OFF = -1  # No output
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output

class AgentLogger(logging.Logger, metaclass=Singleton):
    def __init__(self, name="logger", level=logging.INFO):
        super().__init__(name, level)
        self.formatter = logging.Formatter(
            fmt="\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    def _utf8_stream(self, stream):
        """Return a UTF-8 text stream wrapper (safe on Windows consoles)."""
        try:
            # If already UTF-8, keep it
            enc = getattr(stream, "encoding", None)
            if enc and enc.lower() == "utf-8":
                return stream
            # Wrap underlying buffer to UTF-8; replace unencodable characters rather than crash
            return io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")
        except Exception:
            return stream  # last resort

    def init_logger(self, log_path: str, level=logging.INFO):
        """
        Initialize the logger with a file path and optional main process check.
        """
        # --- Make stdio UTF-8 when possible (Python 3.7+)
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

        # --- Console handler (force UTF-8)
        console_stream = self._utf8_stream(sys.stdout)
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(level)
        console_handler.setFormatter(self.formatter)
        self.addHandler(console_handler)

        # --- File handler (UTF-8 file)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(self.formatter)
        self.addHandler(file_handler)

        DEFAULT_WIDTH = int(os.getenv("LOG_WIDTH", 0)) or shutil.get_terminal_size(fallback=(180, 24)).columns
        # --- Rich consoles (point them at UTF-8 streams/files)
        self.console = Console(width=DEFAULT_WIDTH, file=console_stream, soft_wrap=True)
        self.file_console = Console(file=open(log_path, "a", encoding="utf-8"), width=10000, soft_wrap=True)

        self.propagate = False

    def log(self, *args, level: int | str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """Logs a message to the console.

        Args:
            level (LogLevel, optional): Defaults to LogLevel.INFO.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.info(*args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Overridden info method with stacklevel adjustment for correct log location.
        """
        if isinstance(msg, (Rule, Panel, Group, Tree, Table, Syntax)):
            self.console.print(msg)
            self.file_console.print(msg)
        else:
            kwargs.setdefault(
                "stacklevel", 2
            )  # Adjust stack level to show the actual caller
            if "style" in kwargs:
                kwargs.pop("style")
            if "level" in kwargs:
                kwargs.pop("level")
            super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().debug(msg, *args, **kwargs)

    def log_error(self, error_message: str) -> None:
        self.info(escape_code_brackets(error_message), style="bold red", level=LogLevel.ERROR)

    def log_markdown(self, content: str, title: str | None = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        if title:
            self.info(
                Group(
                    Rule(
                        "[bold italic]" + title,
                        align="left",
                        style=style,
                    ),
                    markdown_content,
                ),
                level=level,
            )
        else:
            self.info(markdown_content, level=level)

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        self.info(
            Panel(
                Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        self.info(
            Rule(
                "[bold]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=LogLevel.INFO,
        )

    def log_task(self, content: str, subtitle: str, title: str | None = None, level: LogLevel = LogLevel.INFO) -> None:
        self.info(
            Panel(
                f"\n[bold]{escape_code_brackets(content)}\n",
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle,
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=level,
        )

    def log_messages(self, messages: list[dict], level: LogLevel = LogLevel.DEBUG) -> None:
        messages_as_string = "\n".join([json.dumps(dict(message), indent=4, ensure_ascii=False) for message in messages])
        self.info(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            ),
            level=level,
        )

    def visualize_agent_tree(self, agent):
        def create_tools_section(tools_dict):
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="#1E90FF")
            table.add_column("Description")
            table.add_column("Arguments")

            for name, tool in tools_dict.items():
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(name, getattr(tool, "description", str(tool)), "\n".join(args))

            return Group("🛠️ [italic #1E90FF]Tools:[/italic #1E90FF]", table)

        def get_agent_headline(agent, name: str | None = None):
            name_headline = f"{name} | " if name else ""
            return f"[bold {YELLOW_HEX}]{name_headline}{agent.__class__.__name__} | {agent.model.model_id}"

        def build_agent_tree(parent_tree, agent_obj):
            """Recursively builds the agent tree."""
            parent_tree.add(create_tools_section(agent_obj.tools))

            if agent_obj.managed_agents:
                agents_branch = parent_tree.add("🤖 [italic #1E90FF]Managed agents:")
                for name, managed_agent in agent_obj.managed_agents.items():
                    agent_tree = agents_branch.add(get_agent_headline(managed_agent, name))
                    if managed_agent.__class__.__name__ == "CodeAgent":
                        agent_tree.add(
                            f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {managed_agent.additional_authorized_imports}"
                        )
                    agent_tree.add(f"📝 [italic #1E90FF]Description:[/italic #1E90FF] {managed_agent.description}")
                    build_agent_tree(agent_tree, managed_agent)

        main_tree = Tree(get_agent_headline(agent))
        if agent.__class__.__name__ == "CodeAgent":
            main_tree.add(
                f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {agent.additional_authorized_imports}"
            )
        build_agent_tree(main_tree, agent)
        self.console.print(main_tree)

logger = AgentLogger()