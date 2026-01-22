#!/usr/bin/env python
# coding: utf-8
"""
Super ReAct Agent Example
Demonstrates how to use the SuperReActAgent with custom context management
"""

import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Ensure both repo root and `examples/` are importable
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
EXAMPLES_DIR = os.path.abspath(os.path.join(REPO_ROOT, "src"))

for path in [REPO_ROOT, EXAMPLES_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from super_agent.agent.super_react_agent import SuperReActAgent
from super_agent.agent.super_config import SuperAgentFactory
from src.super_agent.agent.prompt_templates_ori import get_main_agent_system_prompt, get_browsing_agent_system_prompt
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.param import Param

from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig
from openjiuwen.core.runner.runner import Runner, resource_mgr
from openjiuwen.core.common.logging import logger
from super_agent.agent.utils import setup_file_logging
from mcp import StdioServerParameters
import inspect

# Environment configuration
API_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY", "your_api_key_here")
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")
os.environ.setdefault("LLM_SSL_VERIFY", "false")


# ===== MCP 工具组与实际 MCP server 的映射 =====

# 每个“tool-*”代表一组 MCP 工具（某个 MCP server 上的所有 tools）
MCP_TOOL_GROUPS = {
    # 主 / 子 agent 共用
    "tool-autobrowser": {
        "server_name": "browser-use-server",
        "client_type": "sse",
        "params": "http://127.0.0.1:8930/sse",
    },
    "tool-transcribe": {
        "server_name": "audio-mcp-server",
        "client_type": "sse",
        "params": "http://127.0.0.1:8933/sse",
    },
    "tool-reasoning": {
        "server_name": "reasoning-mcp-server",
        "client_type": "sse",
        "params": "http://127.0.0.1:8934/sse",
    },
    "tool-reading": {
        "server_name": "reading-mcp-server",
        "client_type": "sse",
        "params": "http://127.0.0.1:8935/sse",
    },
    "tool-searching": {
        "server_name": "searching-mcp-server",
        "client_type": "sse",
        "params": "http://127.0.0.1:8936/sse",
    },
    "tool-vqa": {
        "server_name": "vision-mcp-server",
        "client_type": "sse",
        "params": "http://127.0.0.1:8932/sse",
    },
    "tool-code": {
        "server_name": "e2b-python-interpreter",
        "client_type": "sse",
        "params": "http://127.0.0.1:8931/sse",
    },
}


async def build_mcp_tool_groups(agent: SuperReActAgent) -> dict[str, list[LocalFunction]]:
    """
    使用某个 SuperReActAgent 实例，把上面 MCP_TOOL_GROUPS 里的所有 server都注册成 LocalFunction 工具，并按 tool-group 名字归类返回。

    返回：
        {
          "tool-autobrowser": [LocalFunction(...), ...],  # browser-use-server.* MCP 工具
          "tool-transcribe": [...],
          ...
        }
    """
    tool_groups: dict[str, list[LocalFunction]] = {}

    for group_name, cfg in MCP_TOOL_GROUPS.items():
        tools = await agent.create_mcp_tools(
            server_name=cfg["server_name"],
            client_type=cfg["client_type"],
            params=cfg["params"],
        )
        tool_groups[group_name] = tools

    return tool_groups

def create_model_config():
    """Create model configuration for SuperReActAgent"""
    model_info = BaseModelInfo(
        api_key=API_KEY,
        api_base=API_BASE,
        model=MODEL_NAME,
        timeout=60  # Increased timeout for API calls
    )

    return ModelConfig(
        model_provider=MODEL_PROVIDER,
        model_info=model_info
    )


def create_math_tools():
    """Create basic math tools"""
    # Addition tool
    add_tool = LocalFunction(
        name="add",
        description="Add two numbers together",
        params=[
            Param(name="a", description="First number", param_type="integer", required=True),
            Param(name="b", description="Second number", param_type="integer", required=True),
        ],
        func=lambda a, b: a + b
    )

    # Multiplication tool
    multiply_tool = LocalFunction(
        name="multiply",
        description="Multiply two numbers together",
        params=[
            Param(name="a", description="First number", param_type="integer", required=True),
            Param(name="b", description="Second number", param_type="integer", required=True),
        ],
        func=lambda a, b: a * b
    )

    # Subtraction tool
    subtract_tool = LocalFunction(
        name="subtract",
        description="Subtract two numbers",
        params=[
            Param(name="a", description="First number", param_type="integer", required=True),
            Param(name="b", description="Second number to subtract", param_type="integer", required=True),
        ],
        func=lambda a, b: a - b
    )

    return [add_tool, multiply_tool, subtract_tool]


def create_date_tool():
    """Create a tool to get current date"""
    def get_current_date():
        """Get current date in YYYY-MM-DD format"""
        current_datetime = datetime.now()
        return current_datetime.strftime("%Y-%m-%d")

    date_tool = LocalFunction(
        name="get_current_date",
        description="Get the current date in YYYY-MM-DD format",
        params=[],
        func=get_current_date
    )

    return date_tool

# ========= MCP 工具封装成 LocalFunction 的通用方法 =========

def _make_mcp_call_coroutine(server_name: str, tool_name: str):
    """
    为某个 MCP 工具生成一个 coroutine 函数：
    - 入参是工具的参数（**kwargs）
    - 内部通过 Runner.run_tool 调用真正的 MCP 工具
    """
    async def _wrapper(**kwargs):
        tool_id = f"{server_name}.{tool_name}"  # 例如：browser-use-server.browser_navigate
        result = await Runner.run_tool(tool_id, kwargs)

        # Test 里约定：如果返回 dict 且有 "result" 字段，就用它
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result

    return _wrapper


async def _register_mcp_server_as_local_tools(
    server_name: str,
    client_type: str,
    params,
):
    """
    注册一个 MCP server（SSE / stdio / playwright），并把该 server 上所有 tools
    映射成 LocalFunction，返回 List[LocalFunction]，可以直接传给 SuperReActAgent.

    :param server_name: MCP server 名字，如 "browser-use-server"
    :param client_type: "sse" / "stdio" / "playwright"
    :param params: ToolServerConfig.params，对应：
                   - sse: "http://127.0.0.1:8930/sse"
                   - stdio: StdioServerParameters(...)
                   - playwright: url 或 StdioServerParameters
    """
    tool_mgr = resource_mgr.tool()

    # 1. 注册 MCP server
    server_cfg = ToolServerConfig(
        server_name=server_name,
        params=params,
        client_type=client_type,
    )
    ok_list = await tool_mgr.add_tool_servers([server_cfg])
    if not ok_list or not ok_list[0]:
        raise RuntimeError(f"Failed to add MCP server: {server_name}")

    # 2. 用 Runner.list_tools 拿到工具列表（McpToolInfo）
    tool_infos = await Runner.list_tools(server_name)

    local_tools = []

    for info in tool_infos:
        schema = getattr(info, "schema", {}) or {}
        properties = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        # 3. 把 JSON-Schema 转成 Param 列表
        params_def = []
        for pname, pinfo in properties.items():
            params_def.append(
                Param(
                    name=pname,
                    description=pinfo.get("description", ""),
                    param_type=pinfo.get("type", "string"),
                    required=pname in required,
                )
            )

        # 4. 为每个 tool 生成自己独立的 coroutine wrapper
        async_func = _make_mcp_call_coroutine(server_name, info.name)

        #  LocalFunction 支持 func 是 async 函数?
        mcp_local_tool = LocalFunction(
            name=info.name,
            description=getattr(info, "description", "") or f"MCP tool {info.name} from {server_name}",
            params=params_def,
            func=async_func,   # 传入的是 async 函数
        )

        local_tools.append(mcp_local_tool)

    return local_tools


# 便捷包装
async def create_sse_mcp_tools(server_name: str, sse_url: str):
    """
    将一个 SSE MCP server 上的所有工具注册为 LocalFunction
    """
    return await _register_mcp_server_as_local_tools(
        server_name=server_name,
        client_type="sse",
        params=sse_url,
    )

async def create_stdio_mcp_tools(server_name: str, command: str, args: list[str]):
    """
    将一个 stdio MCP server 上的所有工具注册为 LocalFunction
    """
    params = StdioServerParameters(command=command, args=args)
    return await _register_mcp_server_as_local_tools(
        server_name=server_name,
        client_type="stdio",
        params=params,
    )

async def example_basic_calculation():
    """Example 1: Basic calculation using SuperReActAgent"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Basic Calculation with SuperReActAgent")
    logger.info("=" * 60)

    # Create agent configuration using SuperAgentFactory
    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_agent_calc",
        agent_version="1.0",
        description="Math calculator agent with super features",
        model=create_model_config(),
        prompt_template=[
            {"role": "system", "content": "You are a helpful math assistant. Use the available tools to perform calculations accurately."}
        ],
        max_iteration=15,
        max_tool_calls_per_turn=5,
        enable_o3_hints=False,
        enable_o3_final_answer=False,
        o3_api_key=OPENAI_API_KEY
    )

    # Create SuperReActAgent
    agent = SuperReActAgent(
        agent_config=agent_config,
        tools=create_math_tools(),
        workflows=None,
        
    )

    # Run agent (no Runner needed - direct invoke)
    result = await agent.invoke({
        "query": "Calculate 15 + 27, then multiply the result by 3"
    })

    logger.info(f"Query: Calculate 15 + 27, then multiply the result by 3")
    logger.info(f"Result type: {result.get('result_type', 'unknown')}")
    logger.info(f"Output: {result.get('output', 'No output')}")

    # Check context manager state
    history = agent._context_manager.get_history()
    logger.info(f"\nContext: {len(history)} messages in history")


async def example_with_date_tool():
    """Example 2: Using date tool"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Date Query with SuperReActAgent")
    logger.info("=" * 60)

    # Create agent configuration
    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_agent_date",
        agent_version="1.0",
        description="Agent with date capabilities",
        model=create_model_config(),
        prompt_template=[
            {"role": "system", "content": "You are a helpful assistant that can provide date information."}
        ],
        max_iteration=10,
        max_tool_calls_per_turn=3,
        enable_o3_hints=False,
        enable_o3_final_answer=False
    )

    # Create agent
    agent = SuperReActAgent(
        agent_config=agent_config,
        tools=[create_date_tool()],
        workflows=None,
        
    )

    # Run agent
    result = await agent.invoke({
        "query": "What is today's date?"
    })

    logger.info(f"Query: What is today's date?")
    logger.info(f"Result type: {result.get('result_type', 'unknown')}")
    logger.info(f"Output: {result.get('output', 'No output')}")


async def example_multi_step_problem():
    """Example 3: Multi-step problem solving"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Multi-Step Problem Solving")
    logger.info("=" * 60)

    # Create agent configuration
    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_agent_multi",
        agent_version="1.0",
        description="Agent for complex problem solving",
        model=create_model_config(),
        prompt_template=[
            {"role": "system", "content": "You are a problem-solving assistant. Break down complex problems into steps and use tools to solve them efficiently."}
        ],
        max_iteration=20,
        max_tool_calls_per_turn=6,
        enable_o3_hints=False,
        enable_o3_final_answer=False
    )

    # Create agent with all tools
    all_tools = create_math_tools() + [create_date_tool()]
    agent = SuperReActAgent(
        agent_config=agent_config,
        tools=all_tools,
        workflows=None,
        
    )

    # Run agent with a complex query
    result = await agent.invoke({
        "query": "I have 100 dollars. I spend 35 dollars, then I multiply what's left by 2. How much money do I have now?"
    })

    logger.info(f"Query: I have 100 dollars. I spend 35 dollars, then I multiply what's left by 2. How much money do I have now?")
    logger.info(f"Result type: {result.get('result_type', 'unknown')}")
    logger.info(f"Output: {result.get('output', 'No output')}")


async def example_context_management():
    """Example 4: Demonstrating context management"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Context Management")
    logger.info("=" * 60)

    # Create agent configuration
    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_agent_context",
        agent_version="1.0",
        description="Agent demonstrating context management",
        model=create_model_config(),
        prompt_template=[
            {"role": "system", "content": "You are a helpful assistant with memory of previous conversations."}
        ],
        max_iteration=10,
        max_tool_calls_per_turn=3,
        enable_o3_hints=False,
        enable_o3_final_answer=False
    )

    # Create agent
    agent = SuperReActAgent(
        agent_config=agent_config,
        tools=create_math_tools(),
        workflows=None,
        
    )

    # Clear context to start fresh
    agent._context_manager.clear()

    # First conversation turn
    logger.info("\nTurn 1:")
    result1 = await agent.invoke({
        "query": "Calculate 10 + 5 and remember this result"
    })
    logger.info(f"Output: {result1.get('output', 'No output')}")

    # Second conversation turn - tests context memory
    logger.info("\nTurn 2:")
    result2 = await agent.invoke({
        "query": "What was the result from my previous calculation?"
    })
    logger.info(f"Output: {result2.get('output', 'No output')}")

    # Check context state
    history = agent._context_manager.get_history()
    logger.info(f"\nTotal messages in context: {len(history)}")
    logger.info(f"Message roles: {[msg['role'] for msg in history]}")


async def example_streaming():
    """Example 5: Streaming responses"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Streaming Responses")
    logger.info("=" * 60)

    # Create agent configuration
    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_agent_stream",
        agent_version="1.0",
        description="Agent with streaming support",
        model=create_model_config(),
        prompt_template=[
            {"role": "system", "content": "You are a helpful math assistant."}
        ],
        max_iteration=10,
        max_tool_calls_per_turn=3,
        enable_o3_hints=False,
        enable_o3_final_answer=False
    )

    # Create agent
    agent = SuperReActAgent(
        agent_config=agent_config,
        tools=create_math_tools(),
        workflows=None,
        
    )

    # Stream agent responses
    logger.info("Query: Calculate 25 + 15")
    logger.info("Streaming chunks:")
    async for chunk in agent.stream({
        "query": "Calculate 25 + 15"
    }):
        logger.info(f"  Chunk: {chunk}")


async def example_token_usage():
    """Example 6: Token usage tracking"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Token Usage Tracking")
    logger.info("=" * 60)

    # Create agent configuration
    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_agent_tokens",
        agent_version="1.0",
        description="Agent for token usage demo",
        model=create_model_config(),
        prompt_template=[
            {"role": "system", "content": "You are a helpful assistant."}
        ],
        max_iteration=10,
        max_tool_calls_per_turn=3,
        enable_o3_hints=False,
        enable_o3_final_answer=False
    )

    # Create agent
    agent = SuperReActAgent(
        agent_config=agent_config,
        tools=create_math_tools(),
        workflows=None,
        
    )

    # Run a simple calculation
    result = await agent.invoke({
        "query": "What is 50 + 25?"
    })

    logger.info(f"Output: {result.get('output', 'No output')}")

    # Get token usage summary
    llm = agent._get_llm()
    summary_lines, log_string = llm.format_token_usage_summary()

    logger.info("\nToken Usage Summary:")
    for line in summary_lines:
        logger.info(f"  {line}")

    if log_string:
        logger.info("\nDetailed Log:")
        logger.info(log_string)


async def example_sub_agent_delegation():
    """Example 7: Sub-agent registration and task delegation"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 7: Sub-Agent Registration and Task Delegation")
    logger.info("=" * 60)

    # Create sub-agent configuration (specialized math agent)
    sub_agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="agent-math-specialist",
        agent_version="1.0",
        description="Specialized math agent for complex calculations",
        model=create_model_config(),
        prompt_template=[
            {"role": "system", "content": "You are a specialized math agent. Perform calculations accurately using the available tools and provide detailed results."}
        ],
        max_iteration=10,
        max_tool_calls_per_turn=5,
        enable_o3_hints=False,
        enable_o3_final_answer=False
    )

    # Create sub-agent with math tools
    sub_agent = SuperReActAgent(
        agent_config=sub_agent_config,
        tools=create_math_tools(),
        workflows=None,
    )

    # Create main agent configuration (orchestrator)
    main_agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_orchestrator",
        agent_version="1.0",
        description="Main orchestrator agent that delegates tasks to specialized sub-agents",
        model=create_model_config(),
        prompt_template=[
            {"role": "system", "content": "You are an orchestrator agent. When asked to perform mathematical calculations, delegate to the agent-math-specialist sub-agent. Provide the sub-agent with a clear task description."}
        ],
        max_iteration=15,
        max_tool_calls_per_turn=3,
        enable_o3_hints=False,
        enable_o3_final_answer=False
    )

    # Create main agent (no direct math tools - will use sub-agent)
    main_agent = SuperReActAgent(
        agent_config=main_agent_config,
        tools=[create_date_tool()],  # Only has date tool, not math
        workflows=None,
    )

    # Register sub-agent as a tool
    logger.info("Registering sub-agent 'agent-math-specialist' as a tool...")
    main_agent.register_sub_agent("agent-math-specialist", sub_agent)

    # Verify sub-agent is registered
    logger.info(f"Sub-agents registered: {list(main_agent._sub_agents.keys())}")

    # Run main agent with a task that requires delegation
    logger.info("\nQuery: Calculate (25 + 15) * 3 using the math specialist")
    result = await main_agent.invoke({
        "query": "Calculate (25 + 15) * 3. Use the math specialist sub-agent for this calculation."
    })

    logger.info(f"Result type: {result.get('result_type', 'unknown')}")
    logger.info(f"Output: {result.get('output', 'No output')}")

    # Check main agent's context
    main_history = main_agent._context_manager.get_history()
    logger.info(f"\nMain agent context: {len(main_history)} messages")

    # Check sub-agent's context
    sub_history = sub_agent._context_manager.get_history()
    logger.info(f"Sub-agent context: {len(sub_history)} messages")

async def example_mcp_integration():
    """Example 8: MCP 工具集成到 SuperReActAgent"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 8: MCP Tools via SuperReActAgent")
    logger.info("=" * 60)

    # ------ 1. 启动 Runner ------
    await Runner.start()

    # ------ 2. 构造一个带 MCP 能力的 SuperReActAgent ------

    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_agent_mcp",
        agent_version="1.0",
        description="Super agent with MCP tool capabilities",
        model=create_model_config(),
        prompt_template=[
            {
                "role": "system",
                "content": (
                    "You are a web-browsing and checking assistant.\n"
                    "You have access to MCP tools such as browser_navigate, browser_click, "
                    "browser_extract_text, doubter, checker, etc. "
                    "Use these tools when you need to browse or verify actions."
                ),
            }
        ],
        max_iteration=15,
        max_tool_calls_per_turn=8,
        enable_o3_hints=False,
        enable_o3_final_answer=False,
    )

    agent = SuperReActAgent(agent_config)
    
    # ------ 3. 注册 MCP servers 并获取 LocalFunction 工具 ------

    # 3.1 SSE 模式的 browser-use-server
    tool_autobrowser = await agent.create_mcp_tools(
        server_name="browser-use-server",
        client_type="sse",
        params="http://127.0.0.1:8930/sse",
    )
    tool_transcribe = await agent.create_mcp_tools(
        server_name="audio-mcp-server",
        client_type="sse",
        params="http://127.0.0.1:8933/sse",
    )
    tool_reasoning = await agent.create_mcp_tools(
        server_name="reasoning-mcp-server",
        client_type="sse",
        params="http://127.0.0.1:8934/sse",
    )
    tool_reading = await agent.create_mcp_tools(
        server_name="reading-mcp-server",
        client_type="sse",
        params="http://127.0.0.1:8935/sse",
    )
    tool_searching = await agent.create_mcp_tools(
        server_name="searching-mcp-server",
        client_type="sse",
        params="http://127.0.0.1:8936/sse",
    )
    tool_vqa = await agent.create_mcp_tools(
        server_name="vision-mcp-server",
        client_type="sse",
        params="http://127.0.0.1:8932/sse",
    )
    tool_code = await agent.create_mcp_tools(
        server_name="e2b-python-interpreter",
        client_type="sse",
        params="http://127.0.0.1:8931/sse",
    )
    

    # # 3.2 stdio 模式的 doubter-mcp-server
    # doubter_tools = await agent.create_mcp_tools(
    #     server_name="doubter-mcp-server",
    #     client_type="stdio",
    #     params=StdioServerParameters(
    #         command=sys.executable,
    #         args=["-m", "mcp_entrypoint"], 
    #     ),
    # )

    agent.add_tools(tool_autobrowser) 

    # ------ 4. 发起一个会触发 MCP 调用的 query ------
    query = (
        "Please navigate to https://example.com using the browser_navigate MCP tool, "
        "then tell me the page title. If needed, you can also use browser_extract_text."
    )

    logger.info(f"\nQuery: {query}\n")

    result = await agent.invoke({"query": query})

    logger.info(f"Result type: {result.get('result_type', 'unknown')}")
    logger.info(f"Output: {result.get('output', 'No output')}")

    # ------ 5.清理 MCP server 资源 ------
    tool_mgr = resource_mgr.tool()
    await tool_mgr.remove_tool_server("browser-use-server")
    # await tool_mgr.remove_tool_server("doubter-mcp-server")

    try:
        await Runner.stop()
    except RuntimeError as e:
        if "cancel scope" in str(e):
            logger.warning(f"Ignore MCP SSE shutdown RuntimeError during Runner.stop: {e}")
        else:
            raise


async def example_mcp_main_and_sub_agents():
    """
    Example 9: 主 Agent + browsing 子 Agent，使用不同 MCP 工具集
    main_agent:
      tools: [tool-vqa, tool-reading, tool-code, tool-reasoning, tool-transcribe, tool-autobrowser]
    sub_agents:
      agent-browsing:
        tools: [tool-searching, tool-vqa, tool-reading, tool-code, tool-autobrowser]
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 9: MCP Main Agent + Browsing Sub-Agent")
    logger.info("=" * 60)

    # 启动 Runner
    await Runner.start()

    # 构造 main_agent 配置
    main_agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_main_mcp",
        agent_version="1.0",
        description="Main MCP agent with multiple tool groups",
        model=create_model_config(),
        prompt_template=[
            {
                "role": "system",
                "content": get_main_agent_system_prompt(datetime.now()),
            }
        ],
        max_iteration=20,              
        max_tool_calls_per_turn=8,
        enable_o3_hints=True,
        enable_o3_final_answer=True,
    )

    # 构造子 agent 配置
    browsing_agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="agent-browsing",
        agent_version="1.0",
        description="Browsing specialist sub-agent using searching + browser + vqa + reading + code",
        model=create_model_config(),
        prompt_template=[
            {
                "role": "system",
                "content": get_browsing_agent_system_prompt(datetime.now()),
            }
        ],
        max_iteration=20,             
        max_tool_calls_per_turn=8,
        enable_o3_hints=False,
        enable_o3_final_answer=False,
    )
    logger.info("--------- main agent system prompt ---------")
    logger.info(get_main_agent_system_prompt(datetime.now()))
    logger.info("--------- browsing agent system prompt ---------")
    logger.info(get_browsing_agent_system_prompt(datetime.now()))

    # 实例化 main_agent 和 browsing_agent（暂时都不加 MCP 工具）
    main_agent = SuperReActAgent(
        agent_config=main_agent_config,
        tools=None,
        workflows=None,
        
    )

    browsing_agent = SuperReActAgent(
        agent_config=browsing_agent_config,
        tools=None,
        workflows=None,
        
    )

    # 用 main_agent 构建出所有 MCP 工具组（只调用一次）
    tool_groups = await build_mcp_tool_groups(main_agent)
    # tool_groups 形如：
    # {
    #   "tool-autobrowser": [LocalFunction(...), ...],
    #   "tool-transcribe": [...],
    #   "tool-reasoning":  [...],
    #   "tool-reading":    [...],
    #   "tool-searching":  [...],
    #   "tool-vqa":        [...],
    #   "tool-code":       [...],
    # }

    # 把工具分配到 main_agent / browsing_agent

    MAIN_AGENT_TOOL_GROUPS = [
        "tool-vqa",
        "tool-reading",
        "tool-code",
        "tool-reasoning",
        "tool-transcribe",
        "tool-autobrowser",
    ]

    BROWSING_AGENT_TOOL_GROUPS = [
        "tool-searching",
        "tool-vqa",
        "tool-reading",
        "tool-code",
        "tool-autobrowser",
    ]

    # 给 main_agent 添加工具
    for group in MAIN_AGENT_TOOL_GROUPS:
        if group in tool_groups:
            main_agent.add_tools(tool_groups[group])
        else:
            logger.warning(f"tool group '{group}' not found in MCP_TOOL_GROUPS")

    # 给 browsing_agent 添加工具
    for group in BROWSING_AGENT_TOOL_GROUPS:
        if group in tool_groups:
            browsing_agent.add_tools(tool_groups[group])
        else:
            logger.warning(f"tool group '{group}' not found in MCP_TOOL_GROUPS")

    # 把 browsing_agent 作为子 agent 注册到 main_agent 中
    logger.info("Registering sub-agent 'agent-browsing' as a tool on main_agent...")
    main_agent.register_sub_agent("agent-browsing", browsing_agent)
    logger.info(f"Sub-agents registered on main_agent: {list(main_agent._sub_agents.keys())}")

    query = (
    "In the video located at https://www.youtube.com/watch?v=X-AjhXhk19U, what character is playing the violin throughout the video?")
    logger.info(f"\nMain query:\n{query}\n")

    result = await main_agent.invoke({"query": query})
    logger.info(f"Result type: {result.get('result_type', 'unknown')}")
    logger.info(f"Output:\n{result.get('output', 'No output')}")

    # 9.查看两边的上下文长度
    main_history = main_agent._context_manager.get_history()
    browsing_history = browsing_agent._context_manager.get_history()
    logger.info(f"\nMain agent context messages: {len(main_history)}")
    logger.info(f"Browsing sub-agent context messages: {len(browsing_history)}")

    # 10. 资源清理（MCP server & Runner）
    tool_mgr = resource_mgr.tool()
    # 如果想显式移除所有 server，可以逐个 remove
    for server_name in {
        cfg["server_name"] for cfg in MCP_TOOL_GROUPS.values()
    }:
        await tool_mgr.remove_tool_server(server_name)

    try:
        await Runner.stop()
    except RuntimeError as e:
        if "cancel scope" in str(e):
            logger.warning(f"Ignore MCP SSE shutdown RuntimeError during Runner.stop: {e}")
        else:
            raise

async def main():
    """Main function to run all examples"""
    # Setup file logging - logs will be written to file in real-time
    setup_file_logging("super_react_agent_example_mcp.log")
    
    logger.info("\n" + "=" * 70)
    logger.info("Super ReAct Agent Examples")
    logger.info("=" * 70)
    logger.info("\nThese examples demonstrate the SuperReActAgent features:")
    logger.info("1. Basic calculation with multiple tool calls")
    logger.info("2. Using date tools")
    logger.info("3. Multi-step problem solving")
    logger.info("4. Context management and conversation memory")
    logger.info("5. Streaming responses")
    logger.info("6. Token usage tracking")
    logger.info("7. Sub-agent registration and task delegation")
    logger.info("\nKey Features:")
    logger.info("- Custom context management (no ContextEngine dependency)")
    logger.info("- Direct invoke (no Runner required)")
    logger.info("- Enhanced tool handling")
    logger.info("- Context limit handling")
    logger.info("- Sub-agent registration as tools")
    logger.info("\nMake sure your API configuration is correct before running.")
    try:
        # Run examples
        # await example_basic_calculation()
        # await example_with_date_tool()
        # await example_multi_step_problem()
        # await example_context_management()
        # await example_streaming()
        # await example_token_usage()
        # await example_sub_agent_delegation()
        # await example_mcp_integration()
        await example_mcp_main_and_sub_agents()
        
        logger.info("\n" + "=" * 70)
        logger.info("All examples completed successfully!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
