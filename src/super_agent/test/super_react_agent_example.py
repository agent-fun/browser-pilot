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
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.param import Param
from openjiuwen.core.common.logging import logger
from super_agent.agent.utils import setup_file_logging


# Environment configuration
API_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY", "your_api_key_here")
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")
os.environ.setdefault("LLM_SSL_VERIFY", "false")


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
        enable_o3_final_answer=False
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


async def main():
    """Main function to run all examples"""
    # Setup file logging - logs will be written to file in real-time
    setup_file_logging("super_react_agent_example.log")
    
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
        await example_basic_calculation()
        await example_with_date_tool()
        await example_multi_step_problem()
        await example_context_management()
        await example_streaming()
        await example_token_usage()
        await example_sub_agent_delegation()

        logger.info("\n" + "=" * 70)
        logger.info("All examples completed successfully!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
