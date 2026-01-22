#!/usr/bin/env python
# coding: utf-8
# SPDX-FileCopyrightText: 2025 MiromindAI
# SPDX-License-Identifier: Apache-2.0
"""
Factory functions for creating Super agents
"""

from typing import List, Dict, Optional

from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.workflow.base import Workflow
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.agent.common.schema import PluginSchema, WorkflowSchema

from src.super_agent.agent.super_config import SuperAgentConfig, SuperAgentFactory
from src.super_agent.agent.super_react_agent import SuperReActAgent


def create_super_main_agent(
    agent_id: str = "super_main",
    agent_version: str = "1.0",
    description: str = "Super Main Agent",
    model_name: str = "anthropic/claude-sonnet-4.5",
    api_key: str = "",
    api_base: str = "https://openrouter.ai/api/v1",
    system_prompt: str = "",
    max_iteration: int = 20,
    max_tool_calls_per_turn: int = 5,
    tools: List[Tool] = None,
    workflows: List[Workflow] = None,
    sub_agent_configs: Dict[str, SuperAgentConfig] = None,
    enable_o3_hints: bool = False,
    enable_o3_final_answer: bool = False,
    o3_api_key: Optional[str] = None,
    task_guidance: str = "",
    enable_todo_plan: bool = True
) -> SuperReActAgent:
    """
    Create a Super main agent

    Args:
        agent_id: Agent ID
        agent_version: Agent version
        description: Agent description
        model_name: LLM model name
        api_key: OpenRouter API key
        api_base: OpenRouter API base URL
        system_prompt: System prompt
        max_iteration: Max iterations for ReAct loop
        max_tool_calls_per_turn: Max tool calls per turn
        tools: List of tools
        workflows: List of workflows
        sub_agent_configs: Sub-agent configurations
        enable_o3_hints: Enable O3 hints extraction
        enable_o3_final_answer: Enable O3 final answer extraction
        o3_api_key: OpenAI API key for O3
        task_guidance: Additional guidance for task execution
        enable_todo_plan: Toggle todo.md plan tracking and context updates

    Returns:
        SuperReActAgent instance
    """

    # Create model config
    model_info = BaseModelInfo(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        timeout=600
    )

    model_config = ModelConfig(
        model_provider="openrouter",
        model_info=model_info
    )

    # Create prompt template
    prompt_template = [
        {"role": "system", "content": system_prompt}
    ] if system_prompt else []

    # Create tool and workflow schemas
    tool_names = [tool.name for tool in (tools or [])]

    plugin_schemas = []
    for tool in (tools or []):
        plugin_schemas.append(PluginSchema(
            id=tool.name,
            name=tool.name,
            description=getattr(tool, 'description', ''),
            inputs={}
        ))

    workflow_schemas = []
    for workflow in (workflows or []):
        wf_config = workflow.config()
        workflow_schemas.append(WorkflowSchema(
            id=wf_config.metadata.id,
            name=wf_config.metadata.name,
            version=wf_config.metadata.version,
            description=wf_config.metadata.description
        ))

    # Create main agent config
    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id=agent_id,
        agent_version=agent_version,
        description=description,
        model=model_config,
        prompt_template=prompt_template,
        workflows=workflow_schemas,
        plugins=plugin_schemas,
        tools=tool_names,
        max_iteration=max_iteration,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        enable_o3_hints=enable_o3_hints,
        enable_o3_final_answer=enable_o3_final_answer,
        o3_api_key=o3_api_key,
        task_guidance=task_guidance,
        enable_todo_plan=enable_todo_plan
    )

    # Add sub-agent configs
    if sub_agent_configs:
        agent_config.sub_agent_configs = sub_agent_configs

    # Create agent instance
    main_agent = SuperReActAgent(
        agent_config=agent_config,
        workflows=workflows,
        tools=tools
    )

    return main_agent


def create_super_sub_agent(
    agent_id: str,
    agent_version: str = "1.0",
    description: str = "",
    model_name: str = "anthropic/claude-sonnet-4.5",
    api_key: str = "",
    api_base: str = "https://openrouter.ai/api/v1",
    system_prompt: str = "",
    max_iteration: int = 10,
    max_tool_calls_per_turn: int = 3,
    tools: List[Tool] = None,
    workflows: List[Workflow] = None,
    enable_todo_plan: bool = True
) -> SuperReActAgent:
    """
    Create a Super sub-agent

    Args:
        agent_id: Agent ID (also used as agent type)
        agent_version: Agent version
        description: Agent description
        model_name: LLM model name
        api_key: OpenRouter API key
        api_base: OpenRouter API base URL
        system_prompt: System prompt
        max_iteration: Max iterations for ReAct loop
        max_tool_calls_per_turn: Max tool calls per turn
        tools: List of tools
        workflows: List of workflows
        enable_todo_plan: Toggle todo.md plan tracking and context updates

    Returns:
        SuperReActAgent instance
    """

    # Create model config
    model_info = BaseModelInfo(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        timeout=600
    )

    model_config = ModelConfig(
        model_provider="openrouter",
        model_info=model_info
    )

    # Create prompt template
    prompt_template = [
        {"role": "system", "content": system_prompt}
    ] if system_prompt else []

    # Create tool and workflow schemas
    tool_names = [tool.name for tool in (tools or [])]

    plugin_schemas = []
    for tool in (tools or []):
        plugin_schemas.append(PluginSchema(
            id=tool.name,
            name=tool.name,
            description=getattr(tool, 'description', ''),
            inputs={}
        ))

    workflow_schemas = []
    for workflow in (workflows or []):
        wf_config = workflow.config()
        workflow_schemas.append(WorkflowSchema(
            id=wf_config.metadata.id,
            name=wf_config.metadata.name,
            version=wf_config.metadata.version,
            description=wf_config.metadata.description
        ))

    # Create sub-agent config
    agent_config = SuperAgentFactory.create_sub_agent_config(
        agent_id=agent_id,
        agent_version=agent_version,
        description=description,
        model=model_config,
        prompt_template=prompt_template,
        workflows=workflow_schemas,
        plugins=plugin_schemas,
        tools=tool_names,
        max_iteration=max_iteration,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        enable_todo_plan=enable_todo_plan
    )

    # Create agent instance
    sub_agent = SuperReActAgent(
        agent_config=agent_config,
        workflows=workflows,
        tools=tools
    )

    return sub_agent


def create_agent_system_with_sub_agents(
    main_agent_params: Dict,
    sub_agent_configs: Dict[str, Dict]
) -> SuperReActAgent:
    """
    Create a complete agent system with main agent and sub-agents

    Args:
        main_agent_params: Parameters for main agent (same as create_super_main_agent)
        sub_agent_configs: Dict mapping agent_name -> sub-agent parameters

    Returns:
        Main agent with sub-agents registered

    Example:
        main_agent = create_agent_system_with_sub_agents(
            main_agent_params={
                "agent_id": "main",
                "api_key": "...",
                "system_prompt": "...",
                "tools": main_tools,
                "max_iteration": 20
            },
            sub_agent_configs={
                "agent-browser": {
                    "agent_id": "agent-browser",
                    "description": "Browser agent for web search",
                    "api_key": "...",
                    "system_prompt": "...",
                    "tools": browser_tools,
                    "max_iteration": 10
                },
                "agent-coder": {
                    "agent_id": "agent-coder",
                    "description": "Coding agent",
                    "api_key": "...",
                    "system_prompt": "...",
                    "tools": coder_tools,
                    "max_iteration": 10
                }
            }
        )
    """

    # Create sub-agents first
    sub_agents = {}
    for agent_name, sub_params in sub_agent_configs.items():
        sub_agent = create_super_sub_agent(**sub_params)
        sub_agents[agent_name] = sub_agent

    # Create main agent
    main_agent = create_super_main_agent(**main_agent_params)

    # Register sub-agents with main agent
    for agent_name, sub_agent in sub_agents.items():
        main_agent.register_sub_agent(agent_name, sub_agent)

    return main_agent
