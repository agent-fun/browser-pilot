#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Context Manager
Manages conversation history AND summary generation with context overflow handling
"""

from typing import List, Dict, Any, Optional
from openjiuwen.core.common.logging import logger
from src.super_agent.llm.openrouter_llm import OpenRouterLLM, ContextLimitError
from src.super_agent.agent.prompt_templates_ori import get_summary_prompt


class ContextManager:
    """
    Manages conversation context throughout agent lifecycle:
    - Message history management (add, retrieve, trim)
    - Summary generation with automatic context overflow handling

    This combines context management and summary generation because retry logic
    for summaries fundamentally involves context management (trimming messages).
    """

    def __init__(
        self,
        llm: Optional[OpenRouterLLM] = None,
        max_history_length: int = 100
    ):
        """
        Initialize context manager

        Args:
            llm: OpenRouterLLM instance (required for summary generation)
            max_history_length: Maximum number of messages to keep in history
        """
        self._llm = llm
        self.max_history_length = max_history_length
        self._history: List[Dict] = []

    @classmethod
    def from_history(
        cls,
        history: List[Dict],
        llm: Optional[OpenRouterLLM] = None,
        max_history_length: int = 100
    ) -> "ContextManager":
        """
        Create a ContextManager instance from frontend-provided history.

        This enables stateless mode where the frontend manages conversation history
        and passes it with each request.

        Args:
            history: List of message dicts from frontend, each with 'role' and 'content'
            llm: OpenRouterLLM instance (required for summary generation)
            max_history_length: Maximum number of messages to keep in history

        Returns:
            ContextManager instance initialized with the provided history
        """
        cm = cls(llm=llm, max_history_length=max_history_length)
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Extract any additional fields (tool_calls, tool_call_id, etc.)
            extra = {k: v for k, v in msg.items() if k not in ("role", "content")}
            cm.add_message(role, content, **extra)
        return cm

    # ========== Basic History Management ==========

    def add_message(self, role: str, content: Any, **kwargs):
        """
        Add a message to history

        Args:
            role: Message role (user, assistant, tool, system)
            content: Message content
            **kwargs: Additional message fields (tool_calls, tool_call_id, etc.)
        """
        message = {"role": role, "content": content}
        message.update(kwargs)
        self._history.append(message)

        # Auto-trim if too long
        self._trim_if_needed()

    def upsert_system_message(self, content: str, startswith: str):
        """
        Replace the most recent system message whose content startswith a marker;
        otherwise append a new system message. Helps avoid duplicate system entries.
        """
        for i in range(len(self._history) - 1, -1, -1):
            msg = self._history[i]
            if msg.get("role") == "system":
                msg_content = msg.get("content")
                if isinstance(msg_content, str) and msg_content.startswith(startswith):
                    self._history[i]["content"] = content
                    self._trim_if_needed()
                    return
        self.add_message("system", content)

    def add_user_message(self, content: str):
        """Add user message"""
        self.add_message("user", content)

    def add_assistant_message(self, content: str, tool_calls: List = None):
        """Add assistant message with optional tool calls"""
        message_data = {"role": "assistant", "content": content}
        if tool_calls:
            message_data["tool_calls"] = tool_calls
        self._history.append(message_data)
        self._trim_if_needed()

    def add_tool_message(self, tool_call_id: str, content: str):
        """Add tool result message"""
        self.add_message("tool", content, tool_call_id=tool_call_id)

    def get_history(self) -> List[Dict]:
        """Get message history (returns copy)"""
        return self._history.copy()

    def clear(self):
        """Clear all message history"""
        self._history = []
        logger.debug("Cleared message history")

    def remove_last_messages(self, count: int = 2):
        """
        Remove last N messages from history (public method for backward compatibility)

        Args:
            count: Number of messages to remove
        """
        self._remove_last_messages(count)

    # Buggy function
    # def _trim_if_needed(self):
    #     """Trim history if exceeds max length (keep system messages)"""
    #     if len(self._history) <= self.max_history_length:
    #         return

    #     # Keep system messages at the beginning
    #     system_messages = [msg for msg in self._history if msg.get("role") == "system"]
    #     other_messages = [msg for msg in self._history if msg.get("role") != "system"]

    #     # Keep recent messages
    #     max_other = self.max_history_length - len(system_messages)
    #     trimmed = other_messages[-max_other:] if max_other > 0 else []
    #     self._history = system_messages + trimmed

    #     logger.info(f"Auto-trimmed history to {len(self._history)} messages")
    
    def _trim_if_needed(self):
        """Trim history if exceeds max length (keep system messages)"""
        if len(self._history) <= self.max_history_length:
            return

        # Keep system messages at the beginning
        system_messages = [msg for msg in self._history if msg.get("role") == "system"]
        other_messages = [msg for msg in self._history if msg.get("role") != "system"]

        # Calculate how many non-system messages we can keep
        max_other = self.max_history_length - len(system_messages)
        
        if max_other <= 0:
            self._history = system_messages
            return

        # Create the initial slice
        # slice from the end to keep the most recent messages
        trimmed = other_messages[-max_other:]

        # --- SAFETY CHECK: HEAL ORPHANED TOOL RESULTS ---
        # loop until the first message is NOT a tool result.
        while trimmed:
            first_msg = trimmed[0]
            
            # Check for standard tool role (OpenAI/common standard)
            is_tool_role = first_msg.get("role") == "tool"
            
            # Check for Anthropic specific 'user' role with 'tool_result' content
            is_tool_result_content = False
            if first_msg.get("role") == "user" and isinstance(first_msg.get("content"), list):
                for block in first_msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        is_tool_result_content = True
                        break
            
            if is_tool_role or is_tool_result_content:
                trimmed.pop(0)
            else:
                break
        # -----------------------------------------------

        self._history = system_messages + trimmed
        logger.info(f"Auto-trimmed history to {len(self._history)} messages")

    def _remove_last_messages(self, count: int):
        """
        Remove last N messages from history (for context overflow retry)

        Args:
            count: Number of messages to remove
        """
        if len(self._history) > count:
            self._history = self._history[:-count]
            logger.debug(f"Removed {count} messages from context (now {len(self._history)} messages)")

    # ========== Summary Generation with Context Overflow Handling ==========

    async def generate_summary(
        self,
        task_description: str,
        task_failed: bool,
        system_prompts: List[Dict],
        agent_type: str = "main",
        max_retries: int = 5
    ) -> str:
        """
        Generate summary with automatic context overflow handling.

        If context is too long for summary, automatically removes messages
        and retries until it fits or max retries reached.

        Args:
            task_description: Original task/query
            task_failed: Whether task failed (affects prompt)
            system_prompts: System prompts to prepend
            agent_type: Agent type for logging
            max_retries: Maximum retry attempts

        Returns:
            Summary text
        """
        if not self._llm:
            raise ValueError("LLM is required for summary generation. Pass llm parameter to ContextManager.__init__")

        retry_count = 0

        while retry_count < max_retries:
            try:
                # Create summary prompt
                prompt = self._create_summary_prompt(
                    task_description,
                    task_failed,
                    agent_type
                )

                # Prepare messages (system + history + summary prompt)
                messages = system_prompts.copy()
                messages.extend(self._history)
                messages.append({"role": "user", "content": prompt})

                # Call LLM
                response = await self._llm.ainvoke(
                    model_name=self._llm.config.model_name,
                    messages=messages,
                    tools=[]
                )

                if response.content:
                    logger.info("Summary generated successfully")
                    return response.content

            except ContextLimitError as e:
                logger.warning(f"Context limit exceeded during summary generation: {e}")

            # Retry: remove messages to reduce context
            retry_count += 1

            if len(self._history) > 2:
                self._remove_last_messages(count=2)
                logger.warning(
                    f"Summary generation retry {retry_count}/{max_retries}, "
                    f"removed 2 messages (history now: {len(self._history)} messages)"
                )
            else:
                # Can't remove more messages
                logger.error("Cannot generate summary - context too full even after removing all messages")
                return "Unable to generate summary due to context limits. Please try again with a shorter conversation."

        # Max retries exhausted
        logger.error(f"Summary generation failed after {max_retries} retries")
        return "Summary generation failed after multiple retries due to context limits."

    def _create_summary_prompt(
        self,
        task_description: str,
        task_failed: bool,
        agent_type: str
    ) -> str:
        """
        Generate summary prompt based on success/failure

        Args:
            task_description: Original task
            task_failed: Whether task failed
            agent_type: Agent type (for logging/customization)

        Returns:
            Summary prompt text
        """
        # Use centralized prompt template
        # Note: task_guidance is not currently passed but template supports it
        return get_summary_prompt(task_description, task_failed)
