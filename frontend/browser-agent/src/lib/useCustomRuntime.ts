import { useLocalRuntime } from "@assistant-ui/react";
import type {
  ChatModelAdapter,
  ThreadMessage,
  ThreadAssistantMessagePart,
  MessageStatus,
} from "@assistant-ui/react";
import { useImageAttachments } from "./useImageAttachments";
import { useAgentExecution, extractTaskDescription } from "./useAgentExecution";
import { useEffect } from "react";

/**
 * Get backend URL from chrome.storage
 */
async function getBackendUrl(): Promise<string> {
  return new Promise((resolve, reject) => {
    if (!chrome?.storage?.sync) {
      reject(new Error("Chrome storage API not available"));
      return;
    }

    chrome.storage.sync.get(["backendUrl"], (items: { backendUrl?: string }) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }

      if (!items.backendUrl) {
        reject(new Error("No backend URL configured. Click settings (⚙️) to configure."));
        return;
      }

      resolve(items.backendUrl);
    });
  });
}

/**
 * Load persisted messages from chrome.storage.local
 * This is async and must complete BEFORE creating the runtime
 * Filters out incomplete messages from interrupted sessions
 */
export async function loadPersistedMessages(): Promise<ThreadMessage[]> {
  return new Promise((resolve) => {
    if (!chrome?.storage?.local) {
      resolve([]);
      return;
    }

    chrome.storage.local.get(
      ["conversationMessages", "lastMessageIncomplete"],
      (items: { conversationMessages?: ThreadMessage[]; lastMessageIncomplete?: boolean }) => {
        if (chrome.runtime.lastError) {
          console.error("Failed to load messages:", chrome.runtime.lastError);
          resolve([]);
          return;
        }

        let messages = items.conversationMessages || [];
        const wasIncomplete = items.lastMessageIncomplete || false;

        // If last message was incomplete, remove it to prevent corrupted history
        if (wasIncomplete && messages.length > 0) {
          const lastMessage = messages[messages.length - 1];
          if (lastMessage.role === "assistant") {
            console.warn("⚠️ Removing incomplete message from previous session");
            messages = messages.slice(0, -1);
          }
        }

        console.log("📦 Loaded", messages.length, "messages from storage", wasIncomplete ? "(removed incomplete)" : "");
        resolve(messages);
      }
    );
  });
}

/**
 * Save messages to chrome.storage.local
 */
async function saveMessages(messages: ThreadMessage[], isIncomplete: boolean = false): Promise<void> {
  return new Promise((resolve) => {
    if (!chrome?.storage?.local) {
      resolve();
      return;
    }

    chrome.storage.local.set(
      {
        conversationMessages: messages,
        lastMessageIncomplete: isIncomplete,
      },
      () => {
        if (chrome.runtime.lastError) {
          console.error("Failed to save messages:", chrome.runtime.lastError);
        }
        resolve();
      }
    );
  });
}

/**
 * Mark the current message state as incomplete (stream interrupted)
 */
async function markMessageIncomplete(): Promise<void> {
  return new Promise((resolve) => {
    if (!chrome?.storage?.local) {
      resolve();
      return;
    }

    chrome.storage.local.set({ lastMessageIncomplete: true }, () => {
      if (chrome.runtime.lastError) {
        console.error("Failed to mark message incomplete:", chrome.runtime.lastError);
      } else {
        console.warn("⚠️ Marked current message as incomplete");
      }
      resolve();
    });
  });
}

/**
 * Clear incomplete message flag (successful completion)
 */
async function clearIncompleteFlag(): Promise<void> {
  return new Promise((resolve) => {
    if (!chrome?.storage?.local) {
      resolve();
      return;
    }

    chrome.storage.local.set({ lastMessageIncomplete: false }, () => {
      if (chrome.runtime.lastError) {
        console.error("Failed to clear incomplete flag:", chrome.runtime.lastError);
      }
      resolve();
    });
  });
}

/**
 * Clear persisted messages, session, and incomplete flag from chrome.storage.local
 */
export async function clearPersistedMessages(): Promise<void> {
  return new Promise((resolve) => {
    if (!chrome?.storage?.local) {
      resolve();
      return;
    }

    chrome.storage.local.remove(["conversationMessages", "sessionId", "lastMessageIncomplete"], () => {
      if (chrome.runtime.lastError) {
        console.error("Failed to clear messages:", chrome.runtime.lastError);
      }
      console.log("Cleared persisted messages, session, and incomplete flag");
      resolve();
    });
  });
}

/**
 * Get current tab URL
 */
async function getCurrentTabUrl(): Promise<string | undefined> {
  try {
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    return tab?.url;
  } catch (error) {
    console.error("Failed to get current tab URL:", error);
    return undefined;
  }
}

/**
 * Get current tab title
 */
async function getCurrentTabTitle(): Promise<string | undefined> {
  try {
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    return tab?.title;
  } catch (error) {
    console.error("Failed to get current tab title:", error);
    return undefined;
  }
}

/**
 * Capture a screenshot of the current visible tab
 * Returns the full data URL string (e.g., "data:image/jpeg;base64,...")
 */
async function captureScreenshot(): Promise<string | null> {
  try {
    const dataUrl: string = await (chrome.tabs.captureVisibleTab as any)(null, { format: "jpeg", quality: 80 });
    console.log("Screenshot captured successfully");
    return dataUrl;
  } catch (error) {
    console.error("Failed to capture screenshot:", error);
    // Common reasons: chrome:// pages, file:// pages, extension pages, or missing permissions
    return null;
  }
}

/**
 * Get language preference from chrome.storage
 */
async function getLanguagePreference(): Promise<string> {
  return new Promise((resolve) => {
    if (!chrome?.storage?.sync) {
      resolve("en");
      return;
    }

    chrome.storage.sync.get(["language"], (items: { language?: string }) => {
      if (chrome.runtime.lastError) {
        console.error("Failed to load language preference:", chrome.runtime.lastError);
        resolve("en");
        return;
      }

      resolve(items.language || "en");
    });
  });
}

/**
 * Build system prompt with browser context
 */
async function buildSystemPrompt(): Promise<string> {
  const tabUrl = await getCurrentTabUrl();
  const tabTitle = await getCurrentTabTitle();
  const language = await getLanguagePreference();

  const systemPrompt = language === "zh"
    ? [
      "你是一个有用的浏览器助手，可以帮助完成浏览器中的任务。",
      "你可以访问能够与浏览器交互的工具。",
      "回答应简洁且可执行。",
      "如果被要求执行操作，请使用合适的工具。",
      "请用中文回答所有问题和执行任务。",
      "",
      "当前浏览器上下文：",
      `- 日期/时间：${new Date().toLocaleString()}`,
      `- 标签页 URL：${tabUrl || "未知"}`,
      `- 标签页标题：${tabTitle || "未知"}`,
    ].join("\n")
    : [
      "You are a helpful browser assistant that can help with tasks in the browser.",
      "You have access to tools that can interact with the browser.",
      "Keep answers concise and actionable.",
      "If asked to perform actions, use the appropriate tools.",
      "Please respond in English.",
      "",
      "Current browser context:",
      `- Date/Time: ${new Date().toLocaleString()}`,
      `- Tab URL: ${tabUrl || "unknown"}`,
      `- Tab Title: ${tabTitle || "unknown"}`,
    ].join("\n")

  return systemPrompt;
}

/**
 * Load persisted session ID from chrome.storage.local
 */
async function loadSessionId(): Promise<string | null> {
  return new Promise((resolve) => {
    if (!chrome?.storage?.local) {
      resolve(null);
      return;
    }

    chrome.storage.local.get(["sessionId"], (items: { sessionId?: string }) => {
      if (chrome.runtime.lastError) {
        console.error("Failed to load session ID:", chrome.runtime.lastError);
        resolve(null);
        return;
      }

      resolve(items.sessionId || null);
    });
  });
}

/**
 * Save session ID to chrome.storage.local
 */
async function saveSessionId(sessionId: string): Promise<void> {
  return new Promise((resolve) => {
    if (!chrome?.storage?.local) {
      resolve();
      return;
    }

    chrome.storage.local.set({ sessionId }, () => {
      if (chrome.runtime.lastError) {
        console.error("Failed to save session ID:", chrome.runtime.lastError);
      } else {
        console.log("💾 Saved session ID:", sessionId);
      }
      resolve();
    });
  });
}

/**
 * Generate or get session ID for this conversation
 * Persists session ID across reloads to maintain backend context
 */
let cachedSessionId: string | null = null;

async function getSessionId(): Promise<string> {
  // Return cached session ID if available
  if (cachedSessionId) {
    return cachedSessionId;
  }

  // Try to load from storage
  const storedSessionId = await loadSessionId();
  if (storedSessionId) {
    cachedSessionId = storedSessionId;
    console.log("📦 Loaded existing session ID:", storedSessionId);
    return storedSessionId;
  }

  // Generate new session ID
  const newSessionId = `session-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  cachedSessionId = newSessionId;

  // Save to storage
  await saveSessionId(newSessionId);
  console.log("🆕 Generated new session ID:", newSessionId);

  return newSessionId;
}

/**
 * Convert the last user message to backend format
 * Backend expects: { role: "user", content: [{type: "text", text: "..."}, {type: "image", data: "data:image/jpeg;base64,..."}] }
 */
function convertLastMessage(
  messages: readonly ThreadMessage[],
  imageDataUrls: string[]
): {
  role: string;
  content: Array<{ type: string; text?: string; data?: string }>;
} | null {
  // Get the last user message
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      const msg = messages[i];
      const content: Array<{ type: string; text?: string; data?: string }> = [];
      
      // Extract text parts
      const textParts = msg.content
        .filter((part): part is { type: "text"; text: string } => part.type === "text")
        .map((part) => part.text)
        .join("\n");
      
      if (textParts) {
        content.push({ type: "text", text: textParts });
      }
      
      // Add images to content array (already in data URL format)
      for (const dataUrl of imageDataUrls) {
        content.push({
          type: "image",
          data: dataUrl,
        });
      }
      
      if (content.length > 0) {
        return {
          role: "user",
          content: content,
        };
      }
    }
  }
  
  return null;
}

/**
 * Custom runtime hook for assistant-ui
 * Streams from your Python backend via SSE
 * Sends browser context (screenshot, URL, system prompt) to backend
 */
export function useCustomRuntime(initialMessages: ThreadMessage[] = []) {
  const adapter: ChatModelAdapter = {
    async *run({ messages, abortSignal }) {
      let backendUrl: string;

      try {
        backendUrl = await getBackendUrl();
      } catch (error) {
        yield {
          content: [{ type: "text", text: String(error) } as ThreadAssistantMessagePart],
          status: { type: "incomplete", reason: "error", error: String(error) } as MessageStatus,
        };
        return;
      }

      try {
        // Get language preference
        const language = await getLanguagePreference();

        // Build system prompt with browser context
        const systemPrompt = await buildSystemPrompt();

        // Capture screenshot of current tab
        const screenshot = await captureScreenshot();

        // Get manually attached images
        const { images: manualImages, clearImages } = useImageAttachments.getState();

        // Combine screenshot and manual images as data URLs
        const imageDataUrls: string[] = [];
        if (screenshot) {
          imageDataUrls.push(screenshot);
        }
        for (const img of manualImages) {
          // Convert manual images to data URL format
          const dataUrl = `data:${img.mimeType};base64,${img.data}`;
          imageDataUrls.push(dataUrl);
        }

        // Clear manual images after including them
        clearImages();

        console.log("=== Backend Request ===");
        console.log("language:", language);
        console.log("systemPrompt:", systemPrompt);
        console.log("messages:", messages.length);
        console.log("manualImages:", manualImages.length);
        

        // Get session ID (persisted across reloads to maintain backend context)
        const sessionId = await getSessionId();

        // Convert last user message to backend format
        const lastMessage = convertLastMessage(messages, imageDataUrls);
        if (!lastMessage) {
          throw new Error("No user message found");
        }

        // Convert all messages to history format (excluding the last user message)
        const history: Array<{
          role: string;
          content: string;
        }> = [];

        for (let i = 0; i < messages.length - 1; i++) {
          const msg = messages[i];
          if (msg.role === "user" || msg.role === "assistant") {
            // Extract text content from message parts
            const textContent = msg.content
              .filter((part): part is { type: "text"; text: string } => part.type === "text")
              .map((part) => part.text)
              .join("\n");

            if (textContent) {
              history.push({
                role: msg.role,
                content: textContent,
              });
            }
          }
        }

        // Build request payload matching backend format
        const payload: {
          session_id: string;
          language: string;
          system_prompt?: string;
          message: {
            role: string;
            content: Array<{ type: string; text?: string; data?: string }>;
          };
          history?: Array<{ role: string; content: string }>;
        } = {
          session_id: sessionId,
          language: language,
          message: lastMessage,
          history: history.length > 0 ? history : undefined,
        };

        // Always include system_prompt with every message
        payload.system_prompt = systemPrompt;

        console.log("=== Backend Request ===");
        console.log("session_id:", sessionId);
        console.log("language:", language);
        console.log("system_prompt:", systemPrompt);
        console.log("history:", history.length, "messages");
        console.log("history:", JSON.stringify(history, null, 2));
        console.log("message:", lastMessage);
        console.log("images:", imageDataUrls.length);

        const response = await fetch(`${backendUrl}/v1/chat/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
          },
          body: JSON.stringify(payload),
          signal: abortSignal,
        });

        if (!response.ok) {
          const errorText = await response.text().catch(() => "Unknown error");
          throw new Error(`Backend error (${response.status}): ${errorText}`);
        }

        if (!response.body) {
          throw new Error("Response body is null");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        
        // Ordered content parts to maintain correct sequence of text and tool calls
        const contentParts: ThreadAssistantMessagePart[] = [];
        // Map for quick lookup when updating tool calls with results
        const toolCallsMap = new Map<string, ThreadAssistantMessagePart>();
        
        // Helper: Add or append text content (maintains order)
        const addText = (text: string) => {
          if (!text) return;
          const lastPart = contentParts[contentParts.length - 1];
          if (lastPart && lastPart.type === "text") {
            // Append to existing text part
            const beforeLength = (lastPart as any).text.length;
            (lastPart as any).text += text;
            console.log(`✍️ [TEXT] Appended ${text.length} chars (total: ${(lastPart as any).text.length}, was: ${beforeLength})`);
          } else {
            // Create new text part
            contentParts.push({ type: "text", text } as ThreadAssistantMessagePart);
            console.log(`✍️ [TEXT] Created new text part with ${text.length} chars`);
          }
        };
        
        // Helper: Add a tool call (maintains order)
        const addToolCall = (id: string, toolName: string, args: any, argsText: string) => {
          const toolCallPart = {
            type: "tool-call",
            toolCallId: id,
            toolName: toolName,
            args: args,
            argsText,
          } as ThreadAssistantMessagePart;
          contentParts.push(toolCallPart);
          toolCallsMap.set(id, toolCallPart);
          return toolCallPart;
        };
        
        // Helper: Update tool call with result (in place, preserves order)
        const updateToolCallResult = (id: string, result: any) => {
          const existing = toolCallsMap.get(id);
          if (existing) {
            (existing as any).result = result;
          }
        };
        
        // Track current event type for proper SSE parsing
        let currentEvent: string | null = null;
        let receivedSessionId: string | null = null;
        let isDone = false;

        // Track if we've received streaming tokens (to avoid duplication with assistant_message)
        let hasReceivedTokens = false;

        // Buffer tokens until #PLAN# is detected to avoid showing stale/intermediate reasoning
        let tokenBuffer = "";
        let planDetected = false;
        const PLAN_MARKER = "#PLAN#";

        // Typewriter effect: throttle token yields for smoother visual effect
        let lastTokenYieldTime = 0;
        const TOKEN_YIELD_INTERVAL_MS = 30; // ~33 characters per second for typewriter effect

        // Track current executing agent tool for nested sub-tasks
        let currentExecutingAgentTool: string | null = null;

        // Get agent execution store actions
        const agentExecution = useAgentExecution.getState();

        // Mark message as incomplete at start of streaming
        // This will be cleared on successful completion
        await markMessageIncomplete();

        // Streaming diagnostics
        let chunkCount = 0;
        let yieldCount = 0;
        const startTime = Date.now();

        // Throttle yields to smooth out streaming (max once per 16ms = ~60fps)
        let lastYieldTime = 0;
        const YIELD_THROTTLE_MS = 16;

        // Helper: Yield with throttling to smooth streaming
        const throttledYield = (force: boolean = false) => {
          const now = Date.now();
          const timeSinceLastYield = now - lastYieldTime;

          // Only yield if throttle period passed or forced (e.g., done event)
          if (force || timeSinceLastYield >= YIELD_THROTTLE_MS) {
            if (contentParts.length > 0) {
              yieldCount++;
              const textLength = contentParts
                .filter(p => p.type === "text")
                .reduce((sum, p) => sum + ((p as any).text?.length || 0), 0);
              const toolCallsCount = contentParts.filter(p => p.type === "tool-call").length;
              const elapsedSinceStart = Date.now() - startTime;
              console.log(`📤 [YIELD #${yieldCount}] @${elapsedSinceStart}ms (${timeSinceLastYield}ms since last) - textLength: ${textLength}, toolCalls: ${toolCallsCount}, parts: ${contentParts.length}`);

              lastYieldTime = now;
              return {
                content: [...contentParts], // Clone to avoid mutations
                status: { type: "running" } as MessageStatus,
              };
            }
          }
          return null;
        };

        try {
          while (true) {
            // Check if request was aborted
            if (abortSignal.aborted) {
              break;
            }

            const { done, value } = await reader.read();

            if (done) break;

            chunkCount++;
            const chunkSize = value.byteLength;
            const elapsed = Date.now() - startTime;
            console.log(`🌊 [STREAM] Chunk #${chunkCount} received (${chunkSize} bytes, ${elapsed}ms elapsed)`);

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            console.log(`📦 [STREAM] Processing ${lines.length} lines from chunk #${chunkCount}`);

            for (const line of lines) {
              const trimmed = line.trim();
              
              // Skip empty lines
              if (!trimmed) {
                // Empty line indicates end of SSE event, reset currentEvent
                currentEvent = null;
                continue;
              }

              // Parse event: line
              if (trimmed.startsWith("event: ")) {
                currentEvent = trimmed.slice(7).trim();
                continue;
              }

              // Parse data: line
              if (trimmed.startsWith("data: ")) {
                const data = trimmed.slice(6);
                
                // Handle [DONE] marker
                if (data === "[DONE]") {
                  isDone = true;
                  break;
                }

                try {
                  const parsed = JSON.parse(data);

                  // Handle different event types based on SSE event: line or fallback to parsed.type
                  const eventType = currentEvent || parsed.event || parsed.type;

                  // Debug: Log all events
                  console.log("📨 SSE Event received:", { eventType, parsed });

                  // Handle session_start event
                  if (eventType === "session_start" || parsed.session_id) {
                    receivedSessionId = parsed.session_id || receivedSessionId;
                    console.log("📋 Frontend: Received session_start", { session_id: receivedSessionId, sent_session_id: sessionId });
                    // Reset agent execution state on new session
                    agentExecution.reset();
                    continue;
                  }

                  // Handle iteration_start event (main or sub agent)
                  if (eventType === "iteration_start") {
                    const { agent_id, agent_type, data } = parsed;
                    const iteration = data?.iteration || 1;
                    const maxIteration = data?.max_iteration || 10;
                    
                    if (agent_type === "main") {
                      agentExecution.startMainAgent(agent_id, iteration, maxIteration);
                      console.log("🚀 Frontend: Main agent started", { agent_id, iteration, maxIteration });
                    } else if (agent_type === "sub") {
                      console.log("🔄 Frontend: Sub-agent iteration", { agent_id, iteration, maxIteration });
                    }
                    continue;
                  }

                  // Handle tool_executing event
                  if (eventType === "tool_executing") {
                    const { agent_id, agent_type, data } = parsed;
                    const toolName = data?.tool_name || "unknown";
                    const iteration = data?.iteration || 1;
                    const taskId = `${agent_id}-${toolName}-${iteration}`;
                    
                    if (agent_type === "main") {
                      // Main agent tool - add as top-level subtask
                      agentExecution.addSubTask({
                        id: taskId,
                        toolName,
                        agentId: agent_id,
                        agentType: "main",
                        iteration,
                        status: "executing",
                      });
                      // Track current agent tool for nested sub-tasks
                      if (toolName.startsWith("agent-")) {
                        currentExecutingAgentTool = toolName;
                      }
                      console.log("🔧 Frontend: Main agent tool executing", { taskId, toolName });
                    } else if (agent_type === "sub" && currentExecutingAgentTool) {
                      // Sub-agent tool - add as nested subtask
                      agentExecution.addNestedSubTask(currentExecutingAgentTool, {
                        id: taskId,
                        toolName,
                        agentId: agent_id,
                        agentType: "sub",
                        iteration,
                        status: "executing",
                      });
                      console.log("🔧 Frontend: Sub-agent tool executing", { taskId, toolName, parent: currentExecutingAgentTool });
                    }
                    continue;
                  }

                  // Handle tool_completed event
                  if (eventType === "tool_completed") {
                    const { agent_id, agent_type, data } = parsed;
                    const toolName = data?.tool_name || "unknown";
                    const iteration = data?.iteration || 1;
                    const resultPreview = data?.result_preview;
                    const taskId = `${agent_id}-${toolName}-${iteration}`;

                    if (agent_type === "main") {
                      agentExecution.updateSubTaskStatus(taskId, "completed", resultPreview);
                      // Clear current agent tool when it completes
                      if (toolName === currentExecutingAgentTool) {
                        currentExecutingAgentTool = null;
                      }
                      console.log("✅ Frontend: Main agent tool completed", { taskId, toolName });
                    } else if (agent_type === "sub" && currentExecutingAgentTool) {
                      agentExecution.updateNestedSubTaskStatus(currentExecutingAgentTool, taskId, "completed", resultPreview);
                      console.log("✅ Frontend: Sub-agent tool completed", { taskId, toolName, parent: currentExecutingAgentTool });
                    }
                    continue;
                  }

                  // Handle tool_error event (tool execution failed)
                  if (eventType === "tool_error") {
                    const { agent_id, agent_type, data } = parsed;
                    const toolName = data?.tool_name || "unknown";
                    const iteration = data?.iteration || 1;
                    const error = data?.error || "Unknown error";
                    const taskId = `${agent_id}-${toolName}-${iteration}`;

                    if (agent_type === "main") {
                      agentExecution.updateSubTaskStatus(taskId, "failed", error);
                      // Clear current agent tool when it fails
                      if (toolName === currentExecutingAgentTool) {
                        currentExecutingAgentTool = null;
                      }
                      console.log("❌ Frontend: Main agent tool failed", { taskId, toolName, error });
                    } else if (agent_type === "sub" && currentExecutingAgentTool) {
                      agentExecution.updateNestedSubTaskStatus(currentExecutingAgentTool, taskId, "failed", error);
                      console.log("❌ Frontend: Sub-agent tool failed", { taskId, toolName, error, parent: currentExecutingAgentTool });
                    }
                    continue;
                  }

                  // Handle system_message event (informational, not displayed in UI)
                  if (eventType === "system_message" || parsed.role === "system") {
                    console.log("📝 Frontend: Received system_message", { content: parsed.content?.substring(0, 100) });
                    continue;
                  }

                  // Handle user_message event (informational, not displayed in UI)
                  if (eventType === "user_message" || (parsed.role === "user" && currentEvent === "user_message")) {
                    console.log("👤 Frontend: Received user_message");
                    continue;
                  }

                  // Skip historical assistant message replays (echoed back from history)
                  // These come with SSE event: assistant_message and have content but no tool_calls
                  // Actual streaming responses either have tool_calls or come via different event types
                  if (currentEvent === "assistant_message" && parsed.content && !parsed.tool_calls) {
                    console.log("📜 Frontend: Skipping historical assistant_message replay");
                    continue;
                  }

                  // Handle assistant_message event
                  if (eventType === "assistant_message" || parsed.role === "assistant") {
                    const hasToolCalls = parsed.tool_calls && Array.isArray(parsed.tool_calls) && parsed.tool_calls.length > 0;

                    // Only add text if we haven't already received streaming tokens (avoids duplication)
                    // Tokens handle both intermediate reasoning and final answer streaming
                    if (parsed.content && !hasReceivedTokens) {
                      addText(parsed.content);

                      // Extract task description for agent execution popup
                      if (agentExecution.isActive && !agentExecution.mainTaskDescription) {
                        const description = extractTaskDescription(parsed.content);
                        if (description) {
                          agentExecution.setMainTaskDescription(description);
                          console.log("📝 Frontend: Extracted task description", { description: description.substring(0, 100) });
                        }
                      }
                    } else if (parsed.content && hasReceivedTokens) {
                      console.log("📝 Frontend: Skipping text (already received via tokens)");
                    }

                    // Handle tool_calls array in assistant_message
                    if (hasToolCalls) {
                      for (const toolCall of parsed.tool_calls) {
                        const id = toolCall.id || toolCall.tool_call_id || crypto.randomUUID();
                        const toolName = toolCall.function?.name || toolCall.function?.name || toolCall.name || "unknown";
                        const args = typeof toolCall.function?.arguments === "string"
                          ? JSON.parse(toolCall.function.arguments)
                          : toolCall.function?.arguments || toolCall.args || {};
                        const argsText = JSON.stringify(args, null, 2);

                        const toolCallPart = addToolCall(id, toolName, args, argsText);
                        console.log("🔧 Frontend: Added tool_call from assistant_message", toolCallPart);
                      }
                    }
                  }

                  // Handle tool_message event
                  if (eventType === "tool_message" || parsed.role === "tool") {
                    const id = parsed.tool_call_id || crypto.randomUUID();
                    const result = parsed.content !== undefined ? parsed.content : parsed.output || parsed.result;
                    
                    // Update existing tool call with result (preserves order)
                    if (toolCallsMap.has(id)) {
                      updateToolCallResult(id, result);
                      console.log("✅ Frontend: Updated tool_call result from tool_message", { id, result: typeof result === 'string' ? result.substring(0, 100) : result });
                    } else {
                      // Tool call not found, create it (shouldn't happen normally)
                      const toolName = parsed.name || "unknown";
                      const toolCallPart = addToolCall(id, toolName, {}, "{}");
                      (toolCallPart as any).result = result;
                      console.log("⚠️ Frontend: Created tool_call from tool_message (no prior tool_call)", toolCallPart);
                    }
                  }

                  // Handle assistant_final event
                  if (eventType === "assistant_final" || parsed.result_type) {
                    // If we still have buffered content (no #PLAN# was detected), flush it now
                    if (tokenBuffer && !planDetected) {
                      addText(tokenBuffer);
                      tokenBuffer = "";
                      planDetected = true;
                      console.log("📋 Frontend: Flushing buffered content (no #PLAN# detected)");
                    }

                    // Append final reply without discarding earlier plan text
                    if (parsed.reply && parsed.reply.trim()) {
                      contentParts.push({ type: "text", text: parsed.reply } as ThreadAssistantMessagePart);

                      const toolCallsCount = contentParts.filter(part => part.type === "tool-call").length;
                      const existingTextCount = contentParts.filter(part => part.type === "text").length;

                      console.log("🎯 Frontend: Received assistant_final with reply", {
                        result_type: parsed.result_type,
                        replyLength: parsed.reply.length,
                        toolCallsCount,
                        existingTextCount
                      });
                    } else {
                      // If no reply, keep all existing content (tool calls and text)
                      console.log("🎯 Frontend: Received assistant_final without reply, keeping existing content", {
                        result_type: parsed.result_type,
                        contentPartsCount: contentParts.length
                      });
                    }
                  }

                  // Handle error event (e.g. max iterations reached)
                  if (eventType === "error") {
                    const errorMsg = parsed.data?.error || parsed.error || "An error occurred";
                    console.log("❌ Frontend: Received error event:", errorMsg);
                    contentParts.push({ type: "text", text: `\n\n#WARNING#${errorMsg}` } as ThreadAssistantMessagePart);
                  }

                  // Handle done event
                  if (eventType === "done") {
                    isDone = true;
                    // Don't reset agent execution state here - it will persist across iterations
                    // State will be reset on next session_start or when user dismisses
                    console.log("✅ Frontend: Received done event");
                    break;
                  }

                  // Handle token streaming events (ChatGPT-style)
                  if (eventType === "token" || parsed.type === "token") {
                    const tokenContent = parsed.content || parsed.data?.content || "";
                    if (tokenContent) {
                      hasReceivedTokens = true;

                      // Buffer tokens until #PLAN# is detected
                      if (!planDetected) {
                        tokenBuffer += tokenContent;
                        // Check if #PLAN# marker is now in the buffer
                        const planIndex = tokenBuffer.indexOf(PLAN_MARKER);
                        if (planIndex !== -1) {
                          planDetected = true;
                          // Add text starting from #PLAN# marker
                          const textFromPlan = tokenBuffer.substring(planIndex);
                          addText(textFromPlan);
                          tokenBuffer = ""; // Clear buffer
                          console.log("📋 Frontend: #PLAN# detected, starting display");
                          // Yield immediately for first display
                          lastTokenYieldTime = Date.now();
                          yieldCount++;
                          yield {
                            content: [...contentParts],
                            status: { type: "running" } as MessageStatus,
                          };
                        }
                        // Don't yield while buffering (waiting for #PLAN#)
                      } else {
                        // Plan already detected, stream with typewriter effect
                        addText(tokenContent);

                        // Throttle yields for typewriter effect
                        const now = Date.now();
                        if (now - lastTokenYieldTime >= TOKEN_YIELD_INTERVAL_MS) {
                          lastTokenYieldTime = now;
                          yieldCount++;
                          yield {
                            content: [...contentParts],
                            status: { type: "running" } as MessageStatus,
                          };
                        }
                      }
                    }
                    continue;
                  }

                  // Handle final_answer event - clear intermediate text and show only final answer
                  if (eventType === "final_answer") {
                    const finalContent = parsed.content || parsed.data?.content || "";
                    if (finalContent) {
                      // Clear any buffered content since final answer replaces everything
                      tokenBuffer = "";
                      planDetected = true; // Mark as detected to prevent further buffering

                      // Remove all text parts (keep tool_calls)
                      for (let i = contentParts.length - 1; i >= 0; i--) {
                        if (contentParts[i].type === "text") {
                          contentParts.splice(i, 1);
                        }
                      }
                      // Add final answer as the only text
                      contentParts.push({ type: "text", text: finalContent } as ThreadAssistantMessagePart);
                      hasReceivedTokens = true; // Prevent duplication from assistant_message
                      console.log("🎯 Frontend: Replaced intermediate text with final answer");
                    }
                    continue;
                  }

                  // Handle tool_start (old format)
                  if (parsed.type === "tool_start" && parsed.tool) {
                    console.log("🔧 Frontend: Received tool_start (old format)", parsed);
                    const id = parsed.tool_call_id || parsed.tool || crypto.randomUUID();
                    const argsText = parsed.args ? JSON.stringify(parsed.args, null, 2) : "{}";
                    const toolCallPart = addToolCall(id, parsed.tool, parsed.args || {}, argsText);
                    console.log("🔧 Frontend: Added to contentParts", toolCallPart);
                  }

                  // Handle tool_result (old format)
                  if (parsed.type === "tool_result" && parsed.tool) {
                    console.log("✅ Frontend: Received tool_result (old format)", parsed);
                    const id = parsed.tool_call_id || parsed.tool || crypto.randomUUID();
                    const result = parsed.output !== undefined ? parsed.output : parsed.result;
                    
                    if (toolCallsMap.has(id)) {
                      updateToolCallResult(id, result);
                      console.log("✅ Frontend: Updated tool result in place", { id });
                    } else {
                      // Tool call not found, create it
                      const argsText = parsed.args ? JSON.stringify(parsed.args, null, 2) : "{}";
                      const toolCallPart = addToolCall(id, parsed.tool, parsed.args || {}, argsText);
                      (toolCallPart as any).result = result;
                      console.log("⚠️ Frontend: Created tool_call from tool_result", toolCallPart);
                    }
                  }

                  // Yield current content parts with throttling to smooth streaming
                  const yieldResult = throttledYield();
                  if (yieldResult) {
                    yield yieldResult;
                  }
                } catch (parseError) {
                  // Skip invalid JSON - continue processing other lines
                  console.warn("Failed to parse SSE data:", data, parseError);
                }
              }
            }

            // Break if done signal received
            if (isDone) {
              break;
            }
          }

          // Force final yield to ensure any throttled content is displayed
          const finalYieldResult = throttledYield(true);
          if (finalYieldResult) {
            yield finalYieldResult;
          }
        } finally {
          // Ensure reader is released even if loop breaks early
          reader.releaseLock();

          // Print streaming summary
          const totalTime = Date.now() - startTime;
          console.log(`
╔════════════════════════════════════════════════════════════╗
║               STREAMING SESSION SUMMARY                    ║
╠════════════════════════════════════════════════════════════╣
║ Total Chunks Received:  ${chunkCount.toString().padEnd(32)} ║
║ Total Yields to UI:     ${yieldCount.toString().padEnd(32)} ║
║ Total Duration:         ${totalTime}ms${' '.repeat(32 - (totalTime.toString().length + 2))}║
║ Avg Chunk Interval:     ${chunkCount > 0 ? Math.round(totalTime / chunkCount) : 0}ms${' '.repeat(32 - (chunkCount > 0 ? Math.round(totalTime / chunkCount).toString().length + 2 : 3))}║
║ Avg Yield Interval:     ${yieldCount > 0 ? Math.round(totalTime / yieldCount) : 0}ms${' '.repeat(32 - (yieldCount > 0 ? Math.round(totalTime / yieldCount).toString().length + 2 : 3))}║
╚════════════════════════════════════════════════════════════╝
          `);
        }

        // Only yield final status if not aborted
        if (!abortSignal.aborted) {
          const textLength = contentParts
            .filter(p => p.type === "text")
            .reduce((sum, p) => sum + ((p as any).text?.length || 0), 0);
          const toolCallsCount = contentParts.filter(p => p.type === "tool-call").length;

          console.log("🏁 Frontend: Final yield", {
            textLength,
            toolCallsCount,
            partsCount: contentParts.length,
            firstTextPreview: contentParts
              .filter(p => p.type === "text")
              .map(p => (p as any).text?.substring(0, 100))
              .join(", ")
          });

          // Clear incomplete flag on successful completion
          await clearIncompleteFlag();

          yield {
            content: [...contentParts], // Clone to avoid mutations
            status: { type: "complete", reason: "stop" } as MessageStatus,
          };
        }
      } catch (error) {
        // Don't yield error if request was aborted (user cancelled)
        if (error instanceof Error && error.name === "AbortError") {
          console.log("🛑 Request aborted by user - cancelling all executing tasks");
          useAgentExecution.getState().cancelAllExecuting();
          return;
        }

        const errorMessage = error instanceof Error ? error.message : String(error);
        yield {
          content: [{ type: "text", text: `Error: ${errorMessage}` } as ThreadAssistantMessagePart],
          status: { type: "incomplete", reason: "error", error: errorMessage } as MessageStatus,
        };
      }
    },
  };

  // Create runtime with provided initial messages
  console.log("🔄 Creating runtime with", initialMessages.length, "initial messages");

  const runtime = useLocalRuntime(adapter, {
    initialMessages,
  });

  // Save messages whenever they change
  useEffect(() => {
    let saveTimeoutId: NodeJS.Timeout | null = null;

    const unsubscribe = runtime.thread.subscribe(() => {
      const messages = runtime.thread.getState().messages;
      if (messages && messages.length > 0) {
        if (saveTimeoutId) clearTimeout(saveTimeoutId);

        saveTimeoutId = setTimeout(() => {
          saveMessages([...messages]);
          console.log("💾 Saved", messages.length, "messages to storage");
        }, 500);
      }
    });

    return () => {
      if (saveTimeoutId) clearTimeout(saveTimeoutId);
      unsubscribe();
    };
  }, [runtime]);

  return runtime;
}
