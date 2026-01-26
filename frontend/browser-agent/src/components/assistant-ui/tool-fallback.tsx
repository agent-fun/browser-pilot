import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { CheckIcon, ChevronDownIcon, ChevronUpIcon, XIcon, Loader2Icon, MinusCircleIcon } from "lucide-react";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { useAgentExecution, type SubTask } from "@/lib/useAgentExecution";
import { useShallow } from "zustand/shallow";

/**
 * Renders a single subtask item with optional nested sub-tasks
 */
function SubTaskItem({ task, depth = 0 }: { task: SubTask; depth?: number }) {
  const [isExpanded, setIsExpanded] = useState(true); // Default expanded for better visibility
  const hasSubTasks = task.subTasks && task.subTasks.length > 0;
  const isExecuting = task.status === "executing";
  const isFailed = task.status === "failed";
  const isCancelled = task.status === "cancelled";
  const isAgentTool = task.toolName.startsWith("agent-");

  // Format display name for agent tools
  const displayName = isAgentTool
    ? hasSubTasks || task.resultPreview
      ? `${task.toolName}: ${hasSubTasks
          ? (task.subTasks!.length === 1
              ? task.subTasks![0].toolName
              : `called ${task.subTasks!.length} tools`)
          : 'completed'}`
      : task.toolName
    : task.toolName;

  return (
    <div className={`${depth > 0 ? "ml-4 border-l-2 border-blue-200 dark:border-blue-800 pl-3 mt-1" : ""}`}>
      <div
        className="flex items-center gap-2 py-1.5 cursor-pointer hover:bg-muted/50 rounded px-2 -mx-2"
        onClick={() => hasSubTasks && setIsExpanded(!isExpanded)}
      >
        {/* Status indicator */}
        {isExecuting ? (
          <Loader2Icon className="size-3.5 animate-spin text-blue-500 shrink-0" />
        ) : isFailed ? (
          <XIcon className="size-3.5 text-red-500 shrink-0" />
        ) : isCancelled ? (
          <MinusCircleIcon className="size-3.5 text-gray-500 shrink-0" />
        ) : (
          <CheckIcon className="size-3.5 text-green-500 shrink-0" />
        )}

        {/* Tool name */}
        <span className="text-sm font-medium flex-grow truncate">
          {displayName}
        </span>

        {/* Expand/collapse for nested tasks */}
        {hasSubTasks && (
          <Button
            variant="ghost"
            size="sm"
            className="h-5 w-5 p-0"
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
          >
            {isExpanded ? (
              <ChevronUpIcon className="size-3" />
            ) : (
              <ChevronDownIcon className="size-3" />
            )}
          </Button>
        )}
      </div>

      {/* Expanded content: nested sub-tasks */}
      {isExpanded && hasSubTasks && (
        <div className="mt-0.5 space-y-0.5">
          {task.subTasks!.map((subTask) => (
            <SubTaskItem key={subTask.id} task={subTask} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Agent Execution Popup - shows when agent is actively running
 */
export function AgentExecutionPopup() {
  const [language, setLanguage] = useState("en");

  // Load language preference
  useEffect(() => {
    if (chrome?.storage?.sync) {
      chrome.storage.sync.get(["language"], (items: { language?: string }) => {
        if (!chrome.runtime.lastError && items.language) {
          setLanguage(items.language);
        }
      });

      // Listen for language changes
      const handleStorageChange = (changes: { [key: string]: chrome.storage.StorageChange }, areaName: string) => {
        if (areaName === "sync" && changes.language?.newValue) {
          setLanguage(changes.language.newValue as string);
        }
      };
      chrome.storage.onChanged.addListener(handleStorageChange);

      return () => {
        chrome.storage.onChanged.removeListener(handleStorageChange);
      };
    }
  }, []);

  const { isActive, isDismissed, isMinimized, mainAgentId, mainTaskDescription, currentIteration, subtasks, toggleMinimize, dismiss } =
    useAgentExecution(
      useShallow((state) => ({
        isActive: state.isActive,
        isDismissed: state.isDismissed,
        isMinimized: state.isMinimized,
        mainAgentId: state.mainAgentId,
        mainTaskDescription: state.mainTaskDescription,
        currentIteration: state.currentIteration,
        subtasks: state.subtasks,
        toggleMinimize: state.toggleMinimize,
        dismiss: state.dismiss,
      }))
    );

  // Debug: log the state
  console.log("🔍 AgentExecutionPopup render:", {
    isActive,
    isDismissed,
    mainAgentId,
    hasDescription: !!mainTaskDescription,
    subtasksCount: subtasks.length
  });

  // Don't show if not active or dismissed
  if (!isActive || isDismissed) {
    console.log("❌ AgentExecutionPopup hidden:", { isActive, isDismissed });
    return null;
  }

  console.log("✅ AgentExecutionPopup showing!");

  const hasExecutingTasks = subtasks.some((t) => t.status === "executing");
  const completedCount = subtasks.filter((t) => t.status === "completed").length;
  const failedCount = subtasks.filter((t) => t.status === "failed").length;
  const cancelledCount = subtasks.filter((t) => t.status === "cancelled").length;

  return (
    <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50/50 dark:border-blue-800 dark:bg-blue-950/30 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2.5 bg-blue-100/50 dark:bg-blue-900/30 border-b border-blue-200 dark:border-blue-800">
        {hasExecutingTasks ? (
          <Loader2Icon className="size-4 animate-spin text-blue-600 dark:text-blue-400" />
        ) : (
          <CheckIcon className="size-4 text-green-600 dark:text-green-400" />
        )}
        <span className="font-medium text-sm flex-grow">
          {language === "zh" ? "Agent执行中" : "Agent Executing"}
          {currentIteration > 0 && (
            <span className="text-muted-foreground ml-1.5 font-normal">
              ({language === "zh" ? "步骤" : "iteration"} {currentIteration})
            </span>
          )}
        </span>
        {subtasks.length > 0 && (
          <span className="text-xs text-muted-foreground">
            {completedCount}/{subtasks.length} {language === "zh" ? "任务" : "tasks"}
            {failedCount > 0 && (
              <span className="text-red-500 ml-1">({failedCount} {language === "zh" ? "失败" : "failed"})</span>
            )}
            {cancelledCount > 0 && (
              <span className="text-gray-500 ml-1">({cancelledCount} {language === "zh" ? "已取消" : "cancelled"})</span>
            )}
          </span>
        )}
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0 hover:bg-blue-200 dark:hover:bg-blue-800"
          onClick={toggleMinimize}
          title={isMinimized
            ? (language === "zh" ? "展开" : "Expand")
            : (language === "zh" ? "最小化" : "Minimize")
          }
        >
          {isMinimized ? (
            <ChevronDownIcon className="size-3.5" />
          ) : (
            <ChevronUpIcon className="size-3.5" />
          )}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0 hover:bg-blue-200 dark:hover:bg-blue-800"
          onClick={dismiss}
          title={language === "zh" ? "关闭" : "Dismiss"}
        >
          <XIcon className="size-3.5" />
        </Button>
      </div>

      {/* Content - only show when not minimized */}
      {!isMinimized && (
        <div className="px-4 py-3 max-h-64 overflow-y-auto">
          {/* Main task description */}
          {mainTaskDescription && (
            <div className="mb-3 text-sm text-muted-foreground">
              <pre className="whitespace-pre-wrap font-sans">{mainTaskDescription}</pre>
            </div>
          )}

          {/* Agent ID fallback if no description */}
          {!mainTaskDescription && mainAgentId && (
            <div className="mb-3 text-sm text-muted-foreground">
              {language === "zh" ? "正在运行：" : "Running:"} <span className="font-medium">{mainAgentId}</span>
            </div>
          )}

          {/* Subtasks list */}
          {subtasks.length > 0 && (
            <div className="space-y-0.5 border-t border-blue-200 dark:border-blue-800 pt-3 mt-2">
              <div className="text-xs font-medium text-muted-foreground mb-2">{language === "zh" ? "任务：" : "Tasks:"}</div>
              {subtasks.map((task) => (
                <SubTaskItem key={task.id} task={task} />
              ))}
            </div>
          )}

          {/* Empty state */}
          {subtasks.length === 0 && !mainTaskDescription && (
            <div className="text-sm text-muted-foreground flex items-center gap-2">
              <Loader2Icon className="size-3.5 animate-spin" />
              {language === "zh" ? "初始化中..." : "Initializing..."}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Tool Fallback Component - displays individual tool calls
 */
export const ToolFallback: ToolCallMessagePartComponent = ({
  toolName,
  argsText,
  result,
  toolCallId,
}) => {
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [language, setLanguage] = useState("en");

  // Get agent execution state to sync status with AgentExecutionPopup
  const subtasks = useAgentExecution(useShallow((state) => state.subtasks));

  // Find matching subtask by toolName to get authoritative status
  // This ensures ToolFallback shows same success/failure as AgentExecutionPopup
  const matchingSubtask = subtasks.find((task) => {
    // Try to match by toolName
    if (task.toolName === toolName) return true;

    // For nested subtasks, search recursively
    if (task.subTasks) {
      return task.subTasks.some((subTask) => subTask.toolName === toolName);
    }

    return false;
  });

  // Load language preference
  useEffect(() => {
    if (chrome?.storage?.sync) {
      chrome.storage.sync.get(["language"], (items: { language?: string }) => {
        if (!chrome.runtime.lastError && items.language) {
          setLanguage(items.language);
        }
      });

      // Listen for language changes
      const handleStorageChange = (changes: { [key: string]: chrome.storage.StorageChange }, areaName: string) => {
        if (areaName === "sync" && changes.language?.newValue) {
          setLanguage(changes.language.newValue as string);
        }
      };
      chrome.storage.onChanged.addListener(handleStorageChange);

      return () => {
        chrome.storage.onChanged.removeListener(handleStorageChange);
      };
    }
  }, []);

  // Determine error status: prefer subtask status if available, fallback to result inspection
  const isError = (() => {
    // Priority 1: Use subtask status from backend events (authoritative source)
    if (matchingSubtask) {
      return matchingSubtask.status === "failed" || matchingSubtask.status === "cancelled";
    }

    // Priority 2: Fallback to inspecting result content (backward compatibility)
    if (!result) return false;

    // Check if result is a string containing error
    if (typeof result === "string") {
      // Try to parse string as JSON/object
      try {
        const parsed = JSON.parse(result);
        if (typeof parsed === "object" && parsed !== null && "error" in parsed) {
          return true;
        }
      } catch {
        // Not valid JSON, check if string contains 'error'
        if (result.toLowerCase().includes("error")) {
          return true;
        }
      }
    }

    // Check if result is an object with error key
    if (typeof result === "object" && result !== null && "error" in result) {
      return true;
    }

    return false;
  })();

  // Hide tool calls when toolName is 'unknown'
  if (toolName === "unknown") {
    return null;
  }

  // Display "agent called" for agent tools (e.g., agent-browsing, agent-coding)
  const displayName = toolName.startsWith("agent-") ? "agent called" : toolName;
  const label = toolName.startsWith("agent-")
    ? language === "zh"
      ? `调用代理：<b>${toolName.replace("agent-", "")}</b>`
      : `Agent called: <b>${toolName.replace("agent-", "")}</b>`
    : language === "zh"
      ? `使用工具：<b>${toolName}</b>`
      : `Used tool: <b>${toolName}</b>`;

  return (
    <div className={`aui-tool-fallback-root mb-4 flex w-full flex-col gap-3 rounded-lg border py-3 ${isError ? "border-red-300 dark:border-red-800" : ""}`}>
      <div className="aui-tool-fallback-header flex items-center gap-2 px-4">
        {isError ? (
          <XIcon className="aui-tool-fallback-icon size-4 text-red-500" />
        ) : (
          <CheckIcon className="aui-tool-fallback-icon size-4 text-green-500" />
        )}
        <p className="aui-tool-fallback-title flex-grow" dangerouslySetInnerHTML={{ __html: label }} />
        <Button onClick={() => setIsCollapsed(!isCollapsed)}>
          {isCollapsed ? <ChevronDownIcon /> : <ChevronUpIcon />}
        </Button>
      </div>
      {!isCollapsed && (
        <div className="aui-tool-fallback-content flex flex-col gap-2 border-t pt-2">
          <div className="aui-tool-fallback-args-root px-4">
            <pre className="aui-tool-fallback-args-value whitespace-pre-wrap">
              {argsText}
            </pre>
          </div>
          {result !== undefined && (
            <div className="aui-tool-fallback-result-root border-t border-dashed px-4 pt-2">
              <p className="aui-tool-fallback-result-header font-semibold">
                {language === "zh" ? "结果：" : "Result:"}
              </p>
              <pre className="aui-tool-fallback-result-content whitespace-pre-wrap">
                {typeof result === "string"
                  ? result
                  : JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
