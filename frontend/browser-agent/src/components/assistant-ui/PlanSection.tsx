import { ListIcon, ChevronDownIcon, ChevronUpIcon } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";

interface PlanSectionProps {
  content: string;
}

/**
 * Strips the #PLAN# section from content so it doesn't appear twice
 * (once in PlanSection and once in the markdown)
 */
export function stripPlanContent(content: string): string {
  if (!content) return content;

  // Match PLAN marker and everything up to the end of numbered items
  // Uses [\s\S] to match across newlines
  // More permissive patterns to handle various formats
  let result = content
    // Ultra permissive catch-all: any line that is just "PLAN" with symbols/spaces around it
    .replace(/^[\s#*]*PLAN[\s#*]*$/gim, '')
    // Pattern 1: #+PLAN#+ with any spaces (e.g., #  PLAN  #, # PLAN #, #PLAN#)
    .replace(/^#+\s*PLAN\s*#*\s*$/gim, '')
    // Pattern 2: **PLAN** or *PLAN*
    .replace(/^\*{1,2}\s*PLAN\s*\*{1,2}\s*$/gim, '')
    // Pattern 3: Just PLAN on its own line
    .replace(/^PLAN\s*$/gim, '');

  // Now remove consecutive numbered items that follow (the actual plan steps)
  // This is a bit tricky - we need to remove lines like "1. Do something" that come after the PLAN marker
  const lines = result.split('\n');
  const filteredLines: string[] = [];
  let inPlanSection = false;
  let lastWasPlanMarker = false;

  for (const line of lines) {
    const trimmed = line.trim();

    // Check if this line was where a PLAN marker used to be (now empty after replacement)
    if (trimmed === '' && lastWasPlanMarker) {
      // Skip empty lines right after plan marker removal
      continue;
    }

    // If we're seeing numbered items at the start of content or after empty lines
    if (/^\d+\.\s/.test(trimmed)) {
      // Check if this looks like a plan item (starts with action verb)
      const isPlanItem = /^\d+\.\s+(Use|Verify|Present|Check|Analyze|Create|Update|Find|Search|Navigate|Click|Open|Close|Submit|Wait|Extract|Download|Upload|Review|Validate|Test|Debug|Fix|Implement|Add|Remove|Delete|Modify|Configure|Setup|Install|Run|Execute|Call|Send|Get|Post|Put|Fetch|First|Then|Next|Finally|Read|Write|Examine|Explore|Locate|Identify)/i.test(trimmed);

      if (isPlanItem && (inPlanSection || filteredLines.length === 0 || filteredLines.every(l => l.trim() === ''))) {
        inPlanSection = true;
        continue; // Skip this plan item
      }
    }

    // Non-numbered, non-empty line ends the plan section
    if (trimmed !== '' && !/^\d+\./.test(trimmed)) {
      inPlanSection = false;
    }

    lastWasPlanMarker = false;
    filteredLines.push(line);
  }

  return filteredLines.join('\n').trim();
}

/**
 * Extracts and displays the #PLAN# section from assistant messages
 */
// Store the last successfully extracted plan so it persists after execution
let lastExtractedPlan: string[] | null = null;

export function PlanSection({ content }: PlanSectionProps) {
  const [isMinimized, setIsMinimized] = useState(false);

  // Debug: log what content we receive (with escaped newlines for visibility)
  console.log("🎯 PlanSection content:", content ? JSON.stringify(content.substring(0, 300)) : "null/empty");
  
  // Also log the first few lines to see exact format
  const firstLines = content.split('\n').slice(0, 5);
  console.log("🎯 First 5 lines:", firstLines.map((l, i) => `[${i}]: "${l}"`));

  if (!content) {
    // If no content but we have a cached plan, show it
    if (lastExtractedPlan && lastExtractedPlan.length > 0) {
      console.log("📋 Using cached plan:", lastExtractedPlan);
      return renderPlan(lastExtractedPlan, isMinimized, setIsMinimized);
    }
    return null;
  }

  // SUPER permissive patterns - match PLAN with any combination of #, *, spaces
  // The key insight: we just need to find a line that contains "PLAN" surrounded by
  // optional markers (#, *, spaces)
  const patterns = [
    // ULTRA permissive catch-all: any line that is just "PLAN" with symbols/spaces around it
    /^[\s#*]*PLAN[\s#*]*$/im,
    // More specific patterns as fallbacks
    /^#+\s*PLAN\s*#*\s*$/im,               // # PLAN, #PLAN#, # PLAN #, #  PLAN  #, etc.
    /^[#]+[ \t]*PLAN[ \t]*[#]*[ \t]*$/im,  // Same but explicit space/tab
    /[#]+\s*PLAN\s*[#]+/i,                  // #PLAN# anywhere (not just start of line)
    /^\*{1,2}\s*PLAN\s*\*{1,2}\s*$/im,     // **PLAN**, *PLAN*, ** PLAN **, etc.
    /^PLAN\s*$/im,                          // Just PLAN on its own line
  ];

  let markerMatch: RegExpMatchArray | null = null;
  let matchedPatternIndex = -1;
  for (let i = 0; i < patterns.length; i++) {
    markerMatch = content.match(patterns[i]);
    if (markerMatch) {
      matchedPatternIndex = i;
      console.log(`✅ Pattern ${i} matched:`, markerMatch[0], "at index", content.indexOf(markerMatch[0]));
      break;
    } else {
      console.log(`❌ Pattern ${i} did not match`);
    }
  }

  if (!markerMatch) {
    console.log("⚠️ No PLAN marker found in content");
    // No marker found, but if we have a cached plan, show it
    if (lastExtractedPlan && lastExtractedPlan.length > 0) {
      console.log("📋 Using cached plan (no marker):", lastExtractedPlan);
      return renderPlan(lastExtractedPlan, isMinimized, setIsMinimized);
    }
    return null;
  }

  // Get content after the marker
  const markerIndex = content.indexOf(markerMatch[0]);
  const afterMarker = content.substring(markerIndex + markerMatch[0].length);
  console.log("📝 Content after marker:", afterMarker.substring(0, 300));

  // Extract all numbered items from the content after the marker
  const planItems: string[] = [];
  const lines = afterMarker.split('\n');
  console.log("📋 Lines after marker:", lines.slice(0, 10));

  for (const line of lines) {
    const trimmed = line.trim();
    if (/^\d+\./.test(trimmed)) {
      planItems.push(trimmed);
      console.log("  ✓ Found plan item:", trimmed.substring(0, 50));
    } else if (planItems.length > 0 && trimmed === '') {
      continue; // Allow empty lines between items
    } else if (planItems.length > 0 && trimmed !== '') {
      console.log("  ⏹ Stopping at non-numbered line:", trimmed.substring(0, 50));
      break; // Non-numbered, non-empty line = end of plan
    }
  }

  console.log("📊 Extracted plan items:", planItems.length);

  if (planItems.length === 0) {
    console.log("⚠️ No plan items found after marker");
    if (lastExtractedPlan && lastExtractedPlan.length > 0) {
      console.log("📋 Using cached plan (no items):", lastExtractedPlan);
      return renderPlan(lastExtractedPlan, isMinimized, setIsMinimized);
    }
    return null;
  }

  // Cache the plan for persistence
  lastExtractedPlan = planItems;
  console.log("💾 Cached plan for persistence:", planItems.length, "items");

  return renderPlan(planItems, isMinimized, setIsMinimized);
}

function renderPlan(
  planItems: string[],
  isMinimized: boolean,
  setIsMinimized: (value: boolean) => void
) {

  return (
    <div className="mb-4 rounded-lg border border-purple-200 bg-purple-50/50 dark:border-purple-800 dark:bg-purple-950/30 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2.5 bg-purple-100/50 dark:bg-purple-900/30 border-b border-purple-200 dark:border-purple-800">
        <ListIcon className="size-4 text-purple-600 dark:text-purple-400" />
        <span className="font-medium text-sm flex-grow">Plan</span>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0 hover:bg-purple-200 dark:hover:bg-purple-800"
          onClick={() => setIsMinimized(!isMinimized)}
          title={isMinimized ? "Expand" : "Minimize"}
        >
          {isMinimized ? (
            <ChevronDownIcon className="size-3.5" />
          ) : (
            <ChevronUpIcon className="size-3.5" />
          )}
        </Button>
      </div>

      {/* Content - only show when not minimized */}
      {!isMinimized && (
        <div className="px-4 py-3">
          <ol className="space-y-2 list-none">
            {planItems.map((item, index) => {
              // Remove the number prefix for display
              const text = item.replace(/^\d+\.\s*/, '');
              return (
                <li key={index} className="text-sm flex gap-2">
                  <span className="text-purple-600 dark:text-purple-400 font-medium shrink-0">
                    {index + 1}.
                  </span>
                  <span className="text-muted-foreground">{text}</span>
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </div>
  );
}
