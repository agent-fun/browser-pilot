"use client";

import "@assistant-ui/react-markdown/styles/dot.css";

import {
  type CodeHeaderProps,
  MarkdownTextPrimitive,
  unstable_memoizeMarkdownComponents as memoizeMarkdownComponents,
  useIsMarkdownCodeBlock,
} from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import { type FC, memo, useState, Children } from "react";
import { CheckIcon, CopyIcon } from "lucide-react";
import type { Root, Paragraph, List, Text, Heading } from "mdast";

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { cn } from "@/lib/utils";

/**
 * Check if text content matches a PLAN marker pattern
 * Handles: #PLAN#, # PLAN #, #  PLAN  #, PLAN #, #PLAN, PLAN, **PLAN**, etc.
 * Also handles when markdown parser strips leading # (e.g., "PLAN  #" from "#  PLAN  #")
 */
function isPlanMarker(text: string): boolean {
  const trimmed = text.trim();
  // Very flexible pattern: text that is essentially just "PLAN" with optional #, *, or spaces around it
  // This handles all variations including "PLAN  #" (from parsed heading)
  const result = /^[#*\s]*PLAN[#*\s]*$/i.test(trimmed);
  if (result) {
    console.log("🎯 isPlanMarker matched:", JSON.stringify(trimmed));
  }
  return result;
}

/**
 * Check if text looks like a plan item (numbered step)
 * e.g., "1. Use VQA tool to analyze..."
 */
function isPlanItem(text: string): boolean {
  const trimmed = text.trim();
  // Match: starts with number, dot, space, then typical plan verbs
  // Added more verbs including "First", "Then", "Next", "Read", "Write", "Examine", etc.
  const result = /^\d+\.\s+(Use|Verify|Present|Check|Analyze|Create|Update|Find|Search|Navigate|Click|Open|Close|Submit|Wait|Extract|Download|Upload|Review|Validate|Test|Debug|Fix|Implement|Add|Remove|Delete|Modify|Configure|Setup|Install|Run|Execute|Call|Send|Get|Post|Put|Fetch|First|Then|Next|Finally|Read|Write|Examine|Explore|Locate|Identify)/i.test(trimmed);
  if (result) {
    console.log("🎯 isPlanItem matched:", trimmed.substring(0, 50));
  }
  return result;
}

/**
 * Remark plugin to strip #PLAN# sections from markdown
 * This removes the #PLAN# paragraph/heading and following plan items
 */
function remarkStripPlan() {
  return (tree: Root) => {
    const indicesToRemove: number[] = [];
    let inPlanSection = false;

    console.log("🔄 remarkStripPlan processing tree with", tree.children.length, "children");

    tree.children.forEach((node, index) => {
      let textContent = "";

      // Extract text content from paragraphs
      if (node.type === "paragraph") {
        textContent = (node as Paragraph).children
          .filter((child): child is Text => child.type === "text")
          .map((child) => child.value)
          .join("");
      }

      // Extract text content from headings
      if (node.type === "heading") {
        textContent = (node as Heading).children
          .filter((child): child is Text => child.type === "text")
          .map((child) => child.value)
          .join("");
        console.log("📝 Found heading:", JSON.stringify(textContent));
      }

      // Check if this is a PLAN marker
      if (textContent && isPlanMarker(textContent)) {
        console.log("✅ Marking for removal (PLAN marker):", index);
        indicesToRemove.push(index);
        inPlanSection = true;
        return;
      }

      // If we're in a plan section, check for plan items or ordered lists
      if (inPlanSection) {
        // Check for ordered list (the plan items as a list)
        if (node.type === "list" && (node as List).ordered) {
          console.log("✅ Marking for removal (ordered list):", index);
          indicesToRemove.push(index);
          inPlanSection = false; // End of plan section
          return;
        }

        // Check for plan items as paragraphs (numbered lines)
        if (node.type === "paragraph" && /^\d+\./.test(textContent.trim())) {
          console.log("✅ Marking for removal (numbered paragraph):", index);
          indicesToRemove.push(index);
          return;
        }

        // If we hit something else, we're out of the plan section
        if (textContent && !textContent.trim().match(/^\d+\./)) {
          console.log("⏹ Exiting plan section at:", index);
          inPlanSection = false;
        }
      }
    });

    console.log("🗑️ Removing", indicesToRemove.length, "nodes");

    // Remove nodes in reverse order to maintain correct indices
    for (let i = indicesToRemove.length - 1; i >= 0; i--) {
      tree.children.splice(indicesToRemove[i], 1);
    }
  };
}

const MarkdownTextImpl = () => {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm, remarkStripPlan]}
      className="aui-md"
      components={defaultComponents}
    />
  );
};

export const MarkdownText = memo(MarkdownTextImpl);

const CodeHeader: FC<CodeHeaderProps> = ({ language, code }) => {
  const { isCopied, copyToClipboard } = useCopyToClipboard();
  const onCopy = () => {
    if (!code || isCopied) return;
    copyToClipboard(code);
  };

  return (
    <div className="aui-code-header-root mt-4 flex items-center justify-between gap-4 rounded-t-lg bg-muted-foreground/15 px-4 py-2 text-sm font-semibold text-foreground dark:bg-muted-foreground/20">
      <span className="aui-code-header-language lowercase [&>span]:text-xs">
        {language}
      </span>
      <TooltipIconButton tooltip="Copy" onClick={onCopy}>
        {!isCopied && <CopyIcon />}
        {isCopied && <CheckIcon />}
      </TooltipIconButton>
    </div>
  );
};

const useCopyToClipboard = ({
  copiedDuration = 3000,
}: {
  copiedDuration?: number;
} = {}) => {
  const [isCopied, setIsCopied] = useState<boolean>(false);

  const copyToClipboard = (value: string) => {
    if (!value) return;

    navigator.clipboard.writeText(value).then(() => {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), copiedDuration);
    });
  };

  return { isCopied, copyToClipboard };
};

/**
 * Helper to extract text content from React children
 */
function getTextContent(children: React.ReactNode): string {
  if (typeof children === 'string') return children;
  return Children.toArray(children)
    .map(child => typeof child === 'string' ? child : '')
    .join('');
}

/**
 * Check if rendered content is a PLAN marker (for component fallbacks)
 */
function isRenderedPlanMarker(children: React.ReactNode): boolean {
  const text = getTextContent(children).trim();
  // Very flexible pattern: text that is essentially just "PLAN" with optional #, *, or spaces around it
  return /^[#*\s]*PLAN[#*\s]*$/i.test(text);
}

/**
 * Check if rendered content is a plan item (numbered step that follows plan marker)
 */
function isRenderedPlanItem(children: React.ReactNode): boolean {
  const text = getTextContent(children).trim();
  // Match numbered items that look like plan steps
  return /^\d+\.\s+(Use|Verify|Present|Check|Analyze|Create|Update|Find|Search|Navigate|Click|Open|Close|Submit|Wait|Extract|Download|Upload|Review|Validate|Test|Debug|Fix|Implement|Add|Remove|Delete|Modify|Configure|Setup|Install|Run|Execute|Call|Send|Get|Post|Put|Fetch)/i.test(text);
}

const defaultComponents = memoizeMarkdownComponents({
  h1: ({ className, children, ...props }) => {
    // Hide if this is a PLAN marker (e.g., "# PLAN #" parsed as H1 with "PLAN #")
    if (isRenderedPlanMarker(children)) return null;
    return (
      <h1
        className={cn(
          "aui-md-h1 mb-8 scroll-m-20 text-4xl font-extrabold tracking-tight last:mb-0",
          className,
        )}
        {...props}
      >
        {children}
      </h1>
    );
  },
  h2: ({ className, children, ...props }) => {
    if (isRenderedPlanMarker(children)) return null;
    return (
      <h2
        className={cn(
          "aui-md-h2 mt-8 mb-4 scroll-m-20 text-3xl font-semibold tracking-tight first:mt-0 last:mb-0",
          className,
        )}
        {...props}
      >
        {children}
      </h2>
    );
  },
  h3: ({ className, children, ...props }) => {
    if (isRenderedPlanMarker(children)) return null;
    return (
      <h3
        className={cn(
          "aui-md-h3 mt-6 mb-4 scroll-m-20 text-2xl font-semibold tracking-tight first:mt-0 last:mb-0",
          className,
        )}
        {...props}
      >
        {children}
      </h3>
    );
  },
  h4: ({ className, children, ...props }) => {
    if (isRenderedPlanMarker(children)) return null;
    return (
      <h4
        className={cn(
          "aui-md-h4 mt-6 mb-4 scroll-m-20 text-xl font-semibold tracking-tight first:mt-0 last:mb-0",
          className,
        )}
        {...props}
      >
        {children}
      </h4>
    );
  },
  h5: ({ className, children, ...props }) => {
    if (isRenderedPlanMarker(children)) return null;
    return (
      <h5
        className={cn(
          "aui-md-h5 my-4 text-lg font-semibold first:mt-0 last:mb-0",
          className,
        )}
        {...props}
      >
        {children}
      </h5>
    );
  },
  h6: ({ className, children, ...props }) => {
    if (isRenderedPlanMarker(children)) return null;
    return (
      <h6
        className={cn(
          "aui-md-h6 my-4 font-semibold first:mt-0 last:mb-0",
          className,
        )}
        {...props}
      >
        {children}
      </h6>
    );
  },
  p: ({ className, children, ...props }) => {
    // Fallback: hide paragraphs that are PLAN markers or plan items
    if (isRenderedPlanMarker(children)) return null;
    if (isRenderedPlanItem(children)) return null;
    return (
      <p
        className={cn(
          "aui-md-p mt-5 mb-5 leading-7 first:mt-0 last:mb-0",
          className,
        )}
        {...props}
      >
        {children}
      </p>
    );
  },
  a: ({ className, ...props }) => (
    <a
      className={cn(
        "aui-md-a font-medium text-primary underline underline-offset-4",
        className,
      )}
      {...props}
    />
  ),
  blockquote: ({ className, ...props }) => (
    <blockquote
      className={cn("aui-md-blockquote border-l-2 pl-6 italic", className)}
      {...props}
    />
  ),
  ul: ({ className, ...props }) => (
    <ul
      className={cn("aui-md-ul my-5 ml-6 list-disc [&>li]:mt-2", className)}
      {...props}
    />
  ),
  ol: ({ className, children, ...props }) => {
    // Check if this looks like a plan list (first item starts with plan-like verb)
    const childArray = Children.toArray(children);
    if (childArray.length > 0) {
      const firstChild = childArray[0];
      if (firstChild && typeof firstChild === 'object' && 'props' in firstChild) {
        const firstItemText = getTextContent(firstChild.props.children);
        if (/^(Use|Verify|Present|Check|Analyze|Create|Update|Find|Search|Navigate|Click|Open|Close|Submit|Wait|Extract|Download|Upload|Review|Validate|Test|Debug|Fix|Implement|Add|Remove|Delete|Modify|Configure|Setup|Install|Run|Execute|Call|Send|Get|Post|Put|Fetch)/i.test(firstItemText.trim())) {
          return null; // Hide plan-like ordered lists
        }
      }
    }
    return (
      <ol
        className={cn("aui-md-ol my-5 ml-6 list-decimal [&>li]:mt-2", className)}
        {...props}
      >
        {children}
      </ol>
    );
  },
  hr: ({ className, ...props }) => (
    <hr className={cn("aui-md-hr my-5 border-b", className)} {...props} />
  ),
  table: ({ className, ...props }) => (
    <table
      className={cn(
        "aui-md-table my-5 w-full border-separate border-spacing-0 overflow-y-auto",
        className,
      )}
      {...props}
    />
  ),
  th: ({ className, ...props }) => (
    <th
      className={cn(
        "aui-md-th bg-muted px-4 py-2 text-left font-bold first:rounded-tl-lg last:rounded-tr-lg [&[align=center]]:text-center [&[align=right]]:text-right",
        className,
      )}
      {...props}
    />
  ),
  td: ({ className, ...props }) => (
    <td
      className={cn(
        "aui-md-td border-b border-l px-4 py-2 text-left last:border-r [&[align=center]]:text-center [&[align=right]]:text-right",
        className,
      )}
      {...props}
    />
  ),
  tr: ({ className, ...props }) => (
    <tr
      className={cn(
        "aui-md-tr m-0 border-b p-0 first:border-t [&:last-child>td:first-child]:rounded-bl-lg [&:last-child>td:last-child]:rounded-br-lg",
        className,
      )}
      {...props}
    />
  ),
  sup: ({ className, ...props }) => (
    <sup
      className={cn("aui-md-sup [&>a]:text-xs [&>a]:no-underline", className)}
      {...props}
    />
  ),
  pre: ({ className, ...props }) => (
    <pre
      className={cn(
        "aui-md-pre overflow-x-auto !rounded-t-none rounded-b-lg bg-black p-4 text-white",
        className,
      )}
      {...props}
    />
  ),
  code: function Code({ className, ...props }) {
    const isCodeBlock = useIsMarkdownCodeBlock();
    return (
      <code
        className={cn(
          !isCodeBlock &&
            "aui-md-inline-code rounded border bg-muted font-semibold",
          className,
        )}
        {...props}
      />
    );
  },
  CodeHeader,
});
