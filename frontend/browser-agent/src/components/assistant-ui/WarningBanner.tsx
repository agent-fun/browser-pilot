import { AlertTriangleIcon } from "lucide-react";

interface WarningBannerProps {
  content: string;
}

/**
 * Extracts warning message from content marked with #WARNING#
 * and renders it as a yellow banner.
 */
export function WarningBanner({ content }: WarningBannerProps) {
  if (!content) return null;

  const match = content.match(/#WARNING#\s*([\s\S]*?)(?=$)/);
  if (!match) return null;

  const warningText = match[1].trim();
  if (!warningText) return null;

  return (
    <div className="mt-3 mb-1 rounded-lg border border-amber-300 bg-amber-50/80 dark:border-amber-700 dark:bg-amber-950/30 px-4 py-2.5 flex items-start gap-2.5">
      <AlertTriangleIcon className="size-4 text-amber-600 dark:text-amber-400 mt-0.5 shrink-0" />
      <span className="text-sm text-amber-800 dark:text-amber-200">{warningText}</span>
    </div>
  );
}

/**
 * Strips the #WARNING# section from content so it doesn't appear in markdown.
 */
export function stripWarningContent(content: string): string {
  if (!content) return content;
  return content.replace(/#WARNING#[\s\S]*$/, "").trim();
}
