import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Settings as SettingsIcon, ChevronDown, ChevronRight } from "lucide-react";

interface PluginSettings {
  selfEvolution: {
    enabled: boolean;
    memoryPersistence: boolean;
    autoLearning: boolean;
    knowledgeBaseUpdates: boolean;
  };
  todoList: {
    enabled: boolean;
  };
}

const defaultPlugins: PluginSettings = {
  selfEvolution: {
    enabled: false,
    memoryPersistence: true,
    autoLearning: true,
    knowledgeBaseUpdates: true,
  },
  todoList: {
    enabled: false,
  },
};

export function Settings() {
  const [backendUrl, setBackendUrl] = useState("");
  const [language, setLanguage] = useState("en");
  const [plugins, setPlugins] = useState<PluginSettings>(defaultPlugins);
  const [open, setOpen] = useState(false);
  const [expandedPlugins, setExpandedPlugins] = useState<Record<string, boolean>>({});

  function t(en: string, zh: string): string;
  function t(en: React.ReactNode, zh: React.ReactNode): React.ReactNode;
  function t(en: React.ReactNode, zh: React.ReactNode): React.ReactNode {
    return language === "zh" ? zh : en;
  }

  useEffect(() => {
    if (chrome?.storage?.sync) {
      chrome.storage.sync.get(["backendUrl", "language", "plugins"], (items: Record<string, any>) => {
        if (chrome.runtime.lastError) {
          console.error("Error loading settings:", chrome.runtime.lastError);
          return;
        }
        if (items.backendUrl) setBackendUrl(items.backendUrl);
        if (items.language) setLanguage(items.language);
        if (items.plugins) setPlugins({ ...defaultPlugins, ...items.plugins });
      });
    }
  }, []);

  const handleSave = () => {
    if (!backendUrl.trim()) {
      alert(t("Please enter a backend URL", "请输入后端 URL"));
      return;
    }
    try {
      new URL(backendUrl.trim());
    } catch {
      alert(t("Please enter a valid URL (e.g., http://localhost:8000)", "请输入有效的 URL（例如：http://localhost:8000）"));
      return;
    }

    chrome.storage.sync.set(
      { backendUrl: backendUrl.trim(), language, plugins },
      () => {
        if (chrome.runtime.lastError) {
          console.error("Error saving settings:", chrome.runtime.lastError);
          alert("Failed to save settings: " + chrome.runtime.lastError.message);
        } else {
          setOpen(false);
          window.location.reload();
        }
      }
    );
  };

  const toggleExpanded = (key: string) => {
    setExpandedPlugins((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const updatePlugin = <K extends keyof PluginSettings>(
    pluginKey: K,
    field: keyof PluginSettings[K],
    value: boolean
  ) => {
    setPlugins((prev) => ({
      ...prev,
      [pluginKey]: { ...prev[pluginKey], [field]: value },
    }));
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon">
          <SettingsIcon className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t("Settings", "设置")}</DialogTitle>
          <DialogDescription>
            {t(
              "Configure your agent settings including backend URL, language, and plugins.",
              "配置您的代理设置，包括后端 URL、语言偏好和插件。"
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* General Section */}
          <section className="space-y-4">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              {t("General", "通用")}
            </h3>

            <div className="space-y-2">
              <label htmlFor="backend-url" className="text-sm font-medium">
                {t("Backend URL", "后端 URL")}
              </label>
              <Input
                id="backend-url"
                type="url"
                placeholder={t("e.g., http://localhost:8000", "例如：http://localhost:8000")}
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                {t(
                  <>Enter the base URL of your Python backend. The extension will POST to <code>{backendUrl || "..."}/v1/chat/stream</code></>,
                  <>输入您的 Python 后端的基础 URL。扩展将向 <code>{backendUrl || "..."}/v1/chat/stream</code> 发送 POST 请求</>
                )}
              </p>
            </div>

            <div className="space-y-2">
              <label htmlFor="language" className="text-sm font-medium">
                {t("Language", "语言")}
              </label>
              <select
                id="language"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <option value="en">English</option>
                <option value="zh">中文 (Chinese)</option>
              </select>
            </div>
          </section>

          {/* Divider */}
          <div className="border-t" />

          {/* Plugins Section */}
          <section className="space-y-4">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
              {t("Plugins", "插件")}
            </h3>

            {/* Self-Evolution Plugin */}
            <div className="rounded-lg border p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div
                  className="flex items-center gap-2 cursor-pointer select-none"
                  onClick={() => toggleExpanded("selfEvolution")}
                >
                  {expandedPlugins.selfEvolution ? (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  )}
                  <div>
                    <p className="text-sm font-medium">{t("Self-Evolution", "自我进化")}</p>
                    <p className="text-xs text-muted-foreground">
                      {t(
                        "Enable the agent to learn and evolve from interactions.",
                        "使代理能够从交互中学习和进化。"
                      )}
                    </p>
                  </div>
                </div>
                <Switch
                  checked={plugins.selfEvolution.enabled}
                  onCheckedChange={(v) => updatePlugin("selfEvolution", "enabled", v)}
                />
              </div>

              {expandedPlugins.selfEvolution && (
                <div className="ml-6 space-y-3 border-l pl-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm">{t("Memory Persistence", "记忆持久化")}</p>
                      <p className="text-xs text-muted-foreground">
                        {t("Persist learned knowledge across sessions.", "在会话间保留学到的知识。")}
                      </p>
                    </div>
                    <Switch
                      checked={plugins.selfEvolution.memoryPersistence}
                      onCheckedChange={(v) => updatePlugin("selfEvolution", "memoryPersistence", v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm">{t("Auto-Learning", "自动学习")}</p>
                      <p className="text-xs text-muted-foreground">
                        {t("Automatically learn from user corrections.", "自动从用户纠正中学习。")}
                      </p>
                    </div>
                    <Switch
                      checked={plugins.selfEvolution.autoLearning}
                      onCheckedChange={(v) => updatePlugin("selfEvolution", "autoLearning", v)}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm">{t("Knowledge Base Updates", "知识库更新")}</p>
                      <p className="text-xs text-muted-foreground">
                        {t("Allow updates to the knowledge base.", "允许更新知识库。")}
                      </p>
                    </div>
                    <Switch
                      checked={plugins.selfEvolution.knowledgeBaseUpdates}
                      onCheckedChange={(v) => updatePlugin("selfEvolution", "knowledgeBaseUpdates", v)}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Todo List Plugin */}
            <div className="rounded-lg border p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">{t("Todo List", "待办事项")}</p>
                  <p className="text-xs text-muted-foreground">
                    {t(
                      "Enable task tracking and todo list management.",
                      "启用任务跟踪和待办事项管理。"
                    )}
                  </p>
                </div>
                <Switch
                  checked={plugins.todoList.enabled}
                  onCheckedChange={(v) => updatePlugin("todoList", "enabled", v)}
                />
              </div>
            </div>
          </section>

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setOpen(false)}>
              {t("Cancel", "取消")}
            </Button>
            <Button onClick={handleSave}>{t("Save", "保存")}</Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
