<p align="center">
  <img src="assets/logo2.png" width="180" alt="Super Agent Logo">
</p>

<h1 align="center">Browser Pilot</h1>

<p align="center">
  <b>🧠 新一代浏览器级 AI 助手 —— 理解网页语义、执行复杂任务、自动决策与行动</b>
</p>

<p align="center">
  <a href="#-演示视频">演示视频</a> •
  <a href="#-核心功能">核心功能</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-架构详解">架构详解</a> •
  <a href="README.en.md">English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/framework-OpenJiuwen-orange.svg" alt="OpenJiuwen">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
</p>

---

## 💡 项目简介

**Browser Pilot** 是一款基于 [OpenJiuwen Agent Framework](https://gitcode.com/openJiuwen/agent-core) 构建的**浏览器智能助手插件**。它不仅具备对网页内容的深度理解能力，还能够结合用户意图进行推理、规划与执行，作为真正的 AI 助手，自动完成多步骤、跨页面的复杂任务。

> 🎯 **告诉它你想做什么，它会自己思考、执行、甚至从失败中学习。**

### 它能做什么？

| 🔍 智能问答 | 💼 协助办公 | 🛒 复杂任务 | 🔄 自我演进 |
|:---:|:---:|:---:|:---:|
| 网页内容直接问答 | 自动回复邮件 | 看菜谱视频 → 自动加购食材 | 失败自动反思 |
| 截图区域精准识别 | Excel 数据处理 | 跨应用多步骤工作流 | 上下文学习优化 |
| Deep Search 深度搜索 | 文档填写提交 | 搜价、预订、下单 | 越用越聪明 |


## ✨ 核心功能

### 1️⃣ 视觉问答 (VQA)

**让 AI 看懂你看到的一切**

- **网页问答**：打开任意网页，直接向 AI 提问页面内容
- **截图识别**：框选屏幕任意区域，AI 根据图像内容回答
- **Deep Search**：复杂问题自动拆解，多轮搜索，深度汇总

### 2️⃣ 智能办公

**把重复劳动交给 AI**

- **邮件处理**：阅读邮件内容，理解上下文，自动撰写专业回复
- **Excel 处理**：数据清洗、公式计算、图表生成、报表导出
- **表单填写**：智能识别字段，自动填充并提交

### 3️⃣ 长时程复杂任务

**跨应用、多步骤，一气呵成**

- **场景示例**：观看美食视频/文章 → 提取食材清单 → 打开购物软件 → 自动加入购物车
- **工作流编排**：自动规划任务步骤，按序执行，异常处理
- **跨平台协同**：浏览器、本地应用无缝衔接

### 4️⃣ 功能自演进

**越用越聪明的 AI**

- **失败反思**：任务失败时自动分析原因
- **上下文学习**：基于历史交互优化执行策略
- **自主修正**：调整方案后重新执行，直到成功

---

## 🚀 快速开始

Browser Pilot 由**浏览器插件（前端）** 和 **Agent 服务（后端）** 两部分组成。

### 环境要求

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) 包管理器
- Node.js 18+
- Chrome 浏览器

---

### 后端安装

#### 1. 克隆仓库

```bash
git clone https://github.com/xxxx/browser-pilot.git
cd browser-pilot
```

#### 2. 安装依赖

```bash
# 安装 uv（如果尚未安装）
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1
##安装openjiuwen和browser-use
uv pip install openjiuwen==0.1.2
uv pip install browser-use==0.11.2

```

#### 3. 配置环境变量

创建 `.env` 文件：

```bash
# === LLM 提供商配置（至少配置一个）===

# 方式一：OpenRouter（推荐，支持多种模型）
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # 可选，已有默认值

# 方式二：Anthropic 直连
# ANTHROPIC_API_KEY=your_anthropic_api_key
# ANTHROPIC_BASE_URL=https://api.anthropic.com  # 可选，已有默认值

# 方式三：OpenAI 直连
# OPENAI_API_KEY=your_openai_api_key
# OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，已有默认值

# === 模型配置 ===
MODEL_NAME=anthropic/claude-sonnet-4-20250514  # 可选，默认 anthropic/claude-sonnet-4.5
MODEL_PROVIDER=openrouter  # 可选，默认 openrouter

# === 搜索功能（按需配置）===
# SERPER_API_KEY=your_serper_api_key
# GOOGLE_API_KEY=your_google_api_key
# PERPLEXITY_API_KEY=your_perplexity_api_key
# GEMINI_API_KEY=your_gemini_api_key

# === 浏览器配置 ===
# BROWSER_USE_CDP_URL=http://127.0.0.1:9222  # 连接远程 Chrome 时配置
# BROWSER_USE_LLM_MODEL=google/gemini-2.5-pro  # 可选，已有默认值
```

#### 4. 启动服务

**Windows:**
```powershell
# 启动
.\start_agent.ps1
```

**macOS:**
```bash
./start_agent.sh 

```

---

### 前端安装

#### 打开chrome 浏览器
**Windows:**
```powershell
# 启动
.\browser_start_client.ps1
```

**macOS:**
```bash
./browser_start_client.sh
```

#### 加载插件到浏览器

1. 访问 `chrome://extensions/`
2. 开启右上角「开发者模式」
3. 点击「加载已解压的扩展程序」
4. 选择 `./frontend/dist` 目录
5. 打开插件，点击设置在Backend_URL输入 http://localhost:8000

---

### 验证安装

1. 确保后端服务（8930 端口 + 8000 端口）正常运行
2. 打开任意网页
3. 点击浏览器工具栏的 Browser Pilot 图标
4. 输入问题，测试 AI 是否正常响应

---

## 🏗️ 架构详解

### 整体架构

<p align="center">
  <img src="assets/架构图_zh.svg" alt="Architecture" width="800">
</p>


### 技术亮点

| 特点 | 描述 |
|------|------|
| **SuperReAct** | 增强的 ReAct 循环，思考 → 行动 → 观察，支持多轮迭代推理 |
| **Browser Use** | 浏览器自动化操作，跨页面、多步骤任务执行 |
| **反思演进** | 失败后自动反思原因，调整策略重试，越用越聪明 |
| **多模态视觉理解** | 支持网页内容、截图区域的视觉理解与问答 |

### 支持的模型

推荐使用 **[OpenRouter](https://openrouter.ai/)** 统一接入（支持 100+ 模型），同时也兼容各厂商 API 直连：

| 提供商 | 模型 |
|--------|------|
| **Anthropic** | Claude Sonnet 4 (推荐), Claude Opus 4 |
| **OpenAI** | GPT-4o, O3, O3-mini |
| **Google** | Gemini 2.0 Pro/Flash |
| **其他** | Qwen, DeepSeek, Llama 等 |

---

## 🤝 参与贡献

欢迎参与贡献！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/new-feature`)
3. 提交更改 (`git commit -m 'Add new feature'`)
4. 推送分支 (`git push origin feature/new-feature`)
5. 提交 Pull Request

## 📄 许可证

[Apache License 2.0](LICENSE)

## 🔗 相关链接

- [OpenJiuwen Agent Framework](https://gitcode.com/openJiuwen/agent-core) - 底层 Agent 框架

---

<p align="center">
  <b>⭐ 如果觉得有帮助，请给个 Star！</b>
</p>
