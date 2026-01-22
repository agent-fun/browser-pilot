#!/usr/bin/env bash
set -e

echo "Starting Super Agent services..."

# 1. 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
else
  echo "Virtual env not found: .venv/bin/activate"
  exit 1
fi

# 2. 启动 Browser MCP Server（后台）
echo "Starting Browser MCP Server (CDP)..."
uv run python src/super_agent/tool/mcp_servers/browser_use_mcp_server_cdp.py --host 127.0.0.1 --port 8930 &
MCP_PID=$!

# 等待 MCP Server
sleep 3

# 3. 启动 Agent API Server（前台）
echo "Starting Agent API Server..."
uvicorn src.super_agent.api.server:app --host 0.0.0.0 --port 8000

# 如果 uvicorn 退出，顺便关 MCP
trap "kill $MCP_PID" EXIT
