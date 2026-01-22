# ============================
# One-click startup script
# Browser MCP + Agent API
# ============================

Write-Host "Starting Super Agent services..." -ForegroundColor Cyan

# 1. 激活虚拟环境
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..."
    . $venvPath
} else {
    Write-Error "Virtual env not found: $venvPath"
    exit 1
}

# 2. 启动 Browser MCP Server（新窗口）
Write-Host "Starting Browser MCP Server (CDP)..."
Start-Process powershell `
    -ArgumentList "-NoExit", "-Command", "uv run python src/super_agent/tool/mcp_servers/browser_use_mcp_server_cdp.py --host 127.0.0.1 --port 8930"

# 等待 MCP Server 启动
Write-Host "⏳ Waiting for MCP server to be ready..."
Start-Sleep -Seconds 3

# 3. 启动 Agent API Server（当前窗口）
Write-Host "Starting Agent API Server..."
uvicorn src.super_agent.api.server:app --host 0.0.0.0 --port 8000
