param(
    [switch]$NoEnv
)

$servers = @(
    @{ module = "src.super_agent.tool.mcp_servers.browser_use_mcp_server"; args = "" },
    @{ module = "src.super_agent.tool.mcp_servers.python_server"; args = "" },
    @{ module = "src.super_agent.tool.mcp_servers.vision_mcp_server"; args = "" },
    @{ module = "src.super_agent.tool.mcp_servers.audio_mcp_server"; args = "" },
    @{ module = "src.super_agent.tool.mcp_servers.reasoning_mcp_server"; args = "" },
    @{ module = "src.super_agent.tool.mcp_servers.reading_mcp_server"; args = "--transport sse" },
    @{ module = "src.super_agent.tool.mcp_servers.searching_mcp_server"; args = "" },
    @{ module = "src.super_agent.tool.mcp_servers.wiki_mcp_server"; args = "" }
    
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..\..')
$toolingDir = Join-Path $repoRoot "src\super_agent\tool"

if ($NoEnv) {
    Write-Output "Running in NoEnv mode: using python from current shell (respects active venv if any)"
}

if (-not $NoEnv) {
    # Function to check if a directory is a valid venv
    function Test-IsValidVenv {
        param([string]$VenvPath)
        $unixActivate = Join-Path $VenvPath "bin\activate"
        $windowsActivate = Join-Path $VenvPath "Scripts\Activate.ps1"
        return (Test-Path $windowsActivate) -or (Test-Path $unixActivate)
    }

    # Auto-detect venv name
    # Priority: 1) VENV_NAME env var, 2) Auto-detect in tool dir, 3) Default to .venv-tool
    if ($env:VENV_NAME) {
        # User explicitly set VENV_NAME
        $detectedVenvName = $env:VENV_NAME
        Write-Output "Using venv from VENV_NAME environment variable: $detectedVenvName"
    } else {
        # Auto-detect: scan tool directory for venv directories
        # Prefer common names in order: .venv-tool, .venv, venv
        $detectedVenvName = $null

        # Check preferred names first
        $preferredNames = @(".venv-tool", ".venv", "venv")
        foreach ($preferredName in $preferredNames) {
            $venvPath = Join-Path $toolingDir $preferredName
            if (Test-IsValidVenv -VenvPath $venvPath) {
                $detectedVenvName = $preferredName
                Write-Output "Auto-detected venv: $detectedVenvName (found in tool directory)"
                break
            }
        }

        # If no preferred name found, scan all directories in tool dir
        if (-not $detectedVenvName) {
            $directories = Get-ChildItem -Path $toolingDir -Directory -ErrorAction SilentlyContinue
            foreach ($dir in $directories) {
                $dirName = $dir.Name
                # Skip if this venv exists at repo root (we want tool-specific venvs)
                $repoRootVenvPath = Join-Path $repoRoot $dirName
                if (Test-Path $repoRootVenvPath) {
                    continue
                }
                if (Test-IsValidVenv -VenvPath $dir.FullName) {
                    $detectedVenvName = $dirName
                    Write-Output "Auto-detected venv: $detectedVenvName (scanned tool directory)"
                    break
                }
            }
        }

        # Fall back to default if nothing found
        if (-not $detectedVenvName) {
            $detectedVenvName = ".venv-tool"
            Write-Output "No venv found in tool directory, using default: $detectedVenvName"
        }
    }

    $venvName = $detectedVenvName

    # Validate that the venv name is different from any venv at repo root
    $repoRootVenvPath = Join-Path $repoRoot $venvName
    if (Test-Path $repoRootVenvPath) {
        Write-Error "ERROR: Virtual environment '$venvName' already exists at repository root: $repoRootVenvPath"
        Write-Error "Please use a different venv name (set VENV_NAME environment variable) or remove the existing venv."
        exit 1
    }

    # Determine venv activation script path (works for both Unix and Windows)
    $venvPath = Join-Path $toolingDir $venvName
    $venvActivateUnix = Join-Path $venvPath "bin\activate"
    $venvActivateWindows = Join-Path $venvPath "Scripts\Activate.ps1"

    if (Test-Path $venvActivateWindows) {
        $venvActivate = $venvActivateWindows
    } elseif (Test-Path $venvActivateUnix) {
        $venvActivate = $venvActivateUnix
    } else {
        Write-Error "ERROR: Could not find venv activation script in $venvPath"
        Write-Error "Please create the virtual environment or set VENV_NAME to an existing venv name."
        exit 1
    }

    Write-Output "Repository root: $repoRoot"
    Write-Output "Tooling directory: $toolingDir"
    Write-Output "Virtual environment name: $venvName"
    Write-Output "Virtual environment path: $venvActivate"
}
Write-Output "Starting MCP servers (parallel)..."
if ($NoEnv) {
    # If VIRTUAL_ENV is set, use that venv's Python directly (respects active venv)
    if ($env:VIRTUAL_ENV) {
        $venvPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        $venvPythonUnix = Join-Path $env:VIRTUAL_ENV "bin\python"
        if (Test-Path $venvPython) {
            $pythonExe = $venvPython
        } elseif (Test-Path $venvPythonUnix) {
            $pythonExe = $venvPythonUnix
        } else {
            # Fall back to command resolution if venv Python not found
            $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
            if (-not $pythonCmd) {
                $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
            }
            if (-not $pythonCmd) {
                Write-Error "ERROR: Python not found in venv or PATH. Please ensure python is available."
                exit 1
            }
            $pythonExe = $pythonCmd.Source
        }
    } else {
        # No venv detected, use system python (check python first, then python3)
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCmd) {
            $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
        }
        if (-not $pythonCmd) {
            Write-Error "ERROR: Python not found. Please ensure python is in your PATH or activate your venv."
            exit 1
        }
        $pythonExe = $pythonCmd.Source
    }
} else {
    $pythonExe = Join-Path $venvPath "Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        Write-Error "ERROR: Python executable not found in venv: $pythonExe"
        exit 1
    }
}
Write-Output "Python executable: $pythonExe"

foreach ($server in $servers) {
    $pythonCommand = if ([string]::IsNullOrWhiteSpace($server.args)) {
        "python -u -m $($server.module)"
    } else {
        "python -u -m $($server.module) $($server.args)"
    }
    
    # Build the command using the already-selected $pythonExe
    if ([string]::IsNullOrWhiteSpace($server.args)) {
        $fullCommand = "`$env:PYTHONPATH='$repoRoot'; `"$pythonExe`" -u -m $($server.module)"
    } else {
        $fullCommand = "`$env:PYTHONPATH='$repoRoot'; `"$pythonExe`" -u -m $($server.module) $($server.args)"
    }
    
    Write-Output "----------------------------------------------------"
    Write-Output "Launching: $pythonCommand"
    Write-Output "Module: $($server.module)"
    Write-Output "Working directory: $repoRoot"
    Write-Output "----------------------------------------------------"
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $fullCommand `
        -WorkingDirectory $repoRoot
}