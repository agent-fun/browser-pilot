#!/usr/bin/env bash

# Parse command-line arguments
NO_ENV=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-env)
            NO_ENV=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--no-env]" >&2
            exit 1
            ;;
    esac
done

# Auto-detect repo root from script location
# Script is in: examples/super_agent/tool/mcp_servers/activate/
if [[ -z "${REPO_ROOT:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # Try going up 6 levels
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
    
    # Verify this is actually the repo root by checking for mcp_servers directory
    if [[ ! -d "$REPO_ROOT/src/super_agent/tool/mcp_servers" ]]; then
        echo "Warning: mcp_servers directory not found at calculated repo root: $REPO_ROOT" >&2
        echo "Trying alternative detection method..." >&2
        
        # Alternative: walk up from script directory until we find libs/
        current_dir="$SCRIPT_DIR"
        found=false
        for i in {1..10}; do
            current_dir="$(cd "$current_dir/.." && pwd)"
            if [[ -d "$current_dir/src/super_agent/tool/mcp_servers" ]]; then
                REPO_ROOT="$current_dir"
                found=true
                break
            fi
        done
        
        if [[ "$found" != "true" ]]; then
            echo "Error: Could not find repository root (looking for src/super_agent/tool/mcp_servers/ directory)" >&2
            echo "Please set REPO_ROOT environment variable manually" >&2
            exit 1
        fi
    fi
fi

# Validate repo root exists and has libs directory
if [[ ! -d "$REPO_ROOT" ]]; then
    echo "Error: Repository root directory does not exist: $REPO_ROOT" >&2
    exit 1
fi

if [[ ! -d "$REPO_ROOT/src/super_agent/tool/mcp_servers" ]]; then
    echo "Error: mcp_servers directory not found at: $REPO_ROOT/src/super_agent/tool/mcp_servers" >&2
    echo "This doesn't appear to be the correct repository root." >&2
    exit 1
fi

echo "Detected repository root: $REPO_ROOT"

# 定义 servers（module + args）
servers=(
    "src.super_agent.tool.mcp_servers.browser_use_mcp_server|"
    "src.super_agent.tool.mcp_servers.python_server|"
    "src.super_agent.tool.mcp_servers.vision_mcp_server|"
    "src.super_agent.tool.mcp_servers.audio_mcp_server|"
    "src.super_agent.tool.mcp_servers.reasoning_mcp_server|"
    "src.super_agent.tool.mcp_servers.reading_mcp_server|--transport sse"
    "src.super_agent.tool.mcp_servers.searching_mcp_server|"
    "src.super_agent.tool.mcp_servers.wiki_mcp_server|"
)

TOOLING_DIR="$REPO_ROOT/src/super_agent/tool"

if [[ "$NO_ENV" == "true" ]]; then
    echo "Running in NoEnv mode: using python from current shell (respects active venv if any)"
fi

if [[ "$NO_ENV" != "true" ]]; then
    # Function to check if a directory is a valid venv
    is_valid_venv() {
        local venv_path="$1"
        [[ -f "$venv_path/bin/activate" ]] || [[ -f "$venv_path/Scripts/activate" ]]
    }

    # Auto-detect venv name
    # Priority: 1) VENV_NAME env var, 2) Auto-detect in tool dir, 3) Default to .venv-tool
    if [[ -n "${VENV_NAME:-}" ]]; then
        # User explicitly set VENV_NAME
        DETECTED_VENV_NAME="$VENV_NAME"
        echo "Using venv from VENV_NAME environment variable: $DETECTED_VENV_NAME"
    else
        # Auto-detect: scan tool directory for venv directories
        # Prefer common names in order: .venv-tool, .venv, venv
        DETECTED_VENV_NAME=""
        
        # Check preferred names first
        for preferred_name in ".venv-tool" ".venv" "venv"; do
            if is_valid_venv "$TOOLING_DIR/$preferred_name"; then
                DETECTED_VENV_NAME="$preferred_name"
                echo "Auto-detected venv: $DETECTED_VENV_NAME (found in tool directory)"
                break
            fi
        done
        
        # If no preferred name found, scan all directories in tool dir
        if [[ -z "$DETECTED_VENV_NAME" ]]; then
            while IFS= read -r -d '' dir; do
                # Skip the tooling directory itself
                if [[ "$dir" == "$TOOLING_DIR" ]]; then
                    continue
                fi
                dir_name=$(basename "$dir")
                # Skip if this venv exists at repo root (we want tool-specific venvs)
                if [[ -d "$REPO_ROOT/$dir_name" ]]; then
                    continue
                fi
                if is_valid_venv "$dir"; then
                    DETECTED_VENV_NAME="$dir_name"
                    echo "Auto-detected venv: $DETECTED_VENV_NAME (scanned tool directory)"
                    break
                fi
            done < <(find "$TOOLING_DIR" -maxdepth 1 -type d -print0 2>/dev/null)
        fi
        
        # Fall back to default if nothing found
        if [[ -z "$DETECTED_VENV_NAME" ]]; then
            DETECTED_VENV_NAME=".venv-tool"
            echo "No venv found in tool directory, using default: $DETECTED_VENV_NAME"
        fi
    fi

    VENV_NAME="$DETECTED_VENV_NAME"

    # Validate that the venv name is different from any venv at repo root
    if [[ -d "$REPO_ROOT/$VENV_NAME" ]]; then
        echo "ERROR: Virtual environment '$VENV_NAME' already exists at repository root: $REPO_ROOT/$VENV_NAME" >&2
        echo "Please use a different venv name (set VENV_NAME environment variable) or remove the existing venv." >&2
        exit 1
    fi

    # Determine venv Python path (works for both Unix and Windows/Git Bash)
    if [[ -f "$TOOLING_DIR/$VENV_NAME/bin/python" ]]; then
        VENV_PYTHON="$TOOLING_DIR/$VENV_NAME/bin/python"
    elif [[ -f "$TOOLING_DIR/$VENV_NAME/Scripts/python.exe" ]]; then
        VENV_PYTHON="$TOOLING_DIR/$VENV_NAME/Scripts/python.exe"
    else
        echo "ERROR: Could not find Python executable in $TOOLING_DIR/$VENV_NAME" >&2
        echo "Please create the virtual environment or set VENV_NAME to an existing venv name." >&2
        exit 1
    fi

    echo "Repository root: $REPO_ROOT"
    echo "Tooling directory: $TOOLING_DIR"
    echo "Virtual environment name: $VENV_NAME"
    echo "Virtual environment Python: $VENV_PYTHON"
fi

# Select Python executable
if [[ "$NO_ENV" == "true" ]]; then
    # If VIRTUAL_ENV is set, use that venv's Python directly (respects active venv)
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        if [[ -f "$VIRTUAL_ENV/bin/python" ]]; then
            PYTHON_EXE="$VIRTUAL_ENV/bin/python"
        elif [[ -f "$VIRTUAL_ENV/Scripts/python.exe" ]]; then
            PYTHON_EXE="$VIRTUAL_ENV/Scripts/python.exe"
        else
            # Fall back to command resolution if venv Python not found
            if command -v python &> /dev/null; then
                PYTHON_EXE="$(command -v python)"
            elif command -v python3 &> /dev/null; then
                PYTHON_EXE="$(command -v python3)"
            else
                echo "Error: Python not found in venv or PATH. Please install Python 3." >&2
                exit 1
            fi
        fi
    else
        # No venv detected, use system python (check python first, then python3)
        if command -v python &> /dev/null; then
            PYTHON_EXE="$(command -v python)"
        elif command -v python3 &> /dev/null; then
            PYTHON_EXE="$(command -v python3)"
        else
            echo "Error: Python not found. Please install Python 3." >&2
            exit 1
        fi
    fi
    # Verify the resolved path exists
    if [[ ! -f "$PYTHON_EXE" ]]; then
        echo "ERROR: Resolved Python executable does not exist: $PYTHON_EXE" >&2
        exit 1
    fi
else
    # Use venv Python
    PYTHON_EXE="$VENV_PYTHON"
    if [[ ! -f "$PYTHON_EXE" ]]; then
        echo "ERROR: Python executable not found in venv: $PYTHON_EXE" >&2
        exit 1
    fi
fi

echo "Python executable: $PYTHON_EXE"
echo "Starting MCP servers (parallel)..."

# 捕获 Ctrl+C — 杀死所有子进程
trap 'echo "Stopping all servers..."; kill 0' SIGINT

# 并行启动所有 server
for entry in "${servers[@]}"; do
    module="${entry%%|*}"
    args="${entry##*|}"

    if [[ -z "$args" ]]; then
        python_cmd="python -u -m $module"
    else
        python_cmd="python -u -m $module $args"
    fi

    echo "----------------------------------------------------"
    echo "Launching: $python_cmd"
    echo "Module: $module"
    echo "Working directory: $REPO_ROOT"
    echo "----------------------------------------------------"

    # Run command in background using selected Python executable
    (
        cd "$REPO_ROOT" || { 
            echo "ERROR: Failed to cd to $REPO_ROOT for $module" >&2
            exit 1
        }
        
        # Set PYTHONPATH
        if [[ -z "${PYTHONPATH:-}" ]]; then
            export PYTHONPATH="$REPO_ROOT"
        else
            export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
        fi
        
        # Execute using the selected Python executable directly (no activation needed)
        if [[ -z "$args" ]]; then
            "$PYTHON_EXE" -u -m "$module"
        else
            "$PYTHON_EXE" -u -m "$module" $args
        fi
    ) &
done


# 等待所有子进程
wait