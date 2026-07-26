#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PORT="${1:-8888}"

if command -v cygpath >/dev/null 2>&1; then
  # Git Bash / MSYS: Jupyter is a Windows exe, so pass a Windows path.
  REPO_ROOT_FOR_JUPYTER="$(cygpath -w "$REPO_ROOT")"
else
  REPO_ROOT_FOR_JUPYTER="$REPO_ROOT"
  if command -v wslpath >/dev/null 2>&1; then
    # WSL can launch the repo's Windows venv executables, but they need Windows paths.
    REPO_ROOT_FOR_JUPYTER="$(wslpath -w "$REPO_ROOT")"
  fi
fi

if [[ -x "$REPO_ROOT/.venv/Scripts/jupyter-lab.exe" ]]; then
  JUPYTER_LAB="$REPO_ROOT/.venv/Scripts/jupyter-lab.exe"
elif [[ -x "$REPO_ROOT/.venv/bin/jupyter-lab" ]]; then
  JUPYTER_LAB="$REPO_ROOT/.venv/bin/jupyter-lab"
else
  echo "Could not find JupyterLab in the repo venv."
  echo "Expected one of:"
  echo "  $REPO_ROOT/.venv/Scripts/jupyter-lab.exe"
  echo "  $REPO_ROOT/.venv/bin/jupyter-lab"
  exit 1
fi

echo "Starting JupyterLab from:"
echo "  $REPO_ROOT_FOR_JUPYTER"
echo
echo "If port $PORT is already busy, run:"
echo "  bash scripts/start_jupyterlab.sh 8889"
echo

exec "$JUPYTER_LAB" \
  --ServerApp.root_dir="$REPO_ROOT_FOR_JUPYTER" \
  --port="$PORT"
