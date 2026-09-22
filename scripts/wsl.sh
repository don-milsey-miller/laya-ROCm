#!/usr/bin/env bash
# Run a command inside the laya-rocm WSL environment from the repo root:  bash scripts/wsl.sh pytest -q
set -eo pipefail
source "$HOME/.laya-rocm/env.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "$@"
