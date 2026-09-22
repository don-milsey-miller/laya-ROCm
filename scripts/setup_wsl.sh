#!/usr/bin/env bash
# Reproducible, sudo-free ROCm + Laya environment for WSL2 (Ubuntu 22.04/24.04).
#
# Optional convenience script, WSL2 only: on native Linux install ROCm PyTorch the normal way
# (https://rocm.docs.amd.com/projects/ai-ecosystem/) and just `pip install laya-rocm`.
# Defaults target gfx1151 (the machine laya-rocm was developed on); pass your own gfx target.
#
# Prerequisites on the Windows side: Windows 11, AMD Adrenalin >= 26.2.2 (WSL support), WSL2.
#
#   bash scripts/setup_wsl.sh [gfx-target]     # default gfx target: gfx1151 (Ryzen AI Max / Strix Halo)
#   source ~/.laya-rocm/env.sh
#
# Everything lands under ~/.laya-rocm; nothing is written outside $HOME. If you do have sudo,
# `sudo apt install libgomp1 && sudo dpkg -i rocdxg-roct_*.deb` is equivalent to steps 2-3.
set -euo pipefail

GFX="${1:-gfx1151}"
ROCDXG_VERSION="${ROCDXG_VERSION:-1.2.2}"
TORCH_SPEC="${TORCH_SPEC:-torch[device-${GFX}]==2.13.0+rocm10.0.0}"
TORCH_INDEX="${TORCH_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}"
PREFIX="$HOME/.laya-rocm"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$PREFIX/downloads"
cd "$PREFIX/downloads"

# 1. uv (brings its own Python; the distro python lacks ensurepip without sudo)
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

# 2. librocdxg: the user-mode bridge that exposes the Windows GPU to ROCm inside WSL
if [ ! -e "$PREFIX/rocdxg/opt/rocm/lib/librocdxg.so.1" ]; then
  curl -LsSfO "https://github.com/ROCm/librocdxg/releases/download/v${ROCDXG_VERSION}/rocdxg-roct_${ROCDXG_VERSION}_amd64.deb"
  dpkg-deb -x "rocdxg-roct_${ROCDXG_VERSION}_amd64.deb" "$PREFIX/rocdxg"
fi

# 3. libgomp (OpenMP runtime torch links against; not in a minimal WSL image)
if ! ldconfig -p | grep -q libgomp.so.1 && [ ! -e "$PREFIX/sysdeps/usr/lib/x86_64-linux-gnu/libgomp.so.1" ]; then
  apt-get download libgomp1
  dpkg-deb -x libgomp1_*.deb "$PREFIX/sysdeps"
fi

cat > "$PREFIX/env.sh" <<EOF
export LD_LIBRARY_PATH="$PREFIX/rocdxg/opt/rocm/lib:$PREFIX/sysdeps/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}"
export HSA_ENABLE_DXG_DETECTION=1   # required by ROCm < 7.13, harmless after
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
source "$PREFIX/venv/bin/activate"
EOF

# 4. Python env: ROCm PyTorch first, then laya-rocm (which pulls upstream laya)
[ -d "$PREFIX/venv" ] || "$UV" venv --python 3.12 "$PREFIX/venv"
"$UV" pip install -p "$PREFIX/venv/bin/python" --index-url "$TORCH_INDEX" --index-strategy unsafe-best-match "$TORCH_SPEC"
"$UV" pip install -p "$PREFIX/venv/bin/python" -e "$REPO_DIR[bench,test]"

# shellcheck disable=SC1091
source "$PREFIX/env.sh"
python -m laya_rocm doctor --no-smoke
