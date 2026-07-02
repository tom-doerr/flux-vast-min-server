#!/usr/bin/env bash
# Idempotent LightX2V provisioning for the wan proxy runtime (4-step distill).
# Installs LightX2V (pinned), a call-time-failing flash_attn stub (their code
# imports flash_attn unconditionally in unused subsystems), and downloads the
# Wan2.2 T2V base weights + 4-step distill LoRAs. Safe to re-run.
set -euo pipefail
LX2V_REF="00c9da5ee7b2bfd7b13e79a367f2a23eaba04c8a"
BASE_DIR="${LX2V_BASE_DIR:-/workspace}"
MODELS="$BASE_DIR/models"
HERE="$(cd "$(dirname "$0")" && pwd)"

apt-get install -y -q libgl1 || true

if [ ! -d "$BASE_DIR/LightX2V/.git" ]; then
  git clone https://github.com/ModelTC/LightX2V.git "$BASE_DIR/LightX2V"
fi
git -C "$BASE_DIR/LightX2V" fetch -q origin
git -C "$BASE_DIR/LightX2V" checkout -q "$LX2V_REF"
pip install --no-cache-dir "$BASE_DIR/LightX2V"
pip install --no-cache-dir sageattention

# flash_attn stub: real flash-attn wheels track a single torch ABI; LightX2V
# pins its own torch, so a preinstalled wheel breaks with undefined symbols.
pip uninstall -y flash-attn 2>/dev/null || true
SP="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$SP/flash_attn"
cp "$HERE/flash_attn_stub.py" "$SP/flash_attn/__init__.py"
cp "$HERE/flash_attn_stub.py" "$SP/flash_attn/flash_attn_interface.py"
mkdir -p "$MODELS"
python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("Wan-AI/Wan2.2-T2V-A14B",
                  allow_patterns=["models_t5*", "*VAE*", "google/*", "*.json", "*.txt",
                                  "high_noise_model/*", "low_noise_model/*"],
                  local_dir="$MODELS/Wan2.2-T2V-A14B")
snapshot_download("lightx2v/Wan2.2-Distill-Loras", allow_patterns=["*t2v*"],
                  local_dir="$MODELS/Wan2.2-Distill-Loras")
PYEOF

sed "s|/workspace/models|$MODELS|g" "$HERE/wan22_t2v_4step_720p.json" > "$BASE_DIR/lx2v_t2v.json"
echo "LIGHTX2V_SETUP_DONE"
