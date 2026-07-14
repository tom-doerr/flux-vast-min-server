#!/usr/bin/env bash
# NVFP4-Sparse experts for RTX 50-series (sm_120a): ~2.8x faster than the bf16
# block-offload path (49s vs 135s per 720p/81f clip) and both experts stay
# resident in ~18GB, so no CPU offload at all.
#
# Requires: LightX2V checked out at $BASE_DIR/LightX2V (setup_lightx2v.sh).
# Builds lightx2v_kernel from source — the published wheels ship only sm_120
# CUDA sources and no prebuilt binary, so it must be compiled on the box.
set -euo pipefail
BASE_DIR="${LX2V_BASE_DIR:-/workspace}"
MODELS="$BASE_DIR/models"

# CUTLASS: the kernel's CMake demands an explicit path (no vendored copy).
CUTLASS_PATH="${CUTLASS_PATH:-/opt/pytorch/ao/third_party/cutlass}"
if [ ! -d "$CUTLASS_PATH" ]; then
  git clone --depth 1 https://github.com/NVIDIA/cutlass.git "$BASE_DIR/cutlass"
  CUTLASS_PATH="$BASE_DIR/cutlass"
fi

pip install --no-cache-dir scikit-build-core cmake ninja
MAX_JOBS="${MAX_JOBS:-32}" TORCH_CUDA_ARCH_LIST="12.0a" CUTLASS_PATH="$CUTLASS_PATH" \
  pip install --no-cache-dir --no-build-isolation \
    --config-settings=cmake.define.CUTLASS_PATH="$CUTLASS_PATH" \
    "$BASE_DIR/LightX2V/lightx2v_kernel"
python3 -c "import lightx2v_kernel"  # loud failure if the build didn't land

python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("lightx2v/Wan2.2-NVFP4-Sparse",
                  allow_patterns=["*T2V*", "*.json", "*.md"],
                  local_dir="$MODELS/Wan2.2-NVFP4-Sparse")
PYEOF

python3 - <<PYEOF
import json
src = "$BASE_DIR/LightX2V/configs/wan22/wan_moe_t2v_distill_nvfp4_sparse_attn.json"
q = "$MODELS/Wan2.2-NVFP4-Sparse/Wan2.2-T2V-A14B_NVFP4_Sparse_%s.safetensors"
c = json.load(open(src))
c.update(
    target_height=720, target_width=1280,
    high_noise_quantized_ckpt=q % "high", low_noise_quantized_ckpt=q % "low",
    # quantized experts fit resident in ~18GB -> no block offload needed
    cpu_offload=False, t5_cpu_offload=True, vae_cpu_offload=True,
    # cross-attn would fall back to sage_attn2, which has no sm_120 kernel
    cross_attn_1_type="torch_sdpa", cross_attn_2_type="torch_sdpa",
)
json.dump(c, open("$BASE_DIR/lx2v_nvfp4.json", "w"), indent=1)
PYEOF
echo "NVFP4_SM120_SETUP_DONE"
