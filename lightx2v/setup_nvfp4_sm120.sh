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
python3 -c "import torch, lightx2v_kernel"  # torch first: libtorch.so is only on the loader path after torch loads

# Attention kernels for the fast path (benchmarked Jul 19 2026 on a 5090:
# 480p 29s->11s, 720p 49s->27.7s vs the old all-torch_sdpa config).
# SageAttention: cross-attn sage_attn2; builds for sm_120 as-is upstream.
pip install --no-cache-dir pyzmq soundfile sentencepiece librosa
if ! python3 -c "import torch, sageattention._fused" 2>/dev/null; then
  pip uninstall -y sageattention 2>/dev/null || true
  [ -d "$BASE_DIR/SageAttention" ] || git clone --depth 1 https://github.com/thu-ml/SageAttention.git "$BASE_DIR/SageAttention"
  (cd "$BASE_DIR/SageAttention" && TORCH_CUDA_ARCH_LIST="12.0" EXT_PARALLEL=4     NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS="${MAX_JOBS:-32}"     pip install --no-cache-dir --no-build-isolation .)
fi
(cd / && python3 -c "import torch, sageattention._fused; print('sageattention CUDA OK')")
# SpargeAttn (spas_sage_attn): the sage2 block-sparse self-attn kernels. Its
# setup.py WHITELIST is the only sm_120 blocker -- the sm89 fp8 kernels
# compile and run correctly on Blackwell (clips verified real content).
if ! python3 -c "import torch, spas_sage_attn" 2>/dev/null; then
  [ -d "$BASE_DIR/SpargeAttn" ] || git clone --depth 1 https://github.com/thu-ml/SpargeAttn.git "$BASE_DIR/SpargeAttn"
  sed -i 's/SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0"}/SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0", "12.0"}/'     "$BASE_DIR/SpargeAttn/setup.py"
  grep -q '"12.0"' "$BASE_DIR/SpargeAttn/setup.py"  # loud if upstream moved the whitelist
  (cd "$BASE_DIR/SpargeAttn" && TORCH_CUDA_ARCH_LIST="12.0" EXT_PARALLEL=4     NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS="${MAX_JOBS:-32}"     pip install --no-cache-dir --no-build-isolation .)
fi
(cd / && python3 -c "import torch, spas_sage_attn; print('spas_sage_attn OK')")

python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("lightx2v/Wan2.2-NVFP4-Sparse",
                  allow_patterns=["*T2V*", "*.json", "*.md"],
                  local_dir="$MODELS/Wan2.2-NVFP4-Sparse")
PYEOF

python3 - <<PYEOF
import json
src = "$BASE_DIR/LightX2V/configs/wan22/extreme/wan_moe_t2v_distill_nvfp4_sparse_attn.json"
q = "$MODELS/Wan2.2-NVFP4-Sparse/Wan2.2-T2V-A14B_NVFP4_Sparse_%s.safetensors"
c = json.load(open(src))
c.update(
    target_height=720, target_width=1280,
    high_noise_quantized_ckpt=q % "high", low_noise_quantized_ckpt=q % "low",
    # Wan2.2 MoE high/low-noise expert switch. The NVFP4-Sparse template omits
    # it -> the Wan22MoeRunner dies with KeyError: 'boundary'. Match the bf16 cfg.
    boundary=0.875,
    # quantized experts fit resident in ~18GB -> no block offload needed
    cpu_offload=False, t5_cpu_offload=True, vae_cpu_offload=True,
    # The benchmarked fast path: block-sparse int8 self-attn (SpargeAttn
    # kernels, built above) + sage_attn2 cross-attn. 2.6x @480p / 1.8x @720p
    # over the old all-torch_sdpa fallback.
    self_attn_1_type="dynamic_sparse_attn",
    dynamic_sparse_attn_setting={"sparsity_ratio": 0.9, "operator": "sage2"},
    cross_attn_1_type="sage_attn2", cross_attn_2_type="sage_attn2",
    # flashinfer's rope JIT wants CUDA>=12.9 on sm_120; torch rope is
    # equivalent here (rope is a negligible slice of step time).
    rope_type="torch",
)
json.dump(c, open("$BASE_DIR/lx2v_nvfp4.json", "w"), indent=1)
PYEOF
echo "NVFP4_SM120_SETUP_DONE"
