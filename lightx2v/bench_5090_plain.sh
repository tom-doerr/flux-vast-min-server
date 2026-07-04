#!/usr/bin/env bash
# NVFP4 bench on a PLAIN pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel image
# (widely cached on vast hosts; the official 19GB lightx2v image stalls in
# pull on every host tried). Builds the sm_120 kernels, then chains into
# bench_5090_all.sh for weights/configs/benchmarks/self-stop.
set -x
export BENCH_PY="$(command -v python3)"
export BENCH_PIP="$BENCH_PY -m pip"
HERE="$(cd "$(dirname "$0")" && pwd)"

apt-get update -q && apt-get install -y -q libgl1 git curl
$BENCH_PIP install --no-cache-dir scikit_build_core uv huggingface_hub

cd /workspace
git clone --depth 1 https://github.com/NVIDIA/cutlass.git
git clone https://github.com/ModelTC/LightX2V.git
git -C LightX2V checkout 00c9da5ee7b2bfd7b13e79a367f2a23eaba04c8a

# NVFP4 kernel for sm_120 (this is the arch their kernels actually target)
cd /workspace/LightX2V/lightx2v_kernel
MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) TORCH_CUDA_ARCH_LIST=12.0 \
  uv build --wheel -Cbuild-dir=build . \
  -Ccmake.define.CUTLASS_PATH=/workspace/cutlass --no-build-isolation
$BENCH_PIP install dist/*.whl --force-reinstall --no-deps
echo KERNEL_BUILT

TORCH_CUDA_ARCH_LIST=12.0 $BENCH_PIP install --no-cache-dir sageattention
echo SAGE_BUILT

# flash_attn call-time stub (unused subsystems import it unconditionally)
$BENCH_PIP uninstall -y flash-attn 2>/dev/null || true
SP="$($BENCH_PY -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$SP/flash_attn"
cp "$HERE/flash_attn_stub.py" "$SP/flash_attn/__init__.py"
cp "$HERE/flash_attn_stub.py" "$SP/flash_attn/flash_attn_interface.py"

exec bash "$HERE/bench_5090_all.sh"
