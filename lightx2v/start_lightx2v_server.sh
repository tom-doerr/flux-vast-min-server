#!/usr/bin/env bash
# Launch the LightX2V server (foreground). The wan proxy (--runtime lightx2v,
# the default) talks to it on :8912.
set -euo pipefail
BASE_DIR="${LX2V_BASE_DIR:-/workspace}"
PORT="${LX2V_PORT:-8912}"

# Triton must use its own bundled ptxas: the system CUDA 13.x ptxas trips
# triton's version check ("only support CUDA 10.0 ... but got 13.x").
PTXAS="$(python3 -c 'import triton, os, glob; c = glob.glob(os.path.dirname(triton.__file__) + "/backends/nvidia/bin/ptxas"); print(c[0] if c else "")')"
[ -n "$PTXAS" ] && export TRITON_PTXAS_PATH="$PTXAS"
unset CUDA_HOME || true
cd "$BASE_DIR/LightX2V"
exec python -m lightx2v.server \
  --model_path "$BASE_DIR/models/Wan2.2-T2V-A14B" \
  --model_cls wan2.2_moe \
  --task t2v \
  --host 127.0.0.1 \
  --port "$PORT" \
  --config_json "${LX2V_CONFIG:-$BASE_DIR/lx2v_t2v.json}"
