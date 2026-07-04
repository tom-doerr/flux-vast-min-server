#!/usr/bin/env bash
# Self-contained NVFP4-Sparse benchmark for the official lightx2v 5090 image.
# Runs setup -> Bench A (reference offload config) -> Bench B (experts
# resident + VideoMAE co-loaded) -> prints RESULTS -> stops the container.
# All output to stdout so `vastai logs` shows everything without ssh.
set -x
PY="${BENCH_PY:-/app/miniconda/bin/python}"
PIP="${BENCH_PIP:-/app/miniconda/bin/pip}"
M=/workspace/models

cd /workspace
git clone https://github.com/ModelTC/LightX2V.git || true
git -C LightX2V checkout 00c9da5ee7b2bfd7b13e79a367f2a23eaba04c8a
$PIP install --no-cache-dir /workspace/LightX2V
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("lightx2v/Wan2.2-NVFP4-Sparse",
                  allow_patterns=["*T2V*", "*.json", "*.md"],
                  local_dir="/workspace/models/Wan2.2-NVFP4-Sparse")
snapshot_download("Wan-AI/Wan2.2-T2V-A14B",
                  allow_patterns=["models_t5*", "*VAE*", "google/*", "*.json", "*.txt"],
                  local_dir="/workspace/models/Wan2.2-T2V-A14B")
PYEOF
echo WEIGHTS_DONE

# configs: reference (block offload, their published 45s profile) + resident
$PY - <<'PYEOF'
import json
c = json.load(open("/workspace/LightX2V/configs/wan22/wan_moe_t2v_distill_nvfp4_sparse_attn.json"))
c.update(target_height=720, target_width=1280,
         high_noise_quantized_ckpt="/workspace/models/Wan2.2-NVFP4-Sparse/Wan2.2-T2V-A14B_NVFP4_Sparse_high.safetensors",
         low_noise_quantized_ckpt="/workspace/models/Wan2.2-NVFP4-Sparse/Wan2.2-T2V-A14B_NVFP4_Sparse_low.safetensors")
json.dump(c, open("/workspace/ref.json", "w"), indent=1)
c.update(cpu_offload=False, t5_cpu_offload=True, vae_cpu_offload=True)
json.dump(c, open("/workspace/resident.json", "w"), indent=1)
PYEOF

launch_server() {
  for p in $(pgrep -x python; pgrep -x python3); do
    grep -q lightx2v.server "/proc/$p/cmdline" 2>/dev/null && kill -9 "$p"
  done
  sleep 3
  setsid -f bash -c "cd /workspace/LightX2V && $PY -m lightx2v.server \
    --model_path $M/Wan2.2-T2V-A14B --model_cls wan2.2_moe_distill --task t2v \
    --host 127.0.0.1 --port 8912 --config_json $1" \
    >> "/workspace/logs/server_$2.log" 2>&1 < /dev/null
  until curl -sf -m 5 http://127.0.0.1:8912/v1/tasks/queue/status > /dev/null; do
    sleep 8
    grep -q Traceback "/workspace/logs/server_$2.log" && { echo "SERVER_BOOT_ERROR $2"; tail -5 "/workspace/logs/server_$2.log"; exit 1; }
  done
  echo "SERVER_UP $2"
}

cat > /workspace/bench_inline.py <<'PYEOF'
import json, sys, time, urllib.request
B = "http://127.0.0.1:8912"
def post(p, d):
    r = urllib.request.Request(B+p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60))
def get(p):
    return json.load(urllib.request.urlopen(B+p, timeout=30))
label, n = sys.argv[1], int(sys.argv[2])
for i in range(n):
    t = time.time()
    tid = post("/v1/tasks/video/", {"prompt": "a calm ocean wave at sunset, cinematic", "seed": 42+i,
        "target_shape": [1280, 720], "target_video_length": 81, "target_fps": 16,
        "save_result_path": f"{label}_{i}.mp4"})["task_id"]
    while True:
        s = str(get(f"/v1/tasks/{tid}/status").get("task_status") or "").lower()
        if s in {"completed", "success", "succeed", "finished"}: break
        if s in {"failed", "error", "cancelled"}: raise SystemExit(f"TASK_FAILED {label} {i}")
        time.sleep(2)
    print(f"BENCH[{label}] clip{i}: {time.time()-t:.1f}s", flush=True)
print(f"BENCH[{label}] DONE", flush=True)
PYEOF

echo "===== BENCH A: reference (block offload) ====="
launch_server /workspace/ref.json refA
$PY /workspace/bench_inline.py refA 2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo "===== BENCH B: experts resident + VideoMAE co-loaded ====="
cat > /workspace/vmae_hold.py <<'PYEOF'
import time, numpy as np, torch
from transformers import AutoImageProcessor, AutoModel
proc = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = AutoModel.from_pretrained("MCG-NJU/videomae-base").to("cuda").eval()
print(f"VMAE_RESIDENT {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
fr = [np.random.randint(0, 255, (224, 224, 3), np.uint8) for _ in range(16)]
while True:
    with torch.no_grad():
        o = model(**{k: v.to("cuda") for k, v in proc(fr, return_tensors="pt").items()})
    print(f"VMAE_EMBED_OK dims={o.last_hidden_state.shape[-1]}", flush=True)
    time.sleep(45)
PYEOF
setsid -f $PY /workspace/vmae_hold.py >> /workspace/logs/vmae.log 2>&1 < /dev/null
launch_server /workspace/resident.json resB
$PY /workspace/bench_inline.py resB 2
echo "=== VRAM at end of bench B ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
tail -4 /workspace/logs/vmae.log
echo "ALL_RESULTS_DONE"
# stop the container so it never bills idle (results persist in `vastai logs`)
sleep 5
kill -9 1
