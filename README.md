# Flux Vast Minimal Server

Standalone, public, minimal FLUX.2 server for Vast deployments. It is intentionally not copied from any private project.

It implements the subset of the local image-generation API used by Idea Rank:

- `GET /health`
- `GET /generate/params`
- `POST /generate/enqueue`
- `GET /batch/config`
- `POST /batch/config`
- `GET /jobs`
- `GET /jobs/<job_id>`
- `GET /jobs/<job_id>/images`
- `GET /jobs/<job_id>/images/<index>`
- `GET /jobs/<job_id>/images/<index>/decoded`
- `GET /jobs/<job_id>/images/<index>/raw_vae`

The default model is `diffusers/FLUX.2-dev-bnb-4bit`, the supported public Diffusers FLUX.2 dev pipeline. Explicit `flux2-dev-nvfp4`/`dev-nvfp4` aliases remain available, but the current Diffusers backend cannot load BFL's NVFP4 single-file weights cleanly; use a dedicated TensorRT/NVIDIA visual-generation backend for that path. `black-forest-labs/FLUX.2-klein-9b-fp8` remains supported when the deployment supplies its own `HF_TOKEN`/`HF_API`.

`raw_vae` is a compatibility artifact derived from the decoded RGB image and mapped to channel-first `[-1, 1]`; it is not an exact VAE-internal capture.

## Run in an existing Vast vLLM container

```bash
git clone https://github.com/tom-doerr/flux-vast-min-server.git /workspace/flux-vast-min-server
cd /workspace/flux-vast-min-server
python3 -m pip install --no-cache-dir -r requirements.txt
nohup python3 flux_vast_min_server.py --host 0.0.0.0 --port 8910 --offload none --preload-model flux2-dev-bnb-4bit --max-batch-size 2 --batch-wait-ms 1000 > /workspace/logs/flux.log 2>&1 &
```

## Test

```bash
curl http://HOST:8910/health
curl -X POST http://HOST:8910/generate/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"rainy cyberpunk street", "model_id":"flux2-dev-bnb-4bit", "width":1024, "height":1024, "steps":28, "guidance_scale":5, "seed":42}'
```

Then poll `/jobs/<id>` until `status` is `done` and download `/jobs/<id>/images/0`.

For a dedicated GPU instance, use `--offload none --preload-model <model>` so the model is loaded before HTTP readiness and remains resident until the process exits or a different model is explicitly loaded. This server does not run an idle auto-unload timer.

Dynamic batching is server-side. Clients can keep enqueuing one image per job; compatible queued jobs are coalesced up to `max_batch_size` within `batch_wait_ms`. The runtime settings can be changed without restart:

```bash
curl http://HOST:8910/batch/config
curl -X POST http://HOST:8910/batch/config \
  -H 'Content-Type: application/json' \
  -d '{"max_batch_size":2, "batch_wait_ms":1000}'
```

## Docker

```bash
docker build -t flux-vast-min-server:latest .
docker run --gpus all -p 8910:8910 flux-vast-min-server:latest
```
