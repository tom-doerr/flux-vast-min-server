# Flux Vast Minimal Server

Standalone, public, minimal FLUX.2 server for Vast deployments. It is intentionally not copied from any private project.

It implements the subset of the local image-generation API used by Idea Rank:

- `GET /health`
- `GET /generate/params`
- `POST /generate/enqueue`
- `GET /jobs`
- `GET /jobs/<job_id>`
- `GET /jobs/<job_id>/images`
- `GET /jobs/<job_id>/images/<index>`
- `GET /jobs/<job_id>/images/<index>/decoded`
- `GET /jobs/<job_id>/images/<index>/raw_vae`

The default model is `diffusers/FLUX.2-dev-bnb-4bit`, which is usable without copying a Hugging Face token to the Vast host. `black-forest-labs/FLUX.2-klein-9b-fp8` remains supported when the deployment supplies its own `HF_TOKEN`.

`raw_vae` is a compatibility artifact derived from the decoded RGB image and mapped to channel-first `[-1, 1]`; it is not an exact VAE-internal capture.

## Run in an existing Vast vLLM container

```bash
git clone https://github.com/tom-doerr/flux-vast-min-server.git /workspace/flux-vast-min-server
cd /workspace/flux-vast-min-server
python3 -m pip install --no-cache-dir -r requirements.txt
nohup python3 flux_vast_min_server.py --host 0.0.0.0 --port 8910 --offload model > /workspace/logs/flux.log 2>&1 &
```

## Test

```bash
curl http://HOST:8910/health
curl -X POST http://HOST:8910/generate/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"rainy cyberpunk street", "model_id":"diffusers/FLUX.2-dev-bnb-4bit", "width":1024, "height":1024, "steps":28, "guidance_scale":5, "seed":42}'
```

Then poll `/jobs/<id>` until `status` is `done` and download `/jobs/<id>/images/0`.

## Docker

```bash
docker build -t flux-vast-min-server:latest .
docker run --gpus all -p 8910:8910 flux-vast-min-server:latest
```
