#!/usr/bin/env python3
"""Minimal Wan 2.2 text-to-video server for Idea Rank.

Sibling of flux_vast_min_server.py. Runs on the rented Vast.ai box alongside
vLLM. Mirrors the FLUX enqueue/poll/download job contract but for VIDEO, and
adds an on-box /embed endpoint so generated clips are embedded where the bytes
already are (videos are too large to ship back to spark-2).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_MODEL = os.environ.get("WAN_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
# The FINE-TUNED checkpoint, not the raw MAE one. `videomae-base` is the masked
# autoencoder PRETRAINING checkpoint: its features reconstruct pixels and are
# not semantically separable -- measured on 250 rated clips they collapse to an
# effective rank of 8.8 of 768 dims (mean pairwise cosine 0.968) and predict
# ratings with spearman +0.000, i.e. no better than guessing the mean. The
# Kinetics-finetuned checkpoint of the SAME architecture: rank 86.1, +0.193.
DEFAULT_EMBED_MODEL = os.environ.get(
    "WAN_EMBED_MODEL", "MCG-NJU/videomae-base-finetuned-kinetics")
DEFAULT_WIDTH = int(os.environ.get("WAN_WIDTH", "1280"))
DEFAULT_HEIGHT = int(os.environ.get("WAN_HEIGHT", "720"))
DEFAULT_NUM_FRAMES = int(os.environ.get("WAN_NUM_FRAMES", "81"))  # ~5s @16fps
DEFAULT_FPS = int(os.environ.get("WAN_FPS", "16"))
DEFAULT_STEPS = int(os.environ.get("WAN_STEPS", "40"))
DEFAULT_GUIDANCE = float(os.environ.get("WAN_GUIDANCE", "5.0"))
EMBED_FRAMES = int(os.environ.get("WAN_EMBED_FRAMES", "16"))  # VideoMAE clip length

STATE: "AppState | None" = None


def gpu_status() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "free_gb": round(free / 1e9, 2),
            "total_gb": round(total / 1e9, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {"cuda": False, "error": str(exc)}


def _frame_to_pil(frame: Any):
    """WanPipeline returns frames as numpy arrays; convert one to a PIL image
    for the keyframe png. Handles uint8 and float[0,1]/float[0,255]."""
    from PIL import Image
    import numpy as np
    if hasattr(frame, "save"):
        return frame
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        scaled = arr * 255.0 if float(np.nanmax(arr) if arr.size else 0.0) <= 1.0 else arr
        arr = np.clip(scaled, 0, 255).astype("uint8")
    return Image.fromarray(arr)


@dataclass
class VideoRecord:
    path: Path            # mp4
    keyframe_path: Path   # png
    width: int
    height: int
    num_frames: int
    fps: int

    def payload(self, job_id: int, index: int = 0) -> dict[str, Any]:
        return {
            "index": index,
            "video_url": f"/jobs/{job_id}/videos/{index}",
            "keyframe_url": f"/jobs/{job_id}/videos/{index}/keyframe",
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "fps": self.fps,
        }


@dataclass
class Job:
    id: int
    payload: dict[str, Any]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    traceback: str | None = None
    videos: list[VideoRecord] = field(default_factory=list)

    def payload_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "videos": [v.payload(self.id, i) for i, v in enumerate(self.videos)],
            "payload": self.payload,
        }


class WanRuntime:
    def __init__(self, *, output_dir: Path, device: str, dtype: str,
                 model_id: str, embed_model: str, offload: str = "model",
                 runtime: str = "diffusers", lightx2v_url: str = "http://127.0.0.1:8912") -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.dtype = dtype
        self.offload = offload
        self.runtime = runtime  # 'diffusers' (in-process) | 'lightx2v' (proxy)
        self.lightx2v_url = lightx2v_url.rstrip("/")
        self._model_id = model_id
        self._embed_model_id = embed_model
        self._pipe = None
        self._embedder = None
        self._embed_processor = None
        self._lock = threading.Lock()
        self._embed_lock = threading.Lock()

    def _dtype_obj(self):
        import torch
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}.get(self.dtype, torch.bfloat16)

    def _load_pipe(self, model_id: str):
        import torch
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(model_id, torch_dtype=self._dtype_obj())
        cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if not cuda:
            pipe.enable_model_cpu_offload()
        elif self.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif self.offload == "model":
            # Keep only the active expert resident (Wan2.2-A14B is ~56GB in bf16);
            # model offload fits it on an 80GB card alongside 720p activations.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        return pipe

    def pipe(self, model_id: str | None = None):
        target = model_id or self._model_id
        with self._lock:
            if self._pipe is None or target != self._model_id:
                self._pipe = self._load_pipe(target)
                self._model_id = target
            return self._pipe

    def preload(self, model_id: str | None = None) -> str:
        self.pipe(model_id)
        return self._model_id

    def _generate_lightx2v(self, job: Job) -> None:
        """Proxy generation to a LightX2V server (NVFP4 step-distilled Wan2.2).
        steps/guidance from the payload are IGNORED: the distilled model has a
        fixed 4-step expert schedule and cfg disabled -- overriding them would
        break quality. width/height/frames/fps/seed pass through."""
        p = job.payload
        width = int(p.get("width") or DEFAULT_WIDTH)
        height = int(p.get("height") or DEFAULT_HEIGHT)
        num_frames = int(p.get("num_frames") or DEFAULT_NUM_FRAMES)
        fps = int(p.get("fps") or DEFAULT_FPS)
        seed = int(p.get("seed") or 42)
        name = f"job_{job.id:06d}.mp4"
        resp = self._lx2v("POST", "/v1/tasks/video/", {
            "prompt": str(p.get("prompt") or ""),
            "negative_prompt": str(p.get("negative_prompt") or ""),
            "seed": seed,
            "target_shape": [height, width],
            "target_video_length": num_frames,
            "target_fps": fps,
            "save_result_path": name,
        }, timeout=60)
        task_id = resp.get("task_id") or ""
        if not task_id:
            raise RuntimeError(f"lightx2v enqueue failed: {resp}")
        result_name = resp.get("save_result_path") or name
        deadline = time.time() + 1800
        status = ""
        while time.time() < deadline:
            st = self._lx2v("GET", f"/v1/tasks/{task_id}/status", timeout=30)
            status = ""
            if isinstance(st, dict):
                status = str(st.get("task_status") or st.get("status") or "").lower()
            if status in {"completed", "success", "succeed", "finished"}:
                break
            if status in {"failed", "error", "cancelled"}:
                raise RuntimeError(f"lightx2v task {task_id} failed: {st}")
            time.sleep(2)
        else:
            raise TimeoutError(f"lightx2v task {task_id} not done (last status {status!r})")
        mp4_bytes = self._lx2v("GET", f"/v1/files/download/{result_name}", timeout=180)
        if not isinstance(mp4_bytes, (bytes, bytearray)) or len(mp4_bytes) < 1000:
            raise RuntimeError(f"lightx2v result download too small: {len(mp4_bytes) if isinstance(mp4_bytes,(bytes,bytearray)) else type(mp4_bytes)}")
        base = self.output_dir / f"job_{job.id:06d}"
        mp4_path = base.with_suffix(".mp4")
        mp4_path.write_bytes(bytes(mp4_bytes))
        keyframe_path = base.with_suffix(".png")
        frames = self._sample_frames(mp4_path, count=3)
        if not frames:
            raise RuntimeError("lightx2v result has no decodable frames")
        _frame_to_pil(frames[len(frames) // 2]).save(keyframe_path)
        job.videos = [VideoRecord(mp4_path, keyframe_path, width, height, num_frames, fps)]

    def _lx2v(self, method: str, path: str, payload: dict | None = None, timeout: float = 60):
        import urllib.request
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.lightx2v_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
        return json.loads(body) if body[:1] in (b"{", b"[") else body

    def generate(self, job: Job) -> None:
        if self.runtime == "lightx2v":
            return self._generate_lightx2v(job)
        import torch
        from diffusers.utils import export_to_video
        p = job.payload
        width = int(p.get("width") or DEFAULT_WIDTH)
        height = int(p.get("height") or DEFAULT_HEIGHT)
        num_frames = int(p.get("num_frames") or DEFAULT_NUM_FRAMES)
        fps = int(p.get("fps") or DEFAULT_FPS)
        steps = int(p.get("steps") or DEFAULT_STEPS)
        guidance = float(p.get("guidance") if p.get("guidance") is not None else DEFAULT_GUIDANCE)
        seed = int(p.get("seed") or 42)
        prompt = str(p.get("prompt") or "")
        pipe = self.pipe(str(p.get("model_id") or self._model_id))
        gen_device = "cuda" if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=str(p.get("negative_prompt") or ""),
            height=height, width=width, num_frames=num_frames,
            num_inference_steps=steps, guidance_scale=guidance, generator=generator,
        )
        frames = result.frames[0]  # list[PIL.Image]
        base = self.output_dir / f"job_{job.id:06d}"
        mp4_path = base.with_suffix(".mp4")
        export_to_video(frames, str(mp4_path), fps=fps)
        keyframe_path = base.with_suffix(".png")
        _frame_to_pil(frames[len(frames) // 2]).save(keyframe_path)
        job.videos = [VideoRecord(mp4_path, keyframe_path, width, height, num_frames, fps)]

    def _load_embedder(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        if self._embedder is None:
            self._embed_processor = AutoImageProcessor.from_pretrained(self._embed_model_id)
            # float32 for the small embedder: HF image processors emit float32
            # pixel_values, so a bf16 model trips a dtype mismatch.
            model = AutoModel.from_pretrained(self._embed_model_id)
            if self.device.startswith("cuda") and torch.cuda.is_available():
                model = model.to(self.device)
            self._embedder = model.eval()
        return self._embedder, self._embed_processor

    def _sample_frames(self, mp4_path: Path, count: int):
        import imageio.v3 as iio
        import numpy as np
        frames = list(iio.imiter(str(mp4_path), plugin="pyav"))
        if not frames:
            return []
        if len(frames) <= count:
            idx = list(range(len(frames)))
        else:
            idx = [int(round(i * (len(frames) - 1) / (count - 1))) for i in range(count)]
        return [np.asarray(frames[i]) for i in idx]

    def _resolve_video_path(self, payload: dict[str, Any]) -> Path:
        job_id = payload.get("job_id")
        if job_id is not None and STATE is not None:
            job = STATE.get(int(job_id))
            if job is not None and job.videos:
                return job.videos[0].path
        b64 = payload.get("video_b64")
        if b64:
            data = base64.b64decode(b64)
            tmp = self.output_dir / f"embed_{int(time.time()*1000)}.mp4"
            tmp.write_bytes(data)
            return tmp
        raise ValueError("embed requires job_id (with a ready video) or video_b64")

    def embed(self, payload: dict[str, Any]) -> dict[str, Any]:
        import torch
        mp4_path = self._resolve_video_path(payload)
        frames = self._sample_frames(mp4_path, EMBED_FRAMES)
        if not frames:
            raise ValueError("no frames to embed")
        with self._embed_lock:
            model, proc = self._load_embedder()
            inputs = proc(frames, return_tensors="pt")
            mdtype = next(model.parameters()).dtype
            inputs = {k: (v.to(model.device, mdtype) if torch.is_floating_point(v) else v.to(model.device))
                      for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            hidden = out.last_hidden_state  # (1, seq, dim)
            vec = hidden.float().mean(dim=1).squeeze(0).cpu().tolist()
        return {"embedding": vec, "dims": len(vec), "model": self._embed_model_id}


class AppState:
    def __init__(self, runtime: WanRuntime) -> None:
        self.runtime = runtime
        self._jobs: dict[int, Job] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._queue: "queue.Queue[int]" = queue.Queue()
        self._worker = threading.Thread(target=self._loop, name="wan-worker", daemon=True)
        self._worker.start()
        self.preloading = False
        self.preload_error = ""

    def start_preload(self, model_id: str) -> None:
        """Load the generation model in a background thread so the HTTP server
        + the small independent embedder come up immediately."""
        self.preloading = True
        self.preload_error = ""
        threading.Thread(target=self._preload_run, args=(model_id,),
                         name="wan-preload", daemon=True).start()

    def _preload_run(self, model_id: str) -> None:
        try:
            self.runtime.preload(model_id)
            print(f"preloaded {model_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.preload_error = f"{type(exc).__name__}: {exc}"
            print(f"preload failed: {exc}", flush=True)
        finally:
            self.preloading = False

    def enqueue(self, payload: dict[str, Any]) -> Job:
        with self._lock:
            self._counter += 1
            job = Job(id=self._counter, payload=payload)
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            return self._jobs.get(int(job_id))

    def list_jobs(self, *, limit: int) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)
        return jobs[:limit]

    def status(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for j in self._jobs.values():
                counts[j.status] = counts.get(j.status, 0) + 1
            total = len(self._jobs)
        return {"jobs_total": total, "status_counts": counts, "model": self.runtime._model_id}

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            job.status = "running"
            job.started_at = time.time()
            try:
                self.runtime.generate(job)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
            finally:
                job.finished_at = time.time()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.client_address[0]} {fmt % args}", flush=True)

    @property
    def state(self) -> AppState:
        if STATE is None:
            raise RuntimeError("server state not initialized")
        return STATE

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _lightx2v_health(self, rt) -> dict[str, Any]:
        try:
            q = rt._lx2v("GET", "/v1/tasks/queue/status", timeout=5)
            lx_ok = isinstance(q, dict)
        except Exception:
            lx_ok, q = False, {}
        return {"ok": True, "loaded": lx_ok, "preloading": False,
                "preload_error": "" if lx_ok else "lightx2v unreachable",
                "runtime": "lightx2v", "lightx2v": q if isinstance(q, dict) else {},
                "model": rt._model_id, "embed_model": rt._embed_model_id,
                **self.state.status(), "gpu": gpu_status()}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            rt = self.state.runtime
            if rt.runtime == "lightx2v":
                return self._json(self._lightx2v_health(rt))
            return self._json({"ok": True, "loaded": rt._pipe is not None,
                               "preloading": self.state.preloading,
                               "preload_error": self.state.preload_error,
                               "model": rt._model_id, "embed_model": rt._embed_model_id,
                               **self.state.status(), "gpu": gpu_status()})
        if path == "/jobs":
            query = parse_qs(parsed.query)
            limit = max(1, int((query.get("limit") or ["25"])[0]))
            return self._json({"jobs": [j.payload_dict() for j in self.state.list_jobs(limit=limit)]})
        return self._get_job(path)

    def _get_job(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 2 or parts[0] != "jobs":
            return self._json({"error": "not found"}, 404)
        try:
            job_id = int(parts[1])
        except ValueError:
            return self._json({"error": "bad job id"}, 400)
        job = self.state.get(job_id)
        if job is None:
            return self._json({"error": "not found"}, 404)
        if len(parts) == 2:
            return self._json(job.payload_dict())
        if len(parts) >= 4 and parts[2] == "videos":
            try:
                idx = int(parts[3])
            except ValueError:
                return self._json({"error": "bad video index"}, 400)
            if idx < 0 or idx >= len(job.videos):
                return self._json({"error": "video not ready"}, 404)
            rec = job.videos[idx]
            if len(parts) >= 5 and parts[4] == "keyframe":
                return self._bytes(rec.keyframe_path.read_bytes(), "image/png")
            return self._bytes(rec.path.read_bytes(), "video/mp4")
        return self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._json({"error": "invalid json"}, 400)
        if not isinstance(payload, dict):
            return self._json({"error": "payload must be an object"}, 400)
        if path == "/generate/enqueue":
            job = self.state.enqueue(payload)
            return self._json({"id": job.id, "status": job.status})
        if path == "/embed":
            try:
                return self._json(self.state.runtime.embed(payload))
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        if path == "/settings/model":
            try:
                loaded = self.state.runtime.preload(str(payload.get("model_id") or DEFAULT_MODEL))
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return self._json({"loaded_model_id": loaded})
        return self._json({"error": "not found"}, 404)


def main() -> int:
    global STATE, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_NUM_FRAMES, DEFAULT_FPS
    parser = argparse.ArgumentParser(description="Wan 2.2 text-to-video server")
    parser.add_argument("--host", default=os.environ.get("WAN_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WAN_PORT", "8910")))
    parser.add_argument("--output-dir", default=os.environ.get("WAN_OUTPUT_DIR", "/data/out_videos/wan-vast-min"))
    parser.add_argument("--device", default=os.environ.get("WAN_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.environ.get("WAN_DTYPE", "bfloat16"))
    parser.add_argument("--offload", default=os.environ.get("WAN_OFFLOAD", "model"),
                        choices=["none", "model", "sequential"])
    parser.add_argument("--runtime", default=os.environ.get("WAN_RUNTIME", "lightx2v"),
                        choices=["diffusers", "lightx2v"],
                        help="DEFAULT lightx2v: proxy to a LightX2V server (4-step distill, "
                             "~6x faster). Fails LOUDLY if that server is down -- no silent "
                             "fallback to the slow in-process diffusers runtime; pass "
                             "--runtime diffusers explicitly for the legacy path.")
    parser.add_argument("--lightx2v-url", default=os.environ.get("WAN_LIGHTX2V_URL", "http://127.0.0.1:8912"))
    parser.add_argument("--preload-model", default=os.environ.get("WAN_PRELOAD_MODEL", DEFAULT_MODEL))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()
    DEFAULT_WIDTH, DEFAULT_HEIGHT = args.width, args.height
    DEFAULT_NUM_FRAMES, DEFAULT_FPS = args.num_frames, args.fps
    runtime = WanRuntime(output_dir=Path(args.output_dir), device=args.device,
                         dtype=args.dtype, model_id=args.preload_model or DEFAULT_MODEL,
                         embed_model=args.embed_model, offload=args.offload,
                         runtime=args.runtime, lightx2v_url=args.lightx2v_url)
    STATE = AppState(runtime)
    if args.preload_model and args.runtime == "diffusers":
        STATE.start_preload(args.preload_model)  # background: HTTP + embedder ready now
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"wan video server listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
