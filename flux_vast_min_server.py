#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

import numpy as np
from PIL import Image

FLUX2_DEV_BNB = "diffusers/FLUX.2-dev-bnb-4bit"
FLUX2_DEV_BFL = "black-forest-labs/FLUX.2-dev"
FLUX2_DEV_BFL_INT8 = "black-forest-labs/FLUX.2-dev-int8"
FLUX2_DEV_NVFP4 = "black-forest-labs/FLUX.2-dev-NVFP4"
FLUX2_DEV_NVFP4_FILE = "flux2-dev-nvfp4.safetensors"
FLUX2_KLEIN_FP8 = "black-forest-labs/FLUX.2-klein-9b-fp8"
FLUX2_KLEIN_FP8_FILE = "flux-2-klein-9b-fp8.safetensors"
FLUX2_KLEIN_BASE = "ModelsLab/FLUX.2-klein-9B"
DEFAULT_MODEL = FLUX2_DEV_BNB
MODEL_ALIASES = {
    "": DEFAULT_MODEL,
    "dev": FLUX2_DEV_BNB,
    "dev-nvfp4": FLUX2_DEV_NVFP4,
    "dev-int8": FLUX2_DEV_BFL_INT8,
    "dev-bfl": FLUX2_DEV_BFL,
    "dev-bfl-bf16": FLUX2_DEV_BFL,
    "dev-bfl-int8": FLUX2_DEV_BFL_INT8,
    "flux2-dev": FLUX2_DEV_BNB,
    "flux2-dev-nvfp4": FLUX2_DEV_NVFP4,
    "flux2-dev-int8": FLUX2_DEV_BFL_INT8,
    "flux2-dev-bfl-int8": FLUX2_DEV_BFL_INT8,
    "black-forest-labs/flux.2-dev-nvfp4": FLUX2_DEV_NVFP4,
    "black-forest-labs/flux.2-dev-int8": FLUX2_DEV_BFL_INT8,
    FLUX2_DEV_NVFP4.lower(): FLUX2_DEV_NVFP4,
    FLUX2_DEV_BFL_INT8.lower(): FLUX2_DEV_BFL_INT8,
    FLUX2_DEV_BFL.lower(): FLUX2_DEV_BFL,
    "dev-bnb": FLUX2_DEV_BNB,
    "dev-bnb-4bit": FLUX2_DEV_BNB,
    "flux2-dev-bnb": FLUX2_DEV_BNB,
    "flux2-dev-bnb-4bit": FLUX2_DEV_BNB,
    FLUX2_DEV_BNB.lower(): FLUX2_DEV_BNB,
    "klein": FLUX2_KLEIN_FP8,
    "klein-fp8": FLUX2_KLEIN_FP8,
    "klein-9b-fp8": FLUX2_KLEIN_FP8,
    "flux2-klein": FLUX2_KLEIN_FP8,
    "flux2-klein-fp8": FLUX2_KLEIN_FP8,
    "flux2-klein-9b-fp8": FLUX2_KLEIN_FP8,
    FLUX2_KLEIN_FP8.lower(): FLUX2_KLEIN_FP8,
}


def normalize_model_id(value: object) -> str:
    raw = str(value or "").strip()
    return MODEL_ALIASES.get(raw.lower(), raw or DEFAULT_MODEL)


def hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API", "HF_READ_GATED_API_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    return None


def json_safe_job(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if k not in {"traceback"}}


def image_arrays(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    decoded = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    raw_vae_compat = np.moveaxis(decoded * 2.0 - 1.0, -1, 0).astype(np.float32)
    return decoded.astype(np.float32), raw_vae_compat


@dataclass
class ImageRecord:
    index: int
    path: Path
    decoded_path: Path
    raw_vae_path: Path
    width: int
    height: int

    def payload(self, job_id: int) -> dict[str, Any]:
        return {
            "index": self.index,
            "url": f"/jobs/{job_id}/images/{self.index}",
            "image_url": f"/jobs/{job_id}/images/{self.index}",
            "decoded_tensor_url": f"/jobs/{job_id}/images/{self.index}/decoded",
            "raw_vae_tensor_url": f"/jobs/{job_id}/images/{self.index}/raw_vae",
            "width": self.width,
            "height": self.height,
            "format": "png",
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
    images: list[ImageRecord] = field(default_factory=list)

    def payload_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "images": [image.payload(self.id) for image in self.images],
            "payload": self.payload,
        }


class FluxRuntime:
    def __init__(self, *, output_dir: Path, device: str, dtype: str, offload: str) -> None:
        self.output_dir = output_dir
        self.device = device
        self.dtype = dtype
        self.offload = offload
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipe: Any = None
        self._model_id: str | None = None
        self._lock = threading.RLock()
        self._generate_lock = threading.RLock()

    def _dtype_obj(self):
        import torch

        clean = self.dtype.lower().strip()
        if clean in {"fp16", "float16"}:
            return torch.float16
        if clean in {"fp32", "float32"}:
            return torch.float32
        return torch.bfloat16

    def _load_transformer(self, model_id: str):
        from diffusers import Flux2Transformer2DModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        token = hf_token()
        kwargs = {"token": token} if token else {}
        checkpoint_path = hf_hub_download(repo_id=model_id, filename=FLUX2_KLEIN_FP8_FILE, **kwargs)
        state = load_file(checkpoint_path)
        state = {
            key: value
            for key, value in state.items()
            if not (key.endswith(".input_scale") or key.endswith(".weight_scale"))
        }
        if not state:
            raise RuntimeError(f"No model tensors found in {model_id}/{FLUX2_KLEIN_FP8_FILE}")
        attempts = (
            {"torch_dtype": self._dtype_obj(), "config": FLUX2_KLEIN_BASE, "subfolder": "transformer", **kwargs},
            {"dtype": self._dtype_obj(), "config": FLUX2_KLEIN_BASE, "subfolder": "transformer", **kwargs},
            {"config": FLUX2_KLEIN_BASE, "subfolder": "transformer", **kwargs},
        )
        last_exc: Exception | None = None
        for attempt in attempts:
            try:
                return Flux2Transformer2DModel.from_single_file(dict(state), **attempt)
            except Exception as exc:  # pragma: no cover - depends on diffusers version
                last_exc = exc
        raise RuntimeError(f"Failed to load transformer from {model_id}: {last_exc}") from last_exc

    def _apply_offload(self, pipe):
        mode = self.offload.lower().replace("-", "_")
        if mode in {"model", "model_cpu", "cpu"} and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        elif mode in {"sequential", "sequential_cpu"} and hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        return pipe

    def _apply_quantized_cuda(self, pipe):
        import torch

        for name in ("vae",):
            component = getattr(pipe, name, None)
            if component is not None and hasattr(component, "to"):
                component.to(self.device, dtype=torch.float32)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        return pipe

    def _build_klein_fp8_pipe(self, model_id: str):
        from diffusers import Flux2KleinPipeline

        token = hf_token()
        kwargs = {"token": token} if token else {}
        transformer = self._load_transformer(model_id)
        attempts = (
            {"transformer": transformer, "torch_dtype": self._dtype_obj(), **kwargs},
            {"transformer": transformer, "dtype": self._dtype_obj(), **kwargs},
            {"transformer": transformer, **kwargs},
        )
        last_exc: Exception | None = None
        for attempt in attempts:
            try:
                pipe = Flux2KleinPipeline.from_pretrained(FLUX2_KLEIN_BASE, **attempt)
                break
            except Exception as exc:  # pragma: no cover - depends on diffusers version
                last_exc = exc
        else:
            raise RuntimeError(f"Failed to build FLUX.2 Klein pipeline: {last_exc}") from last_exc
        return self._apply_offload(pipe)

    def _build_dev_bnb_pipe(self, model_id: str):
        from diffusers import AutoModel, Flux2Pipeline
        from transformers import Mistral3ForConditionalGeneration

        token = hf_token()
        auth_kwargs = {"token": token} if token else {}
        common = {"torch_dtype": self._dtype_obj(), "device_map": "cpu", **auth_kwargs}
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(model_id, subfolder="text_encoder", **common)
        transformer = AutoModel.from_pretrained(model_id, subfolder="transformer", **common)
        pipe = Flux2Pipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=self._dtype_obj(),
            **auth_kwargs,
        )
        return self._apply_offload(pipe)

    def _build_dev_bfl_pipe(self, *, quantize_8bit: bool):
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        from transformers import Mistral3ForConditionalGeneration

        token = hf_token()
        if not token:
            raise RuntimeError(f"HF_TOKEN/HF_API is required for gated model {FLUX2_DEV_BFL}")
        auth_kwargs = {"token": token}
        if quantize_8bit:
            device_map = "cuda" if self.device.startswith("cuda") else self.device
            text_encoder_kwargs = {
                "quantization_config": TransformersBitsAndBytesConfig(load_in_8bit=True),
                "device_map": device_map,
                **auth_kwargs,
            }
            transformer_kwargs = {
                "quantization_config": DiffusersBitsAndBytesConfig(load_in_8bit=True),
                "device_map": device_map,
                **auth_kwargs,
            }
        else:
            text_encoder_kwargs = {"torch_dtype": self._dtype_obj(), "device_map": "cpu", **auth_kwargs}
            transformer_kwargs = {"torch_dtype": self._dtype_obj(), "device_map": "cpu", **auth_kwargs}
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            FLUX2_DEV_BFL, subfolder="text_encoder", **text_encoder_kwargs
        )
        transformer = Flux2Transformer2DModel.from_pretrained(FLUX2_DEV_BFL, subfolder="transformer", **transformer_kwargs)
        pipe = Flux2Pipeline.from_pretrained(
            FLUX2_DEV_BFL,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=self._dtype_obj(),
            **auth_kwargs,
        )
        if quantize_8bit:
            return self._apply_quantized_cuda(pipe)
        return self._apply_offload(pipe)

    def _load_dev_nvfp4_transformer(self):
        from diffusers import Flux2Transformer2DModel
        from huggingface_hub import hf_hub_download

        token = hf_token()
        auth_kwargs = {"token": token} if token else {}
        checkpoint_path = hf_hub_download(repo_id=FLUX2_DEV_NVFP4, filename=FLUX2_DEV_NVFP4_FILE, **auth_kwargs)
        attempts = (
            {"config": FLUX2_DEV_BNB, "subfolder": "transformer", "torch_dtype": self._dtype_obj(), **auth_kwargs},
            {"config": FLUX2_DEV_BNB, "subfolder": "transformer", "dtype": self._dtype_obj(), **auth_kwargs},
            {"config": FLUX2_DEV_BNB, "subfolder": "transformer", **auth_kwargs},
            {"config": FLUX2_DEV_BFL, "subfolder": "transformer", "torch_dtype": self._dtype_obj(), **auth_kwargs},
        )
        last_exc: Exception | None = None
        for attempt in attempts:
            try:
                return Flux2Transformer2DModel.from_single_file(checkpoint_path, **attempt)
            except Exception as exc:  # pragma: no cover - depends on diffusers version
                last_exc = exc
        raise RuntimeError(f"Failed to load FLUX.2 dev NVFP4 transformer: {last_exc}") from last_exc

    def _build_dev_nvfp4_pipe(self):
        from diffusers import Flux2Pipeline
        from transformers import Mistral3ForConditionalGeneration

        token = hf_token()
        auth_kwargs = {"token": token} if token else {}
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            FLUX2_DEV_BNB,
            subfolder="text_encoder",
            torch_dtype=self._dtype_obj(),
            device_map="cpu",
            **auth_kwargs,
        )
        transformer = self._load_dev_nvfp4_transformer()
        pipe = Flux2Pipeline.from_pretrained(
            FLUX2_DEV_BNB,
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=self._dtype_obj(),
            **auth_kwargs,
        )
        return self._apply_offload(pipe)

    def _effective_model_id(self, model_id: str) -> str:
        if model_id == FLUX2_KLEIN_FP8 and not hf_token():
            print(f"Requested gated {FLUX2_KLEIN_FP8} without HF_TOKEN/HF_API; falling back to {FLUX2_DEV_BNB}", flush=True)
            return FLUX2_DEV_BNB
        return model_id

    def _build_pipe(self, model_id: str):
        if model_id == FLUX2_DEV_BNB:
            return self._build_dev_bnb_pipe(model_id)
        if model_id == FLUX2_DEV_NVFP4:
            return self._build_dev_nvfp4_pipe()
        if model_id == FLUX2_DEV_BFL_INT8:
            return self._build_dev_bfl_pipe(quantize_8bit=True)
        if model_id == FLUX2_DEV_BFL:
            return self._build_dev_bfl_pipe(quantize_8bit=False)
        if model_id == FLUX2_KLEIN_FP8:
            return self._build_klein_fp8_pipe(model_id)
        supported = ", ".join([FLUX2_DEV_NVFP4, FLUX2_DEV_BFL_INT8, FLUX2_DEV_BFL, FLUX2_DEV_BNB, FLUX2_KLEIN_FP8])
        raise RuntimeError(f"Unsupported model_id {model_id!r}; supported models are {supported}")

    def pipe(self, model_id: str):
        model_id = self._effective_model_id(normalize_model_id(model_id))
        with self._lock:
            if self._pipe is None or self._model_id != model_id:
                self._pipe = self._build_pipe(model_id)
                self._model_id = model_id
            return self._pipe

    def generate(self, job: Job) -> None:
        import torch

        payload = job.payload
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("prompt is required")
        width = int(payload.get("width") or 1024)
        height = int(payload.get("height") or 1024)
        steps = int(payload.get("num_inference_steps") or payload.get("steps") or 28)
        guidance_scale = float(payload.get("guidance_scale") or payload.get("guidance") or 5.0)
        seed = int(payload.get("seed") if payload.get("seed") is not None else 42)
        model_id = normalize_model_id(payload.get("model_id"))
        pipeline_kwargs = payload.get("pipeline_kwargs") if isinstance(payload.get("pipeline_kwargs"), dict) else {}

        pipe = self.pipe(model_id)
        generator_device = "cuda" if torch.cuda.is_available() and self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        call_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            **pipeline_kwargs,
        }
        if payload.get("negative_prompt"):
            call_kwargs["negative_prompt"] = str(payload["negative_prompt"])

        with self._generate_lock:
            result = pipe(**call_kwargs)
        images = list(result.images)
        if not images:
            raise RuntimeError("pipeline returned no images")

        job_dir = self.output_dir / str(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        records: list[ImageRecord] = []
        for index, image in enumerate(images):
            image = image.convert("RGB")
            png_path = job_dir / f"{index}.png"
            decoded_path = job_dir / f"{index}.decoded.npz"
            raw_vae_path = job_dir / f"{index}.raw_vae.npz"
            image.save(png_path)
            decoded, raw_vae = image_arrays(image)
            np.savez_compressed(decoded_path, decoded=decoded)
            np.savez_compressed(raw_vae_path, raw_vae=raw_vae)
            records.append(ImageRecord(index=index, path=png_path, decoded_path=decoded_path, raw_vae_path=raw_vae_path, width=image.width, height=image.height))
        job.images = records


class AppState:
    def __init__(self, runtime: FluxRuntime) -> None:
        self.runtime = runtime
        self.lock = threading.RLock()
        self.jobs: dict[int, Job] = {}
        self.next_job_id = 1
        self.queue: queue.Queue[int] = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def enqueue(self, payload: dict[str, Any]) -> Job:
        with self.lock:
            job = Job(id=self.next_job_id, payload=payload)
            self.next_job_id += 1
            self.jobs[job.id] = job
        self.queue.put(job.id)
        return job

    def get(self, job_id: int) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def list_jobs(self, *, limit: int) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda job: job.id, reverse=True)[:limit]

    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            job.status = "running"
            job.started_at = time.time()
            try:
                self.runtime.generate(job)
                job.status = "done"
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
                print(job.traceback, flush=True)
            finally:
                job.finished_at = time.time()
                self.queue.task_done()


STATE: AppState | None = None


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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            runtime = self.state.runtime
            return self._json({"ok": True, "loaded": runtime._pipe is not None, "model": runtime._model_id, "jobs": len(self.state.jobs), "queue_size": self.state.queue.qsize()})
        if path == "/generate/params":
            return self._json({"width": 1024, "height": 1024, "steps": 28, "guidance_scale": 5.0, "seed": 42, "model_id": DEFAULT_MODEL})
        if path in {"/models", "/settings/model"}:
            runtime = self.state.runtime
            return self._json({
                "default_model_id": DEFAULT_MODEL,
                "active_model_id": runtime._model_id,
                "loaded_model_id": runtime._model_id,
                "known_models": [
                    {"id": FLUX2_DEV_NVFP4, "label": "FLUX.2 dev NVFP4", "aliases": sorted(k for k, v in MODEL_ALIASES.items() if v == FLUX2_DEV_NVFP4 and k)},
                    {"id": FLUX2_DEV_BFL_INT8, "label": "FLUX.2 dev BFL 8-bit", "aliases": sorted(k for k, v in MODEL_ALIASES.items() if v == FLUX2_DEV_BFL_INT8 and k)},
                    {"id": FLUX2_DEV_BFL, "label": "FLUX.2 dev BFL BF16", "aliases": sorted(k for k, v in MODEL_ALIASES.items() if v == FLUX2_DEV_BFL and k)},
                    {"id": FLUX2_DEV_BNB, "label": "FLUX.2 dev bnb 4-bit", "aliases": sorted(k for k, v in MODEL_ALIASES.items() if v == FLUX2_DEV_BNB and k)},
                    {"id": FLUX2_KLEIN_FP8, "label": "FLUX.2 Klein 9B FP8", "aliases": sorted(k for k, v in MODEL_ALIASES.items() if v == FLUX2_KLEIN_FP8 and k)},
                ],
            })
        if path == "/jobs":
            query = parse_qs(parsed.query)
            limit = max(1, int((query.get("limit") or ["25"])[0]))
            status = (query.get("status") or [""])[0].lower().strip()
            jobs = [job for job in self.state.list_jobs(limit=limit) if not status or job.status == status]
            return self._json({"jobs": [job.payload_dict() for job in jobs]})

        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "jobs":
            try:
                job_id = int(parts[1])
            except ValueError:
                return self._json({"error": "bad job id"}, 400)
            job = self.state.get(job_id)
            if job is None:
                return self._json({"error": "not found"}, 404)
            if len(parts) == 2:
                return self._json(json_safe_job(job.payload_dict()))
            if len(parts) == 3 and parts[2] == "images":
                return self._json({"job_id": job_id, "images": [image.payload(job_id) for image in job.images]})
            if len(parts) >= 4 and parts[2] == "images":
                try:
                    image_index = int(parts[3])
                except ValueError:
                    return self._json({"error": "bad image index"}, 400)
                if image_index < 0 or image_index >= len(job.images):
                    return self._json({"error": "image not ready"}, 404)
                record = job.images[image_index]
                suffix = parts[4] if len(parts) >= 5 else ""
                if suffix == "decoded":
                    return self._bytes(record.decoded_path.read_bytes(), "application/octet-stream")
                if suffix == "raw_vae":
                    return self._bytes(record.raw_vae_path.read_bytes(), "application/octet-stream")
                return self._bytes(record.path.read_bytes(), "image/png")
        return self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._json({"error": "invalid json"}, 400)
        if path == "/generate/enqueue":
            if not isinstance(payload, dict):
                return self._json({"error": "payload must be an object"}, 400)
            job = self.state.enqueue(payload)
            return self._json({"id": job.id, "status": job.status})
        if path == "/settings/model":
            return self._json({"default_model_id": DEFAULT_MODEL, "active_model_id": self.state.runtime._model_id, "loaded_model_id": self.state.runtime._model_id})
        return self._json({"error": "not found"}, 404)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal Flux server for Vast deployments")
    parser.add_argument("--host", default=os.environ.get("AI_FLUX2_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AI_FLUX2_PORT", "8910")))
    parser.add_argument("--output-dir", default=os.environ.get("AI_FLUX2_OUTPUT_DIR", "/data/out_images/flux-vast-min"))
    parser.add_argument("--device", default=os.environ.get("AI_FLUX2_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.environ.get("AI_FLUX2_DTYPE", "bfloat16"))
    parser.add_argument("--offload", default=os.environ.get("AI_FLUX2_OFFLOAD", "model"), choices=["none", "model", "sequential"])
    args = parser.parse_args()

    global STATE
    STATE = AppState(FluxRuntime(output_dir=Path(args.output_dir), device=args.device, dtype=args.dtype, offload=args.offload))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"flux-vast-min-server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
