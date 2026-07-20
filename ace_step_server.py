#!/usr/bin/env python3
"""ACE-Step music generation job server.

Mirrors wan_vast_min_server.py's job API (enqueue -> poll -> download) so the
idea_rank app client is a near-copy of the video client. Text tags + lyrics ->
a wav clip. Loads ACE-Step-v1-3.5B ONCE (lazy, lock-guarded), non-blocking
preload in a bg thread; a single worker thread drains the queue one job at a
time (one GPU). No silent fallbacks: a failed render marks the job 'error' with
the traceback rather than returning a bogus/empty clip.
"""
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

DEFAULT_MODEL = "ACE-Step/ACE-Step-v1-3.5B"
STATE: "AppState | None" = None


def _wav_info(path: Path) -> tuple[float, int]:
    """(duration_seconds, sample_rate) for a wav, best-effort but NOT silent:
    a failure raises so a truncated/corrupt render is surfaced."""
    import soundfile as sf

    info = sf.info(str(path))
    return (float(info.frames) / float(info.samplerate), int(info.samplerate))


def gpu_status() -> dict[str, Any]:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        used, total, util, draw, limit = [x.strip() for x in out.stdout.strip().split(",")]
        return {"vram_used_mb": float(used), "vram_total_mb": float(total),
                "utilization_gpu_pct": float(util), "power_draw_w": float(draw),
                "power_limit_w": float(limit)}
    except Exception:
        return {}


@dataclass
class AudioRecord:
    path: Path            # wav
    duration: float
    sample_rate: int

    def payload(self, job_id: int, index: int = 0) -> dict[str, Any]:
        return {
            "index": index,
            "audio_url": f"/jobs/{job_id}/audio/{index}",
            "duration": self.duration,
            "sample_rate": self.sample_rate,
        }


@dataclass
class Job:
    id: int
    payload: dict[str, Any]
    status: str = "queued"
    error: str = ""
    traceback: str = ""
    audios: list[AudioRecord] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    def payload_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "audios": [a.payload(self.id, i) for i, a in enumerate(self.audios)],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class AceRuntime:
    def __init__(self, *, output_dir: Path, checkpoint_dir: str, dtype: str = "bfloat16",
                 cpu_offload: bool = False, torch_compile: bool = False) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.dtype = dtype
        self.cpu_offload = cpu_offload
        self.torch_compile = torch_compile
        self._pipe: Any = None
        self._lock = threading.Lock()
        self._model_id = DEFAULT_MODEL

    def preload(self, model_id: str = "") -> str:
        self._load()
        return self._model_id

    def _load(self) -> Any:
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            from acestep.pipeline_ace_step import ACEStepPipeline

            # checkpoint_dir="" -> ACE-Step auto-downloads DEFAULT_MODEL to its cache.
            self._pipe = ACEStepPipeline(
                checkpoint_dir=self.checkpoint_dir or None,
                dtype=self.dtype,
                torch_compile=self.torch_compile,
                cpu_offload=self.cpu_offload,
                overlapped_decode=False,
            )
            print(f"loaded ACE-Step ({self.dtype})", flush=True)
            return self._pipe

    def generate(self, job: Job) -> None:
        pipe = self._load()
        p = job.payload or {}
        prompt = str(p.get("prompt") or "")          # genre / style tags, comma-separated
        lyrics = str(p.get("lyrics") or "[instrumental]")
        save_path = str(self.output_dir / f"job_{job.id:06d}.wav")
        t0 = time.time()
        with self._lock:
            out = pipe(
                format="wav",
                audio_duration=float(p.get("audio_duration", 60.0)),
                prompt=prompt,
                lyrics=lyrics,
                infer_step=int(p.get("infer_step", 60)),
                guidance_scale=float(p.get("guidance_scale", 15.0)),
                scheduler_type=str(p.get("scheduler_type", "euler")),
                cfg_type=str(p.get("cfg_type", "apg")),
                omega_scale=float(p.get("omega_scale", 10.0)),
                manual_seeds=p.get("manual_seeds"),
                save_path=save_path,
                batch_size=1,
            )
        wav = Path(out[0]) if isinstance(out, (list, tuple)) and out else Path(save_path)
        if not (wav.exists() and wav.stat().st_size > 0):
            raise RuntimeError(f"ACE-Step produced no audio for job {job.id}")
        dur, sr = _wav_info(wav)
        job.audios = [AudioRecord(wav, dur, sr)]
        print(f"[ace] job {job.id} -> {wav.name} {dur:.1f}s @ {sr}Hz "
              f"{time.time() - t0:.1f}s", flush=True)


class AppState:
    def __init__(self, runtime: AceRuntime) -> None:
        self.runtime = runtime
        self._jobs: dict[int, Job] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._queue: "queue.Queue[int]" = queue.Queue()
        self._worker = threading.Thread(target=self._loop, name="ace-worker", daemon=True)
        self._worker.start()
        self.preloading = False
        self.preload_error = ""

    def start_preload(self) -> None:
        self.preloading = True
        self.preload_error = ""
        threading.Thread(target=self._preload_run, name="ace-preload", daemon=True).start()

    def _preload_run(self) -> None:
        try:
            self.runtime.preload()
            print("preloaded ACE-Step", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.preload_error = f"{type(exc).__name__}: {exc}"
            print(f"preload failed: {exc}\n{traceback.format_exc()}", flush=True)
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
        pending = counts.get("queued", 0) + counts.get("running", 0)
        return {"jobs_total": total, "status_counts": counts,
                "queue_size": pending, "model": self.runtime._model_id}

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
                print(f"[ace] job {job_id} FAILED: {job.error}\n{job.traceback}", flush=True)
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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            rt = self.state.runtime
            return self._json({"ok": True, "loaded": rt._pipe is not None,
                               "preloading": self.state.preloading,
                               "preload_error": self.state.preload_error,
                               "model": rt._model_id,
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
        if len(parts) >= 4 and parts[2] == "audio":
            try:
                idx = int(parts[3])
            except ValueError:
                return self._json({"error": "bad audio index"}, 400)
            if idx < 0 or idx >= len(job.audios):
                return self._json({"error": "audio not ready"}, 404)
            return self._bytes(job.audios[idx].path.read_bytes(), "audio/wav")
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
        return self._json({"error": "not found"}, 404)


def main() -> int:
    global STATE
    parser = argparse.ArgumentParser(description="ACE-Step music generation server")
    parser.add_argument("--host", default=os.environ.get("ACE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ACE_PORT", "8920")))
    parser.add_argument("--output-dir", default=os.environ.get("ACE_OUTPUT_DIR", "/data/out_music/ace-vast-min"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("ACE_CHECKPOINT_DIR", ""))
    parser.add_argument("--dtype", default=os.environ.get("ACE_DTYPE", "bfloat16"))
    parser.add_argument("--cpu-offload", action="store_true", default=os.environ.get("ACE_CPU_OFFLOAD", "") not in ("", "0", "false"))
    args = parser.parse_args()

    runtime = AceRuntime(output_dir=Path(args.output_dir), checkpoint_dir=args.checkpoint_dir,
                         dtype=args.dtype, cpu_offload=args.cpu_offload)
    STATE = AppState(runtime)
    STATE.start_preload()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ace-step music server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
