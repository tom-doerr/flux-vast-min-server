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
# ACE-Step 1.5 (diffusers AceStepPipeline): XL SFT = commercial-grade, stereo
# 48kHz, LM-planner-assisted long-range structure. Loaded via --engine v15.
DEFAULT_MODEL_V15 = "ACE-Step/acestep-v15-xl-sft-diffusers"
STATE: "AppState | None" = None


def _wav_info(path: Path) -> tuple[float, int]:
    """(duration_seconds, sample_rate) for a wav, best-effort but NOT silent:
    a failure raises so a truncated/corrupt render is surfaced."""
    import soundfile as sf

    info = sf.info(str(path))
    return (float(info.frames) / float(info.samplerate), int(info.samplerate))


def _to_pcm16(path: Path) -> None:
    """Re-encode a wav IN PLACE to 16-bit PCM. ACE-Step writes 32-bit FLOAT wav,
    which most browsers cannot decode (Firefox not at all, Chrome only recently);
    16-bit PCM plays everywhere and is lossless-enough for listening."""
    import soundfile as sf

    if sf.info(str(path)).subtype == "PCM_16":
        return
    audio, sr = sf.read(str(path), dtype="float32")
    sf.write(str(path), audio, sr, subtype="PCM_16")


def _waveform_png(wav_path: Path, out_path: Path, width: int = 900, height: int = 180) -> None:
    """Render a peak-envelope waveform PNG (dark bg, green trace) from a wav so
    an audio card has a visual. Raises on failure (caller logs, non-fatal)."""
    import numpy as np
    import soundfile as sf
    from PIL import Image, ImageDraw

    audio, _sr = sf.read(str(wav_path), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    if n == 0:
        raise RuntimeError("empty audio for waveform")
    step = max(1, n // width)
    peak = float(np.max(np.abs(audio))) or 1.0
    img = Image.new("RGB", (width, height), (11, 18, 32))
    draw = ImageDraw.Draw(img)
    mid = height // 2
    for x in range(width):
        chunk = audio[x * step:(x + 1) * step]
        if len(chunk) == 0:
            continue
        hi = int(float(np.max(chunk)) / peak * (mid - 2))
        lo = int(float(np.min(chunk)) / peak * (mid - 2))
        draw.line([(x, mid - hi), (x, mid - lo)], fill=(74, 222, 128), width=1)
    img.save(str(out_path))


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
    waveform_path: Path | None = None   # peak-envelope PNG, when rendered

    def payload(self, job_id: int, index: int = 0) -> dict[str, Any]:
        data = {
            "index": index,
            "audio_url": f"/jobs/{job_id}/audio/{index}",
            "duration": self.duration,
            "sample_rate": self.sample_rate,
        }
        if self.waveform_path is not None:
            data["waveform_url"] = f"/jobs/{job_id}/audio/{index}/waveform"
        return data


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
                 cpu_offload: bool = False, torch_compile: bool = False,
                 engine: str = "v1", model_id: str = "") -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.dtype = dtype
        self.cpu_offload = cpu_offload
        self.torch_compile = torch_compile
        self.engine = engine
        self._pipe: Any = None
        self._lock = threading.Lock()
        self._model_id = model_id or (DEFAULT_MODEL_V15 if engine == "v15" else DEFAULT_MODEL)
        # CLAP audio embedder for on-box ridge/diversity (kind='audio'), mirrors
        # the wan box's VideoMAE embedder. Loaded lazily, separate from ACE-Step.
        self._embedder: Any = None
        self._embed_lock = threading.Lock()
        self._embed_model_id = os.environ.get("ACE_EMBED_MODEL", "laion/clap-htsat-unfused")
        self._start_output_pruner()

    def _start_output_pruner(self) -> None:
        """Bound disk use: the app fetches each clip within seconds and stores it
        in MinIO, so on-box outputs are disposable. Nothing pruned them and 5900
        clips + one embed temp per clip filled the 80 GB box (music died with
        LibsndfileError: System error). Delete files older than the retention
        window every few minutes. ACE_OUTPUT_RETENTION_MIN=0 disables."""
        retention_min = float(os.environ.get("ACE_OUTPUT_RETENTION_MIN", "20") or 0)
        if retention_min <= 0:
            return

        def prune() -> None:
            while True:
                try:
                    cutoff = time.time() - retention_min * 60
                    removed = 0
                    for f in self.output_dir.glob("*"):
                        try:
                            if f.is_file() and f.stat().st_mtime < cutoff:
                                f.unlink()
                                removed += 1
                        except OSError:
                            pass
                    if removed:
                        print(f"ace output pruner: removed {removed} files older "
                              f"than {retention_min:.0f}m", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"ace output pruner error: {exc}", flush=True)
                time.sleep(300)

        threading.Thread(target=prune, name="ace-output-pruner", daemon=True).start()

    def preload(self, model_id: str = "") -> str:
        self._load()
        return self._model_id

    def model_revision(self) -> str:
        """The loaded checkpoint's HF snapshot sha (first 16 chars) = an exact
        model hash for per-item provenance. Cached; local cache only (no net)."""
        rev = getattr(self, "_model_revision_cache", None)
        if rev is not None:
            return rev
        rev = ""
        try:
            from huggingface_hub import snapshot_download
            p = snapshot_download(self._model_id, local_files_only=True).rstrip("/")
            parts = p.split("/")
            if "snapshots" in parts:
                rev = parts[parts.index("snapshots") + 1][:16]
        except Exception:
            rev = ""
        self._model_revision_cache = rev
        return rev

    def _load(self) -> Any:
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            if self.engine == "v15":
                import torch
                from diffusers import AceStepPipeline
                dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
                self._pipe = AceStepPipeline.from_pretrained(self._model_id, dtype=dtype).to("cuda")
                print(f"loaded ACE-Step 1.5 ({self._model_id}, {self.dtype})", flush=True)
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

    def _resolve_audio_path(self, payload: dict[str, Any]) -> tuple[Path, bool]:
        """Returns (wav_path, is_temp). is_temp files are the app's uploaded
        audio_b64 written to disk only so soundfile can read them -- they MUST be
        deleted after embedding (one per clip at ~11 MB filled the 80 GB box)."""
        import base64
        if payload.get("audio_b64"):
            raw = base64.b64decode(payload["audio_b64"])
            tmp = self.output_dir / f"embed_{int(time.time() * 1000)}.wav"
            tmp.write_bytes(raw)
            return tmp, True
        job_id = payload.get("job_id")
        if job_id is not None:
            candidate = self.output_dir / f"job_{int(job_id):06d}.wav"
            if candidate.exists():
                return candidate, False
        raise ValueError("embed requires job_id (with a ready wav) or audio_b64")

    def _load_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        import torch
        from transformers import ClapModel, ClapProcessor

        proc = ClapProcessor.from_pretrained(self._embed_model_id)
        model = ClapModel.from_pretrained(self._embed_model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embedder = (model.to(device).eval(), proc)
        print(f"loaded CLAP embedder {self._embed_model_id}", flush=True)
        return self._embedder

    def embed(self, payload: dict[str, Any]) -> dict[str, Any]:
        """On-box CLAP audio embedding (kind='audio'). Raises on failure -- no
        silent zero vector (parity with the wan/video embedder)."""
        import numpy as np
        import soundfile as sf
        import torch

        wav_path, is_temp = self._resolve_audio_path(payload)
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)  # mono
            target_sr = 48000
            if sr != target_sr:
                import torchaudio
                audio = torchaudio.functional.resample(
                    torch.from_numpy(np.ascontiguousarray(audio)), sr, target_sr).numpy()
            with self._embed_lock:
                model, proc = self._load_embedder()
                vec, nwin = self._embed_windows_mean(model, proc, audio, target_sr)
            # Report a VERSIONED model id (base@mw{N}s) so multi-window vectors are
            # stored NON-DESTRUCTIVELY (distinct from any single-window rows -- the
            # embeddings PK includes `model`), with the windowing method explicit.
            win_s = int(round(float(os.environ.get("ACE_EMBED_WINDOW_S", "10"))))
            return {"embedding": vec, "dims": len(vec),
                    "model": f"{self._embed_model_id}@mw{win_s}s",
                    "windows": nwin, "method": "multiwindow_mean", "window_s": win_s}
        finally:
            if is_temp:
                try:
                    wav_path.unlink()
                except OSError:
                    pass

    def _embed_windows_mean(self, model, proc, audio, target_sr):
        """CLAP mean over ceil(dur/10) NON-overlapping 10s windows (512-dim).
        CLAP's extractor keeps only ~10s, so one call embeds ~10s of ANY clip --
        fine for a 30s loop but ranks a 4-min track on a random ~4% of it.
        Benchmarked (spark-3): 240s Spearman 0.178 (1 window) -> 0.32 (full
        coverage); overlap/concat did NOT help. Dimension unchanged (drop-in)."""
        import numpy as np
        import torch
        win = int(round(float(os.environ.get("ACE_EMBED_WINDOW_S", "10")) * target_sr))
        n = int(len(audio))
        starts = list(range(0, max(1, n - win + 1), win)) or [0]
        if starts[-1] + win < n:
            starts.append(max(0, n - win))   # cover the tail with a full window
        segs = [np.pad(audio[s:s + win], (0, max(0, win - len(audio[s:s + win]))))[:win] for s in starts]
        batch = int(os.environ.get("ACE_EMBED_WINDOW_BATCH", "16"))
        vecs = []
        for i in range(0, len(segs), batch):
            part = segs[i:i + batch]
            try:
                inputs = proc(audio=part, sampling_rate=target_sr, return_tensors="pt")
            except (TypeError, ValueError):
                # transformers 4.50 ClapProcessor takes audios= (not audio=) and
                # raises ValueError (not TypeError) on the unknown audio= kwarg, so
                # catch both to fall through to the older keyword.
                inputs = proc(audios=part, sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = model.get_audio_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "audio_embeds") and feats.audio_embeds is not None:
                feats = feats.audio_embeds
            vecs.append(feats.float().cpu().numpy())
        return np.concatenate(vecs, 0).mean(axis=0).tolist(), len(segs)

    def generate(self, job: Job) -> None:
        pipe = self._load()
        p = job.payload or {}
        prompt = str(p.get("prompt") or "")          # genre / style tags, comma-separated
        lyrics = str(p.get("lyrics") or "[instrumental]")
        save_path = str(self.output_dir / f"job_{job.id:06d}.wav")
        t0 = time.time()
        if self.engine == "v15":
            import soundfile as sf
            # v1.5 defaults: SFT wants ~30-60 steps, guidance 7 (APG), shift 3.
            # A v1-era guidance (>10) from an un-updated client is reset, since
            # 15 wrecks v1.5 output. Output is stereo 48kHz float.
            g = float(p.get("guidance_scale", 7.0))
            if g > 10.0:
                g = 7.0
            steps = int(p.get("infer_step", 40))
            with self._lock:
                audio = pipe(
                    prompt=prompt,
                    lyrics=lyrics,
                    audio_duration=float(p.get("audio_duration", 30.0)),
                    num_inference_steps=max(1, steps),
                    guidance_scale=g,
                    shift=float(p.get("shift", 3.0)),
                ).audios
            wav = Path(save_path)
            sf.write(str(wav), audio[0].T.cpu().float().numpy(), pipe.sample_rate)
        else:
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
        _to_pcm16(wav)  # ACE-Step writes 32-bit float; make it browser-playable
        dur, sr = _wav_info(wav)
        rec = AudioRecord(wav, dur, sr)
        waveform = wav.with_name(wav.stem + ".waveform.png")
        try:
            _waveform_png(wav, waveform)
            rec.waveform_path = waveform
        except Exception as exc:  # noqa: BLE001 -- visual is non-fatal
            print(f"[ace] waveform failed job {job.id}: {exc}", flush=True)
        job.audios = [rec]
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
        self._started_at = time.time()
        self._gpu_idle_streak = 0
        if os.environ.get("ACE_WATCHDOG", "1") not in ("", "0", "false"):
            threading.Thread(target=self._watchdog, name="ace-watchdog", daemon=True).start()

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

    def _progress_snapshot(self) -> tuple[int, float, float]:
        """(pending, oldest_running_age_s, since_last_progress_s), now-based.

        pending = queued+running. oldest_running = age of the longest-running job
        (the single worker processes one at a time, so this is the current job).
        since_last_progress = seconds since any job last FINISHED (seeded with the
        server start time so a fresh box is not flagged before its first clip).
        """
        now = time.time()
        with self._lock:
            jobs = list(self._jobs.values())
        pending = sum(1 for j in jobs if j.status in ("queued", "running"))
        running = [now - j.started_at for j in jobs if j.status == "running" and j.started_at]
        oldest = max(running) if running else 0.0
        finishes = [j.finished_at for j in jobs if j.finished_at]
        last = max(finishes) if finishes else self._started_at
        return pending, oldest, now - last

    def _watchdog(self) -> None:
        """Hard-restart the process when generation WEDGES: work is pending but
        not progressing AND the GPU is idle -- the ace-worker thread hung
        mid-render (a job stuck 'running' while the HTTP layer still answers
        /health 200, so the box LOOKS up). os._exit lets the tmux `while true`
        loop respawn us (model reload ~5s). False positives are guarded: a real
        render pins the GPU, so the idle STREAK never accumulates; an idle box
        has no pending work; model-load/self-restart is skipped.
        """
        stall_s = float(os.environ.get("ACE_WEDGE_STALL_S", "120"))
        idle_pct = float(os.environ.get("ACE_WEDGE_GPU_UTIL_PCT", "5"))
        need_streak = int(os.environ.get("ACE_WEDGE_IDLE_STREAK", "4"))
        every = float(os.environ.get("ACE_WEDGE_CHECK_S", "20"))
        self._watchdog_loop(stall_s, idle_pct, need_streak, every)

    def _watchdog_loop(self, stall_s: float, idle_pct: float,
                       need_streak: int, every: float) -> None:
        print(f"[ace-watchdog] armed: restart if a job stalls >{stall_s:.0f}s while "
              f"gpu<{idle_pct:.0f}% for {need_streak} x {every:.0f}s checks", flush=True)
        while True:
            time.sleep(every)
            if self.preloading or self.runtime._pipe is None:
                self._gpu_idle_streak = 0  # loading is not a wedge
                continue
            util = gpu_status().get("utilization_gpu_pct")
            if util is None:  # can't read the GPU -> never guess a wedge
                self._gpu_idle_streak = 0
                continue
            self._gpu_idle_streak = self._gpu_idle_streak + 1 if util < idle_pct else 0
            pending, oldest_running, since_progress = self._progress_snapshot()
            stalled = oldest_running > stall_s or (pending > 0 and since_progress > stall_s)
            if stalled and self._gpu_idle_streak >= need_streak:
                print(f"[ace-watchdog] WEDGED: pending={pending} "
                      f"oldest_running={oldest_running:.0f}s since_progress={since_progress:.0f}s "
                      f"gpu_idle_streak={self._gpu_idle_streak} (util={util}%) -> restart",
                      flush=True)
                os._exit(1)


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
                               "model": rt._model_id, "embed_model": rt._embed_model_id,
                               "runtime": rt.engine, "quant": rt.dtype,
                               "model_revision": rt.model_revision(),
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
            rec = job.audios[idx]
            if len(parts) >= 5 and parts[4] == "waveform":
                if rec.waveform_path is None or not rec.waveform_path.exists():
                    return self._json({"error": "waveform not available"}, 404)
                return self._bytes(rec.waveform_path.read_bytes(), "image/png")
            return self._bytes(rec.path.read_bytes(), "audio/wav")
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
        return self._json({"error": "not found"}, 404)


def main() -> int:
    global STATE
    parser = argparse.ArgumentParser(description="ACE-Step music generation server")
    parser.add_argument("--host", default=os.environ.get("ACE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ACE_PORT", "8920")))
    parser.add_argument("--output-dir", default=os.environ.get("ACE_OUTPUT_DIR", "/workspace/ace_out"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("ACE_CHECKPOINT_DIR", ""))
    parser.add_argument("--dtype", default=os.environ.get("ACE_DTYPE", "bfloat16"))
    parser.add_argument("--cpu-offload", action="store_true", default=os.environ.get("ACE_CPU_OFFLOAD", "") not in ("", "0", "false"))
    parser.add_argument("--engine", default=os.environ.get("ACE_ENGINE", "v1"), choices=["v1", "v15"])
    parser.add_argument("--model", default=os.environ.get("ACE_MODEL", ""))
    args = parser.parse_args()

    runtime = AceRuntime(output_dir=Path(args.output_dir), checkpoint_dir=args.checkpoint_dir,
                         dtype=args.dtype, cpu_offload=args.cpu_offload,
                         engine=args.engine, model_id=args.model)
    STATE = AppState(runtime)
    STATE.start_preload()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ace-step music server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
